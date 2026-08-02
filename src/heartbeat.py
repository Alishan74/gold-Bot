"""
Shared heartbeat log: every pipeline job (build_history, build_fundamentals,
build_pattern_library, live_update) records when it started, whether it
succeeded, and how long it took, to data/heartbeats.json. This is what
the dashboard's "pulse check" panel actually reads - real telemetry from
the jobs themselves, not a guess inferred from file mtimes.

Concurrency: write_heartbeat() is a read-modify-write over the WHOLE
file (every job's entry lives in one shared dict) - safe as long as only
one writer touches a given path at a time, which used to be true by
construction (one job, one process, one write). scripts/run_continuous.py
made that assumption stop holding: a continuous loop's own cycle and a
dashboard-triggered action button (both writing THIS ENGINE's
heartbeats.json) can now genuinely run at the same moment. Without a
lock, two concurrent read-modify-writes can race - each reads the same
"before" dict, each patches in its OWN job's key, and whichever writes
last wins, silently discarding the other's update (not file corruption -
atomic_write_text still guarantees no torn file - just a LOST update,
e.g. the Pulse panel briefly missing or showing a stale entry for
whichever job lost the race). _heartbeat_lock() below serializes the
whole read-modify-write with an OS file lock so this can't happen.
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from atomic_io import atomic_write_text

# File locking is NOT cross-platform in the standard library - fcntl only
# exists on POSIX (Linux/Mac), msvcrt only on Windows. This project runs
# on both (the pipeline is commonly deployed on a Windows PC alongside a
# MetaTrader terminal - see mt_bridge/README.md), so both branches are
# real, not a hypothetical. Importing the wrong one unconditionally is a
# hard ImportError at startup on the other platform - verified the hard
# way: an earlier version of this file imported fcntl unconditionally and
# crashed immediately on Windows with "ModuleNotFoundError: No module
# named 'fcntl'", before this system ever got a chance to run anything.
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

DEFAULT_PATH = Path("data") / "heartbeats.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


@contextmanager
def _heartbeat_lock(path: Path):
    """Exclusive OS advisory lock serializing the read-modify-write in
    write_heartbeat() across processes - a SEPARATE sibling `.lock` file,
    not `path` itself, since `path` is only ever touched via
    atomic_write_text's temp-file+rename dance (each writer gets its own
    temp file, so locking `path` directly wouldn't coordinate anything).
    Released automatically when the writing process's file handle closes
    (including on a crash mid-write - the lock can never be left
    permanently held by a dead process, unlike a lock FILE'S EXISTENCE
    would be).

    Cross-platform: fcntl.flock on POSIX (blocks indefinitely until
    acquired), msvcrt.locking on Windows (retries for ~10s, then raises
    PermissionError - acceptable here since the critical section is a
    fast in-memory JSON read + a temp-file write, never something that
    should legitimately hold the lock anywhere near that long; a
    PermissionError surfacing on a genuinely stuck lock is preferable to
    hanging forever silently). msvcrt.locking() locks a byte range
    starting at the file's CURRENT position, and needs that byte to
    actually exist in the file for older Windows/CRT versions to lock it
    reliably - hence writing one placeholder byte the first time the
    lock file is created, then always seeking to 0 before locking so
    every process locks the SAME byte."""
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        if sys.platform == "win32":
            lock_file.seek(0, 2)  # end
            if lock_file.tell() == 0:
                lock_file.write(b"x")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if sys.platform == "win32":
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def write_heartbeat(job: str, status: str, detail: str = "", duration_s: float | None = None,
                     path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _heartbeat_lock(path):
        data = _load(path)
        data[job] = {
            "status": status,  # "running" | "ok" | "error"
            "detail": detail,
            "duration_s": duration_s,
            "timestamp_utc": _now_iso(),
        }
        atomic_write_text(path, json.dumps(data, indent=2))


@contextmanager
def track(job: str, path: Path = DEFAULT_PATH):
    """with track("live_update"): ... - writes 'running' on entry, 'ok' on
    clean exit, 'error' (with the exception message) if it raises."""
    write_heartbeat(job, "running", path=path)
    start = time.monotonic()
    try:
        yield
    except Exception as e:
        write_heartbeat(job, "error", detail=str(e), duration_s=time.monotonic() - start, path=path)
        raise
    else:
        write_heartbeat(job, "ok", duration_s=time.monotonic() - start, path=path)


def read_heartbeats(path: Path = DEFAULT_PATH) -> dict:
    return _load(path)
