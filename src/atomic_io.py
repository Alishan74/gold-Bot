"""
Atomic file writes: write to a temp file in the SAME directory as the
real target, then os.replace() into place. os.replace() is atomic on
both POSIX and Windows (MoveFileExW with MOVEFILE_REPLACE_EXISTING) - a
reader can never observe a half-written file, only the old complete one
or the new complete one.

Why this matters here specifically: scripts/supervise.py exists to kill
and restart a stuck/crashed pipeline command automatically, and every
command in this pipeline is designed to resume cleanly after a restart
(build_history.py's per-hour tick cache, build_pattern_library.py's
per-timeframe rebuild, the signal journal). None of that resumability
means anything if the restart itself finds a CORRUPTED file, because the
previous run got killed mid-write to it - `pd.read_parquet()` and
`json.loads()` both raise on a truncated file, which would turn "restart
and continue" into "restart and fail forever on a broken file." Writing
through a temp file + atomic rename closes that window: a kill at any
point during the write leaves either the untouched old file or the
fully-written new one, never something in between.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _replace_via_tmp(path: Path, write_fn) -> None:
    """write_fn(tmp_path: str) does the actual write to a temp path in
    the same directory as `path`; only once that returns successfully
    does the temp file get atomically renamed into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write_fn(tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    def _write(tmp_name):
        with open(tmp_name, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    _replace_via_tmp(path, _write)


def atomic_write_parquet(df, path: Path, **kwargs) -> None:
    kwargs.setdefault("index", False)
    _replace_via_tmp(path, lambda tmp_name: df.to_parquet(tmp_name, **kwargs))
