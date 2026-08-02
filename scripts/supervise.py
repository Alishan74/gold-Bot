"""
Run a command under a restart-on-crash / restart-on-stall watchdog.

Built for build_history.py's multi-hour Dukascopy backfill - the one
command in this pipeline most likely to get interrupted (a flaky
connection, a laptop sleeping, an SSH session dropping) and the most
expensive to lose progress on - but works for any script in this repo,
including a long-running dashboard/live_update loop.

Why "just restart the same command" is actually safe here, not a hack:
  - build_history.py caches each hour to data/raw_bi5/ ONLY after it's
    downloaded AND decoded AND passed the price sanity check (see
    dukascopy_fetch.py) - re-running the exact same command re-scans the
    same date range, but every already-cached hour is read from disk in
    milliseconds instead of re-fetched from Dukascopy. That's what "start
    fetching exactly from where it left off" means in practice: there's
    no separate resume pointer to manage, the cache IS the resume state.
  - Every file this pipeline writes (candle parquet files, pattern
    library JSON, the signal journal, heartbeats.json) is written
    atomically now (atomic_io.py) - a kill mid-write leaves the previous
    good file in place, never a truncated one that would break the next
    attempt at reading it.
  - build_pattern_library.py is a pure local recomputation (no network) -
    restarting it costs CPU time, not external requests or rate limits.

Two failure modes this watches for:
  1. The wrapped process EXITS non-zero (crash, Ctrl+C, network retries
     exhausted and the script itself gave up, terminal closed) -
     restart with exponential backoff (capped at --max-backoff).
  2. The wrapped process is STUCK - no new stdout/stderr for
     --stall-timeout seconds (default 600 = 10min). A genuinely slow
     step (mining ~824 patterns over 20 years of 1min candles) keeps
     printing progress; a truly hung one doesn't. Treated the same as a
     crash: killed and restarted.
Runs forever by default (stops only once the command exits 0), or up to
--max-restarts consecutive failures if you'd rather it give up and let
you look at what's wrong.

The child's stdout is forced unbuffered (PYTHONUNBUFFERED=1) regardless
of whether the wrapped command remembers `-u` itself - otherwise a
buffered child can go quiet for minutes at a time even while working
normally, which would trip stall detection on a false premise.

Usage:
    python scripts/supervise.py -- python src/build_history.py --years 20 --workers 8
    python scripts/supervise.py --stall-timeout 300 -- python src/build_pattern_library.py
    python scripts/supervise.py --log logs/live_update.log -- python src/live_update.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def _run_once(cmd: list[str], stall_timeout: float, log_path: Path | None):
    state = {"last_output": time.monotonic()}
    lock = threading.Lock()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
    )

    log_file = open(log_path, "a") if log_path else None

    def reader():
        for line in proc.stdout:
            with lock:
                state["last_output"] = time.monotonic()
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_file:
                log_file.write(line)
                log_file.flush()

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                t.join(timeout=5)
                return ret
            with lock:
                idle = time.monotonic() - state["last_output"]
            if idle > stall_timeout:
                print(f"[supervise] no output for {idle:.0f}s (> {stall_timeout:.0f}s) - "
                      f"treating as stuck, killing", file=sys.stderr, flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return "stalled"
            time.sleep(2)
    finally:
        if log_file:
            log_file.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stall-timeout", type=float, default=600,
                         help="seconds with no output before treating the process as stuck (default 600 = 10min)")
    parser.add_argument("--max-restarts", type=int, default=0,
                         help="give up after this many consecutive failures (default 0 = unlimited)")
    parser.add_argument("--initial-backoff", type=float, default=5.0,
                         help="seconds to wait before the first retry (default 5)")
    parser.add_argument("--max-backoff", type=float, default=300.0,
                         help="cap on the exponential backoff between retries (default 300 = 5min)")
    parser.add_argument("--log", default=None,
                         help="also append all wrapped-command output to this file (dir created if needed)")
    parser.add_argument("command", nargs="+", help="the command to run - put -- before it")
    args = parser.parse_args()

    cmd = args.command
    if not cmd:
        parser.error("no command given - usage: supervise.py [options] -- <command> [args...]")

    log_path = Path(args.log) if args.log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    attempt = 0
    backoff = args.initial_backoff
    while True:
        attempt += 1
        print(f"[supervise] attempt {attempt}: {' '.join(cmd)}", flush=True)
        result = _run_once(cmd, args.stall_timeout, log_path)

        if result == 0:
            print(f"[supervise] command finished successfully after {attempt} attempt(s)", flush=True)
            return

        reason = "stalled" if result == "stalled" else f"exited with code {result}"
        print(f"[supervise] {reason}", flush=True)

        if args.max_restarts and attempt >= args.max_restarts:
            print(f"[supervise] giving up after {attempt} attempt(s) (--max-restarts {args.max_restarts})",
                  flush=True)
            sys.exit(1)

        print(f"[supervise] waiting {backoff:.0f}s before retry...", flush=True)
        time.sleep(backoff)
        backoff = min(backoff * 2, args.max_backoff)


if __name__ == "__main__":
    main()
