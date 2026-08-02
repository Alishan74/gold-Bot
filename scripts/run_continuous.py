"""
Genuinely continuous operation: run the requested engine's live-update
cycle (fetch/import the newest market data -> recompute today's signal ->
self-heal check -> log to journal) repeatedly, FOREVER, sleeping
--interval-seconds between cycles, until stopped (Ctrl+C, or SIGTERM if
run detached) - not "live when someone happens to run it," an actual
always-on process that keeps checking the live market on its own from
the moment it starts until the moment it's told to stop.

Each cycle runs the underlying script (live_update.py or
ml_live_update.py) as a FRESH SUBPROCESS, not one long-lived in-process
loop - a clean interpreter every cycle, so nothing this system does can
leak memory or accumulate bad state across days/weeks of continuous
operation the way one giant process eventually could. A cycle that
crashes or hangs (no output for --stall-timeout seconds - same stuck-
detection scripts/supervise.py already uses for the one-shot backfill,
reimplemented here rather than imported so this script can reach in and
kill the live subprocess directly the instant a stop is requested) is
killed and logged, and the loop moves on to the NEXT cycle rather than
giving up entirely - a real feed handler doesn't permanently stop
because one tick had a network blip, it just tries again next tick.

HOW FAST CAN THIS ACTUALLY SEE NEW DATA? Bounded by the upstream data
source, not by this loop's interval:
  - Dukascopy (the default source) publishes ONE FILE PER COMPLETED UTC
    HOUR - so `--engine rules`/`--engine ml` against the default source
    genuinely has nothing NEW to find more than once an hour, no matter
    how tight this loop's interval is. The default 300s interval exists
    to catch that new hourly file promptly once Dukascopy publishes it
    (there's typically a few minutes of publish lag after the hour
    rolls over), not to invent sub-hourly data that doesn't exist.
  - Your OWN broker via mt_bridge/ IS genuinely sub-minute - the
    MetaTrader live bridge (LiveBridgeMT4.mq4 / LiveBridgeMT5.mq5) checks
    every 30s and appends any newly CLOSED candle from your broker's own
    feed. Pass --import-mt-broker-dir (and --mt-symbol) so every cycle
    also runs mt_import.py to pull in whatever the bridge has appended
    since the last cycle, before recomputing the signal - see
    mt_bridge/README.md.

Runs the rule-based and ML challenger engines as SEPARATE processes (one
instance of this script per engine, --engine rules / --engine ml), same
as every other part of this project that keeps the two systems
independent - run two terminals/services, one per engine, to watch both
continuously side by side.

Usage:
    python scripts/run_continuous.py --engine rules
    python scripts/run_continuous.py --engine ml --interval-seconds 300
    python scripts/run_continuous.py --engine rules \\
        --import-mt-broker-dir "/path/to/MQL5/Files/gold_export" --mt-symbol XAUUSD.a

Stop any time with Ctrl+C (or `kill <pid>` / SIGTERM if run detached via
nohup/systemd/tmux).
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Source-aware default polling interval - see module docstring's "HOW
# FAST CAN THIS ACTUALLY SEE NEW DATA?" section. Against Dukascopy
# (publishes once per completed UTC hour), anything much tighter than a
# minute buys nothing but more retries against an hour that isn't
# published yet. Against your own broker's live bridge (mt_bridge/,
# appends newly closed candles every ~30s), a tight interval is where
# the real latency win actually is - so this is genuinely source-aware,
# not one compromise default trying to serve both.
DEFAULT_INTERVAL_DUKASCOPY = 60.0
DEFAULT_INTERVAL_MT_BROKER = 20.0


class _StopRequested(Exception):
    pass


def _install_stop_handler():
    def handler(signum, frame):
        raise _StopRequested()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _run_cycle(cmd: list[str], stall_timeout: float, log_file) -> int | str:
    """Runs `cmd` once to completion, killing it if it goes
    `stall_timeout` seconds without producing any output (a genuinely
    hung cycle, not a slow-but-working one). Returns the exit code, or
    the string "stalled". Re-raises _StopRequested after cleanly
    terminating the child if a stop was requested mid-cycle - a
    continuous process must never leave an orphaned subprocess running
    after it's told to shut down."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

    state = {"last_output": time.monotonic()}
    lock = threading.Lock()

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
                print(f"[run_continuous] cycle stuck ({idle:.0f}s with no output) - killing this cycle", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return "stalled"
            time.sleep(1)
    except _StopRequested:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise


def _build_engine_cmd(args) -> list[str]:
    if args.engine == "rules":
        cmd = [sys.executable, str(ROOT / "src" / "live_update.py"),
               "--symbol", args.symbol, "--data-dir", args.data_dir, "--lib-dir", args.lib_dir]
        if args.skip_fundamentals:
            cmd.append("--skip-fundamentals")
        if args.no_self_heal:
            cmd.append("--no-self-heal")
    else:
        cmd = [sys.executable, str(ROOT / "ml_system" / "ml_live_update.py"),
               "--symbol", args.symbol, "--candles-dir", args.candles_dir,
               "--news-data-dir", args.news_data_dir, "--data-dir", args.data_dir,
               "--registry-dir", args.registry_dir]
        if args.no_self_heal:
            cmd.append("--no-self-heal")
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["rules", "ml"], required=True)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--interval-seconds", type=float, default=None,
                         help="sleep between cycles. Default is source-aware if not set explicitly: "
                              f"{DEFAULT_INTERVAL_MT_BROKER:.0f}s when --import-mt-broker-dir is set (matches the "
                              f"MetaTrader live bridge's own ~30s cadence - this is the genuinely fast path), "
                              f"{DEFAULT_INTERVAL_DUKASCOPY:.0f}s otherwise (Dukascopy only publishes one file per "
                              "completed UTC hour - see module docstring for why going much tighter than this "
                              "against that source doesn't find data any faster, it just retries a not-yet-"
                              "published hour more often)")
    parser.add_argument("--stall-timeout", type=float, default=600,
                         help="kill a single stuck cycle after this many seconds with no output (default 600)")
    parser.add_argument("--data-dir", default=None, help="defaults to data/ (rules) or ml_data/ (ml)")
    parser.add_argument("--lib-dir", default="pattern_library", help="rules engine only")
    parser.add_argument("--candles-dir", default="data/candles", help="ml engine only - shared candle source")
    parser.add_argument("--news-data-dir", default="data", help="ml engine only - shared news calendar")
    parser.add_argument("--registry-dir", default="ml_registry", help="ml engine only")
    parser.add_argument("--skip-fundamentals", action="store_true", help="rules engine only")
    parser.add_argument("--no-self-heal", action="store_true")
    parser.add_argument("--import-mt-broker-dir", default=None,
                         help="if set, every cycle first runs mt_import.py against this MetaTrader export "
                              "directory before the engine's own update - see mt_bridge/README.md")
    parser.add_argument("--mt-symbol", default=None, help="required if --import-mt-broker-dir is set")
    parser.add_argument("--log", default=None,
                         help="also append all cycle output here (dir created if needed) - "
                              "defaults to logs/<engine>_continuous.log")
    parser.add_argument("--max-cycles", type=int, default=0,
                         help="stop after this many cycles (0 = forever, the normal case) - mainly for testing")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = "ml_data" if args.engine == "ml" else "data"
    if args.import_mt_broker_dir and not args.mt_symbol:
        parser.error("--mt-symbol is required when --import-mt-broker-dir is set")
    interval_was_explicit = args.interval_seconds is not None
    if args.interval_seconds is None:
        args.interval_seconds = DEFAULT_INTERVAL_MT_BROKER if args.import_mt_broker_dir else DEFAULT_INTERVAL_DUKASCOPY

    log_path = Path(args.log) if args.log else ROOT / "logs" / f"{args.engine}_continuous.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _install_stop_handler()

    cycle = 0
    interval_note = "explicit" if interval_was_explicit else (
        "auto: MetaTrader broker feed cadence" if args.import_mt_broker_dir else "auto: Dukascopy cadence"
    )
    print(f"[run_continuous] starting continuous '{args.engine}' engine - checking every "
          f"{args.interval_seconds:.0f}s ({interval_note}), logging to {log_path} - Ctrl+C to stop", flush=True)

    with open(log_path, "a") as log_file:
        try:
            while True:
                cycle += 1
                now = dt.datetime.now(dt.timezone.utc).isoformat()
                header = f"\n=== cycle {cycle} @ {now} ==="
                print(f"[run_continuous]{header}", flush=True)
                log_file.write(header + "\n")
                log_file.flush()

                if args.import_mt_broker_dir:
                    import_cmd = [sys.executable, str(ROOT / "src" / "mt_import.py"),
                                  "--export-dir", args.import_mt_broker_dir, "--symbol", args.mt_symbol]
                    result = _run_cycle(import_cmd, args.stall_timeout, log_file)
                    print(f"[run_continuous] mt_import.py: {'ok' if result == 0 else f'FAILED ({result})'}",
                          flush=True)

                cmd = _build_engine_cmd(args)
                result = _run_cycle(cmd, args.stall_timeout, log_file)
                status = "ok" if result == 0 else f"FAILED ({result}) - will retry next cycle"
                print(f"[run_continuous] cycle {cycle} {status}", flush=True)

                if args.max_cycles and cycle >= args.max_cycles:
                    print(f"[run_continuous] reached --max-cycles {args.max_cycles} - stopping", flush=True)
                    break

                print(f"[run_continuous] sleeping {args.interval_seconds:.0f}s until next cycle...", flush=True)
                time.sleep(args.interval_seconds)
        except _StopRequested:
            print(f"\n[run_continuous] stop requested after {cycle} cycle(s) - shutting down cleanly", flush=True)


if __name__ == "__main__":
    main()
