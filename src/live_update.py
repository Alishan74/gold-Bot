"""
Daily/hourly job: pull the newest ticks since the last update, append them
to the historical candle files, emit today's signal, and (weekly) refresh
the pattern library so newly observed live data feeds back into future
pattern stats - "save the data we get live for future patterns".

Meant to be run on a schedule (cron) on a machine with real internet
access, e.g.:

    # every hour
    0 * * * * cd /path/to/gold-signals-bot && python src/live_update.py

    # weekly full pattern-library refresh
    0 3 * * 0 cd /path/to/gold-signals-bot && python src/build_pattern_library.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from atomic_io import atomic_write_parquet
from build_fundamentals import refresh_fundamentals
from build_history import TIMEFRAMES, fetch_all_hours, merge_with_existing, resample_ticks
from build_pattern_library import rebuild_all
from dukascopy_fetch import FetchConfig
from heartbeat import read_heartbeats, track
from news_calendar import build_upcoming_calendar
from circuit_breaker import check_circuit_breaker
from signal_engine import compute_signal, load_inputs, load_suspended
from signal_journal import load_journal, log_signal, should_self_heal, summary, update_journal


def last_candle_time(path: Path) -> dt.datetime | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["timestamp"])
    if df.empty:
        return None
    return pd.Timestamp(df["timestamp"].max()).to_pydatetime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--lib-dir", default="pattern_library")
    parser.add_argument("--discovered-dir", default="discovered_patterns")
    parser.add_argument("--lookback-hours", type=int, default=6,
                         help="always re-fetch this many trailing hours, to catch late-arriving ticks")
    parser.add_argument("--event-config", default="event_config.json")
    parser.add_argument("--skip-fundamentals", action="store_true",
                         help="skip the FRED refresh (e.g. no network / no API key on this run)")
    parser.add_argument("--no-self-heal", action="store_true",
                         help="never auto-trigger a pattern library rebuild, even if enough patterns "
                              "are decaying live or the library has gone stale (see should_self_heal)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    lib_dir = Path(args.lib_dir)
    candles_dir = data_dir / "candles"
    candles_dir.mkdir(parents=True, exist_ok=True)
    cfg = FetchConfig(symbol=args.symbol, cache_dir=data_dir / "raw_bi5")

    with track("live_update", path=data_dir / "heartbeats.json"):
        daily_path = candles_dir / f"{args.symbol}_1d.parquet"
        last_dt = last_candle_time(daily_path)  # naive (candle timestamps are naive-but-UTC - see build_history.py)
        end = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
        start = (last_dt or (end - dt.timedelta(days=1))) - dt.timedelta(hours=args.lookback_hours)

        print(f"fetching new ticks: {start.isoformat()} -> {end.isoformat()}")
        ticks, failed_hours = fetch_all_hours(start, end, cfg, workers=4)
        if failed_hours:
            print(f"  WARNING: {len(failed_hours)} hour(s) failed to fetch this run (not market "
                  f"closure) - {failed_hours}")
        if ticks.empty:
            print("no new ticks (market closed?) - nothing to append")
        else:
            for name, rule in TIMEFRAMES.items():
                new_candles = resample_ticks(ticks, rule)
                out_path = candles_dir / f"{args.symbol}_{name}.parquet"
                merged = merge_with_existing(new_candles, out_path)
                atomic_write_parquet(merged, out_path)
                print(f"  {name}: +{len(new_candles)} candles seen, {len(merged)} total")

        if not args.skip_fundamentals:
            try:
                refresh_fundamentals(Path(args.event_config), data_dir)
                build_upcoming_calendar(Path(args.event_config), data_dir)
            except Exception:
                print("fundamentals/news-calendar refresh failed - continuing with whatever "
                      "was already on disk, technical signal still runs")

        candles_by_tf, library_by_tf, events, upcoming = load_inputs(
            args.symbol, data_dir, lib_dir, discovered_dir=args.discovered_dir
        )

        # Resolve open journal entries against the freshly-appended candles
        # BEFORE logging today's signal, so a trade that just hit its
        # target/stop this run is marked closed, not left dangling as open
        # for one extra cycle.
        update_journal(candles_by_tf, data_dir)

        # Self-healing, step 1: if enough patterns are decaying live, or
        # the library just hasn't been remined in a while, rebuild it
        # automatically right now instead of waiting for a human to
        # notice the dashboard warning or remember the weekly cron. Runs
        # BEFORE computing today's signal so that signal benefits from
        # the freshly-remined library immediately, same run.
        if not args.no_self_heal and library_by_tf:
            journal_df = load_journal(data_dir)
            heartbeats = read_heartbeats(data_dir / "heartbeats.json")
            trigger, reason = should_self_heal(journal_df, lib_dir, args.symbol, heartbeats, args.discovered_dir)
            if trigger:
                print(f"SELF-HEAL: triggering automatic pattern library rebuild - {reason}")
                with track("build_pattern_library", path=data_dir / "heartbeats.json"):
                    rebuild_all(args.symbol, data_dir, lib_dir)
                candles_by_tf, library_by_tf, events, upcoming = load_inputs(
                    args.symbol, data_dir, lib_dir, discovered_dir=args.discovered_dir
                )

        # Self-healing, step 2: even on a freshly-remined library, a
        # pattern/direction with an active live losing streak stays
        # suspended until NEW live evidence (not just a remine) shows
        # it's recovered - see signal_journal.suspended_patterns().
        suspended = load_suspended(data_dir, lib_dir, args.symbol, args.discovered_dir) if library_by_tf else set()
        if suspended:
            print(f"self-heal: {len(suspended)} pattern/timeframe/direction combo(s) suspended "
                  f"(DECAYING live): {sorted(suspended)}")

        # Loaded once, reused for both the circuit breaker AND loss/win
        # attribution (context_penalty) below - both read this system's
        # own journal, no reason to re-parse the parquet file twice.
        journal = load_journal(data_dir)

        # Portfolio-level circuit breaker - checked every run regardless
        # of self-heal state, since it looks at REALIZED outcomes only
        # (see circuit_breaker.py). A tripped breaker forces HOLD
        # system-wide, overriding every individually-qualifying pattern.
        breaker = check_circuit_breaker(journal)
        if breaker["tripped"]:
            print(f"CIRCUIT BREAKER TRIPPED: {breaker['reasons']} - forcing HOLD system-wide until a human clears it")

        if library_by_tf:
            signal = compute_signal(candles_by_tf, library_by_tf, events, upcoming,
                                    suspended=suspended, circuit_breaker=breaker, journal=journal)
            print("current signal:")
            print(json.dumps(signal.to_dict(), indent=2))
            log_signal(signal.to_dict(), candles_by_tf, data_dir)
        else:
            print("no pattern library yet - run build_pattern_library.py once you have enough history")

        print("signal journal:", json.dumps(summary(load_journal(data_dir))))


if __name__ == "__main__":
    main()
