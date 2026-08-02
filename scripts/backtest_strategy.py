"""
Standalone backtest for ONE strategy family at a time, independent of
build_pattern_library.py's full mining run (~30 atomic + ~1300 combo
patterns across every family). Two families are registered:

  - smc: liquidity-sweep / fair-value-gap patterns (smc_patterns.py) -
    the 6 patterns added on top of support_resistance.swing_levels().
  - trend_following: patterns.trend_following_long/short - SMA(50/200)
    trend direction, RSI(14)+OBV non-correlated confirmation, ATR
    volatility filter, stop/target both anchored to THIS candle's own
    ATR (STOP_ATR_MULTIPLE for the stop, RR_RATIO x that for the
    target - never a higher-timeframe level, so nothing here can
    repaint). Designed for the 15min file specifically (see
    patterns.trend_following_long's own docstring) but, like every
    other pattern in this codebase, not hard-restricted to it in code -
    the win-rate gate decides per timeframe, same philosophy as
    everywhere else.

Why a SEPARATE script instead of just filtering pattern_library/*.json
after the fact: build_library() in build_pattern_library.py detects and
simulates EVERY pattern across ALL 6 families plus ~1300 cross-family
combos every time it runs, so re-testing a change to just one family
still means waiting on (and overwriting) the full run. This script
detects and simulates ONLY the chosen family's patterns, so it is fast
enough to rerun after every tweak to smc_patterns.py or
patterns.trend_following_*.

Same functions, same gates, same output shape as production - not a
second implementation: pattern_direction_hint(), simulate_trades(),
summarize_trades(), and the SAME news/regime conditioning helpers
build_pattern_library.py itself uses are imported and called directly
here, not reimplemented. A pattern's stats reported by this script are
computed by the exact identical code path build_pattern_library.py
would use for it inside the full run - the only thing different is
which pattern names get iterated over. Because these patterns are
ALSO detected by signal_engine.py's own base_flags concat (detect_all
already includes trend_following_long/short; detect_smc_events is
concatenated in directly) via the SAME detector functions used here,
a setup backtested by this script is mechanically the same setup
signal_engine.py will fire live - there is no second definition for
live detection to silently drift onto.

This is a research/iteration tool, not a replacement for the full
mined library: pattern_library/*.json (built by build_pattern_library.py)
remains the ONE source signal_engine.py actually reads for live
qualifying/weight decisions. Nothing written by this script is read by
signal_engine.py - rerun build_pattern_library.py (or wait for its next
scheduled run) to get a family's updated stats into the live-serving
library after using this script to iterate on it.

Usage:
    python scripts/backtest_strategy.py --strategy smc --symbol XAUUSD
    python scripts/backtest_strategy.py --strategy trend_following --symbol XAUUSD --timeframes 15min
    python scripts/backtest_strategy.py --strategy all --symbol XAUUSD
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from atomic_io import atomic_write_text  # noqa: E402
from build_pattern_library import _with_news_conditioning, _with_regime_conditioning  # noqa: E402
from heartbeat import track  # noqa: E402
from patterns import pattern_direction_hint, trend_following_long, trend_following_short  # noqa: E402
from regime import combined_regime  # noqa: E402
from risk_reward import (  # noqa: E402
    MIN_RESOLVED_SAMPLES, MIN_WIN_RATE, OOS_MIN_RESOLVED_SAMPLES, RR_RATIO,
    atr, simulate_trades, summarize_trades,
)
from smc_patterns import SMC_PATTERNS  # noqa: E402

STRATEGIES: dict[str, dict] = {
    "smc": SMC_PATTERNS,
    "trend_following": {
        "trend_following_long": trend_following_long,
        "trend_following_short": trend_following_short,
    },
}


def backtest_strategy(strategy: str, candles: pd.DataFrame,
                       events: "pd.DataFrame | None" = None) -> dict:
    """Mines ONLY STRATEGIES[strategy]'s patterns over `candles` - same
    per-pattern branch build_pattern_library.build_library() runs for
    every atomic pattern (direction-hinted patterns simulated once at
    their own direction; direction-agnostic ones simulated both ways),
    just scoped to one family instead of iterating pattern_flags.columns
    for the whole ~1300-column concatenated DataFrame."""
    patterns_map = STRATEGIES[strategy]
    atr_series = atr(candles)
    regime_labels = combined_regime(candles)

    high_impact_events = None
    if events is not None and not events.empty:
        from event_timing import impact_of
        high_impact_events = events[events["event_type"].apply(impact_of) == "high"]

    library = {"_meta": {"strategy": strategy, "rr_ratio": RR_RATIO, "min_win_rate": MIN_WIN_RATE,
                          "min_resolved_samples": MIN_RESOLVED_SAMPLES,
                          "oos_min_resolved_samples": OOS_MIN_RESOLVED_SAMPLES}}
    for name, fn in patterns_map.items():
        occurred = fn(candles).fillna(False)
        hint = pattern_direction_hint(name)

        if hint == 0:
            long_trades = simulate_trades(candles, occurred, direction=+1, atr_series=atr_series)
            short_trades = simulate_trades(candles, occurred, direction=-1, atr_series=atr_series)
            entry = {
                "direction": "ambiguous",
                "as_long": summarize_trades(long_trades),
                "as_short": summarize_trades(short_trades),
            }
            long_cond = _with_news_conditioning(long_trades, candles, high_impact_events)
            short_cond = _with_news_conditioning(short_trades, candles, high_impact_events)
            if long_cond or short_cond:
                entry["news_conditioning"] = {"as_long": long_cond, "as_short": short_cond}
            long_regime = _with_regime_conditioning(long_trades, regime_labels)
            short_regime = _with_regime_conditioning(short_trades, regime_labels)
            if long_regime or short_regime:
                entry["by_regime"] = {"as_long": long_regime, "as_short": short_regime}
        else:
            trades = simulate_trades(candles, occurred, direction=hint, atr_series=atr_series)
            entry = {
                "direction": "bullish" if hint > 0 else "bearish",
                "stats": summarize_trades(trades),
            }
            cond = _with_news_conditioning(trades, candles, high_impact_events)
            if cond:
                entry["news_conditioning"] = cond
            by_regime = _with_regime_conditioning(trades, regime_labels)
            if by_regime:
                entry["by_regime"] = by_regime

        library[name] = entry
    return library


def _qualifies(entry: dict) -> bool:
    if "stats" in entry:
        return entry["stats"].get("qualifies", False)
    return entry.get("as_long", {}).get("qualifies", False) or entry.get("as_short", {}).get("qualifies", False)


def rebuild_all(strategies: list[str], symbol: str, data_dir: Path, out_dir: Path,
                 timeframes: "list[str] | None") -> dict:
    candles_dir = data_dir / "candles"
    out_dir.mkdir(parents=True, exist_ok=True)

    timeframe_files = sorted(candles_dir.glob(f"{symbol}_*.parquet"))
    if not timeframe_files:
        raise SystemExit(f"no candle files found in {candles_dir} - run build_history.py first")
    if timeframes:
        wanted = set(timeframes)
        timeframe_files = [p for p in timeframe_files if p.stem.replace(f"{symbol}_", "") in wanted]
        if not timeframe_files:
            raise SystemExit(f"none of --timeframes {timeframes} matched files in {candles_dir}")

    events_path = data_dir / "events" / "fundamentals.parquet"
    events = pd.read_parquet(events_path) if events_path.exists() else None

    summary = {}
    for path in timeframe_files:
        tf = path.stem.replace(f"{symbol}_", "")
        candles = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
        summary[tf] = {}
        for strategy in strategies:
            print(f"{tf}/{strategy}: backtesting {len(STRATEGIES[strategy])} pattern(s) over "
                  f"{len(candles)} candles ({candles['timestamp'].min()} -> {candles['timestamp'].max()}) "
                  f"at fixed 1:{RR_RATIO:.0f} R:R")
            library = backtest_strategy(strategy, candles, events)
            out_path = out_dir / f"{strategy}_{symbol}_{tf}.json"
            atomic_write_text(out_path, json.dumps(library, indent=2, default=str))

            entries = {k: v for k, v in library.items() if k != "_meta"}
            qualifying = [n for n, v in entries.items() if _qualifies(v)]
            print(f"  -> {len(qualifying)}/{len(entries)} patterns clear the {MIN_WIN_RATE:.0%} "
                  f"win-rate gate: {qualifying or '(none)'}")
            print(f"     written to {out_path}")
            summary[tf][strategy] = {"total_patterns": len(entries), "qualifying": qualifying}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=[*STRATEGIES, "all"], default="all")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="strategy_backtests")
    parser.add_argument("--timeframes", default=None,
                         help="comma-separated (e.g. 15min or 15min,1h) - default: all available candle files")
    args = parser.parse_args()

    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    timeframes = [t.strip() for t in args.timeframes.split(",")] if args.timeframes else None
    data_dir = Path(args.data_dir)

    with track("backtest_strategy", path=data_dir / "heartbeats.json"):
        rebuild_all(strategies, args.symbol, data_dir, Path(args.out_dir), timeframes)


if __name__ == "__main__":
    main()
