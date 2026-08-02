"""
Mine the historical candle data for pattern edge under the system's fixed
1:4 risk:reward trade structure (see risk_reward.py). For every
occurrence of every pattern, simulate the trade that pattern would have
called (long or short, stop/target from ATR at 1:4 R:R) and record
whether it actually hit its target before its stop.

A pattern is only usable as a live signal source if it clears BOTH hard
gates: risk:reward fixed at 1:4 (true for every simulated trade, by
construction) and historical win rate >= 60% on enough samples
(`qualifies: true` in the output). Patterns that don't clear the bar are
still recorded, but flagged `qualifies: false`, and signal_engine.py
refuses to emit anything from them.

If data/events/fundamentals.parquet exists (built by
build_fundamentals.py), fundamental events (CPI, PCE, NFP, GDP, FOMC
decisions) are mined through the exact same pipeline as candlestick
patterns - same trade simulation, same hard gates, same JSON shape, just
prefixed `fundamental_*` in the output so you can tell them apart.

Trading-session opens/closes (Sydney/Tokyo/London/New York) are mined
the same way too, prefixed `session_*` - these need no external data,
so they're always included.

Support/resistance reactions (support_resistance.py) are mined the same
way too - swing-high/low rejections, round-number ($50 grid) rejections/
bounces, and daily-pivot R1/S1 rejections/bounces. Same trade
simulation, same hard gates, same JSON shape as every other pattern -
see support_resistance.py's module docstring for the full look-ahead-
safety reasoning behind each of the three level types. Needs no
external data either, always included.

Smart Money Concepts (smc_patterns.py) are mined the same way too -
liquidity sweeps of a confirmed swing level and fair value gaps (3-candle
imbalances), standalone and combined into one sweep+gap trigger. Same
trade simulation, same hard gates, same JSON shape as every other
pattern - see smc_patterns.py's module docstring for why only these two
SMC concepts (of the many loosely-defined ones in retail SMC folklore)
were picked: both have an unambiguous, look-ahead-safe, mechanically
checkable definition. Needs no external data either, always included.

News conditioning: for every ATOMIC pattern (not combos - see below), IF
historical fundamentals are available, each occurrence is also tagged
with whether a high-impact event (CPI/PCE/NFP/GDP/FOMC) fell inside that
specific trade's entry->resolution window, and win rate is reported
separately for "no_news" vs "news_in_window" under each entry's
"news_conditioning" key. This is what lets a live signal say
"historically, when high-impact news lands mid-trade, this pattern's win
rate drops from X% to Y%" - each side still has to independently clear
the same hard gate to be trusted (small samples just show
`qualifies: false`, not a fabricated rate).

Combo patterns (combo_patterns.py): every valid cross-family PAIR of
already-detected patterns (e.g. bullish_engulfing + session_london_open
firing on the same candle) is ALSO mined through the identical 1:4 R:R /
hard-gate pipeline, prefixed `combo__` in the output. This is the main
lever for getting MORE qualifying signals without weakening the gate at
all: genuine confluence between two conditions is rarer but typically
more selective than either alone. Combos use higher sample thresholds
(risk_reward.COMBO_MIN_RESOLVED_SAMPLES/COMBO_OOS_MIN_RESOLVED_SAMPLES)
than atomic patterns and skip news-conditioning (their samples are
already smaller by construction; splitting further into a third
no_news/news_in_window cut would almost never have enough left to say
anything) - see combo_patterns.py's docstring for the full rationale,
including the multiple-comparisons caveat.

Regime conditioning (regime.py): for every ATOMIC pattern (not combos -
same rationale as news-conditioning above: combo samples are already
small, splitting further into up to 6 regime buckets would rarely leave
enough to say anything), each occurrence is also tagged with the
combined volatility/trend regime (e.g. "HIGH_TRENDING") active at that
trade's entry, and win rate is reported separately per regime under
"by_regime". This is what lets signal_engine.py check "does this pattern
actually have a proven edge in the CURRENT market regime, or only in its
blended-across-20-years average" - a pattern whose edge only ever
existed in one regime can pass the overall gate while quietly not
working right now if the market has since shifted into a different one.
Regime-specific stats influence live signal WEIGHT, not a hard gate - see
signal_engine.py for why (an additional hard gate here would cut deeply
into signal frequency on an already-conservative system, for a benefit
better captured as a scoring adjustment plus the self-healing suspension
mechanism in signal_journal.py, which reacts to actual LIVE
underperformance rather than a backtested regime split alone).

This only needs the historical candles (and, optionally, the
fundamentals table) already downloaded locally - no network access - so
it runs fine locally or in a sandbox.

Usage:
    python src/build_pattern_library.py --symbol XAUUSD
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from atomic_io import atomic_write_text
from combo_patterns import COMBO_PAIRS_BY_NAME, combo_direction_hint, detect_combo_events
from fundamental_patterns import detect_fundamental_events
from heartbeat import track
from patterns import detect_all, pattern_direction_hint
from regime import combined_regime
from risk_reward import (
    COMBO_MIN_RESOLVED_SAMPLES, COMBO_OOS_MIN_RESOLVED_SAMPLES, MIN_RESOLVED_SAMPLES,
    MIN_WIN_RATE, OOS_MIN_RESOLVED_SAMPLES, RR_RATIO, atr, simulate_trades, summarize_trades,
    summarize_trades_by_regime, summarize_trades_conditioned, tag_news_in_window, tag_regime,
)
from session_patterns import detect_session_events
from smc_patterns import detect_smc_events
from support_resistance import detect_support_resistance_events


def _with_news_conditioning(trades: pd.DataFrame, candles: pd.DataFrame,
                             high_impact_events: pd.DataFrame | None) -> dict | None:
    if high_impact_events is None or high_impact_events.empty or trades.empty:
        return None
    tagged = tag_news_in_window(trades, candles, high_impact_events)
    return summarize_trades_conditioned(tagged)


def _with_regime_conditioning(trades: pd.DataFrame, regime_labels: pd.Series) -> dict | None:
    if trades.empty:
        return None
    tagged = tag_regime(trades, regime_labels)
    by_regime = summarize_trades_by_regime(tagged)
    return by_regime or None


def compute_pattern_flags(candles: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every pattern this system knows about, one boolean column each:
    candlestick + indicator (`detect_all`), fundamental, session,
    support/resistance, smc, and every valid cross-family combo
    (`detect_combo_events`, built from the same base_flags) - the SAME
    concatenation `build_library()` below has always used, pulled out on
    its own so `causal_autopsy.py` can compute the identical flag set
    (same patterns, same definitions, same combo list) without a second,
    potentially-drifting implementation. This is the ONE place "what
    patterns exist and how are they detected" is assembled; anything
    that needs "every pattern's occurrence series" should call this,
    not reconstruct its own concat."""
    base_flags = pd.concat(
        [detect_all(candles), detect_fundamental_events(candles, events),
         detect_session_events(candles), detect_support_resistance_events(candles),
         detect_smc_events(candles)],
        axis=1,
    )
    combo_flags = detect_combo_events(base_flags)
    return pd.concat([base_flags, combo_flags], axis=1)


def pattern_direction_and_gates(name: str) -> tuple[int, bool, int, int]:
    """(direction_hint, is_combo, min_resolved, oos_min_resolved) for any
    column name compute_pattern_flags() can produce - the SAME branching
    build_library()'s own loop below uses to decide direction and which
    sample-size gate applies (combos need more, see risk_reward.py's
    COMBO_MIN_RESOLVED_SAMPLES docstring), pulled out so causal_autopsy.py
    can look up the exact same thing for a pattern without re-deriving
    the is-this-a-combo/what's-its-direction logic a second time."""
    if name in COMBO_PAIRS_BY_NAME:
        name_a, name_b = COMBO_PAIRS_BY_NAME[name]
        return combo_direction_hint(name_a, name_b), True, COMBO_MIN_RESOLVED_SAMPLES, COMBO_OOS_MIN_RESOLVED_SAMPLES
    return pattern_direction_hint(name), False, MIN_RESOLVED_SAMPLES, OOS_MIN_RESOLVED_SAMPLES


def build_library(candles: pd.DataFrame, events: pd.DataFrame | None = None) -> dict:
    pattern_flags = compute_pattern_flags(candles, events)

    high_impact_events = None
    if events is not None and not events.empty:
        from event_timing import impact_of
        high_impact_events = events[events["event_type"].apply(impact_of) == "high"]

    # Computed ONCE and reused for every pattern - atomic (~30) plus
    # combo (~700+) - instead of every single simulate_trades() call
    # recomputing the identical rolling ATR over the full candle history.
    atr_series = atr(candles)
    regime_labels = combined_regime(candles)

    library = {"_meta": {"rr_ratio": RR_RATIO, "min_win_rate": MIN_WIN_RATE,
                          "min_resolved_samples": MIN_RESOLVED_SAMPLES,
                          "oos_min_resolved_samples": OOS_MIN_RESOLVED_SAMPLES,
                          "combo_min_resolved_samples": COMBO_MIN_RESOLVED_SAMPLES,
                          "combo_oos_min_resolved_samples": COMBO_OOS_MIN_RESOLVED_SAMPLES}}
    for name in pattern_flags.columns:
        hint, is_combo, min_resolved, oos_min_resolved = pattern_direction_and_gates(name)
        if is_combo:
            name_a, name_b = COMBO_PAIRS_BY_NAME[name]

        if hint == 0:
            long_trades = simulate_trades(candles, pattern_flags[name], direction=+1, atr_series=atr_series)
            short_trades = simulate_trades(candles, pattern_flags[name], direction=-1, atr_series=atr_series)
            entry = {
                "direction": "ambiguous",
                "as_long": summarize_trades(long_trades, min_resolved, oos_min_resolved),
                "as_short": summarize_trades(short_trades, min_resolved, oos_min_resolved),
            }
            if not is_combo:
                long_cond = _with_news_conditioning(long_trades, candles, high_impact_events)
                short_cond = _with_news_conditioning(short_trades, candles, high_impact_events)
                if long_cond or short_cond:
                    entry["news_conditioning"] = {"as_long": long_cond, "as_short": short_cond}
                long_regime = _with_regime_conditioning(long_trades, regime_labels)
                short_regime = _with_regime_conditioning(short_trades, regime_labels)
                if long_regime or short_regime:
                    entry["by_regime"] = {"as_long": long_regime, "as_short": short_regime}
            if is_combo:
                entry["combo_of"] = [name_a, name_b]
            library[name] = entry
        else:
            trades = simulate_trades(candles, pattern_flags[name], direction=hint, atr_series=atr_series)
            entry = {
                "direction": "bullish" if hint > 0 else "bearish",
                "stats": summarize_trades(trades, min_resolved, oos_min_resolved),
            }
            if not is_combo:
                cond = _with_news_conditioning(trades, candles, high_impact_events)
                if cond:
                    entry["news_conditioning"] = cond
                by_regime = _with_regime_conditioning(trades, regime_labels)
                if by_regime:
                    entry["by_regime"] = by_regime
            if is_combo:
                entry["combo_of"] = [name_a, name_b]
            library[name] = entry
    return library


def _qualifies(entry: dict) -> bool:
    if "stats" in entry:
        return entry["stats"].get("qualifies", False)
    return entry.get("as_long", {}).get("qualifies", False) or entry.get("as_short", {}).get("qualifies", False)


def rebuild_all(symbol: str, data_dir: Path, out_dir: Path) -> dict:
    """Mine every timeframe's candle file into pattern_library/*.json.
    Pure function - no argparse, no heartbeat wrapping - so both main()
    below and live_update.py's automatic "enough patterns are decaying
    live, re-mine now" self-heal trigger (signal_journal.should_self_heal)
    can call the exact same rebuild logic identically, instead of the
    self-heal path having to shell out to this script as a subprocess or
    duplicate the loop. Returns a per-timeframe summary dict."""
    candles_dir = data_dir / "candles"
    out_dir.mkdir(parents=True, exist_ok=True)

    timeframe_files = sorted(candles_dir.glob(f"{symbol}_*.parquet"))
    if not timeframe_files:
        raise SystemExit(f"no candle files found in {candles_dir} - run build_history.py first")

    events_path = data_dir / "events" / "fundamentals.parquet"
    events = pd.read_parquet(events_path) if events_path.exists() else None
    if events is not None:
        print(f"loaded {len(events)} fundamental events from {events_path}")
    else:
        print(f"no fundamentals found at {events_path} - mining candlestick/indicator/session "
              f"patterns only, no news conditioning (run build_fundamentals.py to add CPI/PCE/NFP/GDP/FOMC)")

    summary = {}
    for path in timeframe_files:
        tf = path.stem.replace(f"{symbol}_", "")
        candles = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
        print(f"{tf}: mining patterns over {len(candles)} candles "
              f"({candles['timestamp'].min()} -> {candles['timestamp'].max()}) "
              f"at fixed 1:{RR_RATIO:.0f} R:R")

        library = build_library(candles, events)
        out_path = out_dir / f"{symbol}_{tf}.json"
        atomic_write_text(out_path, json.dumps(library, indent=2, default=str))

        pattern_entries = {k: v for k, v in library.items() if k != "_meta"}
        qualifying = [name for name, v in pattern_entries.items() if _qualifies(v)]
        atomic_qualifying = [n for n in qualifying if n not in COMBO_PAIRS_BY_NAME]
        combo_qualifying = [n for n in qualifying if n in COMBO_PAIRS_BY_NAME]
        n_atomic = sum(1 for n in pattern_entries if n not in COMBO_PAIRS_BY_NAME)
        n_combo = sum(1 for n in pattern_entries if n in COMBO_PAIRS_BY_NAME)
        print(f"  -> {len(qualifying)}/{len(pattern_entries)} patterns clear the "
              f"{MIN_WIN_RATE:.0%} win-rate gate at 1:{RR_RATIO:.0f} R:R "
              f"({len(atomic_qualifying)}/{n_atomic} atomic, "
              f"{len(combo_qualifying)}/{n_combo} combo)")
        print(f"     atomic: {atomic_qualifying or '(none)'}")
        print(f"     combo:  {combo_qualifying or '(none)'}")
        print(f"     written to {out_path}")
        summary[tf] = {
            "total_patterns": len(pattern_entries), "qualifying_count": len(qualifying),
            "atomic_qualifying": atomic_qualifying, "combo_qualifying": combo_qualifying,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="pattern_library")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with track("build_pattern_library", path=data_dir / "heartbeats.json"):
        rebuild_all(args.symbol, data_dir, Path(args.out_dir))


if __name__ == "__main__":
    main()
