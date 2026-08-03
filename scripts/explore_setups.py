"""
Exploratory pattern search: run the SAME beam-search "self pattern
maker" as discover_patterns.py (discovery_search.search_conjunctions()
over discovery_primitives.py's 226-primitive catalog), but WITHOUT the
production pipeline's fixed 1:4 R:R / 60% win-rate assumptions baked
into what gets kept.

Why this exists: discover_patterns.py (like build_pattern_library.py)
scores and prunes everything against a fixed 1:4 R:R structure with a
demanding win-rate floor at every step (MIN_START_SCORE=0.45 just to
seed the beam, MIN_IMPROVEMENT to grow past one primitive, then FDR +
blind confirmation on top of that). On this project's real 20-year
XAUUSD history, that pipeline (and the hand-picked catalog before it)
found ZERO patterns clearing the bar - not because of a bug (verified:
both search methods produced a win-rate distribution consistent with
honest, correctly-computed no-edge trading, not a broken one), but
because 60%-win-rate-at-1:4-R:R is a very high bar for a single
technical conjunction to clear on a real, liquid market.

This script asks a different question: "what conjunctions does the
search actually FIND, at all, before any R:R-specific bar gets applied -
and how do THOSE specific setups actually perform across several
different R:R structures?" It does this by:
  1. Running discovery_search.search_conjunctions() with the win-rate-
     based pruning thresholds (MIN_START_SCORE, MIN_IMPROVEMENT) opened
     up (defaults here: 0.0 / 0.0, i.e. "keep growing as long as it
     doesn't get WORSE" rather than "only keep growing if it's already
     clearing a bar tuned for 1:4"), with capture_all=True so every
     multi-primitive conjunction the search actually built is visible,
     not just whatever the (still-real, still fixed-RR) worst-era metric
     ranks as survivors of the DEFAULT thresholds.
  2. Independently re-simulating each surfaced conjunction's exact same
     occurrences at SEVERAL different R:R ratios (--rr-ratios) via
     risk_reward.simulate_trades()'s new `rr_ratio` parameter - a
     genuine re-simulation each time (changing the target distance
     changes which candle gets hit first), never a rescale of the 1:4
     numbers.
  3. Writing every result - win rate, resolved count, expectancy before
     AND after spread cost, at every tested R:R - to disk for manual
     review. Nothing here is auto-accepted into discovered_patterns/ or
     pattern_library/ (which signal_engine.py actually reads for live
     signals) - this is a SEPARATE, clearly-labeled research output
     (explored_setups/) for a human to filter and decide from, exactly
     because loosening the search's own pruning bar this much reopens
     the "test enough things and something looks good by chance" risk
     the rest of this system's design works hard to close. Treat
     anything here as a LEAD to investigate, never as a validated signal.

Cost warning: opening up MIN_START_SCORE/MIN_IMPROVEMENT removes the
early-stopping that made discover_patterns.py's default run finish in
well under a minute per timeframe on this project's real data (the beam
kept going empty almost immediately under the strict thresholds - that
IS the fast path). With looser thresholds the search is far more likely
to keep a full beam alive through every depth, meaning meaningfully more
conjunctions actually get scored. On the smaller timeframes (1d/4h/1h)
this is still fast; on 1min/5min (millions of candles, primitives firing
far more often) it can take much longer. Start with the small
timeframes, and wrap 1min/5min in scripts/supervise.py if running them
unattended.

Usage:
    python scripts/explore_setups.py --symbol XAUUSD --timeframes 1d,4h,1h
    python scripts/explore_setups.py --symbol XAUUSD --timeframes 1min --min-start-score 0.0 --min-improvement 0.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from atomic_io import atomic_write_text  # noqa: E402
from discovery_primitives import PRIMITIVES  # noqa: E402
from discovery_search import search_conjunctions  # noqa: E402
from event_timing import impact_of  # noqa: E402
from heartbeat import track  # noqa: E402
from risk_reward import MIN_RESOLVED_SAMPLES, MAX_LOOKAHEAD, atr, simulate_trades, summarize_trades  # noqa: E402

NEWS_NOTE = (
    "Stats below are computed EXCLUDING any occurrence whose trade window "
    "(entry -> resolution) overlapped a high-impact CPI/PCE/NFP/GDP/FOMC "
    "release, per --exclude-news-window (on by default whenever fundamentals "
    "data is available). This isolates each technical setup's own edge from "
    "news-driven volatility - but it also means a LIVE occurrence of one of "
    "these setups that happens to have real news landing in its window was "
    "never represented in this backtest at all. High-impact news can still "
    "cause a historically-clean technical pattern to fail in ways this data "
    "never modeled - treat news-adjacent occurrences of any setup here with "
    "extra caution, not as backtested."
)


def _dedup(conjunctions: list) -> list:
    """Same de-dup rule search_conjunctions() itself uses for
    final_candidates: a conjunction reachable via different extension
    orders keeps only its highest-scoring instance."""
    seen: dict[tuple[frozenset, int], object] = {}
    for c in conjunctions:
        key = (frozenset(c.primitives), c.direction)
        if key not in seen or c.worst_era_score > seen[key].worst_era_score:
            seen[key] = c
    return list(seen.values())


def _high_impact_events(events: "pd.DataFrame | None") -> "pd.DataFrame | None":
    if events is None or events.empty:
        return None
    high = events[events["event_type"].apply(impact_of) == "high"]
    return high if not high.empty else None


CONTEXT_SERIES_NAMES = ("dxy", "real_yield_10y")


def _load_context_series(data_dir: Path) -> dict[str, pd.DataFrame]:
    """build_fundamentals.py's continuous background series (DXY, 10y
    real yield) - previously only ever loaded to build the fundamental_*
    SURPRISE features (see discovery_primitives.py's fundamental-surprise
    family docstring); the raw LEVEL/TREND of these series was never
    itself exposed to the search at all until the carry family below."""
    context_dir = data_dir / "context"
    out = {}
    for name in CONTEXT_SERIES_NAMES:
        path = context_dir / f"{name}.parquet"
        if path.exists():
            out[name] = pd.read_parquet(path).sort_values("date")
    return out


def _attach_context_series(candles: pd.DataFrame, context: "dict[str, pd.DataFrame]") -> pd.DataFrame:
    """Merges each daily context series onto intraday candles as a new
    column (same name as the series) via merge_asof(direction="backward")
    - every candle gets the most recent PRIOR (or same-day) daily value,
    NEVER a future one, so this is look-ahead-safe the same way every
    other join in this codebase (build_fundamentals.py's own release-date
    alignment, event_autopsy's news-window tagging) is. A context series
    with no file on disk is simply absent as a column - discovery_
    primitives.py's carry family primitives check for the column's
    presence and return an inert (all-False) series when it's missing,
    rather than crashing, so this still works on data without the merge
    applied (e.g. before build_fundamentals.py has ever been run)."""
    if not context:
        return candles
    candles = candles.sort_values("timestamp").reset_index(drop=True)
    for name, series_df in context.items():
        candles = pd.merge_asof(
            candles, series_df.rename(columns={"date": "timestamp"}),
            on="timestamp", direction="backward",
        )
    return candles


def _news_in_window_mask(trades: pd.DataFrame, candles: pd.DataFrame, high_impact: pd.DataFrame) -> np.ndarray:
    """Same semantics as risk_reward.tag_news_in_window() (does a
    high-impact event's datetime_utc fall in [entry_time, resolution_time]
    inclusive, where an unresolved trade's window ends at its
    max-lookahead cutoff like tag_news_in_window's own convention) but
    computed via a single vectorized numpy searchsorted over ALL trades
    at once instead of a Python loop calling events_in_window() per row.
    tag_news_in_window() is fine at build_pattern_library.py's scale
    (~50 atomic patterns, evaluated once each) - this script re-evaluates
    news exposure for potentially thousands of conjunction x R:R
    combinations, each with up to hundreds of thousands of occurrences on
    1min/5min, where the per-row version becomes a real bottleneck."""
    if trades.empty:
        return np.zeros(0, dtype=bool)
    ev_ts = np.sort(pd.to_datetime(high_impact["datetime_utc"]).to_numpy())
    cd_ts = pd.to_datetime(candles["timestamp"]).to_numpy()
    n = len(candles)

    start_idx = trades["index"].to_numpy()
    resolved = trades["resolved_index"]
    resolved_known = resolved.notna().to_numpy()
    end_idx = np.where(
        resolved_known,
        resolved.fillna(0).to_numpy().astype(int),
        np.minimum(start_idx + MAX_LOOKAHEAD, n - 1),
    )
    start_t = cd_ts[start_idx]
    end_t = cd_ts[end_idx]

    lo = np.searchsorted(ev_ts, start_t, side="left")
    hi = np.searchsorted(ev_ts, end_t, side="right")
    return hi > lo


def explore_timeframe(candles: pd.DataFrame, events: "pd.DataFrame | None",
                       rr_ratios: list[float], min_resolved: int,
                       beam_width: int, max_depth: int, min_depth: int,
                       min_start_score: float, min_improvement: float,
                       technicals_only: bool = False,
                       exclude_news_window: bool = True,
                       seed_families: "list[str] | None" = None,
                       checkpoint_path: "Path | None" = None,
                       checkpoint_every: int = 200) -> tuple[list[dict], int]:
    atr_series = atr(candles)
    discovery_end = len(candles)  # no held-out confirmation slice - this is exploration, not final validation

    base_primitives = [p for p in PRIMITIVES if p.family != "fundamental"] if technicals_only else None
    high_impact = _high_impact_events(events) if exclude_news_window else None

    # seed_families (see discovery_search.search_conjunctions' seed_primitives
    # docstring): restricts which primitives may START a conjunction, NOT
    # what a conjunction can grow to include - lets one very large timeframe
    # be searched in several smaller, wall-clock-bounded batches (one per
    # family group) without narrowing what any single search actually
    # explores. merge_explored_setups.py unions multiple batches' output
    # files for the same timeframe back into one.
    seed_primitives = None
    if seed_families is not None:
        pool = base_primitives if base_primitives is not None else PRIMITIVES
        seed_primitives = [p for p in pool if p.family in seed_families]

    _final, n_tested, all_scored = search_conjunctions(
        candles, events, atr_series, discovery_end,
        beam_width=beam_width, max_depth=max_depth, min_depth=min_depth,
        min_start_score=min_start_score, min_improvement=min_improvement,
        capture_all=True, base_primitives=base_primitives, seed_primitives=seed_primitives,
        checkpoint_path=checkpoint_path, checkpoint_every=checkpoint_every,
    )

    # Only genuine multi-primitive conjunctions - a lone primitive was
    # never eligible to become a final pattern in the strict pipeline
    # either (MIN_DEPTH=2), and reporting every single-primitive score
    # here would mostly just be noise at this sample size.
    multi = [c for c in all_scored if len(c.primitives) >= max(2, min_depth)]
    deduped = _dedup(multi)

    rows = []
    for conj in deduped:
        per_rr = {}
        best_after_costs = None
        for rr in rr_ratios:
            trades = simulate_trades(candles, conj.occurred, conj.direction, atr_series=atr_series, rr_ratio=rr)
            n_all_occurrences = len(trades)
            if high_impact is not None and not trades.empty:
                news_mask = _news_in_window_mask(trades, candles, high_impact)
                trades = trades[~news_mask]
            stats = summarize_trades(trades, min_resolved=min_resolved,
                                      oos_min_resolved=max(5, min_resolved // 3))
            per_rr[str(rr)] = {
                "win_rate": stats.get("win_rate"),
                "resolved": stats.get("resolved", 0),
                "n_excluded_for_news": n_all_occurrences - len(trades),
                "expectancy_r": stats.get("expectancy_r"),
                "expectancy_r_after_costs": stats.get("expectancy_r_after_costs"),
                "oos_win_rate": (stats.get("out_of_sample") or {}).get("win_rate"),
                "oos_resolved": (stats.get("out_of_sample") or {}).get("resolved", 0),
            }
            eac = stats.get("expectancy_r_after_costs")
            # Only a real candidate for "best" if THIS r:r's own resolved
            # count actually clears min_resolved - expectancy_r_after_costs
            # is computed on any nonzero sample regardless of size (same as
            # every other stat in this system), so without this check a
            # 3-trade fluke at one r:r could win out over a well-sampled
            # result at another purely because it happened to be luckier.
            if eac is not None and stats.get("resolved", 0) >= min_resolved and (
                best_after_costs is None or eac > best_after_costs[1]
            ):
                best_after_costs = (rr, eac)

        if best_after_costs is None:
            continue  # never resolved min_resolved trades at ANY tested R:R - not enough data to say anything

        rows.append({
            "primitives": list(conj.primitives),
            "direction": "bullish" if conj.direction > 0 else "bearish",
            "families": sorted(conj.families),
            "discovery_worst_era_score": conj.worst_era_score,
            "best_rr_ratio": best_after_costs[0],
            "best_expectancy_r_after_costs": round(best_after_costs[1], 4),
            "by_rr_ratio": per_rr,
        })

    rows.sort(key=lambda r: r["best_expectancy_r_after_costs"], reverse=True)
    return rows, n_tested


def rebuild_all(symbol: str, data_dir: Path, out_dir: Path, timeframes: "list[str] | None",
                 rr_ratios: list[float], min_resolved: int, top_n: int,
                 beam_width: int, max_depth: int, min_depth: int,
                 min_start_score: float, min_improvement: float,
                 technicals_only: bool = False, exclude_news_window: bool = True,
                 seed_families: "list[str] | None" = None,
                 checkpoint_dir: "Path | None" = None, checkpoint_every: int = 200) -> dict:
    candles_dir = data_dir / "candles"
    out_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
    news_filtering_active = exclude_news_window and _high_impact_events(events) is not None
    if exclude_news_window and events is None:
        print("note: --exclude-news-window is on but no fundamentals data was found "
              "(data/events/fundamentals.parquet) - nothing to filter, running unfiltered.")

    context = _load_context_series(data_dir)
    if not context:
        print("note: no DXY/real-yield context data found (data/context/*.parquet) - "
              "carry-family primitives will be inert for this run.")

    summary = {}
    for path in timeframe_files:
        tf = path.stem.replace(f"{symbol}_", "")
        candles = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
        candles = _attach_context_series(candles, context)
        print(f"{tf}: exploring over {len(candles)} candles "
              f"({candles['timestamp'].min()} -> {candles['timestamp'].max()}) "
              f"at R:R in {rr_ratios} (beam_width={beam_width}, max_depth={max_depth}, "
              f"min_start_score={min_start_score}, min_improvement={min_improvement}, "
              f"technicals_only={technicals_only}, exclude_news_window={news_filtering_active}, "
              f"context_series={list(context.keys())})")

        checkpoint_path = (checkpoint_dir / f"{symbol}_{tf}.checkpoint.json") if checkpoint_dir is not None else None
        rows, n_tested = explore_timeframe(
            candles, events, rr_ratios, min_resolved,
            beam_width, max_depth, min_depth, min_start_score, min_improvement,
            technicals_only=technicals_only, exclude_news_window=exclude_news_window,
            seed_families=seed_families, checkpoint_path=checkpoint_path, checkpoint_every=checkpoint_every,
        )
        kept = rows[:top_n]
        out_path = out_dir / f"{symbol}_{tf}.json"
        meta = {"rr_ratios": rr_ratios, "min_resolved": min_resolved,
                "n_tested": n_tested, "n_conjunctions_with_enough_samples": len(rows),
                "technicals_only": technicals_only, "exclude_news_window": news_filtering_active}
        if news_filtering_active:
            meta["news_note"] = NEWS_NOTE
        atomic_write_text(out_path, json.dumps({"_meta": meta, "setups": kept}, indent=2, default=str))

        print(f"  -> {n_tested} conjunctions scored, {len(rows)} had >= {min_resolved} resolved "
              f"trades at some tested R:R, kept top {len(kept)} -> {out_path}")
        for r in kept[:10]:
            best_rr = r["best_rr_ratio"]
            best_stats = r["by_rr_ratio"][str(best_rr)]
            print(f"     [{r['direction']:8s}] {' + '.join(r['primitives']):70s} "
                  f"best @ 1:{best_rr} -> win_rate={best_stats['win_rate']} "
                  f"n={best_stats['resolved']} expectancy_after_costs={r['best_expectancy_r_after_costs']}")
        summary[tf] = {"n_tested": n_tested, "n_with_samples": len(rows), "n_kept": len(kept)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="explored_setups")
    parser.add_argument("--timeframes", default=None,
                         help="comma-separated (e.g. 1d,4h,1h) - default: all available")
    parser.add_argument("--rr-ratios", default="1.5,2,2.5,3,4",
                         help="comma-separated reward multiples to re-simulate every found setup at")
    parser.add_argument("--min-resolved", type=int, default=MIN_RESOLVED_SAMPLES,
                         help="minimum resolved trades (at some tested R:R) before a setup is even reported")
    parser.add_argument("--top-n", type=int, default=300,
                         help="max setups written per timeframe, sorted by best after-cost expectancy")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=3,
                         help="lower than discover_patterns.py's default (4) - opening up the score "
                              "thresholds already means far more conjunctions get scored per depth; "
                              "this bounds runtime on the largest timeframes. Raise if you want depth-4.")
    parser.add_argument("--min-depth", type=int, default=2,
                         help="never report a single-primitive setup - matches this system's own "
                              "'no signal alone' rule (combo_patterns.py / discovery_search.py MIN_DEPTH)")
    parser.add_argument("--min-start-score", type=float, default=0.0,
                         help="discovery_search.py's default is 0.45 (tuned for 1:4 R:R) - 0.0 lets the "
                              "search consider primitives that don't already look good at 1:4")
    parser.add_argument("--min-improvement", type=float, default=0.0,
                         help="discovery_search.py's default is 0.01 - 0.0 means 'keep growing as long as "
                              "it doesn't get worse' instead of requiring a proven improvement at 1:4")
    parser.add_argument("--technicals-only", action="store_true",
                         help="exclude fundamental-family primitives from the search entirely, so no "
                              "conjunction is built out of a CPI/PCE/NFP/GDP/FOMC surprise condition")
    parser.add_argument("--include-news-window", dest="exclude_news_window", action="store_false",
                         help="by default, occurrences whose trade window overlapped a high-impact news "
                              "release are excluded before scoring (isolates each setup's own technical "
                              "edge from news-driven moves) - pass this to disable that and score every "
                              "occurrence regardless of nearby news")
    parser.set_defaults(exclude_news_window=True)
    parser.add_argument("--seed-families", default=None,
                         help="comma-separated primitive families (e.g. momentum,volatility) - restricts "
                              "which primitives may START a conjunction (depth-1 seed) so a large timeframe "
                              "can be searched in several smaller, wall-clock-bounded batches; a seed can "
                              "still GROW into any family at depth 2+, unaffected by this. Run once per "
                              "family group with the same --out-dir, then merge_explored_setups.py to "
                              "combine the batches back into one file per timeframe. Default: no "
                              "restriction, matches every prior behavior.")
    parser.add_argument("--checkpoint-dir", default=None,
                         help="if set, caches every scored conjunction's result to "
                              "<dir>/<symbol>_<tf>.checkpoint.json as the search runs (flushed every "
                              "--checkpoint-every trials), and reuses that cache on the next run with the "
                              "SAME --checkpoint-dir instead of re-simulating trades already scored - "
                              "makes a search resumable across process restarts with an IDENTICAL final "
                              "result to an uninterrupted run (verified: kill -9 mid-run + rerun with the "
                              "same --checkpoint-dir reproduces the uninterrupted output exactly). Default: "
                              "no checkpointing, matches every prior behavior.")
    parser.add_argument("--checkpoint-every", type=int, default=200,
                         help="flush the checkpoint cache to disk every N newly-scored conjunctions")
    args = parser.parse_args()

    rr_ratios = [float(x) for x in args.rr_ratios.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")] if args.timeframes else None
    data_dir = Path(args.data_dir)
    seed_families = [f.strip() for f in args.seed_families.split(",")] if args.seed_families else None
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    with track("explore_setups", path=data_dir / "heartbeats.json"):
        rebuild_all(args.symbol, data_dir, Path(args.out_dir), timeframes, rr_ratios,
                    args.min_resolved, args.top_n, args.beam_width, args.max_depth,
                    args.min_depth, args.min_start_score, args.min_improvement,
                    technicals_only=args.technicals_only, exclude_news_window=args.exclude_news_window,
                    checkpoint_dir=checkpoint_dir, checkpoint_every=args.checkpoint_every,
                    seed_families=seed_families)


if __name__ == "__main__":
    main()
