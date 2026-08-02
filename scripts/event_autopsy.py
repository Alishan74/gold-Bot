"""
CLI over src/causal_autopsy.py: "of the times a pattern fired, what
actually separated the occurrences that WON from the ones that DIDN'T" -
see that module's docstring for the full statistical methodology
(Mann-Whitney U per ml_system/features.py feature, Benjamini-Hochberg FDR
correction across every feature tested).

This used to hard-code its own 6-pattern demo catalog and its own
detector lookup (patterns.ALL_PATTERNS / support_resistance.
SUPPORT_RESISTANCE_PATTERNS only - missing session/fundamental/smc/combo
entirely). Both now live in src/causal_autopsy.py and src/build_pattern_
library.py's compute_pattern_flags(), so this script is a thin CLI over
the SAME pattern vocabulary and SAME detector definitions build_pattern_
library.py itself mines - no second, potentially-drifting catalog.

Three ways to pick which patterns get autopsied:
  --patterns all            every pattern compute_pattern_flags() can
                             produce (candlestick/indicator, session,
                             fundamental, support/resistance, smc, and
                             every cross-family combo) - the default.
                             Fast: ~4s for ~1900 patterns over 5,000
                             candles in testing - most patterns don't have
                             enough resolved occurrences to test at all,
                             so the real cost is dominated by the ones
                             that do, not the full column count.
  --patterns library        only patterns already present in an existing
                             pattern_library/<symbol>_<tf>.json - use this
                             with --merge-into-library to add "why" to
                             exactly what's already been mined, without
                             re-testing patterns that never made the cut.
  --patterns a,b,c           an explicit comma-separated list (the
                             original behavior of this script, before it
                             covered every pattern in the system).

--merge-into-library writes each pattern's causal result directly into
pattern_library/<symbol>_<tf>.json under a new "why" key on that
pattern's own entry (matching causal_autopsy.autopsy_pattern()'s output
shape 1:1 - "as_long"/"as_short" for ambiguous patterns, flat for
directional ones) - preserving every other key already there. This is
what makes the causal analysis actually visible to "the rest of the
system": the dashboard, signal_engine.py, and anything else that already
reads pattern_library/*.json sees it right next to win_rate, without a
second, disconnected report file nobody has to remember to open. Without
this flag, results are written to their own --out-dir instead (the
original standalone-report behavior), for exploratory runs that
shouldn't touch the live-serving library.

Usage:
    python scripts/event_autopsy.py --symbol XAUUSD --patterns all
    python scripts/event_autopsy.py --symbol XAUUSD --patterns library --merge-into-library
    python scripts/event_autopsy.py --symbol XAUUSD --events bullish_engulfing,bearish_engulfing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml_system"))

import pandas as pd

from atomic_io import atomic_write_text  # noqa: E402
from build_pattern_library import compute_pattern_flags  # noqa: E402
from causal_autopsy import FDR_ALPHA_DEFAULT, autopsy_patterns  # noqa: E402
from heartbeat import track  # noqa: E402
from risk_reward import RR_RATIO, atr  # noqa: E402

import features as features_module  # noqa: E402


def _resolve_pattern_names(spec: str, pattern_flags: pd.DataFrame, library_path: Path) -> list[str]:
    if spec == "all":
        return list(pattern_flags.columns)
    if spec == "library":
        if not library_path.exists():
            raise SystemExit(f"--patterns library needs an existing {library_path} - "
                              f"run build_pattern_library.py first, or use --patterns all")
        library = json.loads(library_path.read_text())
        names = [name for name in library if name != "_meta"]
        missing = [n for n in names if n not in pattern_flags.columns]
        if missing:
            print(f"  note: {len(missing)} name(s) in {library_path} aren't in this run's detected "
                  f"patterns (e.g. discovered__ entries from the Pattern Discovery Engine, which aren't "
                  f"re-derivable from compute_pattern_flags() alone) - skipped: {missing[:5]}"
                  f"{'...' if len(missing) > 5 else ''}")
        return [n for n in names if n in pattern_flags.columns]
    return [e.strip() for e in spec.split(",") if e.strip()]


def _merge_into_library(library_path: Path, results: dict) -> int:
    if not library_path.exists():
        raise SystemExit(f"--merge-into-library needs an existing {library_path} - "
                          f"run build_pattern_library.py first")
    library = json.loads(library_path.read_text())
    n_merged = 0
    for name, why in results.items():
        if name not in library:
            continue  # not (yet) a library entry - nothing to attach "why" to
        library[name]["why"] = why
        n_merged += 1
    atomic_write_text(library_path, json.dumps(library, indent=2, default=str))
    return n_merged


def _print_progress(name: str, result: "dict | None") -> None:
    if result is None:
        return
    if result.get("direction") == "ambiguous":
        for side, label in (("as_long", "long"), ("as_short", "short")):
            r = result.get(side)
            if r and r["n_significant"]:
                top = r["significant_factors"][0]
                print(f"    {name} ({label}): {r['n_wins']}W/{r['n_losses']}L "
                      f"(win_rate={r['win_rate']:.1%}), {r['n_significant']}/{r['n_features_tested']} "
                      f"features survive FDR - top: {top['feature']} "
                      f"(p={top['p_value']:.2e}, higher in {top['higher_in']})")
    elif result["n_significant"]:
        top = result["significant_factors"][0]
        print(f"    {name}: {result['n_wins']}W/{result['n_losses']}L "
              f"(win_rate={result['win_rate']:.1%}), {result['n_significant']}/"
              f"{result['n_features_tested']} features survive FDR - top: {top['feature']} "
              f"(p={top['p_value']:.2e}, higher in {top['higher_in']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data", help="shared, read-only root for data/candles/")
    parser.add_argument("--timeframes", default=None,
                         help="comma-separated timeframe labels to run (default: every timeframe file found)")
    parser.add_argument("--patterns", default="all",
                         help="'all' (default, every detected pattern), 'library' (only patterns already in "
                              "pattern_library/<symbol>_<tf>.json), or a comma-separated explicit list")
    parser.add_argument("--events", default=None, dest="patterns_legacy",
                         help="deprecated alias for --patterns, kept for old scripts/cron jobs")
    parser.add_argument("--rr-ratio", type=float, default=RR_RATIO,
                         help="R:R structure used to grade each occurrence win/loss (default: system 1:4)")
    parser.add_argument("--fdr-alpha", type=float, default=FDR_ALPHA_DEFAULT)
    parser.add_argument("--min-resolved", type=int, default=None,
                         help="override each pattern's own atomic/combo sample-size gate (default: use it)")
    parser.add_argument("--out-dir", default="event_autopsy", help="ignored when --merge-into-library is set")
    parser.add_argument("--library-dir", default="pattern_library")
    parser.add_argument("--merge-into-library", action="store_true",
                         help="write each pattern's causal result into pattern_library/<symbol>_<tf>.json "
                              "under a 'why' key, instead of a separate report file")
    args = parser.parse_args()
    patterns_spec = args.patterns_legacy if args.patterns_legacy is not None else args.patterns

    data_dir = Path(args.data_dir)
    candles_dir = data_dir / "candles"
    out_dir = Path(args.out_dir)
    library_dir = Path(args.library_dir)
    if not args.merge_into_library:
        out_dir.mkdir(parents=True, exist_ok=True)

    timeframe_files = sorted(candles_dir.glob(f"{args.symbol}_*.parquet"))
    if not timeframe_files:
        raise SystemExit(f"no candle files found in {candles_dir} - run build_history.py first")
    if args.timeframes:
        wanted = set(args.timeframes.split(","))
        timeframe_files = [p for p in timeframe_files if p.stem.replace(f"{args.symbol}_", "") in wanted]
        if not timeframe_files:
            raise SystemExit(f"none of --timeframes {args.timeframes} matched files in {candles_dir}")

    with track("event_autopsy", path=data_dir / "heartbeats.json"):
        for path in timeframe_files:
            tf = path.stem.replace(f"{args.symbol}_", "")
            candles = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
            library_path = library_dir / f"{args.symbol}_{tf}.json"

            print(f"{tf}: {len(candles)} candles - computing pattern flags + feature table once, "
                  f"reused across every pattern...")
            pattern_flags = compute_pattern_flags(candles)
            feature_table = features_module.compute_features(candles)
            a = atr(candles)

            names = _resolve_pattern_names(patterns_spec, pattern_flags, library_path)
            print(f"  autopsying {len(names)} pattern(s) at {args.rr_ratio:g}R...", flush=True)

            results = autopsy_patterns(names, candles, pattern_flags, feature_table, a,
                                        rr_ratio=args.rr_ratio, fdr_alpha=args.fdr_alpha,
                                        min_resolved=args.min_resolved, progress_cb=_print_progress)
            print(f"  {len(results)}/{len(names)} pattern(s) had enough resolved win/loss occurrences to autopsy")

            if args.merge_into_library:
                n_merged = _merge_into_library(library_path, results)
                print(f"  merged 'why' into {n_merged} entr{'y' if n_merged == 1 else 'ies'} of {library_path}")
            else:
                out_path = out_dir / f"{args.symbol}_{tf}.json"
                atomic_write_text(out_path, json.dumps({
                    "symbol": args.symbol, "timeframe": tf, "rr_ratio": args.rr_ratio,
                    "fdr_alpha": args.fdr_alpha, "patterns": results,
                }, indent=2, default=str))
                print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
