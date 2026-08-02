"""
Rigorous, single-candidate validation: takes ONE specific primitive
conjunction you already have (e.g. surfaced by scripts/explore_setups.py)
and grades it with the SAME acceptance logic discover_patterns.py applies
to every self-learned pattern - not a fresh search, a direct verdict on a
candidate already in hand.

Why explore_setups.py's own "out of sample" number isn't enough on its
own: that's risk_reward.summarize_trades()'s built-in split - the most
recent 30% of THIS PATTERN'S OWN OCCURRENCES. Real and useful, but not
the same guarantee as discover_patterns.py's blind confirmation slice
(the newest 25% of CALENDAR TIME, held out of the search process
ENTIRELY - explore_setups.py's beam search saw that data when scoring and
growing every conjunction; this script's confirmation check does not).

Three checks, all reusing the exact functions discover_patterns.py itself
calls (never a second, possibly-inconsistent implementation):
  1. FDR-corrected acceptance (discovery_validation.fdr_accept) - is this
     candidate's pooled win/loss record significant against the null "no
     better than this system's own 60% bar," once honestly corrected for
     how many other candidates were compared to find it (--n-tested)?
  2. Blind confirmation slice (discovery_validation.
     validate_on_confirmation_slice) - does it independently qualify on
     the newest ~25% of history, never touched by the search that found
     it?
  3. Full-history hard gate (risk_reward.summarize_trades, including its
     own internal out-of-sample sub-split) - the identical check every
     atomic/combo/discovered pattern in this system has to clear.
ALL THREE must pass - matching discover_patterns.discover_for_timeframe's
own accept logic exactly, not a looser standard for a candidate that
started outside the strict pipeline.

`--n-tested` is REQUIRED, not defaulted - guessing it would make the FDR
correction meaninglessly lenient (see discovery_validation.py's own
n_tested warnings). Use the honest count of what was actually searched to
find this candidate: explore_setups.py's own console output/_meta.n_tested
gives the number of CONJUNCTIONS scored for that timeframe; multiply by
how many --rr-ratios that run tried (5 by default) since "best of several
r:r structures" is itself a comparison that needs to count - e.g. a
candidate pulled from a default explore_setups.py run that scored 838
conjunctions should be validated with --n-tested 4190 (838 x 5), not 838.

Usage:
    python scripts/validate_candidate.py --symbol XAUUSD --timeframe 4h \\
        --primitives near_range_high_10_0.9,volatility_contraction_0.7,session_sydney_active \\
        --direction bullish --rr-ratio 1.5 --n-tested 4190
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from atomic_io import atomic_write_text  # noqa: E402
from discovery_primitives import evaluate_primitive  # noqa: E402
from discovery_validation import (  # noqa: E402
    FDR_ALPHA, binomial_p_value, fdr_accept, score_conjunction, split_discovery_confirmation,
    validate_on_confirmation_slice,
)
from risk_reward import atr, simulate_trades, summarize_trades  # noqa: E402


def validate(candles: pd.DataFrame, events: "pd.DataFrame | None", primitives: list[str], direction: int,
             rr_ratio: float, n_tested: int, fdr_alpha: float = FDR_ALPHA) -> dict:
    occurred = pd.Series(True, index=candles.index)
    for name in primitives:
        occurred = occurred & evaluate_primitive(name, candles, events).fillna(False)
    n_raw_occurrences = int(occurred.sum())

    atr_series = atr(candles)
    n = len(candles)
    discovery_end, n_total = split_discovery_confirmation(n)

    era_result = score_conjunction(candles, occurred, direction, atr_series, discovery_end, rr_ratio=rr_ratio)

    # Rank-1 case of the real fdr_accept(): this candidate is the ONLY
    # thing being reported from the n_tested search, so it has to clear
    # the strictest (smallest) BH threshold in that batch - exactly what
    # discover_patterns.py's own candidates are held to. Computed directly
    # via binomial_p_value (not just read back from fdr_accept's return)
    # because fdr_accept() only returns ACCEPTED candidates - on a FAIL it
    # returns [], which would silently throw away the real p-value right
    # when it's most useful to see (how far did this miss the bar by).
    bh_threshold = (1 / n_tested) * fdr_alpha
    p_value = binomial_p_value(era_result["total_wins"], era_result["total_samples"])
    fdr_result = fdr_accept(
        [{"wins": era_result["total_wins"], "n": era_result["total_samples"]}],
        n_tested=n_tested, alpha=fdr_alpha,
    )
    fdr_pass = len(fdr_result) == 1

    confirmation = validate_on_confirmation_slice(
        candles, occurred, direction, atr_series, discovery_end, n_total, rr_ratio=rr_ratio,
    )
    confirmation_pass = bool(confirmation.get("qualifies"))

    full_trades = simulate_trades(candles, occurred, direction, atr_series=atr_series, rr_ratio=rr_ratio)
    full_stats = summarize_trades(full_trades, rr_ratio=rr_ratio)
    full_history_pass = bool(full_stats.get("qualifies"))

    overall = fdr_pass and confirmation_pass and full_history_pass

    return {
        "primitives": primitives, "direction": "bullish" if direction > 0 else "bearish",
        "rr_ratio": rr_ratio, "n_tested": n_tested, "n_raw_occurrences": n_raw_occurrences,
        "worst_era": {"worst_era_score": era_result["worst_era_score"],
                      "era_scores": era_result["era_scores"], "era_samples": era_result["era_samples"],
                      "total_wins": era_result["total_wins"], "total_samples": era_result["total_samples"]},
        "fdr": {"p_value": p_value, "bh_threshold": bh_threshold, "passes": fdr_pass},
        "confirmation_slice": {**confirmation, "passes": confirmation_pass},
        "full_history": {**full_stats, "passes": full_history_pass},
        "OVERALL_QUALIFIES": overall,
    }


def _append_to_log(log_path: Path, result: dict, symbol: str, timeframe: str) -> None:
    """Appends a compact summary of this run to a running JSON array at
    `log_path` - re-running this same command periodically (e.g. whenever
    live_update.py has pulled fresh candles) is this project's live-
    tracking mechanism for a candidate that failed the strict gate but
    hasn't been debunked: the confirmation slice is always the newest
    ~25% of WHATEVER history exists when this script runs, so as real,
    new candles accumulate, re-running it re-checks the honest question
    "does the statistical case for this candidate look any different with
    more real data" - without ever touching signal_engine.py/the journal,
    so a not-yet-qualifying candidate can never leak into a live BUY/SELL
    signal just because it's being watched. Read-modify-write (not a true
    atomic append), same acceptable-for-a-low-frequency-manual-tool
    tradeoff atomic_write_text's own docstring accepts elsewhere - this
    runs at most a few times a week, never concurrently with itself."""
    existing = json.loads(log_path.read_text()) if log_path.exists() else []
    fdr, cs, fh = result["fdr"], result["confirmation_slice"], result["full_history"]
    existing.append({
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "timeframe": timeframe,
        "primitives": result["primitives"], "direction": result["direction"], "rr_ratio": result["rr_ratio"],
        "n_tested": result["n_tested"],
        "fdr_p_value": fdr["p_value"], "fdr_bh_threshold": fdr["bh_threshold"], "fdr_passes": fdr["passes"],
        "confirmation_win_rate": cs.get("win_rate"), "confirmation_resolved": cs.get("resolved"),
        "confirmation_passes": cs["passes"],
        "full_history_win_rate": fh.get("win_rate"), "full_history_resolved": fh.get("resolved"),
        "full_history_passes": fh["passes"],
        "overall_qualifies": result["OVERALL_QUALIFIES"],
    })
    atomic_write_text(log_path, json.dumps(existing, indent=2, default=str))
    print(f"\n({len(existing)} check(s) now logged in {log_path} - re-run this same command "
          f"periodically as live_update.py pulls new candles to watch the trend)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--primitives", required=True, help="comma-separated primitive names")
    parser.add_argument("--direction", required=True, choices=["bullish", "bearish"])
    parser.add_argument("--rr-ratio", type=float, required=True)
    parser.add_argument("--n-tested", type=int, required=True,
                         help="see module docstring - the honest count of everything searched to find this candidate")
    parser.add_argument("--fdr-alpha", type=float, default=FDR_ALPHA)
    parser.add_argument("--log", default=None,
                         help="path to a running JSON log this run's summary gets appended to - re-run the "
                              "same command (same path) over time to build a trend as real data accumulates, "
                              "e.g. --log watchlist\\4h_near_range_session.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    candles_path = data_dir / "candles" / f"{args.symbol}_{args.timeframe}.parquet"
    if not candles_path.exists():
        raise SystemExit(f"no candle file at {candles_path}")
    candles = pd.read_parquet(candles_path).sort_values("timestamp").reset_index(drop=True)

    events_path = data_dir / "events" / "fundamentals.parquet"
    events = pd.read_parquet(events_path) if events_path.exists() else None

    primitives = [p.strip() for p in args.primitives.split(",")]
    direction = 1 if args.direction == "bullish" else -1

    print(f"Validating {' + '.join(primitives)} ({args.direction}) on {args.timeframe} "
          f"at 1:{args.rr_ratio}, n_tested={args.n_tested}...")
    result = validate(candles, events, primitives, direction, args.rr_ratio, args.n_tested, args.fdr_alpha)

    print(f"\nraw occurrences (before entry/resolution filtering): {result['n_raw_occurrences']}")
    we = result["worst_era"]
    print(f"\nworst-era diagnostic (4 disjoint chronological chunks of the discovery portion):")
    print(f"  worst_era_score={we['worst_era_score']:.4f}  era_scores={[round(s,4) for s in we['era_scores']]}"
          f"  era_samples={we['era_samples']}")
    print(f"  pooled: {we['total_wins']}/{we['total_samples']} wins")

    fdr = result["fdr"]
    print(f"\n1) FDR-corrected acceptance (null: no better than this system's 60% bar, "
          f"n_tested={args.n_tested}):")
    print(f"   p_value={fdr['p_value']}  bh_threshold={fdr['bh_threshold']:.6f}  "
          f"-> {'PASS' if fdr['passes'] else 'FAIL'}")

    cs = result["confirmation_slice"]
    print(f"\n2) Blind confirmation slice (newest ~25% of history, NEVER seen by explore_setups.py's search):")
    print(f"   win_rate={cs.get('win_rate')}  resolved={cs.get('resolved')}  "
          f"oos_win_rate={(cs.get('out_of_sample') or {}).get('win_rate')} "
          f"oos_resolved={(cs.get('out_of_sample') or {}).get('resolved')}"
          f"  -> {'PASS' if cs['passes'] else 'FAIL'}")

    fh = result["full_history"]
    print(f"\n3) Full-history hard gate (same summarize_trades() every pattern in this system is graded by):")
    print(f"   win_rate={fh.get('win_rate')}  resolved={fh.get('resolved')}  "
          f"oos_win_rate={(fh.get('out_of_sample') or {}).get('win_rate')} "
          f"oos_resolved={(fh.get('out_of_sample') or {}).get('resolved')}"
          f"  -> {'PASS' if fh['passes'] else 'FAIL'}")

    print(f"\n{'='*70}")
    print(f"OVERALL: {'QUALIFIES' if result['OVERALL_QUALIFIES'] else 'DOES NOT QUALIFY'}")
    print(f"{'='*70}")

    if args.log:
        _append_to_log(Path(args.log), result, args.symbol, args.timeframe)

    out_path = Path(f"validated_{args.symbol}_{args.timeframe}_{'_'.join(primitives)}.json")
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nfull result written to {out_path}")


if __name__ == "__main__":
    main()
