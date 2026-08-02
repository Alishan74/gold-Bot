"""
Pattern Discovery Engine - Layer 4: orchestration, provenance, and the
actual self-learning loop.

Runs, per timeframe:
  0. Layer 0's genetic primitive SYNTHESIS (discovery_synthesis.py) -
     composes brand-new candidate primitives from raw OHLCV, beyond the
     hand-designed catalog.
  1. Layer 2's beam search (discovery_search.py) over Layer 1's fixed
     primitive library (discovery_primitives.py) PLUS whatever Layer 0
     just synthesized.
  2. Every candidate scored by Layer 3's worst-era Wilson lower bound
     (discovery_validation.py), survivors put through FDR-corrected
     final acceptance - critically, `n_tested` for FDR purposes is the
     TRUE total across BOTH Layer 0's synthesis trials and Layer 2's
     search trials, not just one or the other (see discovery_synthesis.
     py's own module docstring for why skipping this would quietly
     reopen the "test enough things and something clears a fixed bar by
     luck" hole).
  3. Each FDR survivor put through ONE LAST blind check against the
     confirmation slice it has never been evaluated against before.
  4. Each survivor of ALL of that additionally scored on the NEXT-
     COARSER timeframe's own candles (cross-timeframe confirmation) -
     informational, not a hard gate (many genuine patterns are honestly
     single-timeframe-scoped - session primitives, microstructure-scale
     conjunctions on 1min bars), but surfaced for the dashboard and used
     by signal_engine.py as a soft live-weight discount, the identical
     "soft discount, never a hard block" precedent signal_engine.
     REGIME_MISMATCH_PENALTY already established for regime mismatches.

Only patterns that clear steps 0-3 get written out at all; step 4 is
diagnostic metadata on top of an already-accepted pattern.

Output, deliberately kept SEPARATE from pattern_library/ (not merged
into build_pattern_library.py's own output files) - two independent
mining scripts writing to the same file would race; this mirrors the
exact same separation ml_registry/lib_view/ already uses to keep the ML
challenger's own mined view distinct from the rule-based system's:
    discovered_patterns/<symbol>_<tf>.json

Each accepted pattern's `stats` is a FULL-history risk_reward.
summarize_trades() result - the identical shape and identical hard-gate
computation every atomic/combo pattern already gets, so a discovered
pattern is graded on EXACTLY the same terms as a hand-picked one, not a
separate, looser standard. The full discovery provenance (which
primitives, era-by-era scores, FDR p-value/threshold, the confirmation-
slice result, cross-timeframe result, and - for any synthesized
component - its full reconstructable expression) is additionally stored
under `discovery_meta` for auditability - never hidden, never just "the
model said so."

Self-learning loop: re-run this periodically (same idea as
build_pattern_library.py's own weekly-ish cadence, or triggered
alongside it) as more real history and live-resolved trades accumulate.
Every discovered pattern's name is a STABLE hash of its primitive set +
direction (see _pattern_name below) - re-discovering the same
conjunction in a later run produces the SAME name, so
signal_journal.py's per-pattern live tracking/drift-detection continues
uninterrupted across re-runs instead of fragmenting into a new bucket
every time.

Usage:
    python src/discover_patterns.py --symbol XAUUSD
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from atomic_io import atomic_write_text
from build_history import TIMEFRAMES
from discovery_primitives import evaluate_primitive
from discovery_search import search_conjunctions
from discovery_synthesis import evaluate_expression, synthesize_primitives
from discovery_validation import (
    N_ERAS, fdr_accept, split_discovery_confirmation, validate_on_confirmation_slice,
)
from heartbeat import track
from risk_reward import atr, simulate_trades, summarize_trades

# Cross-timeframe confirmation floor - deliberately the SAME magnitude as
# discovery_validation.CONFIRMATION_MIN_RESOLVED_SAMPLES (20), not the
# stricter atomic-pattern MIN_RESOLVED_SAMPLES (30): this is a diagnostic
# secondary check on an ALREADY-accepted pattern, not itself a hard gate,
# so it doesn't need the full primary-gate sample bar - but it still
# needs enough samples that "qualifies" here means something, not noise.
CROSS_TF_MIN_RESOLVED = 20


def _pattern_name(primitives: tuple[str, ...], direction: int) -> str:
    """Stable identity: same primitive set + direction always hashes to
    the same name, regardless of the order beam search happened to
    construct them in (primitives are sorted before hashing) - so a
    re-discovery run recognizes "this is the same pattern as before,"
    not a fresh one every time."""
    key = "|".join(sorted(primitives)) + f"|{direction}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return f"discovered__{digest}"


def _evaluate_any_primitive(name: str, candles: pd.DataFrame, events: "pd.DataFrame | None",
                             synthesized_expressions: "dict[str, dict] | None") -> pd.Series:
    """Single shared evaluation path for BOTH kinds of primitive a
    discovered pattern can be built from: hand-designed (looked up by
    name in discovery_primitives.PRIMITIVES_BY_NAME) and synthesized
    (reconstructed from its stored expression dict, since a synthesized
    primitive's original in-memory closure from its discovery run does
    NOT exist anymore - the pattern's own discovery_meta.
    synthesized_expressions is the only record of what it means).
    `synthesized_expressions` is None/empty for any pattern discovered
    before Layer 0 existed, or that simply used no synthesized
    components - falls through to the ordinary hand-designed lookup."""
    if synthesized_expressions and name in synthesized_expressions:
        return evaluate_expression(synthesized_expressions[name], candles, events)
    return evaluate_primitive(name, candles, events)


def _cross_timeframe_confirm(primitives: tuple[str, ...], direction: int,
                              cross_tf_candles: "pd.DataFrame | None", events: "pd.DataFrame | None",
                              synthesized_expressions: "dict[str, dict] | None") -> "dict | None":
    """Re-evaluates this EXACT primitive conjunction on a different
    timeframe's own candles (the next-coarser one - see rebuild_all()'s
    adjacency map) and grades it by the SAME summarize_trades() hard-gate
    function everything else in this system is graded by. None if no
    sibling-timeframe candles were available (e.g. this is the coarsest
    timeframe with history, or that timeframe hasn't been built yet) -
    absence of a cross-timeframe check is NOT evidence against a
    pattern, it just means nothing was checked."""
    if cross_tf_candles is None or cross_tf_candles.empty:
        return None
    occurred = pd.Series(True, index=cross_tf_candles.index)
    for name in primitives:
        occurred = occurred & _evaluate_any_primitive(name, cross_tf_candles, events, synthesized_expressions)
    if not occurred.any():
        return {"checked": True, "win_rate": None, "resolved": 0, "qualifies": False,
                "note": "this conjunction never fires at all on the sibling timeframe"}
    cross_tf_atr = atr(cross_tf_candles)
    trades = simulate_trades(cross_tf_candles, occurred, direction, atr_series=cross_tf_atr)
    stats = summarize_trades(trades, min_resolved=CROSS_TF_MIN_RESOLVED,
                              oos_min_resolved=max(5, CROSS_TF_MIN_RESOLVED // 2))
    return {"checked": True, "win_rate": stats.get("win_rate"), "resolved": stats.get("resolved", 0),
            "qualifies": bool(stats.get("qualifies", False))}


def discover_for_timeframe(candles: pd.DataFrame, events: "pd.DataFrame | None",
                            cross_tf_candles: "pd.DataFrame | None" = None,
                            cross_tf_label: "str | None" = None,
                            synthesis_seed: "int | None" = None) -> tuple[dict, dict]:
    """Runs the full Layer 0 + Layer 2 + Layer 3 pipeline for one
    timeframe's candles, plus the Layer-3-adjacent cross-timeframe check
    on any accepted survivor. Returns (patterns_by_name, run_summary).
    `cross_tf_candles`/`cross_tf_label`: the next-coarser timeframe's own
    candles and its name, if available - see rebuild_all()."""
    n = len(candles)
    atr_series = atr(candles)
    discovery_end, n_total = split_discovery_confirmation(n)

    synth_primitives, synth_expr_by_name, synth_n_tested = synthesize_primitives(
        candles, events, atr_series, discovery_end, seed=synthesis_seed,
    )
    candidates, search_n_tested, _ = search_conjunctions(
        candles, events, atr_series, discovery_end, extra_primitives=synth_primitives,
    )
    # The TRUE total candidates tested this run - Layer 0's synthesis
    # trials AND Layer 2's search trials both count toward how strict the
    # FDR bar has to be (see this module's own docstring + discovery_
    # synthesis.py's for why skipping either half would be wrong).
    n_tested = synth_n_tested + search_n_tested

    fdr_input = [
        {"wins": c.score_result["total_wins"], "n": c.score_result["total_samples"], "_conj": c}
        for c in candidates
    ]
    accepted = fdr_accept(fdr_input, n_tested=n_tested)

    patterns: dict = {}
    for a in accepted:
        conj = a["_conj"]
        confirmation = validate_on_confirmation_slice(
            candles, conj.occurred, conj.direction, atr_series, discovery_end, n_total,
        )
        if not confirmation["qualifies"]:
            continue

        # Full-history stats - the SAME summarize_trades() call, on the
        # SAME occurred series (unmasked, discovery + confirmation
        # combined), that every atomic/combo pattern is graded by. This
        # is what signal_engine.py / the dashboard / signal_journal.py
        # actually read - "discovery_meta" below is the audit trail
        # explaining WHY this pattern was trusted, not the number itself.
        full_trades = simulate_trades(candles, conj.occurred, conj.direction, atr_series=atr_series)
        stats = summarize_trades(full_trades)
        if not stats["qualifies"]:
            continue  # the confirmation-slice-only check can pass while the full-history blend still doesn't - hold discovered patterns to the identical bar as everything else, no exception

        this_pattern_synth_exprs = {
            name: synth_expr_by_name[name] for name in conj.primitives if name in synth_expr_by_name
        }
        cross_tf = _cross_timeframe_confirm(
            conj.primitives, conj.direction, cross_tf_candles, events, this_pattern_synth_exprs,
        )

        name = _pattern_name(conj.primitives, conj.direction)
        patterns[name] = {
            "direction": "bullish" if conj.direction > 0 else "bearish",
            "stats": stats,
            "discovery_meta": {
                "primitives": list(conj.primitives),
                "families": sorted(conj.families),
                "synthesized_expressions": this_pattern_synth_exprs,
                "discovery_worst_era_score": conj.worst_era_score,
                "discovery_era_scores": conj.score_result["era_scores"],
                "discovery_era_samples": conj.score_result["era_samples"],
                "n_eras": N_ERAS,
                "p_value": a["p_value"],
                "bh_threshold": a["bh_threshold"],
                "n_tested_this_run": n_tested,
                "confirmation_slice": confirmation,
                "cross_timeframe": ({**cross_tf, "timeframe": cross_tf_label} if cross_tf else None),
            },
        }

    summary = {
        "n_tested": n_tested,
        "n_tested_synthesis": synth_n_tested,
        "n_tested_search": search_n_tested,
        "n_synthesized_primitives": len(synth_primitives),
        "n_candidates_after_beam_search": len(candidates),
        "n_accepted_after_fdr": len(accepted),
        "n_final_after_confirmation": len(patterns),
    }
    return patterns, summary


def is_discovered_pattern_active(primitives: list[str], candles: pd.DataFrame,
                                  events: "pd.DataFrame | None",
                                  synthesized_expressions: "dict[str, dict] | None" = None) -> bool:
    """Live detection for a discovered pattern - re-evaluates each of ITS
    OWN component primitives against the current candle tail, using
    _evaluate_any_primitive() (the exact same function discover_for_
    timeframe's own cross-timeframe check uses) so live detection can
    never silently drift onto a different definition than the one that
    was validated - true for hand-designed primitives (discovery_
    primitives.evaluate_primitive) AND for synthesized ones
    (reconstructed from `synthesized_expressions`, this pattern's own
    discovery_meta.synthesized_expressions field - see discovery_
    synthesis.py's module docstring for why a synthesized primitive
    can't just be looked up by name in a static catalog the way a
    hand-designed one can). True only if EVERY component primitive is
    true on the LAST candle (logical AND, same confluence definition
    discovery_search.py's beam search itself used to build the
    conjunction in the first place)."""
    if not primitives:
        return False
    for name in primitives:
        series = _evaluate_any_primitive(name, candles, events, synthesized_expressions)
        if not bool(series.iloc[-1]):
            return False
    return True


def _next_coarser_timeframe(tf: str) -> "str | None":
    """TIMEFRAMES (build_history.py) is already ordered finest-to-
    coarsest by construction (1min, 5min, 15min, 1h, 4h, 1d) - the next
    dict key after `tf` is the timeframe cross-timeframe confirmation
    checks a pattern against. None for the coarsest timeframe (1d) or
    any unrecognized label - "no sibling to check," not an error."""
    order = list(TIMEFRAMES)
    if tf not in order:
        return None
    idx = order.index(tf)
    return order[idx + 1] if idx + 1 < len(order) else None


def rebuild_all(symbol: str, data_dir: Path, out_dir: Path) -> dict:
    """Mirrors build_pattern_library.rebuild_all()'s own shape/contract -
    pure function, no argparse, so this can be called identically from
    main() below and from any future self-heal trigger, exactly the
    precedent signal_journal.should_self_heal already established for
    build_pattern_library.py.

    Loads ALL timeframes' candles UP FRONT (not one-at-a-time, as a
    single-timeframe version of this function could) specifically so
    cross-timeframe confirmation has the sibling timeframe's data
    available when discovering on any given timeframe - a modest memory
    cost (candle parquet files, not the raw tick cache) for a real
    accuracy improvement."""
    candles_dir = data_dir / "candles"
    out_dir.mkdir(parents=True, exist_ok=True)

    timeframe_files = sorted(candles_dir.glob(f"{symbol}_*.parquet"))
    if not timeframe_files:
        raise SystemExit(f"no candle files found in {candles_dir} - run build_history.py first")

    events_path = data_dir / "events" / "fundamentals.parquet"
    events = pd.read_parquet(events_path) if events_path.exists() else None

    candles_by_tf: dict[str, pd.DataFrame] = {}
    for path in timeframe_files:
        tf = path.stem.replace(f"{symbol}_", "")
        candles_by_tf[tf] = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)

    summary = {}
    for tf, candles in candles_by_tf.items():
        cross_tf_label = _next_coarser_timeframe(tf)
        cross_tf_candles = candles_by_tf.get(cross_tf_label) if cross_tf_label else None
        print(f"{tf}: discovering patterns over {len(candles)} candles "
              f"({candles['timestamp'].min()} -> {candles['timestamp'].max()})"
              + (f" - cross-timeframe check against {cross_tf_label}" if cross_tf_candles is not None
                 else " - no coarser sibling timeframe available for cross-timeframe check"))

        patterns, run_summary = discover_for_timeframe(candles, events, cross_tf_candles, cross_tf_label)
        out_path = out_dir / f"{symbol}_{tf}.json"
        atomic_write_text(out_path, json.dumps(patterns, indent=2, default=str))

        print(f"  -> tested {run_summary['n_tested']} candidate conjunctions "
              f"({run_summary['n_tested_synthesis']} from primitive synthesis, "
              f"{run_summary['n_tested_search']} from beam search over "
              f"{run_summary['n_synthesized_primitives']} synthesized + the fixed catalog), "
              f"{run_summary['n_candidates_after_beam_search']} survived beam search, "
              f"{run_summary['n_accepted_after_fdr']} survived FDR correction, "
              f"{run_summary['n_final_after_confirmation']} survived the blind confirmation slice")
        print(f"     discovered: {list(patterns) or '(none)'}")
        print(f"     written to {out_path}")
        summary[tf] = {**run_summary, "discovered": list(patterns)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="discovered_patterns")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with track("discover_patterns", path=data_dir / "heartbeats.json"):
        rebuild_all(args.symbol, data_dir, Path(args.out_dir))


if __name__ == "__main__":
    main()
