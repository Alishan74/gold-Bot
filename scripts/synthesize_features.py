"""
Genetic search for brand-new NUMERIC features - the ML challenger's own
"self pattern maker," mirroring discovery_synthesis.py's genetic
primitive search (which invents new BOOLEAN conditions for the rule-
based side) but composing new DERIVED FEATURES for the gradient-boosted
model instead. See ml_system/feature_synthesis.py's module docstring for
the grammar (two existing feature columns, combined via a binary
operator, optionally wrapped in a unary one) and why it's look-ahead-safe
by construction.

Three-layer validation, mirroring discovery_validation.py's exact
discipline (multi-era + FDR + blind confirmation), adapted from "does
this boolean conjunction's win rate clear a bar" to "does this numeric
feature's Spearman correlation with the outcome clear a bar":

  1. MULTI-ERA SCREENING (during the genetic search itself, in
     feature_synthesis... no - in THIS script's _worst_era_correlation):
     a candidate is ranked/pruned by the WORST of several disjoint
     historical eras' |Spearman correlation|, never a blended average -
     a feature that only correlates in one regime and would show nothing
     in the others scores exactly as badly as its worst era.

  2. FDR-CORRECTED ACCEPTANCE: every genetic-search trial (survivor or
     not) is a genuine multiple-comparisons "test" in exactly the sense
     discovery_synthesis.py's own module docstring insists on getting
     right - `n_tested` is the TRUE total number of distinct expressions
     evaluated this run (both directions combined), and Benjamini-
     Hochberg correction (discovery_validation.bh_correct - the SAME
     shared implementation event_autopsy.py and discovery_validation.
     fdr_accept() also use, not a fourth copy of this algorithm) is
     applied to the POOLED discovery-portion correlation p-value before
     anything is accepted.

  3. BLIND CONFIRMATION SLICE: the newest ~25% of history (discovery_
     validation.split_discovery_confirmation - the SAME boundary the
     rule-based Pattern Discovery Engine uses) is held out of the ENTIRE
     genetic search. An FDR-surviving feature is re-tested there, once,
     and must show a nominally significant correlation (p < 0.05) with
     the SAME SIGN as the discovery-portion correlation - sign
     consistency is a real, additional honesty check available here
     that the boolean-pattern side doesn't need in quite the same form
     (a win-rate test is inherently one-directional already; a
     correlation can flip sign between two noisy samples in a way that
     would still individually clear p<0.05 by chance, which sign-
     matching catches).

Only features that survive ALL THREE layers are written to
synthesized_features/<symbol>_<tf>.json for train.py/live_signal.py to
actually use (via ml_system/feature_synthesis.load_synthesized_features) -
everything else this run tried is still recorded in the output's
"all_tested" section for a full audit trail, same "never hidden, never
just the search said so" standard discovery_synthesis.py holds itself
to.

Usage:
    python scripts/synthesize_features.py --symbol XAUUSD --timeframes 1h,4h,1d
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# scipy.stats.spearmanr warns (not raises) on a constant input array -
# already handled correctly everywhere it's called here via an explicit
# np.isfinite(corr) check right after, so this is cosmetic noise on a
# genetic search that WILL hit plenty of constant-within-an-era slices
# by construction, not a signal anything is wrong.
warnings.filterwarnings("ignore", message="An input array is constant")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml_system"))

import numpy as np
import pandas as pd
from scipy import stats

from atomic_io import atomic_write_text  # noqa: E402
from discovery_validation import (  # noqa: E402
    ERA_MIN_RESOLVED_SAMPLES, N_ERAS, bh_correct, era_boundaries, split_discovery_confirmation,
)
from heartbeat import track  # noqa: E402
from risk_reward import RR_RATIO, atr  # noqa: E402

import feature_synthesis as fsyn  # noqa: E402
import features as features_module  # noqa: E402
from labeling import label_all_candles  # noqa: E402

FDR_ALPHA = 0.05
CONFIRMATION_MIN_ABS_CORR = 0.02
CONFIRMATION_MAX_P = 0.05


def _label_series(candles: pd.DataFrame, direction: int, atr_series: pd.Series, rr_ratio: float) -> pd.Series:
    """0.0/1.0/NaN win-label per candle, same construction train.py's
    _label_dataset() uses - this script has its own copy (not imported
    from train.py) because it only needs the y series, not the full
    label DataFrame, and importing train.py here would pull in its
    heavier sklearn/model-registry dependencies for no reason."""
    labels = label_all_candles(candles, direction, atr_series=atr_series, rr_ratio=rr_ratio)
    y = pd.Series(np.nan, index=candles.index)
    resolved = labels[labels["outcome"] != "unresolved"]
    win_mask = resolved["outcome"] == "win"
    y.loc[resolved["signal_index"].to_numpy()] = win_mask.to_numpy().astype(float)
    return y


def _worst_era_correlation(values: pd.Series, y: pd.Series, discovery_end: int, n_eras: int = N_ERAS) -> float:
    """Worst-era |Spearman correlation| between `values` and the binary
    label `y` - the numeric-feature analog of discovery_validation.
    score_conjunction()'s worst-era Wilson lower bound. 0.0 (not
    dropped/skipped) whenever an era has too little resolved data or the
    feature is constant within it - a hard floor, same "can't claim it
    holds everywhere without evidence everywhere" reasoning."""
    eras = era_boundaries(discovery_end, n_eras)
    worst = 1.0
    for era_start, era_end in eras:
        v, yy = values.iloc[era_start:era_end], y.iloc[era_start:era_end]
        mask = v.notna() & yy.notna()
        if int(mask.sum()) < ERA_MIN_RESOLVED_SAMPLES or v[mask].nunique() < 2 or yy[mask].nunique() < 2:
            return 0.0
        corr, _ = stats.spearmanr(v[mask], yy[mask])
        if not np.isfinite(corr):
            return 0.0
        worst = min(worst, abs(float(corr)))
    return worst


def _genetic_search(feature_table: pd.DataFrame, y: pd.Series, discovery_end: int,
                     seed: int) -> tuple[list[tuple[fsyn.SynthesizedFeature, float]], int]:
    """Returns (survivors_with_score, n_tested) - survivors already
    cleared SYNTH_MIN_SCORE (worst-era |correlation|) but are NOT yet
    FDR-corrected or confirmation-checked (the caller does both - see
    module docstring). Structurally mirrors discovery_synthesis.
    synthesize_primitives() exactly: random generation seeded
    population, score-and-keep-top-K survivors per generation, mutate
    survivors + inject fresh random blood, repeat, dedup by name keeping
    each name's best-scoring instance across all generations."""
    rng = np.random.default_rng(seed)
    feature_columns = [c for c in feature_table.columns if feature_table[c].notna().any()]
    n_tested = 0
    seen_names: set[str] = set()
    scored: list[tuple[fsyn.SynthesizedFeature, float]] = []

    population = [fsyn.random_feature(rng, feature_columns) for _ in range(fsyn.GENERATION_SIZE)]
    for _generation in range(fsyn.N_GENERATIONS):
        gen_scored: list[tuple[fsyn.SynthesizedFeature, float]] = []
        for expr in population:
            if expr.is_degenerate or expr.name in seen_names:
                continue
            seen_names.add(expr.name)
            try:
                values = expr.evaluate(feature_table)
            except Exception:
                continue
            n_tested += 1
            score = _worst_era_correlation(values, y, discovery_end)
            gen_scored.append((expr, score))
        gen_scored.sort(key=lambda t: t[1], reverse=True)
        scored.extend(gen_scored)

        survivors = gen_scored[:fsyn.SURVIVORS_PER_GEN]
        if not survivors:
            break
        mutants_per_survivor = max(1, fsyn.GENERATION_SIZE // (2 * len(survivors)))
        population = [fsyn.mutate_feature(s[0], rng, feature_columns) for s in survivors for _ in range(mutants_per_survivor)]
        population += [fsyn.random_feature(rng, feature_columns) for _ in range(fsyn.GENERATION_SIZE // 4)]

    best_by_name: dict[str, tuple[fsyn.SynthesizedFeature, float]] = {}
    for expr, score in scored:
        current = best_by_name.get(expr.name)
        if current is None or score > current[1]:
            best_by_name[expr.name] = (expr, score)

    survivors_final = sorted(best_by_name.values(), key=lambda t: t[1], reverse=True)
    survivors_final = [t for t in survivors_final if t[1] >= fsyn.SYNTH_MIN_SCORE][:fsyn.N_OUTPUT_MAX]
    return survivors_final, n_tested


def synthesize_for_timeframe(candles: pd.DataFrame, rr_ratio: float = RR_RATIO,
                              seed: int = 0) -> dict:
    """Full pipeline for one timeframe: genetic search over BOTH
    directions (a feature useful for calling bullish trades and one
    useful for bearish trades are both worth keeping - the ML model
    trains a separate classifier per direction anyway, see train.py, so
    there's no reason to force one shared direction here), pooled
    n_tested across both, FDR correction, blind confirmation slice.
    Returns the full result dict written to disk (see module docstring
    for the "accepted" vs "all_tested" shape)."""
    feature_table = features_module.compute_features(candles)
    a = atr(candles)
    discovery_end, n_candles = split_discovery_confirmation(len(candles))

    all_candidates: list[dict] = []
    total_tested = 0
    for direction, direction_label in ((1, "bullish"), (-1, "bearish")):
        y = _label_series(candles, direction, a, rr_ratio)
        # Different (but deterministic) seed per direction so the two
        # searches don't retrace identical random trajectories - offset
        # by a fixed, always-non-negative amount rather than `direction`
        # itself (which is -1 for bearish and would otherwise produce a
        # negative seed, invalid for np.random.default_rng).
        direction_seed_offset = 0 if direction > 0 else 10_000
        survivors, n_tested = _genetic_search(feature_table, y, discovery_end, seed=seed + direction_seed_offset)
        total_tested += n_tested
        for expr, worst_era_score in survivors:
            values = expr.evaluate(feature_table)
            disc_v = values.iloc[:discovery_end]
            disc_y = y.iloc[:discovery_end]
            mask = disc_v.notna() & disc_y.notna()
            if int(mask.sum()) < ERA_MIN_RESOLVED_SAMPLES * N_ERAS:
                continue
            pooled_corr, pooled_p = stats.spearmanr(disc_v[mask], disc_y[mask])
            if not np.isfinite(pooled_corr):
                continue
            all_candidates.append({
                "name": expr.name, "expression": expr.to_dict(), "found_via_direction": direction_label,
                "worst_era_score": round(worst_era_score, 5),
                "discovery_correlation": round(float(pooled_corr), 5), "p_value": round(float(pooled_p), 6),
                "discovery_n": int(mask.sum()),
            })

    # Dedup: the same expression can independently survive both
    # directions' searches - keep the stronger (higher worst-era score)
    # instance, note doesn't lose the fact it helps both if it does.
    best_by_name: dict[str, dict] = {}
    for c in all_candidates:
        current = best_by_name.get(c["name"])
        if current is None or c["worst_era_score"] > current["worst_era_score"]:
            best_by_name[c["name"]] = c
    candidates = list(best_by_name.values())

    corrected = bh_correct(candidates, total_tested, FDR_ALPHA) if candidates else []

    # Blind confirmation slice - re-test ONLY the FDR survivors, on data
    # untouched by any part of the search above (see module docstring).
    accepted = []
    for c in corrected:
        if not c.get("significant"):
            c["confirmation"] = None
            continue
        expr = fsyn.SynthesizedFeature.from_dict(c["expression"])
        values = expr.evaluate(feature_table)
        conf_v = values.iloc[discovery_end:n_candles]
        # Confirmation-slice label must ALSO be built only from
        # confirmation-portion candles' own resolution - reusing the
        # already-computed `y` combined for whichever direction found
        # this feature keeps this simple and honest (the label window
        # itself never depended on the discovery/confirmation split to
        # begin with - simulate_trades() walks forward from each row
        # using the full candle series, exactly like risk_reward.py's
        # own OOS check already does).
        y_conf = _label_series(candles, 1 if c["found_via_direction"] == "bullish" else -1,
                                a, rr_ratio).iloc[discovery_end:n_candles]
        mask = conf_v.notna() & y_conf.notna()
        if int(mask.sum()) < ERA_MIN_RESOLVED_SAMPLES:
            c["confirmation"] = {"passed": False, "reason": "too few resolved rows in confirmation slice"}
            continue
        conf_corr, conf_p = stats.spearmanr(conf_v[mask], y_conf[mask])
        same_sign = np.sign(conf_corr) == np.sign(c["discovery_correlation"]) if np.isfinite(conf_corr) else False
        passed = (
            np.isfinite(conf_corr) and abs(conf_corr) >= CONFIRMATION_MIN_ABS_CORR
            and conf_p < CONFIRMATION_MAX_P and same_sign
        )
        c["confirmation"] = {
            "passed": bool(passed), "correlation": round(float(conf_corr), 5) if np.isfinite(conf_corr) else None,
            "p_value": round(float(conf_p), 6) if np.isfinite(conf_p) else None,
            "same_sign_as_discovery": bool(same_sign), "n": int(mask.sum()),
        }
        if passed:
            accepted.append(c)

    return {
        "n_tested": total_tested, "n_candidates_before_fdr": len(candidates),
        "n_fdr_survivors": len([c for c in corrected if c.get("significant")]),
        "n_accepted": len(accepted),
        "fdr_alpha": FDR_ALPHA, "discovery_end": discovery_end, "n_candles": n_candles,
        "accepted": [{"leaf_a": c["expression"]["leaf_a"], "leaf_b": c["expression"]["leaf_b"],
                      "binary_op": c["expression"]["binary_op"], "unary_wrap": c["expression"]["unary_wrap"]}
                     for c in accepted],
        "accepted_provenance": accepted,
        "all_tested_summary": corrected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data", help="shared, read-only root for data/candles/")
    parser.add_argument("--timeframes", default=None,
                         help="comma-separated timeframe labels to run (default: every timeframe file found)")
    parser.add_argument("--rr-ratio", type=float, default=RR_RATIO,
                         help="R:R structure used to grade the search's own win/loss labels (default: system 1:4 - "
                              "the synthesized features themselves are RR-agnostic and get used by every tier "
                              "train.py's RR_GRID trains, same as every hand-engineered feature already is)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="synthesized_features")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    candles_dir = data_dir / "candles"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timeframe_files = sorted(candles_dir.glob(f"{args.symbol}_*.parquet"))
    if not timeframe_files:
        raise SystemExit(f"no candle files found in {candles_dir} - run build_history.py first")
    if args.timeframes:
        wanted = set(args.timeframes.split(","))
        timeframe_files = [p for p in timeframe_files if p.stem.replace(f"{args.symbol}_", "") in wanted]
        if not timeframe_files:
            raise SystemExit(f"none of --timeframes {args.timeframes} matched files in {candles_dir}")

    with track("synthesize_features", path=data_dir / "heartbeats.json"):
        for path in timeframe_files:
            tf = path.stem.replace(f"{args.symbol}_", "")
            candles = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
            print(f"{tf}: {len(candles)} candles - genetic search over derived features "
                  f"({fsyn.GENERATION_SIZE} population x {fsyn.N_GENERATIONS} generations x 2 directions)...",
                  flush=True)
            result = synthesize_for_timeframe(candles, rr_ratio=args.rr_ratio, seed=args.seed)
            print(f"  {tf}: {result['n_tested']} expressions tested, {result['n_candidates_before_fdr']} cleared "
                  f"multi-era screening, {result['n_fdr_survivors']} survived FDR, "
                  f"{result['n_accepted']} confirmed on the blind holdout slice")
            for name in result["accepted"]:
                print(f"    accepted: synth_{name['binary_op']}_{name['leaf_a']}_{name['leaf_b']}_{name['unary_wrap']}")
            out_path = out_dir / f"{args.symbol}_{tf}.json"
            atomic_write_text(out_path, json.dumps(result, indent=2, default=str))
            print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
