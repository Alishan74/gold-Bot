"""
Causal-factor autopsy for ANY mined pattern: "of the times this pattern
fired, what actually separated the occurrences that WON from the ones
that LOST" - a statistically honest answer, not an unfalsifiable
post-hoc story. Originally built as scripts/event_autopsy.py against a
hand-picked demo list of 6 patterns; pulled into src/ and generalized
here so build_pattern_library.py (and anything else) can ask this
question of EVERY pattern this system knows about - candlestick,
indicator, session, fundamental, support/resistance, smc, and every
cross-family combo - not a curated subset, and so the mining pipeline
and the causal-analysis pipeline can never silently test a different
pattern than the one that's actually in pattern_library/*.json.

Why this is NOT "explain why price moved" (which no system can honestly
answer): it never asks the causal question at all. It asks a narrower,
answerable one - for a CLEARLY DEFINED, ALREADY-DETECTED pattern (exactly
the same detector build_pattern_library.py used to decide qualifies:
true/false), does splitting that pattern's own historical occurrences by
their KNOWN, ALREADY-RESOLVED outcome (win vs loss, via the SAME
risk_reward.simulate_trades() every other trade in this system is graded
by) reveal any of ml_system/features.py's 80+ engineered numbers reading
MEASURABLY DIFFERENTLY between the two buckets. That's an association, a
real and falsifiable one, not a narrative - "breakouts that held had
1.8x the volume ratio of breakouts that failed" is a testable claim
about the historical record, checkable against the same reserved data
this whole system already discloses honestly.

Statistical discipline (same spirit as discovery_validation.py, adapted
to a different question - "which FEATURES differ" instead of "which
PATTERN CONJUNCTIONS qualify"):
  - Mann-Whitney U test per feature (win-bucket values vs loss-bucket
    values) - nonparametric, doesn't assume the feature is normally
    distributed (most of these aren't).
  - Benjamini-Hochberg FDR correction (discovery_validation.bh_correct -
    the ONE implementation of this in the codebase) across every feature
    actually tested for a given pattern (~80+ features = ~80+
    simultaneous tests) - `n_features_tested` is recorded honestly in the
    output for the same audit-trail reason train.py's RR_GRID search
    records n_tiers_tested.
  - A feature surviving FDR here is still an ASSOCIATION, not proof of
    mechanism - reported as "measurably different between win/loss
    buckets," never as "the reason" a trade won or lost.

Direction-agnostic patterns (doji, session_*, atr_expansion, adx_trending,
most combos, ...) are autopsied on BOTH interpretations (as_long/as_short)
- the exact same "test it both ways, gate each independently" convention
build_pattern_library.build_library() already applies to win-rate itself,
now applied to the causal factors too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from build_pattern_library import compute_pattern_flags, pattern_direction_and_gates
from discovery_validation import bh_correct
from risk_reward import MIN_RESOLVED_SAMPLES, RR_RATIO, simulate_trades

MIN_BUCKET_SAMPLES = 15  # below this, a bucket's own median/test isn't trustworthy either way
FDR_ALPHA_DEFAULT = 0.05
TOP_FACTORS_KEPT = 10  # how many significant factors get embedded in a "why" block, most-significant first


def autopsy_occurrence(occurred: pd.Series, direction: int, candles: pd.DataFrame,
                        feature_table: pd.DataFrame, atr_series: pd.Series,
                        rr_ratio: float = RR_RATIO, fdr_alpha: float = FDR_ALPHA_DEFAULT,
                        min_resolved: int = MIN_RESOLVED_SAMPLES,
                        min_bucket: int = MIN_BUCKET_SAMPLES) -> dict | None:
    """Full autopsy for one already-detected occurrence series in one
    direction. Returns None if there aren't enough resolved win/loss
    occurrences to say anything honest - same silent-skip posture
    build_library() itself uses for a pattern with too few samples to
    gate at all."""
    trades = simulate_trades(candles, occurred, direction, atr_series=atr_series, rr_ratio=rr_ratio)
    if trades.empty:
        return None  # zero occurrences (or none with a next-open to enter at) - nothing to autopsy
    resolved = trades[trades["outcome"] != "unresolved"]
    if len(resolved) < min_resolved:
        return None

    win_idx = resolved.loc[resolved["outcome"] == "win", "signal_index"].to_numpy()
    loss_idx = resolved.loc[resolved["outcome"] == "loss", "signal_index"].to_numpy()
    if len(win_idx) < min_bucket or len(loss_idx) < min_bucket:
        return None

    win_rows = feature_table.iloc[win_idx]
    loss_rows = feature_table.iloc[loss_idx]

    factors = []
    for col in feature_table.columns:
        w = win_rows[col].dropna().to_numpy()
        lo = loss_rows[col].dropna().to_numpy()
        if len(w) < min_bucket or len(lo) < min_bucket:
            continue
        if np.all(w == w[0]) and np.all(lo == lo[0]) and w[0] == lo[0]:
            continue  # both buckets are the identical constant - nothing to test
        try:
            _, p = stats.mannwhitneyu(w, lo, alternative="two-sided")
        except ValueError:
            continue  # e.g. all-identical values on both sides after all - degenerate, skip
        win_med, loss_med = float(np.median(w)), float(np.median(lo))
        factors.append({
            "feature": col, "p_value": float(p),
            "win_median": win_med, "loss_median": loss_med,
            "higher_in": "win" if win_med > loss_med else ("loss" if loss_med > win_med else "tie"),
            "win_n": int(len(w)), "loss_n": int(len(lo)),
        })

    n_tested = len(factors)
    factors = bh_correct(factors, n_tested, fdr_alpha)
    factors.sort(key=lambda f: f["p_value"])
    significant = [f for f in factors if f.get("significant")]

    return {
        "direction": ("bullish" if direction > 0 else "bearish"),
        "rr_ratio": rr_ratio,
        "n_resolved": int(len(resolved)), "n_wins": int(len(win_idx)), "n_losses": int(len(loss_idx)),
        "win_rate": round(len(win_idx) / len(resolved), 4),
        "n_features_tested": n_tested, "fdr_alpha": fdr_alpha,
        "n_significant": len(significant),
        "significant_factors": significant[:TOP_FACTORS_KEPT],
    }


def autopsy_pattern(name: str, occurred: pd.Series, candles: pd.DataFrame,
                     feature_table: pd.DataFrame, atr_series: pd.Series,
                     rr_ratio: float = RR_RATIO, fdr_alpha: float = FDR_ALPHA_DEFAULT,
                     min_resolved: "int | None" = None) -> dict | None:
    """Autopsy one pattern by NAME, looking up its direction/combo/
    sample-size gate via build_pattern_library.pattern_direction_and_gates
    - the exact same lookup build_library() itself used when it decided
    this pattern's qualifies: true/false, so a pattern's "why" block is
    always talking about the SAME direction(s) its win-rate stats are.
    `min_resolved=None` uses that pattern's own gate (atomic vs combo
    threshold); pass an explicit value to override for exploratory runs."""
    hint, is_combo, default_min_resolved, _ = pattern_direction_and_gates(name)
    effective_min_resolved = min_resolved if min_resolved is not None else default_min_resolved

    if hint == 0:
        long_result = autopsy_occurrence(occurred, +1, candles, feature_table, atr_series,
                                          rr_ratio, fdr_alpha, effective_min_resolved)
        short_result = autopsy_occurrence(occurred, -1, candles, feature_table, atr_series,
                                           rr_ratio, fdr_alpha, effective_min_resolved)
        if long_result is None and short_result is None:
            return None
        return {"direction": "ambiguous", "as_long": long_result, "as_short": short_result}

    return autopsy_occurrence(occurred, hint, candles, feature_table, atr_series,
                               rr_ratio, fdr_alpha, effective_min_resolved)


def autopsy_patterns(names: list[str], candles: pd.DataFrame, pattern_flags: pd.DataFrame,
                      feature_table: pd.DataFrame, atr_series: pd.Series,
                      rr_ratio: float = RR_RATIO, fdr_alpha: float = FDR_ALPHA_DEFAULT,
                      min_resolved: "int | None" = None,
                      progress_cb=None) -> dict:
    """Autopsy every name in `names` (must all be columns of
    `pattern_flags` - the output of build_pattern_library.
    compute_pattern_flags(), the single source of truth for what a
    pattern IS). `feature_table`/`atr_series` are computed ONCE by the
    caller and reused across every pattern, same "compute once, reuse"
    discipline build_library() itself applies to its own ATR series.
    `progress_cb(name, result)` is called after each pattern if given -
    lets a CLI print progress without this function knowing about
    printing. Returns {name: result} - patterns with too few resolved
    occurrences to say anything are simply absent, not present with an
    empty/null entry."""
    out = {}
    for name in names:
        result = autopsy_pattern(name, pattern_flags[name], candles, feature_table, atr_series,
                                  rr_ratio, fdr_alpha, min_resolved)
        if result is not None:
            out[name] = result
        if progress_cb is not None:
            progress_cb(name, result)
    return out
