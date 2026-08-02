"""
Pattern Discovery Engine - Layer 3: validation.

Three defenses against the "search enough things and something passes
by luck" problem discovery_search.py's construction algorithm would
otherwise be exposed to - STACKED, not alternatives to each other:

1. MULTI-ERA SCORING: every candidate conjunction is scored using the
   WORST of several disjoint historical eras' Wilson lower bounds, not
   one blended full-sample number. A pattern that only worked in one
   regime and would have failed in the others scores exactly as badly
   as its worst era - it can never hide behind an average.

2. FDR-CORRECTED FINAL ACCEPTANCE: discovery_search.py evaluates many
   candidate conjunctions over a search run - the more it tests, the
   more of them will clear ANY fixed bar by pure chance alone (classic
   multiple-comparisons - see combo_patterns.py's own docstring for the
   same risk at smaller scale). Benjamini-Hochberg false-discovery-rate
   correction makes the bar a candidate has to clear MEASURABLY
   STRICTER as the total number tested grows, instead of staying fixed
   regardless of how hard the search looked.

3. BLIND CONFIRMATION SLICE: the newest ~25% of history is held out of
   discovery ENTIRELY - not touched by era-scoring, not touched by FDR
   selection, nothing in the search process ever sees it. A surviving
   candidate has to ALSO independently clear the hard gate there, on
   data nothing about its own construction could have adapted to, even
   indirectly through the search algorithm's behavior across many
   candidates - a subtler leak than any single candidate's own
   train/test split closes on its own.

All three reuse risk_reward.py's existing trade-simulation machinery
(simulate_trades/summarize_trades/wilson_lower_bound) rather than a
second implementation - same reasoning ml_system/labeling.py already
documents for why that sharing matters (a fair, apples-to-apples grade
against every other pattern in this system, and no second place for the
same trade-simulation bug to be introduced twice).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from risk_reward import RR_RATIO, MIN_WIN_RATE, simulate_trades, summarize_trades, wilson_lower_bound

DISCOVERY_FRACTION = 0.75      # newest 25% of history is NEVER touched during search
N_ERAS = 4                     # disjoint chronological chunks within the discovery portion
# Per-era floor BELOW which a candidate scores 0.0 for that era outright
# (dragging its worst-era score to 0.0 too) - deliberately lower than
# MIN_RESOLVED_SAMPLES(30), since each era is only ~1/4 of the already-
# reduced discovery portion; this is a per-era screening floor, not the
# bar a pattern is finally accepted on (that's FDR_ALPHA + the
# confirmation slice below).
ERA_MIN_RESOLVED_SAMPLES = 20
FDR_ALPHA = 0.05               # target false discovery rate for final acceptance
CONFIRMATION_MIN_RESOLVED_SAMPLES = 20


def split_discovery_confirmation(n_candles: int, discovery_fraction: float = DISCOVERY_FRACTION) -> tuple[int, int]:
    """(discovery_end, n_candles) index boundaries - candles[0:discovery_end]
    is the DISCOVERY portion (everything discovery_search.py is allowed
    to look at), candles[discovery_end:n_candles] is the CONFIRMATION
    portion (never touched until validate_on_confirmation_slice() below,
    once, per final candidate)."""
    discovery_end = int(n_candles * discovery_fraction)
    return discovery_end, n_candles


def era_boundaries(discovery_end: int, n_eras: int = N_ERAS) -> list[tuple[int, int]]:
    """N_ERAS disjoint, roughly-equal, chronologically-ordered [start, end)
    index ranges within [0, discovery_end)."""
    edges = np.linspace(0, discovery_end, n_eras + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_eras)]


def _score_era(candles: pd.DataFrame, occurred: pd.Series, direction: int,
               atr_series: pd.Series, era_start: int, era_end: int,
               rr_ratio: float = RR_RATIO) -> tuple[float, int, int]:
    """Wilson lower-bound win rate + resolved-sample count for occurrences
    falling WITHIN [era_start, era_end), resolved by walking forward
    through the FULL candle series - a trade opened near the end of an
    era still needs real subsequent candles to know whether it actually
    won or lost, so only WHICH candles count as signal occurrences is
    restricted to the era; the underlying candle data passed to
    simulate_trades is never truncated at the era boundary. Same
    "restrict which trades count, don't truncate the data" principle
    risk_reward.summarize_trades' own out-of-sample split already uses.
    Returns (wilson_lower_bound_or_0, wins, resolved_n).

    `rr_ratio` defaults to the system-wide fixed RR_RATIO (1:4) - every
    real discover_patterns.py call leaves it unset. scripts/
    validate_candidate.py passes a non-default value so a candidate found
    by explore_setups.py at a DIFFERENT r:r (not this system's native 1:4)
    can be graded by this identical worst-era/FDR/confirmation machinery
    at the r:r it actually performs at, instead of being silently
    re-scored at 1:4 regardless of what it was found at."""
    era_mask = pd.Series(False, index=candles.index)
    era_mask.iloc[era_start:era_end] = True
    masked_occurred = occurred & era_mask
    if not masked_occurred.any():
        return 0.0, 0, 0
    trades = simulate_trades(candles, masked_occurred, direction, atr_series=atr_series, rr_ratio=rr_ratio)
    resolved = trades[trades["outcome"] != "unresolved"]
    n = len(resolved)
    if n < ERA_MIN_RESOLVED_SAMPLES:
        return 0.0, 0, n
    wins = int((resolved["outcome"] == "win").sum())
    return wilson_lower_bound(wins, n), wins, n


def score_conjunction(candles: pd.DataFrame, occurred: pd.Series, direction: int,
                       atr_series: pd.Series, discovery_end: int,
                       n_eras: int = N_ERAS, rr_ratio: float = RR_RATIO) -> dict:
    """The search's core scoring function: worst-era Wilson lower bound
    across N_ERAS disjoint chronological chunks of the DISCOVERY portion
    ONLY (never the confirmation slice - see module docstring). A
    conjunction that hasn't fired ERA_MIN_RESOLVED_SAMPLES times in
    EVERY era scores 0.0 for that era, which drags the overall (worst-
    era) score to 0.0 too - a pattern has to have enough real
    occurrences in every regime tested, not just on average across them.

    `rr_ratio`: see _score_era's docstring - defaults to the system-wide
    1:4, unused by discovery_search.py's real search calls."""
    eras = era_boundaries(discovery_end, n_eras)
    era_scores, era_wins, era_samples = [], [], []
    for era_start, era_end in eras:
        score, wins, n = _score_era(candles, occurred, direction, atr_series, era_start, era_end, rr_ratio=rr_ratio)
        era_scores.append(score)
        era_wins.append(wins)
        era_samples.append(n)
    return {
        "worst_era_score": min(era_scores),
        "era_scores": era_scores,
        "era_wins": era_wins,
        "era_samples": era_samples,
        "total_wins": sum(era_wins),
        "total_samples": sum(era_samples),
    }


def binomial_p_value(wins: int, n: int, null_win_rate: float = MIN_WIN_RATE) -> float:
    """One-sided p-value: if the TRUE win rate were exactly `null_win_rate`
    (the hard gate itself, not a generic 50/50 - the honest null here is
    "no better than the bar we're about to hold this candidate to"),
    what's the probability of observing `wins`-or-more successes out of
    `n` purely by chance? Smaller = less likely to be a fluke. Uses the
    exact binomial survival function, not a normal approximation - small-
    n candidates are exactly where a normal approximation is least
    trustworthy, and this only ever runs once per FINAL candidate (not
    per search step), so exactness is cheap here."""
    if n == 0:
        return 1.0
    return float(stats.binom.sf(wins - 1, n, null_win_rate))


def bh_correct(items: list[dict], n_tested: int, alpha: float, p_value_key: str = "p_value") -> list[dict]:
    """Standard Benjamini-Hochberg false-discovery-rate correction over
    any list of dicts that already carry a p-value - the ONE
    implementation of this algorithm in the codebase, shared by
    fdr_accept() below (pattern-conjunction acceptance, binomial win-
    rate p-values) and scripts/event_autopsy.py (per-feature win/loss
    distribution comparison, Mann-Whitney p-values) and ml_system/
    feature_synthesis.py (synthesized-feature correlation p-values) -
    three genuinely different hypothesis tests that all need the exact
    same correction procedure applied to whatever p-values they've
    already computed, not three near-identical inline copies of this
    loop that could silently drift from each other.

    Returns a NEW list, sorted ascending by p-value, with `bh_threshold`
    and `significant` (bool) added to every item - does not filter
    anything out, so a caller that wants "every tested item, verdict
    included" (event_autopsy.py) and a caller that wants "only the
    survivors" (fdr_accept()) both work from the same output.

    `n_tested`: the TRUE total number of hypotheses tested this run -
    NOT necessarily len(items). A caller may pass only pre-filtered
    finalists (fdr_accept() receives only a search's surviving
    candidates, not every one it scored) - the whole point of FDR
    correction is that the acceptance bar gets stricter the more things
    were tried, so it has to be scaled by the true search-wide count,
    never by how many happened to make a shortlist. Standard BH
    procedure: sort ascending by p-value, find the largest rank k where
    p_(k) <= (k / n_tested) * alpha, mark every item at or below that
    rank significant - NOT just every item whose own p-value
    individually clears alpha, which is what makes this correction
    actually scale with search size instead of being a fixed per-item
    bar in disguise."""
    if not items or n_tested <= 0:
        for item in items:
            item["bh_threshold"] = None
            item["significant"] = False
        return sorted(items, key=lambda d: d[p_value_key])

    scored = sorted(items, key=lambda d: d[p_value_key])
    largest_accepted_rank = 0
    for rank, item in enumerate(scored, start=1):
        item["bh_threshold"] = (rank / n_tested) * alpha
        if item[p_value_key] <= item["bh_threshold"]:
            largest_accepted_rank = rank
    for i, item in enumerate(scored, start=1):
        item["significant"] = i <= largest_accepted_rank
    return scored


def fdr_accept(candidates: list[dict], n_tested: int, alpha: float = FDR_ALPHA) -> list[dict]:
    """Benjamini-Hochberg false-discovery-rate correction for pattern
    conjunctions. `candidates`: dicts each with at least "wins" and "n"
    (pooled discovery-portion trade counts - see discover_patterns.py
    for how these get built from score_conjunction()'s era totals).
    `n_tested`: TOTAL candidate conjunctions the search evaluated this
    run (not just these finalists) - see bh_correct()'s own docstring
    for why that distinction matters. Returns the subset of
    `candidates` that survive, each with "p_value"/"bh_threshold" added
    for auditability (see discover_patterns.py's provenance storage)."""
    if not candidates or n_tested <= 0:
        return []
    scored = [{**c, "p_value": binomial_p_value(c["wins"], c["n"])} for c in candidates]
    corrected = bh_correct(scored, n_tested, alpha)
    return [c for c in corrected if c["significant"]]


def validate_on_confirmation_slice(candles: pd.DataFrame, occurred: pd.Series, direction: int,
                                    atr_series: pd.Series, discovery_end: int, n_candles: int,
                                    rr_ratio: float = RR_RATIO) -> dict:
    """The final, blind check - occurrences within [discovery_end,
    n_candles) ONLY, on data the search process never touched at any
    point, for any candidate, at any step. Reuses risk_reward.
    summarize_trades() directly - the identical hard-gate function every
    other pattern in this system is graded by, including its own
    internal out-of-sample sub-split - so a discovered pattern earns
    `qualifies: true` under exactly the same discipline as a hand-picked
    one, not a separate, possibly-weaker check.

    `rr_ratio`: see score_conjunction's docstring."""
    confirmation_mask = pd.Series(False, index=candles.index)
    confirmation_mask.iloc[discovery_end:n_candles] = True
    masked_occurred = occurred & confirmation_mask
    trades = simulate_trades(candles, masked_occurred, direction, atr_series=atr_series, rr_ratio=rr_ratio)
    return summarize_trades(
        trades, min_resolved=CONFIRMATION_MIN_RESOLVED_SAMPLES,
        oos_min_resolved=max(5, CONFIRMATION_MIN_RESOLVED_SAMPLES // 2),
        rr_ratio=rr_ratio,
    )
