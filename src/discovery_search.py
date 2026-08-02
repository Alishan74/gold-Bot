"""
Pattern Discovery Engine - Layer 2: the actual "self pattern maker."

Greedy/BEAM construction, not brute-force enumeration of every possible
primitive combination - the difference between a search that stays
tractable and honest, and one that recreates the "unconstrained search
over raw data" trap this whole design was built to avoid (see
discovery_primitives.py's own docstring for why the primitive count
itself is bounded, not infinite).

How it works, precisely:
  1. Score every single primitive alone (Layer 3's worst-era Wilson
     lower bound). Anything that doesn't clear MIN_START_SCORE is
     dropped immediately - no point spending search budget extending a
     starting point with no baseline promise at all.
  2. From each surviving single primitive, grow it one primitive at a
     time. At each step, only an addition that improves the worst-era
     score by at least MIN_IMPROVEMENT survives - every single primitive
     in a discovered pattern has to have PROVEN incremental value, not
     just be along for the ride inside a big conjunction that happens to
     pass. This is what keeps a discovered pattern genuinely readable
     ("X matters, then Y also independently helps, then Z too") instead
     of an opaque bundle.
  3. BEAM width (not pure greedy, not brute force): at each depth, keep
     the top BEAM_WIDTH partial conjunctions by score, not just the
     single best - explores meaningfully more of the space than greedy
     hill-climbing while staying polynomial in cost, not combinatorial.
  4. Two hard structural constraints on every extension, mirroring
     combo_patterns.py's own rules for hand-picked combos exactly:
       - Cross-family only: never add a primitive from a family already
         present in the conjunction (two momentum primitives together
         tend to be near-duplicates of each other, not real confluence).
       - No contradictions: never add a primitive whose nonzero
         direction hint disagrees with the conjunction's established
         direction.
  5. MIN_DEPTH = 2 - a single primitive is NEVER emitted as a final
     candidate pattern, no matter how well it scores alone. This is the
     literal "not just NFP alone, not just two candles alone" rule -
     genuine multi-condition confluence is required by construction, not
     as an afterthought filter.

Every primitive's boolean Series is computed ONCE and cached (primitives
are pure functions of (candles, events), independent of which
conjunction is being built) - reused across every conjunction that
includes it, not recomputed per trial.

`extra_primitives` (search_conjunctions' own parameter): lets a caller
fold in primitives beyond the fixed discovery_primitives.PRIMITIVES
catalog for just this one run - specifically, discovery_synthesis.py's
(Layer 0) genetically-synthesized primitives. Evaluated via each
Primitive object's OWN `.fn` directly rather than discovery_primitives'
name-indexed evaluate_primitive() (which only knows about the static
catalog) - this works identically for hand-designed and synthesized
Primitive objects, since both are the same dataclass shape.
"""
from __future__ import annotations

import pandas as pd

from discovery_primitives import PRIMITIVES, Primitive
from discovery_validation import score_conjunction

MIN_START_SCORE = 0.45     # a single primitive scoring below this alone isn't worth extending
MIN_IMPROVEMENT = 0.01     # an addition must improve the worst-era score by at least this much
BEAM_WIDTH = 5
MAX_DEPTH = 4
MIN_DEPTH = 2               # never emit a single-primitive "pattern" - see module docstring


class Conjunction:
    """A candidate self-constructed pattern: an ordered list of primitive
    names, its established direction, and its worst-era validation score
    (Layer 3). `occurred`: cached AND of every component's boolean
    Series - computed incrementally (parent's occurred & new primitive's
    Series), never recomputed from scratch as depth grows. `primitives_by_name`:
    the FULL name->Primitive map this conjunction was built against (fixed
    catalog + any extra_primitives for this run) - stored per-instance
    rather than read from a module-level global, so `.families` resolves
    correctly even when some component is a synthesized primitive that
    doesn't exist in discovery_primitives.PRIMITIVES_BY_NAME at all."""

    __slots__ = ("primitives", "direction", "occurred", "score_result", "_primitives_by_name")

    def __init__(self, primitives: tuple[str, ...], direction: int,
                 occurred: pd.Series, score_result: dict, primitives_by_name: dict[str, Primitive]):
        self.primitives = primitives
        self.direction = direction
        self.occurred = occurred
        self.score_result = score_result
        self._primitives_by_name = primitives_by_name

    @property
    def worst_era_score(self) -> float:
        return self.score_result["worst_era_score"]

    @property
    def families(self) -> set[str]:
        return {self._primitives_by_name[p].family for p in self.primitives}


def _is_compatible_extension(existing_families: set[str], existing_direction: int,
                              candidate: Primitive) -> bool:
    if candidate.family in existing_families:
        return False
    if existing_direction != 0 and candidate.direction_hint != 0 and candidate.direction_hint != existing_direction:
        return False
    return True


def _score_and_wrap(primitives: tuple[str, ...], direction: int, occurred: pd.Series,
                     candles: pd.DataFrame, atr_series: pd.Series, discovery_end: int,
                     primitives_by_name: dict[str, Primitive]) -> Conjunction:
    result = score_conjunction(candles, occurred, direction, atr_series, discovery_end)
    return Conjunction(primitives, direction, occurred, result, primitives_by_name)


def search_conjunctions(candles: pd.DataFrame, events: "pd.DataFrame | None", atr_series: pd.Series,
                         discovery_end: int, beam_width: int = BEAM_WIDTH, max_depth: int = MAX_DEPTH,
                         min_depth: int = MIN_DEPTH, min_start_score: float = MIN_START_SCORE,
                         min_improvement: float = MIN_IMPROVEMENT,
                         extra_primitives: "list[Primitive] | None" = None,
                         capture_all: bool = False,
                         base_primitives: "list[Primitive] | None" = None) -> tuple[list[Conjunction], int, list[Conjunction]]:
    """Runs the full beam search. Returns (final_candidates, n_tested,
    all_scored) - `n_tested` is the TOTAL number of distinct conjunctions
    actually scored over the whole run (every starting primitive, every
    extension trial, whether or not it survived) - this is the count
    discovery_validation.fdr_accept() needs to correctly scale its
    acceptance bar, so it has to be the true search-wide total, not just
    len(final_candidates). `extra_primitives`: see module docstring -
    folded into the candidate pool for just this call, not the global
    catalog.

    `capture_all`: when True, `all_scored` holds EVERY conjunction this
    run actually scored (including ones that never cleared
    `min_start_score`/`min_improvement` or never made it into the beam) -
    the strict production path (discover_patterns.py) never sets this,
    since it only ever needs the beam's own survivors. It exists for
    scripts/explore_setups.py: with `min_start_score`/`min_improvement`
    lowered to admit weaker conjunctions into the beam too, `all_scored`
    is how a caller sees the RAW candidate set the search actually built,
    for independent re-evaluation at other R:R ratios - not just whatever
    happened to survive THIS run's particular pruning bar. When False
    (the default), `all_scored` is always []; behavior and return values
    for existing callers are otherwise unchanged.

    `base_primitives`: defaults to None, which uses discovery_primitives.
    PRIMITIVES (the full fixed catalog) exactly as before this parameter
    existed - every production caller (discover_patterns.py) leaves this
    unset. Passing a list REPLACES the base catalog instead of using the
    global one (`extra_primitives` still gets added on top either way) -
    scripts/explore_setups.py uses this for `--technicals-only`, so
    fundamental-family primitives never enter the search at all rather
    than being filtered out after the fact."""
    n_tested = 0
    all_scored: list[Conjunction] = []
    base = base_primitives if base_primitives is not None else PRIMITIVES
    all_primitives = list(base) + list(extra_primitives or [])
    primitives_by_name: dict[str, Primitive] = {p.name: p for p in all_primitives}

    # Cache every primitive's raw boolean Series once - independent of
    # which conjunction it ends up in. Uses each Primitive's OWN `.fn`
    # directly (not the name-indexed discovery_primitives.evaluate_
    # primitive helper) so this works identically whether `p` came from
    # the fixed catalog or from extra_primitives.
    primitive_series: dict[str, pd.Series] = {
        p.name: p.fn(candles, events).fillna(False) for p in all_primitives
    }

    # Step 1: score every primitive alone, keep survivors as depth-1 beam seeds.
    beam: list[Conjunction] = []
    for p in all_primitives:
        direction = p.direction_hint  # a lone ambiguous (0-hint) primitive can't be tested standalone without picking a side
        directions_to_try = (1, -1) if direction == 0 else (direction,)
        for d in directions_to_try:
            n_tested += 1
            conj = _score_and_wrap((p.name,), d, primitive_series[p.name], candles, atr_series,
                                    discovery_end, primitives_by_name)
            if capture_all:
                all_scored.append(conj)
            if conj.worst_era_score >= min_start_score:
                beam.append(conj)
    beam.sort(key=lambda c: c.worst_era_score, reverse=True)
    beam = beam[:beam_width]

    final_candidates: list[Conjunction] = []
    if min_depth <= 1:
        final_candidates.extend(beam)

    depth = 1
    while beam and depth < max_depth:
        depth += 1
        next_round: list[Conjunction] = []
        for parent in beam:
            existing_families = parent.families
            for p in all_primitives:
                if p.name in parent.primitives:
                    continue
                if not _is_compatible_extension(existing_families, parent.direction, p):
                    continue
                trial_direction = parent.direction if parent.direction != 0 else p.direction_hint
                directions_to_try = (1, -1) if trial_direction == 0 else (trial_direction,)
                for d in directions_to_try:
                    n_tested += 1
                    trial_occurred = parent.occurred & primitive_series[p.name]
                    trial = _score_and_wrap(
                        parent.primitives + (p.name,), d, trial_occurred, candles, atr_series,
                        discovery_end, primitives_by_name,
                    )
                    if capture_all:
                        all_scored.append(trial)
                    if trial.worst_era_score >= parent.worst_era_score + min_improvement:
                        next_round.append(trial)
        next_round.sort(key=lambda c: c.worst_era_score, reverse=True)
        beam = next_round[:beam_width]
        if depth >= min_depth:
            final_candidates.extend(beam)

    # Dedup: a conjunction can be re-reached via different extension
    # orders (A then B scores the same as B then A) - keep the
    # highest-scoring instance of each distinct primitive SET+direction.
    seen: dict[tuple[frozenset, int], Conjunction] = {}
    for c in final_candidates:
        key = (frozenset(c.primitives), c.direction)
        if key not in seen or c.worst_era_score > seen[key].worst_era_score:
            seen[key] = c
    deduped = sorted(seen.values(), key=lambda c: c.worst_era_score, reverse=True)

    return deduped, n_tested, all_scored
