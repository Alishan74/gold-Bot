"""
Pattern Discovery Engine - Layer 0: primitive SYNTHESIS.

Everything in discovery_primitives.py (Layer 1) is still, at bottom, a
human-chosen TEMPLATE (RSI, ADX, slope, Donchian position, ...) even
though the search (Layer 2) and validation (Layer 3) decide entirely on
their own which specific instances and combinations of those templates
actually earn a place. This module removes that last human choice: it
COMPOSES brand-new candidate primitives itself, out of a small grammar
of building blocks, and keeps only the ones that empirically clear the
exact same statistical bar as everything else. Nobody chose "rolling
20-bar std of returns compared against its own 80th percentile" - the
search found it, or didn't, on its own.

Grammar: a BASE series (close / hl2 / returns / true-range-%) run
through a rolling TRANSFORM (mean / std / slope / percentile-rank /
z-score) over a WINDOW, compared against a rolling PERCENTILE of its own
trailing history - not a raw fixed number. Self-normalizing on purpose:
a raw price-level threshold silently stops meaning anything as gold
drifts from ~$1200 to $2600+ across a 20-year history; "above the 80th
percentile of its own last 200 bars" stays meaningful throughout, the
same reasoning discovery_primitives._surprise_zscore_on_candles already
applies to fundamental surprises (z-scored against trailing releases,
never a raw number).

Evolutionary, not brute-force enumeration of the whole grammar (which
would run into the thousands): start with GENERATION_SIZE random
expressions, score each SOLO (Layer 3's worst-era Wilson lower bound -
the identical scoring function discovery_search.py's own seeding step
uses), keep the top SURVIVORS_PER_GEN, and build the next generation
from MUTATIONS of those survivors plus a slug of fresh random blood (so
it doesn't prematurely converge on one lucky corner of the grammar).
Repeat N_GENERATIONS times. The final top N_OUTPUT_MAX survivors that
clear SYNTH_MIN_SCORE become real discovery_primitives.Primitive objects
- passed into discovery_search.search_conjunctions(...,
extra_primitives=...), where they compete and combine with the
hand-designed catalog on identical terms.

CRITICAL correctness requirement, not optional plumbing: every
expression tried during synthesis - survivor or not - is a genuine
"test" in exactly the multiple-comparisons sense discovery_search.py's
own primitives/conjunctions already are. discover_patterns.py MUST add
this function's returned `n_tested` to the beam search's own n_tested
BEFORE discovery_validation.fdr_accept() runs. Skipping this would let a
primitive that only cleared the worst-era bar by pure luck (out of
however many random expressions were actually tried) sneak past FDR
under an artificially lenient bar - precisely the "test enough things
and something clears a FIXED bar by chance" failure mode this entire
system was built to prevent. There is no other defense against that
specific leak once synthesis exists; get this one accounting detail
wrong and the rest of the rigor here is theater.

Auditability: every synthesized primitive's exact expression (base,
transform, window, comparator, percentile) is fully serializable via
SynthesizedExpression.to_dict()/from_dict(). discover_patterns.py stores
it under a pattern's own discovery_meta.synthesized_expressions, so
"discovered__<hash> was built from synth_close_slope20_gt80" is always
inspectable months later - never an opaque black box, same "never
hidden, never just the model said so" standard this whole engine holds
itself to. This is also what makes LIVE re-evaluation possible: a
synthesized primitive's `.fn` closure only exists in memory for the run
that created it, so discover_patterns.is_discovered_pattern_active()
reconstructs a fresh SynthesizedExpression from the stored dict instead
of depending on that closure surviving across process restarts.

Look-ahead safety: the percentile-threshold baseline is computed via
`.shift(1).rolling(...).quantile(...)` - the current row is NEVER part
of its own comparison baseline, the identical discipline discovery_
primitives._surprise_zscore_on_candles already uses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from discovery_primitives import Primitive, rolling_slope
from discovery_validation import score_conjunction

GENERATION_SIZE = 60
N_GENERATIONS = 3
SURVIVORS_PER_GEN = 12
N_OUTPUT_MAX = 15
SYNTH_MIN_SCORE = 0.45     # same bar as discovery_search.MIN_START_SCORE - no point keeping a synthesized primitive that wouldn't even seed a beam
PCT_LOOKBACK = 200          # fixed, shared IDENTICALLY between discovery-time scoring and live re-evaluation - a different lookback in either place would be training-serving skew

BASE_SERIES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "close": lambda df: df["close"],
    "hl2": lambda df: (df["high"] + df["low"]) / 2,
    "returns": lambda df: df["close"].pct_change(),
    "rangepct": lambda df: (df["high"] - df["low"]) / df["close"].replace(0, np.nan),
}

TRANSFORMS: dict[str, Callable[[pd.Series, int], pd.Series]] = {
    "raw": lambda s, w: s,
    "mean": lambda s, w: s.rolling(w).mean(),
    "std": lambda s, w: s.rolling(w).std(),
    "slope": lambda s, w: rolling_slope(s, w),
    "pctrank": lambda s, w: s.rolling(w + 1).apply(
        (lambda x: float((x[-1] > x[:-1]).mean()) if len(x) > 1 else np.nan), raw=True),
    "zscore": lambda s, w: (s - s.rolling(w).mean()) / s.rolling(w).std().replace(0, np.nan),
}

WINDOWS = (5, 10, 20, 50)
COMPARATORS = (">", "<")
PCT_THRESHOLDS = (0.6, 0.7, 0.8, 0.9)
# TRANSFORMS["raw"] ignores `window` entirely (it's just the base series,
# unchanged) - without canonicalizing to a single fixed value here,
# _random_expression/_mutate would produce up to len(WINDOWS) DIFFERENT
# names (synth_close_raw5_gt80, synth_close_raw20_gt80, ...) that are
# all, in fact, the IDENTICAL boolean series - wasted search budget, and
# worse, each would count as a SEPARATE trial toward n_tested despite
# being zero genuinely new information, silently making the FDR bar
# stricter than the true number of distinct hypotheses actually tested
# warrants. Canonicalizing keeps every name unique-in-substance, not just
# unique-in-string.
_RAW_CANONICAL_WINDOW = WINDOWS[0]


@dataclass(frozen=True)
class SynthesizedExpression:
    base: str
    transform: str
    window: int
    comparator: str
    pct_threshold: float

    @property
    def name(self) -> str:
        comp = "gt" if self.comparator == ">" else "lt"
        thresh = int(round(self.pct_threshold * 100))
        return f"synth_{self.base}_{self.transform}{self.window}_{comp}{thresh}"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SynthesizedExpression":
        return SynthesizedExpression(
            base=d["base"], transform=d["transform"], window=int(d["window"]),
            comparator=d["comparator"], pct_threshold=float(d["pct_threshold"]),
        )

    def evaluate(self, candles: pd.DataFrame, events: "pd.DataFrame | None" = None) -> pd.Series:
        base_series = BASE_SERIES[self.base](candles)
        computed = TRANSFORMS[self.transform](base_series, self.window)
        q = self.pct_threshold if self.comparator == ">" else (1 - self.pct_threshold)
        rolling_q = computed.shift(1).rolling(PCT_LOOKBACK, min_periods=30).quantile(q)
        if self.comparator == ">":
            return computed > rolling_q
        return computed < rolling_q


def evaluate_expression(expr_or_dict, candles: pd.DataFrame,
                         events: "pd.DataFrame | None" = None) -> pd.Series:
    """NaN-safe (fillna(False), same convention every primitive evaluator
    in this codebase uses). Accepts either a live SynthesizedExpression
    or its serialized dict form (discovery_meta.synthesized_expressions'
    own shape), so a caller reconstructing from disk doesn't need to
    remember to call from_dict() itself first."""
    expr = expr_or_dict if isinstance(expr_or_dict, SynthesizedExpression) else SynthesizedExpression.from_dict(expr_or_dict)
    return expr.evaluate(candles, events).fillna(False)


def _canonicalize(transform: str, window: int) -> int:
    """See _RAW_CANONICAL_WINDOW above - "raw" has no window dimension in
    substance, so every "raw" expression collapses to the SAME window
    regardless of what was drawn, keeping every distinct NAME a distinct
    boolean series too."""
    return _RAW_CANONICAL_WINDOW if transform == "raw" else window


def _random_expression(rng: np.random.Generator) -> SynthesizedExpression:
    transform = str(rng.choice(list(TRANSFORMS)))
    return SynthesizedExpression(
        base=str(rng.choice(list(BASE_SERIES))),
        transform=transform,
        window=_canonicalize(transform, int(rng.choice(WINDOWS))),
        comparator=str(rng.choice(COMPARATORS)),
        pct_threshold=float(rng.choice(PCT_THRESHOLDS)),
    )


def _mutate(expr: SynthesizedExpression, rng: np.random.Generator) -> SynthesizedExpression:
    """Perturbs exactly ONE field - keeps most mutations close to an
    already-promising expression (hill-climbing) instead of jumping
    somewhere unrelated every time, the same "beam search, not brute
    force" tractability reasoning discovery_search.py's own extension
    step is built on."""
    field = rng.choice(["base", "transform", "window", "comparator", "pct_threshold"])
    kwargs = asdict(expr)
    if field == "base":
        kwargs["base"] = str(rng.choice(list(BASE_SERIES)))
    elif field == "transform":
        kwargs["transform"] = str(rng.choice(list(TRANSFORMS)))
    elif field == "window":
        kwargs["window"] = int(rng.choice(WINDOWS))
    elif field == "comparator":
        kwargs["comparator"] = str(rng.choice(COMPARATORS))
    elif field == "pct_threshold":
        kwargs["pct_threshold"] = float(rng.choice(PCT_THRESHOLDS))
    kwargs["window"] = _canonicalize(kwargs["transform"], kwargs["window"])
    return SynthesizedExpression(**kwargs)


def _family_for(expr: SynthesizedExpression) -> str:
    """Grouped by TRANSFORM TYPE, not a borrowed hand-designed-family
    label - two synthesized expressions sharing a transform (e.g. two
    different rolling-std thresholds) are near-duplicates the same way
    two momentum primitives are, so discovery_search.py's cross-family
    rule keeps them from combining with each other; a slope-based and a
    std-based synthesized primitive are NOT near-duplicates, so they
    stay free to combine. The real backstop against a redundant addition
    either way is discovery_search.py's own MIN_IMPROVEMENT requirement
    (a second primitive must independently improve the score, correct
    family label or not) - this tagging is a heuristic layered on top of
    that, not the actual defense."""
    return f"synthesized_{expr.transform}"


def synthesize_primitives(candles: pd.DataFrame, events: "pd.DataFrame | None", atr_series: pd.Series,
                           discovery_end: int, seed: "int | None" = None
                           ) -> tuple[list[Primitive], dict[str, dict], int]:
    """Returns (new_primitives, expr_by_name, n_tested):
      - new_primitives: real Primitive objects, ready to pass into
        search_conjunctions(..., extra_primitives=new_primitives).
      - expr_by_name: {primitive_name: serialized expression dict} for
        every returned primitive - discover_patterns.py stores this
        under discovery_meta.synthesized_expressions on any accepted
        pattern that used one, for later reconstruction.
      - n_tested: see module docstring - MUST be folded into the total
        passed to discovery_validation.fdr_accept().

    Every returned Primitive's `.fn` is a closure over its own
    SynthesizedExpression - fine for use within THIS run's search, but
    NOT how a discovered pattern re-evaluates itself on a later day (see
    discover_patterns.is_discovered_pattern_active, which reconstructs
    from expr_by_name's serialized form instead of depending on this
    in-memory closure surviving)."""
    rng = np.random.default_rng(seed)
    n_tested = 0
    seen_names: set[str] = set()
    scored: list[tuple[SynthesizedExpression, int, dict]] = []

    population = [_random_expression(rng) for _ in range(GENERATION_SIZE)]
    for _generation in range(N_GENERATIONS):
        gen_scored: list[tuple[SynthesizedExpression, int, dict]] = []
        for expr in population:
            if expr.name in seen_names:
                continue
            seen_names.add(expr.name)
            try:
                series = expr.evaluate(candles, events).fillna(False)
            except Exception:
                continue  # a malformed random composition (e.g. window >= available history) - skip, don't crash the whole run
            n_tested += 1
            best = None
            for d in (1, -1):
                result = score_conjunction(candles, series, d, atr_series, discovery_end)
                if best is None or result["worst_era_score"] > best[1]["worst_era_score"]:
                    best = (d, result)
            gen_scored.append((expr, best[0], best[1]))
        gen_scored.sort(key=lambda t: t[2]["worst_era_score"], reverse=True)
        scored.extend(gen_scored)

        survivors = gen_scored[:SURVIVORS_PER_GEN]
        if not survivors:
            break
        mutants_per_survivor = max(1, GENERATION_SIZE // (2 * len(survivors)))
        population = [_mutate(s[0], rng) for s in survivors for _ in range(mutants_per_survivor)]
        population += [_random_expression(rng) for _ in range(GENERATION_SIZE // 4)]

    # Dedup by name (mutation can re-derive the same expression more than
    # once across generations) keeping each name's best-scoring instance.
    best_by_name: dict[str, tuple[SynthesizedExpression, int, dict]] = {}
    for expr, direction, result in scored:
        current = best_by_name.get(expr.name)
        if current is None or result["worst_era_score"] > current[2]["worst_era_score"]:
            best_by_name[expr.name] = (expr, direction, result)

    survivors_final = sorted(best_by_name.values(), key=lambda t: t[2]["worst_era_score"], reverse=True)
    survivors_final = [t for t in survivors_final if t[2]["worst_era_score"] >= SYNTH_MIN_SCORE][:N_OUTPUT_MAX]

    primitives: list[Primitive] = []
    expr_by_name: dict[str, dict] = {}
    for expr, direction, _result in survivors_final:
        primitives.append(Primitive(
            name=expr.name, family=_family_for(expr), direction_hint=direction,
            fn=(lambda df, ev, e=expr: e.evaluate(df, ev)),
        ))
        expr_by_name[expr.name] = expr.to_dict()

    return primitives, expr_by_name, n_tested
