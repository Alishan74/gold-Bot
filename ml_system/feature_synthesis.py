"""
Feature synthesis - the ML challenger's equivalent of discovery_
synthesis.py's genetic PRIMITIVE synthesis, but composing brand-new
NUMERIC FEATURES instead of new boolean primitives. Every one of
features.py's 80+ engineered numbers is still, at bottom, a human-chosen
template (RSI, ATR-normalized distance, efficiency ratio, ...) even
though the gradient-boosted model decides entirely on its own which
INTERACTIONS among them matter. This module removes that remaining human
choice for a specific kind of interaction: it COMPOSES new derived
features itself, out of a small grammar of arithmetic operators over
EXISTING feature columns, and keeps only the ones that empirically show
real, multi-era, FDR-surviving, confirmation-slice-surviving association
with trade outcomes (see scripts/synthesize_features.py for the full
three-layer validation - this module only handles the search/grammar,
not the accept/reject decision).

Grammar, deliberately flat and shallow (not a deep recursive expression
tree) - two LEAF feature columns (from features.compute_features()'s
own output, never raw OHLC/volume - already normalized, already
look-ahead-safe, so a synthesized feature inherits both properties for
free) combined by a BINARY operator, optionally wrapped in a UNARY one.
Same "simple grammar over combinatorial generality" philosophy discovery_
synthesis.py's own module docstring argues for: a flat 4-field
expression (leaf_a, leaf_b, binary_op, unary_wrap) is trivially
mutation-friendly, trivially serializable, and - critically - trivially
provable to be look-ahead-safe, since every operator here is a PURE,
POINTWISE (same-row) function of two already-causal columns: combining
two numbers that are each already known at row i, at row i, can never
introduce a look-ahead leak, no new rolling window is ever opened here.
That is the actual safety argument, not an assumption - re-verify it
before ever adding an operator that looks at more than the current row.

Symmetric operators (add/mul/max/min) canonicalize their leaf order
(alphabetical) at construction time - `add(A, B)` and `add(B, A)`
compute the IDENTICAL series, and without canonicalizing, both would
count as separate trials toward n_tested for zero genuinely new
information, silently making the FDR bar looser than the true number of
distinct hypotheses tested actually warrants (the exact inverse of the
mistake discovery_synthesis.py's own `_RAW_CANONICAL_WINDOW` comment
warns about, same underlying principle).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

GENERATION_SIZE = 60
N_GENERATIONS = 3
SURVIVORS_PER_GEN = 12
N_OUTPUT_MAX = 15
# A raw Spearman correlation against a noisy binary trading outcome is
# NOT the same scale as a boolean pattern's win-rate-based Wilson lower
# bound (which naturally clusters well above 0 since win rates cluster
# around 50%) - real, useful single-feature correlations in this domain
# are typically small (0.01-0.05). This floor is intentionally modest;
# the real rigor is the FDR correction + blind confirmation slice
# scripts/synthesize_features.py applies on top of it, not this number
# alone.
SYNTH_MIN_SCORE = 0.02

BINARY_OPS: dict[str, Callable[[pd.Series, pd.Series], pd.Series]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "safe_div": lambda a, b: a / b.replace(0, np.nan),
    "max": lambda a, b: pd.concat([a, b], axis=1).max(axis=1),
    "min": lambda a, b: pd.concat([a, b], axis=1).min(axis=1),
}
UNARY_OPS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "none": lambda x: x,
    "abs": lambda x: x.abs(),
    "neg": lambda x: -x,
}
SYMMETRIC_OPS = {"add", "mul", "max", "min"}


def _canonicalize_leaves(binary_op: str, leaf_a: str, leaf_b: str) -> tuple[str, str]:
    """See module docstring - a symmetric op's two leaves are sorted so
    the SAME pair always produces the SAME name regardless of which
    order they were drawn/mutated in."""
    if binary_op in SYMMETRIC_OPS and leaf_a > leaf_b:
        return leaf_b, leaf_a
    return leaf_a, leaf_b


@dataclass(frozen=True)
class SynthesizedFeature:
    leaf_a: str
    leaf_b: str
    binary_op: str
    unary_wrap: str = "none"

    def __post_init__(self):
        a, b = _canonicalize_leaves(self.binary_op, self.leaf_a, self.leaf_b)
        object.__setattr__(self, "leaf_a", a)
        object.__setattr__(self, "leaf_b", b)

    @property
    def name(self) -> str:
        return f"synth_{self.binary_op}_{self.leaf_a}_{self.leaf_b}_{self.unary_wrap}"

    @property
    def is_degenerate(self) -> bool:
        """leaf_a == leaf_b makes sub/safe_div trivially constant
        (always 0 or always 1) and add/mul/max/min trivially redundant
        with the leaf itself - never a genuinely new feature, so this
        is checked and skipped BEFORE spending any search budget on it,
        not discovered after the fact via a degenerate correlation."""
        return self.leaf_a == self.leaf_b

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SynthesizedFeature":
        return SynthesizedFeature(
            leaf_a=d["leaf_a"], leaf_b=d["leaf_b"],
            binary_op=d["binary_op"], unary_wrap=d.get("unary_wrap", "none"),
        )

    def evaluate(self, feature_table: pd.DataFrame) -> pd.Series:
        a, b = feature_table[self.leaf_a], feature_table[self.leaf_b]
        combined = BINARY_OPS[self.binary_op](a, b)
        return UNARY_OPS[self.unary_wrap](combined)


def evaluate_expression(expr_or_dict, feature_table: pd.DataFrame) -> pd.Series:
    """Accepts either a live SynthesizedFeature or its serialized dict
    form (synthesized_features/<symbol>_<tf>.json's own shape), so a
    caller reconstructing from disk (train.py, live_signal.py) doesn't
    need to remember to call from_dict() itself first - same convenience
    wrapper discovery_synthesis.evaluate_expression() already provides
    for the boolean-primitive side."""
    expr = expr_or_dict if isinstance(expr_or_dict, SynthesizedFeature) else SynthesizedFeature.from_dict(expr_or_dict)
    return expr.evaluate(feature_table)


def apply_synthesized_features(feature_table: pd.DataFrame, synth_defs: list[dict]) -> pd.DataFrame:
    """Evaluates every synthesized feature definition against
    `feature_table` and returns a NEW DataFrame of just those columns
    (same index as feature_table), ready to `pd.concat([feature_table,
    apply_synthesized_features(...)], axis=1)` - the exact pattern
    train.py already uses for cross-timeframe context columns. A
    definition whose leaf columns aren't present in `feature_table`
    (e.g. it references a feature that doesn't exist in this run's
    single-timeframe feature set for some reason) evaluates to an
    all-NaN column rather than crashing - same graceful-degradation
    convention every optional feature block in features.py already
    follows for missing source columns (volume, order-flow, ...).
    Empty (0-column) DataFrame if `synth_defs` is empty - "nothing
    synthesized for this timeframe yet" is normal, not an error."""
    if not synth_defs:
        return pd.DataFrame(index=feature_table.index)
    out = {}
    for d in synth_defs:
        expr = SynthesizedFeature.from_dict(d)
        try:
            out[expr.name] = evaluate_expression(expr, feature_table)
        except KeyError:
            out[expr.name] = pd.Series(np.nan, index=feature_table.index)
    return pd.DataFrame(out, index=feature_table.index)


def load_synthesized_features(synth_dir: "Path | str", symbol: str, timeframe: str) -> list[dict]:
    """Every accepted synthesized feature's serialized definition for
    this (symbol, timeframe) - see scripts/synthesize_features.py for
    what writes this file and the full validation each entry already
    passed before landing here. Empty list (not an error) if the file
    doesn't exist - "synthesis hasn't been run for this timeframe yet"
    is the normal, expected state until a human runs the script."""
    path = Path(synth_dir) / f"{symbol}_{timeframe}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload.get("accepted", [])


def random_feature(rng: np.random.Generator, feature_columns: list[str]) -> SynthesizedFeature:
    leaf_a, leaf_b = rng.choice(feature_columns, size=2, replace=False)
    binary_op = str(rng.choice(list(BINARY_OPS)))
    return SynthesizedFeature(
        leaf_a=str(leaf_a), leaf_b=str(leaf_b), binary_op=binary_op,
        unary_wrap=str(rng.choice(list(UNARY_OPS))),
    )


def mutate_feature(expr: SynthesizedFeature, rng: np.random.Generator, feature_columns: list[str]) -> SynthesizedFeature:
    """Perturbs exactly ONE field - keeps most mutations close to an
    already-promising expression (hill-climbing) instead of jumping
    somewhere unrelated every time, the identical reasoning discovery_
    synthesis._mutate() already documents for the boolean-primitive
    side."""
    field = rng.choice(["leaf_a", "leaf_b", "binary_op", "unary_wrap"])
    kwargs = asdict(expr)
    if field == "leaf_a":
        kwargs["leaf_a"] = str(rng.choice(feature_columns))
    elif field == "leaf_b":
        kwargs["leaf_b"] = str(rng.choice(feature_columns))
    elif field == "binary_op":
        kwargs["binary_op"] = str(rng.choice(list(BINARY_OPS)))
    elif field == "unary_wrap":
        kwargs["unary_wrap"] = str(rng.choice(list(UNARY_OPS)))
    return SynthesizedFeature(**kwargs)
