"""
Labels for the ML challenger - deliberately NOT a new labeling scheme
or a reimplementation. `label_all_candles()` is a thin wrapper around
risk_reward.simulate_trades() - the EXACT function the rule-based system
uses to grade its own patterns - called with "occurred" true on every
candle instead of only where a hand-coded pattern fired. Same fixed
entry at the next candle's open, same ATR-based 1:4 R:R stop/target,
same walk-forward resolution with the same conservative same-candle-
tie-break, same MAX_LOOKAHEAD cutoff for "unresolved."

Why reuse instead of reimplementing: the whole point of running this as
a challenger to the rule-based system (see the top-level README) is a
fair, apples-to-apples comparison later. If the two systems graded trade
outcomes even slightly differently, a difference in results could just
mean a different label definition, not a better model - and it would be
a second place for the same trade-simulation bug to be introduced twice
with subtly different logic. Sharing simulate_trades() makes that
impossible by construction, and its output schema (signal_index, index,
resolved_index, candles_to_resolve, outcome, entry, stop, target, risk,
net_r) is exactly what risk_reward.summarize_trades() expects - so any
model-selected SUBSET of these rows (e.g. "every row the model scored
above threshold") can be graded with the identical hard-gate function
the rule-based system's pattern qualification uses. See model_registry.py.

Labels both directions separately (long and short) - like an ambiguous
candlestick pattern, a given market state might only have a real edge in
one direction, and that's for validation results to reveal, not a
hard-coded assumption baked into the label.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from risk_reward import MAX_LOOKAHEAD, RR_RATIO, atr, simulate_trades  # noqa: E402


def label_all_candles(candles: pd.DataFrame, direction: int, atr_series: pd.Series | None = None,
                       rr_ratio: float = RR_RATIO) -> pd.DataFrame:
    """One row per candle (indexed by `signal_index`, 0..len(candles)-1
    except the last, which has no next-open to enter at) - see module
    docstring for why this is simulate_trades() itself, not a parallel
    implementation.

    `rr_ratio` defaults to the system-wide 1:4 every existing caller
    relies on - it exists so train.py's multi-tier search (see
    train.RR_GRID) can re-label the SAME candles at a DIFFERENT reward
    multiple (a tighter scalp target or a wider swing target) without a
    second labeling implementation. Passing it changes only the target
    distance simulate_trades() uses - stop distance, entry timing, and
    MAX_LOOKAHEAD are all unaffected, so label_window() below still
    correctly bounds every rr_ratio's label."""
    a = atr_series if atr_series is not None else atr(candles)
    occurred = pd.Series(True, index=candles.index)
    return simulate_trades(candles, occurred, direction, atr_series=a, rr_ratio=rr_ratio)


def label_window() -> int:
    """How many future candles a label depends on - needed by
    validation.py's purging logic (a training example's label window
    must fully resolve before the validation period starts, or its
    "known outcome" is really information leaking backward from the
    future). Fixed at MAX_LOOKAHEAD + 1 (the +1 for the next-candle-open
    entry shift)."""
    return MAX_LOOKAHEAD + 1
