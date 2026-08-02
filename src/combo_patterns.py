"""
Pattern CONFLUENCE mining: two individually-detected patterns firing on
the SAME candle at once. This is the highest-leverage way to get MORE
qualifying signals without loosening the hard 1:4 R:R / 60% win-rate gate
even slightly - confluence of two independent-ish conditions is both
rarer (fewer, more selective triggers) and typically more reliable (each
condition filters out noise the other one doesn't catch alone) than
either pattern by itself. That is what "more signals, but more perfect"
actually requires, instead of trading one for the other.

Design choices, and why:

- Only CROSS-family pairs are combined - candlestick, indicator, session,
  fundamental, support_resistance (support_resistance.py: swing/
  round-number/pivot rejections and bounces), and smc (smc_patterns.py:
  liquidity sweeps and fair value gaps), every pairwise combination
  across those 6 families - two patterns from the SAME family (e.g. doji
  + inside_bar, both candlestick shapes) tend to be near-duplicates of
  each other by construction (heavily overlapping candles), not real
  confluence between independent signals. A candlestick reversal shape
  firing exactly AT a level (e.g. combo__bearish_engulfing__swing_high_
  rejection) is the textbook case this mechanism exists for: two
  genuinely independent conditions - "the shape says reversal" and "the
  price says this is a level that's mattered before" - agreeing at once.
- Directly CONTRADICTORY pairs are dropped: if both components have a
  nonzero directional hint (patterns.DIRECTION_HINT) and they disagree
  (one bullish, one bearish), they can't describe the same trade and are
  never combined.
- The pair list (combo_pairs()) is a SINGLE source of truth, imported by
  both build_pattern_library.py (mining) and signal_engine.py (live
  detection) - so live signals can never fire on a combo the miner never
  actually tested, or vice versa. That exact "two systems silently drift
  apart" failure mode has bitten this codebase before (see README.md,
  "Second pass" / dedup fix) - sharing this module is how it's avoided
  here from the start.

Honest statistical caveat: testing hundreds of combinations and keeping
only the ones that clear a 60% win-rate bar is exactly the setup for
false positives from chance alone (multiple-comparisons risk). Two
mitigations are in place, both real, neither a complete fix:
  1. Combos require MORE resolved samples than atomic patterns
     (COMBO_MIN_RESOLVED_SAMPLES / COMBO_OOS_MIN_RESOLVED_SAMPLES in
     risk_reward.py) before being trusted at all - rarer joint events
     need more confirmations.
  2. Every combo still has to independently clear the SAME out-of-sample
     split every atomic pattern does (risk_reward.summarize_trades) - a
     combo that only looks good from overfitting the full sample is far
     less likely to also clear the bar on a held-out slice it never
     touched while being selected.
This is a real reduction in false-positive risk, not a guarantee - a
combo showing `qualifies: true` still deserves the same skepticism as any
statistically-mined signal, just as documented for atomic patterns.
"""
from __future__ import annotations

import pandas as pd

from fundamental_patterns import FUNDAMENTAL_PATTERN_NAMES
from patterns import CANDLESTICK_PATTERNS, INDICATOR_PATTERNS, pattern_direction_hint
from session_patterns import SESSION_PATTERN_NAMES
from smc_patterns import SMC_PATTERN_NAMES
from support_resistance import SUPPORT_RESISTANCE_PATTERN_NAMES

FAMILIES: dict[str, set[str]] = {
    "candlestick": set(CANDLESTICK_PATTERNS),
    "indicator": set(INDICATOR_PATTERNS),
    "session": set(SESSION_PATTERN_NAMES),
    "fundamental": set(FUNDAMENTAL_PATTERN_NAMES),
    "support_resistance": set(SUPPORT_RESISTANCE_PATTERN_NAMES),
    "smc": set(SMC_PATTERN_NAMES),
}

_FAMILY_NAMES = sorted(FAMILIES)

COMBO_PREFIX = "combo__"


def combo_name(name_a: str, name_b: str) -> str:
    return f"{COMBO_PREFIX}{name_a}__{name_b}"


def combo_pairs() -> list[tuple[str, str]]:
    """Every valid (name_a, name_b) cross-family pair, deterministic
    order (alphabetical by family, then by name) so this list - and
    therefore every combo's identity - never depends on dict/set
    iteration order. Shared by mining and live detection; see module
    docstring for why that sharing matters."""
    pairs = []
    for i, fam_a in enumerate(_FAMILY_NAMES):
        for fam_b in _FAMILY_NAMES[i + 1:]:
            for name_a in sorted(FAMILIES[fam_a]):
                hint_a = pattern_direction_hint(name_a)
                for name_b in sorted(FAMILIES[fam_b]):
                    hint_b = pattern_direction_hint(name_b)
                    if hint_a != 0 and hint_b != 0 and hint_a != hint_b:
                        continue  # contradictory components - not real confluence
                    pairs.append((name_a, name_b))
    return pairs


def combo_direction_hint(name_a: str, name_b: str) -> int:
    """0 only if BOTH components are direction-agnostic (mined both ways,
    same as any other ambiguous pattern). Otherwise the nonzero
    component's direction wins - combo_pairs() already guarantees the two
    can't disagree if both are nonzero."""
    return pattern_direction_hint(name_a) or pattern_direction_hint(name_b)


# Built once at import time (deterministic, a few hundred entries) so
# both mining and live detection can look up "is this name a combo, and
# which two atomic patterns is it made of" without re-parsing the name
# string (pattern names may themselves contain underscores, so string
# splitting would be ambiguous - this dict sidesteps that entirely).
COMBO_PAIRS_BY_NAME: dict[str, tuple[str, str]] = {
    combo_name(a, b): (a, b) for a, b in combo_pairs()
}


def detect_combo_events(base_flags: pd.DataFrame) -> pd.DataFrame:
    """base_flags: the concatenated technical + fundamental + session +
    support_resistance + smc boolean DataFrame (patterns.detect_all +
    fundamental_patterns.detect_fundamental_events +
    session_patterns.detect_session_events +
    support_resistance.detect_support_resistance_events +
    smc_patterns.detect_smc_events), same index/contract used everywhere
    else in this codebase. Returns one boolean column per valid
    cross-family pair, True where BOTH components fired on the same
    candle."""
    out = {}
    for name, (name_a, name_b) in COMBO_PAIRS_BY_NAME.items():
        if name_a not in base_flags.columns or name_b not in base_flags.columns:
            continue  # e.g. no fundamentals loaded this run - that family's columns are absent, not all-False
        out[name] = base_flags[name_a] & base_flags[name_b]
    return pd.DataFrame(out, index=base_flags.index)
