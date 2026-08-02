"""
Support/Resistance: "did this candle react to a price level the market
has independently paid attention to before" - as its own mined pattern
family, given the EXACT SAME treatment as every other family in this
codebase (patterns.py, session_patterns.py, fundamental_patterns.py):
boolean Series per candle, fed through build_pattern_library.py's
identical 1:4 R:R simulation, 60% win-rate / sample-size hard gates,
out-of-sample check, combo-pairing (combo_patterns.py), and regime/news
conditioning. Nothing here is a hard-coded "S/R is good" assumption -
every pattern below still has to independently PROVE its edge on real
historical trade simulation to ever be usable live, same as a doji.

Three independent level sources, each look-ahead-safe by construction -
built only from candles STRICTLY BEFORE the one being tested, same
shift(1)-before-rolling discipline patterns.donchian_breakout_up/down
already uses in this codebase:

1. Swing highs/lows (`swing_levels`) - the classic N-bar fractal: a high
   that's the highest of the `2*SWING_LOOKBACK+1`-candle window centered
   on it (equivalently, a low that's the lowest of its window). A
   fractal is inherently not confirmed until SWING_LOOKBACK candles
   AFTER it forms - that's the definition, not a shortcut - and it's
   only usable starting ONE candle later still (final .shift(1)) so that
   the candle whose own high/low happened to be the one that FINALLY
   confirmed the fractal is never itself tested against that same level
   (it was one of the confirming candles - testing its own reaction to a
   level whose existence depended on its own price would be circular).
   Net effect: a real, honestly-paid lag of SWING_LOOKBACK+1 candles
   between a swing point forming and it becoming usable - documented
   here rather than hidden, same spirit as every other tradeoff called
   out elsewhere in this codebase (see combo_patterns.py's multiple-
   comparisons caveat, signal_journal.py's diagnostic-only framing).

2. Round numbers (`nearest_round_number`) - fixed psychological levels
   at a flat $ROUND_NUMBER_STEP grid ($2000, $2050, $2100, ... for
   gold). No look-ahead question at all: always a pure function of the
   CURRENT candle's own price, recomputed fresh every candle, nothing
   historical to leak.

3. Classic floor-trader daily pivots (`daily_pivots`) - PP/R1/S1 from
   the PRIOR completed calendar day's OHLC (the standard definition -
   not a rolling recalculation), held constant through the current day.
   Look-ahead-safe by construction: "yesterday" is always already fully
   closed by the time today's candles are forming, regardless of
   timeframe.

Deliberately excluded from this first pass (see README.md for the full
discussion): the pivot point (PP) itself as a rejection/bounce level -
R1/S1 have a clean, textbook directional meaning (first resistance /
first support), while PP is more of a rotational "fair value" level
that behaves differently depending on which side of it price is
already on; and "broken level becomes the opposite" (role-reversal)
retests, a genuinely different pattern from a clean rejection that
deserves its own honest mining rather than being folded in here.

A "reaction" pattern requires proximity to a level (within
PROXIMITY_ATR_MULT * ATR - regime-normalized, so "close" means the same
thing in calm and volatile conditions, same reasoning atr_expansion in
patterns.py already applies), a real approach (price was strictly on
the other side of the level one candle prior - so a level already
sitting comfortably beyond current price, or a level being crossed
mid-trend, doesn't falsely count as a "rejection"), AND a genuinely
ONE-SIDED rejection-shaped candle - not just "a big wick somewhere."
This reuses patterns.py's exact two-sided hammer/shooting_star
discipline (_upper_shadow/_lower_shadow/_body - "reused, not
reimplemented" like everywhere else), not a single one-sided check:
  1. The wick AGAINST the level must be large relative to the body
     (>= REJECTION_WICK_MULT x body, with NO fallback when body is ~0 -
     unlike a "must be small" comparison, "must be large" should become
     MORE easily satisfied as body shrinks toward 0, not impossible,
     since a near-zero body with one real wick is precisely the
     textbook gravestone/dragonfly doji rejection shape).
  2. The OPPOSITE wick must be small relative to the body-or-range
     (<= OPPOSITE_WICK_MAX_MULT x body, falling back to range when body
     is ~0 - the same fallback patterns.py's hammer/shooting_star
     already use for this direction). Without this second check, a
     long-legged doji with a big wick on BOTH sides - real indecision,
     not a clean rejection - would wrongly qualify on check 1 alone.
Proximity alone says nothing about direction (same reasoning
atr_expansion documents for why IT is combo-only) - these patterns are
standalone-eligible because the rejection SHAPE, not just the
proximity, is what supplies the directional claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from patterns import _body, _bearish, _bullish, _lower_shadow, _range, _upper_shadow
from risk_reward import atr as _atr

SWING_LOOKBACK = 5          # candles each side of a fractal swing point
ROUND_NUMBER_STEP = 50.0    # gold-specific psychological grid ($2000, $2050, ...)
PROXIMITY_ATR_MULT = 0.5    # "near" a level = within half an ATR
REJECTION_WICK_MULT = 1.5   # matches patterns._hammer_shape/_star_shape's own threshold
OPPOSITE_WICK_MAX_MULT = 0.3  # matches patterns._hammer_shape/_star_shape's own threshold


def confirmed_swing_points(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> tuple[pd.Series, pd.Series]:
    """(swing_high_confirmed, swing_low_confirmed): SPARSE series - a real
    price only on the candle where a fractal swing high/low FINALLY
    becomes confirmed (see module docstring for the lookback+1
    confirmation-lag reasoning), NaN everywhere else. This is the raw
    material swing_levels() itself ffills into a persistent "current
    level" - split out as its own public function so a caller that needs
    the SEQUENCE of confirmed swing points (not just the latest one) can
    reuse the identical fractal detection instead of reimplementing it.
    smc_patterns.py's market-structure functions (bos_bullish/bearish,
    choch_bullish/bearish) are exactly that caller: telling a
    continuation (Break of Structure) apart from a reversal (Change of
    Character) requires comparing the two MOST RECENT confirmed swing
    points against each other, not just knowing the latest one."""
    high, low = df["high"], df["low"]
    window = 2 * lookback + 1

    is_swing_high = high == high.rolling(window, center=True).max()
    is_swing_low = low == low.rolling(window, center=True).min()

    swing_high_confirmed = high.where(is_swing_high).shift(lookback)
    swing_low_confirmed = low.where(is_swing_low).shift(lookback)
    return swing_high_confirmed, swing_low_confirmed


def swing_levels(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> tuple[pd.Series, pd.Series]:
    """(swing_high_level, swing_low_level): the most recently CONFIRMED
    swing high/low price known as of the candle strictly BEFORE the one
    being evaluated. See module docstring for the confirmation-lag
    reasoning. Public (not underscored) because ml_system/features.py
    reuses this exact function for its own distance-to-level features -
    one implementation, no risk of the live/training feature code
    silently drifting from the pattern-mining definition of the same
    level."""
    swing_high_confirmed, swing_low_confirmed = confirmed_swing_points(df, lookback)
    swing_high_level = swing_high_confirmed.ffill().shift(1)
    swing_low_level = swing_low_confirmed.ffill().shift(1)
    return swing_high_level, swing_low_level


def nearest_round_number(df: pd.DataFrame, step: float = ROUND_NUMBER_STEP) -> pd.Series:
    """Nearest fixed psychological grid level to this candle's own
    midpoint. Pure function of the current candle - no history needed,
    no look-ahead question.

    Explicit round-half-UP, not pandas/numpy's default .round() - that
    default uses banker's rounding (round-half-to-EVEN), which at an
    EXACT halfway point (e.g. 2025.00, exactly between 2000 and 2050)
    silently favors whichever grid point happens to be an even multiple
    of `step` (2000, 2100, 2200, ...) over the odd ones (2050, 2150,
    ...) - an arbitrary, undocumented bias, not a neutral tie-break.
    Round-half-up is an equally arbitrary tie-break for a genuinely
    equidistant point, but at least a CONSISTENT, predictable one that
    doesn't quietly favor one family of levels over another."""
    mid = (df["high"] + df["low"]) / 2
    return np.floor(mid / step + 0.5) * step


def daily_pivots(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(pp, r1, s1) classic floor-trader pivots from the PRIOR completed
    calendar day, held constant through the current day. Assumes `df` is
    already sorted ascending by timestamp - the same assumption every
    other detector in this codebase already makes about its input,
    enforced once by build_pattern_library.py/signal_engine.py before
    any detector runs, not re-checked here."""
    ts = pd.to_datetime(df["timestamp"])
    day = ts.dt.floor("D")
    daily = df.groupby(day).agg(day_high=("high", "max"), day_low=("low", "min"), day_close=("close", "last"))
    daily["pp"] = (daily["day_high"] + daily["day_low"] + daily["day_close"]) / 3
    daily["r1"] = 2 * daily["pp"] - daily["day_low"]
    daily["s1"] = 2 * daily["pp"] - daily["day_high"]
    prior = daily.shift(1)  # prior day's OHLC -> today's pivot levels

    pp = day.map(prior["pp"])
    r1 = day.map(prior["r1"])
    s1 = day.map(prior["s1"])
    pp.index = r1.index = s1.index = df.index
    return pp, r1, s1


def _resistance_rejection(df: pd.DataFrame, level: pd.Series, atr_series: pd.Series) -> pd.Series:
    """Price approached `level` from below, poked into it, and got
    pushed back down without closing through it - a bearish signal.

    Wick geometry reuses patterns.py's exact two-sided hammer/
    shooting_star discipline, not a single one-sided check: the
    REJECTING wick must be large relative to the body (`_upper_shadow
    >= REJECTION_WICK_MULT * body`, with NO range-fallback when body is
    ~0 - unlike a "must be small" comparison, a "must be large" one
    should become MORE easily satisfied, not impossible, as body
    shrinks to 0, since a body of 0 with a real wick is the textbook
    gravestone/dragonfly doji rejection shape, not a shape to exclude)
    AND the OPPOSITE wick must be small relative to the range (`_lower_
    shadow <= 0.3 * body-or-range-fallback`, the exact same fallback
    patterns.hammer/shooting_star already use for this direction, since
    "must be small" SHOULD fall back to a real range-based threshold
    when body is 0). Without this second check, a long-legged doji with
    a big wick on BOTH sides - genuine indecision, not a clean rejection
    - would wrongly qualify on the first check alone."""
    buffer = PROXIMITY_ATR_MULT * atr_series
    body, rng = _body(df), _range(df)
    opposite_wick_threshold = OPPOSITE_WICK_MAX_MULT * body.replace(0, np.nan).fillna(rng)
    was_below = df["close"].shift(1) < level
    approached = df["high"] >= (level - buffer)
    held = df["close"] < level
    big_rejection_wick = _upper_shadow(df) >= REJECTION_WICK_MULT * body
    small_opposite_wick = _lower_shadow(df) <= opposite_wick_threshold
    return was_below & approached & held & big_rejection_wick & small_opposite_wick


def _support_bounce(df: pd.DataFrame, level: pd.Series, atr_series: pd.Series) -> pd.Series:
    """Mirror of _resistance_rejection: approached `level` from above,
    poked into it, and got pushed back up without closing through it -
    a bullish signal. See _resistance_rejection's docstring for the
    two-sided wick-geometry reasoning."""
    buffer = PROXIMITY_ATR_MULT * atr_series
    body, rng = _body(df), _range(df)
    opposite_wick_threshold = OPPOSITE_WICK_MAX_MULT * body.replace(0, np.nan).fillna(rng)
    was_above = df["close"].shift(1) > level
    approached = df["low"] <= (level + buffer)
    held = df["close"] > level
    big_rejection_wick = _lower_shadow(df) >= REJECTION_WICK_MULT * body
    small_opposite_wick = _upper_shadow(df) <= opposite_wick_threshold
    return was_above & approached & held & big_rejection_wick & small_opposite_wick


def sr_swing_high_rejection(df: pd.DataFrame) -> pd.Series:
    swing_high, _ = swing_levels(df)
    return _resistance_rejection(df, swing_high, _atr(df))


def sr_swing_low_bounce(df: pd.DataFrame) -> pd.Series:
    _, swing_low = swing_levels(df)
    return _support_bounce(df, swing_low, _atr(df))


def sr_round_number_rejection(df: pd.DataFrame) -> pd.Series:
    level = nearest_round_number(df)
    return _resistance_rejection(df, level, _atr(df))


def sr_round_number_bounce(df: pd.DataFrame) -> pd.Series:
    level = nearest_round_number(df)
    return _support_bounce(df, level, _atr(df))


def sr_pivot_r1_rejection(df: pd.DataFrame) -> pd.Series:
    _, r1, _ = daily_pivots(df)
    return _resistance_rejection(df, r1, _atr(df))


def sr_pivot_s1_bounce(df: pd.DataFrame) -> pd.Series:
    _, _, s1 = daily_pivots(df)
    return _support_bounce(df, s1, _atr(df))


# "sr_" prefix (not the bare shape names) so this family is
# distinguishable by name alone, the same way every other family in
# this codebase already is (session_*, fundamental_*, combo__*) -
# dashboard/server.py's _pattern_category() and its JS mirror
# (dashboard/static/index.html's patternCategory()) both key off this
# prefix to badge/filter these patterns as their own category instead
# of silently falling into "technical".
SUPPORT_RESISTANCE_PATTERNS = {
    "sr_swing_high_rejection": sr_swing_high_rejection,
    "sr_swing_low_bounce": sr_swing_low_bounce,
    "sr_round_number_rejection": sr_round_number_rejection,
    "sr_round_number_bounce": sr_round_number_bounce,
    "sr_pivot_r1_rejection": sr_pivot_r1_rejection,
    "sr_pivot_s1_bounce": sr_pivot_s1_bounce,
}

SUPPORT_RESISTANCE_PATTERN_NAMES = list(SUPPORT_RESISTANCE_PATTERNS)

# Every pattern here has an unambiguous textbook direction (a rejection
# is bearish, a bounce is bullish, by construction of the shape test
# itself) - merged into patterns.DIRECTION_HINT (see patterns.py) so
# there is exactly ONE direction lookup for every pattern name in the
# system, regardless of which module defines the pattern, matching how
# CANDLESTICK_PATTERNS and INDICATOR_PATTERNS already share that single
# dict today.
DIRECTION_HINT = {
    "sr_swing_high_rejection": -1,
    "sr_swing_low_bounce": +1,
    "sr_round_number_rejection": -1,
    "sr_round_number_bounce": +1,
    "sr_pivot_r1_rejection": -1,
    "sr_pivot_s1_bounce": +1,
}


def detect_support_resistance_events(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a boolean DataFrame, same index/contract as
    patterns.detect_all/session_patterns.detect_session_events/
    fundamental_patterns.detect_fundamental_events - one column per
    pattern in SUPPORT_RESISTANCE_PATTERNS."""
    out = {}
    for name, fn in SUPPORT_RESISTANCE_PATTERNS.items():
        out[name] = fn(df).fillna(False)
    return pd.DataFrame(out, index=df.index)
