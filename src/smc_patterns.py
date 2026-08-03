"""
Smart Money Concepts (SMC): given the EXACT SAME treatment as every other
pattern family in this codebase (patterns.py, session_patterns.py,
fundamental_patterns.py, support_resistance.py) - boolean Series per
candle, fed through build_pattern_library.py's identical 1:4 R:R
simulation, 60% win-rate / sample-size hard gates, out-of-sample check,
combo-pairing (combo_patterns.py), and regime/news conditioning. Nothing
here is a hard-coded "SMC works" assumption - every pattern below still
has to independently PROVE its edge on real historical trade simulation
to ever be usable live, same as a doji.

"SMC" covers a lot of loosely-defined retail-trading folklore (order
blocks, "breaker blocks", premium/discount zones, inducement, etc.) with
no single agreed-upon definition. This module implements the SMC concepts
that have an unambiguous, mechanically checkable definition and can be
built look-ahead-safe from data already in this codebase, rather than
inventing a bespoke indicator to cover every piece of folklore:

1. Liquidity sweep (`liquidity_sweep_low` / `liquidity_sweep_high`) - a
   candle that trades THROUGH a confirmed swing level (running the stops
   resting just beyond it) and then closes back on the origin side. This
   reuses support_resistance.swing_levels() - the SAME look-ahead-safe,
   confirmed swing high/low machinery already mined by
   support_resistance.py - but tests a STRICTER condition than that
   module's own sr_swing_*_rejection/bounce patterns: those only require
   price to *approach* the level (get within PROXIMITY_ATR_MULT * ATR)
   and hold; a liquidity sweep requires price to *break* the level
   outright before reclaiming it. Approaching-and-holding and
   breaking-and-reclaiming are genuinely different price behaviors (the
   latter implies stops on the far side were actually triggered), so
   this is a new, independent pattern family, not a duplicate of
   support_resistance.py's rejection/bounce shapes.

2. Fair Value Gap / imbalance (`fvg_bullish` / `fvg_bearish`) - the
   classic three-candle price gap: candle 1's high sits strictly below
   candle 3's low (bullish) or candle 1's low sits strictly above candle
   3's high (bearish), leaving an untouched price region between them
   that the impulsive middle candle jumped over. Confirmed the instant
   candle 3 closes - a pure function of the current candle and the one
   two bars back (`.shift(2)`), so there is no look-ahead question at
   all, the same reasoning nearest_round_number's "pure function of the
   current candle" gets in support_resistance.py.

3. Market structure - Break of Structure / Change of Character
   (`bos_bullish`/`bos_bearish`/`choch_bullish`/`choch_bearish`) - SMC's
   actual foundational concept, arguably more central than either of the
   above: does the SEQUENCE of confirmed swing points show higher-highs-
   and-higher-lows (bullish structure) or lower-highs-and-lower-lows
   (bearish structure), and when price closes beyond the most recent
   confirmed swing point, is that break WITH the prevailing structure
   (BOS - continuation) or AGAINST it (CHoCH - the first sign of
   reversal)? Built on `support_resistance.confirmed_swing_points()` -
   the same fractal detection swing_levels() itself is built from, only
   without the final ffill that collapses the SEQUENCE of swing points
   down to just "the current one" - see `_swing_sequence()` below.

4. Equal highs / equal lows liquidity pool sweep (`eq_high_sweep` /
   `eq_low_sweep`) - a stricter, higher-conviction variant of the plain
   liquidity sweep above: the two most recently confirmed swing points on
   one side sit within EQUAL_LEVEL_ATR_MULT of ATR of each other (the
   "equal highs/lows" retail stop-cluster SMC specifically calls out as a
   bigger liquidity pool than a single swing point), AND that pool gets
   swept-and-reclaimed the same way liquidity_sweep_low/high tests a
   single level.

5. Order blocks (`near_bullish_order_block` / `near_bearish_order_block`)
   and premium/discount zones (`range_position`, thresholded into
   `smc_discount_zone`/`smc_premium_zone`) - originally excluded here
   (see prior revisions) as needing a PERSISTENT, mutable zone rather
   than a single stateless per-candle boolean, unlike everything else in
   this module. Implemented now, with the ambiguity in both concepts'
   definitions (no single agreed-upon standard exists in retail ICT
   material for either) made explicit in each function's own docstring
   rather than silently picking one convention and presenting it as THE
   definition. Breaker blocks remain excluded - a breaker block is
   specifically a FAILED, invalidated order block re-purposed for the
   opposite direction, meaningfully more state to track correctly than
   order_block_zone's single active-zone-per-direction model handles.

All primitives are exposed standalone (so combo_patterns.py can pair any
one of them with a pattern from another family, e.g. a candlestick
reversal shape landing exactly on a liquidity sweep) AND the sweep/FVG
pair is additionally combined into one deterministic "sweep + gap"
trigger (`smc_bullish_sweep_fvg` / `smc_bearish_sweep_fvg`): the sweep
candle itself must ALSO be the third (gap-confirming) candle of the FVG -
a single, non-repainting condition, fully knowable as of the sweep
candle's own close, not "wait and see if a gap forms later."
"""
from __future__ import annotations

import pandas as pd

from risk_reward import atr as _atr
from support_resistance import confirmed_swing_points, swing_levels

FVG_LOOKBACK = 2  # classic 3-candle gap: compare current candle to the one 2 bars back
EQUAL_LEVEL_ATR_MULT = 0.15  # "equal" highs/lows tolerance, ATR-normalized like PROXIMITY_ATR_MULT elsewhere


def liquidity_sweep_low(df: pd.DataFrame) -> pd.Series:
    """Bullish: this candle's low traded BELOW the most recent confirmed
    swing low (clearing resting sell-stops below that support) but its
    close reclaimed back ABOVE the swing low - a stop-hunt that failed to
    hold, textbook SMC "liquidity grab" at support. Stricter than
    support_resistance.sr_swing_low_bounce, which only requires price to
    approach the level and hold, not actually break it first."""
    _, swing_low = swing_levels(df)
    swept = df["low"] < swing_low
    reclaimed = df["close"] > swing_low
    return swept & reclaimed


def liquidity_sweep_high(df: pd.DataFrame) -> pd.Series:
    """Mirror of liquidity_sweep_low: this candle's high traded ABOVE the
    most recent confirmed swing high (clearing resting buy-stops above
    that resistance) but its close reclaimed back BELOW the swing high -
    a failed stop-hunt at resistance, bearish."""
    swing_high, _ = swing_levels(df)
    swept = df["high"] > swing_high
    reclaimed = df["close"] < swing_high
    return swept & reclaimed


def fvg_bullish(df: pd.DataFrame) -> pd.Series:
    """Bullish Fair Value Gap: this candle's low sits strictly above the
    high of the candle FVG_LOOKBACK (2) bars ago, i.e. the middle candle's
    range jumped clean over a price band neither candle 1 nor candle 3
    touched. Confirmed exactly as of this candle's own close - candle 1's
    high and this candle's own low are both already-known values, nothing
    from the future."""
    prior_high = df["high"].shift(FVG_LOOKBACK)
    return df["low"] > prior_high


def fvg_bearish(df: pd.DataFrame) -> pd.Series:
    """Mirror of fvg_bullish: this candle's high sits strictly below the
    low of the candle FVG_LOOKBACK (2) bars ago - a bearish imbalance."""
    prior_low = df["low"].shift(FVG_LOOKBACK)
    return df["high"] < prior_low


def _swing_sequence(sparse: pd.Series) -> tuple[pd.Series, pd.Series]:
    """From a SPARSE confirmed-swing-point series (support_resistance.
    confirmed_swing_points()'s own output - a real value only on the
    candle where a swing point is confirmed, NaN elsewhere), returns
    (current_level, prior_level): the most recently confirmed swing
    point's price, and the ONE BEFORE THAT - both usable as of the
    candle strictly before the one being evaluated, the identical
    ffill-then-shift(1) discipline swing_levels() itself uses (so the
    candle that finally confirms a point is never evaluated against a
    level whose existence depended on its own price)."""
    current = sparse.ffill().shift(1)
    confirmed_only = sparse.dropna()
    prior_at_confirmation = confirmed_only.shift(1)  # the confirmed point BEFORE each confirmed point
    prior = prior_at_confirmation.reindex(sparse.index).ffill().shift(1)
    return current, prior


def _market_structure(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """(cur_high, prior_high, cur_low, prior_low): the two most recently
    confirmed swing highs and the two most recently confirmed swing lows,
    each already lagged for look-ahead-safe use. Shared by bos_*/choch_*/
    eq_*_sweep below so the fractal detection and swing-sequence pairing
    only happens once per call site, not once per pattern function."""
    swing_high_sparse, swing_low_sparse = confirmed_swing_points(df)
    cur_high, prior_high = _swing_sequence(swing_high_sparse)
    cur_low, prior_low = _swing_sequence(swing_low_sparse)
    return cur_high, prior_high, cur_low, prior_low


def _edge_triggered(signal: pd.Series) -> pd.Series:
    """True only on the FIRST candle a condition holds, not every candle
    it persists - cur_high/cur_low stay constant (ffilled) between swing
    confirmations, so an un-gated structure/break condition would stay
    True for potentially hundreds of candles after the actual break,
    flooding risk_reward.py's win-rate statistics with near-duplicate
    occurrences of the same single event. Same discipline patterns.
    trend_following_long/short already apply to their own multi-
    condition signal for the identical reason.

    `shift(1, fill_value=False)`, NOT `.shift(1).fillna(False)`: shifting
    a bool-dtype Series with no explicit fill value introduces a NaN at
    the first row, which silently upcasts the WHOLE series to `object`
    dtype (bool can't hold NaN) - and `~` on an object array of Python
    bools does INTEGER bitwise-NOT (`~False == -1`, `~True == -2`), not
    logical negation, since Python bool is an int subclass. Both -1 and
    -2 are truthy, so `signal & ~signal.shift(1).fillna(False)` silently
    degrades into just `signal` again - the edge-trigger never actually
    filters anything. Passing `fill_value=False` keeps the shifted
    result bool-dtype throughout, so `~` means what it looks like it
    means. Verified the hard way: an earlier version of this function
    used the broken form, and a hand-built continuously-True test signal
    came back with every single row still True instead of only the
    first."""
    return signal & ~signal.shift(1, fill_value=False)


def bos_bullish(df: pd.DataFrame) -> pd.Series:
    """Break of Structure, bullish: the market was already in a bullish
    structure (the two most recently confirmed swing points show a
    higher high AND a higher low) and this candle's close breaks above
    the most recent confirmed swing high - a break WITH the prevailing
    structure, i.e. trend continuation. Edge-triggered to the candle
    where the break first happens."""
    cur_high, prior_high, cur_low, prior_low = _market_structure(df)
    bullish_structure = (cur_high > prior_high) & (cur_low > prior_low)
    return _edge_triggered(bullish_structure & (df["close"] > cur_high))


def bos_bearish(df: pd.DataFrame) -> pd.Series:
    """Mirror of bos_bullish: bearish structure (lower high AND lower
    low) and close breaks below the most recent confirmed swing low -
    continuation of a downtrend."""
    cur_high, prior_high, cur_low, prior_low = _market_structure(df)
    bearish_structure = (cur_high < prior_high) & (cur_low < prior_low)
    return _edge_triggered(bearish_structure & (df["close"] < cur_low))


def choch_bullish(df: pd.DataFrame) -> pd.Series:
    """Change of Character, bullish: the market was in a BEARISH
    structure (lower high AND lower low) and this candle's close breaks
    ABOVE the most recent confirmed swing high - a break AGAINST the
    prevailing structure, the first mechanical sign the downtrend may be
    reversing. Same swing points as bos_bearish's bearish_structure test,
    but the break direction is the OPPOSITE of what bos_bearish requires
    - the two are mutually exclusive by construction, never double-firing
    on the same candle."""
    cur_high, prior_high, cur_low, prior_low = _market_structure(df)
    bearish_structure = (cur_high < prior_high) & (cur_low < prior_low)
    return _edge_triggered(bearish_structure & (df["close"] > cur_high))


def choch_bearish(df: pd.DataFrame) -> pd.Series:
    """Mirror of choch_bullish: bullish structure but close breaks BELOW
    the most recent confirmed swing low - first mechanical sign an
    uptrend may be reversing."""
    cur_high, prior_high, cur_low, prior_low = _market_structure(df)
    bullish_structure = (cur_high > prior_high) & (cur_low > prior_low)
    return _edge_triggered(bullish_structure & (df["close"] < cur_low))


def eq_high_sweep(df: pd.DataFrame) -> pd.Series:
    """Bearish: the two most recently confirmed swing highs sit within
    EQUAL_LEVEL_ATR_MULT of ATR of each other - an "equal highs"
    liquidity pool, a bigger cluster of resting buy-stops than a single
    swing point - and this candle sweeps through the more recent of the
    two (clearing BOTH resting-stop clusters at once, since they sit at
    essentially the same price) before closing back below it. Stricter
    and rarer than liquidity_sweep_high, which fires on a single swing
    point with no "is this actually a pool" check at all."""
    cur_high, prior_high, _, _ = _market_structure(df)
    a = _atr(df)
    is_equal = (cur_high - prior_high).abs() <= EQUAL_LEVEL_ATR_MULT * a
    swept = df["high"] > cur_high
    reclaimed = df["close"] < cur_high
    return is_equal & swept & reclaimed


def eq_low_sweep(df: pd.DataFrame) -> pd.Series:
    """Mirror of eq_high_sweep: an "equal lows" liquidity pool (the two
    most recently confirmed swing lows within EQUAL_LEVEL_ATR_MULT of ATR
    of each other) swept and reclaimed - bullish."""
    _, _, cur_low, prior_low = _market_structure(df)
    a = _atr(df)
    is_equal = (cur_low - prior_low).abs() <= EQUAL_LEVEL_ATR_MULT * a
    swept = df["low"] < cur_low
    reclaimed = df["close"] > cur_low
    return is_equal & swept & reclaimed


def order_block_zone(df: pd.DataFrame, bullish: bool) -> tuple[pd.Series, pd.Series]:
    """(zone_low, zone_high): the ACTIVE order block's price range, as of
    each candle - persistent/mutable, unlike every other function in this
    module, which is exactly why this module's own docstring originally
    excluded order blocks. Implemented now at explicit user request
    ("no constraints... SMC ICT whatever you want") with ONE reasonable,
    fully mechanical definition among several real ones retail ICT
    material uses (no single agreed-upon standard exists, same ambiguity
    this module's docstring already flags for premium/discount below):

    A bullish order block is the LAST opposite-colored (bearish) candle
    immediately before a Break of Structure (bos_bullish) - the
    institutional "footprint" candle right before the impulsive move that
    broke structure. Its [low, high] range becomes a zone traders watch
    for price to return to (mitigation) as a lower-risk continuation
    entry. The zone is set at each BOS event (from whichever bearish
    candle most recently preceded it) and PERSISTS - unlike this
    codebase's other stateless per-candle booleans - until superseded by
    the next BOS's own new order block. Bearish mirror: last opposite
    (bullish) candle before a bos_bearish.

    Look-ahead safety: the "last opposite candle before this BOS" lookup
    is `.ffill().shift(1)` - the candle used is always strictly BEFORE
    the evaluation row, and the zone itself only updates on rows where
    the BOS condition is already true (itself already edge-triggered,
    look-ahead-safe by construction - see bos_bullish/bos_bearish)."""
    close, open_, high, low = df["close"], df["open"], df["high"], df["low"]
    bos = bos_bullish(df) if bullish else bos_bearish(df)
    opposite_candle = (close < open_) if bullish else (close > open_)

    last_opp_high = high.where(opposite_candle).ffill().shift(1)
    last_opp_low = low.where(opposite_candle).ffill().shift(1)

    zone_high = last_opp_high.where(bos).ffill()
    zone_low = last_opp_low.where(bos).ffill()
    return zone_low, zone_high


def near_bullish_order_block(df: pd.DataFrame) -> pd.Series:
    """Price has returned into the most recent active bullish order
    block's [low, high] range (see order_block_zone) - the classic ICT
    "wait for mitigation" continuation entry, rather than chasing the
    original breakout candle itself."""
    zone_low, zone_high = order_block_zone(df, bullish=True)
    close = df["close"]
    return (close >= zone_low) & (close <= zone_high) & zone_low.notna()


def near_bearish_order_block(df: pd.DataFrame) -> pd.Series:
    """Mirror of near_bullish_order_block for bearish order blocks."""
    zone_low, zone_high = order_block_zone(df, bullish=False)
    close = df["close"]
    return (close >= zone_low) & (close <= zone_high) & zone_low.notna()


def range_position(df: pd.DataFrame, window: int) -> pd.Series:
    """0.0 = price sits at the trailing `window`-candle range's LOW,
    1.0 = at its HIGH - ICT's premium/discount framing: below the
    range's midpoint (0.5) is a "discount" (favor longs, price is
    "cheap" relative to its recent range), above is a "premium" (favor
    shorts). Like order blocks above, "premium/discount" has no single
    universally-agreed range definition in retail ICT material (some use
    the current swing leg, some a fixed lookback, some a session's own
    range) - this uses a fixed trailing lookback window for the same
    reason discovery_primitives.py's own docstring gives for bounded,
    parameterized primitives generally: a small fixed grid the search can
    evaluate, not an unbounded/ambiguous continuum. Range of exactly zero
    (flat market) divides by zero -> NaN, correctly "undefined," not a
    fabricated 0 or 1."""
    high = df["high"].rolling(window).max()
    low = df["low"].rolling(window).min()
    return (df["close"] - low) / (high - low)


def smc_bullish_sweep_fvg(df: pd.DataFrame) -> pd.Series:
    """The combined "smart money" setup: the SAME candle that sweeps a
    swing low (liquidity_sweep_low) is ALSO the third, gap-confirming
    candle of a bullish FVG (fvg_bullish) - the reversal impulse that
    reclaims the swept level is forceful enough to leave an imbalance
    behind it. One deterministic, non-repainting trigger, fully known as
    of this candle's close - not "sweep now, wait for a gap to maybe form
    later"."""
    return liquidity_sweep_low(df) & fvg_bullish(df)


def smc_bearish_sweep_fvg(df: pd.DataFrame) -> pd.Series:
    """Mirror of smc_bullish_sweep_fvg: the same candle sweeps a swing
    high (liquidity_sweep_high) and confirms a bearish FVG (fvg_bearish)."""
    return liquidity_sweep_high(df) & fvg_bearish(df)


# "smc_" prefix (not the bare shape names) so this family is
# distinguishable by name alone, the same way every other family in this
# codebase already is (session_*, fundamental_*, sr_*, combo__*) -
# dashboard/server.py's _pattern_category() and its JS mirror
# (dashboard/static/index.html's patternCategory()) both key off this
# prefix to badge/filter these patterns as their own category instead of
# silently falling into "technical".
SMC_PATTERNS = {
    "smc_liquidity_sweep_low": liquidity_sweep_low,
    "smc_liquidity_sweep_high": liquidity_sweep_high,
    "smc_fvg_bullish": fvg_bullish,
    "smc_fvg_bearish": fvg_bearish,
    "smc_bullish_sweep_fvg": smc_bullish_sweep_fvg,
    "smc_bearish_sweep_fvg": smc_bearish_sweep_fvg,
    "smc_bos_bullish": bos_bullish,
    "smc_bos_bearish": bos_bearish,
    "smc_choch_bullish": choch_bullish,
    "smc_choch_bearish": choch_bearish,
    "smc_eq_high_sweep": eq_high_sweep,
    "smc_eq_low_sweep": eq_low_sweep,
    "smc_near_bullish_order_block": near_bullish_order_block,
    "smc_near_bearish_order_block": near_bearish_order_block,
    "smc_discount_zone": lambda df: range_position(df, 50) < 0.3,
    "smc_premium_zone": lambda df: range_position(df, 50) > 0.7,
}

SMC_PATTERN_NAMES = list(SMC_PATTERNS)

# Every pattern here has an unambiguous textbook direction (a low-side
# sweep, bullish gap, bullish break/reversal is bullish; the mirror image
# is bearish, by construction of the shape test itself) - looked up as a
# fallback by patterns.pattern_direction_hint so this stays the ONE
# direction lookup for every pattern name in the system regardless of
# which module defines it, same as support_resistance.DIRECTION_HINT
# already is.
DIRECTION_HINT = {
    "smc_liquidity_sweep_low": +1,
    "smc_liquidity_sweep_high": -1,
    "smc_fvg_bullish": +1,
    "smc_fvg_bearish": -1,
    "smc_bullish_sweep_fvg": +1,
    "smc_bearish_sweep_fvg": -1,
    "smc_bos_bullish": +1,
    "smc_bos_bearish": -1,
    "smc_choch_bullish": +1,
    "smc_choch_bearish": -1,
    "smc_eq_high_sweep": -1,
    "smc_eq_low_sweep": +1,
}


def detect_smc_events(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a boolean DataFrame, same index/contract as
    patterns.detect_all/session_patterns.detect_session_events/
    fundamental_patterns.detect_fundamental_events/
    support_resistance.detect_support_resistance_events - one column per
    pattern in SMC_PATTERNS."""
    out = {}
    for name, fn in SMC_PATTERNS.items():
        out[name] = fn(df).fillna(False)
    return pd.DataFrame(out, index=df.index)
