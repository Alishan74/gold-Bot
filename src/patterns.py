"""
Pattern detectors: candlestick shapes + simple indicator crosses.

Everything here is vectorized pandas over a candle DataFrame with columns
[timestamp, open, high, low, close, volume]. Each detector returns a
boolean Series aligned to the input index - True on the candle where the
pattern completes.

Deliberately no TA-Lib dependency (painful C-extension install); these are
the standard, well-documented rule definitions reimplemented directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk_reward import atr as _atr


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _range(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]).replace(0, np.nan)


def _body_high(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].max(axis=1)


def _body_low(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1)


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def _bullish(df: pd.DataFrame) -> pd.Series:
    return df["close"] > df["open"]


def _bearish(df: pd.DataFrame) -> pd.Series:
    return df["close"] < df["open"]


def _body_ratio(df: pd.DataFrame) -> pd.Series:
    """Real body as a fraction of the candle's full high-low range. Low
    values mean big wicks relative to the body - i.e. the market pushed
    price further and then rejected it. Used below to stop O/C-only shape
    tests (engulfing, piercing line, ...) from firing on a candle whose
    wicks contradict the story its open/close would otherwise tell."""
    return (_body(df) / _range(df)).fillna(0)


def _prior_trend_down(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """True if the close going into this candle sits below the average of
    the `lookback` closes before that - a simple, look-ahead-free downtrend
    proxy (only uses candles strictly before the pattern candle)."""
    prior_close = df["close"].shift(1)
    return prior_close < prior_close.rolling(lookback).mean()


def _prior_trend_up(df: pd.DataFrame, lookback: int = 10) -> pd.Series:
    prior_close = df["close"].shift(1)
    return prior_close > prior_close.rolling(lookback).mean()


# ---- single-candle patterns ------------------------------------------------

def doji(df: pd.DataFrame, body_ratio: float = 0.1) -> pd.Series:
    return _body(df) <= body_ratio * _range(df)


def _hammer_shape(df: pd.DataFrame) -> pd.Series:
    body, rng = _body(df), _range(df)
    return (
        (_lower_shadow(df) >= 2 * body)
        & (_upper_shadow(df) <= 0.3 * body.replace(0, np.nan).fillna(rng))
        & (body <= 0.4 * rng)
    )


def _star_shape(df: pd.DataFrame) -> pd.Series:
    body, rng = _body(df), _range(df)
    return (
        (_upper_shadow(df) >= 2 * body)
        & (_lower_shadow(df) <= 0.3 * body.replace(0, np.nan).fillna(rng))
        & (body <= 0.4 * rng)
    )


def hammer(df: pd.DataFrame) -> pd.Series:
    # Bullish reversal: the long-lower-wick shape only means "buyers
    # defended a dip" if there was a dip to defend - i.e. a prior
    # downtrend. The identical shape after an UPTREND is a hanging_man
    # (bearish) instead, see below.
    return _hammer_shape(df) & _prior_trend_down(df)


def shooting_star(df: pd.DataFrame) -> pd.Series:
    # Bearish reversal: long-upper-wick rejection only means something
    # after a prior UPTREND. The identical shape after a downtrend is an
    # inverted_hammer (bullish) instead, see below.
    return _star_shape(df) & _prior_trend_up(df)


def hanging_man(df: pd.DataFrame) -> pd.Series:
    # Same raw shape as hammer, but only counted after a prior UPTREND -
    # that context is what makes the long lower wick a bearish warning
    # (a top-heavy market suddenly finding aggressive selling intraday)
    # instead of a bullish one. Without this trend gate, hammer and
    # hanging_man would fire on the literal same candles and get mined
    # with opposite hard-coded directions for no real reason.
    return _hammer_shape(df) & _prior_trend_up(df)


def inverted_hammer(df: pd.DataFrame) -> pd.Series:
    # Same raw shape as shooting_star, but only after a prior DOWNTREND -
    # see hanging_man above for why the trend context (not just the
    # shape) is what determines which of the two this actually is.
    return _star_shape(df) & _prior_trend_down(df)


def marubozu_bullish(df: pd.DataFrame) -> pd.Series:
    # A candle that's almost ALL body - open near the low, close near the
    # high, virtually no wicks either side. Reads as one-sided conviction
    # for the full period, not just a strong close after a fight.
    return _bullish(df) & (_body_ratio(df) >= 0.9)


def marubozu_bearish(df: pd.DataFrame) -> pd.Series:
    return _bearish(df) & (_body_ratio(df) >= 0.9)


# ---- two-candle patterns ----------------------------------------------------

def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    # Body test (open/close) defines the classic shape, but a candle with
    # a dominant upper wick means the market rejected the highs even
    # though the close was strong - not real conviction. body_ratio (uses
    # high/low) filters that out.
    return (
        _bearish(prev)
        & _bullish(df)
        & (df["open"] <= prev["close"])
        & (df["close"] >= prev["open"])
        & (_body_ratio(df) >= 0.6)
    )


def bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    return (
        _bullish(prev)
        & _bearish(df)
        & (df["open"] >= prev["close"])
        & (df["close"] <= prev["open"])
        & (_body_ratio(df) >= 0.6)
    )


def piercing_line(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    prev_mid = (prev["open"] + prev["close"]) / 2
    return (
        _bearish(prev)
        & _bullish(df)
        & (df["open"] < prev["close"])
        & (df["close"] > prev_mid)
        & (df["close"] < prev["open"])
        & (_body_ratio(df) >= 0.5)
    )


def dark_cloud_cover(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    prev_mid = (prev["open"] + prev["close"]) / 2
    return (
        _bullish(prev)
        & _bearish(df)
        & (df["open"] > prev["close"])
        & (df["close"] < prev_mid)
        & (df["close"] > prev["open"])
        & (_body_ratio(df) >= 0.5)
    )


def bullish_harami(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    # Opposite of engulfing: the SMALL candle's whole body (open AND
    # close, using _body_high/_body_low - not just its close) sits
    # entirely inside the prior candle's body, after a real (not tiny)
    # prior move - the market suddenly stalling inside yesterday's range.
    return (
        _bearish(prev)
        & _bullish(df)
        & (_body_high(df) <= _body_high(prev))
        & (_body_low(df) >= _body_low(prev))
        & (_body(df) <= 0.6 * _body(prev))
        & (_body_ratio(prev) >= 0.4)
    )


def bearish_harami(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    return (
        _bullish(prev)
        & _bearish(df)
        & (_body_high(df) <= _body_high(prev))
        & (_body_low(df) >= _body_low(prev))
        & (_body(df) <= 0.6 * _body(prev))
        & (_body_ratio(prev) >= 0.4)
    )


def tweezer_bottom(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    rng = pd.concat([_range(df), _range(prev)], axis=1).max(axis=1)
    # Two candles rejecting the SAME low (within a tolerance scaled to
    # the bigger of the two candles' own ranges, not a fixed price
    # distance - so it means the same thing on 1min and on 1d) - a
    # visible double-rejection of that price, not just "closed higher."
    matching_lows = (df["low"] - prev["low"]).abs() <= 0.15 * rng
    return _bearish(prev) & _bullish(df) & matching_lows & (_body_ratio(df) >= 0.3)


def tweezer_top(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    rng = pd.concat([_range(df), _range(prev)], axis=1).max(axis=1)
    matching_highs = (df["high"] - prev["high"]).abs() <= 0.15 * rng
    return _bullish(prev) & _bearish(df) & matching_highs & (_body_ratio(df) >= 0.3)


def outside_bar(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    return (df["high"] > prev["high"]) & (df["low"] < prev["low"])


def inside_bar(df: pd.DataFrame) -> pd.Series:
    prev = df.shift(1)
    return (df["high"] < prev["high"]) & (df["low"] > prev["low"])


# ---- three-candle patterns --------------------------------------------------

def morning_star(df: pd.DataFrame) -> pd.Series:
    c1, c2, c3 = df.shift(2), df.shift(1), df
    small_body = _body(c2) <= 0.5 * _body(c1)
    # A real "star" is a small candle sitting apart from the trend - that
    # means small TOTAL RANGE (body + wicks), not just a small body inside
    # an otherwise huge, wick-heavy candle. _range uses high/low.
    small_range = _range(c2) <= 0.6 * _range(c1)
    return (
        _bearish(c1)
        & small_body
        & small_range
        & _bullish(c3)
        & (c3["close"] >= (c1["open"] + c1["close"]) / 2)
        & (_body_ratio(c3) >= 0.5)
    )


def evening_star(df: pd.DataFrame) -> pd.Series:
    c1, c2, c3 = df.shift(2), df.shift(1), df
    small_body = _body(c2) <= 0.5 * _body(c1)
    small_range = _range(c2) <= 0.6 * _range(c1)
    return (
        _bullish(c1)
        & small_body
        & small_range
        & _bearish(c3)
        & (c3["close"] <= (c1["open"] + c1["close"]) / 2)
        & (_body_ratio(c3) >= 0.5)
    )


def three_white_soldiers(df: pd.DataFrame) -> pd.Series:
    c1, c2, c3 = df.shift(2), df.shift(1), df
    # Each candle should close near its own high (small upper wick) to
    # show sustained buying, not just three up-closes propped up by noise.
    strong_close = lambda c: _upper_shadow(c) <= 0.3 * _body(c)
    return (
        _bullish(c1) & _bullish(c2) & _bullish(c3)
        & (c2["open"] > c1["open"]) & (c2["close"] > c1["close"])
        & (c3["open"] > c2["open"]) & (c3["close"] > c2["close"])
        & strong_close(c1) & strong_close(c2) & strong_close(c3)
    )


def three_black_crows(df: pd.DataFrame) -> pd.Series:
    c1, c2, c3 = df.shift(2), df.shift(1), df
    strong_close = lambda c: _lower_shadow(c) <= 0.3 * _body(c)
    return (
        _bearish(c1) & _bearish(c2) & _bearish(c3)
        & (c2["open"] < c1["open"]) & (c2["close"] < c1["close"])
        & (c3["open"] < c2["open"]) & (c3["close"] < c2["close"])
        & strong_close(c1) & strong_close(c2) & strong_close(c3)
    )


# ---- indicator-based signals -------------------------------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def golden_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    f, s = sma(df["close"], fast), sma(df["close"], slow)
    return (f > s) & (f.shift(1) <= s.shift(1))


def death_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    f, s = sma(df["close"], fast), sma(df["close"], slow)
    return (f < s) & (f.shift(1) >= s.shift(1))


def rsi_oversold_cross(df: pd.DataFrame, window: int = 14, level: float = 30) -> pd.Series:
    r = rsi(df["close"], window)
    return (r > level) & (r.shift(1) <= level)


def rsi_overbought_cross(df: pd.DataFrame, window: int = 14, level: float = 70) -> pd.Series:
    r = rsi(df["close"], window)
    return (r < level) & (r.shift(1) >= level)


def macd_bullish_cross(df: pd.DataFrame) -> pd.Series:
    macd_line, signal_line = macd(df["close"])
    return (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))


def macd_bearish_cross(df: pd.DataFrame) -> pd.Series:
    macd_line, signal_line = macd(df["close"])
    return (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))


DONCHIAN_WINDOW = 20


def donchian_breakout_up(df: pd.DataFrame, window: int = DONCHIAN_WINDOW) -> pd.Series:
    # Close breaks above the highest HIGH of the prior `window` candles
    # (shift(1) before the rolling max, so the channel is built only from
    # candles strictly before this one - no look-ahead). A real high/low
    # channel breakout, not a close-only moving-average cross.
    prior_high = df["high"].shift(1).rolling(window).max()
    return df["close"] > prior_high


def donchian_breakout_down(df: pd.DataFrame, window: int = DONCHIAN_WINDOW) -> pd.Series:
    prior_low = df["low"].shift(1).rolling(window).min()
    return df["close"] < prior_low


def atr_expansion(df: pd.DataFrame, window: int = 14, lookback: int = 50, multiple: float = 1.3) -> pd.Series:
    """Volatility regime shift: current ATR meaningfully above its own
    recent average. Direction-ambiguous on its own (a volatility spike
    isn't bullish or bearish by itself) - its real value is as a COMBO
    component (see combo_patterns.py): "bullish_engulfing during an ATR
    expansion" is a materially different, more selective claim than
    "bullish_engulfing" alone."""
    a = _atr(df, window)
    baseline = a.shift(1).rolling(lookback).mean()
    return a > multiple * baseline


# ---- additional indicators: each genuinely non-redundant with what's
# already above, not a second formula wearing a different name -------------

BOLLINGER_WINDOW = 20
BOLLINGER_STD_MULT = 2.0


def bollinger_bands(series: pd.Series, window: int = BOLLINGER_WINDOW,
                     std_mult: float = BOLLINGER_STD_MULT) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(upper, mid, lower): mid is the plain SMA(20) already used
    elsewhere in this file; upper/lower are mid +/- std_mult standard
    deviations of the SAME window's closes - a volatility-NORMALIZED
    band, unlike donchian_breakout_up/down's fixed high/low channel."""
    mid = sma(series, window)
    std = series.rolling(window).std()
    return mid + std_mult * std, mid, mid - std_mult * std


def bb_upper_breakout(df: pd.DataFrame) -> pd.Series:
    """Close breaks above its own rolling Bollinger upper band. Distinct
    from donchian_breakout_up: Donchian reacts to the literal extreme
    (highest HIGH of the last N candles), Bollinger to the DISTRIBUTION
    of recent CLOSES (mean + 2 std) - the two can and often do disagree
    on when a "breakout" actually happened. Same directional convention
    as donchian_breakout_up (+1): a push outside the recent statistical
    range, not a mean-reversion fade back toward it."""
    upper, _, _ = bollinger_bands(df["close"])
    return (df["close"] > upper) & (df["close"].shift(1) <= upper.shift(1))


def bb_lower_breakout(df: pd.DataFrame) -> pd.Series:
    """Mirror of bb_upper_breakout: close breaks below the lower band."""
    _, _, lower = bollinger_bands(df["close"])
    return (df["close"] < lower) & (df["close"].shift(1) >= lower.shift(1))


STOCH_WINDOW = 14
STOCH_SMOOTH = 3


def stochastic(df: pd.DataFrame, window: int = STOCH_WINDOW,
               smooth: int = STOCH_SMOOTH) -> tuple[pd.Series, pd.Series]:
    """(%K, %D): bounded 0-100 momentum oscillator built from the
    HIGH/LOW range over `window` candles, not from closes alone - the
    reason it's picked as a SECOND momentum-family indicator alongside
    RSI rather than skipped as redundant: RSI is a close-only magnitude
    measure of recent gains vs losses, Stochastic is a positional measure
    of where the current close sits within the recent high/low range -
    different inputs, same "genuinely disjoint, not two names for the
    same signal" standard _obv()'s own docstring sets for OBV vs RSI."""
    low_n = df["low"].rolling(window).min()
    high_n = df["high"].rolling(window).max()
    percent_k = 100 * (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)
    percent_d = percent_k.rolling(smooth).mean()
    return percent_k, percent_d


def stoch_oversold_cross(df: pd.DataFrame, level: float = 20) -> pd.Series:
    k, _ = stochastic(df)
    return (k > level) & (k.shift(1) <= level)


def stoch_overbought_cross(df: pd.DataFrame, level: float = 80) -> pd.Series:
    k, _ = stochastic(df)
    return (k < level) & (k.shift(1) >= level)


def adx_trending(df: pd.DataFrame, window: int = 14, threshold: float = 25.0) -> pd.Series:
    """Trend-STRENGTH filter (not direction) - reuses regime.adx(), the
    exact same ADX implementation regime.py already computes to classify
    TRENDING/RANGING for the by-regime conditioning every atomic pattern
    gets in build_pattern_library.py (reused, not reimplemented - regime.
    py imports only risk_reward.py, so importing it here creates no
    cycle). Direction-ambiguous on its own (a strong trend isn't
    inherently bullish or bearish) - the same role atr_expansion plays
    for volatility: real value as a COMBO component, e.g. "bullish
    engulfing during a strong (ADX > 25) trend" is a materially
    different, more selective claim than "bullish engulfing" alone."""
    from regime import adx
    return adx(df, window) > threshold


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored Volume-Weighted Average Price: cumulative
    (volume x typical price) / cumulative volume, RESETTING at each
    calendar day boundary (the same day-grouping support_resistance.
    daily_pivots() already uses for its own daily-anchored levels) - the
    standard institutional-benchmark definition, not a rolling window.
    Fully causal: within a day, cumsum only ever looks backward, so
    there's no look-ahead question. Note the reset itself is a genuinely
    noisy moment (each day's first candle's vwap equals that SAME
    candle's own typical price, so a "cross" detected right at the open
    is comparing a candle to a level derived only from itself) - a real
    characteristic of session VWAP, not a bug specific to this
    implementation."""
    ts = pd.to_datetime(df["timestamp"])
    day = ts.dt.floor("D")
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_pv = (typical * df["volume"]).groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def vwap_bullish_cross(df: pd.DataFrame) -> pd.Series:
    v = vwap(df)
    return (df["close"] > v) & (df["close"].shift(1) <= v.shift(1))


def vwap_bearish_cross(df: pd.DataFrame) -> pd.Series:
    v = vwap(df)
    return (df["close"] < v) & (df["close"].shift(1) >= v.shift(1))


# ---- trend-following strategy (1 trend indicator + 2 non-correlated
# confirmations + 1 volatility filter) -----------------------------------

TREND_SMA_FAST = 50
TREND_SMA_SLOW = 200
TREND_RSI_WINDOW = 14
TREND_OBV_SLOPE_WINDOW = 20
TREND_VOL_LOOKBACK = 50
TREND_VOL_MULTIPLE = 1.1  # looser than atr_expansion()'s 1.3 - this is a
                          # persistent "not dead/choppy" FILTER meant to
                          # pass often, not a rare standalone signal


def _obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (Granville, 1963): cumulative volume, signed by
    close-to-close direction. Deliberately the SECOND confirmation
    alongside RSI below, not a second price-momentum indicator wearing a
    different name - RSI is computed purely from the SIZE of price
    changes and never looks at volume; OBV is computed purely from the
    DIRECTION of price changes and the volume behind them, and never
    looks at how far price actually moved. Two indicators built from
    genuinely disjoint inputs are what "non-correlated confirmation"
    has to mean in practice, not just two different formulas applied to
    the same close-price series (RSI and MACD, for instance, are both
    close-price-only and move together far more often than a "second,
    independent" check should)."""
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def _trend_following_conditions(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Shared by both directions below - returns (trend_up, momentum_up,
    participation_up); each direction's entry function combines these
    (or their negations) with its own volatility-filter check. Every
    piece is a plain deterministic function of OHLCV history up to and
    including the current candle - no discretionary judgment, no
    parameter fit to this specific dataset (all four thresholds are the
    textbook defaults for their respective indicators: RSI 50 midline,
    SMA 50/200 "golden/death cross" pair, standard OBV, ATR's own
    existing atr_expansion() convention)."""
    close = df["close"]
    trend_up = sma(close, TREND_SMA_FAST) > sma(close, TREND_SMA_SLOW)
    momentum_up = rsi(close, TREND_RSI_WINDOW) > 50
    obv = _obv(df)
    participation_up = (obv - obv.shift(TREND_OBV_SLOPE_WINDOW)) > 0
    return trend_up, momentum_up, participation_up


def trend_following_long(df: pd.DataFrame) -> pd.Series:
    """Trend-following long entry, 15min-timeframe design (mined/gated
    like every other pattern here - see module note in the
    INDICATOR_PATTERNS registration below for why it isn't hard-
    restricted to only that one timeframe in code).

    Fires on the candle where ALL FOUR conditions first become true
    together (edge-triggered against the immediately prior candle, same
    "this is where the pattern completes" convention golden_cross/
    death_cross already use) - not on every candle the trend persists,
    which would flood risk_reward.py's win-rate statistics with
    hundreds of near-duplicate, highly autocorrelated "occurrences" of
    the same single trend move instead of one real entry per move.

      1. TREND DIRECTION (one indicator): SMA(50) > SMA(200) - price's
         own longer-run trend state, not a momentary crossover event.
      2 & 3. CONFIRMATION (two non-correlated indicators): RSI(14) > 50
         (price-momentum family) AND rising OBV over the last 20 candles
         (volume/participation family) - see _obv()'s docstring for why
         this pairing, not RSI+MACD, is genuine non-correlation rather
         than two names for the same underlying signal.
      4. VOLATILITY FILTER: ATR(14) above 1.1x its own 50-candle trailing
         average (atr_expansion()'s own convention, reused directly) -
         keeps this out of dead/choppy stretches where even a genuine
         trend-following setup has no real room to run before its stop.

    Stop-loss and take-profit are NOT computed here - like every other
    pattern in this file, this function only says WHEN to enter;
    risk_reward.py's shared simulate_trades()/resolve_trade() then
    prices every trade at STOP_ATR_MULTIPLE (1.5x) times ATR AT THIS
    SAME CANDLE for the stop, and RR_RATIO (4x that same risk) for the
    target - both anchored to this timeframe's own ATR, never a
    higher-timeframe level, so there is nothing here that can repaint."""
    trend_up, momentum_up, participation_up = _trend_following_conditions(df)
    vol_ok = atr_expansion(df, lookback=TREND_VOL_LOOKBACK, multiple=TREND_VOL_MULTIPLE)
    signal = trend_up & momentum_up & participation_up & vol_ok
    return signal & ~signal.shift(1, fill_value=False)


def trend_following_short(df: pd.DataFrame) -> pd.Series:
    """Mirror image of trend_following_long() - see its docstring for
    the full rationale. SMA(50) < SMA(200) for trend direction, RSI < 50
    and falling OBV for confirmation, the same volatility filter (a
    volatility contraction is directionless, so the filter itself is
    identical for both sides)."""
    trend_up, momentum_up, participation_up = _trend_following_conditions(df)
    vol_ok = atr_expansion(df, lookback=TREND_VOL_LOOKBACK, multiple=TREND_VOL_MULTIPLE)
    signal = (~trend_up) & (~momentum_up) & (~participation_up) & vol_ok
    return signal & ~signal.shift(1, fill_value=False)


CANDLESTICK_PATTERNS = {
    "doji": doji,
    "hammer": hammer,
    "shooting_star": shooting_star,
    "hanging_man": hanging_man,
    "inverted_hammer": inverted_hammer,
    "marubozu_bullish": marubozu_bullish,
    "marubozu_bearish": marubozu_bearish,
    "bullish_engulfing": bullish_engulfing,
    "bearish_engulfing": bearish_engulfing,
    "piercing_line": piercing_line,
    "dark_cloud_cover": dark_cloud_cover,
    "bullish_harami": bullish_harami,
    "bearish_harami": bearish_harami,
    "tweezer_bottom": tweezer_bottom,
    "tweezer_top": tweezer_top,
    "outside_bar": outside_bar,
    "inside_bar": inside_bar,
    "morning_star": morning_star,
    "evening_star": evening_star,
    "three_white_soldiers": three_white_soldiers,
    "three_black_crows": three_black_crows,
}

INDICATOR_PATTERNS = {
    "golden_cross": golden_cross,
    "death_cross": death_cross,
    "rsi_oversold_cross": rsi_oversold_cross,
    "rsi_overbought_cross": rsi_overbought_cross,
    "macd_bullish_cross": macd_bullish_cross,
    "macd_bearish_cross": macd_bearish_cross,
    "donchian_breakout_up": donchian_breakout_up,
    "donchian_breakout_down": donchian_breakout_down,
    "atr_expansion": atr_expansion,
    "bb_upper_breakout": bb_upper_breakout,
    "bb_lower_breakout": bb_lower_breakout,
    "stoch_oversold_cross": stoch_oversold_cross,
    "stoch_overbought_cross": stoch_overbought_cross,
    "adx_trending": adx_trending,
    "vwap_bullish_cross": vwap_bullish_cross,
    "vwap_bearish_cross": vwap_bearish_cross,
    # Registered here like every other pattern, not hard-restricted to
    # 15min in code: this system's whole philosophy (see build_pattern_
    # library.py / risk_reward.py's hard 60%-win-rate gate) is "mine
    # everywhere, let the gate decide," never "assume a timeframe works
    # without checking." It was DESIGNED for 15min, and its qualifying/
    # promoted status per timeframe is exactly how you'd confirm that -
    # check model_registry/pattern_library results for "trend_following_
    # long"/"trend_following_short" specifically on the 15min file
    # rather than assuming.
    "trend_following_long": trend_following_long,
    "trend_following_short": trend_following_short,
}

ALL_PATTERNS = {**CANDLESTICK_PATTERNS, **INDICATOR_PATTERNS}

# Nominal direction bias for patterns whose textbook interpretation has a
# clear side. Patterns not listed here (doji, outside/inside_bar,
# atr_expansion, adx_trending, and every session_*/fundamental_* pattern)
# are direction-agnostic: both a long and a short version are mined, and
# each is gated independently. Lives here (not in build_pattern_library.py)
# so combo_patterns.py can reuse it without importing the mining script.
DIRECTION_HINT = {
    "hammer": +1,
    "shooting_star": -1,
    "hanging_man": -1,
    "inverted_hammer": +1,
    "marubozu_bullish": +1,
    "marubozu_bearish": -1,
    "bullish_engulfing": +1,
    "bearish_engulfing": -1,
    "piercing_line": +1,
    "dark_cloud_cover": -1,
    "bullish_harami": +1,
    "bearish_harami": -1,
    "tweezer_bottom": +1,
    "tweezer_top": -1,
    "morning_star": +1,
    "evening_star": -1,
    "three_white_soldiers": +1,
    "three_black_crows": -1,
    "golden_cross": +1,
    "death_cross": -1,
    "rsi_oversold_cross": +1,
    "rsi_overbought_cross": -1,
    "macd_bullish_cross": +1,
    "macd_bearish_cross": -1,
    "donchian_breakout_up": +1,
    "donchian_breakout_down": -1,
    "bb_upper_breakout": +1,
    "bb_lower_breakout": -1,
    "stoch_oversold_cross": +1,
    "stoch_overbought_cross": -1,
    "vwap_bullish_cross": +1,
    "vwap_bearish_cross": -1,
    "trend_following_long": +1,
    "trend_following_short": -1,
    "doji": 0,
    "outside_bar": 0,
    "inside_bar": 0,
    "atr_expansion": 0,
    "adx_trending": 0,
}

def pattern_direction_hint(name: str) -> int:
    """+1/-1 for patterns with a textbook directional bias, 0 for
    direction-agnostic ones - including every session_*/fundamental_*
    name, which aren't in DIRECTION_HINT at all and correctly default to
    0 (ambiguous, mined both ways) via .get().

    support_resistance.py's patterns (swing/round-number/pivot
    rejection-or-bounce) and smc_patterns.py's patterns (liquidity
    sweeps / fair value gaps) also have an unambiguous textbook
    direction - a rejection/high-side sweep/bearish gap is bearish, a
    bounce/low-side sweep/bullish gap is bullish, by construction of the
    shape test itself - looked up here as a fallback so this stays the
    ONE direction lookup for every pattern name in the system regardless
    of which module defines it, same as CANDLESTICK_PATTERNS/
    INDICATOR_PATTERNS already sharing DIRECTION_HINT above. Imported
    lazily, inside the function, not at module top-level: support_
    resistance.py itself imports FROM patterns.py (for _body/_range/...),
    so a top-level import here would be circular (smc_patterns.py in
    turn imports FROM support_resistance.py, for swing_levels - same
    reasoning, same fix)."""
    if name in DIRECTION_HINT:
        return DIRECTION_HINT[name]
    from support_resistance import DIRECTION_HINT as _SR_DIRECTION_HINT
    if name in _SR_DIRECTION_HINT:
        return _SR_DIRECTION_HINT[name]
    from smc_patterns import DIRECTION_HINT as _SMC_DIRECTION_HINT
    return _SMC_DIRECTION_HINT.get(name, 0)


def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a boolean DataFrame, same index as df, one column per pattern."""
    out = {}
    for name, fn in ALL_PATTERNS.items():
        out[name] = fn(df).fillna(False)
    return pd.DataFrame(out, index=df.index)
