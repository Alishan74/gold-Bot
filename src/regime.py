"""
Market regime classification: gold doesn't behave the same way in every
market condition, and mining one blended win rate across 20 years of
everything hides that instead of revealing it. A pattern that only
worked during a specific volatility/trend regime can still show a
passing BLENDED win rate if the rest of history was neutral - and the
live system has no way to know the regime has since changed back to one
where that pattern never actually had an edge.

Two independent regime dimensions, each look-ahead-free (computed only
from candles strictly before/at the classification point):

- VOLATILITY regime (LOW / NORMAL / HIGH): current ATR vs. the rolling
  mean of ATR over the preceding window - the same basic comparison
  patterns.atr_expansion already uses for a single binary flag,
  generalized to three buckets here. Deliberately NOT a true rolling
  percentile-rank (which would be O(n * lookback) via pandas
  rolling().apply() and meaningfully slower at 20-years-of-1min scale for
  a real benefit this simpler ratio mostly already captures).
- TREND regime (TRENDING / RANGING): ADX(14) vs. the conventional 25
  threshold - standard trend-strength convention, reimplemented directly
  (no TA-Lib dependency, same policy as patterns.py).

Combined into one label (e.g. "HIGH_TRENDING") for mining/lookup
convenience - see risk_reward.tag_regime / summarize_trades_by_regime
and build_pattern_library.py for how this feeds into per-regime pattern
stats, and signal_engine.py for how the CURRENT regime is looked up at
signal time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from risk_reward import atr

VOLATILITY_BASELINE_WINDOW = 100
VOLATILITY_LOW_MULTIPLE = 0.8
VOLATILITY_HIGH_MULTIPLE = 1.25

TREND_ADX_WINDOW = 14
TREND_ADX_THRESHOLD = 25.0


def volatility_regime(df: pd.DataFrame, atr_window: int = 14,
                       baseline_window: int = VOLATILITY_BASELINE_WINDOW) -> pd.Series:
    """LOW / NORMAL / HIGH, from current ATR vs. the rolling mean of ATR
    over the `baseline_window` candles strictly before it (shift(1)
    before the rolling mean - no look-ahead). None until enough history
    exists to compute a baseline."""
    a = atr(df, atr_window)
    baseline = a.shift(1).rolling(baseline_window).mean()
    ratio = a / baseline

    regime = pd.Series(pd.NA, index=df.index, dtype=object)
    known = baseline.notna() & ratio.notna()
    regime[known] = "NORMAL"
    regime[known & (ratio < VOLATILITY_LOW_MULTIPLE)] = "LOW"
    regime[known & (ratio > VOLATILITY_HIGH_MULTIPLE)] = "HIGH"
    return regime


def _true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def adx(df: pd.DataFrame, window: int = TREND_ADX_WINDOW) -> pd.Series:
    """Standard Average Directional Index - trend STRENGTH (not
    direction). Causal by construction (diff()/ewm() only ever look
    backward), so using it to classify the regime AT a given candle is
    the same known-at-signal-time timing risk_reward.atr() already
    relies on for stop distance."""
    high, low = df["high"], df["low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = _true_range(df)
    atr_ema = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_ema
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr_ema

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=1 / window, adjust=False).mean()


def trend_regime(df: pd.DataFrame, window: int = TREND_ADX_WINDOW,
                  threshold: float = TREND_ADX_THRESHOLD) -> pd.Series:
    """TRENDING / RANGING, from ADX vs. the conventional 25 threshold.
    None for the initial candles before ADX has enough history to mean
    anything (matches adx()'s own warm-up NaNs)."""
    a = adx(df, window)
    regime = pd.Series(pd.NA, index=df.index, dtype=object)
    known = a.notna()
    regime[known] = "RANGING"
    regime[known & (a > threshold)] = "TRENDING"
    return regime


def combined_regime(df: pd.DataFrame) -> pd.Series:
    """"{volatility}_{trend}" (e.g. "HIGH_TRENDING") - None wherever
    either component isn't known yet, so a trade entered too early in
    the history to classify never gets silently assigned a fake regime."""
    vol = volatility_regime(df)
    trend = trend_regime(df)
    both_known = vol.notna() & trend.notna()
    out = pd.Series(pd.NA, index=df.index, dtype=object)
    out[both_known] = vol[both_known].astype(str) + "_" + trend[both_known].astype(str)
    return out


REGIME_LABELS = [f"{v}_{t}" for v in ("LOW", "NORMAL", "HIGH") for t in ("TRENDING", "RANGING")]
