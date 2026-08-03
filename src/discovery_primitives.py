"""
Pattern Discovery Engine - Layer 1: the primitive library.

Instead of a fixed catalog of named patterns (patterns.py's hammer,
engulfing, ...), this is a BOUNDED, FINITE set of reusable, parameterized
CONDITION BUILDING BLOCKS - "5-candle upward slope," "RSI above 70,"
"within 1.5x ATR of a swing high," "CPI surprise more than 1 std above
its own trailing dispersion." discovery_search.py combines these into
multi-primitive conjunctions (genuinely new, self-constructed patterns)
rather than a human hand-picking which shapes to test.

Deliberately BOUNDED, not "every possible threshold on every possible
feature": each primitive TEMPLATE below is instantiated at a small,
fixed set of parameter values (e.g. window in {5, 10, 20}), producing a
few hundred total primitives, not an unconstrained continuum. This
matters - an unbounded parameter space would turn the search in
discovery_search.py back into the "test enough things and something
passes by luck" trap this whole design exists to avoid (see
discovery_validation.py for the other half of that defense: multi-era +
FDR-corrected + blind-confirmation-slice validation).

Every primitive:
  - Is look-ahead-safe by construction (only ever uses candles strictly
    at-or-before the one being evaluated) - same discipline as every
    other detector in this codebase.
  - Reuses existing, already-verified functions (risk_reward.atr,
    patterns.rsi/macd/sma, regime.adx, support_resistance.*,
    session_patterns.*) rather than reimplementing them - one source of
    truth for "what RSI means" everywhere in this system, not a second,
    possibly-drifted copy.
  - Is tagged with a FAMILY (momentum, volatility, structure, level_*,
    fundamental, session) - discovery_search.py enforces cross-family-
    only combination, the exact same anti-redundancy rule
    combo_patterns.py already applies to hand-picked patterns (two
    momentum primitives together tend to be near-duplicates of each
    other, not real confluence).
  - Has a direction_hint (+1/-1/0) - discovery_search.py refuses to
    combine primitives whose nonzero hints disagree, same contradiction
    rule combo_patterns.py already uses.

Signature every primitive function shares: fn(candles, events) ->
pd.Series[bool], aligned to candles.index. `events` may be None (most
primitives ignore it entirely - only the fundamental-surprise family
uses it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import smc_patterns as _smc
from patterns import bb_lower_breakout as _pat_bb_lower
from patterns import bb_upper_breakout as _pat_bb_upper
from patterns import bollinger_bands as _bollinger_bands
from patterns import macd as _macd
from patterns import rsi as _rsi
from patterns import sma as _sma
from patterns import stochastic as _stochastic
from patterns import vwap as _vwap
from regime import adx as _adx
from risk_reward import atr as _atr
from session_patterns import SESSIONS, UTC
from support_resistance import daily_pivots, nearest_round_number, swing_levels


@dataclass(frozen=True)
class Primitive:
    name: str
    family: str
    direction_hint: int  # +1 bullish, -1 bearish, 0 ambiguous
    fn: Callable[[pd.DataFrame, "pd.DataFrame | None"], pd.Series]


PRIMITIVES: list[Primitive] = []


def _register(name: str, family: str, direction_hint: int, fn) -> None:
    PRIMITIVES.append(Primitive(name=name, family=family, direction_hint=direction_hint, fn=fn))


# ---- momentum / trend-shape family ------------------------------------------
# "Movement shape" over more than 1-2 candles, as continuous, combinable
# measurements rather than one hand-coded "head and shoulders" rule - the
# search decides which combinations of these actually predict outcomes.

def _slope(close: pd.Series, window: int) -> pd.Series:
    """Simple linear-regression slope of close over the trailing `window`
    candles (candle strictly before the current one through the current
    one - shift(0), no look-ahead since it only ever uses already-closed
    history up to and including the evaluation candle, same timing
    convention risk_reward.atr() already uses).

    Closed-form rolling OLS slope, not a per-window Python callback via
    rolling().apply() (the original implementation - correct, but ~30x
    slower on large timeframes: benchmarked at 4.5s/window on 15min's
    488K candles, and this primitive is instantiated 12x (slope_up/down)
    plus indirectly 12 more times via _acceleration below, so it was a
    real contributor to search runtime on 15min/5min). Since the x
    values (candle POSITION within each window) are always exactly
    0..window-1 - fixed and known ahead of time, never the actual close
    values - the standard OLS slope formula
        slope = [w*sum(k*y_k) - sum(k)*sum(y_k)] / [w*sum(k^2) - sum(k)^2]
    reduces to two ROLLING SUMS (vectorized, C-level:
    close.rolling(window).sum() and an equivalent rolling sum of
    position-weighted values) plus fixed constants (sum(k), sum(k^2) -
    functions of window only) - no Python-level per-window callback
    needed. sum(k*y_k) for k=0..window-1 (position WITHIN the window) is
    derived from a plain rolling sum of a precomputed (global row index *
    y) series via sum(k*y_k) = sum_j((j-start)*y_j) = RollingSum(j*y_j) -
    start*RollingSum(y_j), where start = (row's global index) - window +
    1 is the window's own starting global index - deterministic per row,
    not itself something that needs rolling.
    Verified against the original Python-callback implementation:
    matches to ~1e-9 (floating-point rounding order only) across every
    tested window, including identical NaN propagation (a NaN anywhere
    in the window still propagates via the rolling sums' own NaN
    handling, same as the original's explicit `if np.isnan(y).any()`
    check)."""
    y = close.to_numpy(dtype=float)
    n = len(y)
    idx = np.arange(n, dtype=float)
    sum_y = close.rolling(window).sum().to_numpy()
    sum_iy = pd.Series(idx * y, index=close.index).rolling(window).sum().to_numpy()
    start = idx - window + 1  # each row's window's own starting global index
    numerator = sum_iy - start * sum_y  # sum(k*y_k), k = position within window (0..window-1)
    w = float(window)
    sum_k = w * (w - 1) / 2
    sum_k2 = (w - 1) * w * (2 * w - 1) / 6
    denom = w * sum_k2 - sum_k ** 2
    return pd.Series((w * numerator - sum_k * sum_y) / denom, index=close.index)


# Public alias - discovery_synthesis.py (Layer 0, primitive SYNTHESIS) reuses
# this exact look-ahead-safe implementation rather than a second copy, same
# "one shared function" discipline this whole codebase applies everywhere.
rolling_slope = _slope


# Threshold/window GRIDS below are deliberately denser than a human would
# hand-pick (e.g. RSI swept every 5 points from 55-85, not just "70, maybe
# 80") - the search (discovery_search.py) and FDR correction (discovery_
# validation.py, which scales its acceptance bar with HOW MANY candidates
# were actually tested) decide which specific value earns its place, not a
# human guessing round numbers. Still deliberately BOUNDED (a fixed grid,
# not a continuum) for the same reason discovery_primitives.py's own module
# docstring gives - an unbounded parameter space is exactly the "test
# enough things and something passes by luck" trap this design avoids.
# Every value already present in the pre-densification grids is KEPT
# (never removed/renamed) so patterns discovered and saved to disk under
# an OLDER, sparser grid stay valid - discover_patterns.is_discovered_
# pattern_active() looks primitives up BY NAME, so a name that used to
# exist must keep existing.
for _window in (5, 8, 10, 15, 20, 30):
    _register(f"slope_up_{_window}", "momentum", +1,
              (lambda df, ev, w=_window: _slope(df["close"], w) > 0))
    _register(f"slope_down_{_window}", "momentum", -1,
              (lambda df, ev, w=_window: _slope(df["close"], w) < 0))


def _acceleration(close: pd.Series, window: int) -> pd.Series:
    """Slope-of-the-slope: is the trend itself speeding up or slowing
    down, not just which way it points."""
    s = _slope(close, window)
    return s - s.shift(window)


for _window in (5, 8, 10, 15, 20, 30):
    _register(f"accelerating_up_{_window}", "momentum", +1,
              (lambda df, ev, w=_window: _acceleration(df["close"], w) > 0))
    _register(f"accelerating_down_{_window}", "momentum", -1,
              (lambda df, ev, w=_window: _acceleration(df["close"], w) < 0))


def _consecutive_run(series_diff_positive: pd.Series, min_count: int) -> pd.Series:
    """True where the current streak of consecutive positive (or,
    inverted, negative) diffs is at least `min_count` long."""
    groups = (series_diff_positive != series_diff_positive.shift()).cumsum()
    run_length = series_diff_positive.groupby(groups).cumcount() + 1
    return series_diff_positive & (run_length >= min_count)


for _count in (3, 4, 5, 6, 8, 10):
    _register(f"consecutive_higher_highs_{_count}", "momentum", +1,
              (lambda df, ev, n=_count: _consecutive_run(df["high"].diff() > 0, n)))
    _register(f"consecutive_lower_lows_{_count}", "momentum", -1,
              (lambda df, ev, n=_count: _consecutive_run(df["low"].diff() < 0, n)))

for _level in (55, 60, 65, 70, 75, 80, 85):
    _register(f"rsi_above_{_level}", "momentum", +1,
              (lambda df, ev, lvl=_level: _rsi(df["close"]) > lvl))
for _level in (15, 20, 25, 30, 35, 40, 45):
    _register(f"rsi_below_{_level}", "momentum", -1,
              (lambda df, ev, lvl=_level: _rsi(df["close"]) < lvl))


def _macd_hist(close: pd.Series) -> pd.Series:
    macd_line, signal_line = _macd(close)
    return macd_line - signal_line


_register("macd_hist_positive", "momentum", +1, lambda df, ev: _macd_hist(df["close"]) > 0)
_register("macd_hist_negative", "momentum", -1, lambda df, ev: _macd_hist(df["close"]) < 0)

for _threshold in (20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0):
    _register(f"adx_above_{int(_threshold)}", "momentum", 0,
              (lambda df, ev, t=_threshold: _adx(df) > t))


# ---- time-series momentum family (raw N-period return) ------------------------
# Deliberately DISTINCT from slope_up/down above, not a duplicate: _slope()
# is a least-squares regression over the trailing window (rewards a SMOOTH,
# consistent trajectory), this is the literal academic time-series-momentum
# factor definition (Jegadeesh & Titman-style: sign/magnitude of the raw
# total return over the trailing N periods, indifferent to how smooth or
# choppy the path there was) - two candles that end up implying the same
# raw return can have very different regression slopes (a smooth grind vs a
# single spike), and the search can now tell whether smoothness or raw
# magnitude is the thing that actually matters here.

def _n_period_return(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(window)


for _window in (10, 20, 50, 100):
    _register(f"momentum_return_up_{_window}", "momentum", +1,
              (lambda df, ev, w=_window: _n_period_return(df["close"], w) > 0))
    _register(f"momentum_return_down_{_window}", "momentum", -1,
              (lambda df, ev, w=_window: _n_period_return(df["close"], w) < 0))
for _window, _pct in [(20, 0.01), (20, 0.02), (50, 0.02), (50, 0.03)]:
    _register(f"momentum_return_strong_up_{_window}_{_pct}", "momentum", +1,
              (lambda df, ev, w=_window, p=_pct: _n_period_return(df["close"], w) > p))
    _register(f"momentum_return_strong_down_{_window}_{_pct}", "momentum", -1,
              (lambda df, ev, w=_window, p=_pct: _n_period_return(df["close"], w) < -p))


# ---- volatility family -------------------------------------------------------

def _volatility_ratio(df: pd.DataFrame, baseline_window: int = 100) -> pd.Series:
    a = _atr(df)
    baseline = a.shift(1).rolling(baseline_window).mean()
    return a / baseline


for _mult in (1.2, 1.3, 1.5, 1.75, 2.0, 2.5):
    _register(f"volatility_expansion_{_mult}", "volatility", 0,
              (lambda df, ev, m=_mult: _volatility_ratio(df) > m))
for _mult in (0.4, 0.5, 0.6, 0.7, 0.8):
    _register(f"volatility_contraction_{_mult}", "volatility", 0,
              (lambda df, ev, m=_mult: _volatility_ratio(df) < m))


def _vol_of_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Is realized volatility itself becoming more erratic - the second-
    order volatility measurement, not just "is ATR big right now.\""""
    a = _atr(df)
    return a.rolling(window).std() / a.rolling(window).mean()


for _threshold in (0.2, 0.3, 0.4, 0.5, 0.6):
    _register(f"vol_of_vol_above_{_threshold}", "volatility", 0,
              (lambda df, ev, t=_threshold: _vol_of_vol(df) > t))


# ---- oscillator family (Bollinger / Stochastic / VWAP) -------------------------
# All three existed in patterns.py (used by the SEPARATE hand-picked
# build_pattern_library.py pipeline) but, like smc_patterns.py before this
# session, were never wired into THIS search - discover_patterns.py /
# explore_setups.py have never been able to combine a Bollinger squeeze,
# Stochastic cross, or VWAP cross with anything from another family until
# now. Reuses patterns.py's own implementations directly, no reimplementation.

def _bb_position(df: pd.DataFrame) -> pd.Series:
    """0 = at the lower band, 1 = at the upper band, can exceed [0,1]
    when price is genuinely outside the bands - continuous position
    analogous to structure family's near_range_high, but volatility-
    normalized (std-based) rather than a fixed high/low channel."""
    upper, _, lower = _bollinger_bands(df["close"])
    return (df["close"] - lower) / (upper - lower).replace(0, np.nan)


_register("bb_upper_breakout", "oscillator", +1, lambda df, ev: _pat_bb_upper(df))
_register("bb_lower_breakout", "oscillator", -1, lambda df, ev: _pat_bb_lower(df))
for _thresh in (0.9, 0.95, 1.0):
    _register(f"bb_near_upper_{_thresh}", "oscillator", -1,
              (lambda df, ev, t=_thresh: _bb_position(df) > t))
for _thresh in (0.1, 0.05, 0.0):
    _register(f"bb_near_lower_{_thresh}", "oscillator", +1,
              (lambda df, ev, t=_thresh: _bb_position(df) < t))

_register("stoch_oversold_cross", "oscillator", +1,
          lambda df, ev: (_stochastic(df)[0] > 20) & (_stochastic(df)[0].shift(1) <= 20))
_register("stoch_overbought_cross", "oscillator", -1,
          lambda df, ev: (_stochastic(df)[0] < 80) & (_stochastic(df)[0].shift(1) >= 80))
for _level in (10, 20, 30):
    _register(f"stoch_below_{_level}", "oscillator", +1,
              (lambda df, ev, lvl=_level: _stochastic(df)[0] < lvl))
for _level in (70, 80, 90):
    _register(f"stoch_above_{_level}", "oscillator", -1,
              (lambda df, ev, lvl=_level: _stochastic(df)[0] > lvl))

_register("vwap_bullish_cross", "oscillator", +1,
          lambda df, ev: (df["close"] > _vwap(df)) & (df["close"].shift(1) <= _vwap(df).shift(1)))
_register("vwap_bearish_cross", "oscillator", -1,
          lambda df, ev: (df["close"] < _vwap(df)) & (df["close"].shift(1) >= _vwap(df).shift(1)))
_register("above_vwap", "oscillator", +1, lambda df, ev: df["close"] > _vwap(df))
_register("below_vwap", "oscillator", -1, lambda df, ev: df["close"] < _vwap(df))


# ---- structure family (position within a recent range) ----------------------

def _donchian_position(df: pd.DataFrame, window: int) -> pd.Series:
    donchian_high = df["high"].shift(1).rolling(window).max()
    donchian_low = df["low"].shift(1).rolling(window).min()
    rng = (donchian_high - donchian_low).replace(0, np.nan)
    return (df["close"] - donchian_low) / rng


for _window in (10, 20, 50):
    _register(f"near_range_high_{_window}", "structure", +1,
              (lambda df, ev, w=_window: _donchian_position(df, w) > 0.8))
    _register(f"near_range_low_{_window}", "structure", -1,
              (lambda df, ev, w=_window: _donchian_position(df, w) < 0.2))

# Denser threshold coverage for the SAME donchian-position measurement -
# separate names (window AND threshold both baked in) rather than
# widening the loop above, so the original near_range_high_10/20/50 (at
# the fixed 0.8/0.2 threshold) names never change meaning.
for _window in (10, 20, 30, 50, 75):
    for _threshold in (0.7, 0.75, 0.85, 0.9):
        _register(f"near_range_high_{_window}_{_threshold}", "structure", +1,
                  (lambda df, ev, w=_window, t=_threshold: _donchian_position(df, w) > t))
    for _threshold in (0.1, 0.15, 0.25, 0.3):
        _register(f"near_range_low_{_window}_{_threshold}", "structure", -1,
                  (lambda df, ev, w=_window, t=_threshold: _donchian_position(df, w) < t))

for _window in (10, 15, 20, 30, 50, 75, 100):
    _register(f"sma_distance_above_{_window}", "structure", +1,
              (lambda df, ev, w=_window: df["close"] > _sma(df["close"], w)))
    _register(f"sma_distance_below_{_window}", "structure", -1,
              (lambda df, ev, w=_window: df["close"] < _sma(df["close"], w)))

# Distance MAGNITUDE, not just sign - "meaningfully above/below the SMA
# by at least N x ATR," not just barely crossing it. A genuinely new
# parametrization axis (how far, not just which window), same reasoning
# as combo_patterns.py's own preference for a real, provable edge over a
# borderline one.
for _window in (10, 20, 50):
    for _mult in (0.3, 0.5, 1.0):
        _register(f"sma_distance_above_{_window}_atr{_mult}", "structure", +1,
                  (lambda df, ev, w=_window, m=_mult:
                   (df["close"] - _sma(df["close"], w)) / _atr(df).replace(0, np.nan) > m))
        _register(f"sma_distance_below_{_window}_atr{_mult}", "structure", -1,
                  (lambda df, ev, w=_window, m=_mult:
                   (_sma(df["close"], w) - df["close"]) / _atr(df).replace(0, np.nan) > m))


# ---- level-proximity families (each its own family - a swing-level
# proximity and a round-number proximity are NOT redundant with each
# other the way two momentum primitives would be, so they're kept
# separate rather than lumped into one "level" family) ----

def _atr_distance(price_a: pd.Series, price_b: pd.Series, atr_series: pd.Series) -> pd.Series:
    return (price_a - price_b).abs() / atr_series.replace(0, np.nan)


for _mult in (0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    _register(f"near_swing_high_{_mult}", "level_swing", -1,
              (lambda df, ev, m=_mult: _atr_distance(df["close"], swing_levels(df)[0], _atr(df)) < m))
    _register(f"near_swing_low_{_mult}", "level_swing", +1,
              (lambda df, ev, m=_mult: _atr_distance(df["close"], swing_levels(df)[1], _atr(df)) < m))

for _mult in (0.2, 0.3, 0.5, 0.75, 1.0):
    _register(f"near_round_number_{_mult}", "level_round", 0,
              (lambda df, ev, m=_mult: _atr_distance(df["close"], nearest_round_number(df), _atr(df)) < m))

for _mult in (0.3, 0.5, 0.75, 1.0, 1.5):
    _register(f"near_pivot_r1_{_mult}", "level_pivot", -1,
              (lambda df, ev, m=_mult: _atr_distance(df["close"], daily_pivots(df)[1], _atr(df)) < m))
    _register(f"near_pivot_s1_{_mult}", "level_pivot", +1,
              (lambda df, ev, m=_mult: _atr_distance(df["close"], daily_pivots(df)[2], _atr(df)) < m))


# ---- fundamental-surprise family (STRICTER than fundamental_patterns.py's
# plain accelerating/decelerating sign check - requires the surprise to be
# a meaningfully LARGE deviation, not just directionally positive/negative) ----

def _surprise_zscore_on_candles(candles: pd.DataFrame, events: "pd.DataFrame | None",
                                 event_type: str, lookback: int = 8) -> pd.Series:
    """Maps each event's surprise size - `vs_trend`, already computed by
    build_fundamentals.py as this reading's deviation from ITS OWN
    trailing trend - onto the candle it was first published on (same
    forward-merge-asof timing fundamental_patterns._map_events_to_candles
    uses), then z-scores it against that event type's own trailing
    `lookback` releases (a rolling, expanding-safe z-score - only ever
    uses releases strictly before the current one, so no look-ahead).
    Forward-filled onto every candle between releases (the surprise
    "stays in effect" as background context until the next release,
    same convention `daily_pivots()` uses for holding a level constant).
    None (an all-False Series, via .fillna(False) at the call site) if
    `events` wasn't supplied or this event type isn't present - fails
    open like every other optional-fundamentals code path in this
    system (see build_pattern_library.py's own "no fundamentals loaded"
    handling)."""
    if events is None or events.empty or "vs_trend" not in events.columns:
        return pd.Series(np.nan, index=candles.index)
    sub = events[events["event_type"] == event_type].sort_values("datetime_utc")
    if sub.empty:
        return pd.Series(np.nan, index=candles.index)

    vs_trend = sub["vs_trend"].astype(float)
    rolling_std = vs_trend.shift(1).rolling(lookback, min_periods=3).std()
    z = vs_trend / rolling_std.replace(0, np.nan)

    ev_ts = pd.to_datetime(sub["datetime_utc"]).to_numpy()
    cd_ts = pd.to_datetime(candles["timestamp"]).to_numpy()
    idx = np.searchsorted(ev_ts, cd_ts, side="right") - 1
    z_vals = z.to_numpy()
    out = np.where(idx >= 0, z_vals[np.clip(idx, 0, len(z_vals) - 1)], np.nan)
    return pd.Series(out, index=candles.index)


for _event_type in ("CPI", "PCE", "NFP", "GDP"):
    for _z in (0.75, 1.0, 1.25, 1.5, 2.0):
        _register(f"fundamental_{_event_type.lower()}_surprise_hot_{_z}", "fundamental", +1,
                   (lambda df, ev, et=_event_type, z=_z:
                    (_surprise_zscore_on_candles(df, ev, et) > z).fillna(False)))
        _register(f"fundamental_{_event_type.lower()}_surprise_cool_{_z}", "fundamental", -1,
                   (lambda df, ev, et=_event_type, z=_z:
                    (_surprise_zscore_on_candles(df, ev, et) < -z).fillna(False)))


# ---- carry family (DXY / real-yield level + trend) -----------------------------
# Gold pays no yield, so its "carry" (the return from just holding the
# position, independent of price direction - the same concept FX/rates
# carry trades are named for) is the OPPORTUNITY COST of not holding a
# yielding asset instead: when real (inflation-adjusted) yields rise,
# holding gold gets relatively more expensive (bearish gold); when they
# fall, relatively cheaper (bullish gold) - a real, causal, widely-cited
# mechanism, not a hand-wavy analogy. DXY strength/weakness is gold's
# other standard macro cross-check (gold is dollar-priced, so a stronger
# dollar is mechanically bearish all else equal, though the correlation is
# regime-dependent - exactly what the search is for).
#
# These primitives read df["dxy"]/df["real_yield_10y"] columns that
# explore_setups.py's _attach_context_series() merges onto candles from
# data/context/*.parquet BEFORE the search ever sees them (look-ahead-safe
# merge_asof, most-recent-PRIOR-day value only) - NOT columns every caller
# is guaranteed to have (a candles frame without that merge applied simply
# lacks them), so each primitive here checks for the column and returns an
# all-False Series rather than raising when it's absent.

def _context_col_or_none(df: pd.DataFrame, col: str) -> "pd.Series | None":
    return df[col] if col in df.columns else None


def _context_trend(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    series = _context_col_or_none(df, col)
    if series is None:
        return pd.Series(False, index=df.index)
    return series.diff(window)


for _window in (20, 100):
    _register(f"carry_real_yield_rising_{_window}", "carry", -1,
              (lambda df, ev, w=_window: (_context_trend(df, "real_yield_10y", w) > 0).fillna(False)))
    _register(f"carry_real_yield_falling_{_window}", "carry", +1,
              (lambda df, ev, w=_window: (_context_trend(df, "real_yield_10y", w) < 0).fillna(False)))
    _register(f"carry_dxy_rising_{_window}", "carry", -1,
              (lambda df, ev, w=_window: (_context_trend(df, "dxy", w) > 0).fillna(False)))
    _register(f"carry_dxy_falling_{_window}", "carry", +1,
              (lambda df, ev, w=_window: (_context_trend(df, "dxy", w) < 0).fillna(False)))


def _context_vs_own_median(df: pd.DataFrame, col: str, window: int, above: bool) -> pd.Series:
    """Level relative to its own trailing rolling median - a self-
    normalizing "high/low regime" test that needs no hardcoded absolute
    threshold (a real yield of 2% was a totally different regime in 2010
    than in 2023 - this is regime-relative, not an assumed fixed
    cutoff). `above`/"below" are each computed directly with their own
    fillna(False) (NOT one as `~` the other) - during the rolling
    window's warm-up period the median itself is NaN, so `series >
    median` and `series < median` are BOTH correctly False/undefined
    there; naively inverting one to get the other would incorrectly mark
    the entire warm-up period as "below" (or "above")."""
    series = _context_col_or_none(df, col)
    if series is None:
        return pd.Series(False, index=df.index)
    median = series.rolling(window, min_periods=window // 2).median()
    return (series > median).fillna(False) if above else (series < median).fillna(False)


for _window in (250, 1000):
    _register(f"carry_real_yield_above_own_median_{_window}", "carry", -1,
              (lambda df, ev, w=_window: _context_vs_own_median(df, "real_yield_10y", w, above=True)))
    _register(f"carry_real_yield_below_own_median_{_window}", "carry", +1,
              (lambda df, ev, w=_window: _context_vs_own_median(df, "real_yield_10y", w, above=False)))


# ---- seasonality family --------------------------------------------------------
# Calendar-based, not lunar-precise (Diwali/Chinese New Year drift ~3-6
# weeks year to year on the Gregorian calendar - pinning an exact date
# window would need a lunar-calendar lookup table this codebase doesn't
# have) - deliberately simple, deterministic, and independent of any
# assumption about which months are actually favorable: registers one
# primitive per calendar month (direction_hint=0, same as session_*
# primitives) plus the two broad windows real-world gold seasonality
# research repeatedly documents (India's Oct-Nov wedding season/Dhanteras/
# Diwali physical demand, and the wider Nov-Feb window where that overlaps
# Chinese New Year buying) - the search finds out empirically whether
# THIS dataset actually shows the effect, rather than the primitive
# itself asserting a direction.

def _month_is(candles: pd.DataFrame, month: int) -> pd.Series:
    return pd.to_datetime(candles["timestamp"]).dt.month == month

def _month_in(candles: pd.DataFrame, months: set[int]) -> pd.Series:
    return pd.to_datetime(candles["timestamp"]).dt.month.isin(months)


for _month in range(1, 13):
    _register(f"seasonality_month_{_month}", "seasonality", 0,
              (lambda df, ev, m=_month: _month_is(df, m)))

_register("seasonality_diwali_window", "seasonality", 0,
          (lambda df, ev: _month_in(df, {10, 11})))
_register("seasonality_golden_window", "seasonality", 0,
          (lambda df, ev: _month_in(df, {11, 12, 1, 2})))


# ---- smc family (Smart Money Concepts / ICT) -----------------------------------
# smc_patterns.py's functions were, until now, only reachable through the
# SEPARATE hand-picked pipeline (build_pattern_library.py's
# compute_pattern_flags()) - never through THIS search, so discover_patterns.py
# / explore_setups.py have never once been able to combine a liquidity sweep,
# FVG, BOS/CHoCH, or order block with anything from another family. Wired in
# here at explicit user request for an unconstrained, any-methodology search.
for _name, _fn, _hint in [
    ("smc_liquidity_sweep_low", _smc.liquidity_sweep_low, +1),
    ("smc_liquidity_sweep_high", _smc.liquidity_sweep_high, -1),
    ("smc_fvg_bullish", _smc.fvg_bullish, +1),
    ("smc_fvg_bearish", _smc.fvg_bearish, -1),
    ("smc_bos_bullish", _smc.bos_bullish, +1),
    ("smc_bos_bearish", _smc.bos_bearish, -1),
    ("smc_choch_bullish", _smc.choch_bullish, +1),
    ("smc_choch_bearish", _smc.choch_bearish, -1),
    ("smc_eq_high_sweep", _smc.eq_high_sweep, -1),
    ("smc_eq_low_sweep", _smc.eq_low_sweep, +1),
    ("smc_bullish_sweep_fvg", _smc.smc_bullish_sweep_fvg, +1),
    ("smc_bearish_sweep_fvg", _smc.smc_bearish_sweep_fvg, -1),
    ("smc_near_bullish_order_block", _smc.near_bullish_order_block, +1),
    ("smc_near_bearish_order_block", _smc.near_bearish_order_block, -1),
]:
    _register(_name, "smc", _hint, (lambda df, ev, f=_fn: f(df)))

for _window in (20, 50, 100):
    _register(f"smc_discount_zone_{_window}", "smc", +1,
              (lambda df, ev, w=_window: _smc.range_position(df, w) < 0.3))
    _register(f"smc_premium_zone_{_window}", "smc", -1,
              (lambda df, ev, w=_window: _smc.range_position(df, w) > 0.7))


# ---- ict_timing family (killzones) ---------------------------------------------
# ICT's "killzones": specific intraday windows retail ICT material claims
# see disproportionate institutional volume/volatility - converted from the
# stated New York local times the same DST-safe way session_patterns.py's
# own SESSIONS already are (not a fixed UTC offset, which would be wrong
# half the year).
ICT_KILLZONES = {
    "london_kz": {"tz": "America/New_York", "open": (2, 0), "close": (5, 0)},
    "ny_am_kz": {"tz": "America/New_York", "open": (7, 0), "close": (10, 0)},
}

def _killzone_active(candles: pd.DataFrame, cfg: dict) -> pd.Series:
    ts = pd.to_datetime(candles["timestamp"])
    ts_utc = ts.dt.tz_localize(UTC) if ts.dt.tz is None else ts.dt.tz_convert(UTC)
    local = ts_utc.dt.tz_convert(ZoneInfo(cfg["tz"]))
    local_minutes = local.dt.hour * 60 + local.dt.minute
    open_h, open_m = cfg["open"]
    close_h, close_m = cfg["close"]
    open_min, close_min = open_h * 60 + open_m, close_h * 60 + close_m
    return (local_minutes >= open_min) & (local_minutes < close_min)

for _name, _cfg in ICT_KILLZONES.items():
    _register(f"ict_{_name}_active", "ict_timing", 0,
              (lambda df, ev, c=_cfg: _killzone_active(df, c)))


# ---- session family -----------------------------------------------------------

def _session_active(candles: pd.DataFrame, name: str) -> pd.Series:
    """Vectorized equivalent of ts.apply(lambda t: name in
    active_sessions_at(t)) - the per-row Python callback + per-call
    ZoneInfo conversion in session_patterns.active_sessions_at() (built
    for single-timestamp live lookups elsewhere, e.g. signal_journal.py)
    benchmarked at ~23s PER SESSION on 15min's 488K candles (~90s for
    all 4, worse again on 5min's 1.4M) - a real contributor to search
    runtime on the larger timeframes. pandas can convert an entire
    Series of timestamps to a given IANA timezone in one vectorized,
    C-level call (Series.dt.tz_convert), so this reimplements
    active_sessions_at()'s exact open<=local_minutes<close comparison
    for ONE named session, elementwise over the whole Series at once,
    instead of calling into that function per row. Verified byte-for-
    byte identical output against the original on a 20K-row sample
    across all 4 sessions (including DST-transition dates, which
    tz_convert handles correctly the same way ZoneInfo does)."""
    cfg = SESSIONS[name]
    ts = pd.to_datetime(candles["timestamp"])
    ts_utc = ts.dt.tz_localize(UTC) if ts.dt.tz is None else ts.dt.tz_convert(UTC)
    local = ts_utc.dt.tz_convert(ZoneInfo(cfg["tz"]))
    local_minutes = local.dt.hour * 60 + local.dt.minute
    open_h, open_m = cfg["open"]
    close_h, close_m = cfg["close"]
    open_min, close_min = open_h * 60 + open_m, close_h * 60 + close_m
    return (local_minutes >= open_min) & (local_minutes < close_min)


for _name in SESSIONS:
    _register(f"session_{_name}_active", "session", 0,
              (lambda df, ev, n=_name: _session_active(df, n)))


PRIMITIVES_BY_NAME: dict[str, Primitive] = {p.name: p for p in PRIMITIVES}
FAMILIES: set[str] = {p.family for p in PRIMITIVES}


def evaluate_primitive(name: str, candles: pd.DataFrame, events: "pd.DataFrame | None") -> pd.Series:
    """Boolean Series for one primitive by name, NaN-safe (fillna(False)
    - same convention every other detector in this codebase uses for
    "not enough history yet")."""
    return PRIMITIVES_BY_NAME[name].fn(candles, events).fillna(False)
