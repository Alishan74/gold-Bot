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

import numpy as np
import pandas as pd

from patterns import macd as _macd
from patterns import rsi as _rsi
from patterns import sma as _sma
from regime import adx as _adx
from risk_reward import atr as _atr
from session_patterns import SESSIONS, active_sessions_at
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
    convention risk_reward.atr() already uses)."""
    x = np.arange(window)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = (x_centered ** 2).sum()

    def _win_slope(y):
        if np.isnan(y).any():
            return np.nan
        return float((x_centered * (y - y.mean())).sum() / denom)

    return close.rolling(window).apply(_win_slope, raw=True)


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


# ---- session family -----------------------------------------------------------

def _session_active(candles: pd.DataFrame, name: str) -> pd.Series:
    ts = pd.to_datetime(candles["timestamp"])
    return ts.apply(lambda t: name in active_sessions_at(t))


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
