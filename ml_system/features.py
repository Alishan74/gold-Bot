"""
Causal feature engineering - the ML challenger's equivalent of
patterns.py, but producing a bank of NUMERIC features instead of named
boolean patterns, for a model to find interactions across on its own
instead of a human hand-picking which ones to test.

THE SINGLE MOST IMPORTANT RULE IN THIS FILE: every feature must be
computable using ONLY candles up to and including the current one - no
peeking forward. This module is imported by BOTH train.py (building the
training table) and live_signal.py (scoring the live tail) - using the
exact same function for both is what prevents training-serving skew
(computing a feature slightly differently at inference time than at
training time is one of the most common, hardest-to-notice bugs in real
ML systems). Never duplicate this logic anywhere else.

Reuses risk_reward.atr, patterns.sma/rsi/macd, session_patterns'
detectors, and regime.adx directly rather than reimplementing them -
single source of truth for every one of those calculations, same as the
rule-based system already relies on.

All-NaN rows (the warm-up period before the slowest feature - currently
SMA_200 - has enough history) are the caller's responsibility to drop
before training; compute_features() itself never drops rows, so its
output index always lines up 1:1 with the input candles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_history import TIMEFRAME_MINUTES  # noqa: E402
from patterns import macd, rsi, sma  # noqa: E402
from regime import adx  # noqa: E402
from risk_reward import atr  # noqa: E402
from session_patterns import SESSIONS, detect_session_events  # noqa: E402
from support_resistance import PROXIMITY_ATR_MULT, daily_pivots, nearest_round_number, swing_levels  # noqa: E402

N_COARSER_TIMEFRAMES_DEFAULT = 2  # how many next-coarser timeframes' context to merge in by default


def coarser_timeframes(tf: str, n: int = N_COARSER_TIMEFRAMES_DEFAULT) -> list[str]:
    """Up to `n` next-coarser timeframe labels after `tf`, in TIMEFRAME_
    MINUTES' own finest-to-coarsest order (mirrors discover_patterns.
    _next_coarser_timeframe, extended from 1 sibling to N) - e.g. for
    "5min" with n=2: ["15min", "1h"]. Empty for the coarsest timeframe
    (1d) or an unrecognized label - "no coarser context available," not
    an error, same convention as everywhere else cross-timeframe context
    is optional in this codebase. Shared by train.py (building the
    training table) and live_signal.py (scoring live) so both agree on
    exactly which coarser timeframes a given model's cross-timeframe
    context comes from - a mismatch here would silently break
    live_signal.py's reindex-to-stored-feature_columns compatibility."""
    order = list(TIMEFRAME_MINUTES)
    if tf not in order:
        return []
    idx = order.index(tf)
    return order[idx + 1: idx + 1 + n]

RETURN_WINDOWS = (1, 3, 5, 10, 20, 50)
SMA_WINDOWS = (10, 20, 50, 100, 200)
DONCHIAN_WINDOW = 20
VOLATILITY_BASELINE_WINDOW = 100
BODY_TREND_WINDOWS = (3, 5, 8)
VOLUME_TREND_WINDOWS = (3, 5, 10)
AUTOCORR_WINDOW = 50
EFFICIENCY_RATIO_WINDOW = 20
OPENING_RANGE_CANDLES = 3  # first N candles after a session's open boundary define its opening range
EFFICIENCY_TREND_THRESHOLD = 0.3  # efficiency_ratio_20 at/above this = "trending", below = "ranging/choppy"

FEATURE_COLUMNS: list[str] = []  # populated at import time, see bottom of file


def _body_ratio(df: pd.DataFrame) -> pd.Series:
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (body / rng).fillna(0)


def _upper_wick_ratio(df: pd.DataFrame) -> pd.Series:
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (upper / rng).fillna(0)


def _lower_wick_ratio(df: pd.DataFrame) -> pd.Series:
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (lower / rng).fillna(0)


def _streak(condition: pd.Series) -> pd.Series:
    """Length of the current unbroken run of True values in `condition`,
    ending at (and including) the current row - 0 wherever condition is
    currently False. Fully vectorized (no per-row Python loop, no
    rolling().apply()) so it stays fast even on 1min's ~7M-row history:
    group consecutive equal values via a cumulative sum over "did the
    value change from the previous row", then count position within each
    group. NaN-safe: a NaN in `condition` breaks the streak (treated as
    False) rather than silently propagating forward or backward."""
    c = condition.fillna(False).to_numpy()
    if len(c) == 0:
        return pd.Series(dtype=float, index=condition.index)
    change = np.empty(len(c), dtype=bool)
    change[0] = True
    change[1:] = c[1:] != c[:-1]
    group_id = np.cumsum(change)
    position_in_group = pd.Series(np.arange(len(c))).groupby(group_id).cumcount().to_numpy() + 1
    return pd.Series(np.where(c, position_in_group, 0), index=condition.index).astype(float)


def _session_opening_range(candles: pd.DataFrame, open_flags: pd.Series,
                            n_candles: int = OPENING_RANGE_CANDLES) -> tuple[pd.Series, pd.Series]:
    """High/low of the first `n_candles` candles (starting from, and
    including, each session-open boundary candle) - held constant from
    the moment that window itself has fully closed until the NEXT
    occurrence of the same session's open boundary. Look-ahead-safe by
    construction: the range for a given session-day only becomes valid
    (non-NaN) starting at the candle where its OWN defining window last
    closed (index `end-1`), never before - a candle still inside its own
    still-forming opening range reads NaN, not a partial/leaking value.

    Iterates over session-open EVENTS (at most ~1/day - roughly 7300 over
    a 20-year history), not over every candle, so this stays fast
    regardless of timeframe (a 1min run's 7M rows doesn't turn this into
    a 7M-iteration loop)."""
    n = len(candles)
    open_idx = np.flatnonzero(open_flags.fillna(False).to_numpy())
    high_arr = candles["high"].to_numpy()
    low_arr = candles["low"].to_numpy()
    range_high = np.full(n, np.nan)
    range_low = np.full(n, np.nan)

    for i, idx in enumerate(open_idx):
        window_end = min(idx + n_candles, n)
        next_open = open_idx[i + 1] if i + 1 < len(open_idx) else n
        valid_start = window_end - 1  # candle where the defining window itself just closed
        if valid_start >= next_open:
            continue  # next session-open arrives before this range ever fully forms - nothing to hold
        range_high[valid_start:next_open] = high_arr[idx:window_end].max()
        range_low[valid_start:next_open] = low_arr[idx:window_end].min()

    return pd.Series(range_high, index=candles.index), pd.Series(range_low, index=candles.index)


def _categorical_run_length(labels: pd.Series) -> pd.Series:
    """How many consecutive candles (ending at, and including, this one)
    have shared the SAME categorical value as this candle - the general
    version of _streak() above for a multi-valued label instead of a
    boolean condition (used below for regime_duration: how long has the
    CURRENT market phase held). NaN-safe the same way _streak() is: a
    NaN breaks continuity in both directions (a NaN row starts a new,
    NaN-valued "streak" rather than silently bridging the run on either
    side of it) - the final `np.where(is_nan, np.nan, ...)` is what
    actually enforces that, not just the grouping itself. Fully
    vectorized, same cumulative-group-change technique as _streak()."""
    vals = labels.to_numpy()
    n = len(vals)
    if n == 0:
        return pd.Series(dtype=float, index=labels.index)
    is_nan = pd.isna(labels).to_numpy()
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = is_nan[1:] | is_nan[:-1] | (vals[1:] != vals[:-1])
    group_id = np.cumsum(change)
    position_in_group = pd.Series(np.arange(n)).groupby(group_id).cumcount().to_numpy() + 1
    run_length = np.where(is_nan, np.nan, position_in_group)
    return pd.Series(run_length, index=labels.index).astype(float)


def _level_test_history(level: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
                         atr_series: pd.Series, is_resistance: bool,
                         touch_atr_mult: float = PROXIMITY_ATR_MULT) -> tuple[pd.Series, pd.Series]:
    """(test_count, break_rate): "level memory" - how many times has THIS
    SPECIFIC confirmed swing level been tested up to AND INCLUDING the
    current candle, and what fraction of those tests closed beyond it
    (broke) rather than reversing off it (rejected)? This is a genuinely
    different question from dist_to_swing_high_atr/dist_to_swing_low_atr
    above (which only say how FAR away the level is right now) - two
    candles equally close to a level can behave completely differently
    depending on whether that level has already been tested 6 times and
    broken through 5 of them, or has never been tested at all.

    `level` is support_resistance.swing_levels()'s OWN output (already
    look-ahead-safe by construction - see that module's docstring for
    the shift(lookback)+ffill+shift(1) confirmation-lag reasoning), so
    this can never silently disagree with the rule-based system's own
    notion of "which level is this." A "level" here is a whole SEGMENT
    of consecutive candles sharing the identical confirmed swing price
    (`level` only changes value when a NEWER swing point confirms) -
    test_count/break_rate reset to fresh (0 tests, NaN rate) at the
    start of each new segment, since a brand new level has no test
    history of its own, regardless of how the PREVIOUS level (a
    different price) performed.

    "Tested" reuses support_resistance.py's OWN `approached` definition
    exactly (price's relevant extreme within touch_atr_mult * ATR of the
    level, one-sided, no upper bound - see _resistance_rejection/
    _support_bounce there) - not reimplemented, so "near a level" can
    never mean something subtly different here than it does to the
    pattern-mining side. "Broke" = this test candle's CLOSE ended up
    beyond the level - the direct inverse of _resistance_rejection/
    _support_bounce's own `held` check. NaN wherever `level` itself is
    NaN (no swing has confirmed yet this early in the history)."""
    buffer = touch_atr_mult * atr_series
    if is_resistance:
        is_test = (high >= (level - buffer)).fillna(False)
        broke = (close > level).fillna(False) & is_test
    else:
        is_test = (low <= (level + buffer)).fillna(False)
        broke = (close < level).fillna(False) & is_test

    same_as_prev = (level == level.shift(1)).fillna(False)
    both_nan = level.isna() & level.shift(1).isna()
    new_segment = ~(same_as_prev | both_nan)
    segment_id = new_segment.cumsum()

    test_count = is_test.astype(int).groupby(segment_id).cumsum()
    break_count = broke.astype(int).groupby(segment_id).cumsum()
    known = level.notna()
    test_count = test_count.where(known).astype(float)
    break_rate = (break_count / test_count.replace(0, np.nan)).where(known)
    return test_count, break_rate


def _rolling_autocorr_lag1(x: pd.Series, window: int) -> pd.Series:
    """Rolling lag-1 autocorrelation of `x`, computed via the standard
    correlation-from-rolling-moments identity (mean/var/covariance) rather
    than `.rolling(window).apply(lambda w: pd.Series(w).autocorr(), ...)`
    - the .apply() form re-runs a Python-level callback once per row,
    which is prohibitively slow at 1min's row count; this is fully
    vectorized pandas/numpy and computes the identical statistic."""
    x1 = x
    x2 = x.shift(1)
    mean1 = x1.rolling(window).mean()
    mean2 = x2.rolling(window).mean()
    cov = (x1 * x2).rolling(window).mean() - mean1 * mean2
    std1 = x1.rolling(window).std()
    std2 = x2.rolling(window).std()
    denom = (std1 * std2).replace(0, np.nan)
    return cov / denom


def compute_features(candles: pd.DataFrame) -> pd.DataFrame:
    """candles: DataFrame with [timestamp, open, high, low, close] sorted
    ascending. Returns a DataFrame of numeric features, same index as
    candles, values NaN wherever there isn't enough history yet."""
    close = candles["close"]
    out = {}

    # --- price action / candle shape (same OHLC-depth discipline as patterns.py) ---
    out["body_ratio"] = _body_ratio(candles)
    out["upper_wick_ratio"] = _upper_wick_ratio(candles)
    out["lower_wick_ratio"] = _lower_wick_ratio(candles)
    out["is_bullish"] = (candles["close"] > candles["open"]).astype(float)

    a = atr(candles)
    out["atr_norm"] = a / close  # ATR as a fraction of price - comparable across price levels/eras

    # --- returns at multiple lookback windows, ATR-normalized so a 1-hour
    # move means roughly the same thing whether volatility is high or low ---
    for w in RETURN_WINDOWS:
        out[f"return_{w}"] = (close - close.shift(w)) / close.shift(w)
        out[f"return_{w}_atr_norm"] = (close - close.shift(w)) / a.replace(0, np.nan)

    # --- moving-average distance, ATR-normalized ---
    for w in SMA_WINDOWS:
        s = sma(close, w)
        out[f"dist_sma_{w}_atr_norm"] = (close - s) / a.replace(0, np.nan)

    # --- momentum / trend-strength indicators (reused, not reimplemented) ---
    out["rsi_14"] = rsi(close, 14)
    macd_line, macd_signal = macd(close)
    out["macd_line_norm"] = macd_line / close
    out["macd_hist_norm"] = (macd_line - macd_signal) / close
    out["adx_14"] = adx(candles, 14)

    # --- Donchian channel position: where does the current close sit
    # within the recent high/low range (0 = at the low, 1 = at the high) -
    # shift(1) before the rolling max/min so the channel is built only
    # from candles strictly before this one ---
    donchian_high = candles["high"].shift(1).rolling(DONCHIAN_WINDOW).max()
    donchian_low = candles["low"].shift(1).rolling(DONCHIAN_WINDOW).min()
    donchian_range = (donchian_high - donchian_low).replace(0, np.nan)
    out["donchian_position"] = (close - donchian_low) / donchian_range

    # --- volatility regime, as a continuous ratio (more informative to a
    # model than regime.py's 3-bucket LOW/NORMAL/HIGH label - the bucket
    # boundaries exist for human-readable pattern conditioning, not for a
    # model that can use the raw ratio directly) ---
    vol_baseline = a.shift(1).rolling(VOLATILITY_BASELINE_WINDOW).mean()
    out["volatility_ratio"] = a / vol_baseline

    # --- session/time-of-day context - cyclic encoding so midnight and
    # 23:00 read as adjacent, not maximally far apart ---
    ts = pd.to_datetime(candles["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_of_week"] = ts.dt.dayofweek.astype(float)

    session_flags = detect_session_events(candles)
    for col in session_flags.columns:
        out[col] = session_flags[col].astype(float)

    # --- support/resistance proximity, ATR-normalized so distance means
    # the same thing in calm and volatile conditions - reuses
    # support_resistance.py's OWN level functions (not reimplemented
    # here) so the ML challenger's notion of "near a swing high" can
    # never silently drift from the rule-based system's mined
    # definition of the same level. Signed: positive means price is
    # still on the "normal" side of the level (below a resistance-type
    # level, above a support-type one), negative means price has
    # already pushed through it - the model can use magnitude for
    # closeness and sign for which side, rather than getting a single
    # collapsed distance-only number the way a boolean pattern trigger
    # would. Same "continuous, not bucketed" reasoning volatility_ratio
    # above already documents. ---
    swing_high, swing_low = swing_levels(candles)
    round_level = nearest_round_number(candles)
    _, pivot_r1, pivot_s1 = daily_pivots(candles)
    safe_atr = a.replace(0, np.nan)
    out["dist_to_swing_high_atr"] = (swing_high - close) / safe_atr
    out["dist_to_swing_low_atr"] = (close - swing_low) / safe_atr
    out["dist_to_round_number_atr"] = (close - round_level) / safe_atr
    out["dist_to_pivot_r1_atr"] = (pivot_r1 - close) / safe_atr
    out["dist_to_pivot_s1_atr"] = (close - pivot_s1) / safe_atr

    # --- level memory: not just HOW FAR the nearest swing level is
    # (dist_to_swing_*_atr above), but how many times it has ALREADY been
    # tested and what fraction of those tests broke through vs rejected -
    # a genuinely different question ("has the market already fought over
    # this exact price and who won") that pure distance can't answer. See
    # _level_test_history()'s own docstring for the full reasoning. ---
    resistance_test_count, resistance_break_rate = _level_test_history(
        swing_high, candles["high"], candles["low"], close, a, is_resistance=True,
    )
    support_test_count, support_break_rate = _level_test_history(
        swing_low, candles["high"], candles["low"], close, a, is_resistance=False,
    )
    out["resistance_test_count"] = resistance_test_count
    out["resistance_break_rate"] = resistance_break_rate
    out["support_test_count"] = support_test_count
    out["support_break_rate"] = support_break_rate

    # --- session opening-range breakout: is price still inside, or has it
    # broken beyond, the high/low established in the first
    # OPENING_RANGE_CANDLES candles after each session opened - a genuinely
    # different question from "is this session currently active"
    # (session_flags above) or "near a swing level" (dist_to_swing_*
    # above, which uses a fixed-window fractal, not a session-anchored
    # one). Signed the same way as the swing/pivot distances: positive =
    # still inside the range, negative = already broken through. ---
    for session_name in SESSIONS:
        open_col = f"session_{session_name}_open"
        range_high, range_low = _session_opening_range(candles, session_flags[open_col])
        out[f"dist_to_{session_name}_open_range_high_atr"] = (range_high - close) / safe_atr
        out[f"dist_to_{session_name}_open_range_low_atr"] = (close - range_low) / safe_atr

    # --- multi-candle shape/sequence features - beyond a single candle's
    # own body/wick ratios, how does a RUN of recent candles look. Signed
    # streaks (positive = N bullish/higher-highs in a row, negative = N
    # bearish/lower-lows in a row) rather than two separate unsigned
    # columns, so the model sees direction and length as one number. ---
    is_bullish_candle = candles["close"] > candles["open"]
    out["candle_direction_streak"] = (
        _streak(is_bullish_candle) - _streak(~is_bullish_candle)
    )
    # Higher-high and lower-low are NOT mutually exclusive (an outside bar
    # is both at once), unlike bullish/bearish above - so these stay two
    # separate unsigned streaks rather than one signed column.
    higher_high = candles["high"] > candles["high"].shift(1)
    lower_low = candles["low"] < candles["low"].shift(1)
    out["higher_high_streak"] = _streak(higher_high)
    out["lower_low_streak"] = _streak(lower_low)

    # Body-size trend: is the market compressing (bodies shrinking) or
    # expanding (bodies growing) over the last few candles - a slope
    # proxy (later minus earlier, divided by the window), not just the
    # single-candle body_ratio above. ATR-normalized so it means the same
    # thing across price levels/eras, same convention as everything else
    # in this file.
    body_size = (candles["close"] - candles["open"]).abs()
    for w in BODY_TREND_WINDOWS:
        out[f"body_size_trend_{w}"] = (body_size - body_size.shift(w)) / w / safe_atr

    # Overlap between this candle's range and the PREVIOUS candle's range -
    # low overlap after a big move signals a clean break; high overlap
    # signals chop/indecision. Clipped at 0 (no negative "overlap").
    prev_high, prev_low = candles["high"].shift(1), candles["low"].shift(1)
    overlap = (
        pd.concat([candles["high"], prev_high], axis=1).min(axis=1)
        - pd.concat([candles["low"], prev_low], axis=1).max(axis=1)
    ).clip(lower=0)
    out["candle_overlap_atr"] = overlap / safe_atr

    # Opening gap vs the PRIOR candle's close, normalized by the ATR known
    # AT that prior candle's close (not this candle's own ATR, which
    # wouldn't be fully known until this candle's range is in - a subtle
    # but real look-ahead-adjacent distinction, same reasoning the
    # Donchian channel above already applies via shift(1)).
    out["gap_atr_norm"] = (candles["open"] - candles["close"].shift(1)) / a.shift(1).replace(0, np.nan)

    # Is THIS candle's own range unusually large or small relative to its
    # own ATR - distinct from volatility_ratio (which compares the ATR
    # itself, a smoothed measure, against its own baseline).
    out["range_vs_atr"] = (candles["high"] - candles["low"]) / safe_atr

    # --- volume / order flow. "volume" (ask+bid ticks summed per candle)
    # is always present post-resampling; degrades to all-NaN (not a
    # crash) if it isn't, e.g. a hand-built candle frame in a test that
    # omits it. ---
    if "volume" in candles.columns:
        vol = candles["volume"].astype(float)
        vol_baseline = vol.shift(1).rolling(VOLATILITY_BASELINE_WINDOW).mean()
        out["volume_ratio"] = vol / vol_baseline.replace(0, np.nan)
        out["volume_percentile"] = vol.rolling(VOLATILITY_BASELINE_WINDOW).rank(pct=True)
        for w in VOLUME_TREND_WINDOWS:
            out[f"volume_trend_{w}"] = (vol - vol.shift(w)) / w / vol_baseline.replace(0, np.nan)
    else:
        out["volume_ratio"] = np.nan
        out["volume_percentile"] = np.nan
        for w in VOLUME_TREND_WINDOWS:
            out[f"volume_trend_{w}"] = np.nan

    # --- ask/bid volume imbalance and tick-count activity. Only present
    # in candle files backfilled via scripts/backfill_order_flow.py (or
    # produced by build_history.py after ask_volume/bid_volume/tick_count
    # were added to resample_ticks()) - degrades to all-NaN, same
    # convention as "volume" above, for any older candle file that hasn't
    # been backfilled yet. Honest framing: Dukascopy's ask_volume/
    # bid_volume are QUOTED liquidity at each tick, not a trade's actual
    # aggressor side - this is a relative-liquidity signal ("was more
    # size showing on the ask or the bid side while this candle formed"),
    # not a confirmed buy/sell trade-flow measure the way a exchange-
    # reported tape would give. ---
    if "ask_volume" in candles.columns and "bid_volume" in candles.columns:
        ask_vol = candles["ask_volume"].astype(float)
        bid_vol = candles["bid_volume"].astype(float)
        total_ab = (ask_vol + bid_vol).replace(0, np.nan)
        out["ask_bid_volume_imbalance"] = (ask_vol - bid_vol) / total_ab
    else:
        out["ask_bid_volume_imbalance"] = np.nan

    if "tick_count" in candles.columns:
        tick_count = candles["tick_count"].astype(float)
        tick_baseline = tick_count.shift(1).rolling(VOLATILITY_BASELINE_WINDOW).mean()
        out["tick_count_ratio"] = tick_count / tick_baseline.replace(0, np.nan)
    else:
        out["tick_count_ratio"] = np.nan

    # --- volatility percentile: where does the CURRENT ATR rank within
    # its own trailing history (rank-based, robust to one outlier period
    # skewing a simple ratio) - complements volatility_ratio above rather
    # than replacing it. rolling(...).rank(pct=True) only ever looks at
    # values up to and including the current row, so this is exactly as
    # look-ahead-safe as volatility_ratio, just a different statistic. ---
    out["volatility_percentile"] = a.rolling(VOLATILITY_BASELINE_WINDOW).rank(pct=True)

    # --- calendar richness beyond hour-of-day/day-of-week: month-of-year
    # position (cyclic) and month/quarter-end proximity, both real,
    # recurring flow effects (rebalancing, month-end positioning) that
    # hour/day-of-week alone can't capture. ---
    day_of_month = ts.dt.day.astype(float)
    days_in_month = ts.dt.days_in_month.astype(float)
    out["day_of_month_sin"] = np.sin(2 * np.pi * day_of_month / days_in_month)
    out["day_of_month_cos"] = np.cos(2 * np.pi * day_of_month / days_in_month)
    out["is_month_end"] = ((days_in_month - day_of_month) <= 2).astype(float)
    out["is_quarter_end"] = (out["is_month_end"].astype(bool) & ts.dt.month.isin([3, 6, 9, 12])).astype(float)

    # --- statistical/regime features ---
    # Rolling lag-1 autocorrelation of returns: persistently positive =
    # trending/momentum regime, persistently negative/near-zero = choppy/
    # mean-reverting - a genuinely different signal from any single
    # indicator above, computed on the SAME return series (not
    # ATR-normalized, autocorrelation is scale-invariant already).
    returns = close.pct_change()
    out["return_autocorr_50"] = _rolling_autocorr_lag1(returns, AUTOCORR_WINDOW)

    # Kaufman's Efficiency Ratio: net directional move over a window
    # divided by the total path length traveled to get there - 1.0 means
    # price moved in a straight line (pure trend), near 0 means it
    # churned back and forth to end up roughly where it started (pure
    # noise/mean-reversion). A standard, cheap, fully vectorized proxy for
    # "is this a trending or mean-reverting regime" - not a full Hurst-
    # exponent estimate (which needs a slower log-log regression across
    # multiple sub-window sizes), but the same underlying question.
    net_change = (close - close.shift(EFFICIENCY_RATIO_WINDOW)).abs()
    path_length = close.diff().abs().rolling(EFFICIENCY_RATIO_WINDOW).sum()
    out["efficiency_ratio_20"] = net_change / path_length.replace(0, np.nan)

    # --- regime / market phase (Wyckoff-style, causal proxy built from
    # features already computed above, not a new indicator) ---
    # Trending vs ranging comes from efficiency_ratio_20 (net directional
    # move vs total path length - see its own comment above); WHICH SIDE
    # of the recent range a ranging period sits on (donchian_position,
    # 0 = at the recent low, 1 = at the recent high, computed earlier in
    # this function) is what distinguishes accumulation (ranging near
    # recent lows) from distribution (ranging near recent highs) - the
    # same distinction a chart-reader means by those Wyckoff terms,
    # expressed as two already-computed numbers rather than a new one.
    # Genuinely different information from efficiency_ratio_20 alone: two
    # candles with IDENTICAL trend strength can be in opposite phases
    # depending on where in the broader range they currently sit.
    #
    # `known` guards against NaN inputs (the shared warm-up period)
    # reading as false "ranging" - a bare `~trending` would otherwise
    # treat "trend strength is unknown yet" the same as "genuinely not
    # trending," which is wrong (NaN comparisons evaluate False in
    # pandas, so `trending` itself is already correctly False during
    # warm-up, but plain `~trending` would then be True - `known` is
    # what stops that True from cascading into a false accumulation/
    # distribution label).
    known = out["efficiency_ratio_20"].notna() & out["donchian_position"].notna() & out["return_20"].notna()
    trending = known & (out["efficiency_ratio_20"] >= EFFICIENCY_TREND_THRESHOLD)
    trending_up = trending & (out["return_20"] > 0)
    trending_down = trending & (out["return_20"] <= 0)
    ranging = known & ~trending
    near_range_low = out["donchian_position"] < 0.5
    accumulation = ranging & near_range_low
    distribution = ranging & ~near_range_low

    out["regime_markup"] = trending_up.astype(float)
    out["regime_markdown"] = trending_down.astype(float)
    out["regime_accumulation"] = accumulation.astype(float)
    out["regime_distribution"] = distribution.astype(float)

    regime_label = pd.Series(np.select(
        [trending_up.to_numpy(), trending_down.to_numpy(), accumulation.to_numpy(), distribution.to_numpy()],
        ["markup", "markdown", "accumulation", "distribution"],
        default=None,
    ), index=candles.index)
    # How long (in candles) has the CURRENT regime held - a genuinely
    # different signal from efficiency_ratio_20's instantaneous reading:
    # a market 40 candles into a markup phase behaves differently from
    # one that just flipped into markup last candle, even at an
    # identical efficiency_ratio_20 value right now.
    out["regime_duration"] = _categorical_run_length(regime_label)

    features = pd.DataFrame(out, index=candles.index)
    return features


FEATURE_COLUMNS = list(compute_features(pd.DataFrame({
    "timestamp": pd.date_range("2020-01-01", periods=250, freq="1h"),
    "open": 1900.0, "high": 1901.0, "low": 1899.0, "close": 1900.0,
    "volume": 100.0, "ask_volume": 55.0, "bid_volume": 45.0, "tick_count": 20.0,
})).columns)


# ---- cross-timeframe context (optional, separate from single-timeframe
# compute_features() above so every existing caller that doesn't pass
# coarser-timeframe data is completely unaffected - a genuinely fine
# entry (5m) can see genuinely coarser context (15m/1h/4h/...) without
# either timeframe's OWN feature computation ever changing shape) ----

# A deliberately SMALL, meaningful subset - not all 65+ single-timeframe
# features re-imported per coarser timeframe, which would 2-3x the total
# feature count for mostly-redundant information. Trend, momentum, trend
# STRENGTH, two different volatility-regime views, structure/range
# position, and trend-vs-noise - each answers a genuinely different
# "what is the coarser timeframe doing right now" question.
CROSS_TF_CONTEXT_COLUMNS = [
    "dist_sma_50_atr_norm", "rsi_14", "adx_14",
    "volatility_ratio", "volatility_percentile",
    "donchian_position", "efficiency_ratio_20",
]


def compute_cross_timeframe_features(
        fine_candles: pd.DataFrame,
        coarser_candles_by_tf: "dict[str, tuple[pd.DataFrame, float]]") -> pd.DataFrame:
    """For every (label -> (that timeframe's OWN candles, its duration in
    minutes)) entry in `coarser_candles_by_tf`, computes that coarser
    timeframe's OWN compute_features() (self-contained - a coarser
    timeframe's features never depend on the fine timeframe, so there is
    no recursion risk) and merges CROSS_TF_CONTEXT_COLUMNS onto
    `fine_candles`' own timestamps.

    Look-ahead safety is the entire point of this function, so it's
    worth being explicit: the merge key on the coarse side is each candle's
    CLOSE time (its own open timestamp + its duration), not its open
    time - a coarse candle's OHLC-derived features aren't actually KNOWN
    until that candle has finished forming. `pd.merge_asof(...,
    direction="backward")` then matches each fine-timeframe row to the
    most recent coarse candle whose close time is <= that fine row's OWN
    open time - the same "only look back, only at what's already
    finished" discipline every other cross-timeframe/cross-source join in
    this codebase already uses (event_timing.py's fundamentals-to-candles
    merge, discover_patterns.py's own cross-timeframe confirmation).

    Column names are prefixed `ctx_<label>_<feature>` (e.g. `ctx_1h_rsi_14`)
    so a model's feature table can carry context from several coarser
    timeframes side by side without any name collisions. Returns an
    empty (0-column) DataFrame, same index as fine_candles, if
    `coarser_candles_by_tf` is empty - e.g. 1d has no coarser sibling at
    all, same "nothing to check, not an error" convention discover_
    patterns.py's own cross-timeframe check already established."""
    if not coarser_candles_by_tf:
        return pd.DataFrame(index=fine_candles.index)

    fine_ts = pd.to_datetime(fine_candles["timestamp"]).to_numpy()
    order = np.argsort(fine_ts, kind="stable")
    inverse_order = np.argsort(order, kind="stable")
    fine_ts_sorted = fine_ts[order]

    frames = []
    for label, (coarse_candles, duration_minutes) in coarser_candles_by_tf.items():
        coarse_features = compute_features(coarse_candles)
        coarse_close_time = (
            pd.to_datetime(coarse_candles["timestamp"]) + pd.to_timedelta(duration_minutes, unit="m")
        ).to_numpy()

        lookup = coarse_features[CROSS_TF_CONTEXT_COLUMNS].reset_index(drop=True)
        lookup_order = np.argsort(coarse_close_time, kind="stable")
        lookup_sorted = lookup.iloc[lookup_order].reset_index(drop=True)
        close_time_sorted = coarse_close_time[lookup_order]

        merged = pd.merge_asof(
            pd.DataFrame({"_ts": fine_ts_sorted}), lookup_sorted.assign(_close_time=close_time_sorted),
            left_on="_ts", right_on="_close_time", direction="backward",
        )
        merged = merged[CROSS_TF_CONTEXT_COLUMNS].iloc[inverse_order].reset_index(drop=True)
        merged.index = fine_candles.index
        frames.append(merged.add_prefix(f"ctx_{label}_"))

    return pd.concat(frames, axis=1)
