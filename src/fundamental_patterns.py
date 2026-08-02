"""
Turn dated fundamental events (from build_fundamentals.py) into pattern
flags aligned to a specific candle series - i.e. figure out exactly which
candle was live when each event hit - so fundamentals plug into the same
mining and hard-gate machinery as candlestick/indicator patterns
(patterns.py, risk_reward.py) instead of being a separate, looser system.

Every fundamental reading here is direction-ambiguous by design: we don't
assume from macro theory whether, say, a hot CPI print is bullish or
bearish for gold (that relationship has flipped across regimes over the
last 10 years). The 1:4 R:R win-rate mining decides empirically, exactly
like it does for `doji` / `inside_bar` in patterns.py.
"""
from __future__ import annotations

import pandas as pd

FUNDAMENTAL_PATTERN_NAMES = [
    "fundamental_cpi_accelerating", "fundamental_cpi_decelerating",
    "fundamental_pce_accelerating", "fundamental_pce_decelerating",
    "fundamental_nfp_accelerating", "fundamental_nfp_decelerating",
    "fundamental_gdp_accelerating", "fundamental_gdp_decelerating",
    "fundamental_fomc_hike", "fundamental_fomc_cut", "fundamental_fomc_hold",
]

_MACRO_SHORT_NAMES = {"CPI": "cpi", "PCE": "pce", "NFP": "nfp", "GDP": "gdp"}


def _map_events_to_candles(events: pd.DataFrame, candles: pd.DataFrame) -> pd.Series:
    """For each event, the row-position of the first candle whose open
    time is at or after the event's UTC datetime - the candle the market
    reaction actually shows up in. `candles` must have a plain 0..n-1
    RangeIndex (call .reset_index(drop=True) before passing it in)."""
    ev = events.sort_values("datetime_utc").reset_index(drop=True)
    cd = (
        candles[["timestamp"]]
        .reset_index()
        .rename(columns={"index": "candle_idx"})
        .sort_values("timestamp")
    )
    merged = pd.merge_asof(
        ev, cd, left_on="datetime_utc", right_on="timestamp", direction="forward",
    )
    return merged["candle_idx"]


def detect_fundamental_events(candles: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Returns a boolean DataFrame aligned to candles.index, one column
    per fundamental pattern in FUNDAMENTAL_PATTERN_NAMES - same shape and
    contract as patterns.detect_all(), so the two can be concatenated."""
    out = pd.DataFrame(False, index=candles.index, columns=FUNDAMENTAL_PATTERN_NAMES)
    if events is None or events.empty:
        return out

    events = events.copy()
    events["candle_idx"] = _map_events_to_candles(events, candles)
    events = events.dropna(subset=["candle_idx"])
    events["candle_idx"] = events["candle_idx"].astype(int)

    # Columns are only guaranteed to exist if the corresponding event type
    # is actually present in `events` (build_fundamentals.py may be run
    # with only some event types configured, e.g. FOMC's release_id still
    # unset) - guard each block rather than assuming the full schema.
    if "vs_trend" in events.columns:
        for event_type, short in _MACRO_SHORT_NAMES.items():
            sub = events[events["event_type"] == event_type]
            accel_idx = sub.loc[sub["vs_trend"] > 0, "candle_idx"]
            decel_idx = sub.loc[sub["vs_trend"] < 0, "candle_idx"]
            out.loc[out.index.isin(accel_idx), f"fundamental_{short}_accelerating"] = True
            out.loc[out.index.isin(decel_idx), f"fundamental_{short}_decelerating"] = True

    if "fomc_direction" in events.columns:
        fomc = events[events["event_type"] == "FOMC"]
        for direction in ["hike", "cut", "hold"]:
            idx = fomc.loc[fomc["fomc_direction"] == direction, "candle_idx"]
            out.loc[out.index.isin(idx), f"fundamental_fomc_{direction}"] = True

    return out
