"""
Trading-session open/close as mineable patterns. Unlike fundamentals,
these need no external data source - session times are a fixed daily
convention, so the UTC instant of "London open" on any given date is
pure calendar math (with proper DST handling per session's own
timezone). That also means this runs fully offline, including in a
network-restricted sandbox.

Gold sees real, recurring volatility patterns around session
opens/closes (liquidity stepping in/out) - conventionally traded on, but
whether it's actually a statistical edge on this specific instrument is
exactly what mining is for. Every session pattern here is direction-
ambiguous (hint=0) for that reason, same as the fundamental events.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

UTC = ZoneInfo("UTC")

# (session name, timezone, open (h,m), close (h,m)) - conventional cash/FX
# session hours. Weekends are naturally skipped since gold doesn't trade
# then and no candle will ever match those timestamps.
SESSIONS = {
    "sydney":   {"tz": "Australia/Sydney", "open": (9, 0), "close": (17, 0)},
    "tokyo":    {"tz": "Asia/Tokyo", "open": (9, 0), "close": (17, 0)},
    "london":   {"tz": "Europe/London", "open": (8, 0), "close": (16, 30)},
    "new_york": {"tz": "America/New_York", "open": (8, 0), "close": (17, 0)},
}

SESSION_PATTERN_NAMES = [
    f"session_{name}_{edge}" for name in SESSIONS for edge in ("open", "close")
]


def _local_to_utc(date: pd.Timestamp, hour: int, minute: int, tz_name: str) -> pd.Timestamp:
    local = pd.Timestamp(date.year, date.month, date.day, hour, minute, tz=ZoneInfo(tz_name))
    return local.tz_convert(UTC).tz_localize(None)


def _session_boundary_timestamps(start: pd.Timestamp, end: pd.Timestamp,
                                  hour: int, minute: int, tz_name: str) -> pd.Series:
    days = pd.date_range(start.normalize() - pd.Timedelta(days=1), end.normalize() + pd.Timedelta(days=1), freq="1D")
    return pd.Series([_local_to_utc(d, hour, minute, tz_name) for d in days])


def active_sessions_at(ts_utc: pd.Timestamp) -> list[str]:
    """Which of the 4 sessions are currently OPEN (a live span, not a
    boundary instant) at `ts_utc` - answers "was this candle during
    active London hours", not "did a session open/close exactly on this
    candle" (that's what SESSION_PATTERN_NAMES/detect_session_events()
    above answer, and they're a genuinely different question - a trade
    can fire hours into an active session, long after its own open
    boundary candle). Used for live context-attribution
    (signal_journal.py's confluence/session tracking), not mining.
    Returns 0 names (off-hours, e.g. between New York close and Sydney
    open), 1, or more than 1 (session overlaps, e.g. London + New York)
    - never assume exactly one. `ts_utc` may be naive (assumed UTC, the
    convention every candle timestamp in this codebase already uses) or
    tz-aware."""
    ts = pd.Timestamp(ts_utc)
    ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)

    active = []
    for name, cfg in SESSIONS.items():
        local = ts.tz_convert(ZoneInfo(cfg["tz"]))
        local_minutes = local.hour * 60 + local.minute
        open_h, open_m = cfg["open"]
        close_h, close_m = cfg["close"]
        if open_h * 60 + open_m <= local_minutes < close_h * 60 + close_m:
            active.append(name)
    return active


def detect_session_events(candles: pd.DataFrame) -> pd.DataFrame:
    """Returns a boolean DataFrame aligned to candles.index, one column
    per SESSION_PATTERN_NAMES entry - same shape/contract as
    patterns.detect_all() and fundamental_patterns.detect_fundamental_events(),
    so it concatenates directly alongside them."""
    out = pd.DataFrame(False, index=candles.index, columns=SESSION_PATTERN_NAMES)
    if candles.empty:
        return out

    cd = (
        candles[["timestamp"]]
        .reset_index()
        .rename(columns={"index": "candle_idx"})
        .sort_values("timestamp")
    )
    start, end = candles["timestamp"].min(), candles["timestamp"].max()

    for name, cfg in SESSIONS.items():
        for edge, (hour, minute) in (("open", cfg["open"]), ("close", cfg["close"])):
            boundaries = _session_boundary_timestamps(start, end, hour, minute, cfg["tz"])
            boundaries = boundaries.sort_values().reset_index(drop=True)
            # Boundaries computed before the data's own start are only
            # there to fill a 1-day lookback so we don't miss one near
            # the edge - but forward-matching them still lands on candle
            # 0, which would falsely stack every session's boundary onto
            # the dataset's very first candle. Drop anything that fell
            # before the actual candle range - it isn't really "this
            # session opened around this candle" for our data window.
            boundaries = boundaries[boundaries >= start]
            merged = pd.merge_asof(
                boundaries.to_frame("boundary_utc"), cd,
                left_on="boundary_utc", right_on="timestamp", direction="forward",
            ).dropna(subset=["candle_idx"])
            idx = merged["candle_idx"].astype(int)
            out.loc[out.index.isin(idx), f"session_{name}_{edge}"] = True

    return out
