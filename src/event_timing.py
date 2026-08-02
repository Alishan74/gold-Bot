"""
Shared release-time convention and impact classification for the
fundamental events this system tracks. Used by both build_fundamentals.py
(historical mining) and news_calendar.py (forward-looking live labeling)
so the two never drift out of sync on what time an event actually posts.
"""
from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# BLS/BEA releases (CPI/PCE/NFP/GDP) publish 8:30 AM Eastern by
# long-standing convention; FOMC decisions announce 2:00 PM Eastern.
RELEASE_TIME_ET = {
    "CPI": (8, 30),
    "PCE": (8, 30),
    "NFP": (8, 30),
    "GDP": (8, 30),
    "FOMC": (14, 0),
}

# All five tracked events were deliberately chosen (see README) BECAUSE
# they're the highest-impact scheduled releases for gold/USD - that was
# the whole point of scoping to "core drivers only" instead of a full
# economic calendar. So all five are "high" impact; session opens/closes
# (session_patterns.py) are lower-impact liquidity/volatility events, not
# directional macro surprises - labeled "normal".
EVENT_IMPACT = {
    "CPI": "high", "PCE": "high", "NFP": "high", "GDP": "high", "FOMC": "high",
}


def et_to_utc(date: pd.Timestamp, hour: int, minute: int) -> pd.Timestamp:
    local = pd.Timestamp(date.year, date.month, date.day, hour, minute, tz=EASTERN)
    return local.tz_convert(UTC).tz_localize(None)


def impact_of(event_type: str) -> str:
    return EVENT_IMPACT.get(event_type, "normal")
