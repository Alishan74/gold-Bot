"""
Data-quality auditing for the historical candle series: surfaces exactly
what "how reliable is this data" actually needs answered, instead of
leaving backfill failures buried in build_history.py's stdout log lines
that nobody's watching hours into an unattended 20-year run.

Two distinct things "reliable" means here:
  1. Every hour that SHOULD have data (a real trading hour) actually got
     some - tracked via build_history.py's fetch failures (network/format
     errors - see dukascopy_fetch.DukascopyFormatError), which are a
     different thing from the EXPECTED empty-hour case (weekend/holiday,
     a clean 404 - not a problem, not counted as a failure).
  2. The resulting candle series has no unexplained holes - gaps between
     consecutive candles bigger than a normal weekend/holiday close would
     produce, on any given timeframe. A failed-hours count of zero
     doesn't guarantee this on its own (e.g. a whole day's worth of
     hours all independently succeeding as "empty" during what should
     have been a live trading day would never show up as a fetch
     failure) - gap detection is the independent check for that.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from atomic_io import atomic_write_text

# Gold trades ~23/5 - the only routine gap is the weekly close, roughly
# Friday ~21:00 UTC to Sunday ~22:00 UTC (~49h). EXPECTED_MAX_WEEKEND_GAP
# is that plus a buffer for late Sunday-open data, NOT padded out to cover
# every possible multi-day holiday weekend - a 3-4 day holiday closure
# WILL get flagged here too. That's a deliberate call: this has no
# trading-holiday calendar to consult, and a false positive on a known
# holiday (a human glances at the gap list and recognizes "that's
# Christmas") is a far cheaper mistake than a false negative that hides a
# genuine multi-day data outage inside a threshold sized to wave through
# any gap "long enough to plausibly be a holiday."
#
# A gap counts as anomalous if it clears BOTH this absolute floor AND a
# multiple of the timeframe's own candle spacing - the multiple-of-
# timeframe half is what catches a shorter-but-still-abnormal hole on a
# FAST timeframe (e.g. a broken 6-hour stretch of the 1min feed) that
# would otherwise be silently smaller than the weekend-sized floor.
EXPECTED_MAX_WEEKEND_GAP = pd.Timedelta(hours=56)
GAP_MULTIPLE_OF_TIMEFRAME = 3


def detect_gaps(candles: pd.DataFrame, timeframe_minutes: float) -> list[dict]:
    """Consecutive-candle gaps exceeding the threshold above. Returns a
    list of {"from", "to", "gap_hours"} dicts, oldest first."""
    if candles.empty or len(candles) < 2:
        return []
    ts = candles["timestamp"].sort_values().reset_index(drop=True)
    deltas = ts.diff()
    expected = pd.Timedelta(minutes=timeframe_minutes)
    threshold = max(EXPECTED_MAX_WEEKEND_GAP, expected * GAP_MULTIPLE_OF_TIMEFRAME)

    gaps = []
    for i in range(1, len(ts)):
        delta = deltas.iloc[i]
        if delta > threshold:
            gaps.append({
                "from": ts.iloc[i - 1].isoformat(),
                "to": ts.iloc[i].isoformat(),
                "gap_hours": round(delta.total_seconds() / 3600, 1),
            })
    return gaps


def build_report(data_dir: Path, symbol: str, timeframe_minutes: dict[str, int],
                  failed_hours: list[str] | None = None,
                  hours_requested: int | None = None,
                  hours_with_data: int | None = None) -> dict:
    """Full report: backfill failure counts (if build_history.py just ran
    and passed them in) plus a fresh gap scan over whatever candle files
    currently exist on disk - the gap scan runs regardless of whether
    fresh failure counts are available, so `python -m data_quality` (or
    the dashboard) can always answer "does the data on disk look OK
    right now", not just "did the last backfill run cleanly"."""
    candles_dir = data_dir / "candles"
    report: dict = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "backfill": {
            "hours_requested": hours_requested,
            "hours_with_data": hours_with_data,
            "hours_failed": len(failed_hours) if failed_hours else 0,
            "failed_hours_sample": (failed_hours or [])[:50],
        },
        "gaps_by_timeframe": {},
    }
    for tf, minutes in timeframe_minutes.items():
        path = candles_dir / f"{symbol}_{tf}.parquet"
        if not path.exists():
            continue
        candles = pd.read_parquet(path)
        report["gaps_by_timeframe"][tf] = detect_gaps(candles, minutes)
    return report


def write_report(report: dict, data_dir: Path) -> Path:
    out_path = data_dir / "data_quality_report.json"
    atomic_write_text(out_path, json.dumps(report, indent=2, default=str))
    return out_path


def load_report(data_dir: Path) -> dict | None:
    path = data_dir / "data_quality_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
