"""
Forward-looking news calendar: for signals, we don't just need "did news
happen" (that's build_fundamentals.py, historical), we need "is
high-impact news SCHEDULED before this trade is expected to resolve" -
so a live signal can be labeled with what's coming, before it happens.

FRED/ALFRED tracks release calendars ahead of time - release_dates(...,
include_future=True) returns dates that don't have data yet, i.e.
already-announced future release dates (see fred_client.py). This reuses
that, plus event_timing.py's impact classification, to build
data/events/upcoming.parquet and to answer "what high-impact news falls
inside [window_start, window_end]" for both:
  - live signals (signal_engine.py): what's coming before this trade
    would typically resolve
  - historical mining (build_pattern_library.py): did news actually fall
    inside each past occurrence's entry->resolution window, so we can
    report a news-conditioned win rate separately from the baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from event_timing import RELEASE_TIME_ET, et_to_utc, impact_of
from fred_client import release_dates
from heartbeat import track, write_heartbeat


def build_upcoming_calendar(config_path: Path, data_dir: Path) -> Path | None:
    """Fetch the forward-looking release schedule for every configured
    event and write data/events/upcoming.parquet. Safe to re-run often -
    it's a small, cheap fetch (just dates, no series values)."""
    hb_path = data_dir / "heartbeats.json"
    if not config_path.exists():
        write_heartbeat("news_calendar", "skipped", detail="event_config.json not found", path=hb_path)
        return None

    with track("news_calendar", path=hb_path):
        config = json.loads(config_path.read_text())
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)

        rows = []
        for event_type, cfg in config.items():
            if event_type.startswith("_"):
                continue
            release_id = cfg.get("release_id")
            if release_id is None:
                continue
            dates = release_dates(release_id, include_future=True)
            future = dates[dates["date"] >= now.normalize()]
            hour, minute = RELEASE_TIME_ET.get(event_type, (8, 30))
            for d in future["date"]:
                rows.append({
                    "datetime_utc": et_to_utc(d, hour, minute),
                    "event_type": event_type,
                    "impact": impact_of(event_type),
                })

        events_dir = data_dir / "events"
        events_dir.mkdir(parents=True, exist_ok=True)
        out_path = events_dir / "upcoming.parquet"
        df = pd.DataFrame(rows).sort_values("datetime_utc") if rows else pd.DataFrame(
            columns=["datetime_utc", "event_type", "impact"])
        df.to_parquet(out_path, index=False)
        print(f"wrote {len(df)} upcoming scheduled events -> {out_path}")
        return out_path


def load_upcoming(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "events" / "upcoming.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["datetime_utc", "event_type", "impact"])
    return pd.read_parquet(path)


def events_in_window(events: pd.DataFrame, start, end, impact: str | None = None) -> pd.DataFrame:
    """Events (upcoming or historical fundamentals) whose datetime_utc
    falls in [start, end]. impact="high" filters to high-impact only."""
    if events is None or events.empty:
        return events if events is not None else pd.DataFrame()
    mask = (events["datetime_utc"] >= start) & (events["datetime_utc"] <= end)
    sub = events[mask]
    if impact is not None and "impact" in sub.columns:
        sub = sub[sub["impact"] == impact]
    elif impact is not None and "event_type" in sub.columns:
        sub = sub[sub["event_type"].apply(impact_of) == impact]
    return sub.sort_values("datetime_utc")


def label_signal_news_risk(now: pd.Timestamp, median_candles_to_resolve: float | None,
                            timeframe_minutes: float, upcoming: pd.DataFrame) -> dict:
    """For a live signal: is high-impact news scheduled before this trade
    is expected to resolve? Returns a dict ready to attach to the
    signal's trade_plan. `median_candles_to_resolve` comes from the
    pattern's mined stats (risk_reward.summarize_trades); if unknown,
    we can't estimate a resolution window and say so explicitly rather
    than guessing."""
    if median_candles_to_resolve is None:
        return {"window_known": False, "high_impact_events": [], "note":
                "no median resolution time available for this pattern - can't estimate the trade's news window"}

    window_end = now + pd.Timedelta(minutes=median_candles_to_resolve * timeframe_minutes)
    hits = events_in_window(upcoming, now, window_end, impact="high")

    events_out = []
    for _, row in hits.iterrows():
        hours_into_trade = (row["datetime_utc"] - now).total_seconds() / 3600
        events_out.append({
            "event_type": row["event_type"],
            "datetime_utc": row["datetime_utc"].isoformat(),
            "hours_from_now": round(hours_into_trade, 1),
        })

    return {
        "window_known": True,
        "expected_resolution_utc": window_end.isoformat(),
        "expected_resolution_hours": round((window_end - now).total_seconds() / 3600, 1),
        "high_impact_events": events_out,
        "news_in_window": len(events_out) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="event_config.json")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    build_upcoming_calendar(Path(args.config), Path(args.data_dir))


if __name__ == "__main__":
    main()
