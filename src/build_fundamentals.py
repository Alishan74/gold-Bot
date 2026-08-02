"""
Build the fundamentals event table: for CPI, PCE, NFP, GDP and FOMC rate
decisions, pull every historical release over the backfill window and
record WHEN it happened (exact UTC datetime, not just a date) and WHAT it
showed (value, change from prior, direction), so it can be aligned to the
exact candle that was live at that instant.

Also pulls two continuous background series gold actually trades off of:
the US Dollar Index (DXY) and 10-year real yields.

Timing methodology (the "what time" the user needs, since FRED gives
dates but not intraday times):
  - CPI / PCE / NFP / GDP are BLS/BEA releases, published 8:30 AM Eastern
    Time by long-standing convention.
  - FOMC rate decisions are announced 2:00 PM Eastern Time.
  These are converted to UTC via America/New_York, which correctly
  handles the EST/EDT boundary across all 10 years - NOT a fixed UTC
  offset, which would be off by an hour half the year.

Which release date goes with which data point: each FRED release_id
returns a plain list of calendar publication dates; each series has its
own list of (period, value) observations. We pair them by matching every
observation to the closest release date ON OR AFTER that observation's
reference period ends (pandas merge_asof, forward). This handles GDP
correctly too - GDP gets 3 release dates per quarter (advance/second/
third estimate), and "closest date after quarter-end" naturally picks
the advance estimate release, which is the market-moving one.

IMPORTANT CAVEAT: FRED gives us the actual published VALUE, not the
Wall-Street CONSENSUS FORECAST at the time. We do not have a "beat vs
miss" figure. Instead, "hot/cool" or "accelerating/decelerating" is
defined here as the change relative to this series' OWN recent trend
(e.g. this month's YoY change vs the trailing 3-month average YoY
change) - a legitimate, fully data-derived reading, but explicitly NOT
the same thing as "beat consensus." Don't confuse the two.

LOOK-AHEAD FIX (point-in-time vintages, not latest-revised): CPI/PCE/
NFP/GDP all get revised after initial publication - NFP and GDP
routinely and sometimes materially so. `value`/`change`/`vs_trend`
below are computed from series_observations_as_first_published()
(fred_client.py) - the ALFRED vintage AS IT WAS ORIGINALLY REPORTED on
the release date, not whatever FRED's database says today after every
subsequent revision. Getting this wrong would mean mining a pattern's
win rate using information that didn't exist yet at the time the trade
would have been entered - a real, if usually small, look-ahead bias.
FOMC's rate-decision series (build_fomc_event, below) is unaffected by
this and deliberately still uses the plain series_observations() - a
policy rate isn't "revised" after the fact the way a survey-based
economic statistic is; what the Fed announced on a given date is a
historical fact, not a preliminary estimate.

Requires: export FRED_API_KEY=... (see fred_client.py) and a filled-in
event_config.json (see discover_release_ids.py). Makes real network
calls - run locally, not in a network-restricted sandbox.

Usage:
    python src/build_fundamentals.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from event_timing import RELEASE_TIME_ET, et_to_utc
from fred_client import release_dates, series_observations, series_observations_as_first_published
from heartbeat import track, write_heartbeat

# Continuous background series - not point events, just daily context.
CONTEXT_SERIES = {
    "dxy": "DTWEXBGS",        # Trade-weighted broad US dollar index
    "real_yield_10y": "DFII10",  # 10-year TIPS yield
}


def _align_observations_to_releases(obs: pd.DataFrame, releases: pd.DataFrame) -> pd.DataFrame:
    """For each observation (a reference period + value), find the actual
    calendar date it was first published: the closest release date on or
    after the period date."""
    obs = obs.sort_values("date").reset_index(drop=True)
    releases = releases.sort_values("date").reset_index(drop=True)
    merged = pd.merge_asof(
        obs, releases.rename(columns={"date": "release_date"}),
        left_on="date", right_on="release_date", direction="forward",
    )
    return merged.dropna(subset=["release_date"])


def build_macro_event(event_type: str, release_id: int, series_id: str) -> pd.DataFrame:
    releases = release_dates(release_id)
    # As-first-published, NOT today's latest-revised number - see the
    # "LOOK-AHEAD FIX" module docstring section above.
    obs = series_observations_as_first_published(series_id)
    aligned = _align_observations_to_releases(obs, releases)

    aligned["prior_value"] = aligned["value"].shift(1)
    aligned["change"] = aligned["value"] - aligned["prior_value"]
    aligned["pct_change"] = aligned["change"] / aligned["prior_value"].abs()
    # "vs own recent trend" reading (see module docstring caveat) - NOT
    # vs consensus forecast, which FRED doesn't provide.
    trailing_avg_change = aligned["change"].rolling(3).mean().shift(1)
    aligned["vs_trend"] = aligned["change"] - trailing_avg_change

    hour, minute = RELEASE_TIME_ET[event_type]
    aligned["datetime_utc"] = aligned["release_date"].apply(lambda d: et_to_utc(d, hour, minute))

    aligned["event_type"] = event_type
    return aligned[[
        "datetime_utc", "event_type", "date", "value", "prior_value",
        "change", "pct_change", "vs_trend",
    ]].rename(columns={"date": "reference_period"})


def build_fomc_event(series_id: str) -> pd.DataFrame:
    """FOMC rate DECISIONS - i.e. actual hikes/cuts - derived directly from
    where the target-range series itself changes value, NOT from a FRED
    "release" calendar. Originally this used release_dates() against
    FRED's "FOMC Press Release" release (id 101) on the assumption it was
    a meeting calendar - verified WRONG against the real fetched data:
    that release fires at a median 1-DAY gap (3737 dates over ~12 years),
    not the ~6-week FOMC meeting cadence, so it's evidently a general
    press-release feed (statements, minutes, speeches, ...), not a
    decision calendar. FRED has no dedicated FOMC-meeting-calendar
    release under any "FOMC"/"monetary policy"/"interest rate" keyword
    (checked the full releases catalog). Deriving events from the rate
    series' own change points is self-verifying (every row here IS a
    real, confirmed hike or cut - never a guess) and trades away only
    "hold" meetings (no rate change = nothing for a level-diff to find),
    which is an honest, disclosed simplification given the data
    available, not a hidden approximation."""
    rate = series_observations(series_id).sort_values("date").reset_index(drop=True)
    rate["prior_value"] = rate["value"].shift(1)
    rate["change"] = (rate["value"] - rate["prior_value"]).round(4)
    changes = rate[rate["change"].notna() & (rate["change"] != 0)].reset_index(drop=True)

    rows = []
    for _, row in changes.iterrows():
        direction = "hike" if row["change"] > 0 else "cut"
        rows.append({
            "datetime_utc": et_to_utc(row["date"], *RELEASE_TIME_ET["FOMC"]),
            "event_type": "FOMC",
            "reference_period": row["date"],
            "value": row["value"],
            "prior_value": row["prior_value"],
            "change": row["change"],
            "pct_change": None,
            "vs_trend": None,
            "fomc_direction": direction,
        })
    return pd.DataFrame(rows)


def refresh_fundamentals(config_path: Path, data_dir: Path) -> Path | None:
    """Fetch everything in config_path and write data/events/fundamentals.parquet
    (plus context series). Returns the output path, or None if nothing was
    configured yet. Callable directly (used by live_update.py) or via CLI."""
    hb_path = data_dir / "heartbeats.json"

    if not config_path.exists():
        print(f"{config_path} not found - skipping fundamentals refresh. Copy "
              f"event_config.example.json to {config_path}, run "
              "discover_release_ids.py, and fill in release_id for each event.")
        write_heartbeat("build_fundamentals", "skipped", detail="event_config.json not found", path=hb_path)
        return None

    with track("build_fundamentals", path=hb_path):
        config = json.loads(config_path.read_text())

        events_dir = data_dir / "events"
        events_dir.mkdir(parents=True, exist_ok=True)

        all_events = []
        for event_type, cfg in config.items():
            if event_type.startswith("_"):
                continue
            release_id = cfg.get("release_id")
            series_id = cfg.get("series_id")
            if release_id is None:
                print(f"skipping {event_type}: release_id not set in {config_path}")
                continue
            print(f"fetching {event_type} (release_id={release_id}, series={series_id})...")
            if event_type == "FOMC":
                df = build_fomc_event(series_id)
            else:
                df = build_macro_event(event_type, release_id, series_id)
            print(f"  {len(df)} historical releases")
            all_events.append(df)

        if not all_events:
            print("no events configured with a release_id yet - nothing to write")
            return None

        events = pd.concat(all_events, ignore_index=True).sort_values("datetime_utc").reset_index(drop=True)
        out_path = events_dir / "fundamentals.parquet"
        events.to_parquet(out_path, index=False)
        print(f"wrote {len(events)} total events -> {out_path}")

        context_dir = data_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        for name, series_id in CONTEXT_SERIES.items():
            df = series_observations(series_id).rename(columns={"value": name})
            out = context_dir / f"{name}.parquet"
            df.to_parquet(out, index=False)
            print(f"wrote {len(df)} {name} observations -> {out}")

        return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="event_config.json")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    out_path = refresh_fundamentals(Path(args.config), Path(args.data_dir))
    if out_path is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
