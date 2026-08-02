"""
Thin client for FRED (Federal Reserve Economic Data), the free/official
source for the fundamental events that actually move gold: CPI, PCE,
Non-Farm Payrolls, GDP, and the Fed funds target rate (FOMC decisions).

Needs a free API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set it as the FRED_API_KEY environment variable.

NOTE: like dukascopy_fetch.py, this makes real HTTPS requests
(api.stlouisfed.org) and will NOT work inside a network-restricted
sandbox - run it on a machine with normal internet access.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

BASE_URL = "https://api.stlouisfed.org/fred"


class FredError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise FredError(
            "FRED_API_KEY environment variable is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and export it."
        )
    return key


def _get(path: str, params: dict, max_retries: int = 5) -> dict:
    params = {**params, "api_key": _api_key(), "file_type": "json"}
    delay = 1.0
    for attempt in range(max_retries):
        resp = requests.get(f"{BASE_URL}/{path}", params=params, timeout=20)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:  # rate limited
            time.sleep(delay)
            delay *= 2
            continue
        raise FredError(f"FRED API error {resp.status_code} on {path}: {resp.text[:300]}")
    raise FredError(f"FRED API rate-limited after {max_retries} retries on {path}")


def list_releases() -> pd.DataFrame:
    """All FRED releases (id + name). Paginated 1000 at a time."""
    rows, offset = [], 0
    while True:
        data = _get("releases", {"limit": 1000, "offset": offset})
        releases = data.get("releases", [])
        if not releases:
            break
        rows.extend(releases)
        offset += len(releases)
        if len(releases) < 1000:
            break
    return pd.DataFrame(rows)[["id", "name"]]


def find_releases(keyword: str) -> pd.DataFrame:
    all_releases = list_releases()
    mask = all_releases["name"].str.contains(keyword, case=False, na=False)
    return all_releases[mask].reset_index(drop=True)


def series_info(series_id: str) -> dict:
    data = _get("series", {"series_id": series_id})
    seriess = data.get("seriess", [])
    if not seriess:
        raise FredError(f"series {series_id} not found")
    return seriess[0]


def release_dates(release_id: int, include_future: bool = False) -> pd.DataFrame:
    """Publication dates for a release, oldest first.

    include_future=False (default): only dates that already have data -
    i.e. releases that actually happened. Used for historical mining.

    include_future=True: also includes dates FRED has scheduled but that
    haven't published yet - FRED/ALFRED tracks the release CALENDAR ahead
    of time, not just after-the-fact publication dates. Used to build a
    forward-looking "when's the next high-impact release" calendar for
    live signals. Verify this returns what you expect for your release_id
    the first time you use it - scheduled-date coverage isn't guaranteed
    identical across every release series.
    """
    rows, offset = [], 0
    while True:
        data = _get("release/dates", {
            "release_id": release_id,
            "limit": 1000,
            "offset": offset,
            "sort_order": "asc",
            "include_release_dates_with_no_data": "true" if include_future else "false",
        })
        dates = data.get("release_dates", [])
        if not dates:
            break
        rows.extend(dates)
        offset += len(dates)
        if len(dates) < 1000:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df[["date"]].sort_values("date").reset_index(drop=True)


def series_observations(series_id: str, start: str = "1990-01-01") -> pd.DataFrame:
    """Plain (latest-vintage) observations for a series: one row per
    reference period with its final/current value. WARNING: this is the
    number as it stands TODAY, after every subsequent revision - for a
    series that gets revised after initial publication (NFP, GDP - see
    series_observations_as_first_published below), this is NOT what a
    trader actually saw on the release date. Safe to use only for series
    that aren't meaningfully revised (e.g. FOMC's rate-decision series,
    which is a policy level, not a survey estimate) or for purely
    descriptive/contextual use, never for anything tagged to a historical
    release date and used as if it were known then."""
    data = _get("series/observations", {
        "series_id": series_id,
        "observation_start": start,
        "sort_order": "asc",
    })
    obs = data.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]].dropna().reset_index(drop=True)


def series_observations_as_first_published(series_id: str, start: str = "1990-01-01") -> pd.DataFrame:
    """ALFRED point-in-time observations: for each reference period, the
    value AS IT WAS FIRST PUBLISHED - not today's latest-revised number
    (see series_observations() above). This is what a trader actually
    saw at the time, which matters for any series that gets revised
    after initial release (NFP and GDP both are, routinely and
    sometimes materially - CPI/PCE are rarely revised in practice but
    this applies the same correct methodology to all of them rather than
    special-casing).

    How: requesting a WIDE realtime window with output_type=2
    ("observations by real-time period, all observations") from FRED's
    series/observations endpoint. VERIFIED against a real call (this
    endpoint could not be reached from the sandbox this was originally
    built in, so the first version of this function guessed at a
    long-format shape - one row per (date, realtime_start) pair with a
    `value`/`realtime_start` column each - that turned out to be wrong
    and crashed with KeyError('realtime_start') the first time it
    actually ran). What FRED's output_type=2 genuinely returns is WIDE,
    not long: one row per reference `date`, and one column PER VINTAGE
    DATE the series has ever had, named "{series_id}_{YYYYMMDD}" - e.g.
    for CPIAUCSL, columns like "CPIAUCSL_20230214". A reference period's
    value only starts APPEARING from whichever vintage column corresponds
    to when it was first published - earlier vintage columns are simply
    ABSENT (not null, not ".", just not a key) for that row, since the
    number didn't exist yet as of that vintage. Confirmed directly:
    "CPIAUCSL_20230214" is present for the 2023-01-01 (January CPI) row
    but absent for the 2023-02-01 (February CPI) row, because February's
    reading hadn't been published yet as of Feb 14 2023. So for each
    row, the EARLIEST vintage column that's actually present IS the
    value as first published - exactly what this function now extracts.
    """
    data = _get("series/observations", {
        "series_id": series_id,
        "observation_start": start,
        "sort_order": "asc",
        "realtime_start": "1776-07-04",  # FRED's own convention for "the beginning of time"
        "realtime_end": "9999-12-31",    # FRED's own convention for "forever" - together these
        "output_type": 2,                # request every vintage, not just the one in effect today
    })
    obs = data.get("observations", [])
    if not obs:
        return pd.DataFrame(columns=["date", "value"])

    vintage_prefix = f"{series_id}_"
    rows = []
    for row in obs:
        date = row.get("date")
        # Every non-"date" key is "{series_id}_{YYYYMMDD}" - the value as
        # of that vintage. Missing keys (not "." or None - genuinely
        # absent from the dict) mean the reference period didn't exist
        # as of that vintage yet. The chronologically-earliest PRESENT
        # vintage column is the value as first published - "." is FRED's
        # own separate convention for an explicit missing value within a
        # vintage that does exist, handled the same as any other missing
        # value (skipped, not treated as a real 0/empty reading).
        vintages = [
            (key[len(vintage_prefix):], val) for key, val in row.items()
            if key != "date" and key.startswith(vintage_prefix) and val not in (None, ".", "")
        ]
        if not vintages:
            continue
        vintages.sort(key=lambda pair: pair[0])  # "YYYYMMDD" strings sort chronologically as-is
        rows.append({"date": date, "value": vintages[0][1]})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
