"""
Dependency-free FRED (Federal Reserve Economic Data) raw fetcher for
environments where pandas/pyarrow can't be (easily) installed - e.g.
Termux/Android, mirroring termux_fetch_raw.py's role for tick data.

Needs a free API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set it as the FRED_API_KEY environment variable before running.

This does NOT do any of build_fundamentals.py's alignment/vintage-
extraction/look-ahead-safety logic - it does the ONE thing that
genuinely needs to run on a machine with real internet access: make the
raw HTTPS calls to api.stlouisfed.org and dump every response verbatim
into one JSON bundle file. scripts/replay_fred_bundle.py (run in the
sandbox, with pandas available) replays that bundle through the EXACT
SAME processing functions build_fundamentals.py already uses (by
monkeypatching fred_client._get to serve cached responses instead of
hitting the network) - so none of the alignment/point-in-time-vintage
logic is duplicated or re-implemented here, only the fetch.

Fetches, for each of CPI/PCE/NFP/GDP/FOMC:
  - every FRED release whose name matches the event's search keyword
    (not just one guessed id - discover_release_ids.py's "eyeball and
    confirm" safety property is preserved by fetching every plausible
    candidate here and letting the sandbox-side replay verify which one
    actually looks right from its real title/date range, rather than
    ever hardcoding a possibly-wrong numeric id)
  - that release's full publication-date history
  - the series' full point-in-time vintage history (output_type=2 - see
    fred_client.series_observations_as_first_published's docstring)
Plus the FOMC rate series (plain, unrevised) and the two continuous
context series (DXY, 10y real yield).

Usage:
    export FRED_API_KEY=your_key_here
    python scripts/termux_fetch_fundamentals.py
    # writes fred_raw_bundle.json in the current directory - push that
    # file to the repo (or copy it over) for the sandbox side to replay.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://api.stlouisfed.org/fred"

# (event label, search keyword for the release name, FRED series id)
CANDIDATES = [
    ("CPI",  "Consumer Price Index",        "CPIAUCSL"),
    ("PCE",  "Personal Income and Outlays", "PCEPI"),
    ("NFP",  "Employment Situation",        "PAYEMS"),
    ("GDP",  "Gross Domestic Product",      "GDPC1"),
    ("FOMC", "FOMC",                        "DFEDTARU"),
]
CONTEXT_SERIES = {
    "dxy": "DTWEXBGS",
    "real_yield_10y": "DFII10",
}


def _api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        sys.exit("FRED_API_KEY environment variable is not set. Get a free key at "
                  "https://fred.stlouisfed.org/docs/api/api_key.html and export it.")
    return key


def _get(path: str, params: dict, max_retries: int = 5) -> dict:
    params = {**params, "api_key": _api_key(), "file_type": "json"}
    url = f"{BASE_URL}/{path}?{urllib.parse.urlencode(params)}"
    delay = 1.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            body = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"FRED API error {e.code} on {path}: {body}") from e
    raise RuntimeError(f"FRED API rate-limited after {max_retries} retries on {path}")


def list_releases() -> list:
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
    return rows


def release_dates_raw(release_id: int) -> list:
    rows, offset = [], 0
    while True:
        data = _get("release/dates", {
            "release_id": release_id, "limit": 1000, "offset": offset,
            "sort_order": "asc", "include_release_dates_with_no_data": "false",
        })
        dates = data.get("release_dates", [])
        if not dates:
            break
        rows.extend(dates)
        offset += len(dates)
        if len(dates) < 1000:
            break
    return rows


def main():
    _api_key()  # fail fast if unset
    bundle = {"releases": [], "series_info": {}, "release_dates": {}, "series_observations_vintage": {},
              "series_observations_plain": {}}

    print("fetching full FRED releases catalog...")
    bundle["releases"] = list_releases()
    print(f"  {len(bundle['releases'])} releases")

    all_candidate_ids = set()
    for label, keyword, series_id in CANDIDATES:
        print(f"\n=== {label} ===")
        info = _get("series", {"series_id": series_id})
        seriess = info.get("seriess", [])
        bundle["series_info"][series_id] = seriess[0] if seriess else None
        if seriess:
            print(f"  series {series_id}: \"{seriess[0]['title']}\"")

        matches = [r for r in bundle["releases"] if keyword.lower() in r.get("name", "").lower()]
        print(f"  {len(matches)} release(s) matched '{keyword}':")
        for r in matches:
            print(f"    release_id={r['id']:<6} \"{r['name']}\"")
            all_candidate_ids.add(r["id"])

    print(f"\nfetching publication-date history for {len(all_candidate_ids)} candidate release(s)...")
    for rid in sorted(all_candidate_ids):
        bundle["release_dates"][str(rid)] = release_dates_raw(rid)
        print(f"  release_id={rid}: {len(bundle['release_dates'][str(rid)])} dates")

    print("\nfetching point-in-time vintage history (as-first-published) for CPI/PCE/NFP/GDP...")
    for label, keyword, series_id in CANDIDATES:
        if label == "FOMC":
            continue
        data = _get("series/observations", {
            "series_id": series_id, "observation_start": "1990-01-01", "sort_order": "asc",
            "realtime_start": "1776-07-04", "realtime_end": "9999-12-31", "output_type": 2,
        })
        bundle["series_observations_vintage"][series_id] = data.get("observations", [])
        print(f"  {series_id}: {len(bundle['series_observations_vintage'][series_id])} rows")

    print("\nfetching plain (latest-vintage) series for FOMC rate + context series...")
    for series_id in ["DFEDTARU"] + list(CONTEXT_SERIES.values()):
        data = _get("series/observations", {
            "series_id": series_id, "observation_start": "1990-01-01", "sort_order": "asc",
        })
        bundle["series_observations_plain"][series_id] = data.get("observations", [])
        print(f"  {series_id}: {len(bundle['series_observations_plain'][series_id])} rows")

    out_path = "fred_raw_bundle.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nwrote {out_path} ({size_kb:.0f} KB) - push this file to the repo for the sandbox side to replay.")


if __name__ == "__main__":
    main()
