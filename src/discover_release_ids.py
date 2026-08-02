"""
One-time setup step: find FRED's actual release_id for each fundamental
event category and the display name of each series, so you can confirm
them before anything gets fetched at scale.

We deliberately do NOT hardcode guessed release_id numbers anywhere in
this codebase - a wrong numeric id would silently pull the wrong release
(or an empty one) and no one would notice until the pattern stats looked
strange. Run this once, eyeball the matches, then fill in
event_config.json yourself.

Usage:
    export FRED_API_KEY=your_key_here
    python src/discover_release_ids.py
"""
from __future__ import annotations

from fred_client import find_releases, series_info

# (event label, search keyword for the release name, FRED series id we'll
#  use for the actual values - these tickers are standard/stable and
#  fine to hardcode; only the numeric release_id is being discovered)
CANDIDATES = [
    ("CPI",  "Consumer Price Index",        "CPIAUCSL"),
    ("PCE",  "Personal Income and Outlays", "PCEPI"),
    ("NFP",  "Employment Situation",        "PAYEMS"),
    ("GDP",  "Gross Domestic Product",      "GDPC1"),
    ("FOMC", "FOMC",                        "DFEDTARU"),
]


def main():
    print("Confirm these before filling in event_config.json:\n")
    for label, keyword, series_id in CANDIDATES:
        print(f"=== {label} ===")
        try:
            info = series_info(series_id)
            print(f"  series {series_id}: \"{info['title']}\" "
                  f"(units: {info['units']}, freq: {info['frequency']})")
        except Exception as e:
            print(f"  series {series_id}: ERROR - {e}")

        try:
            matches = find_releases(keyword)
            if matches.empty:
                print(f"  no releases matched '{keyword}'")
            else:
                for _, row in matches.iterrows():
                    print(f"  release_id={row['id']:<6} \"{row['name']}\"")
        except Exception as e:
            print(f"  release search error - {e}")
        print()

    print("Pick the correct release_id for each event, then write them into "
          "event_config.json (see event_config.example.json for the shape).")


if __name__ == "__main__":
    main()
