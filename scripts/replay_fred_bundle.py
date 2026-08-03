"""
Sandbox-side counterpart to termux_fetch_fundamentals.py: replays a
fred_raw_bundle.json (fetched on a machine with real internet access)
through the EXACT SAME processing code fred_client.py / discover_release_ids.py
/ build_fundamentals.py already use, by monkeypatching fred_client._get
to serve cached responses instead of making HTTP calls. Nothing about
the alignment / point-in-time-vintage / look-ahead-safety logic is
duplicated here - this file's only job is the cache shim.

Usage (from repo root, with the bundle file already in place):
    python scripts/replay_fred_bundle.py --discover    # step 1: eyeball release_ids
    # ... fill in event_config.json using the printed candidates ...
    python scripts/replay_fred_bundle.py --build        # step 2: write fundamentals.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fred_client  # noqa: E402


def _cache_key(path: str, params: dict) -> tuple:
    relevant = {k: v for k, v in params.items() if k not in ("api_key", "file_type")}
    return (path, tuple(sorted(relevant.items())))


def install_replay(bundle_path: Path) -> None:
    bundle = json.loads(bundle_path.read_text())
    cache: dict[tuple, dict] = {}

    # releases (paginated in the original fetch, but bundle stores the
    # already-concatenated full list - replay it as a single page so
    # list_releases()'s pagination loop terminates after one call)
    cache[_cache_key("releases", {"limit": 1000, "offset": 0})] = {"releases": bundle["releases"]}

    for series_id, info in bundle["series_info"].items():
        cache[_cache_key("series", {"series_id": series_id})] = {"seriess": [info] if info else []}

    # release/dates can exceed 1000 rows (e.g. release_id 101 "FOMC Press
    # Release" has 3739) - fred_client.release_dates() paginates with
    # offset in steps of 1000 until a page comes back short, so replay
    # every offset that a real paginated fetch would have requested, each
    # sliced from the bundle's already-concatenated full list.
    for rid_str, dates in bundle["release_dates"].items():
        for offset in range(0, len(dates) + 1, 1000):
            cache[_cache_key("release/dates", {
                "release_id": int(rid_str), "limit": 1000, "offset": offset,
                "sort_order": "asc", "include_release_dates_with_no_data": "false",
            })] = {"release_dates": dates[offset:offset + 1000]}

    for series_id, obs in bundle["series_observations_vintage"].items():
        cache[_cache_key("series/observations", {
            "series_id": series_id, "observation_start": "1990-01-01", "sort_order": "asc",
            "realtime_start": "1776-07-04", "realtime_end": "9999-12-31", "output_type": 2,
        })] = {"observations": obs}

    for series_id, obs in bundle["series_observations_plain"].items():
        cache[_cache_key("series/observations", {
            "series_id": series_id, "observation_start": "1990-01-01", "sort_order": "asc",
        })] = {"observations": obs}

    def _replay_get(path: str, params: dict, max_retries: int = 5) -> dict:
        key = _cache_key(path, params)
        if key not in cache:
            raise fred_client.FredError(
                f"replay bundle has no cached response for {path} {params} - "
                "re-run termux_fetch_fundamentals.py to refresh the bundle"
            )
        return cache[key]

    fred_client._get = _replay_get
    # os.environ.get("FRED_API_KEY") is still checked by _api_key() inside
    # the real fred_client functions this replay doesn't touch - set a
    # dummy value so that check doesn't block a replay run that never
    # needs a real key.
    import os
    os.environ.setdefault("FRED_API_KEY", "replay-bundle-no-real-key-needed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="fred_raw_bundle.json")
    parser.add_argument("--config", default="event_config.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--discover", action="store_true", help="print release_id candidates (step 1)")
    parser.add_argument("--build", action="store_true", help="write fundamentals.parquet (step 2)")
    args = parser.parse_args()

    install_replay(Path(args.bundle))

    if args.discover:
        import discover_release_ids
        discover_release_ids.main()
    elif args.build:
        import build_fundamentals
        out_path = build_fundamentals.refresh_fundamentals(Path(args.config), Path(args.data_dir))
        if out_path is None:
            raise SystemExit(1)
    else:
        parser.error("pass --discover or --build")


if __name__ == "__main__":
    main()
