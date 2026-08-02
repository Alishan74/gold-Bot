"""
Dependency-free raw-tick fetcher for environments where pandas/pyarrow
can't be (easily) installed - e.g. Termux/Android, where PyPI often has
no prebuilt wheel and pip falls back to a slow/fragile from-source build.
See dukascopy_fetch.py's module docstring for the protocol this mirrors.

This does NOT decode ticks into a DataFrame or write candle parquet
files - it does the ONE thing that genuinely needs to run on a machine
with real internet access: download and cache Dukascopy's raw per-hour
.bi5 files under data/raw_bi5/, in EXACTLY the cache layout
dukascopy_fetch.fetch_hour_ticks() reads from. Once that cache exists
(push it to the repo, or copy it over), build_history.py can run
ANYWHERE with pandas/pyarrow installed - including a machine with NO
network access at all - and resample the cached ticks into candles with
ZERO new network requests, exactly the cache-reuse guarantee
scripts/backfill_order_flow.py's own docstring already documents for
this same cache directory.

Validation mirrors dukascopy_fetch.fetch_hour_ticks() exactly - LZMA
decompress, record-size alignment, gold price sanity range - using only
stdlib (lzma, struct). None of those checks actually need a DataFrame,
so a corrupt/garbled download is never cached here either, same
guarantee as the pandas version.

Usage:
    python scripts/termux_fetch_raw.py --check          # one-hour smoke test first
    python scripts/termux_fetch_raw.py --years 20 --workers 4
"""
from __future__ import annotations

import argparse
import datetime as dt
import lzma
import struct
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
SYMBOL_POINT = {"XAUUSD": 0.001}
TICK_STRUCT = struct.Struct(">3I2f")
PRICE_SANITY_RANGE = (200.0, 20000.0)
# See dukascopy_fetch.py's own REQUEST_HEADERS docstring: the default
# User-Agent gets 429'd by whatever sits in front of Dukascopy's feed,
# a plain browser-looking one is accepted - not a documented requirement,
# just matching what a normal browser sends.
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}


def hour_url(symbol: str, hour_utc: dt.datetime) -> str:
    # Dukascopy indexes months zero-based (January = 00).
    return (
        f"{BASE_URL}/{symbol}/{hour_utc.year:04d}/{hour_utc.month - 1:02d}/"
        f"{hour_utc.day:02d}/{hour_utc.hour:02d}h_ticks.bi5"
    )


def cache_path(cache_dir: Path, symbol: str, hour_utc: dt.datetime) -> Path:
    return (
        cache_dir / symbol / f"{hour_utc.year:04d}" / f"{hour_utc.month:02d}"
        / f"{hour_utc.day:02d}" / f"{hour_utc.hour:02d}h_ticks.bi5"
    )


def download_raw(url: str, timeout_s: int = 20, max_retries: int = 5) -> "bytes | None":
    """Returns raw bytes, empty bytes for "no data this hour" (weekend/
    holiday - a 404), or None after exhausting retries on a real error -
    same three-way contract as dukascopy_fetch._download_raw()."""
    delay = 1.0
    for _ in range(max_retries):
        req = urllib.request.Request(url, headers=REQUEST_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            time.sleep(delay)
            delay *= 2
        except urllib.error.URLError:
            time.sleep(delay)
            delay *= 2
    return None


def validate(raw: bytes, symbol: str) -> "str | None":
    """None if `raw` is valid (or legitimately empty), else a reason
    string - mirrors dukascopy_fetch.fetch_hour_ticks()'s exact checks
    (LZMA decompress, record-size alignment, price sanity range) so a
    corrupt/garbled download is never written to the cache."""
    if not raw:
        return None  # legitimate "no ticks this hour"
    try:
        decompressed = lzma.decompress(raw)
    except lzma.LZMAError as e:
        return f"corrupt LZMA data: {e}"
    n = len(decompressed) // TICK_STRUCT.size
    if n * TICK_STRUCT.size != len(decompressed):
        return f"decompressed size {len(decompressed)} not a multiple of record size {TICK_STRUCT.size}"
    point = SYMBOL_POINT[symbol]
    lo, hi = PRICE_SANITY_RANGE
    bad = 0
    for i in range(n):
        _, ask_raw, bid_raw, _, _ = TICK_STRUCT.unpack_from(decompressed, i * TICK_STRUCT.size)
        mid = (ask_raw + bid_raw) * point / 2.0
        if mid < lo or mid > hi:
            bad += 1
    if n and bad > 0.01 * n:  # more than 1% out of range -> format is wrong, not just noise
        return f"{bad}/{n} ticks outside sane gold price range {PRICE_SANITY_RANGE}"
    return None


def fetch_hour(hour_utc: dt.datetime, symbol: str, cache_dir: Path) -> str:
    """Returns 'cached' (already had it - zero network calls), 'ok'
    (freshly fetched and cached), 'empty' (no data this hour, cached as
    empty so it's not re-requested next run), or 'failed'."""
    cfile = cache_path(cache_dir, symbol, hour_utc)
    if cfile.exists():
        return "cached"
    raw = download_raw(hour_url(symbol, hour_utc))
    if raw is None:
        return "failed"
    reason = validate(raw, symbol)
    if reason is not None:
        return "failed"
    cfile.parent.mkdir(parents=True, exist_ok=True)
    cfile.write_bytes(raw)
    return "empty" if not raw else "ok"


def hour_range(start: dt.datetime, end: dt.datetime):
    cur = start
    while cur < end:
        yield cur
        cur += dt.timedelta(hours=1)


def check(symbol: str, cache_dir: Path) -> None:
    """One-hour smoke test - same purpose as dukascopy_fetch.
    verify_format(): confirm real ticks come back before committing to a
    multi-year run. Walks back through recent hours (skipping ones
    already cached, so re-running --check doesn't just report stale
    cache hits) until it finds one with real data."""
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    probe = now - dt.timedelta(hours=2)
    for _ in range(48):
        cfile = cache_path(cache_dir, symbol, probe)
        if cfile.exists():
            probe -= dt.timedelta(hours=1)
            continue
        result = fetch_hour(probe, symbol, cache_dir)
        if result == "ok":
            print(f"OK: {probe.isoformat()} fetched and validated real ticks -> {cfile}")
            return
        if result == "failed":
            print(f"  {probe.isoformat()}: failed, trying an earlier hour")
        probe -= dt.timedelta(hours=1)
    raise SystemExit("no real (non-cached) ticks found in the last 48 hours - check connectivity, "
                      "or the cache already covers this whole range")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4,
                         help="lower than build_history.py's desktop default (8) - gentler on "
                              "mobile data/battery/thermals")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--check", action="store_true",
                         help="fetch one recent hour and exit, don't run the full backfill")
    args = parser.parse_args()

    cache_dir = Path(args.data_dir) / "raw_bi5"

    if args.check:
        check(args.symbol, cache_dir)
        return

    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    start = now.replace(year=now.year - args.years)
    hours = list(hour_range(start, now))
    print(f"fetching {len(hours)} hours of {args.symbol} from {start.isoformat()} to {now.isoformat()}")
    print(f"caching to {cache_dir} - safe to Ctrl+C and re-run any time, already-cached hours are "
          f"skipped with zero network calls")

    counts = {"cached": 0, "ok": 0, "empty": 0, "failed": 0}
    failed_hours = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_hour, h, args.symbol, cache_dir): h for h in hours}
        for fut in as_completed(futures):
            h = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"  error on {h.isoformat()}: {e}")
                result = "failed"
            counts[result] += 1
            if result == "failed":
                failed_hours.append(h)
            done += 1
            if done % 200 == 0:
                print(f"  progress: {done}/{len(hours)} (cached={counts['cached']} ok={counts['ok']} "
                      f"empty={counts['empty']} failed={counts['failed']})")

    print(f"done: {done}/{len(hours)} - cached={counts['cached']} ok={counts['ok']} "
          f"empty={counts['empty']} failed={counts['failed']}")
    if failed_hours:
        print(f"{len(failed_hours)} hour(s) failed - just re-run this same command, "
              f"already-cached hours are skipped automatically so this only retries the gaps")


if __name__ == "__main__":
    main()
