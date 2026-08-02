"""
Low-level client for Dukascopy's public historical tick-data feed.

No API key needed. Dukascopy publishes one compressed file per instrument
per UTC hour, going back 20+ years for major instruments including spot
gold (XAUUSD). This module downloads and decodes those files into plain
tick rows.

NOTE: this makes real HTTPS requests to datafeed.dukascopy.com. It will
NOT work inside a network-restricted sandbox - run it on a machine with
normal internet access.
"""
from __future__ import annotations

import datetime as dt
import lzma
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Price point (divisor) per instrument: raw integer price * POINT = real price.
# Only gold is in scope for this project.
SYMBOL_POINT = {
    "XAUUSD": 0.001,
}

# Each decompressed tick record is 20 bytes, big-endian:
#   uint32 time_offset_ms   (ms since the start of the hour)
#   uint32 ask_price_raw
#   uint32 bid_price_raw
#   float32 ask_volume
#   float32 bid_volume
TICK_STRUCT = struct.Struct(">3I2f")

# Sanity bounds for gold spot price (USD). Used to catch a decode/format
# error early instead of silently writing garbage into the dataset.
PRICE_SANITY_RANGE = (200.0, 20000.0)

# Dukascopy's servers (or whatever CDN/bot-protection sits in front of
# them) rate-limit/reject requests carrying the default `requests`
# library User-Agent ("python-requests/x.y.z") with a 429, even on the
# very first request from a given IP - a plain browser-looking
# User-Agent is accepted immediately. Verified against the real live
# endpoint (this couldn't be caught in development - real network access
# to Dukascopy isn't available in that sandbox, only on an actual
# deployment machine): the identical request returned 429 with the
# default requests User-Agent and 200 with this one, nothing else
# differed. Not a workaround for anything Dukascopy documents as
# required - just matching what a normal browser sends.
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}


class DukascopyFormatError(RuntimeError):
    """Decoded data failed the sanity check - format assumption is wrong."""


@dataclass
class FetchConfig:
    symbol: str = "XAUUSD"
    cache_dir: Path = Path("data/raw_bi5")
    max_retries: int = 5
    timeout_s: int = 20


def _hour_url(symbol: str, hour_utc: dt.datetime) -> str:
    # Dukascopy indexes months zero-based (January = 00).
    return (
        f"{BASE_URL}/{symbol}/{hour_utc.year:04d}/{hour_utc.month - 1:02d}/"
        f"{hour_utc.day:02d}/{hour_utc.hour:02d}h_ticks.bi5"
    )


def _cache_path(cache_dir: Path, symbol: str, hour_utc: dt.datetime) -> Path:
    return (
        cache_dir
        / symbol
        / f"{hour_utc.year:04d}"
        / f"{hour_utc.month:02d}"
        / f"{hour_utc.day:02d}"
        / f"{hour_utc.hour:02d}h_ticks.bi5"
    )


def _download_raw(url: str, cfg: FetchConfig) -> bytes | None:
    """Returns raw bytes, empty bytes for "no data this hour" (weekend/holiday),
    or None after exhausting retries on a real error."""
    delay = 1.0
    for attempt in range(cfg.max_retries):
        try:
            resp = requests.get(url, timeout=cfg.timeout_s, headers=REQUEST_HEADERS)
        except requests.RequestException:
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 404:
            return b""  # market closed / no ticks this hour
        if resp.status_code == 200:
            return resp.content
        time.sleep(delay)
        delay *= 2
    return None


def _write_cache(cache_file: Path, raw: bytes) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(raw)


def fetch_hour_ticks(hour_utc: dt.datetime, cfg: FetchConfig | None = None) -> pd.DataFrame:
    """Fetch and decode one UTC hour of ticks for cfg.symbol.

    Returns a DataFrame with columns: timestamp (UTC), ask, bid, ask_volume,
    bid_volume, mid. Empty DataFrame if there were no ticks that hour.

    Freshly-downloaded bytes are only written to the on-disk cache AFTER
    they decode cleanly and pass the price sanity check below - not
    immediately after download. A previously-cached file that fails
    validation (corrupt/truncated bytes from a one-off bad download) gets
    DELETED here so the next run re-downloads it, instead of permanently
    re-reading and re-failing on the same bad cached bytes forever - the
    module docstring's "just run it again to fill gaps" promise otherwise
    silently didn't hold for this one failure mode.
    """
    cfg = cfg or FetchConfig()
    point = SYMBOL_POINT[cfg.symbol]
    cache_file = _cache_path(cfg.cache_dir, cfg.symbol, hour_utc)
    from_cache = cache_file.exists()

    if from_cache:
        raw = cache_file.read_bytes()
    else:
        raw = _download_raw(_hour_url(cfg.symbol, hour_utc), cfg)
        if raw is None:
            raise DukascopyFormatError(
                f"failed to download {hour_utc.isoformat()} after {cfg.max_retries} retries"
            )

    if not raw:
        if not from_cache:
            _write_cache(cache_file, raw)  # "no data this hour" is a stable result - safe to cache
        return pd.DataFrame(columns=["timestamp", "ask", "bid", "ask_volume", "bid_volume", "mid"])

    try:
        decompressed = lzma.decompress(raw)
    except lzma.LZMAError as e:
        if from_cache:
            cache_file.unlink(missing_ok=True)
        raise DukascopyFormatError(
            f"corrupt/undecodable data for {hour_utc.isoformat()}: {e}"
        ) from e

    n_records = len(decompressed) // TICK_STRUCT.size
    if n_records * TICK_STRUCT.size != len(decompressed):
        if from_cache:
            cache_file.unlink(missing_ok=True)
        raise DukascopyFormatError(
            f"decompressed size {len(decompressed)} is not a multiple of "
            f"record size {TICK_STRUCT.size} for {hour_utc.isoformat()} - "
            "the bi5 record format assumption may be wrong"
        )

    rows = []
    for i in range(n_records):
        offset_ms, ask_raw, bid_raw, ask_vol, bid_vol = TICK_STRUCT.unpack_from(
            decompressed, i * TICK_STRUCT.size
        )
        rows.append((offset_ms, ask_raw * point, bid_raw * point, ask_vol, bid_vol))

    df = pd.DataFrame(rows, columns=["offset_ms", "ask", "bid", "ask_volume", "bid_volume"])
    df["timestamp"] = hour_utc + pd.to_timedelta(df["offset_ms"], unit="ms")
    df["mid"] = (df["ask"] + df["bid"]) / 2.0
    df = df[["timestamp", "ask", "bid", "ask_volume", "bid_volume", "mid"]]

    lo, hi = PRICE_SANITY_RANGE
    bad = df[(df["mid"] < lo) | (df["mid"] > hi)]
    if len(bad) > 0.01 * len(df):  # more than 1% out of range -> format is wrong, not just noise
        if from_cache:
            cache_file.unlink(missing_ok=True)
        raise DukascopyFormatError(
            f"{len(bad)}/{len(df)} ticks for {hour_utc.isoformat()} fell outside the "
            f"sane gold price range {PRICE_SANITY_RANGE} - check SYMBOL_POINT / struct format"
        )

    if not from_cache:
        _write_cache(cache_file, raw)

    return df


def verify_format(cfg: FetchConfig | None = None) -> None:
    """Quick smoke test: fetch one recent trading hour and confirm the
    decoded prices look like real gold prices. Run this FIRST, before
    kicking off a multi-year backfill, so a wrong format assumption fails
    fast instead of corrupting 10 years of data.

    Walks back through up to 48 candidate hours looking for one with real
    ticks - a SINGLE probed hour hitting a one-off bad download
    (DukascopyFormatError: corrupt bytes, a transient garbled response)
    must not abort the whole check, since this runs on every single
    build_history.py startup (unless --skip-verify), including every
    automatic restart under scripts/supervise.py - without this, one
    unlucky recent hour could put the entire backfill into a permanent
    crash/restart/crash loop before the resumable bulk-fetch logic (which
    already handles exactly this kind of one-off failure correctly) ever
    gets a chance to run. Only genuinely exhausting all 48 candidates
    without finding real data - the true "something's structurally wrong"
    signal - still raises.
    """
    cfg = cfg or FetchConfig()
    # Naive-but-UTC-valued - see build_history.py's `end` for why this
    # matters: it becomes `hour_utc` in fetch_hour_ticks() below and flows
    # straight into df["timestamp"], which must stay naive to match every
    # other timestamp in this codebase.
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    probe = now - dt.timedelta(hours=2)
    last_error: Exception | None = None
    for _ in range(48):  # walk back until we hit an hour with actual ticks
        try:
            df = fetch_hour_ticks(probe, cfg)
        except DukascopyFormatError as e:
            last_error = e
            probe -= dt.timedelta(hours=1)
            continue
        if not df.empty:
            print(f"OK: {probe.isoformat()} -> {len(df)} ticks, "
                  f"price range {df['mid'].min():.2f}-{df['mid'].max():.2f}")
            return
        probe -= dt.timedelta(hours=1)
    detail = f" (last error: {last_error})" if last_error else ""
    raise DukascopyFormatError(f"no ticks found in the last 48 hours - check symbol/connectivity{detail}")


if __name__ == "__main__":
    verify_format()
