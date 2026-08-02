"""
Backfill N years of XAUUSD tick data and resample into multiple candle
timeframes, saved as Parquet.

Run locally (needs real internet access to datafeed.dukascopy.com):

    python src/build_history.py --years 20 --workers 8

Default is 20 years, not just enough for the original ask - more
historical occurrences means more trustworthy pattern stats (the
MIN_RESOLVED_SAMPLES gate in risk_reward.py needs real sample size to
mean anything). This is safe to push further than Dukascopy's actual
coverage: hours with no data return an empty tick set (see
dukascopy_fetch.py's 404 handling), not an error, so asking for more
years than exist just fetches nothing for the years that aren't there -
it won't corrupt or crash the backfill. Bump --years higher if you want
to find out exactly how far back XAUUSD data actually goes.

Resumable: already-downloaded hours are cached under data/raw_bi5/ and
skipped on re-run, so if it's interrupted you can just run it again.

Timeframes are all derived from the same tick cache, so downloading once
gives you 1min, 5min, 15min, 1h, 4h and 1d candles for free.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import data_quality
from atomic_io import atomic_write_parquet
from dukascopy_fetch import FetchConfig, fetch_hour_ticks, verify_format
from heartbeat import track

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_history")

TIMEFRAMES = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

# Minutes per candle, for converting a pattern's median_candles_to_resolve
# (from risk_reward.summarize_trades) into a real-world time estimate.
TIMEFRAME_MINUTES = {
    "1min": 1, "5min": 5, "15min": 15, "1h": 60, "4h": 240, "1d": 1440,
}


def hour_range(start: dt.datetime, end: dt.datetime):
    cur = start
    while cur < end:
        yield cur
        cur += dt.timedelta(hours=1)


def _fetch_batch(hours: list[dt.datetime], cfg: FetchConfig, workers: int,
                  frames: list, label: str) -> list[dt.datetime]:
    """Fetches one batch of hours, appending successful non-empty results
    to `frames` (shared across passes) and returning the hours that
    raised an exception (network error after retries, or a
    DukascopyFormatError - see dukascopy_fetch.py), NOT the expected
    "market closed, 404" case (which returns an empty-but-successful
    DataFrame and isn't a failure at all)."""
    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_hour_ticks, h, cfg): h for h in hours}
        for fut in as_completed(futures):
            h = futures[fut]
            try:
                df = fut.result()
                if not df.empty:
                    frames.append(df)
            except Exception:
                log.exception("failed on hour %s (%s pass) - will retry", h, label)
                failed.append(h)
            done += 1
            if done % 500 == 0:
                log.info("%s progress: %d/%d hours", label, done, len(hours))
    return failed


def fetch_all_hours(start: dt.datetime, end: dt.datetime, cfg: FetchConfig,
                     workers: int) -> tuple[pd.DataFrame, list[str]]:
    """Returns (ticks, failed_hours) - failed_hours is every hour that
    STILL raised an exception after an automatic retry pass (network
    error after retries, or a DukascopyFormatError - see
    dukascopy_fetch.py), NOT the expected "market closed, 404" case
    (which returns an empty-but-successful DataFrame and isn't a failure
    at all). Reported back to the caller instead of only ever appearing
    in a log line, so a silent data hole from an unattended multi-hour
    backfill run is actually auditable afterwards (see data_quality.py).

    A single hour can fail its first attempt for reasons that are purely
    transient (a brief network blip, a momentary server hiccup) and
    would succeed moments later - each hour already retries up to
    cfg.max_retries times internally (dukascopy_fetch.fetch_hour_ticks),
    but that's all back-to-back, over a matter of seconds; it doesn't
    help against an issue that clears up over a longer stretch of real
    wall-clock time. So after the main pass over every requested hour
    completes, this automatically retries ONLY the hours that failed,
    once, after a short pause - by the time a large backfill's main pass
    has been running for minutes to hours, any short-lived blip has
    almost certainly already cleared, so this quietly recovers most of
    those without requiring you to notice a warning and manually re-run
    the whole command. Only hours that fail AGAIN on this retry pass are
    reported as real gaps."""
    hours = list(hour_range(start, end))
    log.info("fetching %d hours of ticks (%s -> %s)", len(hours), start, end)

    frames: list = []
    failed_dt = _fetch_batch(hours, cfg, workers, frames, "initial")

    if failed_dt:
        log.warning(
            "%d hour(s) failed on the initial pass - waiting 15s then retrying just "
            "those (transient blips often clear up moments later)", len(failed_dt),
        )
        time.sleep(15)
        failed_dt = _fetch_batch(failed_dt, cfg, workers, frames, "retry")
        if failed_dt:
            log.warning("%d hour(s) still failed after the retry pass - real gaps, "
                        "see the list below / data_quality_report.json", len(failed_dt))

    failed = [h.isoformat() for h in failed_dt]

    if not frames:
        return pd.DataFrame(columns=["timestamp", "ask", "bid", "ask_volume", "bid_volume", "mid"]), failed
    ticks = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    ticks = ticks.drop_duplicates(subset="timestamp")
    return ticks, failed


def resample_ticks(ticks: pd.DataFrame, rule: str) -> pd.DataFrame:
    idx_ticks = ticks.set_index("timestamp")
    s = idx_ticks["mid"]
    ask_vol, bid_vol = idx_ticks["ask_volume"], idx_ticks["bid_volume"]
    ohlc = s.resample(rule).ohlc()
    ohlc["volume"] = (ask_vol + bid_vol).resample(rule).sum()
    # ask_volume/bid_volume kept SEPARATE (not just their sum, "volume"
    # above) so ml_system/features.py can compute a real bid/ask order-
    # flow imbalance signal - genuinely different information from "how
    # much traded," a real question this system used to just throw away
    # by summing the two before this line ever ran. tick_count is a raw
    # activity/liquidity proxy (some candles cover a burst of many small
    # ticks, others a few large ones - "volume" alone can't tell those
    # apart). All three are purely additive columns - every existing
    # reader of a candle file only ever asked for open/high/low/close/
    # volume/source, so this changes nothing for them.
    ohlc["ask_volume"] = ask_vol.resample(rule).sum()
    ohlc["bid_volume"] = bid_vol.resample(rule).sum()
    ohlc["tick_count"] = s.resample(rule).count()
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"])
    ohlc["source"] = "dukascopy"
    return ohlc.reset_index()


# Merge priority when two sources disagree on the same candle timestamp -
# higher wins. Dukascopy is deliberately authoritative: it's the deep
# mining source everything was backtested against, so a broker import
# (mt_import.py, tags rows "mt_broker") must never silently overwrite it.
# This is an explicit, order-independent rule - not an accident of
# whichever source happened to be written to disk first (that was the
# previous, weaker behavior: plain drop_duplicates(keep="first") only
# protects whatever's ALREADY on disk, regardless of which source it
# actually came from).
SOURCE_PRIORITY = {"dukascopy": 2, "mt_broker": 1}


def merge_with_existing(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return new_df
    existing = pd.read_parquet(path)

    # Data written before source-tracking existed was Dukascopy-only by
    # construction (mt_bridge didn't exist yet) - backfill the column
    # rather than let it fall through to the lowest merge priority.
    if "source" not in existing.columns:
        existing = existing.assign(source="dukascopy")
    if "source" not in new_df.columns:
        new_df = new_df.assign(source="dukascopy")

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined["_priority"] = combined["source"].map(SOURCE_PRIORITY).fillna(0)
    combined = combined.sort_values(["timestamp", "_priority"], ascending=[True, False])
    combined = combined.drop_duplicates(subset="timestamp", keep="first")
    combined = combined.drop(columns="_priority").sort_values("timestamp")
    return combined.reset_index(drop=True)


def _chunk_already_covered(candles_dir: Path, symbol: str,
                            chunk_start: dt.datetime, chunk_end: dt.datetime) -> bool:
    """True only if EVERY timeframe's candle file already has REASONABLY
    DENSE data strictly WITHIN [chunk_start, chunk_end) - a row-count
    check against a generous floor, not a boundary-proximity check.

    A boundary-proximity check (had a nearby row on each end, within
    some tolerance margin) was tried first and is a real, demonstrated
    bug, not just a theoretical risk: caught by this function's own
    test. Two SEPARATELY-fetched date ranges that happen to sit close
    together (e.g. one ending 2025-07-27, the next starting 2025-07-28 -
    one day apart) can each have a row near a margin-tolerant "boundary"
    of a THIRD chunk that falls entirely in the one-day gap between
    them, wrongly marking that middle chunk "covered" despite it having
    zero real data anywhere inside it - the tolerance margin needed to
    survive normal weekend/holiday gaps was, in that case, wide enough
    to bridge a real missing chunk too. Counting actual rows INSIDE the
    chunk's own range has no such loophole: a genuinely-missing chunk
    has (close to) zero rows in range, full stop, regardless of what
    other date ranges happen to be nearby.

    Uses the 1h timeframe's expected density (roughly `span_hours *
    5/7`, discounting for weekend closure) as the reference, with a
    generous 40% floor - deliberately loose enough to tolerate holidays/
    outages/thin trading without falsely triggering a re-fetch, while
    still firmly rejecting a chunk with near-zero real coverage. Still
    not exhaustive gap-free verification within the interior beyond that
    (data_quality.build_report()'s own detect_gaps() is the real
    backstop for that, run unconditionally at the end over the FULL
    final candle files regardless of what got skipped here).

    Deliberately conservative beyond that: if ANY single timeframe looks
    like it might be missing this chunk, re-fetch rather than risk
    silently skipping real work - re-fetching an already-cached chunk is
    cheap (raw_bi5 makes it a fast disk-read pass, not a network
    refetch), while wrongly skipping a genuinely-missing chunk would
    silently leave a permanent hole. Exists because build_history.py has
    no persisted "resume at chunk N" pointer - every restart (including
    scripts/supervise.py's own auto-restarts) begins its chunk loop at
    chunk 1 again; without this check, a restart late in a 20-year run
    would re-fetch every already-completed year before reaching real new
    work, every time."""
    span_hours = (chunk_end - chunk_start).total_seconds() / 3600
    expected_1h_candles = span_hours * (5 / 7)  # weekend-discounted estimate
    min_required = expected_1h_candles * 0.4     # generous floor - see docstring

    for name in TIMEFRAMES:
        path = candles_dir / f"{symbol}_{name}.parquet"
        if not path.exists():
            return False
        try:
            existing = pd.read_parquet(path, columns=["timestamp"])
        except Exception:
            return False
        if existing.empty:
            return False
        ts = existing["timestamp"]
        in_range_count = int(((ts >= chunk_start) & (ts < chunk_end)).sum())
        # Scale the floor by this timeframe's own candle spacing relative
        # to 1h, so 1min/5min/etc. get a proportionally higher expected
        # count and 4h/1d get a proportionally lower one - the SAME
        # underlying density requirement, just expressed per timeframe.
        scale = 60 / TIMEFRAME_MINUTES[name]
        if in_range_count < min_required * scale:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--workers", type=int, default=8, help="concurrent downloads")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cfg = FetchConfig(symbol=args.symbol, cache_dir=data_dir / "raw_bi5")

    with track("build_history", path=data_dir / "heartbeats.json"):
        if not args.skip_verify:
            log.info("verifying bi5 format assumptions against a recent hour before bulk backfill...")
            verify_format(cfg)

        # NAIVE-but-UTC-valued, matching every other timestamp in this
        # codebase (session_patterns.py, event_timing.py, signal_engine.py,
        # signal_journal.py, news_calendar.py all explicitly
        # tz_localize(None) their "now" for the same reason) - this value
        # becomes `hour_utc` inside fetch_hour_ticks() and flows straight
        # into df["timestamp"] for every candle this backfill writes.
        # Leaving it tz-AWARE here (tzinfo=UTC) would make every candle
        # this script produces tz-aware too, which then raises
        # "can't compare/subtract offset-naive and offset-aware datetimes"
        # the moment it meets any of those naive timestamps downstream -
        # this exact code path was never actually exercised before now
        # (it needs real network access this sandbox doesn't have), so
        # the mismatch had never been triggered until this pass tested it
        # end-to-end with the network call mocked out.
        end = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0, tzinfo=None)
        start = end.replace(year=end.year - args.years)
        hours_requested = len(list(hour_range(start, end)))

        candles_dir = data_dir / "candles"
        candles_dir.mkdir(parents=True, exist_ok=True)

        # Chunked by ~365 days, NOT one single fetch_all_hours() call over
        # the entire requested range - fetch_all_hours() keeps every
        # fetched hour's decoded tick DataFrame in memory (in its `frames`
        # list) until the whole range is done, so a full 20-year/175,320-
        # hour run in one shot means holding ~20 years of raw tick data in
        # RAM simultaneously before writing a single candle - multiple GB,
        # confirmed in practice to be enough to trigger MemoryError deep
        # inside lzma.decompress on an ordinary machine, worsening as the
        # run progresses and more accumulates. Chunking bounds peak memory
        # to roughly ONE chunk's worth regardless of --years, and writes/
        # merges each chunk's candles to disk immediately - a genuine
        # bonus, not just a memory fix: a crash partway through year 15 of
        # 20 now leaves years 1-14 already safely written, instead of the
        # previous all-or-nothing behavior where NOTHING was written until
        # the entire multi-year fetch finished end to end.
        chunk_span = dt.timedelta(days=365)
        all_failed_hours: list[str] = []
        chunk_start = start
        chunk_num = 0
        any_ticks_ever = False
        while chunk_start < end:
            chunk_end = min(chunk_start + chunk_span, end)
            chunk_num += 1

            if _chunk_already_covered(candles_dir, args.symbol, chunk_start, chunk_end):
                log.info("=== chunk %d: %s -> %s === already covered on disk from an earlier "
                         "run - skipping re-fetch", chunk_num, chunk_start.date(), chunk_end.date())
                any_ticks_ever = True
                chunk_start = chunk_end
                continue

            log.info("=== chunk %d: %s -> %s ===", chunk_num, chunk_start.date(), chunk_end.date())

            ticks, failed_hours = fetch_all_hours(chunk_start, chunk_end, cfg, args.workers)
            all_failed_hours.extend(failed_hours)

            if not ticks.empty:
                any_ticks_ever = True
                log.info("chunk %d: fetched %d ticks", chunk_num, len(ticks))
                for name, rule in TIMEFRAMES.items():
                    candles = resample_ticks(ticks, rule)
                    out_path = candles_dir / f"{args.symbol}_{name}.parquet"
                    merged = merge_with_existing(candles, out_path)
                    atomic_write_parquet(merged, out_path)
                    log.info("%s: +%d candles this chunk -> %d total -> %s",
                              name, len(candles), len(merged), out_path)
            del ticks  # free this chunk's tick memory before starting the next one

            chunk_start = chunk_end

        if not any_ticks_ever:
            log.error("no ticks fetched across the entire range - check symbol/connectivity")
            return

        if all_failed_hours:
            log.warning(
                "%d/%d hours FAILED to fetch (network/format errors, NOT market closure) - "
                "re-run build_history.py to retry just these (already-good hours are cached "
                "and skipped); see data/data_quality_report.json for the full list",
                len(all_failed_hours), hours_requested,
            )

        report = data_quality.build_report(
            data_dir, args.symbol, TIMEFRAME_MINUTES,
            failed_hours=all_failed_hours, hours_requested=hours_requested,
            hours_with_data=hours_requested - len(all_failed_hours),
        )
        report_path = data_quality.write_report(report, data_dir)
        n_gaps = sum(len(g) for g in report["gaps_by_timeframe"].values())
        log.info("data quality report -> %s (%d failed hours, %d anomalous gaps found across all timeframes)",
                  report_path, len(all_failed_hours), n_gaps)


if __name__ == "__main__":
    main()
