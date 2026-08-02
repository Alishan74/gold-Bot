"""
Backfill ask_volume/bid_volume/tick_count into EXISTING candle files -
these three columns were added to build_history.resample_ticks() after
this project's original 20-year backfill already ran, so every candle
file written before that change is missing them. This script re-derives
ONLY those three columns (never touching open/high/low/close/volume/
source, which stay exactly as they already are) from the SAME cached raw
tick data (data/raw_bi5/) the original backfill already downloaded and
validated - dukascopy_fetch.fetch_hour_ticks() reads from that cache
whenever it's present, so this should need ZERO new network requests for
any date range that already has a candle file (ticks for that range were,
by definition, already fetched and cached to produce those candles in the
first place).

Chunked by year (the same memory-bounded discipline build_history.py's
own main() uses, for the identical reason - accumulating 20 years of raw
ticks in memory at once was the literal MemoryError this project already
hit once during the original backfill), and merges the 3 new columns
onto each EXISTING candle file by timestamp - a left join that can never
alter or drop an existing row, only add to it.

Usage:
    python scripts/backfill_order_flow.py --symbol XAUUSD
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from atomic_io import atomic_write_parquet  # noqa: E402
from build_history import TIMEFRAMES, fetch_all_hours, resample_ticks  # noqa: E402
from dukascopy_fetch import FetchConfig  # noqa: E402

NEW_COLUMNS = ["ask_volume", "bid_volume", "tick_count"]


def merge_order_flow_columns(existing: pd.DataFrame, resampled: pd.DataFrame) -> pd.DataFrame:
    """Upsert `resampled`'s NEW_COLUMNS onto `existing` by timestamp - a
    row whose timestamp appears in `resampled` (this chunk's date range)
    gets THIS run's freshly-resampled values (never stale ones from a
    previous partial attempt, same reasoning as before); a row whose
    timestamp does NOT appear in `resampled` (every OTHER chunk's date
    range, already backfilled by an earlier iteration of the caller's
    per-chunk loop) is left completely untouched.

    Previously this dropped NEW_COLUMNS from the WHOLE of `existing`
    before merging back in only the current chunk's slice - since the
    caller processes one ~year-long chunk at a time and writes the
    result to disk after each one, that drop-then-left-merge silently
    wiped every OTHER already-backfilled chunk's values back to NaN on
    every subsequent chunk's write, leaving only the LAST chunk
    processed with real data by the time the full backfill finished -
    verified against a real run, not a hypothetical (tick_count_ratio
    came out entirely NaN across a whole timeframe's history because of
    exactly this). DataFrame.update() aligns on the index and only
    overwrites cells where `to_merge` has a non-NA value for that
    (timestamp, column) pair, which is precisely the "only touch this
    chunk's rows" semantics needed here."""
    to_merge = resampled[["timestamp"] + NEW_COLUMNS].drop_duplicates(subset="timestamp")
    existing = existing.set_index("timestamp")
    for col in NEW_COLUMNS:
        if col not in existing.columns:
            existing[col] = float("nan")
    existing.update(to_merge.set_index("timestamp"))
    return existing.reset_index().sort_values("timestamp").reset_index(drop=True)


def backfill_symbol(symbol: str, data_dir: Path, workers: int = 6) -> None:
    candles_dir = data_dir / "candles"
    cfg = FetchConfig(symbol=symbol, cache_dir=data_dir / "raw_bi5")

    timeframe_files = {
        name: candles_dir / f"{symbol}_{name}.parquet" for name in TIMEFRAMES
        if (candles_dir / f"{symbol}_{name}.parquet").exists()
    }
    if not timeframe_files:
        raise SystemExit(f"no candle files found in {candles_dir} - nothing to backfill")

    # The date range to re-derive ticks over is the UNION of every
    # existing timeframe file's own min/max timestamp - re-deriving ticks
    # ONCE per chunk and reusing them across all 6 timeframes' resamples,
    # rather than re-fetching per timeframe.
    mins, maxs = [], []
    for path in timeframe_files.values():
        ts = pd.read_parquet(path, columns=["timestamp"])["timestamp"]
        mins.append(ts.min())
        maxs.append(ts.max())
    overall_start = min(mins)
    overall_end = max(maxs)
    print(f"backfilling ask_volume/bid_volume/tick_count for {symbol} over "
          f"{overall_start} -> {overall_end}, {len(timeframe_files)} timeframe file(s): "
          f"{list(timeframe_files)}")

    chunk_span = dt.timedelta(days=365)
    chunk_start = pd.Timestamp(overall_start).to_pydatetime().replace(minute=0, second=0, microsecond=0)
    end = pd.Timestamp(overall_end).to_pydatetime() + dt.timedelta(hours=1)
    chunk_num = 0

    while chunk_start < end:
        chunk_num += 1
        chunk_end = min(chunk_start + chunk_span, end)
        print(f"=== chunk {chunk_num}: {chunk_start.date()} -> {chunk_end.date()} ===")
        ticks, failed_hours = fetch_all_hours(chunk_start, chunk_end, cfg, workers)
        if failed_hours:
            print(f"  note: {len(failed_hours)} hour(s) not available from cache/re-fetch this chunk - "
                  f"those candles keep whatever order-flow columns they already had (NaN if none)")
        if not ticks.empty:
            for name, rule in TIMEFRAMES.items():
                path = timeframe_files.get(name)
                if path is None:
                    continue
                resampled = resample_ticks(ticks, rule)
                existing = pd.read_parquet(path)
                merged = merge_order_flow_columns(existing, resampled)
                atomic_write_parquet(merged, path)
                print(f"  {name}: merged order-flow columns for this chunk -> {path}")
        del ticks
        chunk_start = chunk_end

    print("backfill complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    backfill_symbol(args.symbol, Path(args.data_dir), args.workers)


if __name__ == "__main__":
    main()
