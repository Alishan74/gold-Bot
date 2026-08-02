"""
Loads OHLCV history exported from MetaTrader (mt_bridge/ExportHistoryMT4.mq4
/ ExportHistoryMT5.mq5 for the one-time backfill, LiveBridgeMT4.mq4 /
LiveBridgeMT5.mq5 for ongoing updates) into this project's
data/candles/{SYMBOL}_{tf}.parquet files - the same schema and merge
behavior build_history.py uses, so build_pattern_library.py,
signal_engine.py, and the dashboard all work unchanged regardless of
whether candles came from Dukascopy or your actual broker.

Timestamp handling: MT4/5 record candle times in the BROKER'S SERVER
time, which is frequently NOT UTC (commonly GMT+0, +2, or +3 depending
on the broker, with its own DST convention that may not match any
real-world timezone). Both export scripts write the server's current UTC
offset - from MQL's own TimeGMTOffset() - as a comment on the first line
of every CSV, and this loader uses THAT value, not a hardcoded guess. If
a file is missing that comment, this refuses to import it rather than
assume an offset.

READ BEFORE TRUSTING THIS: the MQL scripts in mt_bridge/ could not be
compiled or run in the environment this was built in (no MetaTrader
available there). They're written against stable, documented MQL4/5 APIs
(CopyRates, MqlRates, TimeGMTOffset, FileOpen/FileWrite), but you should
sanity-check the FIRST export yourself: open the generated CSV, compare
a handful of recent candles against what your MT4/5 chart actually
shows, before trusting a full backfill built from it.

Usage:
    python src/mt_import.py --export-dir "/path/to/MQL5/Files/gold_export" --symbol XAUUSD.a
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from build_history import TIMEFRAMES, merge_with_existing
from heartbeat import track

TIMEFRAME_NAMES = list(TIMEFRAMES.keys())

_GMT_OFFSET_RE = re.compile(r"#\s*gmt_offset_seconds\s*=\s*(-?\d+)")


class MtImportError(RuntimeError):
    pass


def _read_mt_csv(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first_line = f.readline().strip()
    m = _GMT_OFFSET_RE.match(first_line)
    if not m:
        raise MtImportError(
            f"{path}: expected a '# gmt_offset_seconds=<seconds>' comment on the "
            "first line (written by the mt_bridge export/live-bridge scripts) - "
            "got something else instead. Refusing to guess the server/UTC offset."
        )
    gmt_offset_s = int(m.group(1))

    df = pd.read_csv(path, skiprows=1, dtype=str, keep_default_na=False)
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    required = {"timestamp_server", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise MtImportError(f"{path}: missing expected column(s) {missing} - got {list(df.columns)}")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col].str.strip().str.strip('"'), errors="coerce")
    df["timestamp_server"] = df["timestamp_server"].str.strip().str.strip('"')

    df["timestamp"] = (
        pd.to_datetime(df["timestamp_server"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
        - pd.Timedelta(seconds=gmt_offset_s)
    )
    before = len(df)
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    dropped = before - len(df)
    if dropped:
        print(f"  {path.name}: dropped {dropped}/{before} unparseable rows")

    df["source"] = "mt_broker"
    return df[["timestamp", "open", "high", "low", "close", "volume", "source"]]


def import_mt_exports(export_dir: Path, symbol: str, data_dir: Path,
                       output_symbol: str = "XAUUSD") -> dict:
    """For each of the 6 timeframes, load <symbol>_<tf>.csv (bulk
    backfill, optional) and <symbol>_<tf>_live.csv (ongoing updates,
    optional - at least one of the two must exist), merge them, and
    merge into data/candles/<output_symbol>_<tf>.parquet the same way
    build_history.py's live_update.py merges new Dukascopy candles."""
    candles_dir = data_dir / "candles"
    candles_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for tf in TIMEFRAME_NAMES:
        frames = []
        bulk_path = export_dir / f"{symbol}_{tf}.csv"
        live_path = export_dir / f"{symbol}_{tf}_live.csv"

        if bulk_path.exists():
            frames.append(_read_mt_csv(bulk_path))
        if live_path.exists():
            frames.append(_read_mt_csv(live_path))

        if not frames:
            results[tf] = {"status": "no_file", "candles": 0}
            continue

        new_df = pd.concat(frames, ignore_index=True)
        new_df = new_df.drop_duplicates(subset="timestamp").sort_values("timestamp")

        out_path = candles_dir / f"{output_symbol}_{tf}.parquet"
        merged = merge_with_existing(new_df, out_path)
        merged.to_parquet(out_path, index=False)
        results[tf] = {
            "status": "ok", "candles": len(merged), "new_rows_seen": len(new_df),
            "start": merged["timestamp"].min().isoformat() if len(merged) else None,
            "end": merged["timestamp"].max().isoformat() if len(merged) else None,
        }
        print(f"{tf}: +{len(new_df)} rows from MT export -> {len(merged)} total -> {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True,
                         help="path to the gold_export folder under your MT4/5 terminal's "
                              "MQL4/Files or MQL5/Files directory (File > Open Data Folder in the terminal)")
    parser.add_argument("--symbol", required=True,
                         help="exact broker symbol name as it appears in the exported filenames, "
                              "e.g. XAUUSD or XAUUSD.a - check the gold_export folder if unsure")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-symbol", default="XAUUSD",
                         help="symbol name to write into data/candles/ - defaults to XAUUSD "
                              "regardless of your broker's actual symbol suffix, so the rest of "
                              "the pipeline doesn't need to know about broker-specific naming")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    if not export_dir.exists():
        raise SystemExit(f"{export_dir} does not exist - check the path (File > Open Data Folder "
                          "in MT4/5, then MQL4 or MQL5 > Files > gold_export)")

    data_dir = Path(args.data_dir)
    with track("mt_import", path=data_dir / "heartbeats.json"):
        results = import_mt_exports(export_dir, args.symbol, data_dir, args.output_symbol)

    found_any = any(r["status"] == "ok" for r in results.values())
    if not found_any:
        raise SystemExit(
            f"no {args.symbol}_<tf>.csv or {args.symbol}_<tf>_live.csv files found in {export_dir} - "
            "check --symbol matches the exported filenames exactly, and that the export script actually ran"
        )


if __name__ == "__main__":
    main()
