"""
Combine several explore_setups.py --seed-families batch outputs (each run
against a disjoint slice of the primitive catalog as depth-1 seeds, each
written to its OWN --out-dir so batches never clobber each other) back
into one file per timeframe - the counterpart to
discovery_search.search_conjunctions' seed_primitives parameter, whose
own docstring explains why splitting a search this way is a strictly
BROADER union of depth-1 starting points than a single unsplit run, not
a narrower one, so merging is a safe union: dedup by (primitive set,
direction) exactly like search_conjunctions' own final dedup, keeping
the higher best_expectancy_r_after_costs on the rare case a conjunction
was independently found via more than one batch, then re-sort and
re-truncate to --top-n.

Usage:
    python scripts/merge_explored_setups.py --symbol XAUUSD \
        --batch-dirs explored_setups_batch_a,explored_setups_batch_b \
        --timeframes 15min --out-dir explored_setups_relaxed --top-n 300
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from atomic_io import atomic_write_text  # noqa: E402


def merge_timeframe(symbol: str, tf: str, batch_dirs: list[Path], top_n: int) -> dict | None:
    by_key: dict[tuple, dict] = {}
    meta_parts = []
    total_n_tested = 0
    for bd in batch_dirs:
        path = bd / f"{symbol}_{tf}.json"
        if not path.exists():
            print(f"  note: {path} not found - skipping this batch for {tf}")
            continue
        raw = json.loads(path.read_text())
        meta = raw.get("_meta", {})
        meta_parts.append(meta)
        total_n_tested += meta.get("n_tested", 0)
        for row in raw.get("setups", raw if isinstance(raw, list) else []):
            key = (frozenset(row["primitives"]), row["direction"])
            if key not in by_key or row["best_expectancy_r_after_costs"] > by_key[key]["best_expectancy_r_after_costs"]:
                by_key[key] = row

    if not by_key:
        return None

    merged_rows = sorted(by_key.values(), key=lambda r: r["best_expectancy_r_after_costs"], reverse=True)
    kept = merged_rows[:top_n]
    merged_meta = {
        "merged_from_batches": len(meta_parts),
        "n_tested_total_across_batches": total_n_tested,
        "n_unique_conjunctions_merged": len(by_key),
    }
    if meta_parts:
        merged_meta.update({k: v for k, v in meta_parts[0].items()
                             if k not in ("n_tested", "n_conjunctions_with_enough_samples")})
    return {"_meta": merged_meta, "setups": kept}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--batch-dirs", required=True, help="comma-separated directories, one per batch")
    parser.add_argument("--timeframes", required=True, help="comma-separated (e.g. 15min,5min)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=300)
    args = parser.parse_args()

    batch_dirs = [Path(d.strip()) for d in args.batch_dirs.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tf in timeframes:
        result = merge_timeframe(args.symbol, tf, batch_dirs, args.top_n)
        if result is None:
            print(f"{tf}: no batch output found in any of {batch_dirs} - nothing to merge")
            continue
        out_path = out_dir / f"{args.symbol}_{tf}.json"
        atomic_write_text(out_path, json.dumps(result, indent=2, default=str))
        print(f"{tf}: merged {result['_meta']['merged_from_batches']} batch(es), "
              f"{result['_meta']['n_unique_conjunctions_merged']} unique conjunctions, "
              f"kept top {len(result['setups'])} -> {out_path}")


if __name__ == "__main__":
    main()
