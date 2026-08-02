"""
Diagnostic: summarize the actual win-rate distribution across a mined
pattern_library/*.json set, not just the qualify/fail count
build_pattern_library.py prints. Answers "how close did anything get" -
the difference between "the 60%-at-1:4 bar is just genuinely very hard to
clear on real data" (patterns clustering in the 40-55% range, working as
intended) and "something is broken" (every pattern sitting near a fixed/
degenerate value, or resolved counts that don't make sense).

Usage:
    python scripts/inspect_library.py --symbol XAUUSD
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _stats_list(entry: dict) -> list[tuple[str, dict]]:
    if "stats" in entry:
        return [("", entry["stats"])]
    out = []
    if "as_long" in entry:
        out.append((" (long)", entry["as_long"]))
    if "as_short" in entry:
        out.append((" (short)", entry["as_short"]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--lib-dir", default="pattern_library")
    args = parser.parse_args()

    lib_dir = Path(args.lib_dir)
    files = sorted(lib_dir.glob(f"{args.symbol}_*.json"))
    if not files:
        raise SystemExit(f"no files found in {lib_dir} matching {args.symbol}_*.json")

    for path in files:
        tf = path.stem.replace(f"{args.symbol}_", "")
        library = json.loads(path.read_text())
        rows = []  # (name, suffix, win_rate, resolved, oos_win_rate, oos_resolved, qualifies)
        zero_resolved = 0
        zero_samples = 0
        for name, entry in library.items():
            if name == "_meta":
                continue
            for suffix, stats in _stats_list(entry):
                resolved = stats.get("resolved", 0)
                samples = stats.get("samples", 0)
                if samples == 0:
                    zero_samples += 1
                    continue
                if resolved == 0:
                    zero_resolved += 1
                    continue
                wr = stats.get("win_rate")
                oos = stats.get("out_of_sample", {}) or {}
                rows.append((
                    name + suffix, wr, resolved,
                    oos.get("win_rate"), oos.get("resolved", 0),
                    stats.get("qualifies", False),
                ))

        rows.sort(key=lambda r: (r[1] if r[1] is not None else -1), reverse=True)

        print(f"\n=== {tf} ===")
        print(f"total pattern-entries: {len(rows) + zero_resolved + zero_samples} "
              f"  (zero-samples: {zero_samples}, zero-resolved: {zero_resolved}, "
              f"has-stats: {len(rows)})")
        if not rows:
            print("  NOTHING resolved a single trade on this timeframe - definitely worth digging into.")
            continue

        win_rates = [r[1] for r in rows if r[1] is not None]
        resolved_counts = [r[2] for r in rows]
        print(f"  win_rate distribution: min={min(win_rates):.3f} max={max(win_rates):.3f} "
              f"mean={sum(win_rates)/len(win_rates):.3f}")
        print(f"  resolved-count distribution: min={min(resolved_counts)} max={max(resolved_counts)} "
              f"mean={sum(resolved_counts)/len(resolved_counts):.1f}")

        buckets = {"<40%": 0, "40-50%": 0, "50-55%": 0, "55-60%": 0, "60-65%": 0, ">=65%": 0}
        for wr in win_rates:
            if wr < 0.40: buckets["<40%"] += 1
            elif wr < 0.50: buckets["40-50%"] += 1
            elif wr < 0.55: buckets["50-55%"] += 1
            elif wr < 0.60: buckets["55-60%"] += 1
            elif wr < 0.65: buckets["60-65%"] += 1
            else: buckets[">=65%"] += 1
        print(f"  buckets: {buckets}")

        print("  top 15 by win_rate (name, win_rate, resolved, oos_win_rate, oos_resolved, qualifies):")
        for name, wr, resolved, oos_wr, oos_resolved, qualifies in rows[:15]:
            wr_s = f"{wr:.3f}" if wr is not None else "None"
            oos_s = f"{oos_wr:.3f}" if oos_wr is not None else "None"
            print(f"    {name:45s} wr={wr_s:6s} n={resolved:6d}  oos_wr={oos_s:6s} oos_n={oos_resolved:4d}  qualifies={qualifies}")

        real_sample_rows = [r for r in rows if r[2] >= 30]
        print(f"  entries with resolved>=30 (a real sample, not a lucky handful): {len(real_sample_rows)}")
        if real_sample_rows:
            print("  top 15 by win_rate AMONG resolved>=30:")
            for name, wr, resolved, oos_wr, oos_resolved, qualifies in real_sample_rows[:15]:
                wr_s = f"{wr:.3f}" if wr is not None else "None"
                oos_s = f"{oos_wr:.3f}" if oos_wr is not None else "None"
                print(f"    {name:45s} wr={wr_s:6s} n={resolved:6d}  oos_wr={oos_s:6s} oos_n={oos_resolved:4d}  qualifies={qualifies}")


if __name__ == "__main__":
    main()
