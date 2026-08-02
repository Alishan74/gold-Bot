"""
Print the out-of-sample breakdown for the top N setups in an
explore_setups.py output file - the console output from explore_setups.py
itself only shows the full-sample number, which is exactly the number
most likely to look good by chance given how many conjunctions/R:R
combinations that script tests with no FDR correction. This is the next
thing to actually look at before trusting anything in that report.

Usage:
    python scripts/show_explored.py explored_setups\\XAUUSD_1d.json --top 15
"""
from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    data = json.loads(open(args.file).read())
    meta = data["_meta"]
    print(f"meta: {meta}")
    if meta.get("news_note"):
        print(f"\n{meta['news_note']}\n")
    for s in data["setups"][: args.top]:
        rr = s["best_rr_ratio"]
        stats = s["by_rr_ratio"][str(rr)]
        excluded = stats.get("n_excluded_for_news", 0)
        print(f"\n[{s['direction']:8s}] {' + '.join(s['primitives'])}")
        print(f"   best @ 1:{rr}   full: win_rate={stats['win_rate']} n={stats['resolved']}"
              f"   |   out_of_sample: win_rate={stats['oos_win_rate']} n={stats['oos_resolved']}"
              + (f"   |   ({excluded} occurrences excluded for nearby news)" if excluded else ""))


if __name__ == "__main__":
    main()
