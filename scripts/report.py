"""
Plain-text self-assessment report: overall scorecard and per-pattern
live credibility - the exact numbers the dashboard's Self-Assessment
panel shows, for when you want the answer to "is this actually working"
without opening a browser (cron job, piped into a log, checked over SSH).

Reads nothing but the existing signal journal (signal_journal.py) - no
new data source, no network access needed.

Usage:
    python scripts/report.py
    python scripts/report.py --symbol XAUUSD --data-dir data --lib-dir pattern_library
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from circuit_breaker import check_circuit_breaker  # noqa: E402
from signal_journal import (  # noqa: E402
    context_scorecard, equity_curve, load_journal, overall_scorecard, pattern_scorecard,
)


def _fmt_pct(p) -> str:
    return "n/a" if p is None else f"{p * 100:.1f}%"


def _fmt_r(r) -> str:
    return "n/a" if r is None else f"{r:+.2f}R"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--lib-dir", default="pattern_library")
    parser.add_argument("--discovered-dir", default="discovered_patterns")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    lib_dir = Path(args.lib_dir)
    discovered_dir = Path(args.discovered_dir)
    journal = load_journal(data_dir)

    print("=" * 70)
    print(f"  SELF-ASSESSMENT REPORT — {args.symbol}")
    print("=" * 70)

    # Circuit breaker status FIRST, before anything else - checked
    # regardless of whether any signals have been logged yet (matches
    # the dashboard's always-visible-when-tripped banner). This is the
    # one piece of state someone checking in via cron/SSH without the
    # dashboard open most needs to see immediately, not buried below the
    # scorecard.
    breaker = check_circuit_breaker(journal)
    if breaker["tripped"]:
        print(f"\n⛔ CIRCUIT BREAKER TRIPPED - signal forced to HOLD until a human clears it")
        print(f"   Reasons: {'; '.join(breaker['reasons'])}")
        if breaker["current_streak"]:
            st = breaker["current_streak"]
            print(f"   Streak: {st['length']} {st['type']}, "
                  f"max drawdown: {breaker['max_drawdown_r']:+.1f}R, open risk: {breaker['open_risk_r']:.0f}R")
    else:
        print(f"\n✓ Circuit breaker: healthy (max drawdown {breaker['max_drawdown_r']:+.1f}R, "
              f"open risk {breaker['open_risk_r']:.0f}R)")

    sc = overall_scorecard(journal)
    if not sc["total"]:
        print("\nNo signals logged yet - nothing to assess. Run live_update.py "
              "until it starts emitting actionable (non-HOLD) signals.")
        return

    print(f"\nSignals logged: {sc['total']}  (open: {sc['open']}, expired: {sc['expired']})")
    print(f"Resolved: {sc['win'] + sc['loss']}  ({sc['win']} win / {sc['loss']} loss)")
    print(f"Raw live win rate:          {_fmt_pct(sc['live_win_rate'])}")
    print(f"Conservative (95% lower):   {_fmt_pct(sc['win_rate_wilson_lower'])}"
          f"  <- this is the number to trust, not the raw one, on small samples")
    print(f"Total realized R:           {_fmt_r(sc['total_r'])}"
          f"  (net of costs: {_fmt_r(sc['total_r_after_costs'])})")
    print(f"Avg R / resolved trade:     {_fmt_r(sc['avg_r'])}")
    if sc["current_streak"] and sc["current_streak"]["length"]:
        st = sc["current_streak"]
        print(f"Current streak:             {st['length']} {st['type']}{'es' if st['type']=='loss' and st['length']>1 else ('s' if st['length']>1 else '')} in a row")
    print(f"Best / worst single trade:  {_fmt_r(sc['best_trade_r'])} / {_fmt_r(sc['worst_trade_r'])}")

    curve = equity_curve(journal)
    if curve:
        print(f"\nEquity curve: {len(curve)} resolved trades, "
              f"running at {_fmt_r(curve[-1]['cumulative_r'])} cumulative R "
              f"(net {_fmt_r(curve[-1]['cumulative_r_after_costs'])})")

    patterns = pattern_scorecard(journal, lib_dir, args.symbol, discovered_dir)
    if patterns:
        print("\n" + "-" * 70)
        print(f"{'PATTERN':<28}{'TF':<6}{'N':<5}{'LIVE WR':<10}{'MINED WR':<10}{'R':<9}{'LABEL'}")
        print("-" * 70)
        for p in patterns:
            score_str = f"{p['score']:.1f}" if p["score"] is not None else "n/a"
            print(f"{p['pattern']:<28}{p['timeframe']:<6}{p['live_samples']:<5}"
                  f"{_fmt_pct(p['live_win_rate']):<10}{_fmt_pct(p['mined_win_rate']):<10}"
                  f"{_fmt_r(p['total_r']):<9}{p['label']} ({score_str})")
        decaying = [p for p in patterns if p["label"] == "DECAYING"]
        if decaying:
            print(f"\n⚠ {len(decaying)} pattern(s) DECAYING live - see reasons above / dashboard for detail")

    context_rows = [r for r in context_scorecard(journal) if r["decaying"]]
    if context_rows:
        print("\n" + "-" * 70)
        print("LOSS/WIN ATTRIBUTION - context proven to underperform a pattern's own live trades:")
        print("-" * 70)
        for r in context_rows:
            print(f"  {r['pattern']}/{r['timeframe']}/{r['direction']} - {r['dimension']}={r['bucket']}: "
                  f"{_fmt_pct(r['win_rate'])} win rate on {r['samples']} trades (upper bound "
                  f"{_fmt_pct(r['win_rate_wilson_upper'])}) vs {_fmt_pct(r['baseline_win_rate'])} for this "
                  f"pattern's other live trades - live signals in this exact context now get a confidence "
                  f"discount (see the main README's Loss/Win Attribution section)")
    print()


if __name__ == "__main__":
    main()
