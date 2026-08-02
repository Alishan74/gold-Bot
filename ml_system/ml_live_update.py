"""
The ML challenger's equivalent of live_update.py: compute today's
ML-driven signal, log it to THIS system's own journal, and self-heal
(auto-retrain) when triggered - all reusing the rule-based system's
already-verified machinery (signal_journal.py, unmodified) rather than a
parallel implementation, per live_signal.py's docstring.

Does NOT fetch new ticks itself - candles are SHARED, read-only, from
the rule-based system's data/candles/ (or wherever --candles-dir points).
Run build_history.py / live_update.py (or scripts/supervise.py wrapping
them) to keep that data fresh; this script only needs it to already be
there. There is no reason to run a second Dukascopy backfill just
because a second system is reading the same ticks.

WHERE "learning from live data, not just history" actually happens, and
why it's NOT simply "feed the journal's outcomes back into training":
every retrain (train.py) re-labels EVERY candle in data/candles/,
including whatever the shared candle-fetching pipeline appended since
the last retrain - so newly observed real market behavior becomes new
training examples automatically, the same way it would for any
historical candle. The signal JOURNAL (this system's own log of what it
actually predicted and what happened) is deliberately NOT fed back into
training directly - it's a SELECTED subset (only candles the model
itself chose to score highly), and training a model on its own past
selections would bias it toward reinforcing whatever it already
believed instead of learning from the full, honest distribution of
outcomes. The journal's real job is what it already does for the
rule-based system: self-assessment and self-healing suspension.

Self-heal (auto-retrain) triggers, identical in spirit to
signal_journal.should_self_heal() used for the rule-based system:
  1. This system's own live journal shows >= REBUILD_TRIGGER_DECAYING_COUNT
     pattern/timeframe/direction combos DECAYING (i.e. "ml_model" on some
     timeframe/direction, since that's this system's only pattern name -
     see live_signal.PATTERN_NAME).
  2. The model registry hasn't been retrained in
     REBUILD_TRIGGER_MAX_AGE_DAYS, regardless of live drift - a
     time-based backstop so newly accumulated candle history eventually
     gets folded in even if nothing has drifted yet.

Usage:
    python ml_system/ml_live_update.py --symbol XAUUSD \\
        --candles-dir data/candles --data-dir ml_data --registry-dir ml_registry
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import live_signal  # noqa: E402
import model_registry  # noqa: E402
from train import RR_GRID, train_all  # noqa: E402

from circuit_breaker import check_circuit_breaker  # noqa: E402
from heartbeat import read_heartbeats, track  # noqa: E402
from news_calendar import load_upcoming  # noqa: E402
from signal_journal import (  # noqa: E402
    load_journal, log_signal, should_self_heal, summary, update_journal,
)


def _load_candles(candles_dir: Path, symbol: str, tail: int = 300) -> dict[str, pd.DataFrame]:
    candles_by_tf = {}
    for path in sorted(candles_dir.glob(f"{symbol}_*.parquet")):
        tf = path.stem.replace(f"{symbol}_", "")
        candles_by_tf[tf] = pd.read_parquet(path).sort_values("timestamp").tail(tail).reset_index(drop=True)
    return candles_by_tf


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--candles-dir", default="data/candles",
                         help="shared, read-only candle source - normally the rule-based system's data/candles/")
    parser.add_argument("--news-data-dir", default="data",
                         help="shared, read-only root for the forward news calendar (data/events/upcoming.parquet) "
                              "- normally the rule-based system's data/, same sharing rationale as candles")
    parser.add_argument("--data-dir", default="ml_data", help="this system's own journal/heartbeats")
    parser.add_argument("--registry-dir", default="ml_registry", help="this system's own model storage")
    parser.add_argument("--n-splits", type=int, default=5, help="purged walk-forward folds used by any triggered retrain")
    parser.add_argument("--no-self-heal", action="store_true",
                         help="never auto-trigger a retrain, even if triggered by drift or staleness")
    parser.add_argument("--rr-grid", default=None,
                         help="comma-separated R:R tiers used by any SELF-HEAL-TRIGGERED retrain, e.g. '1.5,4,8' - "
                              f"overrides train.py's default {len(RR_GRID)}-tier grid. An automatic retrain "
                              "kicked off mid-run should not silently balloon to a ~10x-longer job without the "
                              "operator having chosen that - same reasoning as --n-splits above.")
    args = parser.parse_args()

    rr_grid = tuple(float(x) for x in args.rr_grid.split(",")) if args.rr_grid else RR_GRID

    data_dir = Path(args.data_dir)
    candles_dir = Path(args.candles_dir)
    registry_dir = Path(args.registry_dir)

    with track("ml_live_update", path=data_dir / "heartbeats.json"):
        candles_by_tf = _load_candles(candles_dir, args.symbol)
        if not candles_by_tf:
            raise SystemExit(f"no candle data found in {candles_dir} - run build_history.py first "
                              f"(this system reads the SAME candles the rule-based one does)")

        update_journal(candles_by_tf, data_dir)

        # Self-heal: retrain automatically if this system's own live
        # journal shows enough drift, or the registry has gone stale -
        # BEFORE computing today's signal, so it benefits from a fresh
        # retrain the same run. Mirrors should_self_heal()'s use in
        # live_update.py exactly, just against this system's own state.
        if not args.no_self_heal:
            journal_df = load_journal(data_dir)
            lib_view_dir = registry_dir / "lib_view"
            heartbeats = read_heartbeats(data_dir / "heartbeats.json")
            train_heartbeats = {"build_pattern_library": heartbeats.get("ml_train")}  # same age-check shape
            trigger, reason = should_self_heal(journal_df, lib_view_dir, args.symbol, train_heartbeats)
            if trigger:
                print(f"SELF-HEAL: triggering automatic ML retrain - {reason} "
                      f"({len(rr_grid)} R:R tier(s): {list(rr_grid)})")
                with track("ml_train", path=data_dir / "heartbeats.json"):
                    train_all(args.symbol, candles_dir, registry_dir, args.n_splits, rr_grid=rr_grid)
                candles_by_tf = _load_candles(candles_dir, args.symbol)  # unchanged, but keep the pattern explicit

        suspended = live_signal.load_suspended_ml(data_dir, registry_dir, args.symbol)
        if suspended:
            print(f"self-heal: {len(suspended)} pattern/timeframe/direction combo(s) suspended "
                  f"(DECAYING live): {sorted(suspended)}")

        # Loaded once, reused for both the circuit breaker AND loss/win
        # attribution (context_penalty) below.
        journal = load_journal(data_dir)

        # Portfolio-level circuit breaker - breakered independently
        # against THIS system's own journal (see circuit_breaker.py) - a
        # tripped rule-based breaker does not automatically halt this
        # system, and vice versa; each engine's realized live outcomes
        # are its own responsibility.
        breaker = check_circuit_breaker(journal)
        if breaker["tripped"]:
            print(f"CIRCUIT BREAKER TRIPPED: {breaker['reasons']} - forcing HOLD system-wide until a human clears it")

        upcoming = load_upcoming(Path(args.news_data_dir))
        # One independent signal PER R:R TIER now (see live_signal.py's
        # module docstring / compute_ml_signal() - a scalp-tier and a
        # swing-tier signal off the same market state are never blended
        # into one composite anymore) - log EACH one. log_signal()'s own
        # dedup keys on (timeframe, pattern, entry_candle_timestamp), and
        # `pattern` is tier-specific (ml_model_rr<tag>), so logging
        # multiple tiers here in the same run can never collide with
        # each other.
        signals = live_signal.compute_ml_signal(candles_by_tf, registry_dir, args.symbol, upcoming=upcoming,
                                                  suspended=suspended, circuit_breaker=breaker, journal=journal)
        print(f"current ML signals ({len(signals)} R:R tier(s)):")
        for sig in signals:
            print(json.dumps(sig, indent=2, default=str))
            log_signal(sig, candles_by_tf, data_dir)

        print("ML journal:", json.dumps(summary(load_journal(data_dir))))


if __name__ == "__main__":
    main()
