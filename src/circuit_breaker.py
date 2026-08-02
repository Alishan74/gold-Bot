"""
Portfolio-level circuit breaker - the thing a real trading desk, broker,
or exchange has that a collection of individually-gated patterns/models
does not: a hard, system-wide stop that doesn't care whether any
particular pattern still shows `qualifies: true`. Every pattern and every
ML model in this system is already gated on its OWN mined/validated
statistics - but statistics describe the past. A strategy can clear
every individual gate and still be in the middle of an actively
catastrophic run RIGHT NOW, for reasons the mined stats haven't caught
up to yet (a regime break severe enough that self-healing hasn't
finished reacting, a data problem, anything). That's exactly the gap a
circuit breaker closes: it looks at REALIZED, live outcomes only - not
predictions, not backtests - and forces a hard HOLD across the entire
system until a human clears it.

Three independent triggers, any one is sufficient to trip:

1. **Consecutive loss streak** (>= MAX_CONSECUTIVE_LOSSES). Under a
   genuinely ~60%+ win-rate system, a streak this long has a probability
   under 0.1% if trades were independent draws from the system's own
   claimed win rate (0.4^8 ≈ 0.07%) - so hitting it is strong evidence
   something is systematically wrong, not normal variance.
2. **Peak-to-trough drawdown** (cumulative realized R falling more than
   MAX_DRAWDOWN_R below its running peak). The system is DESIGNED for
   strongly positive expectancy (a 60%-win-rate pattern at fixed 1:4 R:R
   has expectancy 0.6*4 - 0.4*1 = +2.0R per trade) - a drawdown this
   large is inconsistent with normal variance around a genuine edge.
3. **Too much simultaneous open risk** (>= MAX_OPEN_RISK_R open trades'
   worth of risk at once, each open trade risking a fixed 1R by
   construction). Prevents an unbounded pile-up of simultaneous exposure
   if many patterns/timeframes fire in a short window.

Deliberately evaluated ONLY against REALIZED outcomes (win/loss/expired)
plus open COUNT - never unrealized P&L on open positions, which would
require a live price feed this module doesn't have and shouldn't need:
the breaker must be fully deterministic and auditable from the journal
alone, the same file everything else in this system's self-assessment
already reads.

Tripping the breaker is a HARD override: compute_signal() /
compute_ml_signal() force HOLD regardless of what any individual pattern
or model says, the same way a suspended pattern is excluded regardless
of what the mined library says - see signal_engine.py's `circuit_breaker`
parameter. A human has to look at what happened and clear it; nothing in
this system auto-clears a tripped breaker, on purpose.
"""
from __future__ import annotations

import pandas as pd

MAX_CONSECUTIVE_LOSSES = 8
MAX_DRAWDOWN_R = -15.0
MAX_OPEN_RISK_R = 6.0  # each open trade risks a fixed 1R, so this is "how many can be open at once"


def _max_drawdown_r(cumulative_r: pd.Series) -> float:
    """Largest peak-to-trough decline in a cumulative-R series - the
    standard drawdown definition (running max minus current value,
    most-negative result kept). Returns 0.0 for an empty/flat series."""
    if cumulative_r.empty:
        return 0.0
    running_peak = cumulative_r.cummax()
    drawdown = cumulative_r - running_peak  # always <= 0
    return float(drawdown.min())


def check_circuit_breaker(journal: pd.DataFrame) -> dict:
    """Evaluates all three triggers against `journal` (one engine's own
    signal_journal.load_journal() output - the rule-based and ML
    challenger systems each have their own journal and are breakered
    independently, matching how they're self-healed independently).
    Returns {"tripped": bool, "reasons": [str, ...], "current_streak":
    ..., "max_drawdown_r": ..., "open_risk_r": ...} - the numbers are
    always included, tripped or not, so the dashboard can show how close
    the system is to tripping, not just a binary state."""
    if journal.empty:
        return {"tripped": False, "reasons": [], "current_streak": None,
                "max_drawdown_r": 0.0, "open_risk_r": 0.0}

    reasons = []

    resolved = journal[journal["status"].isin(["win", "loss"])].sort_values("resolved_at_utc")
    streak_type, streak_len = None, 0
    for outcome in reversed(resolved["status"].tolist()):
        if streak_type is None:
            streak_type, streak_len = outcome, 1
        elif outcome == streak_type:
            streak_len += 1
        else:
            break
    if streak_type == "loss" and streak_len >= MAX_CONSECUTIVE_LOSSES:
        reasons.append(f"{streak_len} consecutive losses (>= {MAX_CONSECUTIVE_LOSSES})")

    max_dd = 0.0
    if not resolved.empty:
        cumulative_r = resolved["actual_r"].astype(float).cumsum()
        max_dd = _max_drawdown_r(cumulative_r)
        if max_dd <= MAX_DRAWDOWN_R:
            reasons.append(f"peak-to-trough drawdown {max_dd:.1f}R (<= {MAX_DRAWDOWN_R:.1f}R)")

    open_count = int((journal["status"] == "open").sum())
    open_risk_r = float(open_count)  # each open trade risks a fixed 1R by construction
    if open_risk_r >= MAX_OPEN_RISK_R:
        reasons.append(f"{open_count} trades open simultaneously (>= {MAX_OPEN_RISK_R:.0f}R aggregate open risk)")

    return {
        "tripped": bool(reasons), "reasons": reasons,
        "current_streak": {"type": streak_type, "length": streak_len} if streak_type else None,
        "max_drawdown_r": round(max_dd, 2), "open_risk_r": open_risk_r,
    }
