"""
Compare the most recent (live) candles on each timeframe against the mined
pattern library and emit a composite BUY/SELL/HOLD signal with a concrete
1:4 risk:reward trade plan (entry/stop/target), a freshness/expiry label,
and - when relevant - a high-impact-news proximity warning.

Two hard gates, enforced here independently of build_pattern_library.py
(so a bug upstream can't silently let a bad signal through):
  1. Risk:reward is always exactly 1:4 - stop/target are computed fresh
     here from current ATR, never read from the mined stats.
  2. A pattern's mined win rate under that 1:4 structure must be >= 60%
     (`qualifies: true`) or it is skipped entirely - no partial credit,
     no "weak signal" fallback.

Freshness/expiry: pattern detection only ever looks at CLOSED candles
(the same thing risk_reward.py backtested), so a signal is only as good
as how recently its trigger candle closed. Every signal reports its age
and is labeled FRESH / AGING / EXPIRED based on how many trigger-
timeframe candles have passed since - a signal you see for the first
time after the setup has clearly moved on is marked EXPIRED rather than
presented as actionable. This deliberately does NOT try to detect
patterns on a still-forming candle - that would use data the backtest
never saw, so its true win rate would be unverified.

News proximity: using the pattern's own median time-to-resolution (mined
by risk_reward.py) and the forward-looking FRED release calendar
(news_calendar.py), a signal reports whether high-impact news is
scheduled to land before the trade would typically resolve, and, if
enough historical occurrences overlapped with news before, what this
pattern's win rate looked like specifically when that happened
(`news_conditioning` in the mined library, from build_pattern_library.py).

Pure comparison against precomputed stats - no historical recomputation
happens here. Re-run build_pattern_library.py periodically (e.g. weekly,
via live_update.py) to fold newly observed live data back into the
pattern stats.

Usage:
    python src/signal_engine.py --symbol XAUUSD
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from build_history import TIMEFRAME_MINUTES
from combo_patterns import detect_combo_events
from discover_patterns import is_discovered_pattern_active
from fundamental_patterns import detect_fundamental_events
from news_calendar import label_signal_news_risk, load_upcoming
from patterns import detect_all
from regime import combined_regime
from risk_reward import MIN_RESOLVED_SAMPLES, MIN_WIN_RATE, RR_RATIO, STOP_ATR_MULTIPLE, atr, wilson_lower_bound
from session_patterns import detect_session_events
from smc_patterns import detect_smc_events
from support_resistance import detect_support_resistance_events

# How much weight each timeframe's signal carries in the composite score.
# Higher timeframes get more weight - they're less noisy. This only
# affects ranking among already-qualifying patterns, never the hard gate.
TIMEFRAME_WEIGHTS = {
    "1min": 0.5,
    "5min": 0.75,
    "15min": 1.0,
    "1h": 1.5,
    "4h": 2.0,
    "1d": 2.5,
}

# Regime-conditioned mining (regime.py, build_pattern_library.py's
# "by_regime") shows whether a pattern actually has a proven edge in the
# CURRENT market regime specifically, not just blended across 20 years.
# Deliberately a WEIGHT penalty, not a hard gate: an additional hard gate
# here would cut deeply into signal frequency on an already-conservative
# system for every pattern that simply doesn't have enough regime-
# specific samples yet (most of them, especially for rarer regimes) -
# only applied when there's ENOUGH regime-specific history to say the
# pattern has genuinely been tested in this regime and failed, not
# merely "we don't know yet" (see _regime_stats_for_pattern below).
REGIME_MISMATCH_PENALTY = 0.5

# Same reasoning and same magnitude as REGIME_MISMATCH_PENALTY immediately
# above, applied to discover_patterns.py's cross-timeframe confirmation
# check instead of regime-conditioning: a discovered pattern whose SAME
# conjunction was explicitly checked against the next-coarser timeframe's
# own history and did NOT hold up there gets its live weight discounted,
# never hard-excluded - many genuinely valid discovered patterns are
# honestly single-timeframe-scoped (session-timing components, fine-
# grained microstructure conjunctions that only make sense at 1min/5min
# resolution), so failing a cross-timeframe check is real but soft
# evidence, not proof the pattern itself is false. Only ever applied when
# the check actually RAN (cross_timeframe.checked is true) - "no sibling
# timeframe was available to check" is explicitly NOT the same as
# "checked and failed," and must never be penalized.
CROSS_TIMEFRAME_MISMATCH_PENALTY = 0.5

# A signal's freshness is measured in units of its own trigger candle's
# duration - e.g. on the 1h timeframe, 1 unit = 1 hour old.
FRESH_MAX_CANDLES = 1.0    # still within the candle that triggered it
AGING_MAX_CANDLES = 3.0    # late, but the setup may still be intact
# beyond AGING_MAX_CANDLES -> EXPIRED


@dataclass
class Contribution:
    timeframe: str
    pattern: str
    direction: str
    win_rate: float
    win_rate_wilson_lower: float
    samples: int
    weight: float
    used_for_signal: bool = False
    median_candles_to_resolve: float | None = None
    news_conditioning: dict | None = None
    regime: str | None = None
    regime_win_rate: float | None = None
    regime_samples: int | None = None
    regime_qualifies: bool | None = None
    regime_penalized: bool = False
    cross_timeframe_penalized: bool = False


@dataclass
class Signal:
    direction: str
    confidence: float
    contributions: list = field(default_factory=list)
    trade_plan: dict | None = None
    freshness: dict | None = None
    news_risk: dict | None = None
    primary_pattern: str | None = None
    primary_timeframe: str | None = None
    suspended_skipped: list = field(default_factory=list)
    circuit_breaker: dict | None = None
    context_penalty: dict | None = None

    def to_dict(self):
        return {
            "direction": self.direction,
            "confidence": round(self.confidence, 1),
            "risk_reward": f"1:{RR_RATIO:.0f}",
            "trade_plan": self.trade_plan,
            "freshness": self.freshness,
            "news_risk": self.news_risk,
            # Self-healing: pattern/timeframe/direction combos that WOULD
            # have contributed to this signal but were skipped because
            # signal_journal marks them DECAYING live right now (see
            # compute_signal's `suspended` param) - empty when nothing is
            # currently suspended.
            "suspended_skipped": self.suspended_skipped,
            # Portfolio-level circuit breaker (circuit_breaker.py) - when
            # tripped, this signal is forced to HOLD regardless of what
            # any individual pattern says, same override precedence as
            # suspension. None when the breaker wasn't even checked
            # (caller didn't pass one in); {"tripped": False, ...} when
            # checked and healthy.
            "circuit_breaker": self.circuit_breaker,
            # Loss/win ATTRIBUTION (signal_journal.context_scorecard/
            # context_penalty) - a confidence discount applied ONLY once
            # this specific pattern/timeframe/direction's live trades in
            # this signal's exact context (how many other patterns also
            # agreed, which trading session is active) have PROVEN,
            # Wilson-bound and hard-gated on sample size, to underperform
            # this pattern's OTHER live trades. None when not checked
            # (no journal was passed in) or the signal is HOLD;
            # {"multiplier": 1.0, "reasons": []} is the healthy/no-effect
            # case, always included so it's visible either way, not just
            # when it fires.
            "context_penalty": self.context_penalty,
            # The EXACT contribution that produced trade_plan/freshness/
            # news_risk above - NOT necessarily contributions[0]. With
            # multiple timeframes, the single highest-weight contribution
            # can be in the LOSING direction (many moderate same-direction
            # votes can outweigh one strong opposing one) - callers like
            # signal_journal.log_signal must use these fields, not guess
            # from contributions[0], or they can log a signal under a
            # pattern that had nothing to do with producing it.
            "primary_pattern": self.primary_pattern,
            "primary_timeframe": self.primary_timeframe,
            "contributions": [c.__dict__ for c in self.contributions],
        }


def _active_patterns_on_last_candle(candles: pd.DataFrame, events: pd.DataFrame | None,
                                     library: dict | None = None) -> list[str]:
    """`library`: this timeframe's merged pattern library (pattern_library/
    + discovered_patterns/, see load_inputs below) - needed here ONLY to
    find which `discovered__`-prefixed entries exist and what their own
    component primitives are (discovery_meta.primitives), so each one can
    be re-evaluated live via discover_patterns.is_discovered_pattern_active()
    (also passed this entry's discovery_meta.synthesized_expressions, for
    any component Layer 0 synthesized rather than hand-designed - see
    discover_patterns._evaluate_any_primitive). Whichever kind a
    primitive is, this is the IDENTICAL evaluation path the discovery run
    itself scored it with - live detection can never silently drift onto
    a different definition of a discovered pattern than the one it was
    validated under. None/omitted (a caller with no library yet) simply
    detects zero discovered patterns, same fail-open posture as every
    other optional lookup in this module."""
    base_flags = pd.concat(
        [detect_all(candles), detect_fundamental_events(candles, events), detect_session_events(candles),
         detect_support_resistance_events(candles), detect_smc_events(candles)],
        axis=1,
    )
    # Combos (combo_patterns.py) are detected from the SAME base_flags
    # and the SAME combo_pairs() list build_pattern_library.py mined
    # against - never redefined here, so live detection and mining can't
    # silently drift onto different combo definitions.
    combo_flags = detect_combo_events(base_flags)
    flags = pd.concat([base_flags, combo_flags], axis=1)
    last = flags.iloc[-1]
    active = [name for name, is_on in last.items() if bool(is_on)]

    if library:
        for name, entry in library.items():
            if not name.startswith("discovered__"):
                continue
            meta = entry.get("discovery_meta", {})
            primitives = meta.get("primitives")
            synthesized_expressions = meta.get("synthesized_expressions")
            if not primitives:
                continue
            try:
                is_active = is_discovered_pattern_active(primitives, candles, events, synthesized_expressions)
            except Exception:
                # A single malformed/corrupted/stale discovered_patterns
                # entry (e.g. a hand-edited file, a partial write, a
                # primitive name from a since-changed catalog) must never
                # take down live signal generation for every OTHER
                # pattern and timeframe - same fail-open posture as
                # load_suspended()/check_circuit_breaker_safe() elsewhere
                # in this module. Treated as "not active," not "crash."
                continue
            if is_active:
                active.append(name)
    return active


def _news_conditioning_for(entry: dict, direction: str) -> dict | None:
    """Pull the news_conditioning sub-object matching the direction actually
    used, from a raw pattern_library entry (see build_pattern_library.py)."""
    cond = entry.get("news_conditioning")
    if cond is None:
        return None
    if entry.get("direction") == "ambiguous":
        return cond.get("as_long") if direction == "bullish" else cond.get("as_short")
    return cond


def _regime_stats_for_pattern(library: dict, pattern: str, direction: str,
                               regime_label: str | None) -> dict | None:
    """Pull `pattern`'s mined stats for the CURRENT market regime, if
    known (regime.combined_regime) and if this pattern has a regime
    breakdown at all - "by_regime" in the library entry, populated by
    build_pattern_library.py for ATOMIC patterns only (combos skip
    regime-conditioning at mining time, same rationale as combos
    skipping news-conditioning - see that module's docstring), so this
    always returns None for a combo__ pattern name, by design."""
    if regime_label is None:
        return None
    entry = library.get(pattern)
    if entry is None:
        return None
    by_regime = entry.get("by_regime")
    if by_regime is None:
        return None
    if entry.get("direction") == "ambiguous":
        side = by_regime.get("as_long") if direction == "bullish" else by_regime.get("as_short")
    else:
        side = by_regime
    if side is None:
        return None
    return side.get(regime_label)


def _cross_timeframe_mismatch(library: dict, pattern: str) -> bool:
    """True only when `pattern` is a discovered__ pattern WHOSE cross-
    timeframe check (discover_patterns.discover_for_timeframe) actually
    ran (`checked: true`) and came back qualifies: false - see
    CROSS_TIMEFRAME_MISMATCH_PENALTY above for why that's a soft
    discount, not a hard gate. Always False for a hand-picked pattern
    (no discovery_meta at all) or a discovered pattern with no cross-
    timeframe entry (not checked, e.g. it's already on the coarsest
    available timeframe) - absence of a check is never treated as a
    failed one."""
    entry = library.get(pattern)
    if entry is None:
        return False
    cross_tf = entry.get("discovery_meta", {}).get("cross_timeframe")
    if not cross_tf or not cross_tf.get("checked"):
        return False
    return not cross_tf.get("qualifies", False)


def _qualifying_directions_ranked(library: dict, pattern: str) -> list[tuple[str, dict, dict | None]]:
    """Every direction of `pattern` that clears both hard gates, ranked
    best-win-rate-first. Re-checks `qualifies` here rather than trusting
    the stored flag blindly, since this is the last line of defense
    before a signal goes out. Returns ALL qualifying candidates (not
    just the best) so compute_signal() can skip a SUSPENDED top choice
    (see signal_journal.suspended_patterns) and fall back to another
    direction that independently qualifies, instead of dropping the
    pattern's vote entirely just because its best direction is currently
    decaying live."""
    entry = library.get(pattern)
    if entry is None:
        return []

    candidates = []
    if entry.get("direction") == "ambiguous":
        long_stats = entry.get("as_long")
        short_stats = entry.get("as_short")
        if long_stats:
            candidates.append(("bullish", long_stats))
        if short_stats:
            candidates.append(("bearish", short_stats))
    else:
        stats = entry.get("stats")
        if stats:
            candidates.append((entry["direction"], stats))

    qualifying = [
        (direction, stats) for direction, stats in candidates
        if stats.get("qualifies") and stats.get("win_rate") is not None
        and stats["win_rate"] >= MIN_WIN_RATE
    ]
    qualifying.sort(key=lambda c: c[1]["win_rate"], reverse=True)
    return [(direction, stats, _news_conditioning_for(entry, direction)) for direction, stats in qualifying]


def _qualifying_direction(library: dict, pattern: str):
    """Returns (direction, stats, news_conditioning) for the single BEST
    qualifying direction, or None - unchanged behavior/signature from
    before suspension support was added. See _qualifying_directions_ranked
    above for the full ranked list."""
    ranked = _qualifying_directions_ranked(library, pattern)
    return ranked[0] if ranked else None


def _trade_plan(candles: pd.DataFrame, direction: str, rr_ratio: float = RR_RATIO) -> dict | None:
    """entry uses the latest CLOSED candle's close as a proxy for "right
    now's" price. That's a deliberately different call from
    risk_reward.simulate_trades(), which prices historical entries at the
    NEXT candle's open (see risk_reward.py docstring) - the backtest fix
    matters because a historical candle's own close isn't something a
    backtest could have actually traded on. For a LIVE signal there is no
    equivalent look-ahead problem: the last close IS (approximately)
    "now," since there's no future data being peeked at - it's the best
    available real-time estimate of the next candle's open. Freshness
    labeling (below) is what guards against this approximation going
    stale if live_update.py runs late.

    `rr_ratio` defaults to the rule-based system's fixed 1:4 (every
    existing caller - signal_engine.compute_signal() - omits it and
    behaves identically to before this parameter existed). It exists so
    ml_system/live_signal.py's multi-tier signals (see
    ml_system/train.py's RR_GRID) can build a trade plan with the
    CORRECT target distance for whichever tier a given model was
    actually trained and validated on - stop distance (still
    STOP_ATR_MULTIPLE * ATR) is unaffected, only the target moves."""
    a = atr(candles)
    if a.empty or pd.isna(a.iloc[-1]) or a.iloc[-1] <= 0:
        return None
    entry = float(candles["close"].iloc[-1])
    risk = STOP_ATR_MULTIPLE * float(a.iloc[-1])
    if direction == "bullish":
        stop, target = entry - risk, entry + rr_ratio * risk
    else:
        stop, target = entry + risk, entry - rr_ratio * risk
    return {
        "entry": round(entry, 3),
        "stop_loss": round(stop, 3),
        "take_profit": round(target, 3),
        "risk": round(risk, 3),
        "rr_ratio": rr_ratio,
    }


def _trigger_candle_source(candles: pd.DataFrame) -> dict:
    """Which feed the TRIGGER candle (the one the pattern actually fired
    on) came from - "dukascopy" or "mt_broker" (see build_history.py's
    `source` column / SOURCE_PRIORITY, and mt_bridge/README.md). Matters
    because every pattern's mined win rate was measured almost entirely
    against Dukascopy-sourced history (build_history.merge_with_existing
    always prefers Dukascopy on any overlapping timestamp, permanently -
    a broker-sourced row only ever survives until Dukascopy catches up
    for that same hour) - but the MOST RECENT candle, right at live
    signal time, is often still broker-sourced simply because Dukascopy
    hasn't published that hour yet. Different brokers construct OHLC
    slightly differently from Dukascopy's own feed (different liquidity
    provider, spread construction, occasional requotes - see
    mt_bridge/README.md's own opening paragraph) - usually not enough to
    matter, but candlestick patterns are precise shape/ratio tests
    (e.g. hammer's lower-shadow->=2x-body rule), so a genuinely
    borderline candle COULD register differently on one feed than the
    other. This is reported transparently, not silently hard-blocked -
    if it were a REAL systematic problem for a given pattern, live-drift
    detection (signal_journal.detect_drift/suspended_patterns) already
    catches and suspends it the same as any other cause of live
    underperformance; this field exists so a human can factor
    "is this candle even Dukascopy-confirmed yet" into a specific,
    individual signal, not just wait for aggregate drift to show up."""
    if "source" not in candles.columns:
        return {"source": "unknown", "dukascopy_confirmed": None,
                "note": "this candle predates source-tracking - treat like any pre-mt_bridge candle"}
    source = candles["source"].iloc[-1]
    if source == "mt_broker":
        return {
            "source": "mt_broker", "dukascopy_confirmed": False,
            "note": "this trigger candle came from your broker's live feed, not yet from Dukascopy - "
                    "the pattern's mined win rate was measured almost entirely on Dukascopy-sourced "
                    "history, so a borderline candle shape could occasionally read differently here. "
                    "Dukascopy will silently replace this candle once it publishes that hour.",
        }
    return {"source": "dukascopy", "dukascopy_confirmed": True,
            "note": "this trigger candle is Dukascopy-sourced - the same feed the pattern's mined "
                    "win rate was measured against"}


def _freshness(candles: pd.DataFrame, timeframe: str, now: pd.Timestamp) -> dict:
    trigger_ts = pd.Timestamp(candles["timestamp"].iloc[-1])
    candle_minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
    age_minutes = (now - trigger_ts).total_seconds() / 60
    age_candles = age_minutes / candle_minutes if candle_minutes else 0

    if age_candles <= FRESH_MAX_CANDLES:
        status = "FRESH"
    elif age_candles <= AGING_MAX_CANDLES:
        status = "AGING"
    else:
        status = "EXPIRED"

    return {
        "trigger_candle_closed_at": trigger_ts.isoformat(),
        "signal_evaluated_at": now.isoformat(),
        "age_minutes": round(age_minutes, 1),
        "age_candles": round(age_candles, 2),
        "status": status,
        "note": "signal detection only looks at CLOSED candles - EXPIRED means "
                "too much time has passed since the trigger candle closed for "
                "the original entry/stop/target to still reflect the setup",
        "data_source": _trigger_candle_source(candles),
    }


def compute_signal(
    candles_by_timeframe: dict[str, pd.DataFrame],
    library_by_timeframe: dict[str, dict],
    events: pd.DataFrame | None = None,
    upcoming: pd.DataFrame | None = None,
    now: pd.Timestamp | None = None,
    suspended: set[tuple[str, str, str]] | None = None,
    circuit_breaker: dict | None = None,
    journal: pd.DataFrame | None = None,
) -> Signal:
    """`suspended`: a set of (pattern, timeframe, direction) triples -
    self-healing, see signal_journal.suspended_patterns() and
    load_suspended() below. A pattern/direction currently DECAYING live
    is skipped here even if the mined library still says qualifies:true
    (the library may simply not have been remined since it started
    failing live) - and if a DIFFERENT direction of the same pattern
    still independently qualifies and isn't suspended, that one is used
    instead rather than dropping the pattern's vote entirely.

    `circuit_breaker`: the dict circuit_breaker.check_circuit_breaker()
    returns, if the caller checked one. When `circuit_breaker["tripped"]`
    is true, this is a HARD override - forced HOLD, no contributions
    computed at all - same precedence as a suspended pattern, but
    system-wide rather than per-pattern. Portfolio-level risk state, not
    a per-pattern statistical judgment; see circuit_breaker.py.

    `journal`: this system's own live signal_journal (signal_journal.
    load_journal()'s output), if the caller has one. Used ONLY for
    signal_journal.context_penalty() - loss/win ATTRIBUTION, a confidence
    discount applied once THIS specific pattern/timeframe/direction's own
    live trades, in the SAME context this signal is about to fire in
    (confluence count, active trading session), have proven - Wilson-
    bound, hard-gated on sample size - to underperform that pattern's
    other live trades. None/omitted means "not checked" (confidence is
    never discounted, same fail-open posture as every other self-healing
    lookup in this system if the caller doesn't have a journal handy)."""
    now = now or pd.Timestamp.now(tz="UTC").tz_localize(None)
    suspended = suspended or set()

    if circuit_breaker and circuit_breaker.get("tripped"):
        return Signal(direction="HOLD", confidence=0.0, circuit_breaker=circuit_breaker)

    contributions = []
    suspended_skipped: list[str] = []

    for tf, candles in candles_by_timeframe.items():
        library = library_by_timeframe.get(tf)
        if library is None or len(candles) < 30:
            continue
        # Current market regime for THIS timeframe - same combined_regime()
        # mining used, computed on the live tail so it stays causal (no
        # look-ahead risk, see regime.py). One value per timeframe, reused
        # for every pattern below - not per-pattern, since it only depends
        # on the candles themselves.
        regime_series = combined_regime(candles)
        current_regime = regime_series.iloc[-1] if len(regime_series) else None
        if pd.isna(current_regime):
            current_regime = None

        active = _active_patterns_on_last_candle(candles, events, library)
        for pattern in active:
            ranked = _qualifying_directions_ranked(library, pattern)
            if not ranked:
                continue  # hard gate: doesn't clear 1:4 R:R @ 60%+ win rate (in-sample AND out-of-sample)
            result = None
            for candidate_direction, candidate_stats, candidate_news_cond in ranked:
                if (pattern, tf, candidate_direction) in suspended:
                    suspended_skipped.append(f"{pattern}/{tf}/{candidate_direction}")
                    continue
                result = (candidate_direction, candidate_stats, candidate_news_cond)
                break
            if result is None:
                continue  # every qualifying direction for this pattern is currently suspended (DECAYING live)
            direction, stats, news_cond = result
            resolved = stats["resolved"]
            win_rate = stats["win_rate"]
            wilson_lower = stats.get("win_rate_wilson_lower", wilson_lower_bound(round(win_rate * resolved), resolved))
            # Weighted by the CONSERVATIVE (Wilson lower-bound) win rate,
            # not the raw observed one - a 65% win rate on 30 trades and
            # 65% on 300 trades should not carry equal weight.
            weight = TIMEFRAME_WEIGHTS.get(tf, 1.0) * wilson_lower * min(resolved / 200, 1.0)

            # Regime-conditioned weight adjustment - see REGIME_MISMATCH_PENALTY
            # above for why this is a discount, not a hard gate. Only
            # applied when there's ENOUGH regime-specific mined history to
            # trust the verdict (regime_stats["qualifies"] already folds
            # in its own min-sample check); "not enough regime-specific
            # data yet" leaves weight untouched, deliberately.
            regime_stats = _regime_stats_for_pattern(library, pattern, direction, current_regime)
            regime_penalized = False
            if (regime_stats is not None and regime_stats.get("resolved", 0) >= MIN_RESOLVED_SAMPLES
                    and not regime_stats.get("qualifies", False)):
                weight *= REGIME_MISMATCH_PENALTY
                regime_penalized = True

            # Cross-timeframe weight adjustment - see CROSS_TIMEFRAME_MISMATCH_PENALTY
            # above. Only ever fires for a discovered__ pattern whose check
            # actually ran and came back qualifies: false - see
            # _cross_timeframe_mismatch's own docstring.
            cross_timeframe_penalized = _cross_timeframe_mismatch(library, pattern)
            if cross_timeframe_penalized:
                weight *= CROSS_TIMEFRAME_MISMATCH_PENALTY

            contributions.append(
                Contribution(
                    timeframe=tf,
                    pattern=pattern,
                    direction=direction,
                    win_rate=win_rate,
                    win_rate_wilson_lower=round(wilson_lower, 4),
                    samples=resolved,
                    weight=round(weight, 4),
                    median_candles_to_resolve=stats.get("median_candles_to_resolve"),
                    news_conditioning=news_cond,
                    regime=current_regime,
                    regime_win_rate=regime_stats.get("win_rate") if regime_stats else None,
                    regime_samples=regime_stats.get("resolved") if regime_stats else None,
                    regime_qualifies=regime_stats.get("qualifies") if regime_stats else None,
                    regime_penalized=regime_penalized,
                    cross_timeframe_penalized=cross_timeframe_penalized,
                )
            )

    if not contributions:
        return Signal(direction="HOLD", confidence=0.0, suspended_skipped=suspended_skipped,
                      circuit_breaker=circuit_breaker)

    # Multiple patterns firing on the SAME timeframe in the SAME direction
    # are often correlated (a doji and a hammer both describe a small-body
    # candle), not independent confirmations - summing all of them would
    # overstate conviction. Only the strongest (highest-weight) pattern per
    # (timeframe, direction) counts toward direction/confidence; the rest
    # are still reported in `contributions` for transparency
    # (used_for_signal=False) but don't add weight.
    #
    # Deliberately keyed by (timeframe, direction), NOT timeframe alone: a
    # timeframe can show genuinely CONFLICTING evidence (one pattern
    # bullish, a different pattern bearish) - that's not correlation, it's
    # real mixed signal, and should partially offset in the weighted vote
    # (lowering confidence), not get silently discarded because a
    # same-timeframe bullish pattern happened to have a higher weight.
    best_per_tf_dir: dict[tuple[str, str], Contribution] = {}
    for c in contributions:
        key = (c.timeframe, c.direction)
        current = best_per_tf_dir.get(key)
        if current is None or c.weight > current.weight:
            best_per_tf_dir[key] = c
    for c in best_per_tf_dir.values():
        c.used_for_signal = True

    used = list(best_per_tf_dir.values())
    bull_weight = sum(c.weight for c in used if c.direction == "bullish")
    bear_weight = sum(c.weight for c in used if c.direction == "bearish")
    total = bull_weight + bear_weight
    if total == 0:
        return Signal(direction="HOLD", confidence=0.0, contributions=contributions,
                      suspended_skipped=suspended_skipped, circuit_breaker=circuit_breaker)

    if bull_weight > bear_weight:
        direction = "BUY"
        agreement = (bull_weight - bear_weight) / total
        top = max((c for c in used if c.direction == "bullish"), key=lambda c: c.weight)
    elif bear_weight > bull_weight:
        direction = "SELL"
        agreement = (bear_weight - bull_weight) / total
        top = max((c for c in used if c.direction == "bearish"), key=lambda c: c.weight)
    else:
        direction, agreement, top = "HOLD", 0.0, None

    # Confidence is now a statistical estimate (the strongest contributing
    # pattern's Wilson lower-bound win rate), discounted by how much
    # OTHER timeframes disagree - NOT "how many things agree," which
    # could hit 100% off a single pattern barely over the 60% gate.
    confidence = 100 * top.win_rate_wilson_lower * agreement if top else 0.0

    # Loss/win ATTRIBUTION (signal_journal.context_penalty) - see
    # Signal.to_dict()'s docstring. `confluence_count` here uses the SAME
    # rule signal_journal._confluence_count() applies when this signal
    # eventually gets logged (every contribution sharing the FINAL
    # direction, not just the ones counted toward the weighted vote) -
    # computed identically in both places so a live lookup here and the
    # historical buckets it's compared against can never silently
    # disagree on what "confluence_count=2" means.
    context_penalty_info = None
    if top is not None and journal is not None:
        from signal_journal import context_penalty, session_label_at

        same_direction = "bullish" if direction == "BUY" else "bearish"
        confluence_count = sum(1 for c in contributions if c.direction == same_direction)
        top_candles = candles_by_timeframe[top.timeframe]
        entry_ts = pd.Timestamp(top_candles["timestamp"].iloc[-1])
        multiplier, reasons = context_penalty(
            # NOTE: `direction` here (not top.direction) - the journal
            # stores BUY/SELL (signal_dict's own vocabulary), while
            # Contribution.direction uses bullish/bearish; context_scorecard()
            # groups by the journal's raw column, so this has to match that.
            journal, top.pattern, top.timeframe, direction,
            confluence_count, session_label_at(entry_ts),
        )
        context_penalty_info = {"multiplier": multiplier, "reasons": reasons}
        confidence *= multiplier

    contributions.sort(key=lambda c: (c.used_for_signal, c.weight), reverse=True)

    trade_plan, freshness, news_risk = None, None, None
    if direction != "HOLD":
        top_candles = candles_by_timeframe[top.timeframe]
        signal_direction = "bullish" if direction == "BUY" else "bearish"
        trade_plan = _trade_plan(top_candles, signal_direction)
        freshness = _freshness(top_candles, top.timeframe, now)

        news_risk = label_signal_news_risk(
            now, top.median_candles_to_resolve, TIMEFRAME_MINUTES.get(top.timeframe, 60), upcoming
        )
        if news_risk.get("news_in_window") and top.news_conditioning:
            news_risk["historical_performance_with_news"] = top.news_conditioning.get("news_in_window")
            news_risk["historical_performance_without_news"] = top.news_conditioning.get("no_news")

    return Signal(direction=direction, confidence=confidence, contributions=contributions,
                  trade_plan=trade_plan, freshness=freshness, news_risk=news_risk,
                  primary_pattern=top.pattern if top else None,
                  primary_timeframe=top.timeframe if top else None,
                  suspended_skipped=suspended_skipped, circuit_breaker=circuit_breaker,
                  context_penalty=context_penalty_info)


def load_inputs(symbol: str, data_dir: Path, lib_dir: Path, tail: int = 300,
                 discovered_dir: Path | str = "discovered_patterns"):
    """`discovered_dir`: the Pattern Discovery Engine's own output
    (discover_patterns.py) - kept in a SEPARATE directory from `lib_dir`
    (pattern_library/, written by build_pattern_library.py) rather than
    merged into the same files on disk, so the two mining scripts never
    race writing to the same file (same separation ml_registry/lib_view/
    already uses to keep the ML challenger's mined view distinct from
    the rule-based system's). Merged together only here, in memory, into
    one `library_by_tf[tf]` dict - discovered pattern names are always
    prefixed `discovered__` (see discover_patterns._pattern_name), so
    there's no risk of colliding with an atomic/combo/support-resistance
    name from pattern_library/. Missing entirely (no discovery run yet)
    is a normal, common state, not an error - discovered patterns simply
    don't contribute anything until discover_patterns.py has been run at
    least once."""
    discovered_dir = Path(discovered_dir)
    candles_by_tf, library_by_tf = {}, {}
    for candle_path in sorted((data_dir / "candles").glob(f"{symbol}_*.parquet")):
        tf = candle_path.stem.replace(f"{symbol}_", "")
        candles_by_tf[tf] = pd.read_parquet(candle_path).sort_values("timestamp").tail(tail).reset_index(drop=True)
        lib_path = lib_dir / f"{symbol}_{tf}.json"
        library = json.loads(lib_path.read_text()) if lib_path.exists() else {}
        discovered_path = discovered_dir / f"{symbol}_{tf}.json"
        if discovered_path.exists():
            library.update(json.loads(discovered_path.read_text()))
        if library:
            library_by_tf[tf] = library

    events_path = data_dir / "events" / "fundamentals.parquet"
    events = pd.read_parquet(events_path) if events_path.exists() else None
    upcoming = load_upcoming(data_dir)

    return candles_by_tf, library_by_tf, events, upcoming


def load_suspended(data_dir: Path, lib_dir: Path, symbol: str,
                    discovered_dir: Path | str | None = "discovered_patterns") -> set[tuple[str, str, str]]:
    """(pattern, timeframe, direction) triples currently DECAYING per the
    live journal (signal_journal.suspended_patterns) - translated from
    the journal's BUY/SELL vocabulary into this module's bullish/bearish
    one. `discovered_dir`: same merge load_inputs() does - without it, a
    live-firing discovered__ pattern's mined win rate could never be found
    (it isn't in lib_dir), so it could never be flagged DECAYING/suspended
    no matter how badly it performed live (see signal_journal._load_lib_cached's
    own docstring). Failure to load (no journal yet, or anything else going
    wrong) means an empty suspension set, not a crash - self-healing is a
    refinement on top of the existing hard gates, never something that can
    take signal generation down if it errors."""
    try:
        import signal_journal
        journal = signal_journal.load_journal(data_dir)
        raw = signal_journal.suspended_patterns(journal, lib_dir, symbol, discovered_dir)
    except Exception:
        return set()
    dir_map = {"BUY": "bullish", "SELL": "bearish"}
    return {(pattern, tf, dir_map.get(direction, direction)) for (pattern, tf, direction) in raw}


def check_circuit_breaker_safe(data_dir: Path) -> dict | None:
    """circuit_breaker.check_circuit_breaker() against this system's own
    journal, fail-open the same way load_suspended() does - a portfolio-
    level safety check erroring out must never itself be what takes
    signal generation down. Returns None (not evaluated, not a healthy
    False) on any failure, so a caller can distinguish "checked, all
    clear" from "couldn't check" if it wants to."""
    try:
        import signal_journal
        from circuit_breaker import check_circuit_breaker
        journal = signal_journal.load_journal(data_dir)
        return check_circuit_breaker(journal)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--lib-dir", default="pattern_library")
    parser.add_argument("--discovered-dir", default="discovered_patterns")
    parser.add_argument("--no-journal", action="store_true", help="don't log this signal to the journal")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    lib_dir = Path(args.lib_dir)
    candles_by_tf, library_by_tf, events, upcoming = load_inputs(
        args.symbol, data_dir, lib_dir, discovered_dir=args.discovered_dir
    )
    if not candles_by_tf:
        raise SystemExit("no candle data found - run build_history.py first")
    if not library_by_tf:
        raise SystemExit("no pattern library found - run build_pattern_library.py first")

    suspended = load_suspended(data_dir, lib_dir, args.symbol, args.discovered_dir)
    if suspended:
        print(f"self-heal: {len(suspended)} pattern/timeframe/direction combo(s) suspended "
              f"(DECAYING live): {sorted(suspended)}")

    breaker = check_circuit_breaker_safe(data_dir)
    if breaker and breaker.get("tripped"):
        print(f"CIRCUIT BREAKER TRIPPED: {breaker['reasons']} - forcing HOLD system-wide until cleared")

    import signal_journal
    journal = signal_journal.load_journal(data_dir)

    signal = compute_signal(candles_by_tf, library_by_tf, events, upcoming,
                            suspended=suspended, circuit_breaker=breaker, journal=journal)
    print(json.dumps(signal.to_dict(), indent=2))

    if not args.no_journal:
        signal_journal.update_journal(candles_by_tf, data_dir)
        signal_journal.log_signal(signal.to_dict(), candles_by_tf, data_dir)


if __name__ == "__main__":
    main()
