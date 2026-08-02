"""
Live signal journal: every time signal_engine.py emits an actionable
(non-HOLD) signal, log it with EXACTLY the trade plan it produced (entry/
stop/target, fixed at that moment - never moved after the fact). On
every live_update.py run, walk open entries forward against new candles
using the SAME resolve_trade() logic risk_reward.py uses for historical
mining - so live tracking and the backtest are always measuring outcomes
the same way, and the live win rate is actually comparable to the mined
one.

Entries move from "open" to "win" / "loss" (stop or target touched) or
"expired" (MAX_LOOKAHEAD candles passed with neither touched - the same
policy historical mining uses for "unresolved", so a signal that outlives
its expected window is retired rather than tracked forever).

Deduped by (timeframe, pattern, entry candle) so re-running live_update
while the same candle is still the latest one doesn't create duplicate
rows for what is, mechanically, the same setup.

LOSS ATTRIBUTION (context_scorecard/context_penalty below): beyond just
tracking WHETHER a signal won or lost, every logged entry also captures
a handful of CONTEXT features true at the moment it fired - how many
OTHER patterns independently agreed (confluence_count), the market
regime (regime_at_entry), which trading session(s) were active
(session_at_entry) - plus, once a trade resolves, how the volatility
that actually materialized during the trade compared to what the stop
was sized for (volatility_shock_ratio). None of this changes how a
signal is scored win/loss/expired - it's purely additional, structured
evidence for diagnosing WHY, accumulated the same disciplined way every
other statistic in this system is: many resolved trades, Wilson-bound
confidence, a minimum-sample hard gate before anything is trusted, never
a reaction to one single loss. See context_scorecard() and
context_penalty() further down.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from atomic_io import atomic_write_parquet
from risk_reward import (
    MAX_LOOKAHEAD, RR_RATIO, SPREAD_USD, STOP_ATR_MULTIPLE, atr, resolve_trade, wilson_interval,
)

JOURNAL_COLUMNS = [
    "signal_id", "logged_at_utc", "timeframe", "pattern", "direction",
    "entry_candle_timestamp", "entry", "stop_loss", "take_profit", "risk",
    "confidence", "status", "resolved_at_utc", "outcome", "actual_r", "actual_r_after_costs",
    "confluence_count", "regime_at_entry", "session_at_entry", "volatility_shock_ratio",
]


def _journal_path(data_dir: Path) -> Path:
    return data_dir / "signal_journal.parquet"


def load_journal(data_dir: Path) -> pd.DataFrame:
    path = _journal_path(data_dir)
    if not path.exists():
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    journal = pd.read_parquet(path)
    # Schema evolution: a journal file written before a given column
    # existed (e.g. confluence_count/regime_at_entry/session_at_entry/
    # volatility_shock_ratio, added for loss/win attribution) is missing
    # it entirely on disk - pd.read_parquet() only ever returns whatever
    # columns are ACTUALLY in the file, it doesn't know about
    # JOURNAL_COLUMNS. Backfilled with None here, ONCE, at the single
    # point every caller already goes through to read this file - not a
    # fabricated guess (None honestly means "not captured for this row,
    # it predates the feature"), just guaranteeing every caller can rely
    # on the full schema always being present instead of each one
    # needing its own defensive column-existence check (or, worse,
    # silently crashing - verified this was a real, reproducible
    # KeyError in context_scorecard() before this fix, not hypothetical).
    for col in JOURNAL_COLUMNS:
        if col not in journal.columns:
            journal[col] = None
    return journal


def session_label_at(ts) -> str:
    """The exact `session_at_entry` encoding this journal uses (comma-
    joined active session names, or "off_hours") - a SINGLE source of
    truth so log_signal() (writing it after the fact) and
    context_penalty()'s caller (reading it live, for a signal that
    hasn't been logged yet - see signal_engine.py) can never encode the
    same real-world moment two different ways and silently fail to
    match buckets that actually mean the same thing."""
    from session_patterns import active_sessions_at
    active = active_sessions_at(ts)
    return ",".join(active) if active else "off_hours"


def _confluence_count(signal_dict: dict) -> int:
    """How many independent contributions agreed with the direction this
    signal actually took - the FULL agreeing set (not just the ones
    counted toward the final weighted vote, `used_for_signal`), since
    even a correlated-and-not-separately-counted confirmation is still
    corroborating evidence a trader would see on the dashboard. 0 is a
    real, meaningful value (a signal riding on exactly one pattern, no
    other confirmation anywhere) - see context_scorecard() for whether
    that actually correlates with worse outcomes for a given pattern, or
    turns out not to matter."""
    direction = signal_dict.get("direction")
    same_direction = {"BUY": "bullish", "SELL": "bearish"}.get(direction)
    if same_direction is None:
        return 0
    contributions = signal_dict.get("contributions") or []
    return sum(1 for c in contributions if c.get("direction") == same_direction)


def log_signal(signal_dict: dict, candles_by_tf: dict, data_dir: Path) -> pd.DataFrame:
    """Append a new journal entry for `signal_dict` (a Signal.to_dict()
    from signal_engine.py) if it's actionable (BUY/SELL with a trade_plan)
    and hasn't already been logged for this exact trigger."""
    journal = load_journal(data_dir)

    direction = signal_dict.get("direction")
    trade_plan = signal_dict.get("trade_plan")
    # MUST use these explicit fields, not contributions[0]: with multiple
    # timeframes, the single highest-weight individual contribution can be
    # in the LOSING direction (several moderate same-direction votes can
    # outweigh one strong opposing one) - contributions[0] would then name
    # a pattern that had nothing to do with producing trade_plan. See
    # signal_engine.Signal.to_dict() for why these fields exist.
    pattern = signal_dict.get("primary_pattern")
    tf = signal_dict.get("primary_timeframe")
    if direction not in ("BUY", "SELL") or not trade_plan or not pattern or not tf:
        return journal

    candles = candles_by_tf.get(tf)
    if candles is None or candles.empty:
        return journal
    entry_candle_ts = pd.Timestamp(candles["timestamp"].iloc[-1])

    if not journal.empty:
        dup = journal[
            (journal["timeframe"] == tf) & (journal["pattern"] == pattern) &
            (pd.to_datetime(journal["entry_candle_timestamp"]) == entry_candle_ts)
        ]
        if not dup.empty:
            return journal

    # Context captured at THIS exact moment, never revisited later - what
    # a trader would genuinely have known when this signal fired (no
    # look-ahead risk: regime/session are both computed from the trigger
    # candle itself and earlier, same as everything else in signal_engine.py).
    # See module docstring / context_scorecard() below for what this is for.
    from regime import combined_regime

    regime_series = combined_regime(candles)
    regime_at_entry = regime_series.iloc[-1] if len(regime_series) else None
    regime_at_entry = None if pd.isna(regime_at_entry) else str(regime_at_entry)

    session_at_entry = session_label_at(entry_candle_ts)

    row = {
        "signal_id": f"{tf}-{pattern}-{entry_candle_ts.value}",
        "logged_at_utc": pd.Timestamp.now(tz="UTC").tz_localize(None),
        "timeframe": tf, "pattern": pattern, "direction": direction,
        "entry_candle_timestamp": entry_candle_ts,
        "entry": trade_plan["entry"], "stop_loss": trade_plan["stop_loss"],
        "take_profit": trade_plan["take_profit"], "risk": trade_plan["risk"],
        "confidence": signal_dict.get("confidence"),
        "status": "open", "resolved_at_utc": pd.NaT, "outcome": None,
        "actual_r": None, "actual_r_after_costs": None,
        "confluence_count": _confluence_count(signal_dict),
        "regime_at_entry": regime_at_entry,
        "session_at_entry": session_at_entry,
        # Only knowable once the trade has actually played out - see
        # _volatility_shock_ratio() / update_journal() below.
        "volatility_shock_ratio": None,
    }
    journal = pd.concat([journal, pd.DataFrame([row])], ignore_index=True)
    atomic_write_parquet(journal, _journal_path(data_dir))
    return journal


def _volatility_shock_ratio(candles: pd.DataFrame, entry_idx: int, resolved_idx: int, risk) -> float | None:
    """Realized ATR during [entry_idx, resolved_idx] vs the ATR the stop
    was originally sized on - back-derived from the journal's own
    `risk` column (risk == STOP_ATR_MULTIPLE * atr_at_signal_time,
    exactly how signal_engine._trade_plan() computed it - see
    risk_reward.py - so there's no separate value to store, just invert
    the one formula). >1 means the market was genuinely MORE volatile
    during the trade than the stop assumed - a real, distinct candidate
    explanation for a loss ("the pattern wasn't wrong, a volatility
    spike right after entry overran a stop that was reasonable for the
    conditions AT signal time") separate from "the pattern itself just
    doesn't have an edge here". Purely diagnostic: computed strictly
    AFTER the fact using candles the trade has already walked through by
    the time this runs, never fed back into THIS trade's own entry/
    stop/target (already fixed and logged) - only ever into the
    AGGREGATE, hard-gated analysis future signals get judged against
    (context_scorecard() below) - the same no-look-ahead-into-a-live-
    decision rule every other part of this system already follows."""
    if not risk or float(risk) <= 0 or resolved_idx < entry_idx:
        return None
    atr_series = atr(candles)
    window = atr_series.iloc[entry_idx:resolved_idx + 1]
    if window.empty or window.isna().all():
        return None
    realized_atr = float(window.mean())
    atr_at_entry = float(risk) / STOP_ATR_MULTIPLE
    if atr_at_entry <= 0:
        return None
    return round(realized_atr / atr_at_entry, 4)


def update_journal(candles_by_tf: dict, data_dir: Path) -> pd.DataFrame:
    """Walk every OPEN journal entry forward against the latest candles on
    its own timeframe, using the identical resolve_trade() the historical
    mining uses, and mark win/loss/expired as appropriate. Requires the
    entry candle to still be inside the loaded candle window (the default
    tail(300) in signal_engine.load_inputs() is comfortably larger than
    MAX_LOOKAHEAD, so this holds under normal operation)."""
    journal = load_journal(data_dir)
    if journal.empty:
        return journal

    open_mask = journal["status"] == "open"
    if not open_mask.any():
        return journal

    changed = False
    for idx in journal[open_mask].index:
        row = journal.loc[idx]
        candles = candles_by_tf.get(row["timeframe"])
        if candles is None or candles.empty:
            continue

        ts = pd.to_datetime(candles["timestamp"]).values.astype("datetime64[ns]")
        target_ts = pd.Timestamp(row["entry_candle_timestamp"]).to_datetime64().astype("datetime64[ns]")
        matches = np.flatnonzero(ts == target_ts)
        if len(matches) == 0:
            continue  # trigger candle has aged out of the loaded tail window

        # `entry_candle_timestamp` marks the TRIGGER candle (the one the
        # pattern fired on) - same convention risk_reward.simulate_trades
        # uses for mining: the actual entry is priced at the NEXT candle's
        # open, so resolution checking starts there too, not on the
        # trigger candle itself. If that next candle hasn't happened yet
        # (this signal was only just logged), there's nothing to check -
        # stays "open" until it does.
        trigger_idx = int(matches[0])
        entry_idx = trigger_idx + 1
        if entry_idx >= len(candles):
            continue

        direction = 1 if row["direction"] == "BUY" else -1
        high = candles["high"].to_numpy()
        low = candles["low"].to_numpy()

        outcome, resolved_idx = resolve_trade(
            high, low, entry_idx, direction, float(row["stop_loss"]), float(row["take_profit"]),
        )
        candles_since_entry = len(candles) - 1 - entry_idx

        if outcome in ("win", "loss"):
            spread_cost_r = SPREAD_USD / float(row["risk"]) if row["risk"] else 0.0
            journal.loc[idx, "status"] = outcome
            journal.loc[idx, "outcome"] = outcome
            journal.loc[idx, "resolved_at_utc"] = candles["timestamp"].iloc[resolved_idx]
            journal.loc[idx, "actual_r"] = RR_RATIO if outcome == "win" else -1.0
            journal.loc[idx, "actual_r_after_costs"] = (
                RR_RATIO - spread_cost_r if outcome == "win" else -1.0 - spread_cost_r
            )
            journal.loc[idx, "volatility_shock_ratio"] = _volatility_shock_ratio(
                candles, entry_idx, resolved_idx, row["risk"],
            )
            changed = True
        elif candles_since_entry >= MAX_LOOKAHEAD:
            journal.loc[idx, "status"] = "expired"
            journal.loc[idx, "outcome"] = "expired"
            last_idx = min(entry_idx + MAX_LOOKAHEAD - 1, len(candles) - 1)
            journal.loc[idx, "volatility_shock_ratio"] = _volatility_shock_ratio(
                candles, entry_idx, last_idx, row["risk"],
            )
            changed = True

    if changed:
        atomic_write_parquet(journal, _journal_path(data_dir))
    return journal


def summary(journal: pd.DataFrame) -> dict:
    if journal.empty:
        return {"total": 0, "open": 0, "win": 0, "loss": 0, "expired": 0, "live_win_rate": None}
    counts = journal["status"].value_counts().to_dict()
    resolved = counts.get("win", 0) + counts.get("loss", 0)
    win_rate = counts.get("win", 0) / resolved if resolved else None
    return {
        "total": int(len(journal)),
        "open": int(counts.get("open", 0)),
        "win": int(counts.get("win", 0)),
        "loss": int(counts.get("loss", 0)),
        "expired": int(counts.get("expired", 0)),
        "live_win_rate": round(win_rate, 4) if win_rate is not None else None,
    }


MIN_LIVE_SAMPLES_FOR_DRIFT = 10


def _load_lib_cached(lib_dir: Path, symbol: str, cache: dict, tf: str,
                      discovered_dir: Path | None = None) -> dict:
    """`discovered_dir`: merges in discovered_patterns/<symbol>_<tf>.json
    (see discover_patterns.py / signal_engine.load_inputs()'s identical
    merge) - without this, a live-firing `discovered__` pattern's mined
    win rate would never be found here (it isn't in `lib_dir` at all),
    silently disabling drift detection/self-healing for EVERY discovered
    pattern (credibility() treats mined_win_rate=None as "can't be
    decaying" - see its own docstring). None (the default) preserves the
    old behavior for callers that don't have a discovered_dir to offer
    (e.g. the ML challenger's own lib_view, which never has discovered__
    entries to begin with)."""
    import json
    if tf not in cache:
        lib_path = lib_dir / f"{symbol}_{tf}.json"
        lib = json.loads(lib_path.read_text()) if lib_path.exists() else {}
        if discovered_dir is not None:
            discovered_path = Path(discovered_dir) / f"{symbol}_{tf}.json"
            if discovered_path.exists():
                lib.update(json.loads(discovered_path.read_text()))
        cache[tf] = lib
    return cache[tf]


def _mined_win_rate_for(entry: dict | None, directions_seen: set) -> float | None:
    """Pull the mined win rate that corresponds to whichever direction(s)
    this pattern actually fired live, from a raw pattern_library entry
    (see build_pattern_library.py) - shared by detect_drift() and
    pattern_scorecard() so the two can't quietly disagree on what "the
    mined win rate" means for an ambiguous-direction pattern."""
    if entry is None:
        return None
    if "stats" in entry:
        return entry["stats"].get("win_rate")
    if "BUY" in directions_seen and entry.get("as_long"):
        return entry["as_long"].get("win_rate")
    if "SELL" in directions_seen and entry.get("as_short"):
        return entry["as_short"].get("win_rate")
    return None


def detect_drift(journal: pd.DataFrame, lib_dir: Path, symbol: str = "XAUUSD",
                  discovered_dir: Path | None = None) -> list[dict]:
    """For every (pattern, timeframe, direction) with enough RESOLVED live
    journal entries, compare the live win rate against what was mined for
    that exact pattern/direction. Flags `decaying: true` only when the
    live win rate's Wilson UPPER bound is still below the mined win rate
    - i.e. we can be reasonably confident this isn't just sampling noise,
    the pattern is actually performing worse live than its backtest. This
    is the check that would have caught a pattern quietly stopping
    working after the library was last rebuilt. A thin wrapper over
    pattern_scorecard() (single source of truth for "is this
    pattern/timeframe/direction decaying live", also used by
    suspended_patterns() to actually act on it, not just report it)."""
    scorecard = pattern_scorecard(journal, lib_dir, symbol, discovered_dir)
    out = []
    for row in scorecard:
        if row["live_samples"] < MIN_LIVE_SAMPLES_FOR_DRIFT or row["mined_win_rate"] is None:
            continue
        out.append({
            "pattern": row["pattern"], "timeframe": row["timeframe"], "direction": row["direction"],
            "live_win_rate": row["live_win_rate"],
            "live_samples": row["live_samples"],
            "live_win_rate_upper_bound": row["live_win_rate_wilson_upper"],
            "mined_win_rate": row["mined_win_rate"],
            "decaying": row["label"] == "DECAYING",
        })
    return out


def credibility(wins: int, n: int, mined_win_rate: float | None) -> dict:
    """Turns live resolved-trade counts into a single, transparent
    credibility read - not a black-box score. It's the exact same
    Wilson-lower-bound-on-win-rate logic risk_reward.py already uses to
    turn backtested stats into a trustworthy number, applied here to
    what actually happened LIVE, plus the same decay check detect_drift()
    runs. No live samples yet -> no score, on purpose: this system never
    fabricates a number from data it doesn't have, live or mined (same
    principle as the MIN_RESOLVED_SAMPLES hard gate in risk_reward.py)."""
    from risk_reward import MIN_WIN_RATE, wilson_interval

    if n < MIN_LIVE_SAMPLES_FOR_DRIFT:
        return {
            "score": None, "label": "UNPROVEN",
            "reason": f"only {n} live resolved trade(s) so far - need >= {MIN_LIVE_SAMPLES_FOR_DRIFT} "
                      f"before a live number means anything",
            "live_win_rate_wilson_lower": None, "live_win_rate_wilson_upper": None,
        }

    lo, hi = wilson_interval(wins, n)
    decaying = mined_win_rate is not None and hi < mined_win_rate

    if decaying:
        label = "DECAYING"
        reason = (f"live win rate's upper bound ({hi:.1%}) is still below the mined "
                  f"{mined_win_rate:.1%} - statistically worse than backtested, not just noise")
    elif lo >= MIN_WIN_RATE:
        label = "CONFIRMED"
        reason = f"live Wilson lower bound ({lo:.1%}) independently clears the {MIN_WIN_RATE:.0%} gate"
    else:
        label = "WATCH"
        reason = (f"live Wilson lower bound ({lo:.1%}) hasn't cleared the {MIN_WIN_RATE:.0%} gate yet "
                  f"on {n} live trades - could still be noise, needs more samples")

    return {
        "score": round(lo * 100, 1), "label": label, "reason": reason,
        "live_win_rate_wilson_lower": round(lo, 4), "live_win_rate_wilson_upper": round(hi, 4),
    }


def pattern_scorecard(journal: pd.DataFrame, lib_dir: Path, symbol: str = "XAUUSD",
                       discovered_dir: Path | None = None) -> list[dict]:
    """Live performance + credibility for EVERY (pattern, timeframe,
    direction) that has fired live and resolved at least once - not
    gated to only the ones with enough samples for a drift verdict (see
    detect_drift() for that stricter subset). This is the self-
    assessment table: per pattern AND direction, exactly how many
    wins/losses it's actually had live, the realized R (not just a
    win/loss count - a pattern can have a >50% win rate and still be a
    net loser at the wrong R multiples, or vice versa), and a
    credibility verdict you can read at a glance instead of having to
    interpret a raw win-rate number yourself.

    Grouped by DIRECTION too, not just (pattern, timeframe): an
    ambiguous pattern (doji, session_*, fundamental_*, ambiguous combos)
    can genuinely perform differently long vs short, and blending both
    together could hide one direction decaying behind the other still
    working - or, now that this feeds suspended_patterns() below, wrongly
    suspend a direction that's actually fine because its blended-in
    opposite direction is what's actually failing."""
    if journal.empty:
        return []
    resolved = journal[journal["status"].isin(["win", "loss"])]
    if resolved.empty:
        return []

    lib_cache: dict = {}
    out = []
    for (pattern, tf, direction), group in resolved.groupby(["pattern", "timeframe", "direction"]):
        n = len(group)
        wins = int((group["status"] == "win").sum())
        total_r = float(group["actual_r"].astype(float).sum())
        total_r_after_costs = float(group["actual_r_after_costs"].astype(float).sum())

        lib = _load_lib_cached(lib_dir, symbol, lib_cache, tf, discovered_dir)
        mined_win_rate = _mined_win_rate_for(lib.get(pattern), {direction})
        cred = credibility(wins, n, mined_win_rate)

        out.append({
            "pattern": pattern, "timeframe": tf, "direction": direction,
            "live_samples": n, "wins": wins, "losses": n - wins,
            "live_win_rate": round(wins / n, 4),
            "total_r": round(total_r, 2), "total_r_after_costs": round(total_r_after_costs, 2),
            "avg_r": round(total_r / n, 3), "avg_r_after_costs": round(total_r_after_costs / n, 3),
            "mined_win_rate": mined_win_rate,
            **cred,
        })
    out.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0)))
    return out


def suspended_patterns(journal: pd.DataFrame, lib_dir: Path, symbol: str = "XAUUSD",
                        discovered_dir: Path | None = None) -> dict:
    """(pattern, timeframe, direction) triples currently DECAYING live -
    the self-healing mechanism this exists for: signal_engine.py refuses
    to use these for NEW signals (see load_suspended/compute_signal's
    `suspended` parameter there), even though they may still show
    `qualifies: true` in the mined library. A pattern stops being
    suspended once either the library is remined and its live streak
    ages out/improves enough that credibility() no longer calls it
    DECAYING, or enough new live evidence accumulates to prove it's
    recovered - never just because a rebuild happened (see
    should_self_heal() below: a remine alone doesn't erase a live
    losing streak, on purpose)."""
    return {
        (row["pattern"], row["timeframe"], row["direction"]): row
        for row in pattern_scorecard(journal, lib_dir, symbol, discovered_dir)
        if row["label"] == "DECAYING"
    }


# Higher than MIN_LIVE_SAMPLES_FOR_DRIFT (10) on purpose: context_scorecard()
# below tests several dimensions x several buckets x every pattern - a much
# bigger multiple-comparisons search than a single pattern-level drift
# verdict, so it needs more evidence per bucket before anything here is
# trusted. Same reasoning risk_reward.py already applies to combo patterns
# needing a higher sample bar than atomic ones (COMBO_MIN_RESOLVED_SAMPLES).
CONTEXT_MIN_LIVE_SAMPLES = 15

# Soft discount applied by context_penalty() below, same magnitude and same
# reasoning as signal_engine.REGIME_MISMATCH_PENALTY: a live-evidence-backed
# but still comparatively young finding (CONTEXT_MIN_LIVE_SAMPLES is a much
# smaller bar than the pattern's own backtested mined stats) earns a weight
# discount, never a hard block - the existing DECAYING-pattern suspension
# mechanism (suspended_patterns, above) is what actually silences a pattern
# outright, and only once ITS OWN, stricter bar is cleared.
CONTEXT_PENALTY_MULTIPLIER = 0.5


def _confluence_bucket(n) -> str | None:
    """None (not a bucket - excluded from context_scorecard() entirely,
    same convention _volatility_bucket() uses below) for a row that
    predates this column existing (see load_journal()'s schema-evolution
    backfill). Deliberately NOT a fake "unknown" bucket: a LIVE signal's
    confluence_count is always a real int (0, 1, 2, ...), so an
    "unknown" bucket could never be matched against by context_penalty()
    anyway - it would just be dead weight sitting in the scorecard,
    confusing rather than informative."""
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return None
    n = int(n)
    return "3+" if n >= 3 else str(n)


def _volatility_bucket(ratio) -> str | None:
    """None (not a bucket - excluded from context_scorecard() entirely)
    for an unresolved/pre-existing row with no ratio yet - deliberately
    NOT lumped into a false "normal" bucket, which would silently dilute
    the real distribution. See _volatility_shock_ratio()'s docstring for
    why this dimension is diagnostic-only and never reaches
    context_penalty() (it isn't knowable before a trade resolves, so a
    about-to-fire live signal has no value to look up here)."""
    if ratio is None or (isinstance(ratio, float) and pd.isna(ratio)):
        return None
    if ratio < 1.3:
        return "normal (<1.3x)"
    if ratio < 2.0:
        return "elevated (1.3-2x)"
    return "shock (>=2x)"


_CONTEXT_DIMENSIONS = (
    ("confluence_count", "confluence_bucket"),
    ("session_at_entry", "session_at_entry"),
    ("volatility_shock_ratio", "volatility_bucket"),
)


def context_scorecard(journal: pd.DataFrame) -> list[dict]:
    """LOSS/WIN ATTRIBUTION: not "did this pattern work overall"
    (pattern_scorecard() above already answers that), but "WITHIN a
    pattern's own live trades, does the specific CONTEXT a trade fired
    in correlate with a meaningfully different outcome than its OWN
    peers" - confluence_count (how many other patterns independently
    agreed at signal time), session_at_entry (which trading session(s)
    were active), and volatility_shock_ratio (was the stop overrun by a
    volatility spike right after entry - diagnostic only, see
    _volatility_shock_ratio()'s docstring for why this one specifically
    can never feed back into a live decision the way the other two can).

    Grouped by (pattern, timeframe, direction) FIRST - a context factor
    can genuinely matter for one pattern and not another (a tight-stop
    scalping pattern is far more exposed to a volatility shock than a
    wide-stop swing one), blending everything together would wash that
    out. Each bucket's live win rate is compared against that SAME
    pattern's OTHER live trades (its own peers, not the mined/backtested
    number - the question here is "does this context matter", not "is
    this pattern still working", which detect_drift() already answers) -
    Wilson-bound, hard-gated by CONTEXT_MIN_LIVE_SAMPLES on BOTH the
    bucket and the rest-of-pattern comparison group, so a verdict never
    rests on a handful of trades on either side.

    regime_at_entry is captured (every journal row has it) but
    deliberately NOT broken out here - the mined pattern library already
    reports a regime-conditioned win rate (build_pattern_library.py's
    "by_regime") and signal_engine.py already applies a live weight
    discount from it (REGIME_MISMATCH_PENALTY); a second, separate
    regime analysis here would only duplicate that mechanism against a
    much smaller live-only sample, not add anything new."""
    if journal.empty:
        return []
    resolved = journal[journal["status"].isin(["win", "loss"])].copy()
    if resolved.empty:
        return []

    resolved["confluence_bucket"] = resolved["confluence_count"].apply(_confluence_bucket)
    resolved["volatility_bucket"] = resolved["volatility_shock_ratio"].apply(_volatility_bucket)

    out = []
    for (pattern, tf, direction), group in resolved.groupby(["pattern", "timeframe", "direction"]):
        for dimension, col in _CONTEXT_DIMENSIONS:
            valid = group[group[col].notna()]
            for bucket_value, bucket_group in valid.groupby(col):
                n = len(bucket_group)
                if n < CONTEXT_MIN_LIVE_SAMPLES:
                    continue
                wins = int((bucket_group["status"] == "win").sum())
                win_rate = wins / n
                lo, hi = wilson_interval(wins, n)

                rest = valid[valid[col] != bucket_value]
                if len(rest) < CONTEXT_MIN_LIVE_SAMPLES:
                    baseline_win_rate = None
                    decaying = False
                else:
                    baseline_win_rate = float((rest["status"] == "win").mean())
                    decaying = hi < baseline_win_rate

                out.append({
                    "pattern": pattern, "timeframe": tf, "direction": direction,
                    "dimension": dimension, "bucket": str(bucket_value),
                    "samples": n, "wins": wins, "win_rate": round(win_rate, 4),
                    "win_rate_wilson_lower": round(lo, 4), "win_rate_wilson_upper": round(hi, 4),
                    "baseline_win_rate": round(baseline_win_rate, 4) if baseline_win_rate is not None else None,
                    "decaying": bool(decaying),
                })
    out.sort(key=lambda r: (not r["decaying"], r["pattern"], r["dimension"]))
    return out


def context_penalty(journal: pd.DataFrame, pattern: str, timeframe: str, direction: str,
                     confluence_count: int, session_at_entry: str) -> tuple[float, list[str]]:
    """Live lookup for signal_engine.py/live_signal.py: does THIS
    about-to-fire signal's own context match a bucket already PROVEN
    (context_scorecard() above, hard-gated) to underperform for this
    EXACT (pattern, timeframe, direction)? Only confluence_count and
    session_at_entry are checked - the two context dimensions actually
    knowable BEFORE a trade resolves (volatility_shock_ratio, by
    construction, never is).

    Returns (multiplier, reasons): multiplier is 1.0 (no penalty, the
    overwhelmingly common case - most pattern/context combinations will
    never accumulate CONTEXT_MIN_LIVE_SAMPLES live trades) unless a
    hard-gated DECAYING bucket matches, in which case
    CONTEXT_PENALTY_MULTIPLIER - a confidence discount, same precedent
    and severity as signal_engine.REGIME_MISMATCH_PENALTY, never a hard
    suppression (suspended_patterns() is the mechanism that silences a
    pattern outright, and only once ITS stricter bar is cleared).

    Fails open (1.0, []) on any error - same policy as every other self-
    healing lookup in this system (load_suspended(),
    check_circuit_breaker_safe()): a diagnostic feature going wrong must
    never be what takes signal generation down."""
    try:
        scorecard = context_scorecard(journal)
    except Exception:
        return 1.0, []

    confluence_bucket = _confluence_bucket(confluence_count)
    multiplier = 1.0
    reasons: list[str] = []
    for row in scorecard:
        if not row["decaying"]:
            continue
        if (row["pattern"], row["timeframe"], row["direction"]) != (pattern, timeframe, direction):
            continue
        if row["dimension"] == "confluence_count" and row["bucket"] == confluence_bucket:
            multiplier = min(multiplier, CONTEXT_PENALTY_MULTIPLIER)
            reasons.append(
                f"confluence={confluence_bucket}: live win rate {row['win_rate']:.1%} on {row['samples']} "
                f"trades is statistically worse than this pattern's other live trades ({row['baseline_win_rate']:.1%})"
            )
        elif row["dimension"] == "session_at_entry" and row["bucket"] == session_at_entry:
            multiplier = min(multiplier, CONTEXT_PENALTY_MULTIPLIER)
            reasons.append(
                f"session={session_at_entry}: live win rate {row['win_rate']:.1%} on {row['samples']} "
                f"trades is statistically worse than this pattern's other live trades ({row['baseline_win_rate']:.1%})"
            )
    return multiplier, reasons


REBUILD_TRIGGER_DECAYING_COUNT = 3
# Sized for CONTINUOUS operation (scripts/run_continuous.py), not the
# original once-an-hour cron assumption this was first set under: with a
# live broker feed (mt_bridge/) importing new candles every ~20-30s, a
# week is a long time for genuinely new data to sit un-mined. Remining
# is a cheap, fully deterministic recomputation through the SAME hard
# gates every time (MIN_RESOLVED_SAMPLES, MIN_WIN_RATE, the out-of-
# sample check) - there's no "overfitting to noise" risk from doing it
# more often, only CPU time (seconds, per the perf notes in the main
# README), so tightening this backstop is pure upside, not a tradeoff.
# The reactive trigger above (genuine live decay) is intentionally left
# alone - lowering ITS threshold would make the system twitchier, not
# more accurate; this one is about freshness, not sensitivity.
REBUILD_TRIGGER_MAX_AGE_DAYS = 1.0


def should_self_heal(journal: pd.DataFrame, lib_dir: Path, symbol: str, heartbeats: dict,
                      discovered_dir: Path | None = None) -> tuple[bool, str]:
    """Decides whether live_update.py should trigger an automatic
    build_pattern_library.py rebuild THIS run, instead of a human having
    to notice decay on the dashboard and remember to run it, or waiting
    on a fixed cron regardless of whether anything's actually wrong.
    Two independent triggers, either is sufficient:

      1. Enough (pattern, timeframe, direction) combos are DECAYING live
         right now (>= REBUILD_TRIGGER_DECAYING_COUNT) - the market has
         moved enough that the CURRENT library is measurably wrong, not
         just one pattern having a rough patch.
      2. The library hasn't been successfully rebuilt in
         REBUILD_TRIGGER_MAX_AGE_DAYS - a time-based backstop for when
         decay hasn't (yet) accumulated enough live samples to trip
         trigger #1, but enough new data has piled up that a rebuild
         would fold it in regardless.

    Heartbeat timestamps are tz-AWARE ISO strings (heartbeat.py's own
    convention, unrelated to and not to be confused with the naive-UTC
    convention candle timestamps use elsewhere in this codebase - see
    build_history.py) so comparing against a tz-aware "now" here is
    correct, not an oversight."""
    decaying = suspended_patterns(journal, lib_dir, symbol, discovered_dir)
    if len(decaying) >= REBUILD_TRIGGER_DECAYING_COUNT:
        return True, f"{len(decaying)} pattern/timeframe/direction combos DECAYING live (>= {REBUILD_TRIGGER_DECAYING_COUNT})"

    hb = heartbeats.get("build_pattern_library")
    if hb and hb.get("status") == "ok":
        last_run = pd.Timestamp(hb["timestamp_utc"])
        now = pd.Timestamp.now(tz="UTC")
        age_days = (now - last_run).total_seconds() / 86400
        if age_days >= REBUILD_TRIGGER_MAX_AGE_DAYS:
            return True, f"pattern library is {age_days:.1f} days old (>= {REBUILD_TRIGGER_MAX_AGE_DAYS})"

    return False, ""


def overall_scorecard(journal: pd.DataFrame) -> dict:
    """The single "how is this system actually doing, for real" view -
    everything summary() has, plus what actually answers "do I trust
    this": a Wilson lower bound on the OVERALL live win rate (not just
    the raw observed one), realized R gross and cost-adjusted (a >50%
    win rate can still be a net loser at the wrong R multiples), and the
    current streak so a run of losses is visible instead of buried
    inside an aggregate."""
    base = summary(journal)
    empty_extra = {
        "win_rate_wilson_lower": None, "win_rate_wilson_upper": None,
        "total_r": None, "total_r_after_costs": None, "avg_r": None, "avg_r_after_costs": None,
        "current_streak": None, "best_trade_r": None, "worst_trade_r": None,
    }
    if journal.empty:
        return {**base, **empty_extra}

    resolved = journal[journal["status"].isin(["win", "loss"])]
    if resolved.empty:
        return {**base, **empty_extra}

    n = len(resolved)
    wins = int((resolved["status"] == "win").sum())
    lo, hi = wilson_interval(wins, n)
    actual_r = resolved["actual_r"].astype(float)
    actual_r_after_costs = resolved["actual_r_after_costs"].astype(float)

    ordered_outcomes = resolved.sort_values("resolved_at_utc")["status"].tolist()
    streak_type, streak_len = None, 0
    for outcome in reversed(ordered_outcomes):
        if streak_type is None:
            streak_type, streak_len = outcome, 1
        elif outcome == streak_type:
            streak_len += 1
        else:
            break

    return {
        **base,
        "win_rate_wilson_lower": round(lo, 4),
        "win_rate_wilson_upper": round(hi, 4),
        "total_r": round(float(actual_r.sum()), 2),
        "total_r_after_costs": round(float(actual_r_after_costs.sum()), 2),
        "avg_r": round(float(actual_r.mean()), 3),
        "avg_r_after_costs": round(float(actual_r_after_costs.mean()), 3),
        "current_streak": {"type": streak_type, "length": streak_len},
        "best_trade_r": round(float(actual_r.max()), 2),
        "worst_trade_r": round(float(actual_r.min()), 2),
    }


def equity_curve(journal: pd.DataFrame) -> list[dict]:
    """Cumulative realized R over time across every RESOLVED trade, in
    chronological order - the actual "proof in the pudding" chart: if
    this stays above zero and trends up, the system's LIVE performance
    is actually backing up its backtested numbers, without you having to
    risk real money to find that out yourself. Gross and cost-adjusted
    both reported since they can diverge meaningfully on tight-stop
    patterns (see SPREAD_USD in risk_reward.py)."""
    if journal.empty:
        return []
    resolved = journal[journal["status"].isin(["win", "loss"])].sort_values("resolved_at_utc")
    if resolved.empty:
        return []

    cum_r = resolved["actual_r"].astype(float).cumsum()
    cum_r_after_costs = resolved["actual_r_after_costs"].astype(float).cumsum()

    out = []
    for (_, row), c_r, c_r_costs in zip(resolved.iterrows(), cum_r, cum_r_after_costs):
        out.append({
            "resolved_at_utc": row["resolved_at_utc"].isoformat() if pd.notna(row["resolved_at_utc"]) else None,
            "pattern": row["pattern"], "timeframe": row["timeframe"], "outcome": row["status"],
            "trade_r": round(float(row["actual_r"]), 3),
            "cumulative_r": round(float(c_r), 3),
            "cumulative_r_after_costs": round(float(c_r_costs), 3),
        })
    return out
