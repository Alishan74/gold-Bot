"""
Fixed-structure trade simulation and the hard gates the whole system is
built around:

  1. Every trade the system will ever propose risks R and targets 4R -
     risk:reward is 1:4, always, by construction. There is no other ratio
     anywhere in this codebase.
  2. A pattern is only allowed to produce a live signal if, historically,
     that exact 1:4 trade actually hit its target before its stop at
     least MIN_WIN_RATE (60%) of the time, on enough samples to trust it,
     AND held up out-of-sample (see OOS_* below) - not just in a single
     fit over the whole dataset.

Both gates are enforced here at mining time (a pattern that misses either
is stored with `qualifies: false`) and re-checked in signal_engine.py
before anything is emitted - so a bug in one layer can't silently bypass
the other.

Trade simulation methodology, and what changed after a critical review:

- Entry is priced at the NEXT candle's open, not the signal candle's own
  close. You cannot know a candle closed as (say) a bullish engulfing
  until it closes - by the time that's true, price has already moved to
  the next candle's open. Pricing entry at the signal candle's own close
  is a look-ahead-adjacent assumption that inflates backtested results
  versus what's actually tradeable, especially across gold's session/
  weekend gaps. Stop distance (R) is still computed from ATR AT the
  signal candle (that's legitimately known at signal time), but entry,
  stop, and target are all anchored to the next candle's open.
- Transaction costs (SPREAD_USD) are modeled: win/loss classification
  itself is unaffected (spread doesn't move the market, only your fill),
  but expectancy_r is reported both before and after round-trip spread
  cost, because a pattern that "wins" 62% of the time on paper can still
  be unprofitable after real costs on tight-stop setups. SPREAD_USD is a
  placeholder - calibrate it to your actual broker's typical gold spread
  before trusting the after-cost numbers.
- Walk forward candle by candle until either level is touched (checked
  against that candle's high/low, not just its close) or MAX_LOOKAHEAD
  candles pass with neither hit ("unresolved" - excluded from win rate,
  not counted as a win).

`resolve_trade()` is the ONE walk-forward implementation used both here
(historical mining) and by signal_journal.py (checking real live signals
against new candles as they arrive). That sharing matters: if live
tracking used different logic than the backtest, the two win rates
wouldn't be measuring the same thing.

Out-of-sample validation: mining a pattern's stats over the ENTIRE
history and then testing it against that same entire history is fitting
and grading on the same data - a pattern that only worked in one regime
(say, the zero-rate 2009-2021 era) can still show a passing aggregate
win rate today. summarize_trades() now also reports out-of-sample stats
(the most recent OOS_FRACTION of occurrences, held out from nothing else
in this file but reported separately) and `qualifies` requires BOTH the
full-sample gate AND the out-of-sample win rate clearing the bar (on a
smaller, relaxed sample requirement - there's less data in a slice).
This is a single chronological holdout, not full walk-forward
reoptimization - a real edge, meaningfully improved, but flagged
honestly as a partial mitigation, not a complete one.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

RR_RATIO = 4.0            # reward = RR_RATIO * risk. Fixed system-wide.
ATR_WINDOW = 14
STOP_ATR_MULTIPLE = 1.5   # stop-loss distance = STOP_ATR_MULTIPLE * ATR
MAX_LOOKAHEAD = 60        # candles to wait for TP/SL before giving up
MIN_WIN_RATE = 0.60       # hard gate #2
MIN_RESOLVED_SAMPLES = 30 # below this, we don't trust the win rate either way

# Out-of-sample validation (see module docstring). OOS_FRACTION is the
# most-recent slice of each pattern's occurrences held out as a
# secondary check; OOS_MIN_RESOLVED_SAMPLES is relaxed vs the full-sample
# bar since a slice has less data by construction.
OOS_FRACTION = 0.30
OOS_MIN_RESOLVED_SAMPLES = 10

# combo_patterns.py mines cross-family PAIRS of already-vetted patterns
# (hundreds of them - see that module's docstring for why). Testing that
# many combinations and keeping only the ones that clear the bar is a
# much bigger multiple-comparisons search than the ~30-40 atomic patterns
# ever were, so combos are held to a higher sample bar before being
# trusted at all - on top of, not instead of, the same OOS gate below.
COMBO_MIN_RESOLVED_SAMPLES = 40
COMBO_OOS_MIN_RESOLVED_SAMPLES = 15

# PLACEHOLDER - calibrate to your actual broker's typical XAUUSD spread
# before trusting expectancy_r_after_costs. Retail CFD/spot gold spreads
# commonly range from well under $1 (tight ECN accounts) to several
# dollars (retail market-maker accounts, or any account during low
# liquidity / news). This default is a conservative placeholder, not a
# verified figure for any specific broker - it is NOT safe to treat as
# accurate without checking your own account.
SPREAD_USD = 0.30


def atr(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(window).mean()


def resolve_trade(high: np.ndarray, low: np.ndarray, start_idx: int, direction: int,
                   stop: float, target: float, max_lookahead: int = MAX_LOOKAHEAD) -> tuple[str, int | None]:
    """Walk forward from start_idx (inclusive) candle by candle until stop
    or target is touched (checked against each candle's high/low).
    Returns (outcome, resolved_idx) where outcome is
    "win"/"loss"/"unresolved" and resolved_idx is the candle position it
    resolved at (None if unresolved). start_idx should be the ENTRY
    candle itself (its own high/low after the open can already touch
    stop/target) - shared by historical mining and live signal tracking."""
    n = len(high)
    for j in range(start_idx, min(start_idx + max_lookahead, n)):
        hi, lo = high[j], low[j]
        hit_target = hi >= target if direction > 0 else lo <= target
        hit_stop = lo <= stop if direction > 0 else hi >= stop
        if hit_target and hit_stop:
            # Both levels touched within the same candle - OHLC alone
            # can't tell which came first. Assume the stop, so the
            # simulation never overstates the win rate.
            return "loss", j
        if hit_target:
            return "win", j
        if hit_stop:
            return "loss", j
    return "unresolved", None


def simulate_trades(df: pd.DataFrame, occurred: pd.Series, direction: int,
                     atr_series: pd.Series | None = None, rr_ratio: float = RR_RATIO) -> pd.DataFrame:
    """direction: +1 long, -1 short. For every True in `occurred` at
    candle i, simulate a fixed R:R trade ENTERED AT CANDLE i+1's OPEN
    (see module docstring - not candle i's own close, which isn't
    actually tradeable at signal time). Risk distance still comes from
    ATR measured at i (legitimately known then). One output row per
    occurrence with outcome in {"win", "loss", "unresolved"}, plus
    entry_index/resolved_index so callers can reconstruct the exact
    [entry, resolution] time window (used for news-window conditioning).

    `atr_series` lets a caller mining many patterns over the SAME candles
    (build_pattern_library.py, now mining hundreds of combo patterns on
    top of the atomic ones) compute ATR once and reuse it, instead of
    every single pattern recomputing the identical rolling calculation
    over the full candle history. Purely a performance cache - passing it
    in changes nothing about the math, since it's still exactly `atr(df)`.

    `rr_ratio` defaults to the system-wide fixed RR_RATIO (1:4) every
    live/production caller relies on and never overrides - it exists so
    an EXPLORATORY caller (scripts/explore_setups.py) can re-simulate the
    exact same occurrences at a different reward multiple without a
    second implementation of the walk-forward logic. Passing it changes
    only the target distance (stop distance is unaffected - still
    STOP_ATR_MULTIPLE * ATR); every existing call site that omits it
    behaves identically to before this parameter existed.
    """
    a = atr_series if atr_series is not None else atr(df)
    open_, high, low, close = (
        df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    )
    atr_vals = a.to_numpy()
    n = len(df)

    rows = []
    for i in np.flatnonzero(occurred.fillna(False).to_numpy()):
        entry_idx = i + 1
        if entry_idx >= n:
            continue  # signal candle is the last one available - no next-open to enter at yet
        risk = STOP_ATR_MULTIPLE * atr_vals[i]
        if not np.isfinite(risk) or risk <= 0:
            continue  # not enough history yet to compute ATR at this candle
        entry = open_[entry_idx]
        if direction > 0:
            stop, target = entry - risk, entry + rr_ratio * risk
        else:
            stop, target = entry + risk, entry - rr_ratio * risk

        outcome, resolved_idx = resolve_trade(high, low, entry_idx, direction, stop, target)
        candles_to_resolve = (resolved_idx - entry_idx) if resolved_idx is not None else None

        spread_cost_r = SPREAD_USD / risk if risk > 0 else 0.0
        if outcome == "win":
            net_r = rr_ratio - spread_cost_r
        elif outcome == "loss":
            net_r = -1.0 - spread_cost_r
        else:
            net_r = None

        rows.append({
            "signal_index": i, "index": entry_idx, "resolved_index": resolved_idx,
            "candles_to_resolve": candles_to_resolve,
            "direction": direction, "outcome": outcome,
            "entry": entry, "stop": stop, "target": target, "risk": risk,
            "rr_ratio": rr_ratio,
            "net_r": net_r,
        })
    return pd.DataFrame(rows)


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """95% Wilson score interval lower bound on a win rate - a
    sample-size-aware, conservative estimate of the TRUE win rate, not
    just the observed one. A 65% observed win rate on 30 trades and a
    65% observed win rate on 300 trades should not carry equal
    confidence; this is how signal_engine.py tells them apart instead of
    reporting raw win_rate as if it were certainty."""
    if n == 0:
        return 0.0
    phat = wins / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """(lower, upper) 95% Wilson score bounds - used by signal_journal.py
    to tell whether a live win rate has genuinely fallen below what was
    mined, or is just within normal sampling noise of it."""
    if n == 0:
        return 0.0, 1.0
    phat = wins / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom)


def _effective_rr_ratio(trades: pd.DataFrame, rr_ratio: float) -> float:
    """The R:R actually used to produce these trades, read back from
    simulate_trades()'s own `rr_ratio` column when present, falling back
    to the caller-supplied default otherwise (older/hand-built trade
    frames in tests may not carry the column) - so expectancy math and
    the reported `rr_ratio` always reflect what these specific trades
    were ACTUALLY simulated at, even when a caller passed a non-default
    rr_ratio into simulate_trades()."""
    if "rr_ratio" in trades.columns and not trades.empty:
        return float(trades["rr_ratio"].iloc[0])
    return rr_ratio


def _summarize(trades: pd.DataFrame, min_resolved: int,
                rr_ratio: float = RR_RATIO, min_win_rate: float = MIN_WIN_RATE) -> dict:
    if trades.empty:
        return {"samples": 0, "resolved": 0, "win_rate": None,
                "qualifies": False, "rr_ratio": rr_ratio}

    resolved = trades[trades["outcome"] != "unresolved"]
    n_resolved = len(resolved)
    if n_resolved == 0:
        return {"samples": int(len(trades)), "resolved": 0, "win_rate": None,
                "qualifies": False, "rr_ratio": rr_ratio}

    effective_rr = _effective_rr_ratio(trades, rr_ratio)
    wins = int((resolved["outcome"] == "win").sum())
    win_rate = wins / n_resolved
    expectancy_r = win_rate * effective_rr - (1 - win_rate) * 1.0  # in units of R, before costs
    expectancy_r_after_costs = float(resolved["net_r"].mean()) if "net_r" in resolved else None
    qualifies = (n_resolved >= min_resolved) and (win_rate >= min_win_rate)
    median_candles = resolved["candles_to_resolve"].median()

    return {
        "samples": int(len(trades)),
        "resolved": int(n_resolved),
        "unresolved_rate": round(1 - n_resolved / len(trades), 4),
        "win_rate": round(float(win_rate), 4),
        "win_rate_wilson_lower": round(wilson_lower_bound(wins, n_resolved), 4),
        "expectancy_r": round(float(expectancy_r), 4),
        "expectancy_r_after_costs": (round(expectancy_r_after_costs, 4)
                                      if expectancy_r_after_costs is not None else None),
        "rr_ratio": effective_rr,
        "avg_risk": round(float(resolved["risk"].mean()), 4),
        "median_candles_to_resolve": (round(float(median_candles), 1)
                                       if pd.notna(median_candles) else None),
        "qualifies": bool(qualifies),
    }


def summarize_trades(trades: pd.DataFrame, min_resolved: int = MIN_RESOLVED_SAMPLES,
                      oos_min_resolved: int = OOS_MIN_RESOLVED_SAMPLES,
                      rr_ratio: float = RR_RATIO, min_win_rate: float = MIN_WIN_RATE) -> dict:
    """Full-sample stats (as before) PLUS an out-of-sample check: the
    most recent OOS_FRACTION of this pattern's occurrences, summarized
    separately. `qualifies` now requires the out-of-sample slice to also
    clear MIN_WIN_RATE (on `oos_min_resolved`, relaxed vs `min_resolved`
    since it's a smaller slice) - a pattern that only worked in the older
    regime and has since stopped working will fail this even if its
    full-history average still looks fine.

    `min_resolved`/`oos_min_resolved` default to the atomic-pattern
    thresholds; combo_patterns.py-derived patterns pass the higher
    COMBO_* thresholds instead (see those constants above for why).

    `rr_ratio`/`min_win_rate` default to this system's fixed, live-signal
    values and are normally never overridden - they exist purely so an
    EXPLORATORY caller can ask "would this occurrence set qualify under a
    DIFFERENT R:R/bar" without a second copy of this function. When
    `trades` already carries simulate_trades()'s own `rr_ratio` column
    (the normal case), that's what's actually used for expectancy math -
    the `rr_ratio` parameter here only matters as a fallback for empty/
    legacy trade frames and for computing the OOS split's win-rate bar."""
    full = _summarize(trades, min_resolved, rr_ratio, min_win_rate)
    if trades.empty:
        full["out_of_sample"] = _summarize(trades, oos_min_resolved, rr_ratio, min_win_rate)
        return full

    split_at = int(len(trades) * (1 - OOS_FRACTION))
    oos_trades = trades.sort_values("signal_index").iloc[split_at:]
    oos = _summarize(oos_trades, oos_min_resolved, rr_ratio, min_win_rate)
    full["out_of_sample"] = oos

    full["qualifies"] = bool(full["qualifies"] and oos["qualifies"])
    return full


def tag_news_in_window(trades: pd.DataFrame, candles: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Adds a news_in_window boolean column: did a high-impact event fall
    between this trade's entry and its resolution (or, for unresolved
    trades, its max-lookahead cutoff)? Feeds summarize_trades_conditioned()
    to report a news-conditioned win rate alongside the baseline."""
    if trades.empty:
        return trades.assign(news_in_window=pd.Series(dtype=bool))
    from news_calendar import events_in_window  # local import: avoids a cycle at module load

    timestamps = candles["timestamp"].to_numpy()
    n = len(candles)
    flags = []
    for _, row in trades.iterrows():
        start_idx = int(row["index"])
        resolved = row["resolved_index"]
        end_idx = int(resolved) if pd.notna(resolved) else min(start_idx + MAX_LOOKAHEAD, n - 1)
        start_t = pd.Timestamp(timestamps[start_idx])
        end_t = pd.Timestamp(timestamps[end_idx])
        hits = events_in_window(events, start_t, end_t, impact="high")
        flags.append(len(hits) > 0)
    return trades.assign(news_in_window=flags)


def summarize_trades_conditioned(trades: pd.DataFrame) -> dict:
    """Split summarize_trades() by whether high-impact news fell inside
    the trade's window (requires tag_news_in_window() to have run first).
    Each side is gated independently against the same hard gates as
    everything else (including the out-of-sample check) - a pattern's
    news-window performance has to earn `qualifies: true` on its own,
    same as any other stat in this system."""
    if trades.empty or "news_in_window" not in trades.columns:
        return {"no_news": summarize_trades(trades), "news_in_window": summarize_trades(trades.iloc[0:0])}
    return {
        "no_news": summarize_trades(trades[~trades["news_in_window"]]),
        "news_in_window": summarize_trades(trades[trades["news_in_window"]]),
    }


def tag_regime(trades: pd.DataFrame, regime_labels: pd.Series) -> pd.DataFrame:
    """Adds a `regime` column: whichever combined volatility/trend regime
    (regime.combined_regime) was active at each trade's ENTRY candle -
    uses `index` (the entry index), not `signal_index`, since the regime
    at entry time is what a live signal would actually observe (same
    entry-timing convention simulate_trades() itself uses)."""
    if trades.empty:
        return trades.assign(regime=pd.Series(dtype=object))
    entry_idx = trades["index"].to_numpy()
    return trades.assign(regime=regime_labels.to_numpy()[entry_idx])


def summarize_trades_by_regime(trades: pd.DataFrame, min_resolved: int = MIN_RESOLVED_SAMPLES,
                                oos_min_resolved: int = OOS_MIN_RESOLVED_SAMPLES) -> dict:
    """Per-regime win rate/gate check (requires tag_regime() to have run
    first) - the SAME hard-gate rigor summarize_trades() applies overall,
    applied independently within EACH regime bucket. A pattern's overall
    blended win rate can look fine while it's actually only working in
    one specific market regime and losing in others; this is how that
    gets caught instead of averaged away. Regimes with too few trades to
    classify (None from regime.combined_regime's warm-up period) are
    excluded, not lumped into a bucket that would misrepresent them."""
    if trades.empty or "regime" not in trades.columns:
        return {}
    out = {}
    for regime_label, group in trades.groupby("regime"):
        if regime_label is None or pd.isna(regime_label):
            continue
        out[regime_label] = summarize_trades(group, min_resolved, oos_min_resolved)
    return out
