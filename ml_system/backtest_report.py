"""
Walk-forward backtest report for a single ML challenger model version -
built entirely from the OUT-OF-FOLD validation trades train.py already
persists at training time (model_registry.save_candidate's
`validation_trades` parquet - see that function's own docstring), never
by re-simulating anything here. These are the EXACT, never-seen-during-
training predictions that earned this candidate its validation_summary,
so this report and the promotion decision can never silently describe
two different things - the same "one source of truth" discipline every
other shared calculation in this codebase already follows.

Deliberately NOT signal_journal.equity_curve() (which already exists,
tracks the LIVE journal, and starts empty for a freshly-promoted model,
growing slowly in real time as new signals actually fire and resolve).
This is the FULL historical walk-forward backtest instead: every
out-of-fold call the model made across every purged CV fold over the
model's ENTIRE training history, laid out chronologically - the honest
answer to "what would my equity curve have looked like trading every
signal this model called, going back years" rather than "what has it
done since I turned it on." The two are complementary, not redundant:
a model can have a beautiful multi-year backtest here and still be
watched closely via the live journal for regime drift the backtest
can't see coming.

Nothing here is a NEW statistical claim about the model - every number
is a re-summarization of trades that already independently cleared
risk_reward.summarize_trades()'s hard gate at promotion time. This
module exists to make that already-earned track record VISIBLE (equity
curve, drawdown, calibration), not to re-litigate whether it's valid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(trades: "pd.DataFrame | None") -> list[dict]:
    """Same output SHAPE as signal_journal.equity_curve() (resolved_at_utc/
    pattern/timeframe/outcome/trade_r/cumulative_r/cumulative_r_after_costs)
    on purpose, so the dashboard's existing drawEquityCurve() SVG
    renderer works against this completely unmodified - `pattern`/
    `timeframe` are always None here (this is one specific model
    version's own trades, not a mixed multi-pattern journal), kept only
    for shape compatibility with that shared renderer.

    Gross R is derived from `rr_ratio` + `outcome` (a win nets +rr_ratio,
    a loss nets -1.0, before spread) rather than stored directly -
    risk_reward.simulate_trades() only persists `net_r` (already spread-
    adjusted, see that function's docstring), so re-deriving the gross
    figure here is arithmetic on already-known columns, not a second
    trade-grading implementation."""
    if trades is None or trades.empty:
        return []
    resolved = trades[trades["outcome"].isin(["win", "loss"])].copy()
    if resolved.empty:
        return []
    resolved = resolved.sort_values("resolved_timestamp").reset_index(drop=True)

    gross_r = np.where(resolved["outcome"] == "win", resolved["rr_ratio"].astype(float), -1.0)
    cum_gross = pd.Series(gross_r).cumsum()
    cum_after_costs = resolved["net_r"].astype(float).cumsum()

    out = []
    for i, row in resolved.iterrows():
        out.append({
            "resolved_at_utc": row["resolved_timestamp"].isoformat() if pd.notna(row["resolved_timestamp"]) else None,
            "pattern": None, "timeframe": None, "outcome": row["outcome"],
            "trade_r": round(float(gross_r[i]), 3),
            "cumulative_r": round(float(cum_gross.iloc[i]), 3),
            "cumulative_r_after_costs": round(float(cum_after_costs.iloc[i]), 3),
        })
    return out


def summary_stats(trades: "pd.DataFrame | None") -> dict:
    """Headline backtest numbers, all derived from `net_r` (spread-
    adjusted) in chronological (resolution) order - drawdown and streaks
    are ORDER-DEPENDENT statistics, so getting the sort right here isn't
    cosmetic. `r_efficiency` (mean_r / std_r across trades) is
    deliberately NOT called "Sharpe ratio" - a real Sharpe ratio is
    annualized against a time period, and these trades have wildly
    different holding times (candles_to_resolve varies per trade); this
    is a real, useful "return per unit of variance" number, just not the
    specific textbook statistic, and mislabeling it would be exactly the
    kind of overclaiming this codebase avoids everywhere else."""
    if trades is None or trades.empty:
        return {"resolved": 0}
    resolved = trades[trades["outcome"].isin(["win", "loss"])].copy()
    if resolved.empty:
        return {"resolved": 0}
    resolved = resolved.sort_values("resolved_timestamp").reset_index(drop=True)
    net_r = resolved["net_r"].astype(float)

    cum = net_r.cumsum()
    running_peak = cum.cummax()
    drawdown = cum - running_peak  # same formula as circuit_breaker._max_drawdown_r, applied to this model's own trade sequence
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    wins = resolved.loc[resolved["outcome"] == "win", "net_r"].astype(float)
    losses = resolved.loc[resolved["outcome"] == "loss", "net_r"].astype(float)
    gross_loss = float(losses.sum())  # <= 0
    profit_factor = (float(wins.sum()) / abs(gross_loss)) if gross_loss < 0 else None

    std_r = float(net_r.std())
    r_efficiency = (float(net_r.mean()) / std_r) if std_r > 0 else None

    # Current streak - same logic circuit_breaker.check_circuit_breaker
    # uses on the live journal, applied here to this model's own
    # out-of-fold trade order instead.
    streak_type, streak_len = None, 0
    for outcome in reversed(resolved["outcome"].tolist()):
        if streak_type is None:
            streak_type, streak_len = outcome, 1
        elif outcome == streak_type:
            streak_len += 1
        else:
            break

    first_ts = resolved["resolved_timestamp"].iloc[0]
    last_ts = resolved["resolved_timestamp"].iloc[-1]
    return {
        "resolved": int(len(resolved)),
        "win_rate": round(float((resolved["outcome"] == "win").mean()), 4),
        "total_r_after_costs": round(float(net_r.sum()), 2),
        "mean_r_per_trade": round(float(net_r.mean()), 4),
        "r_efficiency": round(r_efficiency, 3) if r_efficiency is not None else None,
        "max_drawdown_r": round(max_dd, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "best_trade_r": round(float(net_r.max()), 3),
        "worst_trade_r": round(float(net_r.min()), 3),
        "current_streak": {"type": streak_type, "length": streak_len} if streak_type else None,
        "first_trade_at_utc": first_ts.isoformat() if pd.notna(first_ts) else None,
        "last_trade_at_utc": last_ts.isoformat() if pd.notna(last_ts) else None,
    }


def calibration_curve(trades: "pd.DataFrame | None", n_bins: int = 10) -> dict:
    """Predicted-vs-actual win rate, binned by the model's OWN
    out-of-fold probability (`model_probability` - see train.py's
    _pooled_validation_trades()) - a model is "calibrated" if, among
    every signal it scored ~0.55, roughly 55% of them actually won. This
    is a genuinely DIFFERENT question from win_rate_wilson_lower (which
    only checks the AGGREGATE win rate clears MIN_WIN_RATE) - a model
    can clear the aggregate bar while being badly miscalibrated at the
    edges (e.g. systematically underconfident on its strongest calls),
    and nothing else in this system currently checks that.

    Brier score (mean squared error between predicted probability and
    the realized 0/1 outcome) is the standard single-number calibration
    summary - lower is better; 0.25 is what an "always predict 0.5"
    model would score, a rough orientation point, not a pass/fail bar
    this system gates promotion on (it doesn't - calibration is reported
    for a human to read, same "diagnostic, not a hard gate" framing
    signal_journal.py's context_scorecard() already uses for its own
    non-gating statistics)."""
    if trades is None or trades.empty:
        return {"bins": [], "brier_score": None, "n": 0}
    resolved = trades[trades["outcome"].isin(["win", "loss"])].dropna(subset=["model_probability"])
    if resolved.empty:
        return {"bins": [], "brier_score": None, "n": 0}

    proba = resolved["model_probability"].astype(float).to_numpy()
    win = (resolved["outcome"] == "win").astype(float).to_numpy()
    brier = float(np.mean((proba - win) ** 2))

    lo, hi = float(proba.min()), float(proba.max())
    if hi <= lo:
        edges = np.array([lo, lo + 1e-9])
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(proba, edges[1:-1], right=True), 0, len(edges) - 2)

    bins = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        bins.append({
            "predicted_proba_mean": round(float(proba[mask].mean()), 4),
            "actual_win_rate": round(float(win[mask].mean()), 4),
            "n": n,
        })
    return {"bins": bins, "brier_score": round(brier, 4), "n": int(len(resolved))}


def full_report(trades: "pd.DataFrame | None") -> dict:
    """Everything above, bundled - the single call dashboard/server.py's
    ML backtest endpoint actually needs."""
    return {
        "summary": summary_stats(trades),
        "equity_curve": equity_curve(trades),
        "calibration": calibration_curve(trades),
    }


def compare_versions(candles: pd.DataFrame, direction: int, rr_ratio: float,
                      version_a: dict, version_b: dict,
                      feature_table: "pd.DataFrame | None" = None,
                      atr_series: "pd.Series | None" = None) -> dict:
    """Re-scores TWO model versions (each model_registry.load_active()'s
    own {"version_id","model","meta"} shape - or load_meta()+joblib.load()
    for an arbitrary non-active version a caller wants to compare) against
    the EXACT SAME candle window - deliberately NOT each version's own
    persisted validation_trades (see model_registry.save_candidate's
    docstring), because two retrains see different, GROWING candle
    history, so their own validation windows differ and aren't directly
    comparable to each other. This answers a genuinely different
    question than full_report() above: "would version A or version B
    have called more/better trades over this SAME specific stretch," not
    "how did each perform over its own full training history."

    Uses each version's OWN stored `feature_columns` to reindex the
    shared feature table - exactly live_signal.py's own reindex
    convention - so an older, smaller-feature-set version is graded
    fairly on what IT actually saw at the time, never silently handed
    columns it was never trained with.

    Deliberately a SEPARATE, simpler "score this specific already-fitted
    model against this specific candle window" path rather than reusing
    train.py's _pooled_validation_trades() - that function is purged-
    walk-forward-CV-specific machinery (re-fits a fresh model per fold);
    this is scoring two ALREADY-FITTED models once each, a genuinely
    different operation that would only be complicated by forcing it
    through the CV-fold plumbing.

    IMPORTANT CALLER RESPONSIBILITY, not something this function can
    enforce for you: `candles` should be a window that is genuinely
    OUT-OF-SAMPLE for BOTH versions (strictly after whichever model was
    trained most recently) for this to mean what it looks like it means.
    Neither version's own training cutoff is threaded through here, so
    this function has no way to warn you if the window you passed
    overlaps one or both models' own training data - doing that would
    silently inflate both "backtests" into in-sample scoring, the exact
    look-ahead-adjacent mistake this entire codebase works hard to avoid
    everywhere else. Pick the comparison window deliberately."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from risk_reward import atr as _atr, simulate_trades  # noqa: E402

    if feature_table is None:
        import features as features_module
        feature_table = features_module.compute_features(candles)
    if atr_series is None:
        atr_series = _atr(candles)

    out = {}
    for label, version in (("version_a", version_a), ("version_b", version_b)):
        feature_columns = version["meta"]["feature_columns"]
        X = feature_table.reindex(columns=feature_columns)
        known = ~X.isna().all(axis=1)
        proba = pd.Series(np.nan, index=candles.index)
        if known.any():
            proba.loc[known] = version["model"].predict_proba(X.loc[known])[:, 1]
        threshold = version["meta"].get("prediction_threshold", 0.5)
        occurred = (proba >= threshold).fillna(False)

        trades = simulate_trades(candles, occurred, direction, atr_series=atr_series, rr_ratio=rr_ratio)
        proba_by_signal_index = proba.to_dict()
        trades["model_probability"] = trades["signal_index"].map(proba_by_signal_index)
        ts = candles["timestamp"]
        trades["entry_timestamp"] = ts.to_numpy()[trades["index"].to_numpy()]
        resolved_pos = trades["resolved_index"]
        has_resolution = resolved_pos.notna()
        trades["resolved_timestamp"] = pd.NaT
        if has_resolution.any():
            trades.loc[has_resolution, "resolved_timestamp"] = ts.to_numpy()[
                resolved_pos[has_resolution].to_numpy().astype(int)
            ]

        out[label] = {"version_id": version["version_id"], "n_signals_called": int(occurred.sum()),
                       **full_report(trades)}
    return out
