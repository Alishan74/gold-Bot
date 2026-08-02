"""
Versioned on-disk storage for the ML challenger's per-(symbol,
timeframe, direction, rr_ratio) models, with a hard promotion gate: a
newly trained model version is saved but NEVER made "active" (used for
live scoring) unless it independently earns it - by the EXACT SAME hard
gate the rule-based system uses for its patterns
(risk_reward.summarize_trades: win_rate >= MIN_WIN_RATE on
>= MIN_RESOLVED_SAMPLES resolved trades, with its own out-of-sample
check), applied here to the AGGREGATE of every purged walk-forward
validation fold's model-selected trades (every candle the model scored
>= PREDICTION_THRESHOLD), not just one holdout split. Reusing that exact
function - not a second, parallel "is this good enough" implementation -
is what guarantees the ML challenger is held to the identical bar as the
rule-based system, byte for byte, so a later comparison between the two
means what it appears to mean.

If a new version doesn't qualify, or qualifies but scores WORSE than the
currently active version, the currently active version keeps serving
live signals - training a model is never allowed to silently make
things worse. Every version, promoted or not, stays on disk (never
overwritten), so a promotion decision can always be inspected or manually
rolled back by pointing active.json at an older version_id.

rr_ratio dimension (added alongside train.py's RR_GRID multi-tier
search): a scalp-sized target (say 1.5R) and a swing-sized target (say
8R) off the SAME candle are genuinely different trades with different
win rates - qualification, promotion, and "currently active" are all
scoped PER TIER, independently, so a tier that stops working doesn't
drag down or get propped up by a different tier's numbers. Each tier
gets its own directory and its own active.json - see rr_tag() below for
the on-disk/pattern-name formatting shared with train.py and
live_signal.py, so all three always agree on what a given rr_ratio is
called.

On-disk layout:
    <registry_dir>/<symbol>/<timeframe>/<direction>/rr_<tag>/
        <version_id>/model.joblib
        <version_id>/meta.json       (validation summary, features, threshold)
        active.json                  ({"active_version": "<version_id>"} or absent)

Old (pre-multi-tier) promoted versions live at the old, un-tiered path
(no rr_<tag> segment) and are simply no longer looked up once this
schema is in use - they stay on disk (nothing is deleted) but a fresh
retrain under the new schema starts each tier's "currently active"
comparison from scratch, which is expected, not a bug: those old
versions predate the tier concept entirely and were trained on an
older, smaller feature set anyway.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from atomic_io import atomic_write_parquet, atomic_write_text  # noqa: E402
from risk_reward import RR_RATIO, summarize_trades  # noqa: E402

PREDICTION_THRESHOLD = 0.5  # the model's own decision boundary - "more likely to win than not"


def rr_tag(rr_ratio: float) -> str:
    """Canonical string form of an rr_ratio, shared by model_registry
    (directory names), train.py (progress logging), and live_signal.py
    (pattern names / journal keys) - having ONE formatting function
    means all three can never silently disagree on what "the 1.5R tier"
    is called. `:g` drops trailing zeros (4.0 -> "4", 1.5 -> "1.5") so
    the common integer ratios don't get an ugly ".0" suffix everywhere."""
    return f"{rr_ratio:g}"


def _rr_dir(registry_dir: Path, symbol: str, timeframe: str, direction: str, rr_ratio: float) -> Path:
    return Path(registry_dir) / symbol / timeframe / direction / f"rr_{rr_tag(rr_ratio)}"


def _active_pointer_path(registry_dir: Path, symbol: str, timeframe: str, direction: str, rr_ratio: float) -> Path:
    return _rr_dir(registry_dir, symbol, timeframe, direction, rr_ratio) / "active.json"


def list_rr_tiers(registry_dir: Path, symbol: str, timeframe: str, direction: str) -> list[float]:
    """Every rr_ratio that has at least one saved candidate on disk for
    this (symbol, timeframe, direction), sorted ascending - live_signal.py
    uses this to know which tiers to even attempt scoring, instead of
    hard-coding train.py's RR_GRID a second time (the grid could change
    between a training run and a later live-scoring run; this always
    reflects what's ACTUALLY on disk right now)."""
    d = Path(registry_dir) / symbol / timeframe / direction
    if not d.exists():
        return []
    tiers = []
    for child in d.iterdir():
        if not child.is_dir() or not child.name.startswith("rr_"):
            continue
        try:
            tiers.append(float(child.name[len("rr_"):]))
        except ValueError:
            continue
    return sorted(tiers)


def save_candidate(registry_dir: Path, symbol: str, timeframe: str, direction: str, model,
                    feature_columns: list[str], validation_summary: dict,
                    hyperparameters: dict | None = None,
                    hyperparameter_search: list[dict] | None = None,
                    rr_ratio: float = RR_RATIO, n_tiers_tested: int | None = None,
                    validation_trades=None,
                    shap_global: dict | None = None, distilled_rules: dict | None = None) -> str:
    """Writes a new, NOT-YET-ACTIVE model version to disk. Returns its
    version_id (a UTC timestamp - unique as long as two candidates for
    the same symbol/timeframe/direction/rr_ratio aren't saved in the
    same second, true for any realistic retrain cadence).

    `hyperparameters`: the winning config train.py's adaptive search
    selected for this candidate (see train.py's HYPERPARAMETER_GRID) -
    stored so any promoted (or rejected) version is auditable: which
    settings actually produced these numbers, not just the numbers.
    `hyperparameter_search`: every candidate config's own validation
    summary from that same search, for full transparency into what was
    tried and rejected, not just the winner - also lets a human sanity-
    check the winner wasn't a fluke against its nearest competitor.
    `n_tiers_tested`: how many rr_ratio tiers train.py's RR_GRID search
    tried for this (symbol, timeframe, direction) in the run that
    produced this candidate - stored purely for multiple-comparisons
    auditability (10 independent tiers being searched means any ONE
    tier's "qualifies" is more likely to be a lucky pass than if only
    one tier were ever tried; recording the honest denominator here is
    the same discipline validate_candidate.py's n_tested already
    applies to the rule-based side's pattern search).

    `validation_trades`: the pooled out-of-fold trades summarize_trades()
    graded this candidate on (train.py's _pooled_validation_trades(),
    now carrying model_probability/entry_timestamp/resolved_timestamp
    columns - see that function's own docstring) - persisted verbatim as
    a parquet file alongside model.joblib/meta.json so a REAL walk-
    forward backtest (equity curve, drawdown, calibration - see
    ml_system/backtest_report.py) can be built LATER from exactly the
    predictions that earned this candidate its validation_summary,
    without needing to reload candle history and re-run CV to
    reconstruct them. None (the default) skips writing this file -
    backward compatible for any caller that doesn't have trades to
    offer (there are none among train.py's own call sites, but keeping
    this optional avoids forcing every future caller to thread a
    DataFrame through just to save a model).

    `shap_global`/`distilled_rules`: ml_system/explainability.py's two
    reports (global feature-importance ranking, surrogate rule
    distillation) - see that module's docstring. Both optional/None the
    same way `validation_trades` is (a caller without them, or without
    `shap` installed, still gets a fully functional saved model)."""
    version_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    d = _rr_dir(registry_dir, symbol, timeframe, direction, rr_ratio) / version_id
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / "model.joblib")
    meta = {
        "version_id": version_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "timeframe": timeframe, "direction": direction,
        "rr_ratio": rr_ratio, "n_tiers_tested": n_tiers_tested,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "feature_columns": feature_columns,
        "validation_summary": validation_summary,
        "hyperparameters": hyperparameters,
        "hyperparameter_search": hyperparameter_search,
        "has_validation_trades": bool(validation_trades is not None and len(validation_trades) > 0),
        "shap_global": shap_global,
        "distilled_rules": distilled_rules,
    }
    atomic_write_text(d / "meta.json", json.dumps(meta, indent=2, default=str))
    if validation_trades is not None and len(validation_trades) > 0:
        atomic_write_parquet(validation_trades, d / "validation_trades.parquet")
    return version_id


def load_validation_trades(registry_dir: Path, symbol: str, timeframe: str, direction: str,
                            version_id: str, rr_ratio: float = RR_RATIO):
    """The persisted out-of-fold trade log for one specific model version
    (see save_candidate's `validation_trades` docstring), or None if this
    version predates persisted trades or genuinely had none. Returns a
    real pandas DataFrame - imported lazily here (not at module top)
    purely to keep model_registry.py's own import list minimal for
    callers that never touch this function."""
    import pandas as pd
    path = _rr_dir(registry_dir, symbol, timeframe, direction, rr_ratio) / version_id / "validation_trades.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_meta(registry_dir: Path, symbol: str, timeframe: str, direction: str, version_id: str,
              rr_ratio: float = RR_RATIO) -> dict | None:
    path = _rr_dir(registry_dir, symbol, timeframe, direction, rr_ratio) / version_id / "meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def promote(registry_dir: Path, symbol: str, timeframe: str, direction: str, version_id: str,
            rr_ratio: float = RR_RATIO) -> None:
    pointer_path = _active_pointer_path(registry_dir, symbol, timeframe, direction, rr_ratio)
    atomic_write_text(pointer_path, json.dumps({"active_version": version_id}, indent=2))


def load_active(registry_dir: Path, symbol: str, timeframe: str, direction: str,
                 rr_ratio: float = RR_RATIO) -> dict | None:
    """Returns {"version_id", "model", "meta"} for the currently active
    version of this (symbol, timeframe, direction, rr_ratio) tier, or
    None if nothing has ever been promoted for it."""
    pointer_path = _active_pointer_path(registry_dir, symbol, timeframe, direction, rr_ratio)
    if not pointer_path.exists():
        return None
    version_id = json.loads(pointer_path.read_text())["active_version"]
    d = _rr_dir(registry_dir, symbol, timeframe, direction, rr_ratio) / version_id
    model_path = d / "model.joblib"
    meta_path = d / "meta.json"
    if not model_path.exists() or not meta_path.exists():
        return None
    return {"version_id": version_id, "model": joblib.load(model_path), "meta": json.loads(meta_path.read_text())}


def write_lib_view(registry_dir: Path, symbol: str, out_dir: Path) -> None:
    """Writes a small pattern_library-SHAPED JSON per timeframe
    (out_dir/<symbol>_<timeframe>.json, one entry PER rr TIER, keyed
    "ml_model_rr<tag>") purely so signal_journal.py's EXISTING
    mined-win-rate lookup (_mined_win_rate_for, underlying detect_drift/
    pattern_scorecard/suspended_patterns/should_self_heal) can compare
    the ML challenger's LIVE performance against its own backtested
    validation performance - without a second, parallel "look up the
    reference win rate" implementation, and without a second, parallel
    self-healing/suspension mechanism: signal_journal.py's functions are
    already fully generic over the `pattern` string (see
    pattern_scorecard's `lib.get(pattern)`), so giving each tier its own
    pattern name here is all that's needed for every tier to be tracked,
    drift-detected, and suspended completely independently, with zero
    changes to signal_journal.py itself. Each tier's entry is
    "ambiguous" direction (as_long/as_short) since the same tier pattern
    name is used for both the bullish and bearish models of that tier -
    see live_signal.py's PATTERN_NAME_FOR. Call this after every
    train.py run so it reflects whatever is CURRENTLY active, not what
    was active at the time of some earlier training run."""
    import json

    registry_dir = Path(registry_dir)
    symbol_dir = registry_dir / symbol
    out_dir = Path(out_dir)
    if not symbol_dir.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    for tf_dir in sorted(p for p in symbol_dir.iterdir() if p.is_dir()):
        tf = tf_dir.name
        tiers = set(list_rr_tiers(registry_dir, symbol, tf, "bullish")) | \
            set(list_rr_tiers(registry_dir, symbol, tf, "bearish"))
        out = {}
        for rr in sorted(tiers):
            entry = {"direction": "ambiguous", "as_long": None, "as_short": None}
            for direction, key in (("bullish", "as_long"), ("bearish", "as_short")):
                active = load_active(registry_dir, symbol, tf, direction, rr)
                if active is not None:
                    vs = active["meta"].get("validation_summary", {})
                    entry[key] = {"win_rate": vs.get("win_rate"), "qualifies": vs.get("qualifies")}
            out[f"ml_model_rr{rr_tag(rr)}"] = entry
        atomic_write_text(out_dir / f"{symbol}_{tf}.json", json.dumps(out, indent=2, default=str))


def evaluate_and_maybe_promote(registry_dir: Path, symbol: str, timeframe: str, direction: str,
                                model, feature_columns: list[str], validation_trades,
                                hyperparameters: dict | None = None,
                                hyperparameter_search: list[dict] | None = None,
                                rr_ratio: float = RR_RATIO, n_tiers_tested: int | None = None,
                                shap_global: dict | None = None, distilled_rules: dict | None = None) -> dict:
    """The main entry point train.py calls after fitting a candidate
    model. `validation_trades`: every held-out (purged walk-forward)
    trade the model scored >= PREDICTION_THRESHOLD, in the SAME schema
    risk_reward.simulate_trades() produces (labeling.py guarantees this) -
    already simulated at THIS `rr_ratio`, so summarize_trades() below
    grades them at the tier they were actually generated for. Grades
    them with the identical summarize_trades() the rule-based system
    uses, saves the candidate regardless of outcome (so it's always
    inspectable), and promotes it ONLY if it independently qualifies AND
    is not worse than whatever's currently active FOR THIS SAME TIER (a
    candidate that still clears the bar but scores lower than the
    deployed model for that tier does not replace it - see module
    docstring). Tiers are never compared against each other here - a
    scalp tier and a swing tier qualifying or not qualifying are
    completely independent outcomes.

    `hyperparameters`/`hyperparameter_search`/`n_tiers_tested`: passed
    straight through to save_candidate() for auditability - see its
    docstring. `validation_trades` is ALSO passed straight through to
    save_candidate() for on-disk persistence (see that function's own
    docstring for why) - the exact same trades summarize_trades() below
    grades, so the persisted file and the reported validation_summary
    can never silently disagree about which trades they describe."""
    validation_summary = summarize_trades(validation_trades, rr_ratio=rr_ratio)
    version_id = save_candidate(registry_dir, symbol, timeframe, direction, model,
                                 feature_columns, validation_summary,
                                 hyperparameters, hyperparameter_search,
                                 rr_ratio=rr_ratio, n_tiers_tested=n_tiers_tested,
                                 validation_trades=validation_trades,
                                 shap_global=shap_global, distilled_rules=distilled_rules)

    result = {
        "version_id": version_id, "qualifies": validation_summary["qualifies"],
        "validation_summary": validation_summary, "promoted": False, "reason": "",
        "rr_ratio": rr_ratio,
    }

    if not validation_summary["qualifies"]:
        result["reason"] = (f"candidate does not independently qualify (win_rate="
                             f"{validation_summary.get('win_rate')}, resolved={validation_summary.get('resolved')})")
        return result

    current = load_active(registry_dir, symbol, timeframe, direction, rr_ratio)
    if current is not None:
        current_win_rate = current["meta"].get("validation_summary", {}).get("win_rate")
        new_win_rate = validation_summary.get("win_rate")
        if current_win_rate is not None and new_win_rate is not None and new_win_rate < current_win_rate:
            result["reason"] = (f"candidate qualifies ({new_win_rate:.1%}) but scores below the "
                                 f"currently active version for this tier ({current_win_rate:.1%}) - not promoted")
            return result

    promote(registry_dir, symbol, timeframe, direction, version_id, rr_ratio)
    result["promoted"] = True
    result["reason"] = "candidate qualifies and " + (
        "is the first version for this symbol/timeframe/direction/tier" if current is None
        else "matches or beats the currently active version for this tier"
    )
    return result
