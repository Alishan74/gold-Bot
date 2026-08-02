"""
Train (or retrain) the ML challenger's per-timeframe, per-direction
models against every candle in history - not just occurrences of a
hand-picked pattern - and promote a new model version only if it
independently clears the SAME hard gate the rule-based system's patterns
have to clear, evaluated honestly via purged/embargoed walk-forward
cross-validation (see validation.py for why a plain train/test split
isn't enough here).

Pipeline, per (timeframe, direction):
  1. features.compute_features() - one row of ~40 numeric features per
     candle, strictly causal.
  2. labeling.label_all_candles() - risk_reward.simulate_trades() itself,
     called on EVERY candle instead of only pattern occurrences: does the
     fixed 1:4 R:R trade win, lose, or stay unresolved from here.
  3. validation.purged_walk_forward_splits() over the FULL candle-index
     range (not the resolved-only subset - the purge boundary has to
     reflect real candle-time distance, so filtering to resolved rows
     happens AFTER splitting, not before, or the purge window's meaning
     would be silently wrong).
  4. A fresh model is trained on each fold's (purged) training rows and
     scored on that fold's validation rows - the validation predictions
     from every fold are pooled into one honest, non-overlapping,
     never-trained-on-its-own-future set of "trades the model would have
     called."
  5. model_registry.evaluate_and_maybe_promote() grades that pooled set
     with the identical risk_reward.summarize_trades() the rule-based
     system uses, and promotes a new model version only if it qualifies
     and doesn't score below whatever's currently active.
  6. The model that actually gets SAVED (and promoted, if it earns it)
     is trained on ALL available resolved data, not just one fold's
     training slice - cross-validation estimates how well an approach
     generalizes, the deployed model should still use everything
     available.

Usage:
    python ml_system/train.py --symbol XAUUSD --data-dir data \\
        --registry-dir ml_registry --n-splits 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent / "."))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import explainability  # noqa: E402
import model_registry  # noqa: E402
from ensemble import EnsembleClassifier  # noqa: E402
from feature_synthesis import apply_synthesized_features, load_synthesized_features  # noqa: E402
from features import compute_cross_timeframe_features, compute_features, coarser_timeframes  # noqa: E402
from labeling import label_all_candles, label_window  # noqa: E402
from validation import purged_walk_forward_splits  # noqa: E402

from build_history import TIMEFRAME_MINUTES  # noqa: E402
from heartbeat import track  # noqa: E402
from risk_reward import RR_RATIO, atr as _atr, summarize_trades  # noqa: E402

N_SPLITS_DEFAULT = 5
EMBARGO_CANDLES = 10
MIN_CANDLES_TO_TRAIN = 2000  # below this, purged CV folds are too thin to mean anything

# Multi-tier R:R search: instead of every model answering only "does the
# fixed 1:4 trade from here win," each (timeframe, direction) is trained
# and independently qualified/promoted at EVERY tier in this grid - a
# tight scalp target and a wide swing target off the SAME candle are
# genuinely different trades with different win rates, and forcing
# everything through one fixed R:R would systematically bias what gets
# found toward whichever style that one ratio happens to favor. Spacing
# is denser at the low end (1.25 -> 3.0) where the win-rate/R:R tradeoff
# curve moves fastest, coarser toward the swing end (6 -> 10) where
# resolved-sample counts get naturally thinner anyway (a wider target
# takes longer to hit, so MAX_LOOKAHEAD cuts off more occurrences as
# "unresolved" the wider the target gets).
#
# Honest cost of "all tiers, not a handful": this is 10 fully independent
# purged-CV + hyperparameter-search + promotion decisions per
# (timeframe, direction) instead of 1 - roughly 10x today's training
# runtime, AND 10x the number of "does this qualify" tests being run
# (10 tiers x 6 timeframes x 2 directions = 120), which is exactly the
# multiple-comparisons exposure validate_candidate.py's n_tested
# discipline exists for on the rule-based side. Each tier's own hard gate
# (MIN_WIN_RATE + out-of-sample check, both from risk_reward.py, applied
# independently per tier) is real protection, but it is NOT a formal FDR
# correction across tiers the way validate_candidate.py applies one
# across mined pattern conjunctions - model_registry.save_candidate()
# records `n_tiers_tested` in every candidate's meta.json specifically so
# this exposure stays visible and auditable rather than silently assumed
# away. --rr-grid lets a run search a smaller/different grid (e.g. for a
# quick test) without editing this file.
RR_GRID = (1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)

# Adaptive hyperparameter search: instead of one fixed, hand-picked
# HistGradientBoostingClassifier config, every retrain tries a small,
# BOUNDED set of genuinely different configs - shallower/faster,
# medium (the config this system shipped with originally), and
# deeper/more-regularized - through the SAME purged walk-forward CV
# used for everything else, and keeps whichever one actually earns the
# best honest out-of-fold performance. This is the literal meaning of
# "self-teaching": the system doesn't just learn from new data with a
# fixed learning process, it also periodically re-asks "is this still
# the right way to learn from this data," the same way a real
# quant desk revisits model architecture, not just retrains on a
# schedule with the same knobs forever.
#
# Deliberately kept SMALL (3 configs, not a sweep of 30): every extra
# config evaluated against the same validation folds is one more
# opportunity to pick a winner that got lucky on THIS data rather than
# one that's genuinely better - the classic multiple-comparisons /
# "selection bias" problem. Three bounded, qualitatively different
# configs (not a fine grid around one idea) keeps that inflation small
# while still giving the system a real choice; the promotion gate in
# model_registry.py (independent qualification vs. whatever's currently
# deployed) is the actual backstop against a spuriously-selected
# config making it to live signals, not the search itself.
HYPERPARAMETER_GRID = [
    {
        "name": "shallow_fast",
        "params": {"max_depth": 3, "learning_rate": 0.1, "max_iter": 150, "l2_regularization": 1.0},
    },
    {
        "name": "medium_default",
        "params": {"max_depth": 6, "learning_rate": 0.05, "max_iter": 200, "l2_regularization": 1.0},
    },
    {
        "name": "deep_regularized",
        "params": {"max_depth": 10, "learning_rate": 0.03, "max_iter": 300, "l2_regularization": 3.0},
    },
]


def _compute_feature_table(candles: pd.DataFrame,
                            coarser_candles_by_tf: "dict[str, tuple[pd.DataFrame, float]] | None" = None,
                            synth_defs: "list[dict] | None" = None) -> pd.DataFrame:
    """The feature table (features.compute_features(), optionally plus
    cross-timeframe context AND synthesized features) - direction- and
    rr_ratio-INDEPENDENT, so train_all() computes this exactly ONCE per
    timeframe and reuses it across both directions and every RR_GRID
    tier, instead of recomputing an 89-column feature pass over the full
    candle history up to 20 times (2 directions x 10 tiers) for
    identical output every time. `coarser_candles_by_tf`: optional -
    when given, cross-timeframe context columns are concatenated on.
    Omitting it (the default) gives byte-identical behavior to before
    cross-timeframe features existed.

    `synth_defs`: ml_system/feature_synthesis.load_synthesized_features()'s
    output for this (symbol, timeframe) - each already-accepted
    synthesized feature (see scripts/synthesize_features.py's three-
    layer validation) gets evaluated and concatenated the SAME way
    cross-timeframe context is, right after the BASE compute_features()
    call and before cross-timeframe context is added - synthesized
    features' leaves reference only base feature columns (never ctx_*
    ones), so this ordering is what guarantees their leaf lookups can
    never accidentally resolve against a run-specific cross-timeframe
    column that may or may not be present. Omitting it (the default)
    gives byte-identical behavior to before feature synthesis existed."""
    features = compute_features(candles)
    if synth_defs:
        synth = apply_synthesized_features(features, synth_defs)
        features = pd.concat([features, synth], axis=1)
    if coarser_candles_by_tf:
        cross_tf = compute_cross_timeframe_features(candles, coarser_candles_by_tf)
        features = pd.concat([features, cross_tf], axis=1)
    return features


def _label_dataset(candles: pd.DataFrame, direction: int, atr_series: pd.Series,
                    rr_ratio: float = RR_RATIO):
    """Returns (y, labels_df) aligned to `candles`'s original position -
    y is 1.0/0.0/NaN (NaN = unresolved, excluded from training but
    keeping the row's POSITION intact is what lets the purge logic
    reason about real candle-time distance). Unlike features, labels DO
    depend on both `direction` and `rr_ratio` (a wider target changes
    which trades resolve win/loss/unresolved) - re-run once per
    (direction, rr_ratio) combination, but cheap relative to feature
    computation since it reuses the SAME precomputed `atr_series` rather
    than recomputing ATR from scratch each time."""
    labels = label_all_candles(candles, direction, atr_series=atr_series, rr_ratio=rr_ratio)

    y = pd.Series(np.nan, index=candles.index)
    resolved = labels[labels["outcome"] != "unresolved"]
    win_mask = resolved["outcome"] == "win"
    y.loc[resolved["signal_index"].to_numpy()] = win_mask.to_numpy().astype(float)
    return y, labels


def _fit_model(X: pd.DataFrame, y: pd.Series, hyperparams: dict) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(**hyperparams, random_state=0)
    model.fit(X, y)
    return model


def _pooled_validation_trades(candles: pd.DataFrame, features: pd.DataFrame, y: pd.Series,
                               labels: pd.DataFrame, n_splits: int, hyperparams: dict,
                               progress_label: str = "") -> pd.DataFrame:
    """Runs purged walk-forward CV, pools every fold's validation-set
    predictions where the model scored >= PREDICTION_THRESHOLD, and
    returns the corresponding trade rows (in simulate_trades()'s own
    schema, via `labels`) for model_registry to grade. Empty DataFrame
    if there isn't enough data for even one valid fold.

    `progress_label` prints one line per fold (config name + fold index +
    how long that fold's fit+predict took) - without this, a single call
    silently runs up to n_splits full HistGradientBoostingClassifier fits
    (each potentially minutes on a large timeframe's full resolved-trade
    count, especially the deeper HYPERPARAMETER_GRID configs) with zero
    stdout in between. Verified against a real run: that silence alone
    was long enough to trip scripts/supervise.py's 600s stall-detector
    and kill a training run that was genuinely still working, not stuck -
    this is the fix, not a cosmetic addition."""
    n = len(candles)
    folds = list(purged_walk_forward_splits(n, label_window(), n_splits=n_splits, embargo=EMBARGO_CANDLES))
    if not folds:
        return labels.iloc[0:0]

    labels_by_signal_index = labels.set_index("signal_index")
    selected_signal_indices = []
    proba_by_signal_index: dict[int, float] = {}

    for fold_i, (train_idx, val_idx) in enumerate(folds, start=1):
        train_resolved = train_idx[~y.iloc[train_idx].isna().to_numpy()]
        val_resolved = val_idx[~y.iloc[val_idx].isna().to_numpy()]
        if len(train_resolved) < 30 or len(val_resolved) == 0:
            if progress_label:
                print(f"    {progress_label} fold {fold_i}/{len(folds)}: too few resolved rows - skipped", flush=True)
            continue

        X_train, y_train = features.iloc[train_resolved], y.iloc[train_resolved]
        X_val = features.iloc[val_resolved]

        # A feature that's ENTIRELY missing across this fold's training
        # window (typically a long-warmup feature - cross-timeframe
        # context, swing-level memory, a synthesized feature - that
        # hasn't accumulated enough history yet this early in a
        # walk-forward fold) crashes HistGradientBoostingClassifier's own
        # binning step below sklearn's radar: _find_binning_thresholds()
        # has a safe fallback for a column with exactly 1 distinct value,
        # but none for 0 (sliding_window_view on an empty array raises
        # "window shape cannot be larger than input array shape") -
        # verified against the real crash, not a hypothetical. Skipping
        # the fold (same handling as "too few resolved rows" above) is
        # the correct fix, not silently dropping the column: dropping it
        # here would desync X_train's columns from X_val's below, and
        # every OTHER fold - later in the walk-forward split, with more
        # history behind it - almost certainly has real values for this
        # same feature, so the feature itself stays valid for the model
        # this search is choosing between.
        all_nan_cols = X_train.columns[X_train.isna().all(axis=0)]
        if len(all_nan_cols) > 0:
            if progress_label:
                print(f"    {progress_label} fold {fold_i}/{len(folds)}: {len(all_nan_cols)} feature(s) "
                      f"entirely missing in this fold's training window (not enough history yet) - skipped: "
                      f"{list(all_nan_cols)}",
                      flush=True)
            continue

        fold_start = time.monotonic()
        model = _fit_model(X_train, y_train, hyperparams)
        proba = model.predict_proba(X_val)[:, 1]
        if progress_label:
            print(f"    {progress_label} fold {fold_i}/{len(folds)}: "
                  f"{len(train_resolved)} train / {len(val_resolved)} val rows, "
                  f"{time.monotonic() - fold_start:.0f}s", flush=True)
        called_mask = proba >= model_registry.PREDICTION_THRESHOLD
        called_win = val_resolved[called_mask]
        selected_signal_indices.extend(called_win.tolist())
        # Each candle's signal_index falls in exactly ONE fold's validation
        # block (walk-forward blocks are disjoint), so this dict is never
        # overwritten by a later fold for the same key - it's purely a
        # convenience lookup, not a "last write wins" resolution.
        for sig_idx, p in zip(called_win.tolist(), proba[called_mask].tolist()):
            proba_by_signal_index[sig_idx] = p

    return _finalize_pooled_trades(candles, labels_by_signal_index, selected_signal_indices, proba_by_signal_index)


def _finalize_pooled_trades(candles: pd.DataFrame, labels_by_signal_index: pd.DataFrame,
                             selected_signal_indices: list, proba_by_signal_index: dict) -> pd.DataFrame:
    """Shared tail for both _pooled_validation_trades() (single model)
    and _pooled_validation_trades_ensemble() (blended prediction across
    several models) - once a caller has ITS OWN selected_signal_indices/
    proba_by_signal_index (however it computed those probabilities),
    building the final graded trade frame is identical: look up each
    selected candle's label row, attach the probability that selected
    it, attach real wall-clock timestamps. Factored out specifically so
    the two callers can never silently drift on this part."""
    if not selected_signal_indices:
        return labels_by_signal_index.iloc[0:0].reset_index()

    matched = labels_by_signal_index.loc[
        labels_by_signal_index.index.intersection(selected_signal_indices)
    ].reset_index()
    matched = matched.sort_values("signal_index").reset_index(drop=True)
    # Persisted alongside the trade for two later uses (backtest_report.py's
    # calibration check, dashboard's per-signal audit trail): the EXACT
    # out-of-fold probability that selected this row - not re-derived
    # later from a model that may have since been retrained, which would
    # silently answer a different question ("what would today's model
    # have said") than "what did the OOF-selecting model actually say at
    # validation time."
    matched["model_probability"] = matched["signal_index"].map(proba_by_signal_index)
    # Real wall-clock timestamps, not just positional indices - so a later
    # backtest report/chart never needs to reload the full candle history
    # just to plot on a real time axis. `index`/`resolved_index` are
    # simulate_trades()'s own entry/resolution candle positions (see
    # risk_reward.py) - resolved_index is None for "unresolved" rows, kept
    # as NaT rather than crashing on a missing lookup.
    ts = candles["timestamp"]
    matched["entry_timestamp"] = ts.to_numpy()[matched["index"].to_numpy()]
    resolved_pos = matched["resolved_index"]
    has_resolution = resolved_pos.notna()
    matched["resolved_timestamp"] = pd.NaT
    matched.loc[has_resolution, "resolved_timestamp"] = ts.to_numpy()[
        resolved_pos[has_resolution].to_numpy().astype(int)
    ]
    return matched


def _pooled_validation_trades_ensemble(candles: pd.DataFrame, features: pd.DataFrame, y: pd.Series,
                                        labels: pd.DataFrame, n_splits: int, hyperparams_list: list[dict],
                                        progress_label: str = "") -> pd.DataFrame:
    """Same purged walk-forward CV loop as _pooled_validation_trades(),
    except EACH fold fits one model PER entry in `hyperparams_list` (on
    that fold's own purged training rows, same as always) and pools the
    EQUAL-WEIGHT AVERAGE of their predicted probabilities, instead of a
    single model's own probability - the honest out-of-fold estimate of
    how an ENSEMBLE of these configs would actually have performed, not
    an assumption that blending helps. See ensemble.EnsembleClassifier
    for the matching wrapper used to deploy the winning blend for real,
    and _select_hyperparameters_or_ensemble() for how this gets compared
    against the single best config on equal footing."""
    n = len(candles)
    folds = list(purged_walk_forward_splits(n, label_window(), n_splits=n_splits, embargo=EMBARGO_CANDLES))
    if not folds:
        return labels.iloc[0:0]

    labels_by_signal_index = labels.set_index("signal_index")
    selected_signal_indices = []
    proba_by_signal_index: dict[int, float] = {}

    for fold_i, (train_idx, val_idx) in enumerate(folds, start=1):
        train_resolved = train_idx[~y.iloc[train_idx].isna().to_numpy()]
        val_resolved = val_idx[~y.iloc[val_idx].isna().to_numpy()]
        if len(train_resolved) < 30 or len(val_resolved) == 0:
            if progress_label:
                print(f"    {progress_label} fold {fold_i}/{len(folds)}: too few resolved rows - skipped", flush=True)
            continue

        X_train, y_train = features.iloc[train_resolved], y.iloc[train_resolved]
        X_val = features.iloc[val_resolved]

        # See the identical guard in _pooled_validation_trades() above for
        # why this is needed - same crash (HistGradientBoostingClassifier's
        # binning step on a feature with zero non-missing values in this
        # fold's training window), same fix (skip the fold, don't drop the
        # column - every ensemble member below would otherwise crash on
        # the same fit).
        all_nan_cols = X_train.columns[X_train.isna().all(axis=0)]
        if len(all_nan_cols) > 0:
            if progress_label:
                print(f"    {progress_label} fold {fold_i}/{len(folds)}: {len(all_nan_cols)} feature(s) "
                      f"entirely missing in this fold's training window (not enough history yet) - skipped: "
                      f"{list(all_nan_cols)}",
                      flush=True)
            continue

        fold_start = time.monotonic()
        member_probas = [_fit_model(X_train, y_train, hp).predict_proba(X_val)[:, 1] for hp in hyperparams_list]
        proba = np.mean(member_probas, axis=0)
        if progress_label:
            print(f"    {progress_label} fold {fold_i}/{len(folds)}: "
                  f"{len(train_resolved)} train / {len(val_resolved)} val rows, "
                  f"{len(hyperparams_list)} ensemble members, {time.monotonic() - fold_start:.0f}s", flush=True)
        called_mask = proba >= model_registry.PREDICTION_THRESHOLD
        called_win = val_resolved[called_mask]
        selected_signal_indices.extend(called_win.tolist())
        for sig_idx, p in zip(called_win.tolist(), proba[called_mask].tolist()):
            proba_by_signal_index[sig_idx] = p

    return _finalize_pooled_trades(candles, labels_by_signal_index, selected_signal_indices, proba_by_signal_index)


def _select_hyperparameters(candles: pd.DataFrame, features: pd.DataFrame, y: pd.Series,
                             labels: pd.DataFrame, n_splits: int) -> tuple[str, dict, pd.DataFrame, list[dict]]:
    """Runs the full purged walk-forward CV loop once PER CANDIDATE
    config in HYPERPARAMETER_GRID, grades each with the identical
    summarize_trades() the final promotion decision uses, and returns
    the winner - (winning_name, winning_params, its pooled validation
    trades, every candidate's own summary for the audit trail).

    Ranked by win_rate_wilson_lower (the SAME conservative,
    sample-size-aware number every other qualification decision in this
    system already uses - not raw win rate, which would favor whichever
    config's OOF trades happened to be a smaller, noisier sample), tied
    by resolved sample count. Using the Wilson lower bound here isn't
    just consistency with the rest of the system - it's also a partial
    defense against the multiple-comparisons problem described above:
    a config that wins on a small, lucky sample gets penalized for that
    smallness rather than rewarded for it."""
    candidates = []
    for entry in HYPERPARAMETER_GRID:
        trades = _pooled_validation_trades(candles, features, y, labels, n_splits, entry["params"],
                                            progress_label=f"[{entry['name']}]")
        summary = summarize_trades(trades)
        candidates.append({"name": entry["name"], "params": entry["params"],
                            "validation_summary": summary, "trades": trades})

    def _rank_key(c):
        s = c["validation_summary"]
        return (s.get("win_rate_wilson_lower") or -1.0, s.get("resolved") or 0)

    best = max(candidates, key=_rank_key)
    search_report = [
        {"name": c["name"], "params": c["params"], "validation_summary": c["validation_summary"]}
        for c in candidates
    ]
    return best["name"], best["params"], best["trades"], search_report


# How many of the top-ranked HYPERPARAMETER_GRID configs (by their OWN
# individual Wilson lower bound) get blended into an EnsembleClassifier
# candidate for _select_hyperparameters_or_ensemble() to compare against
# the single best config - see that function's own docstring for why
# blending is a genuine comparison, not an assumed win. Capped at
# HYPERPARAMETER_GRID's own length (currently 3) regardless of this
# constant, so raising it is safe even before the grid itself grows.
ENSEMBLE_TOP_K = 3


def _select_hyperparameters_or_ensemble(candles: pd.DataFrame, features: pd.DataFrame, y: pd.Series,
                                         labels: pd.DataFrame, n_splits: int
                                         ) -> tuple[str, "dict | list[dict]", pd.DataFrame, list[dict], bool]:
    """Extends _select_hyperparameters()'s single-config search with a
    genuine comparison against BLENDING the top ENSEMBLE_TOP_K configs
    (by their OWN individual Wilson lower bound) into an equal-weight
    ensemble (ensemble.EnsembleClassifier), evaluated on the SAME honest
    purged walk-forward out-of-fold criterion every other candidate here
    is graded by (_pooled_validation_trades_ensemble(), not an assumption
    that blending helps.

    Returns (winning_name, winning_params_or_list, its pooled validation
    trades, every candidate's own summary INCLUDING the ensemble's,
    is_ensemble). `winning_params_or_list` is a single hyperparams dict
    when a lone config wins (identical shape to _select_hyperparameters()'s
    old return) or a LIST of hyperparams dicts (the ensemble's members)
    when the ensemble wins - train_one() checks `is_ensemble` to decide
    which final-fit path to take.

    Why this can genuinely go either way, not "ensembles are always
    better": blending several models' probabilities smooths out any ONE
    config's idiosyncratic overfitting, but it can just as easily dilute
    a genuinely strong single config's confident, correct calls with two
    weaker configs' noisier ones - especially plausible here, since
    HYPERPARAMETER_GRID's three configs are DELIBERATELY very different
    in complexity (shallow_fast/medium_default/deep_regularized), not
    near-duplicates of each other the way an ensemble usually wants its
    members to be. Letting the honest out-of-fold score decide, every
    retrain, is what keeps this a real comparison instead of a coin flip
    dressed up as one."""
    candidates = []
    for entry in HYPERPARAMETER_GRID:
        trades = _pooled_validation_trades(candles, features, y, labels, n_splits, entry["params"],
                                            progress_label=f"[{entry['name']}]")
        summary = summarize_trades(trades)
        candidates.append({"name": entry["name"], "params": entry["params"],
                            "validation_summary": summary, "trades": trades})

    def _rank_key(c):
        s = c["validation_summary"]
        return (s.get("win_rate_wilson_lower") or -1.0, s.get("resolved") or 0)

    candidates_ranked = sorted(candidates, key=_rank_key, reverse=True)
    ensemble_members = candidates_ranked[:min(ENSEMBLE_TOP_K, len(candidates_ranked))]

    ensemble_result = None
    if len(ensemble_members) >= 2:
        ensemble_params = [c["params"] for c in ensemble_members]
        ensemble_trades = _pooled_validation_trades_ensemble(
            candles, features, y, labels, n_splits, ensemble_params,
            progress_label=f"[ensemble_top{len(ensemble_members)}]",
        )
        ensemble_result = {
            "name": f"ensemble_top{len(ensemble_members)}", "params": ensemble_params,
            "member_names": [c["name"] for c in ensemble_members],
            "validation_summary": summarize_trades(ensemble_trades), "trades": ensemble_trades,
        }

    all_results = candidates + ([ensemble_result] if ensemble_result else [])
    winner = max(all_results, key=_rank_key)
    search_report = [
        {"name": c["name"], "params": c["params"], "validation_summary": c["validation_summary"]}
        for c in all_results
    ]
    is_ensemble = winner is ensemble_result
    return winner["name"], winner["params"], winner["trades"], search_report, is_ensemble


def train_one(candles: pd.DataFrame, symbol: str, timeframe: str, direction: int,
              registry_dir: Path, n_splits: int = N_SPLITS_DEFAULT,
              coarser_candles_by_tf: "dict[str, tuple[pd.DataFrame, float]] | None" = None,
              rr_ratio: float = RR_RATIO, n_tiers_tested: int | None = None,
              features: pd.DataFrame | None = None, atr_series: pd.Series | None = None,
              synth_defs: "list[dict] | None" = None) -> dict | None:
    """`features`/`atr_series`: optional precomputed inputs - train_all()
    below computes both ONCE per timeframe and passes them in for every
    (direction, rr_ratio) combination, since neither depends on either
    (see _compute_feature_table()/_label_dataset()). Omitting them (the
    default) computes fresh, exactly as before multi-tier search existed -
    a direct call to train_one() for a single tier still works unchanged.
    `synth_defs`: only used when `features` is omitted (a caller
    supplying its own precomputed `features` has, by construction,
    already decided what's in it - see train_all())."""
    direction_label = "bullish" if direction > 0 else "bearish"
    tier_label = f"rr{model_registry.rr_tag(rr_ratio)}"
    if len(candles) < MIN_CANDLES_TO_TRAIN:
        print(f"  {timeframe}/{direction_label}/{tier_label}: only {len(candles)} candles "
              f"(< {MIN_CANDLES_TO_TRAIN}) - skipping")
        return None

    if features is None:
        features = _compute_feature_table(candles, coarser_candles_by_tf, synth_defs)
    if atr_series is None:
        atr_series = _atr(candles)
    y, labels = _label_dataset(candles, direction, atr_series, rr_ratio=rr_ratio)
    n_resolved = int((~y.isna()).sum())
    ctx_note = f", cross-tf context from {list(coarser_candles_by_tf)}" if coarser_candles_by_tf else ""
    print(f"  {timeframe}/{direction_label}/{tier_label}: {len(candles)} candles, "
          f"{n_resolved} resolved trades to learn from ({len(features.columns)} features{ctx_note})")

    # Adaptive hyperparameter search: try every config in
    # HYPERPARAMETER_GRID through the same purged walk-forward CV, PLUS
    # an equal-weight ensemble of the top few configs (see ENSEMBLE_TOP_K/
    # _select_hyperparameters_or_ensemble() above), keep whichever
    # candidate - single config or ensemble - earned the best honest
    # out-of-fold Wilson lower bound.
    winning_name, winning_params, validation_trades, search_report, is_ensemble = (
        _select_hyperparameters_or_ensemble(candles, features, y, labels, n_splits)
    )
    if validation_trades.empty:
        print(f"  {timeframe}/{direction_label}/{tier_label}: not enough data for a single purged CV fold - skipping")
        return None
    print(f"  {timeframe}/{direction_label}/{tier_label}: hyperparameter search picked '{winning_name}' "
          f"({'ensemble of ' + str(len(winning_params)) + ' configs' if is_ensemble else winning_params}) - "
          + ", ".join(f"{c['name']}={c['validation_summary'].get('win_rate_wilson_lower')}" for c in search_report))

    # The deployed model uses ALL resolved history, not just one fold's
    # training slice - CV above only ESTIMATES how well this approach
    # generalizes; the promotion decision is what actually gates it.
    # Uses the WINNING config (or ensemble of configs) from the search
    # above, not a fixed one.
    resolved_mask = ~y.isna()
    print(f"  {timeframe}/{direction_label}/{tier_label}: fitting final deployed model on all "
          f"{int(resolved_mask.sum())} resolved rows ('{winning_name}')...", flush=True)
    final_fit_start = time.monotonic()
    if is_ensemble:
        members = [_fit_model(features.loc[resolved_mask], y.loc[resolved_mask], p) for p in winning_params]
        # Recover each member's HYPERPARAMETER_GRID name (for auditability
        # in the saved model's own meta, not needed for scoring) by
        # matching its params dict back to the grid - winning_params
        # preserves ensemble_members' rank order, not necessarily
        # HYPERPARAMETER_GRID's own order.
        member_names = [
            next((g["name"] for g in HYPERPARAMETER_GRID if g["params"] == p), "unknown")
            for p in winning_params
        ]
        final_model = EnsembleClassifier(members, member_names=member_names)
    else:
        final_model = _fit_model(features.loc[resolved_mask], y.loc[resolved_mask], winning_params)
    print(f"  {timeframe}/{direction_label}/{tier_label}: final fit done in "
          f"{time.monotonic() - final_fit_start:.0f}s", flush=True)

    # Global explainability - "what did THIS model learn overall," a
    # genuinely different question from live_signal.py's per-signal
    # SHAP breakdown. Computed ONCE per trained candidate (not per
    # hyperparameter-search trial above) on the SAME resolved rows the
    # final model was just fit on, so the report describes exactly the
    # model being saved, not some other slice of history. See
    # ml_system/explainability.py's own docstring for what each piece
    # means and why the surrogate rules are reported with an honest
    # fidelity score rather than presented as the real model's logic.
    explain_start = time.monotonic()
    shap_global = explainability.global_shap_importance(final_model, features.loc[resolved_mask])
    distilled_rules = explainability.distill_rules(final_model, features.loc[resolved_mask])
    print(f"  {timeframe}/{direction_label}/{tier_label}: explainability (global SHAP + rule distillation) "
          f"in {time.monotonic() - explain_start:.0f}s"
          + (f" - surrogate fidelity {distilled_rules.get('fidelity')}" if distilled_rules else ""), flush=True)

    # The ACTUAL columns this run's feature table has - not the fixed,
    # single-timeframe-only features.FEATURE_COLUMNS constant - since
    # cross-timeframe context (when coarser_candles_by_tf is given) adds
    # extra columns on top of it. live_signal.py reindexes to whatever
    # list is stored here at score time, so this has to be the ground
    # truth of what this specific model version was actually trained on.
    # `winning_params` is a single hyperparams dict for a lone-config
    # winner (old shape, flattened via ** for backward-compatible
    # meta.json readability) or a LIST of dicts (one per ensemble
    # member) for an ensemble winner - stored under "members" instead,
    # since ** can't flatten a list. Either way `is_ensemble` in the
    # same dict is what a reader checks to know which shape to expect.
    hyperparameters_meta = (
        {"name": winning_name, "is_ensemble": True, "members": winning_params}
        if is_ensemble else
        {"name": winning_name, "is_ensemble": False, **winning_params}
    )
    result = model_registry.evaluate_and_maybe_promote(
        registry_dir, symbol, timeframe, direction_label, final_model, list(features.columns), validation_trades,
        hyperparameters=hyperparameters_meta, hyperparameter_search=search_report,
        rr_ratio=rr_ratio, n_tiers_tested=n_tiers_tested,
        shap_global=shap_global, distilled_rules=distilled_rules,
    )
    status = "PROMOTED" if result["promoted"] else "not promoted"
    vs = result["validation_summary"]
    print(f"  {timeframe}/{direction_label}/{tier_label}: candidate {result['version_id']} -> {status} "
          f"(win_rate={vs.get('win_rate')}, resolved={vs.get('resolved')}) - {result['reason']}")
    return result


def train_all(symbol: str, candles_dir: Path, registry_dir: Path, n_splits: int = N_SPLITS_DEFAULT,
              use_cross_timeframe: bool = True, rr_grid: "tuple[float, ...]" = RR_GRID,
              synth_dir: "Path | str | None" = "synthesized_features") -> dict:
    """`candles_dir`: the CANDLES folder to read from - normally the
    rule-based system's `data/candles/`, SHARED read-only between both
    systems (see ml_system/README.md - there is no reason to run a
    second Dukascopy backfill just because a second system is reading
    the ticks). `registry_dir`: this system's own model storage,
    completely separate.

    Loads EVERY timeframe's candles UP FRONT (not one at a time) - same
    reasoning discover_patterns.rebuild_all() already documents for its
    own cross-timeframe confirmation check: a timeframe's coarser
    siblings need to be in memory before its own training run needs
    them. `use_cross_timeframe=False` reproduces the exact pre-cross-
    timeframe behavior (single-timeframe features only) for comparison
    or if a coarser sibling's data looks wrong somehow.

    `rr_grid`: every R:R tier trained and independently promoted per
    (timeframe, direction) - see RR_GRID's own docstring above for the
    default and the honest multiple-comparisons cost of a large grid.
    Returns summary[tf][direction_label][f"rr{tag}"] = train_one()'s
    result for that tier (was summary[tf][direction_label] = result
    before multi-tier search existed - any caller reading the old shape
    needs updating, same as any caller of compute_ml_signal's new
    list-of-signals return - see ml_system/README.md).

    `synth_dir`: where to look for scripts/synthesize_features.py's
    output (synth_dir/<symbol>_<tf>.json) - `None` disables loading
    synthesized features entirely (single-timeframe + cross-timeframe
    features only, the exact pre-synthesis behavior), the default path
    is read if present and silently produces zero extra columns if the
    script has never been run for a given timeframe (feature_synthesis.
    load_synthesized_features()'s own "empty list, not an error"
    convention) - so leaving this at its default is always safe whether
    or not synthesis has ever actually been run."""
    timeframe_files = sorted(candles_dir.glob(f"{symbol}_*.parquet"))
    if not timeframe_files:
        raise SystemExit(f"no candle files found in {candles_dir} - run build_history.py first "
                          f"(the ML challenger reads the SAME candle files the rule-based system does)")

    candles_by_tf: dict[str, pd.DataFrame] = {}
    for path in timeframe_files:
        tf = path.stem.replace(f"{symbol}_", "")
        candles_by_tf[tf] = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)

    summary = {}
    for tf, candles in candles_by_tf.items():
        coarser_by_tf = None
        if use_cross_timeframe:
            coarser_labels = [t for t in coarser_timeframes(tf) if t in candles_by_tf]
            if coarser_labels:
                coarser_by_tf = {t: (candles_by_tf[t], TIMEFRAME_MINUTES[t]) for t in coarser_labels}

        print(f"{tf}: training ML challenger over {len(candles)} candles "
              f"({candles['timestamp'].min()} -> {candles['timestamp'].max()}), "
              f"{len(rr_grid)} R:R tier(s): {list(rr_grid)}")

        synth_defs = load_synthesized_features(synth_dir, symbol, tf) if synth_dir else []
        if synth_defs:
            print(f"  {tf}: {len(synth_defs)} synthesized feature(s) loaded from {synth_dir} "
                  f"({[d['leaf_a'] + '_' + d['binary_op'] + '_' + d['leaf_b'] for d in synth_defs]})")

        # Computed ONCE per timeframe, reused across both directions and
        # every rr_grid tier - see _compute_feature_table()/
        # _label_dataset() docstrings for why this is safe (features
        # don't depend on direction or rr_ratio; only labels do). Without
        # this, a 10-tier grid would recompute the same 89-feature pass
        # over the full candle history 20 times (2 directions x 10
        # tiers) for byte-identical output every time.
        features = _compute_feature_table(candles, coarser_by_tf, synth_defs)

        # A feature that's NaN across this timeframe's ENTIRE history (not
        # just an early warmup window) - typically an order-flow feature
        # (ask_bid_volume_imbalance, tick_count_ratio - see features.py's
        # own comment) whose source columns (ask_volume/bid_volume/
        # tick_count) were never backfilled into THIS timeframe's candle
        # file (scripts/backfill_order_flow.py) - is fundamentally
        # different from the "not enough history yet in this fold"
        # case _pooled_validation_trades()'s per-fold skip already
        # handles: skipping every fold forever because of ONE dead column
        # would waste every OTHER genuinely-informative feature for the
        # WHOLE timeframe, every tier, every direction - verified against
        # a real run where exactly this happened (every single CV fold
        # for every rr_grid tier on 15min skipped, because
        # ask_bid_volume_imbalance/tick_count_ratio are NaN in literally
        # every row of that timeframe's candle file). Dropping it here,
        # once, up front - instead of relying on the per-fold skip to
        # paper over it thousands of times - is the correct fix for a
        # column that will NEVER have real values for this timeframe,
        # while the per-fold skip in _pooled_validation_trades() stays in
        # place for the genuinely-different case of a feature with real
        # values LATER in history, just not yet in an early fold.
        dead_cols = features.columns[features.isna().all(axis=0)]
        if len(dead_cols) > 0:
            print(f"  {tf}: dropping {len(dead_cols)} feature(s) entirely missing across this timeframe's "
                  f"WHOLE history (likely a data source never backfilled for it, e.g. order flow) - "
                  f"{list(dead_cols)}")
            features = features.drop(columns=dead_cols)

        atr_series = _atr(candles)

        summary[tf] = {}
        for direction, direction_label in ((1, "bullish"), (-1, "bearish")):
            summary[tf][direction_label] = {}
            for rr_ratio in rr_grid:
                tier_key = f"rr{model_registry.rr_tag(rr_ratio)}"
                summary[tf][direction_label][tier_key] = train_one(
                    candles, symbol, tf, direction, registry_dir, n_splits,
                    coarser_candles_by_tf=coarser_by_tf, rr_ratio=rr_ratio, n_tiers_tested=len(rr_grid),
                    features=features, atr_series=atr_series,
                )

    # Regenerate the lib-view NOW, reflecting whatever is currently
    # active across every timeframe/direction/tier after this run - see
    # model_registry.write_lib_view for why this exists.
    model_registry.write_lib_view(registry_dir, symbol, registry_dir / "lib_view")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--candles-dir", default="data/candles",
                         help="where to read candle parquet files from - normally the rule-based "
                              "system's data/candles/, shared read-only between both systems")
    parser.add_argument("--data-dir", default="ml_data",
                         help="this system's OWN state: heartbeats, and (via ml_live_update.py) its journal")
    parser.add_argument("--registry-dir", default="ml_registry", help="this system's own model storage")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS_DEFAULT)
    parser.add_argument("--no-cross-timeframe", action="store_true",
                         help="disable cross-timeframe context features (single-timeframe features only, "
                              "the exact pre-cross-timeframe behavior)")
    parser.add_argument("--rr-grid", default=None,
                         help="comma-separated R:R tiers to search, e.g. '1.5,4,8' - overrides the default "
                              f"{len(RR_GRID)}-tier grid ({list(RR_GRID)}). Useful for a fast test run before "
                              "committing to the full grid's ~10x runtime.")
    parser.add_argument("--synth-dir", default="synthesized_features",
                         help="where to read scripts/synthesize_features.py's output from - safe to leave at "
                              "the default even if that script has never been run (zero extra columns)")
    parser.add_argument("--no-synthesized-features", action="store_true",
                         help="disable synthesized features entirely (the exact pre-synthesis behavior)")
    args = parser.parse_args()

    rr_grid = tuple(float(x) for x in args.rr_grid.split(",")) if args.rr_grid else RR_GRID

    data_dir = Path(args.data_dir)
    with track("ml_train", path=data_dir / "heartbeats.json"):
        train_all(args.symbol, Path(args.candles_dir), Path(args.registry_dir), args.n_splits,
                  use_cross_timeframe=not args.no_cross_timeframe, rr_grid=rr_grid,
                  synth_dir=None if args.no_synthesized_features else args.synth_dir)


if __name__ == "__main__":
    main()
