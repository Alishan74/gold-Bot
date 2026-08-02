"""
Global model explainability - "what did this model learn OVERALL," a
genuinely different question from live_signal.py's per-signal
_shap_top_factors() ("what drove THIS specific call"). Two
complementary reports, both computed ONCE per trained model version
(train.py calls both right after fitting the final deployed model on
all resolved data, never per live signal):

1. Global SHAP importance: mean |SHAP value| (and mean SIGNED SHAP
   value, for directionality) per feature, across the rows the final
   model was actually fit on. Answers "which of this model's 80+
   features does it actually lean on, on average" - the aggregate
   companion to the per-signal breakdown.

2. Surrogate rule distillation: a shallow, human-readable decision tree
   fit to APPROXIMATE the real model's own predictions (not the true
   labels) - turns "the ensemble learned some opaque combination of
   features" into readable IF/THEN text, with an honestly-reported
   fidelity score so it's never mistaken for the real model's exact
   logic (see distill_rules()'s own docstring for why that distinction
   matters).

Neither of these is a NEW claim about the model's validity - both are
read-outs of a model that already independently cleared
risk_reward.summarize_trades()'s hard gate (or didn't - this runs either
way, since an inspectable non-promoted candidate is still useful to
understand). This module explains an already-graded model; it doesn't
re-grade it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

try:
    import shap as _shap
except ImportError:
    _shap = None

MAX_SHAP_SAMPLE = 20000  # see global_shap_importance()'s own docstring
DISTILL_MAX_DEPTH = 4
DISTILL_MIN_LEAF_FRACTION = 0.02


def _shap_values_for_model(model, X: pd.DataFrame) -> "np.ndarray | None":
    """SHAP values for ANY model this codebase can deploy - a single
    HistGradientBoostingClassifier (shap.TreeExplainer directly - the
    exact call verified empirically before shipping, see live_signal.
    py's _shap_top_factors docstring) OR an ensemble.EnsembleClassifier
    (train.py's blended top-K-hyperparameter-config candidate - see
    ensemble.py's own docstring for why that exists) - shap.TreeExplainer
    has no idea what an EnsembleClassifier even is, so this branches
    explicitly rather than letting that call fail.

    For an ensemble: computes each MEMBER's own SHAP values separately
    (TreeExplainer works fine on each individual HistGradientBoosting
    Classifier member) and averages them with equal weight - the same
    weighting EnsembleClassifier.predict_proba() itself uses. This is an
    HONEST APPROXIMATION, not an exact decomposition of the ensemble's
    blended probability (SHAP's additivity guarantee is exact per
    member, in that member's own log-odds space; averaging PROBABILITIES
    across members - what the ensemble actually does - isn't perfectly
    linear the way averaging log-odds would be) - stated plainly here
    rather than presented as more precise than it is, same standard
    distill_rules()'s fidelity score already holds itself to."""
    if _shap is None:
        return None
    members = getattr(model, "estimators", None)
    if members is None:
        return np.asarray(_shap.TreeExplainer(model).shap_values(X))
    per_member = [np.asarray(_shap.TreeExplainer(m).shap_values(X)) for m in members]
    return np.mean(per_member, axis=0)


def global_shap_importance(model, X: pd.DataFrame, seed: int = 0) -> "dict | None":
    """Mean |SHAP value| and mean SIGNED SHAP value per feature, across
    (a sample of, if large) the rows the FINAL deployed model was
    actually fit on. None if `shap` isn't installed - same optional-
    dependency, fail-open convention live_signal.py's per-signal version
    already established, not a second policy invented here.

    `MAX_SHAP_SAMPLE`: a cap for the pathological case (a 1min-timeframe
    model fit on hundreds of thousands of resolved rows), not a
    compute-avoidance shortcut in disguise - shap.TreeExplainer's cost
    is roughly linear in rows, and this is a GLOBAL AGGREGATE statistic
    (verified empirically before shipping: exact additivity to
    floating-point precision against HistGradientBoostingClassifier's
    own decision_function, same check live_signal.py's per-signal
    version already passed), so a random (not chronological-tail, which
    would bias toward one era) sample of up to 20,000 rows is already
    enough for a stable average - unlike a live per-signal call, no
    individual row here needs to be exact."""
    if _shap is None:
        return None
    if len(X) > MAX_SHAP_SAMPLE:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=MAX_SHAP_SAMPLE, replace=False)
        X_sample = X.iloc[idx]
    else:
        X_sample = X
    values = _shap_values_for_model(model, X_sample)
    if values is None:
        return None

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    order = np.argsort(-mean_abs)
    ranked = [
        {"feature": str(X_sample.columns[i]), "mean_abs_shap": round(float(mean_abs[i]), 5),
         "mean_signed_shap": round(float(mean_signed[i]), 5)}
        for i in order
    ]
    return {"n_rows_sampled": int(len(X_sample)), "n_rows_total": int(len(X)), "ranked_features": ranked}


def distill_rules(model, X: pd.DataFrame, seed: int = 0) -> dict:
    """Fits a SHALLOW, human-readable DecisionTreeClassifier to
    APPROXIMATE the real model's own predictions (`model.predict(X)` -
    NOT the true win/loss labels) on the same rows the real model was
    fit on. This is surrogate-model distillation: the real model
    (HistGradientBoostingClassifier, up to hundreds of boosted trees) is
    not remotely human-readable; a depth-4 single tree fit to mimic ITS
    OUTPUTS is - not because it predicts better (it's deliberately much
    weaker, and never used to score anything for real), but because it
    turns "the ensemble learned some opaque combination of 80+ features"
    into readable IF/THEN text a human can actually sanity-check against
    domain intuition ("does 'IF resistance_break_rate > 0.6 AND
    regime_markup' actually make sense" is a question a human can ask of
    THIS, never of the real ensemble directly).

    `fidelity`: how often the surrogate tree's own prediction agrees
    with the real model's prediction on these same rows - reported
    HONESTLY alongside the rules specifically so this is never mistaken
    for "the real model's exact logic." A low fidelity means the real
    model is doing something genuinely too complex for a shallow tree to
    approximate (real, deep, multi-feature interaction effects) - the
    rules are then a rougher sketch of a rough consensus, not a precise
    account, and this number says so plainly instead of hiding it behind
    confident-looking rule text.

    DecisionTreeClassifier (unlike HistGradientBoostingClassifier)
    doesn't accept NaN natively, so missing values are median-imputed
    per column purely for fitting this diagnostic surrogate - never for
    the real model, which keeps training/scoring NaN natively
    throughout."""
    y_proxy = model.predict(X)
    if len(np.unique(y_proxy)) < 2:
        return {"fidelity": None, "rules": None,
                "note": "the real model predicts only one class on this data - nothing for a surrogate to distinguish"}

    fill_values = X.median(numeric_only=True)
    X_filled = X.fillna(fill_values)
    min_leaf = max(5, int(len(X) * DISTILL_MIN_LEAF_FRACTION))
    surrogate = DecisionTreeClassifier(max_depth=DISTILL_MAX_DEPTH, min_samples_leaf=min_leaf, random_state=seed)
    surrogate.fit(X_filled, y_proxy)
    surrogate_pred = surrogate.predict(X_filled)
    fidelity = float((surrogate_pred == y_proxy).mean())
    rules_text = export_text(surrogate, feature_names=list(X.columns), max_depth=DISTILL_MAX_DEPTH)
    return {
        "fidelity": round(fidelity, 4), "max_depth": DISTILL_MAX_DEPTH,
        "min_samples_leaf": min_leaf, "n_rows": int(len(X)), "rules": rules_text,
    }
