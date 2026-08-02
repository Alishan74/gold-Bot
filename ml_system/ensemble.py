"""
A small, explicitly-typed, picklable ensemble wrapper - averages
predict_proba across several already-fitted HistGradientBoostingClassifier
models with equal weight. Exists so train.py's adaptive hyperparameter
search (HYPERPARAMETER_GRID) can compare "the single best config" against
"an equal-weight blend of the top few configs" on the SAME honest,
purged walk-forward out-of-fold criterion (Wilson lower bound) it
already uses for everything else, and deploy whichever one actually
earns better out-of-fold performance - never an assumption that blending
automatically helps (see train.py's _select_hyperparameters_or_ensemble
docstring for the full comparison logic and why it's a genuine, not
foregone, comparison).

Deliberately NOT a sklearn VotingClassifier/StackingClassifier - those
fit their own members internally in ways that don't compose cleanly with
this codebase's OWN purged walk-forward CV loop (which needs to control
exactly which rows each member sees, per fold, for the SAME leakage
reasons validation.py's own docstring explains for a single model). This
wrapper only ever averages ALREADY-FITTED members' predict_proba
outputs - it never fits anything itself.

A plain top-level class (not a closure or local class inside a
function) purely so joblib can pickle it - model_registry.save_candidate
joblib.dump()s whatever train.py hands it, exactly like a single
HistGradientBoostingClassifier, with zero special-casing needed there.
"""
from __future__ import annotations

import numpy as np


class EnsembleClassifier:
    """Equal-weight average of `estimators`' own predict_proba - NOT a
    sklearn-compatible fit()/predict() estimator (it is never fit
    itself, only ever constructed from already-fitted children), just
    enough of the interface (predict_proba, predict, classes_) for
    model_registry/live_signal.py to use it interchangeably with a
    single HistGradientBoostingClassifier everywhere they call
    .predict_proba(). `member_names`: the HYPERPARAMETER_GRID config
    name each member came from - stored purely for auditability (so a
    saved ensemble's meta.json can honestly say what it's made of, not
    just "an ensemble")."""

    def __init__(self, estimators: list, member_names: "list[str] | None" = None):
        if not estimators:
            raise ValueError("EnsembleClassifier needs at least one fitted estimator")
        self.estimators = estimators
        self.member_names = member_names or [f"member_{i}" for i in range(len(estimators))]
        self.classes_ = estimators[0].classes_

    def predict_proba(self, X):
        probas = np.stack([e.predict_proba(X) for e in self.estimators], axis=0)
        return probas.mean(axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
