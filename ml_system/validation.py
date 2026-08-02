"""
Purged, embargoed walk-forward cross-validation - the single most
important safety mechanism in this ML challenger, and the thing a naive
"just train a model on historical data" approach gets wrong in a way
that's easy to miss and produces numbers that look great and mean
nothing.

The problem: every label here (labeling.py) depends on up to
MAX_LOOKAHEAD+1 FUTURE candles - "did the 1:4 trade opened at candle
i+1 hit its target before its stop, checking forward." That means a
training example near a train/validation boundary can have its
"known outcome" determined by price action that happens DURING the
validation period. A model trained on it has effectively been shown a
sliver of the future it's about to be "tested" on - the validation
score comes back looking good for a reason that has nothing to do with
the model generalizing. A plain chronological train/test split (which
the rule-based system's OOS check uses, and which is a reasonable choice
FOR THAT PURPOSE - see risk_reward.py) does not fully protect against
this the way it needs to for a model with many more degrees of freedom
to exploit that leak.

Two mechanisms fix it, applied together on every fold:
  1. PURGING: any training example whose label window
     [i, i + label_window) overlaps the validation block is dropped from
     training - if resolving its outcome required looking at candles
     inside (or past) the validation start, it doesn't get to be a
     training example for that fold.
  2. EMBARGO: an additional safety buffer (in candles) beyond the
     minimum purge requirement, for extra margin against near-boundary
     autocorrelation in the features themselves (not just the labels).

Multiple SEQUENTIAL, EXPANDING folds (not k-fold / not shuffled) - fold
k's training set is everything from the start of history up to its own
purge cutoff, fold k's validation block comes strictly after. This
mirrors the walk-forward spirit of risk_reward.py's OOS check but with
several folds instead of one, which is what actually tells you whether
a model's edge is STABLE across different historical periods, not just
present in one particular holdout.
"""
from __future__ import annotations

import numpy as np


def purged_walk_forward_splits(n: int, label_window: int, n_splits: int = 5,
                                embargo: int = 0, min_train_size: int | None = None):
    """Yields (train_idx, val_idx) numpy integer-position arrays for
    `n_splits` sequential, expanding-window folds over `n` candles.

    `label_window`: candles a label at position i depends on
    (labeling.label_window()) - the purge requirement.
    `min_train_size`: candles reserved before the FIRST validation block
    even starts, so early folds aren't trained on almost nothing.
    Defaults to n // (n_splits + 1).

    A fold is skipped (not yielded) if purging would leave it with no
    training data at all, or if there's no validation data left - this
    can legitimately happen with a short history and a large
    label_window/embargo; callers should check they got at least one
    fold back, not assume n_splits folds always arrive.
    """
    if n <= 0 or n_splits <= 0:
        return
    if min_train_size is None:
        min_train_size = n // (n_splits + 1)

    block_size = (n - min_train_size) // n_splits
    if block_size <= 0:
        return

    for k in range(n_splits):
        val_start = min_train_size + k * block_size
        val_end = val_start + block_size if k < n_splits - 1 else n
        if val_start >= n or val_end <= val_start:
            continue
        val_idx = np.arange(val_start, val_end)

        purge_cutoff = val_start - label_window - embargo
        if purge_cutoff <= 0:
            continue
        train_idx = np.arange(0, purge_cutoff)

        yield train_idx, val_idx
