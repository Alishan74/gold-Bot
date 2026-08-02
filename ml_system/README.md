# ML Challenger

A second, independent signal-generation system for the same instrument
(XAUUSD), run alongside the rule-based system in `src/` as a
**challenger, not a replacement** - same candles, same fixed 1:4 R:R
trade structure, same hard 60% win-rate gate, same journal/self-
assessment/self-healing machinery, but signals come from a trained model
searching a much larger feature space instead of ~55 hand-picked
candlestick/indicator/support-resistance patterns. See the main `README.md`'s "ML-driven
pattern discovery" discussion for the full reasoning and the risks this
design deliberately guards against.

**The plan, in one sentence:** run both systems, track both with the
identical self-assessment tooling, and only consider merging anything
once the ML system has *independently earned* `CONFIRMED` credibility on
real live trades - never on a backtest claim alone.

## Why this is safe to run as a genuine second system, not a gamble

Three specific risks were called out before building this, and each has
a concrete, testable answer:

1. **Overfitting via multiple comparisons** (a tree model implicitly
   tests far more candidate splits than the 1046 hand-picked patterns
   ever did) → `validation.py` implements PURGED, EMBARGOED walk-forward
   cross-validation, not a plain train/test split. Verified: no
   train/validation index overlap, every training example's label window
   provably resolves before its validation block starts, folds expand
   chronologically. See `validation.py`'s docstring for exactly what
   "purged" and "embargoed" mean and why a naive split isn't enough here.
2. **Training-serving skew** (computing a feature slightly differently
   live than at training time is one of the most common real-world ML
   bugs) → `features.py` is the ONE feature-computation function, used
   by both `train.py` and `live_signal.py`. There is no second
   implementation to drift out of sync.
3. **A model quietly making things worse** → `model_registry.py`'s
   promotion gate reuses `risk_reward.summarize_trades()` - the EXACT
   function that gates the rule-based system's patterns - so a new model
   version only replaces the active one if it independently clears the
   same 60%/`MIN_RESOLVED_SAMPLES` bar AND doesn't score below whatever
   is currently deployed. Every trained version is kept on disk,
   promoted or not, so any promotion can be inspected or rolled back.

## Architecture

```
features.py       - causal feature engineering (44 features), shared by
                     training and live scoring - the ONLY place this logic
                     lives
labeling.py        - thin wrapper around risk_reward.simulate_trades(),
                     called on every candle instead of only pattern
                     occurrences - same trade definition as the rule-based
                     system, on purpose (see file docstring)
validation.py       - purged, embargoed walk-forward cross-validation
model_registry.py   - versioned model storage + the hard promotion gate
train.py            - orchestrates: features + labels + purged CV +
                     promotion, per (timeframe, direction)
live_signal.py       - live scoring, output shape identical to
                     signal_engine.Signal.to_dict() so the SAME journal/
                     dashboard/self-healing code works unmodified
ml_live_update.py    - the challenger's live_update.py: score today's
                     signal, log it, self-heal (auto-retrain) when
                     triggered
```

### What's genuinely shared with the rule-based system, and what isn't

**Shared (read-only):** candle data (`data/candles/`), the forward news
calendar, `risk_reward.py`'s trade simulation and hard-gate constants,
`signal_journal.py` (unmodified), `regime.py`, several `patterns.py`
helper functions. There's no reason to run a second Dukascopy backfill
just because a second system is reading the same ticks.

**NOT shared - genuinely separate state**, by default under their own
directories so the two systems never write into the same file:
`ml_data/` (this system's own journal + heartbeats, vs. the rule-based
system's `data/`), `ml_registry/` (versioned models, vs.
`pattern_library/`). This is what "run them on different devices" is
free to mean literally - only the read-only shared inputs need to be
synced between devices, `ml_data/`/`ml_registry/` are this system's own.

## How the model is trained and what "qualifies" means

For every timeframe and both directions (long/short) separately - a
market state might only have a real edge in one direction, same as an
ambiguous candlestick pattern:

1. `features.compute_features()` - causal features for every candle.
2. `labeling.label_all_candles()` - `risk_reward.simulate_trades()`
   itself, called on every candle: does the fixed 1:4 R:R trade from here
   win, lose, or stay unresolved.
3. `validation.purged_walk_forward_splits()` over the FULL candle-index
   range (not the resolved-only subset - the purge boundary has to
   reflect real candle-time distance).
4. **Adaptive hyperparameter search** (`train.HYPERPARAMETER_GRID`,
   `_select_hyperparameters()`): step 4 above isn't run once with one
   fixed, hand-picked config - it's run once per candidate config in a
   small, bounded grid of three qualitatively different
   `HistGradientBoostingClassifier` setups (`shallow_fast`,
   `medium_default` - this system's original fixed config -
   `deep_regularized`), each scored through the identical purged
   walk-forward CV. Whichever config earns the best
   `win_rate_wilson_lower` (the same conservative, sample-size-aware
   number every other qualification decision in this system already
   uses, not raw win rate) on its pooled out-of-fold trades is the one
   actually used for the deployed model below. This is what makes the
   system genuinely "self-teaching" rather than just "retrains on a
   schedule with the same knobs forever": it periodically re-asks
   whether its own learning process is still the right one for the
   data it's currently seeing, not just what that process concludes.
   Kept deliberately small (3 configs, not a sweep of 30): every extra
   config tried against the same validation folds is one more chance to
   pick a winner that got lucky on this data rather than one that's
   genuinely better (the multiple-comparisons / selection-bias problem)
   - three bounded, qualitatively different configs keeps that inflation
   small while still giving the system a real choice, and the
   independent promotion gate below (step 6) is the actual backstop
   against a spuriously-selected config reaching live signals, not the
   search itself.
5. `model_registry.evaluate_and_maybe_promote()` grades the WINNING
   config's pooled set with `risk_reward.summarize_trades()` - same
   60%/`MIN_RESOLVED_SAMPLES` gate, same out-of-sample check, as any
   rule-based pattern.
6. The model that's actually SAVED (and promoted, if it earns it) is
   retrained on ALL available resolved data, using the WINNING config -
   cross-validation only *estimates* how well an approach generalizes;
   the deployed model should still use everything available. The winning
   config, and every candidate config's own validation summary, is
   stored in that version's `meta.json` (`hyperparameters` /
   `hyperparameter_search`) - so any promoted (or rejected) version is
   auditable: which settings actually produced these numbers, and what
   else was tried and passed over, not just the numbers themselves.

**Verified, not just described:** on pure random-walk synthetic data,
0 of the trained candidates qualified (correctly - there's no real
edge). On data with a genuine, injected relationship (price reliably
rallying whenever RSI dropped below 25), the model found and promoted it
- 71-74% win rate on 2,700-3,900 pooled out-of-fold validation trades
across different test runs, while the direction with no real edge
correctly failed to qualify. A weaker-but-still-qualifying retrained
candidate was verified to NOT replace a stronger active model; a
genuinely stronger one was verified to correctly replace it. The
hyperparameter search itself was verified end-to-end through a real
`train_all()` run: all three configs' CV results were genuinely
different, the search correctly picked the highest-Wilson-lower-bound
config each time, and the resulting `meta.json` correctly persisted both
the winning hyperparameters and all three candidates' own summaries for
audit.

## Self-healing: how this system "learns from live data, not just history"

Two mechanisms, both automatic, both reusing the rule-based system's
already-verified self-healing infrastructure (`signal_journal.py`)
rather than a parallel implementation:

- **Suspension**: a `(pattern, timeframe, direction)` - always
  `"ml_model"` as the pattern name, since there's one model per
  timeframe/direction rather than many named patterns - currently
  `DECAYING` live (Wilson-upper-bound win rate below what the active
  model's own validation claimed) is excluded from producing new
  signals, same mechanism `signal_engine.py` uses for the rule-based
  system.
- **Auto-retrain**: `ml_live_update.py` checks, every run, whether
  enough live drift has accumulated (>= 3 decaying combos) or the
  registry has gone stale (>= 1 day since the last retrain) and
  triggers `train.py` automatically if so - identical trigger logic to
  `signal_journal.should_self_heal()` (the 1-day backstop is sized for
  continuous operation via `scripts/run_continuous.py`, not the
  original once-an-hour cron assumption - see the top-level README's
  "Self-healing" section).

**Where "learning from live data" actually happens, and why it's not
"feed the journal back into training":** every retrain re-labels EVERY
candle in the shared candle store, including whatever's been appended
since the last retrain - so newly observed real market behavior becomes
new training examples automatically. The signal journal itself
(what this system predicted and what happened) is deliberately NOT fed
back into training directly - it's a SELECTED subset (only candles the
model chose to score highly), and training on your own past selections
would bias the model toward reinforcing what it already believed instead
of learning from the full, honest distribution of outcomes. The
journal's job is what it already does for the rule-based system: self-
assessment and self-healing suspension, not a second training signal.

Verified end-to-end: a seeded journal with 3 decaying
pattern/timeframe/direction combos correctly triggered an automatic
retrain mid-`ml_live_update.py` run; `--no-self-heal` correctly
suppressed the retrain while suspension kept working independently.

## Circuit breaker: this system's own portfolio-level hard stop

`ml_live_update.py` checks `circuit_breaker.check_circuit_breaker()`
against THIS system's own journal every run, independently of the
rule-based system's breaker (each has its own journal, breakered
separately - a tripped rule-based breaker doesn't halt this one, and vice
versa). Same three realized-outcome triggers, same shared
`src/circuit_breaker.py` (not a parallel implementation): >= 8
consecutive losses, peak-to-trough drawdown worse than -15R, or >= 6R of
simultaneous open risk. When tripped, `compute_ml_signal()` forces a hard
`HOLD` regardless of what any active model predicts - same override
precedence as suspension. See the top-level README's "Circuit breaker"
section for the full rationale and verification detail (including the
real end-to-end run of `ml_live_update.py`'s actual CLI, and the
dashboard's `/api/signal` endpoint in `ml` mode, against a seeded
8-loss-streak journal and an always-qualifying dummy model - confirmed
forced `HOLD` both times, with a healthy journal correctly letting a real
`BUY` through).

## Loss/win attribution: this system's own live confidence discount

`compute_ml_signal()` accepts the same `journal` parameter and applies
the identical `signal_journal.context_penalty()` mechanism the
rule-based system uses - see the top-level README's "Loss/win
attribution" section for the full methodology and verification detail.
"Confluence" for this system means how many DIFFERENT TIMEFRAMES'
models independently agreed on the final direction - `PATTERN_NAME` is
always `"ml_model"`, so there's nothing else to be confluent WITH on a
single timeframe (one model per direction each). Verified end-to-end
against a real `compute_ml_signal()` call the same way as the rule-based
system: a healthy journal produces no penalty, a journal proving
confluence=1 underperforms confluence=3 for `ml_model/1h/BUY` correctly
discounts confidence with a transparent reason, and omitting the journal
argument reproduces the exact pre-feature behavior.

## Support/resistance

`features.py` adds 5 continuous features - `dist_to_swing_high_atr`,
`dist_to_swing_low_atr`, `dist_to_round_number_atr`,
`dist_to_pivot_r1_atr`, `dist_to_pivot_s1_atr` - ATR-normalized signed
distance from the current close to each of the three level types
`support_resistance.py` defines for the rule-based system (swing-point
fractals, the $50 round-number grid, daily R1/S1 pivots). These reuse
`support_resistance.py`'s own `swing_levels()`/`nearest_round_number()`/
`daily_pivots()` functions directly, not a second implementation - the
same "one function, no training-serving skew" discipline every other
feature in this file already follows. Continuous rather than boolean on
purpose: a tree model can learn its own nonlinear threshold on "how
close is close enough" per level type, rather than being handed a single
hand-picked cutoff the way the rule-based system's rejection/bounce
patterns necessarily are.

## Setup and usage

```bash
pip install -r requirements.txt   # now includes scikit-learn, joblib

# candles must already exist - this reads the rule-based system's
# data/candles/, it doesn't fetch its own
python src/build_history.py --years 20 --workers 8   # if not already run

# initial training - one model per (timeframe, direction)
python ml_system/train.py --symbol XAUUSD \
    --candles-dir data/candles --data-dir ml_data --registry-dir ml_registry

# get today's signal, log it, self-heal if triggered
python ml_system/ml_live_update.py --symbol XAUUSD \
    --candles-dir data/candles --news-data-dir data \
    --data-dir ml_data --registry-dir ml_registry
```

For genuinely continuous operation (an always-on process that keeps
re-checking the live market and firing signals on its own, not just
"live when you happen to run it manually") - use
`scripts/run_continuous.py`, the same tool the rule-based system uses,
just pointed at this engine:

```bash
python scripts/run_continuous.py --engine ml
```

Run this as its OWN long-running process, separate from the rule-based
system's `--engine rules` instance - see the top-level README's "After
that: keep it fed" section for the full detail (stall/crash handling,
what "how fast can it see new data" actually means for the default
Dukascopy source vs. your own broker feed via `mt_bridge/`, and how it
was verified). If you'd rather use OS-level cron instead of a
long-running process, `scripts/supervise.py` still works the same way
it does for the rule-based system, just checking in hourly instead of
continuously:

```bash
python scripts/supervise.py --stall-timeout 300 --max-restarts 3 -- \
    python ml_system/ml_live_update.py --symbol XAUUSD \
    --candles-dir data/candles --news-data-dir data \
    --data-dir ml_data --registry-dir ml_registry
```

Self-assessment for this system works exactly like the rule-based one:

```bash
python scripts/report.py --data-dir ml_data --lib-dir ml_registry/lib_view
```

### Dashboard

The SAME `dashboard/server.py` serves either system - picked by env var,
so you can run one instance per system (same machine, different ports,
or genuinely different devices):

```bash
# rule-based (default)
python dashboard/server.py

# ML challenger
SIGNAL_ENGINE=ml \
DASHBOARD_DATA_DIR=ml_data \
DASHBOARD_SHARED_DATA_DIR=data \
ML_REGISTRY_DIR=ml_registry \
python dashboard/server.py
```

Everything - the signal hero, journal, and especially the Self-
Assessment panel (scorecard, equity curve, credibility table) - renders
identically for both, because both produce the exact same output shapes.
The "Patterns Data" table shows a single `ml_model` row per timeframe
(reading `ml_registry/lib_view/`, regenerated after every `train.py`
run) rather than the 1046-row breakdown the rule-based system shows -
there's one model per timeframe/direction here, not many named patterns.

## Performance notes (measured, not guessed)

- Feature computation: 44 features over 2,000 candles is sub-second.
- Labeling (`simulate_trades()` on every candle): ~1.5s per 50,000
  candles per direction in testing - extrapolates to a few minutes for a
  full 20-year 1min-timeframe history. This is a periodic training job,
  not a live-critical path, so that's an acceptable cost.
- Full `train.py` run (5 purged folds x 2 directions, features + labels
  + fold fitting + final model): ~11s for 8,000-12,000 candles in
  testing. Scale accordingly for the full history per timeframe;
  slower/higher timeframes have far fewer candles and train much faster.
- Live scoring (`live_signal.py`): scores the last closed candle only,
  same tail-based approach `signal_engine.py` uses - not a live-latency
  concern.

## What's still open / honestly not done

- The model type is a single `HistGradientBoostingClassifier` per
  timeframe/direction, not an ensemble - every retrain now picks its
  hyperparameters from a small, bounded search (see "How the model is
  trained" above), but the model FAMILY itself is still fixed; no
  gradient-boosting-vs-other-model-class comparison is attempted.
- No calibration layer (e.g. isotonic regression) on top of the raw tree
  probabilities - `PREDICTION_THRESHOLD=0.5` is used as the model's own
  decision boundary, which the purged-CV-based promotion gate then grades
  honestly regardless of whether the raw probability itself is
  perfectly calibrated.
- Feature set is intentionally broad-but-not-exhaustive (44 features:
  returns, moving-average distances, RSI/MACD/ADX, Donchian position,
  volatility regime, session/time-of-day, support/resistance distance -
  see "Support/resistance" below) - it does not yet include cross-asset
  features (DXY, real yields) the main README flags as a separate,
  not-yet-built idea.
- No automated comparison report between the two systems' live results
  yet - both are fully self-assessed independently (same tooling,
  directly comparable numbers), but "which one is actually better right
  now" is still a manual read of two dashboards/reports, not a single
  merged view. Worth building once there's enough live history on both
  to make that comparison meaningful.
