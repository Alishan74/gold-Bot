"""
Live scoring for the ML challenger - deliberately mirrors
signal_engine.compute_signal()'s output contract (same dict shape,
same trade-plan math, same freshness labeling) so this system's signals
can be tracked with the EXACT SAME journal / self-assessment /
self-healing machinery already built for the rule-based system
(signal_journal.py, unmodified), not a parallel implementation. That
matters for two reasons: (1) it's what makes "run them separately, see
which one actually performs live, merge later" a fair comparison rather
than two differently-measured things, and (2) it means this challenger
gets self-healing suspension and credibility scoring for free.

Multi-tier R:R (see model_registry.py / train.py's RR_GRID): every
contributing model uses a STABLE pattern name PER TIER -
"ml_model_rr<tag>" per (timeframe, direction, rr_ratio) - NOT versioned
by which specific retrain produced it. This matches how the rule-based
system's own live journal already behaves across pattern-library
rebuilds (a live "hammer" trade keeps being graded against whatever the
CURRENT mined stats say, regardless of which rebuild produced them) -
live tracking should measure "how is the ML approach doing on this
timeframe/direction/tier right now," continuously, not fragment into a
new bucket every time train.py promotes a new model version. Each
promoted version's own backtested validation performance is still fully
visible in model_registry's meta.json, per version.

compute_ml_signal() returns a LIST of signals, one per R:R tier that has
at least one non-suspended, currently-active, qualifying model
contributing - a scalp-sized setup and a swing-sized setup off the SAME
underlying market state are genuinely different trades, so they are
never blended into one composite vote the way every model USED to be
blended together before multi-tier search existed. Callers (
ml_live_update.py, the dashboard) iterate the list and treat each entry
as its own independent signal, with its own trade plan, its own
suspension/drift tracking (via its own tier-specific pattern name), and
its own journal rows.

Live feature computation reuses features.compute_features() - the SAME
function train.py used to build the training table - so there is no
separate "live feature code path" that could silently drift from what
the model was actually trained on (training-serving skew).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import explainability  # noqa: E402
import model_registry  # noqa: E402
from feature_synthesis import apply_synthesized_features, load_synthesized_features  # noqa: E402
from features import coarser_timeframes, compute_cross_timeframe_features, compute_features  # noqa: E402

from build_history import TIMEFRAME_MINUTES  # noqa: E402
from news_calendar import label_signal_news_risk, load_upcoming  # noqa: E402
from signal_engine import TIMEFRAME_WEIGHTS, _freshness, _trade_plan  # noqa: E402

SHAP_TOP_N = 5  # how many of a model's own features to report per signal

# Kept for backward compatibility with anything importing the old
# fixed name directly (e.g. a saved dashboard bookmark/filter) - live
# scoring itself always uses _pattern_name(rr_ratio) below now, never
# this constant.
PATTERN_NAME = "ml_model"


def _shap_top_factors(model, X: pd.DataFrame, top_n: int = SHAP_TOP_N) -> "list[dict] | None":
    """Per-signal explainability: which of THIS model's own features
    actually drove THIS specific call, ranked by |SHAP value| (raw
    log-odds/decision-function space, not calibrated probability - fine
    for ranking which factors mattered most relative to each other,
    doesn't need probability calibration to do that). This is the
    "reverse engineering" readout for the ML challenger specifically -
    instead of a human-written narrative about why a signal fired, it's
    the model's own, exact, additive attribution.

    Verified empirically before shipping (not assumed): shap.
    TreeExplainer's per-row SHAP values sum EXACTLY to
    HistGradientBoostingClassifier's own decision_function output for
    that row (checked to floating-point precision, max abs diff
    ~1.8e-15, across rows with and without missing values), and it
    handles NaN inputs the same missing-value-binning way the model
    itself was trained on - so this can never silently disagree with
    what the model actually computed, the same "no second, possibly-
    drifting implementation" discipline every other shared calculation
    in this codebase already follows.

    Returns None (not an error, not an empty list - genuinely "not
    computed") if the `shap` package isn't installed, OR if `model` is
    an ensemble.EnsembleClassifier (train.py's blended top-K-config
    candidate) and shap isn't available - either way this reuses
    explainability._shap_values_for_model() rather than calling
    shap.TreeExplainer directly, so a live signal from an ENSEMBLE model
    gets the same honest per-member-averaged approximation the global
    report already uses, not a crash (TreeExplainer has no idea what an
    EnsembleClassifier is) and not a second, differently-behaved
    ensemble-handling implementation."""
    values = explainability._shap_values_for_model(model, X)
    if values is None:
        return None
    row = values[0]
    order = np.argsort(-np.abs(row))[:top_n]
    return [{"feature": str(X.columns[i]), "shap_value": round(float(row[i]), 5)} for i in order]


def _pattern_name(rr_ratio: float) -> str:
    """Stable, tier-specific pattern name - "ml_model_rr<tag>" - shared
    naming (via model_registry.rr_tag()) with train.py's tier logging
    and model_registry.write_lib_view()'s lib_view keys, so live
    scoring, training, and the mined-win-rate lookup used by
    signal_journal.py's self-healing/suspension machinery can never
    silently disagree on what a given tier is called. Using a distinct
    pattern name per tier is ALL that's needed for signal_journal.py
    (unmodified - already generic over the `pattern` string, see
    pattern_scorecard()'s `lib.get(pattern)`) to track, drift-detect,
    and suspend every tier completely independently."""
    return f"ml_model_rr{model_registry.rr_tag(rr_ratio)}"


def _score_timeframe(candles: pd.DataFrame, tf: str, registry_dir: Path, symbol: str,
                      suspended: set[tuple[str, str, str]],
                      candles_by_timeframe: "dict[str, pd.DataFrame] | None" = None,
                      synth_dir: "Path | str | None" = "synthesized_features") -> list[dict]:
    """Every (direction, R:R tier) with a currently-active, non-suspended
    model that scores >= PREDICTION_THRESHOLD on the latest closed
    candle - same "could contribute" list shape signal_engine's
    per-pattern loop builds, just sourced from models instead of a mined
    pattern library, and now spanning every tier on disk for this
    (timeframe, direction) instead of a single fixed-R:R model.

    `candles_by_timeframe`: the full set of timeframes' candles (already
    loaded by the caller, compute_ml_signal, for its own per-timeframe
    loop) - used to build the SAME cross-timeframe context train.py
    builds for training, via the SAME coarser_timeframes() timeframe
    selection, so live scoring can never silently drift onto a different
    definition of "cross-timeframe context" than what a model was
    actually trained on. A model trained WITHOUT cross-timeframe context
    (--no-cross-timeframe) simply has no ctx_* names in its stored
    feature_columns, so the reindex below harmlessly ignores whatever
    extra ctx_* columns get computed here - no special-casing needed."""
    contributions = []
    if len(candles) < 30:
        return contributions
    feats = compute_features(candles)
    synth_defs = load_synthesized_features(synth_dir, symbol, tf) if synth_dir else []
    if synth_defs:
        synth = apply_synthesized_features(feats, synth_defs)
        feats = pd.concat([feats, synth], axis=1)
    if candles_by_timeframe:
        coarser_labels = [t for t in coarser_timeframes(tf) if t in candles_by_timeframe]
        if coarser_labels:
            from build_history import TIMEFRAME_MINUTES as _TF_MIN
            coarser_by_tf = {t: (candles_by_timeframe[t], _TF_MIN[t]) for t in coarser_labels}
            cross_tf = compute_cross_timeframe_features(candles, coarser_by_tf)
            feats = pd.concat([feats, cross_tf], axis=1)
    latest_row = feats.iloc[[-1]]

    for direction_label in ("bullish", "bearish"):
        for rr_ratio in model_registry.list_rr_tiers(registry_dir, symbol, tf, direction_label):
            pattern_name = _pattern_name(rr_ratio)
            if (pattern_name, tf, direction_label) in suspended:
                continue
            active = model_registry.load_active(registry_dir, symbol, tf, direction_label, rr_ratio)
            if active is None:
                continue
            feature_columns = active["meta"]["feature_columns"]
            X = latest_row.reindex(columns=feature_columns)
            if X.isna().all(axis=1).iloc[0]:
                continue  # not enough live history yet for any feature to be known
            proba = float(active["model"].predict_proba(X)[:, 1][0])
            if proba < model_registry.PREDICTION_THRESHOLD:
                continue
            shap_top_factors = _shap_top_factors(active["model"], X)

            vs = active["meta"].get("validation_summary", {})
            resolved = vs.get("resolved", 0)
            wilson_lower = vs.get("win_rate_wilson_lower", 0.0) or 0.0
            weight = TIMEFRAME_WEIGHTS.get(tf, 1.0) * wilson_lower * proba

            contributions.append({
                # Same field names signal_engine.Contribution uses
                # (win_rate/win_rate_wilson_lower/samples/weight/
                # used_for_signal/median_candles_to_resolve) - the dashboard's
                # contributions table and any other code reading a
                # contribution dict is shared with the rule-based system and
                # reads these exact keys; a differently-named field here
                # would silently render as blank rather than error, which is
                # worse than getting it right up front. There's at most one
                # candidate per (timeframe, direction, tier) here (one model
                # each), so used_for_signal is always true - no
                # same-timeframe dedup step is needed the way the rule-based
                # system's many candlestick/combo patterns require.
                "timeframe": tf, "pattern": pattern_name, "direction": direction_label,
                "rr_ratio": rr_ratio,
                "win_rate": vs.get("win_rate"), "win_rate_wilson_lower": vs.get("win_rate_wilson_lower"),
                "samples": resolved, "weight": round(weight, 4), "used_for_signal": True,
                "median_candles_to_resolve": vs.get("median_candles_to_resolve"),
                # ML-specific extras, additive - harmless for any renderer
                # that only reads the standard fields above.
                "model_version": active["version_id"], "model_probability": round(proba, 4),
                "shap_top_factors": shap_top_factors,
            })
    return contributions


def compute_ml_signal(candles_by_timeframe: dict[str, pd.DataFrame], registry_dir: Path, symbol: str,
                       upcoming: pd.DataFrame | None = None, now: pd.Timestamp | None = None,
                       suspended: set[tuple[str, str, str]] | None = None,
                       circuit_breaker: dict | None = None,
                       journal: pd.DataFrame | None = None,
                       synth_dir: "Path | str | None" = "synthesized_features") -> list[dict]:
    """Returns a LIST of signal dicts, each the SAME shape
    signal_engine.Signal.to_dict() does (direction/confidence/trade_plan/
    freshness/news_risk/suspended_skipped/circuit_breaker/
    context_penalty/primary_pattern/primary_timeframe/contributions) -
    ONE PER R:R TIER that has at least one non-suspended, currently-
    active, qualifying model contributing, NOT one blended composite
    signal (see module docstring for why: a scalp-tier and a swing-tier
    setup off the same market state are genuinely different trades).
    Callers (ml_live_update.py, the dashboard) must iterate the list -
    see ml_system/README.md for the shape change from the pre-multi-tier
    single-dict return.

    Circuit-breaker-tripped and "no active model anywhere" both return a
    single-item list with one system-wide HOLD entry (`risk_reward:
    None` - there is no tier-specific trade to report in either case).

    `circuit_breaker`: same hard override signal_engine.compute_signal()
    honors - when tripped, forces HOLD regardless of what any model
    says. This system is breakered independently, against its OWN
    journal (see ml_live_update.py) - a tripped rule-based breaker does
    NOT automatically halt this system, and vice versa.

    `journal`: this system's own live signal_journal - same loss/win
    ATTRIBUTION mechanism signal_engine.compute_signal() uses
    (signal_journal.context_penalty()), against this system's own
    journal, applied independently within each tier's own aggregation."""
    now = now or pd.Timestamp.now(tz="UTC").tz_localize(None)
    suspended = suspended or set()
    suspended_skipped = [f"{p}/{tf}/{d}" for (p, tf, d) in suspended if p.startswith("ml_model")]

    if circuit_breaker and circuit_breaker.get("tripped"):
        return [_to_dict(direction="HOLD", confidence=0.0, contributions=[], trade_plan=None,
                          freshness=None, news_risk=None, primary_pattern=None, primary_timeframe=None,
                          suspended_skipped=suspended_skipped, circuit_breaker=circuit_breaker,
                          risk_reward=None)]

    all_contributions = []
    for tf, candles in candles_by_timeframe.items():
        all_contributions.extend(
            _score_timeframe(candles, tf, registry_dir, symbol, suspended, candles_by_timeframe, synth_dir)
        )

    if not all_contributions:
        return [_to_dict(direction="HOLD", confidence=0.0, contributions=[], trade_plan=None,
                          freshness=None, news_risk=None, primary_pattern=None, primary_timeframe=None,
                          suspended_skipped=suspended_skipped, circuit_breaker=circuit_breaker,
                          risk_reward=None)]

    # ONE independent aggregation per R:R tier - see _tier_signal() below.
    # A tier with contributions in only one direction, or none at all
    # this run, simply doesn't appear (nothing to report for it, same as
    # today's single-tier "no active model" HOLD case, just scoped).
    tiers = sorted({c["rr_ratio"] for c in all_contributions})
    return [
        _tier_signal(
            rr_ratio, [c for c in all_contributions if c["rr_ratio"] == rr_ratio],
            candles_by_timeframe, upcoming, now, suspended_skipped, circuit_breaker, journal,
        )
        for rr_ratio in tiers
    ]


def _tier_signal(rr_ratio: float, contributions: list[dict], candles_by_timeframe: dict[str, pd.DataFrame],
                  upcoming: pd.DataFrame | None, now: pd.Timestamp, suspended_skipped: list[str],
                  circuit_breaker: dict | None, journal: pd.DataFrame | None) -> dict:
    """The SAME weighted-vote aggregation this module used to run ONCE
    across every active model at once, before multi-tier search existed
    - now run ONCE PER TIER, scoped to that tier's own contributions
    only, by compute_ml_signal() above. Same dedup principle
    signal_engine.py uses: only the strongest contribution per
    (timeframe, direction) counts toward this tier's composite vote -
    here that's moot per-timeframe (at most one bullish + one bearish
    contribution can exist per timeframe within a single tier, one model
    each), but keeping the same structure keeps this genuinely
    comparable to the rule-based system's aggregation logic."""
    bull = [c for c in contributions if c["direction"] == "bullish"]
    bear = [c for c in contributions if c["direction"] == "bearish"]
    bull_weight = sum(c["weight"] for c in bull)
    bear_weight = sum(c["weight"] for c in bear)
    total = bull_weight + bear_weight

    if total == 0 or bull_weight == bear_weight:
        direction, agreement, top = "HOLD", 0.0, None
    elif bull_weight > bear_weight:
        direction = "BUY"
        agreement = (bull_weight - bear_weight) / total
        top = max(bull, key=lambda c: c["weight"])
    else:
        direction = "SELL"
        agreement = (bear_weight - bull_weight) / total
        top = max(bear, key=lambda c: c["weight"])

    confidence = 100 * (top["win_rate_wilson_lower"] or 0.0) * agreement if top else 0.0

    # Loss/win ATTRIBUTION (signal_journal.context_penalty) - identical
    # mechanism and reasoning to signal_engine.compute_signal(), applied
    # against THIS tier's own contributions/confluence only.
    context_penalty_info = None
    if top is not None and journal is not None:
        from signal_journal import context_penalty, session_label_at

        same_direction = "bullish" if direction == "BUY" else "bearish"
        confluence_count = sum(1 for c in contributions if c["direction"] == same_direction)
        top_candles = candles_by_timeframe[top["timeframe"]]
        entry_ts = pd.Timestamp(top_candles["timestamp"].iloc[-1])
        multiplier, reasons = context_penalty(
            journal, top["pattern"], top["timeframe"], direction,
            confluence_count, session_label_at(entry_ts),
        )
        context_penalty_info = {"multiplier": multiplier, "reasons": reasons}
        confidence *= multiplier

    trade_plan, freshness, news_risk = None, None, None
    if direction != "HOLD":
        top_candles = candles_by_timeframe[top["timeframe"]]
        signal_direction = "bullish" if direction == "BUY" else "bearish"
        trade_plan = _trade_plan(top_candles, signal_direction, rr_ratio=rr_ratio)
        freshness = _freshness(top_candles, top["timeframe"], now)
        news_risk = label_signal_news_risk(
            now, top["median_candles_to_resolve"], TIMEFRAME_MINUTES.get(top["timeframe"], 60), upcoming,
        )

    contributions_sorted = sorted(contributions, key=lambda c: c["weight"], reverse=True)
    return _to_dict(
        direction=direction, confidence=confidence, contributions=contributions_sorted,
        trade_plan=trade_plan, freshness=freshness, news_risk=news_risk,
        primary_pattern=top["pattern"] if top else None,
        primary_timeframe=top["timeframe"] if top else None,
        suspended_skipped=suspended_skipped, circuit_breaker=circuit_breaker,
        context_penalty=context_penalty_info, risk_reward=f"1:{rr_ratio:g}",
    )


def _to_dict(direction, confidence, contributions, trade_plan, freshness, news_risk,
             primary_pattern, primary_timeframe, suspended_skipped, circuit_breaker=None,
             context_penalty=None, risk_reward=None) -> dict:
    return {
        "direction": direction,
        "confidence": round(confidence, 1),
        "risk_reward": risk_reward,
        "trade_plan": trade_plan,
        "freshness": freshness,
        "news_risk": news_risk,
        "suspended_skipped": suspended_skipped,
        "circuit_breaker": circuit_breaker,
        "context_penalty": context_penalty,
        "primary_pattern": primary_pattern,
        "primary_timeframe": primary_timeframe,
        "contributions": contributions,
    }


def load_suspended_ml(data_dir: Path, registry_dir: Path, symbol: str) -> set[tuple[str, str, str]]:
    """Same self-healing suspension mechanism signal_engine.load_suspended()
    uses, against THIS system's own journal (data_dir) and its own
    "mined" reference (registry_dir/lib_view - see
    model_registry.write_lib_view) instead of the rule-based system's
    pattern_library/. Failure to load means an empty suspension set, not
    a crash - same fail-open policy as the rule-based system's version."""
    try:
        import signal_journal
        journal = signal_journal.load_journal(data_dir)
        lib_view_dir = Path(registry_dir) / "lib_view"
        raw = signal_journal.suspended_patterns(journal, lib_view_dir, symbol)
    except Exception:
        return set()
    # Journal rows (and therefore suspended_patterns' keys) use BUY/SELL -
    # this module's own vocabulary is bullish/bearish (matching
    # model_registry's directory names) - same translation
    # signal_engine.load_suspended() does for the rule-based system.
    dir_map = {"BUY": "bullish", "SELL": "bearish"}
    return {(pattern, tf, dir_map.get(direction, direction)) for (pattern, tf, direction) in raw}
