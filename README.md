# Gold Signals Bot (XAUUSD only)

Single-instrument signals pipeline: mine gold candles for recurring
patterns, then compare live candles against that mined library to
produce BUY/SELL/HOLD signals. No coin/asset selection logic - gold
only, on purpose.

## Critical review: what was wrong, what's fixed, what's still open

A full trading-desk-level audit was run against this system before
trusting it with real signals. Findings, ranked by severity, and what
changed:

1. **Look-ahead-adjacent entry pricing (fixed).** Historical mining used
   to price entry at the SAME candle used to detect the pattern - you
   can't actually know a candle closed as, say, a bullish engulfing
   until it closes, and by then price has moved to the next candle's
   open. `risk_reward.simulate_trades()` now enters at the next candle's
   open; ATR-based risk distance is still measured at the signal candle
   (legitimately known then). Live signal generation was reviewed
   separately and left as-is deliberately - "last close" IS the best
   available real-time price estimate when there's no look-ahead
   problem, see the comment in `signal_engine._trade_plan()`.
2. **Zero transaction costs anywhere (fixed, partially).** Every
   simulated trade now also reports `expectancy_r_after_costs`, net of a
   round-trip spread cost (`risk_reward.SPREAD_USD`). This is a
   PLACEHOLDER, not a verified figure for any broker - calibrate it to
   your own account before trusting the after-cost numbers. Win/loss
   classification itself is unaffected (spread doesn't move the market),
   only realized expectancy.
3. **No out-of-sample validation (fixed).** Patterns used to be fit and
   graded on the exact same full history - a pattern that only worked in
   one regime could still pass. `summarize_trades()` now also computes
   stats on a held-out most-recent slice of each pattern's occurrences,
   and `qualifies` requires BOTH the full-sample gate AND that
   out-of-sample slice clearing 60% independently. Verified: a
   synthetic pattern that stopped working after a regime shift showed a
   full-sample win rate of ~58% but an out-of-sample win rate of ~11-15%
   - exactly the divergence this gate is built to catch.
4. **Misleading "confidence" metric (fixed).** It used to be "how much do
   contributing patterns agree in direction" - a single pattern barely
   over the 60% gate could show 100% confidence, which was verified to
   actually happen in testing. Confidence is now the Wilson-score
   lower-bound win rate of the strongest contributing pattern (a
   sample-size-aware, conservative estimate of the TRUE win rate, not
   the raw observed one), discounted by cross-timeframe disagreement.
   Verified: a 61%-win-rate/32-sample pattern now shows 42.8% confidence
   instead of 100%; a genuinely strong 82%-win-rate/280-sample pattern
   still shows a deservedly high 77%.
5. **Correlated patterns double-counted (fixed).** Two patterns firing on
   the same candle (e.g. `doji` and `inside_bar`, which aren't
   statistically independent) used to have their weights summed as if
   independent confirmations. Only the strongest pattern per timeframe
   now counts toward direction/confidence; the rest still show in
   `contributions` for transparency with `used_for_signal: false`.
6. **No drift detection (fixed).** The signal journal tracked live
   outcomes but never compared them against what was mined.
   `signal_journal.detect_drift()` now flags a pattern whose live win
   rate's 95% Wilson upper bound has fallen below its mined win rate -
   i.e. live performance that's very unlikely to just be noise - surfaced
   as a warning banner in the dashboard's Signal Journal panel.
7. **No position sizing (fixed).** A trade plan is just price levels
   without a way to turn "risk R" into an actual order size.
   `position_sizing.py` + the dashboard's calculator turn account size +
   risk % + your broker's contract spec into a concrete lot size -
   deliberately NOT auto-applied to every signal, since account size
   isn't something this system can know on its own.

**Empirically tested, not just theorized:** before assuming the biggest
theoretical risk was overfitting from testing hundreds of pattern/
timeframe/direction combinations (a real, general statistical concern),
it was checked directly - 2,880 pattern-tests run against pure
random-walk noise across all 6 timeframes produced **zero** spurious
qualifiers. The 60%-win-rate bar (3x the ~20% breakeven rate for a 1:4
R:R trade) turns out to already be a strong defense against random
noise on its own. That doesn't make multiple-testing correction
worthless as a general principle, but it meant time went to the fixes
above (which DID reproduce real failure modes in testing) instead of a
defensive statistical correction for a problem that isn't currently
manifesting.

**Still open (flagged, not fixed this pass) - be aware of these:**
- **Session-aware volatility for stops.** ATR(14) is timeframe-uniform;
  it doesn't widen/narrow based on which trading session is active
  (Asian session liquidity is thinner than the London/NY overlap).
  Modeling this properly needs tick-level intraday data to verify
  against, which wasn't available to test rigorously here - flagged
  rather than shipping a guessed fix.
- **Weekend/session gap risk.** Stops and targets are checked against
  candle high/low, which assumes continuous price action within a
  candle. A real gap (common around weekends or major news) can jump
  straight through a stop level without ever "touching" it in the
  backtest's terms - the backtest doesn't currently model slippage on
  gap-throughs.
- **Cross-timeframe correlation.** Same-timeframe correlated patterns are
  now deduped (#5 above); different timeframes on the same underlying
  price series are still summed as if independent, which they aren't
  fully (a 4h and 1d signal on overlapping data share information).
  Partially mitigated by `TIMEFRAME_WEIGHTS` favoring higher timeframes,
  not fully solved.
- **Single data vendor.** Everything rests on Dukascopy's feed with no
  cross-check against a second source. `dukascopy_fetch.verify_format()`
  catches an obviously wrong price range, not a subtly biased feed.

### Second pass: two more real bugs found on re-audit

A follow-up full re-read (not just re-explaining the fixes above, actually
re-checking the current code for new issues) found two more, both fixed
and verified:

8. **Conflicting-direction patterns on the same timeframe were silently
   discarded, not offset.** `signal_engine.py`'s de-duplication (fix #5
   above) was originally keyed by timeframe alone: if a timeframe had a
   strong bullish pattern AND a genuinely separate bearish pattern (real
   conflicting evidence, not correlation), only the higher-weight one
   counted - the other vanished instead of partially offsetting it in the
   weighted vote. Now keyed by (timeframe, direction): patterns
   correlated in the SAME direction still dedupe (that was the original,
   correct intent), but genuine disagreement on the same timeframe now
   shows up as reduced confidence, as it should. Verified: a scenario
   with a strong bearish 1h pattern and moderate bullish 1d+4h patterns
   went from "the bearish one silently vanishes, ~60% confidence" to
   "both count, 9-17% confidence" - a materially more honest number.
9. **The journal could log the wrong pattern for a signal.** With
   multiple timeframes contributing, the single highest-weight
   contribution isn't always in the WINNING direction (several moderate
   same-direction votes can outweigh one strong opposing one).
   `signal_journal.log_signal()` used to just take `contributions[0]` as
   "the" pattern - which, in that scenario, names a pattern that had
   nothing to do with producing the trade plan being logged (verified:
   logged `bullish_engulfing` / `direction: SELL`, a self-contradictory
   record). Fixed by having `compute_signal()` explicitly report which
   contribution produced the trade plan (`primary_pattern`/
   `primary_timeframe` on the signal), instead of `log_signal` guessing.
   Re-verified end-to-end through the real `signal_engine.py` entry
   point (not just a unit test): journal now always matches the actual
   trade plan.

### Third pass: hammer/hanging_man and shooting_star/inverted_hammer were never actually distinct patterns

A second follow-up re-read (final in-depth pass, again re-checking current
code rather than re-explaining prior fixes) found one more real bug, fixed
and verified:

10. **`hanging_man` and `inverted_hammer` were literal aliases of
    `hammer` and `shooting_star`** - `hanging_man(df)` just called
    `hammer(df)` and returned the identical boolean Series; same for
    `inverted_hammer`/`shooting_star`. A code comment claimed "direction
    is determined by prior trend, which the pattern-library builder
    accounts for via forward-return labeling" - but no such trend logic
    existed anywhere in `build_pattern_library.py` or `patterns.py`
    (verified by search: zero hits). In real candlestick theory, a
    hammer (bullish reversal) and a hanging man (bearish reversal) are
    the *same shape* but only meaningful in opposite trend contexts - a
    long lower wick after a decline suggests buyers stepped in; the same
    shape after a rally suggests sellers just showed up. Without that
    distinction, the two "different" patterns fired on the exact same
    candles and were mined with opposite hard-coded directions
    (`DIRECTION_HINT`: hammer=+1, hanging_man=-1) for no actual reason -
    confirmed empirically (`hammer == hanging_man` was `True` on every
    row of a synthetic dataset before the fix). This also meant they'd
    always both qualify or both fail together and, post-fix-#8, would
    always show up as "conflicting" on the same timeframe/candle even
    though they were never independent evidence to begin with. Fixed by
    adding a simple, look-ahead-free prior-trend check (`_prior_trend_up`/
    `_prior_trend_down`: is the close going into this candle above or
    below the average of the 10 closes before that) and gating each
    pattern on the trend context its real definition requires - hammer
    and inverted_hammer only fire after a prior downtrend, hanging_man
    and shooting_star only after a prior uptrend. Verified on the same
    synthetic dataset: the four patterns are now mutually exclusive
    (hammer and hanging_man never both fire on the same candle) with
    different, independent sample counts, and a full re-run of
    `build_pattern_library.py` + `signal_engine.py` end-to-end against
    synthetic data completed without error.

### Fourth pass: a real bug that only a real ingestion run could have caught

11. **`build_history.py`/`live_update.py` produced tz-AWARE candle
    timestamps; every other module in this codebase (`signal_engine.py`,
    `session_patterns.py`, `event_timing.py`, `signal_journal.py`,
    `news_calendar.py`) deliberately works in NAIVE-but-UTC-valued
    timestamps instead, each explicitly calling `.tz_localize(None)` on
    its own "now" for exactly that reason. `build_history.py`'s `end`/
    `start` (and the matching spot in `live_update.py` and
    `dukascopy_fetch.verify_format()`) never got that same treatment -
    they stayed tz-aware, which flows straight into every candle's
    `timestamp` column via `hour_utc + pd.to_timedelta(offset_ms, ...)`.
    The result: the moment a tz-aware candle timestamp met a naive one
    anywhere downstream (`now - trigger_ts` in freshness calculations,
    `pd.merge_asof` between candles and fundamentals/session boundaries,
    `last_candle_time()` vs `end` in `live_update.py`), pandas raises
    "can't compare/subtract offset-naive and offset-aware datetimes."
    This was NEVER caught in any earlier pass because every previous
    verification ran signal_engine.py/build_pattern_library.py against
    hand-built synthetic candle files (`pd.date_range(...)`, naive by
    default) - the actual `build_history.py`/`live_update.py` ingestion
    code was never executed even once before now, since it needs real
    Dukascopy network access this sandbox doesn't have. Caught this pass
    by mocking ONLY the network call (`fetch_hour_ticks`) and running
    `live_update.main()` and `build_history.main()` for real - that's
    what surfaced it. Fixed by adding `tzinfo=None` at all three
    `dt.datetime.now(dt.timezone.utc).replace(...)` call sites, so
    candle timestamps are naive-but-UTC everywhere, consistent with the
    rest of the codebase. Re-verified: both `build_history.main()` (mocked
    network, injected failures) and `live_update.main()` (mocked network)
    now run to completion end-to-end with no exception, produce all six
    timeframe files, and correctly write `data_quality_report.json`.

## Dashboard

A local monitoring UI (`dashboard/`) reads all of the state below directly
off disk - data storage, pipeline health/pulse, mined patterns, live
signal, fundamentals feed - with controls to trigger a refresh or rebuild
from the browser instead of a terminal:

```bash
pip install -r requirements.txt
python dashboard/server.py
# open http://localhost:8000
```

It's read-only against real files (nothing in it is mocked) and the
action buttons just run the same scripts documented below as background
jobs, streaming their output into an on-page console. See
`dashboard/README.md` for what each panel shows and where the numbers
come from.

## Hard gates (non-negotiable, enforced in code twice)

Every signal this system will ever emit is built on two rules that are
checked at mining time (`build_pattern_library.py`) AND independently
re-checked at signal time (`signal_engine.py`), so one buggy layer can't
silently let a bad signal through:

1. **Risk:reward is always exactly 1:4.** Stop-loss and take-profit are
   never picked freehand - they're computed from ATR (`risk_reward.py`):
   risk R = 1.5x ATR(14), target = entry + 4R, stop = entry - R (mirrored
   for shorts). There is no code path that produces any other ratio.
2. **Historical win rate must be >= 60% - IN-SAMPLE AND OUT-OF-SAMPLE.**
   For a pattern to be allowed to produce a live signal, simulating that
   exact 1:4 trade over its historical occurrences must show the target
   was hit before the stop at least 60% of the time, on >= 30 resolved
   trades, over the FULL history AND independently over a held-out
   most-recent slice of its occurrences (>= 10 resolved trades there -
   see "Critical review" above for why the out-of-sample half exists).
   Patterns that don't clear both are recorded (so you can see *why*
   they're excluded) but flagged `qualifies: false` and never contribute
   to a signal - no partial credit, no "weak signal" fallback.

Verified with synthetic data before shipping: on pure random-walk noise,
0 of the ~824 mined patterns (atomic + combo - see below) clear the gate
(correct - there's no real edge to find). With a genuine edge injected,
the relevant pattern/combo hit a 97-100% win rate and produced a signal
with an exact 1:4 trade. See `risk_reward.py` for the full simulation
methodology.

## Getting more signals without lowering the bar

More signals and more RELIABLE signals sound like they trade off against
each other, but under a hard, non-negotiable 60%-win-rate gate they
don't have to - the lever is mining a bigger, richer SPACE of candidate
patterns, not loosening what it takes for one to qualify.

**More base patterns** (`patterns.py`): on top of the original 21
candlestick patterns, this now also mines `donchian_breakout_up`/
`donchian_breakout_down` (close breaks a real N-candle high/low channel,
not just a moving-average cross), `atr_expansion` (a volatility-regime
flag, direction-neutral on its own - its value is mainly as a combo
component, see below), `bb_upper_breakout`/`bb_lower_breakout`
(Bollinger: mean+/-2std, a volatility-NORMALIZED breakout that reacts to
the DISTRIBUTION of recent closes, distinct from Donchian's literal
high/low extreme), `stoch_oversold_cross`/`stoch_overbought_cross`
(Stochastic %K vs 20/80 - a bounded high/low-RANGE momentum oscillator,
genuinely different inputs from RSI's close-only magnitude measure, not
a second name for the same signal), `adx_trending` (ADX(14) > 25,
reusing the SAME implementation `regime.py` already computes for its own
TRENDING/RANGING classification - direction-neutral, combo-only, same
role as `atr_expansion`), and `vwap_bullish_cross`/`vwap_bearish_cross`
(close crossing the session-anchored, day-resetting Volume-Weighted
Average Price - the institutional intraday benchmark, distinct in
character from every moving-average-based indicator above). Plus
`trend_following_long`/`trend_following_short` - see "Trend-following
strategy" below. All of them use full OHLC(V) - none are
open/close-only shape tests.

**Support/resistance patterns** (`support_resistance.py`) - "did this
candle react to a price level the market has independently paid
attention to before," mined through the exact same pipeline as
everything else, not assumed to matter just because it's a textbook
concept. Six patterns, from three look-ahead-safe level sources: `sr_
swing_high_rejection`/`sr_swing_low_bounce` (classic N-bar fractal swing
points - a level isn't usable until `SWING_LOOKBACK+1` candles after it
forms, an honestly-paid confirmation lag, not a shortcut - see the
module docstring), `sr_round_number_rejection`/`sr_round_number_bounce`
(the $50 psychological grid - $2000, $2050, $2100, ... - gold
specifically respects round numbers more than most instruments), and
`sr_pivot_r1_rejection`/`sr_pivot_s1_bounce` (classic floor-trader R1/S1
from the PRIOR completed calendar day's OHLC, held constant intraday).
Each fires only on a genuinely ONE-SIDED rejection shape, reusing
`hammer`/`shooting_star`'s exact TWO-sided wick/body geometry, not a
single check: the wick against the level must be large relative to the
body, AND the opposite wick must independently be small - without the
second half, a long-legged doji (a big wick on BOTH sides - real
indecision, not a clean rejection) would wrongly qualify on the first
check alone. Caught and fixed in a follow-up critical pass: the
"must-be-large" check originally used the same range-fallback the
"must-be-small" check needs, which made a body of exactly 0 (a textbook
gravestone/dragonfly doji) mechanically IMPOSSIBLE to qualify - backwards,
since a real wick with zero body is the cleanest possible rejection, not
one to exclude. After a genuine approach from the other side - proximity
to a level alone says nothing about direction (same reasoning
`atr_expansion` above is combo-only), so these are standalone-eligible
only because the shape itself, not just distance, supplies the
directional claim. The ML challenger gets the
continuous version of the same idea: `ml_system/features.py` adds
ATR-normalized signed distance to each of these three level types
(`dist_to_swing_high_atr`, `dist_to_round_number_atr`,
`dist_to_pivot_r1_atr`, ...), reusing `support_resistance.py`'s own
level functions rather than a separate implementation, so the model's
notion of "near a level" can never silently drift from the rule-based
system's.

**Trend-following strategy, 15min design** (`patterns.trend_following_long`/
`trend_following_short`) - a fully mechanical, zero-discretion setup, not
a hand-wavy "buy the trend": exactly 1 trend-direction indicator
(SMA(50) vs SMA(200) - price's own longer-run trend state, not a
momentary crossover event), 2 non-correlated confirmations (RSI(14) > 50
for price-momentum, and rising OBV over 20 candles for volume-
participation - see `patterns._obv`'s docstring for why THIS pairing,
not RSI+MACD, is genuine non-correlation and not two names for the same
signal), and 1 volatility filter (ATR above 1.1x its own 50-candle
trailing average - keeps it out of dead/choppy stretches). Fires once,
edge-triggered on the candle where all four conditions first align
together, not on every candle the trend persists. Stop-loss and target
are NOT a second, higher-timeframe decision that could repaint: both are
priced by the SAME `risk_reward.py` machinery every other pattern in this
system uses - `STOP_ATR_MULTIPLE` (1.5x) times ATR AT THE SIGNAL CANDLE
for the stop, `RR_RATIO` (4x that same risk) for the target - so there is
no multi-timeframe take-profit level anywhere in this setup to inflate
the backtest.

**Smart Money Concepts** (`smc_patterns.py`) - the SMC ideas that have an
unambiguous, mechanically checkable definition, each look-ahead-safe by
construction, mined through the identical pipeline as everything else.
Twelve patterns:
- `smc_liquidity_sweep_low`/`smc_liquidity_sweep_high` - price trades
  THROUGH a confirmed swing level (running the stops resting past it)
  and closes back on the origin side - stricter than
  `sr_swing_*_rejection`/`bounce` above, which only requires price to
  *approach* and hold, not actually break the level first.
- `smc_fvg_bullish`/`smc_fvg_bearish` - the classic three-candle Fair
  Value Gap: candle 1's high/low doesn't overlap candle 3's low/high,
  confirmed the instant candle 3 closes.
- `smc_bullish_sweep_fvg`/`smc_bearish_sweep_fvg` - the sweep candle
  itself is ALSO the gap-confirming third candle of an FVG - one
  deterministic, non-repainting "smart money" trigger, not "sweep now,
  wait and see if a gap forms later."
- `smc_bos_bullish`/`smc_bos_bearish` (Break of Structure) and
  `smc_choch_bullish`/`smc_choch_bearish` (Change of Character) - SMC's
  actual foundational concept: does the SEQUENCE of confirmed swing
  points show higher-highs-and-higher-lows (bullish structure) or
  lower-highs-and-lower-lows (bearish structure), and when price closes
  beyond the most recent confirmed swing point, is that break WITH the
  prevailing structure (BOS - continuation) or AGAINST it (CHoCH - the
  first mechanical sign of a reversal)?
- `smc_eq_high_sweep`/`smc_eq_low_sweep` - a stricter, higher-conviction
  liquidity sweep: the two most recently confirmed swing points on one
  side sit within a tight ATR-normalized tolerance of each other (an
  "equal highs/lows" resting-stop cluster, a bigger liquidity pool than a
  single swing point) and that pool gets swept and reclaimed.

Deliberately excluded, documented the same way `support_resistance.py`
documents its own excluded-list: order blocks and breaker blocks. Both
need a PERSISTENT, mutable zone (an origin candle's range, tracked
forward until a first retest "mitigates" it) rather than a single boolean
condition per candle - a genuinely different kind of object than every
other pattern in this codebase, and one that deserves its own honest
design and correctness review rather than a rushed approximation.

**A real bug caught building this out, worth knowing about if you write a
new edge-triggered pattern**: `signal & ~signal.shift(1).fillna(False)`
looks like standard "fire only on the first candle a condition holds"
logic, but it's silently broken - shifting a bool-dtype Series with no
explicit fill value introduces a NaN, which upcasts the WHOLE series to
`object` dtype, and `~` on an object array of Python bools does INTEGER
bitwise-NOT (`~False == -1`, `~True == -2` - both truthy) instead of
logical negation. The edge-trigger silently never filters anything. This
had been live in `trend_following_long`/`trend_following_short` since
they were written - verified the hard way while building `smc_patterns.py`
(a hand-built continuously-True test signal came back with every row
still `True` instead of only the first) and fixed everywhere it appeared,
using `signal.shift(1, fill_value=False)` instead, which keeps the
shifted result bool-dtype throughout so `~` means what it looks like it
means.

**Pattern confluence / combos** (`combo_patterns.py`) - the bigger lever.
Every valid CROSS-family pair of already-detected patterns (candlestick,
indicator, session, fundamental, support_resistance, and smc - every
pairwise combination across those 6 families - never two patterns from
the SAME family, which tend to be near-duplicates of each other) is mined
through the identical 1:4/60% pipeline, e.g. "`bullish_engulfing` AND
`session_london_open` fired on the same candle," or "`bearish_engulfing`
AND `sr_swing_high_rejection`" - a candlestick reversal shape firing
exactly at a level the market has reacted to before, the textbook case
this mechanism exists for.

Confluence like that is rarer but typically more selective than either
signal alone - real signal, not a weaker gate. ~1837 combo patterns get
mined per timeframe on top of the 65 atomic ones (76 when fundamentals
are loaded - CPI/PCE/NFP/GDP/FOMC add 11 more) - both numbers grew
meaningfully once `smc_patterns.py` and the new indicators above joined
the pool of families combos get built from, which is exactly the point:
more raw atomic material means more candidate combos get a shot at the
SAME unchanged 60% bar, not a looser one. Two safeguards specific to
combos, because testing thousands of combinations and keeping only the
ones that clear 60% is a much bigger multiple-comparisons search than
the atomic patterns ever were:
  1. Combos need MORE resolved samples before being trusted at all
     (`COMBO_MIN_RESOLVED_SAMPLES=40` / `COMBO_OOS_MIN_RESOLVED_SAMPLES=15`
     in `risk_reward.py`, vs 30/10 for atomic patterns).
  2. Every combo still has to clear the SAME out-of-sample split every
     atomic pattern does - a combo that only looks good from overfitting
     the full sample is far less likely to also pass on a held-out slice
     it never touched while being selected.
  This reduces false-positive risk, it doesn't eliminate it - a combo
  showing `qualifies: true` deserves the same skepticism as any
  statistically-mined signal. See `combo_patterns.py`'s docstring for
  the full reasoning.

Mining performance: `build_pattern_library.py` computes ATR ONCE per
timeframe and reuses it across all ~1913 patterns (atomic + combo)
instead of every single one recomputing the identical rolling
calculation - mining ~1913 patterns over 20,000 synthetic candles took
5.7s in testing (well within what a weekly/periodic rebuild can absorb
across 6 timeframes).

Verified end-to-end, not just at the mining layer: a synthetic dataset
with a genuine edge injected ONLY on the joint occurrence of two
patterns (not either alone) correctly produced a qualifying
`combo__doji__session_london_open` entry (99.7% win rate, both full-
sample and out-of-sample) while the same patterns tested individually,
and the combo's opposite direction, correctly did NOT qualify. Live
detection (`signal_engine.py`) picked the combo up as a contribution
exactly when it was active on the trigger candle, using the SAME
`combo_pairs()` list `build_pattern_library.py` mined against - the two
can never silently define combos differently, since both import from
the same `combo_patterns.py` module rather than each defining its own
pair list.

## Pattern Discovery Engine: self-learned patterns (`discover_patterns.py`)

Everything above - `patterns.py`, `support_resistance.py`,
`combo_patterns.py` - is a HAND-PICKED catalog: a human decided which
candlestick shapes, level types, and pairings were worth testing, and
the mining pipeline only ever checks whether those specific ideas hold
up statistically. The Pattern Discovery Engine is a different thing
entirely: instead of testing hand-picked ideas, it builds its OWN
candidate patterns directly from raw market primitives - momentum,
volatility, structure, level proximity, fundamentals, session - the
same way a systematic trader would explore a hypothesis space, not just
check a checklist. It writes its output to a SEPARATE directory
(`discovered_patterns/<symbol>_<tf>.json`, never merged on disk into
`pattern_library/`) so it can run independently of `build_pattern_
library.py` without either script racing to write the same file - the
two libraries are merged only in memory, at load time, by every
consumer (`signal_engine.py`, the dashboard, `signal_journal.py`'s
self-healing).

**Why this needs MORE statistical discipline than combos, not less.**
Unconstrained pattern search is exactly the failure mode this system
has spent its whole design avoiding: test enough candidate patterns
against a fixed bar (60% win rate, 1:4 R:R) and SOME of them will clear
it by pure chance alone, no matter how meaningless the underlying
condition is - the more candidates tested, the more false positives
survive a fixed threshold. A self-learning pattern miner tests far more
candidates than 991 hand-picked combos ever did, so it needed a
correspondingly stricter, not looser, set of defenses. Five layers,
each doing one job:

**Layer 0 - primitive SYNTHESIS (`discovery_synthesis.py`).** Even
Layer 1's primitive catalog below is still, at bottom, a human-chosen
set of TEMPLATES (RSI, ADX, slope, Donchian position, ...), even though
the search decides entirely on its own which instances and combinations
of those templates earn a place. Layer 0 removes that last human choice:
a small evolutionary search - random expressions from a grammar (a BASE
series: close/hl2/returns/range-%, run through a rolling TRANSFORM:
mean/std/slope/percentile-rank/z-score, over a WINDOW, compared against
a rolling PERCENTILE of its own trailing history rather than a raw fixed
number, so it stays meaningful as gold's price level drifts across a
20-year history) scored solo by the identical worst-era Wilson bound
Layer 2 uses, with each generation's survivors mutated into the next -
composes brand-new candidate primitives directly from raw OHLCV. Nobody
chose "rolling 20-bar std of returns compared against its own 80th
percentile" - the search found it, or didn't, on its own. Every
synthesized primitive's exact expression is fully serialized into the
accepted pattern's own `discovery_meta.synthesized_expressions` (so live
re-evaluation, months later, reconstructs it from that stored
definition rather than depending on the one-off in-memory closure that
found it) - never an opaque black box, the same "never hidden, never
just the model said so" standard everything else in this engine holds
itself to. Critically, every expression Layer 0 tries - survivor or not
- is folded into the SAME `n_tested` total Layer 3's FDR correction
scales against; skipping that accounting would let a primitive that
only cleared the bar by pure luck sneak through under an artificially
lenient threshold, silently reopening the exact hole this whole design
exists to close.

**Layer 1 - primitives (`discovery_primitives.py`).** 226 bounded,
named, look-ahead-safe building blocks across 8 families - momentum
(slope/acceleration/consecutive-highs-lows/RSI/MACD/ADX), volatility
(expansion/contraction), structure (Donchian position, SMA distance),
level_swing/level_round/level_pivot (reusing `support_resistance.py`'s
own level functions - never a second, possibly-drifted definition of
"near a level"), fundamental (CPI/PCE/NFP/GDP surprise z-scores, reusing
only PRIOR releases as the baseline so nothing look-ahead-leaks), and
session. Deliberately a bounded, curated library - not "read raw OHLC
and invent literally anything" (that's Layer 0's job) - so the search
space stays large enough to find real structure but small enough that
FDR correction (Layer 3) can meaningfully discipline it.

Every threshold/window here is a densER GRID than a human would
hand-pick one-by-one (RSI swept every 5 points from 55-85, not just "70,
maybe 80"; ADX every 5 points from 20-50; slope windows 5/8/10/15/20/30;
and so on) - the search and FDR correction decide which specific value
earns its place, not a human guessing a round number. Still deliberately
BOUNDED (a fixed grid, not a continuum) for the same multiple-
comparisons reason as everything else here. Every value that existed in
the ORIGINAL, sparser grids is still present - a pattern discovered and
saved to disk under an older grid stays valid, since discovered patterns
reference primitives by name and `is_discovered_pattern_active()` looks
them up by that exact name.

**Layer 2 - beam search construction (`discovery_search.py`).** Every
primitive is scored alone first; only primitives clearing a minimum
starting bar survive as seeds. From there, construction proceeds one
primitive at a time - the search keeps the top 5 (`BEAM_WIDTH`)
partial conjunctions at each depth, and an addition survives only if it
IMPROVES the worst-era score by a minimum margin, so every component in
a discovered pattern has proven independent value, not just ridden
along inside a big conjunction that happens to pass. Two hard
structural rules, mirroring `combo_patterns.py`'s own: cross-family
only (never two momentum primitives together - too likely to be near-
duplicates), and no direction contradictions. `MIN_DEPTH=2`: a single
primitive is NEVER emitted as a final pattern, no matter how well it
scores alone - the literal "not NFP alone, not two candles alone" rule
this was built around. Accepts `extra_primitives` (Layer 0's synthesized
ones) alongside the fixed catalog, competing and combining on identical
terms - each primitive's family and direction hint are resolved from a
per-conjunction lookup built for that run, not a module-global that only
knows about the hand-designed catalog, so a synthesized primitive is
never a second-class citizen inside the search.

**Layer 3 - validation (`discovery_validation.py` + `discover_patterns.
py`)** stacks FOUR independent defenses, not alternatives to each other:
  1. **Worst-era scoring**: every candidate is graded by the WORST of 4
     disjoint chronological eras' Wilson-lower-bound win rates, not one
     blended average - a pattern that only worked in one regime scores
     exactly as badly as its worst era, it can never hide behind an
     average across regimes that includes one lucky stretch.
  2. **Benjamini-Hochberg FDR correction**: the more candidates were
     actually tested this run - Layer 0's synthesis trials AND Layer 2's
     search trials, summed - the STRICTER the p-value bar a survivor has
     to clear to be accepted. A fixed significance threshold does not
     scale with search size; this does, and it's the reason Layer 0
     doesn't reopen the "test enough things and something clears a fixed
     bar by luck" hole this whole design exists to close.
  3. **Blind confirmation slice**: the newest 25% of history
     (`DISCOVERY_FRACTION=0.75`) is held out of the ENTIRE search
     process - not touched by era-scoring, not touched by FDR selection
     - so a survivor has to also independently clear the standard hard
     gate (`risk_reward.summarize_trades()`, including its own internal
     out-of-sample sub-split) on data nothing about its own construction
     could have adapted to, even indirectly through the search
     algorithm's behavior across many candidates.
  4. **Cross-timeframe confirmation** (`discover_patterns.
     _cross_timeframe_confirm`): every survivor of 1-3 gets its EXACT
     same primitive conjunction re-evaluated on the next-coarser
     timeframe's own candles (1min→5min→15min→1h→4h→1d), graded by the
     identical `summarize_trades()` hard gate. Deliberately
     INFORMATIONAL, not a 5th hard gate - many genuinely valid discovered
     patterns are honestly single-timeframe-scoped (a session-timing
     component, a fine-grained microstructure conjunction that only
     makes sense at 1min/5min resolution), so failing this check is real
     evidence, not proof of falseness. Stored under `discovery_meta.
     cross_timeframe` and surfaced as a soft LIVE weight discount in
     `signal_engine.py` (`CROSS_TIMEFRAME_MISMATCH_PENALTY = 0.5`) -
     the identical "soft discount, never a hard block" precedent
     `REGIME_MISMATCH_PENALTY` already established for regime mismatches
     - applied ONLY when the check actually ran and failed, never when
     no sibling timeframe was available to check at all.

**Layer 4 - orchestration (`discover_patterns.py`)** runs the full
pipeline per timeframe (Layer 0 synthesis → Layer 2 search → Layer 3
validation, including the cross-timeframe check above) and only writes
out patterns that survive defenses 1-3, THEN additionally computes a
full-history `summarize_trades()` result exactly like every atomic/combo
pattern gets - "the confirmation-slice-only check can pass while the
full-history blend still doesn't" is explicitly checked and rejected, so
a discovered pattern is held to the identical bar as a hand-picked one,
never a separate, looser standard. Every discovered pattern's name is a
stable hash of its primitive set + direction
(`discovered__<10-char-hex>`) - re-discovering the same conjunction on a
later run produces the SAME name, so `signal_journal.py`'s live
drift-tracking for it persists across re-runs instead of fragmenting
into a new bucket every time. Full provenance (which primitives, per-
era scores, the FDR p-value/threshold, the confirmation-slice result,
the cross-timeframe result, and - for any synthesized component - its
complete reconstructable expression) is stored under `discovery_meta` on
every accepted pattern - never hidden, always auditable.
`rebuild_all()` loads every timeframe's candles UP FRONT (not one at a
time) specifically so the cross-timeframe check always has its sibling
timeframe's data on hand.

**Live detection and self-healing integration.** `signal_engine.
load_inputs()` merges `discovered_patterns/<symbol>_<tf>.json` into the
in-memory pattern library alongside `pattern_library/`; live detection
(`_active_patterns_on_last_candle`) re-evaluates each discovered
pattern's OWN component primitives via `discover_patterns.
_evaluate_any_primitive()` - hand-designed primitives through
`discovery_primitives.evaluate_primitive()`, synthesized ones
reconstructed from the pattern's own stored
`discovery_meta.synthesized_expressions` (a synthesized primitive's
original in-memory closure doesn't survive past the discovery run that
created it, so this reconstruction is the ONLY way live detection can
know what it means) - the IDENTICAL evaluation path the discovery run
itself scored everything with either way, so live detection can never
silently drift onto a different definition than the one that was
validated. Self-healing
(`signal_journal.py`) required a specific fix here: `pattern_scorecard`/
`detect_drift`/`suspended_patterns` only ever read `pattern_library/`
for a pattern's mined win rate, and a `discovered__` pattern's stats
live in the separate `discovered_patterns/` directory - without
threading a `discovered_dir` parameter through, a live-firing
discovered pattern's mined win rate would always resolve to `None`,
which the credibility check treats as "can never be flagged DECAYING,"
silently disabling self-healing for every discovered pattern regardless
of how badly it performed live. Fixed by threading an optional
`discovered_dir` parameter through `_load_lib_cached`/`pattern_
scorecard`/`detect_drift`/`suspended_patterns`/`should_self_heal`, and
every caller (`signal_engine.py`, `live_update.py`, `dashboard/
server.py`, `scripts/report.py`) now passes it - verified with a
dedicated test reproducing the bug (mined_win_rate stayed `None`,
pattern never DECAYING, without the fix) and confirming the fix (mined_
win_rate correctly found, pattern correctly suspended once genuinely
losing live).

**Running it:** `python src/discover_patterns.py --symbol XAUUSD` (also
available as a "Discover Patterns (self-learn)" button on the rule-
based dashboard's Controls panel - not shown on the ML challenger's
dashboard, which has no equivalent primitive library to mine). Not
wired into `live_update.py`'s automatic hourly self-heal trigger -
unlike a `build_pattern_library.py` remine, a full discovery run is
comparatively expensive (genetic synthesis PLUS beam search over 226
hand-designed primitives up to depth 4, per timeframe - roughly
15-35 seconds per timeframe against a real multi-thousand-candle
history in testing) and is meant to be re-run periodically as a
deliberate, separate step (manually, or on your own schedule), the same
relationship `ml_registry/lib_view/`'s ML training already has to the
rule-based system's own remine cadence.

**Honest caveats.** This has been verified with targeted synthetic
tests at every layer - primitive look-ahead safety, worst-era
discrimination between a real and a regime-specific-only pattern,
correct FDR acceptance on a 100-candidate synthetic set with exactly 5
real signals, genetic synthesis correctly rediscovering an engineered
edge with no human choosing its base/transform/window/threshold,
correct combined synthesis+search `n_tested` accounting feeding FDR, and
a full end-to-end run (using the REAL 226-primitive catalog, no
monkeypatching) proving a discovered pattern - including one built
partly from a synthesized primitive - can drive a real live BUY/SELL
signal through `compute_signal()`, with live re-evaluation correctly
reconstructing the synthesized primitive from its stored expression and
correctly returning both True (at a known-firing candle) and False (at
a known-non-firing one) - but it has NOT yet been run against your real
20-year Dukascopy history. The stacked defenses above are deliberately
strict (worst-era scoring plus FDR plus a blind confirmation slice, on
top of the same 1:4/60%/out-of-sample hard gate everything else clears)
- expect it to discover FEW patterns, possibly zero on a first run,
rather than a large catalog; that is the design working as intended,
not a bug. A pattern that survives all of this deserves real trust; one
that doesn't survive was correctly not trusted, not a sign the search
failed. Cross-timeframe confirmation is diagnostic, not a hard gate - a
pattern showing `cross_timeframe.qualifies: false` is real information
(its live weight is discounted 50%), not evidence it should be deleted;
some genuinely valid patterns are honestly single-timeframe-scoped.

**Post-build audit.** A full re-read of every file this engine touches,
plus targeted empirical tests against each finding, caught and fixed
three real issues that the original build-and-verify pass missed:
1. `discovery_synthesis.py`'s "raw" transform ignored its `window`
   parameter entirely, so e.g. `synth_close_raw5_gt80` and `synth_
   close_raw50_gt80` were silently IDENTICAL boolean series under
   different names - wasted search budget, and worse, each counted as a
   separate trial toward FDR's `n_tested`, inflating it with zero
   genuinely new information (this biases the FDR bar stricter, not
   dangerously lenient, but was still a real correctness/honesty defect
   given this engine's "show your work" standard). Fixed by
   canonicalizing "raw" to a single fixed window in both random
   generation and mutation; verified the fix holds across 500 generated
   expressions.
2. `signal_engine._active_patterns_on_last_candle()`'s per-pattern
   discovered-pattern loop had no error isolation - one malformed entry
   in `discovered_patterns/*.json` (a stale primitive name, a corrupted
   write, a hand edit) would raise uncaught inside `is_discovered_
   pattern_active()` and crash live signal generation for EVERY pattern
   and timeframe, not just the bad one. This violates the fail-open
   convention this same module already applies to `load_suspended()`/
   `check_circuit_breaker_safe()`. Fixed with a per-pattern try/except
   (a broken entry is now just skipped, not fatal); verified with a
   synthetic corrupted entry that a good pattern alongside it still gets
   detected and the whole signal computation still completes.
3. Dead code in `discovery_search.py` (an unused `_conjunction_direction`
   helper and an unused `itertools` import, both pre-dating this
   session) removed.

Everything else - look-ahead safety across all synthesis grammar
combinations, no closure/late-binding bugs in the densified primitive
grids, correct `n_tested` accounting end-to-end, correct cross-timeframe
diagnostic computation and live-weight penalty application, dashboard
field names matching the API exactly - was re-verified and holds.

## Data sources

**Price:** [Dukascopy](https://www.dukascopy.com/)'s public historical
tick feed. Free, no API key, covers XAUUSD back 20+ years, and gives
tick-level data you can resample into any timeframe. Confirmed working
format:
- Price point for XAUUSD: raw integer / 1000 = USD price
- Tick record: 20 bytes, big-endian, `3x uint32 + 2x float32`

**Fundamentals:** [FRED](https://fred.stlouisfed.org/) (Federal Reserve
Economic Data) - free, official, needs a free API key. Tracks the core
gold drivers: CPI, PCE inflation, Non-Farm Payrolls, GDP, and FOMC rate
decisions, plus USD index (DXY) and 10-year real yields as background
context. See "Fundamentals" section below for the full methodology.

**Important:** both pipelines make real HTTPS requests
(`datafeed.dukascopy.com`, `api.stlouisfed.org`). Neither will run inside
a network-restricted sandbox (like the one this scaffold was built in) -
run the fetch steps on a machine with normal internet access.

### Data quality auditing (`data_quality.py`)

"How reliable is this data" gets answered two ways, both surfaced in the
dashboard's Data Storage panel (`/api/data_quality`), not just left in a
log line from an unattended multi-hour backfill nobody's watching:

1. **Backfill failures.** `build_history.py`/`live_update.py` now track
   every hour that genuinely FAILED to fetch (a network error after
   retries, or `dukascopy_fetch.DukascopyFormatError` - corrupt/
   truncated data, wrong record count, prices outside the sane gold
   range) separately from the expected "market closed, clean 404" case,
   which isn't a failure at all. Written to
   `data/data_quality_report.json` after every backfill run.
2. **Gap detection.** Independent of failure tracking (a run where every
   hour independently "succeeds" empty during what should have been a
   live trading day would never show up as a fetch failure),
   `detect_gaps()` scans each timeframe's actual candle file for
   consecutive-candle gaps bigger than a normal weekly close (~56h,
   deliberately not padded out to cover every possible holiday weekend -
   a human recognizing "that's Christmas" in the gap list is a cheaper
   mistake than silently missing a real multi-day outage) and bigger
   than 3x that timeframe's own candle spacing (catches shorter-but-
   still-abnormal holes on fast timeframes like 1min).

Two real data-handling bugs were fixed alongside adding this (verified
with reproducible tests, not just read and assumed correct):
- **`dukascopy_fetch.py` was caching corrupt downloads permanently.**
  Raw bytes were written to the on-disk cache immediately after
  download, BEFORE decoding/sanity-checking them - so a one-off
  corrupt/truncated download got cached, and every future "just re-run
  it to fill gaps" retry (a promise this module's own docstring makes)
  kept re-reading and re-failing on that same bad cached file forever
  instead of getting a fresh download. Fixed: bytes are only cached
  after they decode cleanly and pass the price sanity check: a
  previously-cached file that fails validation gets deleted so the next
  run actually retries. Verified with a synthetic corrupt cache file:
  before the fix the cache persisted after a failed read; after the fix
  it's gone, and a valid-data round trip (fresh download -> cached ->
  read from cache) still works correctly.
- **Tz-aware vs. naive candle timestamps** - see finding #11 in
  "Critical review" above. Real, previously-undiscovered, only found by
  actually running the ingestion code end-to-end (with the network call
  mocked) instead of only ever testing downstream code against
  hand-built synthetic candle files.

## Pipeline

```
1. build_history.py         -> downloads 20yrs of ticks, resamples to
                                1min/5min/15min/1h/4h/1d candles (Parquet)
2. build_fundamentals.py +
   news_calendar.py         -> historical + forward-looking CPI/PCE/NFP/
                                GDP/FOMC calendar (see Fundamentals below)
3. build_pattern_library.py -> for every pattern (candlestick, indicator,
                                fundamental, session open/close), simulates
                                the fixed 1:4 R:R trade at every historical
                                occurrence, computes win rate, flags
                                qualifies=true/false against the 60% gate,
                                AND news-conditioned win rate where enough
                                news-overlap samples exist (JSON stats)
3b. discover_patterns.py    -> optional, separate step: SELF-LEARNS its
                                own patterns from raw primitives (momentum/
                                volatility/structure/level/fundamental/
                                session) instead of testing a hand-picked
                                catalog - see "Pattern Discovery Engine"
                                above. Writes discovered_patterns/, never
                                touches pattern_library/.
4. signal_engine.py         -> compares the LATEST candle on each timeframe
                                against ONLY the qualifying patterns (both
                                pattern_library/ AND discovered_patterns/,
                                merged in memory) -> BUY/SELL/HOLD +
                                entry/stop/target, tagged with freshness/
                                expiry and news-window risk
5. live_update.py           -> run on a schedule: pulls new ticks, appends
                                to history, refreshes fundamentals/news
                                calendar, logs the signal to the journal,
                                resolves open journal entries. Run
                                build_pattern_library.py weekly/monthly to
                                fold the new live data back into the stats
                                (discover_patterns.py is run separately,
                                on its own cadence - see above).
```

Step 3 (and step 4's signal part) never re-touches history - it's pure
lookup/comparison against precomputed stats, matching the "we don't need
live data to make patterns, just to compare against them" approach.

## Using your actual broker's data alongside Dukascopy (never replacing it)

Dukascopy is a fine, deep default, but it's not necessarily what YOUR
broker prints - retail gold pricing is OTC, not exchange-traded, so
different brokers' liquidity providers construct OHLC slightly
differently, and MT4/5 history retention (especially on fine timeframes)
is usually much shorter than Dukascopy's. The credible use of a broker
feed is for LIVE execution fidelity, not as a replacement for deep
historical mining - see the "reliability/credibility" discussion in this
project's chat history for the full reasoning. `mt_bridge/` bridges
MetaTrader 4/5: an MQL script exports your broker's own history to CSV
from inside your terminal, a live-updating EA keeps it current, and
`src/mt_import.py` merges it into the exact same `data/candles/*.parquet`
files `build_history.py` produces - nothing downstream needs to know or
care which source the candles came from. See `mt_bridge/README.md` for
setup. Also reachable from the dashboard's Controls panel.

**Dukascopy is enforced as authoritative, not just a convention.** Every
candle is tagged with its source (`dukascopy` / `mt_broker`) when
written, and `build_history.merge_with_existing()` resolves any
overlapping timestamp by that tag - Dukascopy always wins, regardless of
which one was imported first. Verified both directions: a broker import
can't overwrite existing Dukascopy candles, AND a later Dukascopy
backfill correctly replaces broker candles that got there first. The
dashboard's Data Storage panel shows the source breakdown per timeframe
so you can see this holding, not just take it on faith. Net effect: your
broker's feed only ever fills gaps Dukascopy doesn't cover (typically
the most recent period via the live bridge) - the deep mining history
can't be silently degraded by a broker's shorter retention window.

Read the caveat at the top of `mt_bridge/README.md` before trusting the
MQL side: those scripts couldn't be compiled or run in the environment
this was built in (no MetaTrader available there), so - unlike
everything else in this project - they weren't tested end-to-end. The
Python side (`mt_import.py`, and the source-priority merge logic) was
tested rigorously, including the order-independence guarantee above.

## Fundamentals (CPI, PCE, NFP, GDP, FOMC)

These are mined through the exact same trade-simulation + hard-gate
pipeline as candlestick patterns - not a separate, looser system. Output
columns are prefixed `fundamental_*` so you can tell them apart in the
JSON library and in a signal's `contributions`.

**What "on what date, what time, what candle" means concretely:**
- **Date + time**: FRED gives the calendar release *date*; the intraday
  release *time* is added from well-established convention - CPI/PCE/
  NFP/GDP publish 8:30 AM Eastern, FOMC decisions 2:00 PM Eastern
  (`RELEASE_TIME_ET` in `build_fundamentals.py`), converted to UTC via
  `America/New_York` so it's correct across the EST/EDT boundary for all
  10 years, not off by an hour half the year like a fixed offset would be.
- **Which candle**: each event's UTC datetime is matched (pandas
  `merge_asof`, forward) to the first candle whose open time is at or
  after it - the candle the market reaction actually lands in
  (`fundamental_patterns.py`). Verified: a synthetic CPI event fired at
  13:30 UTC on a given day correctly landed on the *next* daily candle,
  not the one that had already closed before the release hit.
- **How inspected**: identical trade simulation to candlesticks - ATR-
  based stop, 4x target, walk forward to see which was hit first. A
  fundamental "pattern" (e.g. `fundamental_cpi_accelerating`) only
  contributes to a live signal if it clears the same 60% win rate /
  >=30 resolved trades bar. Verified end-to-end with synthetic data: an
  injected CPI edge reached 75.6% win rate on 41 resolved trades,
  qualified, and produced a signal with an exact 1:4 trade (risk 86.04
  -> reward 344.15).

**Honesty caveat (read this before trusting the numbers):** FRED gives
the actual published value, NOT the Wall-Street consensus forecast at
release time. There is no "beat vs miss vs consensus" here. Instead,
"accelerating"/"decelerating" means the release moved relative to *its
own recent trend* (this print's change vs the trailing 3-month average
change) - a real, fully data-derived reading, but not the same thing as
a forecast surprise. If you get a paid economic-calendar API with actual
consensus figures later, that's a strictly better signal than this one
and worth swapping in.

**Look-ahead fix (point-in-time vintages, not latest-revised):** CPI,
PCE, NFP, and GDP all get revised by BLS/BEA after their initial
release - NFP and GDP routinely and sometimes materially. Naively
fetching "the value for January 2024" from FRED today returns whatever
the CURRENT, most-revised figure is - not what a trader actually saw on
the release date. Mining a pattern's win rate on the revised number
while tagging it to the original release date is a genuine look-ahead
bias (the mined stat reflects information that didn't exist yet at
signal time). `build_fundamentals.py` uses
`fred_client.series_observations_as_first_published()` instead - FRED's
ALFRED vintage history (`output_type=2` with a wide `realtime_start`/
`realtime_end` window), taking each reference period's EARLIEST vintage,
i.e. the number as it was actually first published - so `value`/
`change`/`vs_trend` (and therefore every `fundamental_*_accelerating`/
`_decelerating` pattern) reflect what was genuinely knowable at the
time. FOMC's rate-decision series is deliberately left on the plain
(non-vintage) fetch - a policy rate the Fed announced on a given date is
a historical fact, not a preliminary survey estimate that gets revised
later. Verified: the vintage-selection logic (group by reference
period, keep the earliest `realtime_start`) against a synthetic
multi-vintage response modeled on a real NFP-style revision history -
confirmed it picks the as-first-published value, not the latest-revised
one, both in isolation and through the full `build_macro_event()`
pipeline end-to-end. NOT verified against a live FRED API call (FRED is
unreachable in the sandbox this was built in, same limitation
`dukascopy_fetch.py` has for Dukascopy) - sanity-check this against a
known historical NFP revision yourself before trusting a full mining
run built on it, same as the existing advice for `mt_bridge/`.

**Setup** (one-time, do this locally):
```bash
export FRED_API_KEY=your_key_here   # free: https://fred.stlouisfed.org/docs/api/api_key.html

# find FRED's actual numeric release_id for each event - deliberately not
# hardcoded anywhere, so a wrong guess can't silently pull the wrong data
python src/discover_release_ids.py

# copy the template and fill in the release_ids you just confirmed
cp event_config.example.json event_config.json

# fetch CPI/PCE/NFP/GDP/FOMC history + DXY/real-yield context series
python src/build_fundamentals.py
```
Then re-run `build_pattern_library.py` - it automatically picks up
`data/events/fundamentals.parquet` if present and mines fundamentals
alongside candlesticks. `live_update.py` refreshes fundamentals on every
run by default (pass `--skip-fundamentals` to turn that off for a given
run, e.g. if FRED is unreachable but you still want the technical signal).

## Trading sessions (Sydney/Tokyo/London/New York open+close)

Mined the same way as everything else, but need no external data source
at all - session times are a fixed daily convention (`session_patterns.py`),
computed directly from each session's own timezone with proper DST
handling, so it works fully offline. Every session pattern is direction-
ambiguous by design (same reasoning as `doji`/`inside_bar`): we don't
assume liquidity events are bullish or bearish, the 1:4 R:R win-rate
mining decides. Verified: session boundaries land on exactly one candle
each, repeat identically day to day, and correctly reproduce the known
Sydney-open/New-York-close overlap around 22:00 UTC in winter.

## News-window labeling, signal freshness, and the signal journal

Three related pieces that answer "is this signal actually still good,
right now":

**News proximity** (`news_calendar.py`): every live signal reports
whether high-impact news (CPI/PCE/NFP/GDP/FOMC - the same five tracked
events, all deliberately high-impact by construction, see scoping
rationale above; `session_*` events are labeled "normal" impact) is
scheduled to land before the trade's typical resolution time. The
resolution estimate comes from the pattern's own mined
`median_candles_to_resolve` (risk_reward.py), converted to real time via
the timeframe's candle duration - not guessed. The upcoming schedule
itself comes from FRED's forward-looking release calendar
(`release_dates(..., include_future=True)` - FRED/ALFRED tracks
scheduled dates ahead of time, not just after-the-fact publication
dates).

**News-conditioned win rate**: separately from "is news coming," every
pattern's historical occurrences are split into "a high-impact event
fell inside this trade's entry->resolution window" vs "it didn't"
(`risk_reward.tag_news_in_window` / `summarize_trades_conditioned`),
each gated independently against the same 60%/30-samples bar as
everything else. When a live signal has news in its window AND that
pattern has a trustworthy conditioned sample, the signal shows both
numbers side by side - e.g. a pattern that's normally 78% but drops to
52% (and doesn't independently qualify) when news lands mid-trade. Small
samples are labeled `qualifies: false`, not hidden or rounded up to look
authoritative.

**Signal freshness/expiry** (`signal_engine._freshness`): pattern
detection only ever looks at CLOSED candles - that's what was actually
backtested. So a signal is only as good as how recently its trigger
candle closed. Every signal reports its age in units of its own
timeframe's candle duration and is labeled FRESH (still within the
trigger candle) / AGING (up to 3 candles late) / EXPIRED (later than
that - the setup has likely already played out, don't treat it as
actionable). This is deliberately NOT "detect the pattern on a still-
forming candle to fire earlier" - that would score signals against data
the backtest never saw, so their true win rate would be unverified. The
freshness label is the honest way to make sure you never act on a stale
setup: it doesn't change entry/stop/target, it tells you when NOT to
trust them anymore.

**Signal journal** (`signal_journal.py`): every actionable signal
`live_update.py` emits gets logged (entry/stop/target frozen at that
moment, deduped by trigger candle so re-running on an unchanged candle
doesn't spam rows). On every subsequent run, open entries are walked
forward against new candles using `risk_reward.resolve_trade()` - the
IDENTICAL function the historical backtest uses - and marked `win`,
`loss`, or `expired` (past `MAX_LOOKAHEAD` candles with neither hit, same
policy as "unresolved" in mining). This is how you'd actually notice if
live performance drifts from what was mined: the journal's live win rate
is directly comparable to the backtest's, because both are computed the
same way. Verified end-to-end with synthetic price paths for all three
outcomes (win at the correct resolution candle with `actual_r` = the
system's fixed 4.0, loss at -1.0, and expiry after 60 flat candles).

### Self-assessment: the system grades its own live performance

The point of the signal journal isn't just record-keeping - it's so you
never have to risk real money to find out whether this thing actually
works. Everything below is pure computation over the journal
(`signal_journal.py`: `overall_scorecard`, `pattern_scorecard`,
`equity_curve`), surfaced in the dashboard's "Self-Assessment" panel and
via `python scripts/report.py` for a plain-text version (cron/SSH-
friendly, no browser needed):

- **Overall scorecard**: total signals, open/win/loss/expired, raw live
  win rate AND its 95% Wilson lower bound (the conservative number to
  actually trust on a small sample - matching the exact same statistic
  the mining gate uses, not a different ad hoc metric for live data),
  total realized R gross and cost-adjusted (a >50% win rate can still be
  a net loser at the wrong R multiples, or vice versa - R is what
  actually answers "is this making money"), average R per trade, current
  win/loss streak, and best/worst single trade.
- **Equity curve**: cumulative realized R across every resolved trade,
  in order - the literal "proof in the pudding" chart. If it's flat or
  trending down while individual patterns still show `qualifies: true`
  in the mined library, that's the live system telling you something
  the backtest isn't.
- **Per-pattern credibility** (`signal_journal.credibility()`): every
  pattern that's fired live and resolved at least once gets a verdict,
  not just a raw number to interpret yourself:
  - **UNPROVEN** - fewer than 10 live resolved trades. No score is
    computed, on purpose - this system doesn't fabricate a confidence
    number from data it doesn't have, live or mined (same principle as
    `MIN_RESOLVED_SAMPLES` in `risk_reward.py`).
  - **CONFIRMED** - live Wilson lower bound independently clears the
    same 60% gate the backtest required. The strongest live signal this
    system can give you that a pattern is actually working, not just
    theoretically.
  - **WATCH** - live samples exist but the Wilson lower bound hasn't
    cleared 60% yet. Could still be noise on a small sample (e.g. a
    pattern winning 9/12 live trades scores WATCH, not CONFIRMED,
    because the Wilson lower bound on 12 samples is genuinely wide -
    that's honest uncertainty, not the system being overly harsh).
  - **DECAYING** - live win rate's Wilson UPPER bound is still below
    the mined win rate. Statistically worse than backtested, not
    explainable by sampling noise - the same check `detect_drift()` uses
    for the journal panel's warning banner.

Verified with a synthetic journal covering all four cases at once (a
9W/3L pattern -> WATCH despite a 75% raw win rate, a 3W/9L pattern with
a 65% mined baseline -> DECAYING, and a 3W/1L pattern with only 4 live
trades -> UNPROVEN), plus the overall scorecard's win-rate/Wilson-bound/
total-R/streak math and the equity curve's chronological cumulative sum,
all checked against hand-computed expected values. `scripts/report.py`
was verified against the same synthetic data and against an empty
journal (prints "nothing to assess yet" instead of erroring).

## Self-healing: the system acts on its own self-assessment

Credibility scoring (above) answering "is this working" is only half of
"self-healing" - the other half is the system actually DOING something
about a "no" instead of just reporting it and waiting for a human to
notice. Two closed loops, both automatic:

### 1. Live-drift suspension (`signal_journal.suspended_patterns`)

Every (pattern, timeframe, direction) currently labeled `DECAYING` (see
"Self-assessment" above) is excluded from producing NEW live signals -
not just flagged on a dashboard. `signal_engine.compute_signal()` takes
a `suspended` set and skips any contribution that matches it, even if
the mined library still says `qualifies: true` for it (the library may
simply not have been remined since the pattern started failing live).
If a DIFFERENT direction of the same pattern still independently
qualifies and isn't suspended, that one is used instead of dropping the
pattern's vote entirely - `_qualifying_directions_ranked()` returns
every qualifying direction, ranked, so a suspended top choice falls
through to the next one rather than silencing the whole pattern.

Grouped by DIRECTION, not just pattern+timeframe: an ambiguous pattern
(doji, session_*, fundamental_*, ambiguous combos) can genuinely perform
differently long vs. short, and blending both together could suspend a
direction that's actually fine because its blended-in opposite is what's
really failing. Verified with a synthetic case: `doji` BUY at 11/12 wins
and `doji` SELL at 1/12 wins on the same timeframe - only SELL gets
suspended, BUY stays CONFIRMED.

If every qualifying direction for every active pattern is suspended at
once, the system goes quiet (HOLD) rather than keep firing on things
that no longer work. That's intended: if a real regime change has broken
the whole library's edge, silence until it's remined and re-proven is
the correct behavior, not a bug to route around.

### 2. Automatic re-mining (`signal_journal.should_self_heal`)

`live_update.py` checks, every run, whether the pattern library itself
needs rebuilding - not on a fixed weekly cron alone, but reactively:

1. **>= 3 pattern/timeframe/direction combos are DECAYING live at once**
   - the market has moved enough that the current library is measurably
   wrong, not just one pattern having a rough patch.
2. **The library hasn't been successfully rebuilt in >= 1 day** - a
   time-based backstop for when decay hasn't yet accumulated enough live
   samples to trip trigger #1, but new data has piled up regardless. Sized
   for continuous operation (`scripts/run_continuous.py` importing new
   broker candles every ~20-30s) rather than the original once-an-hour
   cron assumption - remining is a cheap, fully deterministic
   recomputation through the same hard gates every time, so re-mining
   more often is pure upside (fresher stats), not a tradeoff against
   noise or overfitting the way a shorter live-drift window would be.

Either trigger fires `build_pattern_library.rebuild_all()` automatically,
BEFORE that run's signal is computed, so the signal benefits from the
freshly-remined library the same run. A rebuild alone does NOT
un-suspend a pattern still showing a live losing streak - suspension is
driven by live journal history, which a remine doesn't erase; it takes
renewed live evidence (or the streak aging out) to lift it. This is
deliberate: it stops the system from re-mining its way back into
overconfidence about a pattern that's still actively failing live.
Disable with `--no-self-heal` if you'd rather control rebuilds manually.

Verified end-to-end: a synthetic journal with 3 heavily-losing patterns
correctly triggered an automatic rebuild mid-`live_update.py` run (visible
in its own log output and in `heartbeats.json`'s `build_pattern_library`
entry, written with a fresh timestamp despite `live_update.py` being the
process that triggered it); `--no-self-heal` correctly suppressed the
rebuild while suspension kept working independently (confirmed via a
library file left untouched - a test marker survived only when the flag
was set); the time-based trigger was separately verified against both a
10-day-old and a 1-day-old fake heartbeat.

### Regime-conditioned mining (`regime.py`) - the other half of "the market does new things"

Gold doesn't behave the same way in every market condition, and mining
one blended win rate across 20 years of everything hides that instead of
revealing it. `regime.py` classifies every candle into a combined
volatility/trend regime:

- **Volatility**: LOW / NORMAL / HIGH, from current ATR vs. the rolling
  mean of ATR over the preceding 100 candles (same basic comparison
  `patterns.atr_expansion` already uses for one binary flag, generalized
  to three buckets - deliberately not a true rolling percentile-rank,
  which would be meaningfully slower at 20-years-of-1min scale for a
  benefit this simpler ratio mostly already captures).
- **Trend**: TRENDING / RANGING, from ADX(14) vs. the conventional 25
  threshold (standard trend-strength indicator, reimplemented directly -
  same no-TA-Lib policy as `patterns.py`).

Both are causal (no look-ahead - verified by confirming a regime label
early in a series is unaffected by altering a candle far in the future).
Combined into one label (e.g. `"HIGH_TRENDING"`) for every ATOMIC pattern
(not combos - same reasoning as combos skipping news-conditioning: their
samples are already smaller by construction, splitting further into up
to 6 regime buckets would rarely leave enough to say anything), stored
under `"by_regime"` in the mined library, gated by the identical hard-gate
rules (including its own out-of-sample check) as every other stat in this
system.

At signal time, `signal_engine.py` looks up each contributing pattern's
stats for the CURRENT regime and applies a 0.5x weight discount
(`REGIME_MISMATCH_PENALTY`) - but ONLY when there's enough regime-specific
history to say the pattern has genuinely been tested in this regime and
failed (its own `resolved >= MIN_RESOLVED_SAMPLES` and `qualifies: false`),
never for "not enough regime-specific data yet," which leaves weight
untouched. This is a weight discount, not a hard gate, on purpose: an
additional hard gate here would cut deeply into signal frequency for
every pattern that simply doesn't have enough regime-specific samples
yet (most of them, especially for rarer regimes) - the self-healing
suspension mechanism above already handles genuine, currently-measured
live underperformance; this is a softer, backtest-informed nudge on top.
Shown transparently on the dashboard's contributions table (a "REGIME ⚠"
badge when a pattern's weight was discounted this way, plus the
regime-specific win rate/sample count next to it) rather than silently
applied.

Verified: mining runtime impact is negligible (full mining over 20,000
candles x 1046 patterns - atomic + combo - takes ~3.4s total, since
regime-conditioning only runs on the ~44-55 atomic patterns, not the 991
combos); a synthetic case with a pattern
that qualifies overall (65%) but shows 40 resolved / 30% win rate in the
CURRENT regime specifically had its weight exactly halved and
`regime_penalized: true`; the same pattern with only 5 regime-specific
resolved trades (under `MIN_RESOLVED_SAMPLES`) was correctly left
unpenalized - not enough evidence to say anything yet.

## Circuit breaker: portfolio-level hard stop (`circuit_breaker.py`)

Everything above - hard gates, live-drift suspension, regime-conditioning -
is a per-pattern (or per-model) statistical judgment: it decides whether a
SPECIFIC pattern/timeframe/direction should currently be trusted. What none
of it does is look at the system's overall REALIZED trajectory and ask "is
this actively going wrong RIGHT NOW, regardless of which individual
patterns still say they're fine?" That's the gap a real trading desk,
broker, or exchange closes with a portfolio-level circuit breaker - a hard,
system-wide stop that doesn't care whether any particular pattern still
shows `qualifies: true`, because statistics describe the past and a
regime break severe enough can outrun self-healing before it's finished
reacting.

`check_circuit_breaker(journal)` evaluates three independent triggers
against ONLY realized outcomes (win/loss/expired) plus open-trade count -
never unrealized P&L, which would need a live price feed this module
deliberately doesn't have or need. Any one trigger is sufficient to trip:

1. **Consecutive loss streak >= 8.** Under a genuinely ~60%+ win-rate
   system, a streak this long has under a 0.1% chance if trades were
   independent draws from the system's own claimed win rate
   (0.4^8 ≈ 0.07%) - strong evidence something is systematically wrong,
   not normal variance.
2. **Peak-to-trough drawdown worse than -15R.** The system is designed for
   strongly positive expectancy (a 60%-win-rate pattern at fixed 1:4 R:R
   has expectancy 0.6*4 - 0.4*1 = +2.0R per trade) - a drawdown this large
   is inconsistent with normal variance around a genuine edge.
3. **>= 6R of simultaneous open risk** (each open trade risks a fixed 1R
   by construction, so this is just "how many trades open at once") -
   prevents an unbounded pile-up of simultaneous exposure if many
   patterns/timeframes fire in a short window.

When tripped, `compute_signal()` / `compute_ml_signal()` force a hard
`HOLD` - no contributions even computed - overriding every individually-
qualifying pattern or model, the same override precedence a suspended
pattern gets, just system-wide instead of per-pattern. Nothing in this
system auto-clears a trip; a human has to look at what happened. The
result is always included in the signal JSON (`"circuit_breaker": {...}`)
whether tripped or not, so the dashboard can show how close the system is
to tripping, not just a binary state - surfaced as a persistent red banner
above the Signals Engine card when tripped (streak/drawdown/open-risk
numbers included), and absent entirely when healthy.

Each engine is breakered independently against its OWN journal - the
rule-based system and the ML challenger each have their own signal
journal and are self-healed independently (see below), so they're
breakered independently too; a tripped rule-based breaker does not halt
the ML challenger, and vice versa.

Failure-safe by construction, matching every other self-healing check in
this system: `signal_engine.check_circuit_breaker_safe()` returns `None`
(not evaluated) rather than raising, if the journal can't be read for any
reason - a portfolio-level safety check erroring out must never itself be
what takes signal generation down.

Verified: unit-level, each of the three triggers isolated from the other
two (a synthetic `(['loss']*7 + ['win']) * 5` journal accumulates -18R of
drawdown with no single loss run reaching 8 in a row, tripping only on
drawdown; a 7-loss streak, one below threshold, does not trip; 7
simultaneously open trades trips on open-risk alone); a real
`atomic_write_parquet` -> `load_journal` round trip (not just an
in-memory DataFrame) was confirmed to preserve the trip decision,
including the open-risk case's mixed `None`/`float` `actual_r` column.
End-to-end: the REAL `live_update.py` and `ml_live_update.py` CLI entry
points (network fetch stubbed out, everything else real) were run against
a synthetic candle set with a genuinely pattern-detector-qualifying
"hammer" on the last candle (rule-based) and an always-qualifying dummy
model (ML) - a healthy journal let a real `BUY` signal through the CLI in
both cases, and seeding an 8-loss-streak journal beforehand forced `HOLD`
and printed `CIRCUIT BREAKER TRIPPED` in both cases, despite the
otherwise-qualifying pattern/model. The dashboard's `/api/signal` endpoint
was verified the same way against a real running server (subprocess,
real HTTP request) for all four combinations of {rules, ml} x
{healthy, tripped}.

## Loss/win attribution: the system diagnoses WHY, not just WHETHER (`signal_journal.context_scorecard`/`context_penalty`)

Everything above answers "is this pattern working" - live-drift
suspension, regime-conditioning, the circuit breaker. None of it answers
"WHY did this specific trade hit its stop" in a way that accumulates
into better future signals. This closes that gap, but the honest way,
not the naive one.

**The wrong way to build this** (worth naming so it's clear why this
isn't what's here): reacting to a single loss and adjusting the next
signal off it. A single loss tells you almost nothing - even a
genuinely 70%-win-rate pattern loses 3 times in 10 by design. Any
"learns from its losses" feature that reacts per-trade is just
overfitting to noise wearing a different name, and would work against
every other piece of statistical discipline in this system (Wilson
bounds instead of raw win rate, `MIN_RESOLVED_SAMPLES` gates, purged
CV for the ML challenger).

**What's actually here:** every logged signal captures a handful of
CONTEXT features true at the moment it fired - never anything from
after, no look-ahead risk:

- **`confluence_count`** - how many OTHER contributing patterns
  independently agreed with the direction this signal actually took,
  even ones not directly counted toward the weighted vote (correlated-
  but-not-counted contributions are still corroborating evidence a
  trader would see). A signal riding on exactly one pattern with zero
  confirmation is a genuinely different claim than one with three
  independent confirmations, and this system was previously throwing
  that distinction away entirely.
- **`session_at_entry`** - which trading session(s) were actively open
  at that instant (`session_patterns.active_sessions_at()` - a live
  SPAN membership check, e.g. "London+New York overlap" or
  "off_hours", not the boundary-instant OPEN/CLOSE events
  `session_*` patterns already mine separately).
- **`regime_at_entry`** - captured for completeness/transparency, but
  deliberately NOT re-analyzed here - the mined library already
  reports a regime-conditioned win rate and `signal_engine.py` already
  applies a live weight discount from it (`REGIME_MISMATCH_PENALTY`,
  see "Regime-conditioned mining" above); a second, separate regime
  analysis here would only duplicate that against a much smaller
  live-only sample.
- **`volatility_shock_ratio`** (filled in only once a trade RESOLVES,
  by construction - not knowable before) - realized ATR during the
  trade's actual lifetime vs. the ATR the stop was originally sized on
  (back-derived from the journal's own `risk` column, no separate
  value to store). >1 means the market was genuinely more volatile
  during the trade than the stop assumed - a real, distinct candidate
  explanation for a loss ("the pattern wasn't wrong, a volatility
  spike right after entry overran a stop that was reasonable for the
  conditions AT signal time") separate from "the pattern itself just
  doesn't have an edge here." Purely diagnostic, forever - this is the
  one dimension that can NEVER feed back into a live decision, because
  it isn't knowable until after the trade is already resolved.

`context_scorecard()` mines these against actual outcomes with the
IDENTICAL discipline every other statistic in this system already
uses: grouped by (pattern, timeframe, direction) first (a context
factor can matter for one pattern and not another), each bucket's live
win rate Wilson-bound and compared against that SAME pattern's OTHER
live trades (its own peers, not the mined number - the question is
"does this context matter," not "is this pattern still working," which
`detect_drift()` already answers), hard-gated by `CONTEXT_MIN_LIVE_SAMPLES`
(15, deliberately higher than the plain drift-detection bar of 10 -
testing several dimensions x several buckets x every pattern is a much
bigger multiple-comparisons search than one pattern-level verdict, same
reasoning `risk_reward.py` already applies to combo patterns needing a
higher bar than atomic ones) on BOTH the bucket and the comparison group.

**The feedback loop, engineered to only ever activate on real evidence:**
`context_penalty()` is a live lookup `signal_engine.py`/`live_signal.py`
call for every about-to-fire signal - if its own confluence count or
active session matches a bucket already PROVEN (hard-gated, Wilson-bound)
to underperform that exact pattern's other live trades, confidence is
discounted (`CONTEXT_PENALTY_MULTIPLIER = 0.5`, same magnitude and same
reasoning as `REGIME_MISMATCH_PENALTY`) - a soft discount, never a hard
block; `suspended_patterns()` is still the only mechanism that silences
a pattern outright, and only once its own, stricter bar is cleared.
Fails open (no penalty) on any error or if the caller has no journal
handy, same posture as every other self-healing lookup in this system.
For most pattern/context combinations early on, this is simply a no-op
- `CONTEXT_MIN_LIVE_SAMPLES` per bucket is real evidence to accumulate,
not something a handful of live trades produces by chance.

Surfaced transparently, never silently: the dashboard's Signals Engine
card shows a "LOSS ATTRIBUTION: confidence discounted (×0.5)" box with
the exact reason whenever a penalty is active, the Self-Assessment panel
has its own "Loss/win attribution" table showing every proven bucket
(and the healthy ones, for contrast) per pattern, and `scripts/report.py`
prints the same finding for anyone checking via cron/SSH without the
dashboard open.

Verified end-to-end, not just unit-tested in isolation: a synthetic
journal where confluence=0 trades genuinely underperform confluence=2
trades for the same pattern was correctly flagged DECAYING (Wilson-bound)
while the strong bucket was correctly left alone; the identical
disparity at 8 samples/bucket (below `CONTEXT_MIN_LIVE_SAMPLES`)
correctly produced NO verdict, proving the hard gate actually gates; an
unrelated pattern was confirmed unaffected (no cross-pattern
contamination); the session dimension was verified the same way
independently. A real `compute_signal()`/`compute_ml_signal()` call
(not the isolated scorecard function) confirmed confidence is actually
discounted end-to-end with the correct multiplier and a transparent
reason, that a healthy journal produces no penalty, and that omitting
the journal argument entirely reproduces the exact pre-feature behavior
(full backward compatibility). The `volatility_shock_ratio` computation
was verified against an engineered volatility-spike scenario (a calm
warm-up period, then a wild candle that blows through a tight stop) and
correctly reported a large shock ratio. The dashboard's rendering
(both the hero-card penalty box and the Self-Assessment attribution
table) was confirmed against a real running server and screenshots, with
the test journal specifically engineered so the confluence effect could
be observed in isolation from the (unrelated, and correctly
higher-precedence) circuit breaker and whole-pattern suspension checks.

## Causal autopsy: WHY a pattern's own historical occurrences won or lost (`src/causal_autopsy.py`)

The section above answers "did THIS live signal's context matter" from
the (necessarily small) pool of live trades logged so far. This answers
a related but different question against the FULL mined history: "of
EVERY time this pattern fired historically, what actually separated the
occurrences that hit their target from the ones that hit their stop" -
not a single trade's story, a statistically honest comparison across
however many hundreds of occurrences a pattern has.

**Method** (same statistical discipline as `discovery_validation.py`,
adapted to a different question): split a pattern's resolved historical
occurrences into a WIN bucket and a LOSS bucket (by the same
`risk_reward.simulate_trades()` outcome everything else in this system
is graded by), then run a Mann-Whitney U test - nonparametric, no
assumption the feature is normally distributed - on each of
`ml_system/features.py`'s 80+ engineered numbers (ATR-normalized
distances to levels, RSI/MACD/OBV/ADX readings, session/calendar
encodings, ...) between the two buckets. Benjamini-Hochberg FDR
correction is applied across every feature actually tested (~80
simultaneous comparisons - real multiple-comparisons exposure, corrected
for honestly, `n_features_tested` always recorded). A feature surviving
FDR is reported as "measurably different between win/loss buckets," an
association - never as "the reason" a trade won or lost, since no system
can honestly claim that.

**Not a curated demo list** - `causal_autopsy.py` runs over
`build_pattern_library.compute_pattern_flags()`'s full pattern set: every
candlestick/indicator, session, fundamental, support/resistance, and smc
pattern, plus every cross-family combo, direction-agnostic ones tested
both as-long and as-short (mirroring how `build_library()` itself gates
ambiguous patterns both ways). ~4s to autopsy all ~1900 patterns over
5,000 candles in testing - most patterns don't have enough resolved
win/loss occurrences to test at all, so cost is dominated by the ones
that do, not the raw column count.

**Visible wherever the rest of the system already looks**, not a
disconnected report file: `scripts/event_autopsy.py --merge-into-library`
writes each pattern's causal result directly into
`pattern_library/<symbol>_<tf>.json` under a new `"why"` key on that
pattern's own entry, next to its win rate and OOS stats - the dashboard's
Patterns Data table picks it up automatically as a "Why (win vs loss)"
column, no separate view to remember to check. Without `--merge-into-
library`, results go to their own `--out-dir` instead (`event_autopsy/`
by default) for exploratory runs that shouldn't touch the live-serving
library.

```bash
# autopsy everything already in the mined library, merge "why" back into it
python scripts/event_autopsy.py --symbol XAUUSD --patterns library --merge-into-library

# autopsy every DETECTED pattern regardless of whether it's in the library yet
python scripts/event_autopsy.py --symbol XAUUSD --patterns all

# original narrow-list behavior still works
python scripts/event_autopsy.py --symbol XAUUSD --events bullish_engulfing,bearish_engulfing
```

## The ML challenger (`ml_system/`)

A second, independent signal-generation system for the same instrument -
run alongside this one as a **challenger, not a replacement**: same
candles, same fixed 1:4 R:R structure, same hard 60% win-rate gate, same
journal/self-assessment/self-healing machinery (reused unmodified, not
reimplemented), but signals come from a trained model searching a
44-feature space instead of ~55 hand-picked candlestick/indicator/
support-resistance patterns and their pairwise combos.

The plan is champion vs. challenger: run both, track both with the
identical self-assessment tooling this README already documents above,
and only consider merging anything once the ML system has independently
earned `CONFIRMED` credibility on real LIVE trades - never on a backtest
claim alone. Both systems can run on the same machine or genuinely
different devices; only the read-only shared inputs (candle data, the
forward news calendar) need to be kept in sync between them.

This is the highest-risk, highest-reward idea on this project's list
(see the earlier design discussion in this README's history for the
full risk breakdown: overfitting on financial time series' overlapping,
autocorrelated labels; opacity vs. the transparency everything else here
is built around; training-serving skew), so it was built with three
concrete, TESTED safeguards rather than taken on faith:

1. **Purged, embargoed walk-forward cross-validation**
   (`ml_system/validation.py`), not a plain train/test split - a naive
   split lets a training example's label (which depends on up to 61
   future candles) leak information from inside the validation window.
   Verified with real assertions: zero train/validation index overlap,
   every training example's label window provably resolves before its
   validation block starts.
2. **One shared feature-computation function** (`ml_system/features.py`)
   used identically by training and live scoring - no second
   implementation that could quietly drift from what the model was
   actually trained on.
3. **A promotion gate that reuses `risk_reward.summarize_trades()`
   directly** - the exact function that gates the rule-based system's
   patterns - so a new model version only replaces the active one if it
   independently clears the identical hard bar and doesn't score below
   whatever's currently deployed. Verified: a qualifying-but-weaker
   candidate correctly did NOT replace a stronger active model; a
   genuinely stronger one correctly did.

Verified end-to-end on synthetic data: 0/2 directions qualified on pure
random-walk noise (correct - no real edge to find); a genuine, injected
relationship (price reliably rallying whenever RSI dropped below 25) was
correctly found and promoted, clearing 71-74% win rate on 2,700-3,900
pooled out-of-fold validation trades across test runs. Self-healing
(live-drift suspension + auto-retrain triggers) reuses
`signal_journal.should_self_heal()` unmodified and was verified with the
same rigor as the rule-based system's version.

The SAME dashboard (`dashboard/server.py`) serves either system, picked
by an env var (`SIGNAL_ENGINE=ml`), so both can be watched with identical
tooling. Full architecture, setup commands, and an honest "what's still
open" list (no cross-asset features yet, no hyperparameter tuning, no
automated side-by-side comparison report) are in `ml_system/README.md`.

## Setup

```bash
pip install -r requirements.txt
```

## First run (do this locally, not in this sandbox)

```bash
# sanity-check the data format against one recent hour before committing
# to a multi-year download
python src/dukascopy_fetch.py

# full 20-year backfill across all timeframes (resumable - safe to
# re-run if interrupted, already-downloaded hours are cached under
# data/raw_bi5/; safe to ask for more years than actually exist, see
# build_history.py). Wrapped in scripts/supervise.py so a dropped
# connection, a laptop sleeping, or anything else that kills or hangs a
# multi-hour download gets restarted automatically instead of silently
# stopping - see "Running unattended" below for exactly what that buys you.
python scripts/supervise.py -- python src/build_history.py --years 20 --workers 8

# optional but recommended: fundamentals + forward news calendar (see
# "Fundamentals" and "News-window labeling" sections above for setup)
python src/build_fundamentals.py
python src/news_calendar.py

# mine the patterns (candlestick + indicator + session + fundamental,
# with news conditioning if fundamentals are present)
python src/build_pattern_library.py

# optional: self-learn additional patterns from raw primitives instead of
# the hand-picked catalog above - see "Pattern Discovery Engine" above.
# Slower than build_pattern_library.py (genetic synthesis plus beam search over 226 primitives);
# fine to skip on a first run and come back to it once you have real
# history to mine.
python src/discover_patterns.py

# get today's signal (also logs it to the signal journal)
python src/signal_engine.py
```

`--years 20 --workers 8` downloads roughly 175,000 hourly files (many are
empty - market closed on weekends, or predate however far back XAUUSD
data actually goes). Expect this to take a while on the first run; it's
I/O bound and safe to interrupt/resume.

## Running unattended (`scripts/supervise.py`)

The one command in this pipeline most likely to get interrupted - a
multi-hour, multi-thousand-request Dukascopy backfill - is exactly the
one you most don't want to have to babysit or manually restart.
`scripts/supervise.py` wraps any command in this repo with a watchdog:

```bash
python scripts/supervise.py -- python src/build_history.py --years 20 --workers 8
```

It restarts the wrapped command automatically (with exponential backoff)
if it either **crashes** (non-zero exit) or **stalls** (no stdout/stderr
output for `--stall-timeout` seconds, default 10min - a genuinely slow
step keeps printing progress, a truly hung one doesn't). It runs forever
by default, until the command finally exits 0; pass `--max-restarts N`
if you'd rather it give up and let you look at what's actually wrong
after N tries.

**"Start fetching exactly where it left off" is not something
`supervise.py` has to implement itself** - it just re-runs the identical
command. The resumability comes from the pipeline scripts underneath it:
`build_history.py` caches each hour to `data/raw_bi5/` only once it's
downloaded, decoded, AND passed the price sanity check, so a restart
re-scans the same date range but every already-good hour is read from
disk in milliseconds instead of re-fetched. And every file this pipeline
writes (candle parquet files, the pattern library JSON, the signal
journal, `heartbeats.json`) is written atomically (`atomic_io.py` - temp
file + atomic rename) - a kill mid-write leaves the previous good file
in place, never a truncated one that would break the *next* restart's
attempt to read it. Verified with a real crash-mid-run test: a command
wrapped in `supervise.py`, simulating a crash right after caching item 5
of 10, correctly resumed on restart with items 0-5 skipped as
already-cached and only items 6-9 freshly fetched - the same mechanism
`build_history.py`'s real `data/raw_bi5/` cache uses.

**Concurrent writers to `heartbeats.json` specifically:** every OTHER
file this pipeline writes (candles, pattern library, the signal journal)
belongs to exactly one writer at a time by construction, so atomic
writes alone are enough. `heartbeats.json` is different - it's one
shared dict every job's entry lives in, and `scripts/run_continuous.py`
made "two processes write to it at the same moment" a real, not
hypothetical, scenario: a continuous loop's own cycle and a dashboard
action button click can now genuinely race on the same file.
`heartbeat.write_heartbeat()` serializes its read-modify-write with an
OS-level advisory lock (`fcntl.flock` on a sibling `.lock` file) so a
losing writer's update can no longer be silently discarded - verified by
first reproducing the loss (20 real, separate OS processes writing
concurrently to the same file with the lock disabled dropped 7 of 20
updates) and then confirming the same test passes cleanly (20/20) with
the lock in place.

Also useful for anything else in this pipeline you want to survive
unattended - `live_update.py` on a loop, or the dashboard itself:

```bash
python scripts/supervise.py --log logs/live_update.log -- python src/live_update.py
python scripts/supervise.py --stall-timeout 60 --log logs/dashboard.log -- python dashboard/server.py
```

(For the dashboard specifically, a low `--stall-timeout` is fine and
recommended - a healthy `uvicorn` server logs a line per request/access,
so real silence for 60s+ genuinely means it's wedged, not just idle.)

## After that: keep it fed - genuinely continuous operation (`scripts/run_continuous.py`)

**How "live" this actually is, read this before assuming it watches the
market by itself:** `live_update.py` / `ml_live_update.py` each compute a
fully real, live signal from real current market data every time they
run - but only when something runs them. Neither one loops or watches
the market in the background on its own. `scripts/run_continuous.py`
is that missing piece: it runs the requested engine's update cycle
(fetch/import newest data -> recompute signal -> self-heal check -> log
to journal) in a genuine infinite loop, sleeping `--interval-seconds`
between cycles, from the moment you start it until you stop it
(Ctrl+C, or `kill`/SIGTERM if run detached) - not "live when you happen
to run it," an always-on process:

```bash
python scripts/run_continuous.py --engine rules
python scripts/run_continuous.py --engine ml
```

`--interval-seconds` is source-aware if you don't set it explicitly: 60s
against the default Dukascopy source, or 20s automatically if you pass
`--import-mt-broker-dir` (matching your broker's own live-bridge
cadence - see below). An explicit `--interval-seconds` always overrides
either default. This is the actual lever for minimizing lag between a
candle closing and you seeing the signal - see the next paragraph for
why going tighter than the Dukascopy default doesn't buy anything, and
why the broker feed path is where a genuinely tight interval matters.

Run one instance per engine (matching how every other part of this
project keeps the rule-based and ML challenger systems independent) -
two long-running processes (two terminals, or two systemd services /
tmux panes), each continuously watching the market and firing its own
signals until you stop it.

Each cycle runs the underlying script as a **fresh subprocess**, not one
giant in-process loop - a clean interpreter every cycle, so nothing can
leak memory or accumulate bad state across days or weeks of continuous
operation. A cycle that crashes or hangs (no output for `--stall-timeout`
seconds, default 10min - the identical stuck-detection `supervise.py`
uses for the one-shot backfill) is killed and logged, and the loop moves
straight on to the NEXT cycle rather than giving up - a real feed
handler doesn't permanently stop because one tick had a network blip, it
just tries again next tick. Output is both printed live and appended to
`logs/<engine>_continuous.log` (or wherever `--log` points).

**How fast can it actually SEE new data? Bounded by the data source, not
by this loop's interval.** Dukascopy (the default source) publishes ONE
FILE PER COMPLETED UTC HOUR - so no matter how tight you set
`--interval-seconds`, there is genuinely nothing new to find more than
once an hour; the 60s auto-default exists to catch that new hourly file
promptly once it's published, not to invent sub-hourly data that doesn't
exist - and going tighter than that against Dukascopy only means retrying
an hour that isn't published yet more often, not finding anything sooner.
If you're running your own broker's feed via `mt_bridge/` (genuinely
sub-minute - the live bridge appends newly closed candles every 30s),
pass `--import-mt-broker-dir` (and `--mt-symbol`) so every cycle also
pulls in whatever the bridge has appended since the last cycle before
recomputing the signal - this is the actual fast path, and the interval
auto-adjusts to 20s to match it, no extra flag needed:

```bash
python scripts/run_continuous.py --engine rules \
    --import-mt-broker-dir "/path/to/MQL5/Files/gold_export" --mt-symbol XAUUSD.a
```

**Verified:** the loop's own mechanics (repeats for N cycles, honors the
sleep interval, logs every cycle to file, and - critically - a cycle
that fails EVERY time still keeps retrying forever rather than giving up
permanently) were confirmed against a controllable dummy command
standing in for the real engine script, as was the source-aware default
interval (60s with no broker dir, 20s when `--import-mt-broker-dir` is
set, and an explicit `--interval-seconds` always wins over either
default - all three confirmed by reading the script's own startup log
line, not just by inspecting the code). Clean shutdown was verified two
ways: sending SIGTERM mid-cycle correctly terminates the exact wrapped
child subprocess (confirmed by checking its exact PID is gone from
`/proc` afterward, not just assumed) and returns exit 0 with a clean
shutdown message - no orphaned process left running. This was then
re-confirmed against the REAL `live_update.py` subprocess under genuine
(if slow) real-world network failure conditions: killed mid-cycle, it
shut down cleanly with zero orphaned processes left behind.

**If you prefer OS-level scheduling instead of a long-running process**
(e.g. you don't want to manage a systemd service / tmux session), cron +
`supervise.py` is still a fine alternative - it just checks in hourly
rather than continuously:

```cron
0 * * * *  cd /path/to/gold-signals-bot && python scripts/supervise.py --stall-timeout 300 --max-restarts 3 --log logs/live.log -- python src/live_update.py
0 3 * * 0  cd /path/to/gold-signals-bot && python scripts/supervise.py --stall-timeout 1800 --max-restarts 2 --log logs/rebuild.log -- python src/build_pattern_library.py
```

## Tuning knobs worth knowing about

- `src/patterns.py` - the pattern catalog. Add new candlestick/indicator
  rules here; `build_pattern_library.py` and `signal_engine.py` pick up
  anything added to `ALL_PATTERNS` automatically.
- `src/support_resistance.py` - `SWING_LOOKBACK` (5) controls both the
  fractal window and the confirmation lag before a swing point becomes
  usable (see the module docstring - widening this makes swing points
  more selective but slower to confirm). `ROUND_NUMBER_STEP` (50.0) - the
  psychological grid spacing; gold-specific, would need changing for a
  different-priced instrument. `PROXIMITY_ATR_MULT` (0.5) / `REJECTION_
  WICK_MULT` (1.5) - how close counts as "at" a level and how strong the
  rejection wick must be, same tuning tradeoff as `patterns.py`'s hammer/
  shooting_star thresholds. `build_pattern_library.py` and
  `signal_engine.py` pick up anything added to `SUPPORT_RESISTANCE_
  PATTERNS` automatically, same as `ALL_PATTERNS` above.
- `src/smc_patterns.py` - `FVG_LOOKBACK` (2, the classic 3-candle gap -
  changing this changes what "Fair Value Gap" even means, not really a
  tuning knob) and `EQUAL_LEVEL_ATR_MULT` (0.15) - how tight two swing
  points must sit to count as an "equal highs/lows" liquidity pool
  (wider = more pools detected but each less genuinely "equal"). Reuses
  `support_resistance.SWING_LOOKBACK`/`confirmed_swing_points()` for its
  own fractal detection, so widening that knob affects this family's
  patterns too. `build_pattern_library.py` and `signal_engine.py` pick up
  anything added to `SMC_PATTERNS` automatically, same as `ALL_PATTERNS`
  above.
- `src/risk_reward.py` - where both hard gates actually live.
  `RR_RATIO` (4.0) and `MIN_WIN_RATE` (0.60) are the two constraints from
  the spec - don't loosen these without knowing you're doing it.
  `STOP_ATR_MULTIPLE` (1.5) controls how wide the stop is relative to
  volatility (wider stop = fewer whipsaw losses but bigger $ risk per
  trade). `MAX_LOOKAHEAD` (60 candles) - how long a trade is allowed to
  stay open before being marked "unresolved" (excluded from win rate,
  not counted as a loss). `MIN_RESOLVED_SAMPLES` (30) - patterns with
  fewer resolved trades than this can't qualify even at 100% win rate,
  since the sample is too small to trust. `OOS_FRACTION` (0.30) /
  `OOS_MIN_RESOLVED_SAMPLES` (10) - the out-of-sample holdout gate size
  (see "Critical review"). `SPREAD_USD` (0.30, PLACEHOLDER) - round-trip
  transaction cost assumed for `expectancy_r_after_costs`; calibrate to
  your actual broker before trusting the after-cost numbers.
- `signal_engine.py`: `TIMEFRAME_WEIGHTS` - how much each timeframe's
  vote counts when ranking among already-qualifying patterns (higher
  timeframes weighted more, they're less noisy). This never overrides
  the hard gate - a non-qualifying pattern gets zero weight, period.
- `event_config.json` - which FRED release_id/series_id feed each
  fundamental event. Only events with a non-null `release_id` get
  fetched; leave one out (e.g. FOMC not confirmed yet) and the rest still
  work fine. `event_timing.py`'s `RELEASE_TIME_ET` - the assumed intraday
  release time per event type, if a release's schedule ever changes.
  `EVENT_IMPACT` - currently all five tracked events are "high" impact
  by design (see scoping rationale); add entries here if you ever bring
  in lower-relevance events you'd want labeled "normal" instead.
- `session_patterns.py`'s `SESSIONS` dict - session open/close hours per
  timezone, if a market convention ever shifts.
- `signal_engine.py`'s `FRESH_MAX_CANDLES` (1.0) / `AGING_MAX_CANDLES`
  (3.0) - how many trigger-candle-durations old a signal can be before
  it's labeled AGING, then EXPIRED. Tighten these if you're polling
  live_update.py more often than hourly and want stricter freshness.
- `signal_journal.py`'s `MIN_LIVE_SAMPLES_FOR_DRIFT` (10) - how many
  resolved live trades a pattern needs before `detect_drift()` will
  compare it against its mined win rate at all (below this, there isn't
  enough live data to tell drift from noise).
- `position_sizing.py` - not auto-applied to signals; call directly (or
  use the dashboard's calculator) with your own account size, risk %,
  and broker's contract size. `contract_size` has no default for a
  reason - gold contract specs vary by broker (100oz standard / 10oz
  mini / 1oz micro are common patterns, not universal).
- `signal_journal.py`'s `REBUILD_TRIGGER_DECAYING_COUNT` (3) / `REBUILD_TRIGGER_MAX_AGE_DAYS`
  (1) - the two self-heal triggers for an automatic library rebuild (see
  "Self-healing"). `signal_engine.py`'s `REGIME_MISMATCH_PENALTY` (0.5) -
  how much a pattern's weight is discounted when it fails to independently
  qualify in the CURRENT market regime specifically (with enough
  regime-specific samples to trust that verdict).
- `regime.py`'s `VOLATILITY_BASELINE_WINDOW` (100) / `VOLATILITY_LOW_MULTIPLE`
  (0.8) / `VOLATILITY_HIGH_MULTIPLE` (1.25) - the volatility-regime
  thresholds. `TREND_ADX_WINDOW` (14) / `TREND_ADX_THRESHOLD` (25) - the
  trend-regime ADX convention.

## Extending

This intentionally has no multi-asset/exchange abstraction - it's one
symbol (`XAUUSD`), one data source. If you ever want to add correlated
macro features (DXY, real yields, CPI surprises) to sharpen the patterns,
that's a new column-join step feeding into `build_pattern_library.py`,
not a rewrite.
