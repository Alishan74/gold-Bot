# Dashboard

A local FastAPI app (`server.py`) + a single static page (`static/index.html`).
Everything it shows is read directly from the files the pipeline scripts
already produce - `data/candles/`, `pattern_library/`, `data/events/`,
`data/heartbeats.json`. Nothing is mocked or simulated; if a panel is
empty, that's because the underlying data doesn't exist yet, not a
placeholder.

## Run it

```bash
pip install -r ../requirements.txt   # or from repo root: pip install -r requirements.txt
python server.py                     # from this directory, or: python dashboard/server.py
```

Open http://localhost:8000. It polls its own API every 8s, so leave it
open as a live monitor.

## Two systems, one dashboard codebase

This same app also serves the ML challenger (`../ml_system/`) - set
`SIGNAL_ENGINE=ml` (plus `DASHBOARD_DATA_DIR`/`DASHBOARD_SHARED_DATA_DIR`/
`ML_REGISTRY_DIR`, see `ml_system/README.md`) to point it at that
system's own journal/model registry instead. Every panel below renders
identically for both, because both systems produce the exact same output
shapes - run one instance per system (different ports, or different
devices) to watch them side by side.

## What each panel is

- **Circuit breaker banner** - a full-width red banner above the Signals
  Engine card, visible ONLY when the portfolio-level circuit breaker
  (`circuit_breaker.py`, see the main README's "Circuit breaker" section)
  is currently tripped - >= 8 consecutive live losses, peak-to-trough
  drawdown worse than -15R, or >= 6R of simultaneous open risk, any one
  is sufficient. Shows which trigger(s) fired plus the current
  streak/drawdown/open-risk numbers, and explains that the signal is
  forced to `HOLD` system-wide until a human reviews and clears it -
  nothing in the pipeline auto-clears a trip. Absent entirely (no gap,
  nothing rendered) when the breaker is healthy.
- **Signals Engine** (hero) - the current live signal, computed the same
  way `signal_engine.py` does (same hard gates, same trade-plan math).
  Not cached - every load/poll recomputes it from the latest candles.
  Includes a freshness badge (FRESH/AGING/EXPIRED - how many trigger-
  candle-durations old the signal is) and, when relevant, a high-impact
  news warning box showing what's scheduled before the trade would
  typically resolve and the pattern's win rate specifically when news
  has landed mid-trade historically (vs the baseline). Confidence is a
  Wilson-score lower-bound win rate (sample-size-aware), not raw
  agreement - see "Critical review" in the main README for why that
  changed. The contributing-patterns table shows every pattern that
  fired, but marks ones not counted toward the signal ("not counted")
  when a stronger, likely-correlated pattern on the same timeframe
  already spoke for that timeframe's vote, and shows the CURRENT market
  regime (volatility x trend, `regime.py`) each pattern is being judged
  against, with a "REGIME ⚠" badge when a pattern's weight was discounted
  for not independently holding up in that specific regime (see
  "Self-healing" in the main README). An amber banner appears above the
  trade plan when self-healing has suspended any pattern/timeframe/
  direction from this signal for currently DECAYING live - the mined
  library may still say it qualifies, but live performance says
  otherwise, so it's excluded until that recovers or the library is
  remined. A "📉 LOSS ATTRIBUTION" box appears the same way when this
  specific signal's context (how many other patterns agreed, which
  trading session is active) matches a bucket already proven, on real
  live evidence, to underperform this pattern's other live trades - see
  the main README's "Loss/win attribution" section - confidence is
  discounted, never fully suppressed, and the exact reason is shown, not
  just the fact that a discount happened. A position-size calculator
  sits under the trade plan - enter
  your account size, risk %, and broker's contract size to get a
  concrete lot size (never auto-applied, account size isn't something
  this system can know on its own).
- **Stat tiles** - total candles stored, how many mined patterns clear
  the 1:4 R:R / 60% gate, fundamental events tracked, and the freshest
  candle's age (your at-a-glance "is this stale" check).
- **Data Collection Engine — Pulse** - reads `data/heartbeats.json`,
  which `build_history.py`, `live_update.py`, `build_fundamentals.py`,
  and `build_pattern_library.py` each write to on every run (`running` ->
  `ok`/`error`, with duration). This is real telemetry from the jobs
  themselves, not a guess from file timestamps. The header pill
  ("all systems nominal" / "data may be stale" / "job failing") is a
  rollup: any job in `error` -> critical; otherwise good, UNLESS one of
  the cron-cadence jobs (`live_update`, `build_fundamentals`,
  `news_calendar`) hasn't run in 6+ hours -> warning. `build_history` and
  `build_pattern_library` are periodic/manual by nature (backfill once,
  rebuild weekly-ish) and deliberately excluded from that check - they'd
  show permanently stale otherwise even when everything's healthy.
- **Data Storage** - per-timeframe candle counts (broken down by source -
  `dukascopy` vs `mt_broker`, if you've set up the MetaTrader bridge),
  date range, file size, straight off the Parquet files, plus the raw
  Dukascopy tick cache size. This is where you can SEE the "Dukascopy is
  always authoritative" guarantee (see mt_bridge/README.md) actually
  holding - broker-sourced candles only ever appear for dates Dukascopy
  doesn't cover. A data-quality box appears here too (only when there's
  something to report) showing backfill hours that genuinely FAILED to
  fetch and any inter-candle gap bigger than a normal weekly close - see
  "Data quality auditing" in the main README for what counts as
  anomalous and why.
- **Realtime Data Collection** - a price sparkline per timeframe (tab to
  switch) with a freshness flag if the latest candle is more than 3 days
  old for that timeframe.
- **Patterns Data** - every mined pattern for the selected timeframe -
  atomic (technical/session/fundamental/support-resistance/smc) AND
  confluence combos (two patterns firing on the same candle, see "Getting
  more signals without lowering the bar" in the main README), filterable
  by category (Technical/Session/Fundamental/Support-Resistance/SMC/Combo)
  or qualifying-only - win rate
  shown as a meter bar (with its Wilson confidence lower bound
  alongside), sorted qualifying-first. The Out-of-Sample column shows
  whether the held-out slice independently clears the gate too - a
  pattern can look good full-sample and still fail here if its edge has
  decayed (see "Critical review" in the main README). Expectancy shows
  gross and cost-adjusted (net of the placeholder spread assumption) R
  side by side. Filter by category (technical/fundamental) or
  qualifying-only - this is a live filter over already-fetched data, no
  extra request per click. The "Why (win vs loss)" column shows the
  features that read measurably differently between a pattern's own
  historical WIN and LOSS occurrences (see "Causal autopsy" in the main
  README) - "—" until `scripts/event_autopsy.py --merge-into-library` has
  been run at least once, since this is a separate, opt-in analysis pass,
  not part of the regular `build_pattern_library.py` mining run.
- **Fundamentals Feed** - event counts by type (CPI/PCE/NFP/GDP/FOMC) and
  the most recent releases.
- **Self-Assessment** - the system grading its own live performance, so
  you don't have to trade on real money to find out if it works: overall
  scorecard (raw + Wilson-conservative live win rate, total realized R
  gross and cost-adjusted, current streak, best/worst trade), an equity
  curve of cumulative R across every resolved trade, and a per-pattern
  credibility table (CONFIRMED / WATCH / DECAYING / UNPROVEN - see
  "Self-assessment" in the main README for exactly what each means and
  why), and a loss/win ATTRIBUTION table one level deeper than the
  credibility table - not "is this pattern working," but "WITHIN this
  pattern's own live trades, does the confluence count or trading
  session it fired in make a statistically real difference" (see the
  main README's "Loss/win attribution" section). A plain-text version of
  all of this is available without the dashboard via
  `python scripts/report.py`, for cron jobs or checking over SSH.
- **Signal Journal** - every signal `live_update.py` has ever logged, its
  status (open/win/loss/expired), and its actual R when resolved. The
  header note shows the live win rate across resolved signals - directly
  comparable to the mined win rate, since both use the same resolution
  logic (`risk_reward.resolve_trade`). A warning banner appears when a
  pattern's live win rate has fallen statistically significantly below
  what was mined for it (`signal_journal.detect_drift`) - your early
  warning that a pattern may be decaying in real time, before the
  library's next rebuild would otherwise catch it.
- **Upcoming High-Impact News** - the forward-looking release calendar
  (`news_calendar.py`), so you can see what's coming without waiting for
  a signal to surface it.
- **Controls** - buttons run the actual pipeline scripts
  (`live_update.py`, `build_pattern_library.py`, `build_fundamentals.py`,
  `news_calendar.py`) as background subprocesses; output streams into the
  console below in real time. Same network caveat as the rest of this
  project: refresh/fundamentals/news-calendar actions need real internet
  access to Dukascopy/FRED to do anything - rebuilding the pattern
  library is local-only and always works. Note that "Refresh Live Data"
  can trigger an automatic rebuild on its own now too, without you
  pressing "Rebuild Pattern Library" - see "Self-healing" in the main
  README; the pulse panel's `build_pattern_library` row will show a
  fresh timestamp even though you only clicked refresh.

## Notes

- Single-user, local-only. No auth - don't expose port 8000 beyond
  localhost/your own network without adding some.
- The action endpoints refuse to start a second run of the same action
  while one is already in flight (409), but there's no queueing beyond
  that - it's a personal dashboard, not a job scheduler.
