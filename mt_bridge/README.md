# MetaTrader bridge (get your actual broker's chart data)

Why this exists: everything else in this project defaults to Dukascopy's
free feed. That's a fine, deep historical source, but it's not
necessarily what YOUR broker prints - different brokers use different
liquidity providers, so OHLC on a given candle can differ slightly
(spread construction, session gaps, occasional requotes). If you mine
patterns on one feed and trade live on another, there's a data-source
mismatch. This bridge exports and streams data straight from your own
MT4/MT5 terminal instead, so mining and live trading use the exact same
source your broker actually gives you.

**Honesty check before you use this:** the `.mq4`/`.mq5` scripts here
were written against stable, documented MetaTrader APIs (`CopyRates`,
`MqlRates`, `TimeGMTOffset`, `FileOpen`/`FileWrite`) but could NOT be
compiled or run anywhere in the environment this was built in - there's
no MetaTrader available there. Unlike the rest of this project (which
was tested end-to-end against synthetic data), these need your own
verification. Do this before trusting a full backfill:
1. Run the export script.
2. Open the resulting CSV in a text editor or Excel.
3. Compare a handful of recent candles (open/high/low/close, and the
   date/time) against what your MT4/5 chart actually shows for the same
   period.
4. Only then run the full backfill / trust the live bridge.

If a script fails to compile at all, the most likely cause is an
unusually old MT4 build that predates "New MQL4" (pre-2014ish) - modern
MT4/5 terminals from virtually every current broker will be fine.

## Setup

**1. One-time historical backfill**

- Open a chart in MT4/5 for your broker's gold symbol - whatever it's
  actually called (`XAUUSD`, `XAUUSD.a`, `GOLD`, `XAUUSD_i`, etc.). The
  scripts use `Symbol()`, so they export whatever chart they're dropped on.
- **Before running the script**, scroll each timeframe's chart back (End,
  then hold Home / scroll repeatedly) until it stops loading older bars.
  MT4/5 only has locally what's been downloaded from the broker's server -
  skipping this step silently gives you less history than your broker
  actually retains.
- Drag `ExportHistoryMT5.mq5` (or `ExportHistoryMT4.mq4`) from the
  Navigator's Scripts list onto the chart.
- Output lands in `<Terminal Data Folder>\MQL5\Files\gold_export\`
  (or `MQL4\Files\gold_export\`). Find your data folder via
  File > Open Data Folder inside the terminal.

**2. Ongoing live updates**

- Attach `LiveBridgeMT5.mq5` (or `LiveBridgeMT4.mq4`) to any one chart of
  the same symbol as an Expert Advisor (drag from Navigator > Expert
  Advisors) and enable AutoTrading. It never places trades - MT4/5 just
  requires AutoTrading on for any EA's timer to run.
- It checks all six tracked timeframes every 30 seconds and appends any
  newly CLOSED candle to `gold_export\<symbol>_<tf>_live.csv`. Leave the
  terminal running for this to keep working.

**3. Import into the pipeline**

```bash
python src/mt_import.py \
  --export-dir "/path/to/MQL5/Files/gold_export" \
  --symbol XAUUSD.a
```

`--symbol` must match exactly what's in the exported filenames (check the
`gold_export` folder if unsure). This writes into
`data/candles/XAUUSD_<tf>.parquet` - the SAME files and format
`build_history.py` produces, so `build_pattern_library.py`,
`signal_engine.py`, and the dashboard all work completely unchanged.
Run it again any time (e.g. from a scheduled task alongside
`live_update.py`) to pull in whatever the live bridge has appended since.

## Timestamp correctness (read this)

MT4/5 record candle times in your BROKER'S SERVER time, which is
commonly NOT UTC (GMT+0, +2, +3 are all common, each with their own DST
convention). Both scripts write the server's live UTC offset - from
MQL's `TimeGMTOffset()`, not a hardcoded guess - as the first line of
every CSV. `mt_import.py` uses that value to convert to UTC and will
REFUSE to import a file that's missing it rather than assume an offset.
If your broker's server DST schedule ever changes, newly-exported files
will automatically carry the new offset; old already-imported data
doesn't need to be redone (it was correct for the offset in effect when
it was exported).

## Mixing with Dukascopy data - Dukascopy is always authoritative

The recommended setup: **`build_history.py` (Dukascopy) stays the deep
mining source, `mt_import.py` (your broker) only ever fills in gaps
Dukascopy doesn't cover** (typically the most recent period, via the
live bridge). This is enforced in code, not just a convention to
remember:

- Every candle is tagged with where it came from (`source`:
  `"dukascopy"` or `"mt_broker"`) when written.
- `build_history.merge_with_existing()` resolves any overlapping
  timestamp by that tag, not by which import happened to run first -
  Dukascopy always wins. This was actually tested both ways: Dukascopy
  landing first (broker import correctly can't overwrite it) AND broker
  data landing first (a later Dukascopy backfill correctly replaces it).
  Older candle files from before this existed have no `source` column;
  those are treated as Dukascopy (it was the only source before this
  bridge existed), so already-collected history stays protected too.
- The dashboard's Data Storage panel shows the source breakdown per
  timeframe (`dukascopy: 2900, mt_broker: 5`, etc.) so you can see this
  holding, not just take it on faith.

Net effect: the deep historical mine is never silently degraded by a
broker's much shorter retention window, while live signals still benefit
from your broker's real-time feed for whatever Dukascopy hasn't caught
up to yet.

### The remaining risk this doesn't fully eliminate, and how it's surfaced

Different brokers construct OHLC slightly differently from Dukascopy's
own feed (different liquidity provider, spread construction, occasional
requotes - see this file's opening paragraph). That's usually harmless,
but candlestick patterns are precise shape/ratio tests (e.g. `hammer`
requires the lower shadow to be >= 2x the body) - so a genuinely
borderline candle COULD register as a pattern on one feed's OHLC and not
the other's. Because Dukascopy always wins once it catches up, this only
ever affects the MOST RECENT candle at the exact moment a signal is
computed (the deep history used for mining is always Dukascopy,
permanently) - but at that exact moment, the newest candle very often
still IS broker-sourced, simply because Dukascopy hasn't published that
hour yet.

This isn't hard-blocked - if it turned out to be a real, systematic
problem for a given pattern, live-drift detection
(`signal_journal.detect_drift`/`suspended_patterns`) already catches and
auto-suspends it the same as any other cause of live underperformance,
regardless of cause. What's added is **transparency at the individual
signal level**: every signal's `freshness.data_source` field
(`signal_engine._trigger_candle_source`) reports whether its specific
trigger candle was `"dukascopy"` (the same feed the pattern's win rate
was mined against) or `"mt_broker"` (not yet Dukascopy-confirmed) - shown
on the dashboard as a `⚠ BROKER FEED, NOT YET DUKASCOPY-CONFIRMED` badge
next to the freshness label when it applies, so you can factor that into
how much to trust a specific, individual signal rather than only finding
out in aggregate after the fact. Verified end-to-end: a real
`compute_signal()` call correctly reports `dukascopy_confirmed: false`
when the trigger candle is broker-sourced and `true` when Dukascopy has
since replaced it (same fix applies to the ML challenger too - it reuses
this exact function, not a parallel implementation), and the dashboard
badge was confirmed to render/disappear correctly in both states via a
real running server and a screenshot of each.
