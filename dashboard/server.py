"""
Local monitoring dashboard for the gold signals pipeline. Reads the
actual data files on disk (candles, pattern_library JSON, fundamentals,
heartbeats) - nothing here is mocked or simulated, it's a window onto
whatever state the pipeline is really in.

Also exposes controls to trigger the pipeline scripts (refresh live
data, rebuild the pattern library, refresh fundamentals) as background
jobs, so you can operate the system from the browser instead of a
terminal.

Run:
    pip install -r requirements.txt
    python dashboard/server.py
Then open http://localhost:8000
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response

# When this server process started - powers the "uptime" figure in
# /api/system_status. Simple, honest, and impossible to fake: if this
# endpoint is answering at all, the process has been up at least this long.
SERVER_START_TIME = datetime.now(timezone.utc)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
ML_SYSTEM = ROOT / "ml_system"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ML_SYSTEM))

import data_quality  # noqa: E402
from heartbeat import read_heartbeats  # noqa: E402
from risk_reward import MIN_RESOLVED_SAMPLES, MIN_WIN_RATE, RR_RATIO  # noqa: E402

# One dashboard codebase serves EITHER system, picked by env var - so
# both the rule-based and ML challenger can run their own instance
# (same machine or different devices) while reusing the exact same UI
# and self-assessment machinery, rather than maintaining two dashboards.
# See ml_system/README.md for how the two systems compare/merge later.
ENGINE = os.environ.get("SIGNAL_ENGINE", "rules")  # "rules" or "ml"
SYMBOL = os.environ.get("DASHBOARD_SYMBOL", "XAUUSD")

# This dashboard instance's OWN state - journal, heartbeats, pulse.
# "rules": data/ (the rule-based system IS its own data root).
# "ml": ml_data/ (kept separate from the rule-based system's data/, but
# reads candles/fundamentals from it - see SHARED_DATA_DIR below).
DATA_DIR = Path(os.environ.get("DASHBOARD_DATA_DIR", str(ROOT / ("ml_data" if ENGINE == "ml" else "data"))))

# Read-only market data shared between both systems - candles, raw tick
# cache, fundamentals, data-quality report, forward news calendar. There
# is no reason for the ML challenger to run its own Dukascopy backfill
# just to read the same ticks the rule-based system already fetched.
SHARED_DATA_DIR = Path(os.environ.get("DASHBOARD_SHARED_DATA_DIR", str(ROOT / "data")))

ML_REGISTRY_DIR = Path(os.environ.get("ML_REGISTRY_DIR", str(ROOT / "ml_registry")))

# Reference "mined stats" source - pattern_library/*.json for the
# rule-based system, ML_REGISTRY_DIR/lib_view/*.json
# (model_registry.write_lib_view) for the ML challenger, so
# drift-detection/credibility scoring (signal_journal.py, unmodified
# either way) has something to compare live performance against.
# Deliberately DERIVED from ML_REGISTRY_DIR (not independently defaulted
# off ROOT) so overriding ML_REGISTRY_DIR alone still points this at the
# right place - a mismatch here would silently show "0 qualifying
# patterns" even when the registry has a perfectly good active model.
LIB_DIR = Path(os.environ.get(
    "DASHBOARD_LIB_DIR",
    str(ML_REGISTRY_DIR / "lib_view") if ENGINE == "ml" else str(ROOT / "pattern_library"),
))

# Pattern Discovery Engine output (discover_patterns.py) - self-learned
# patterns, kept in their own directory rather than merged into LIB_DIR on
# disk (see discover_patterns.py's own docstring for why: two independent
# mining scripts must never race writing the same file). Rule-based engine
# only - the ML challenger's own "mined view" is model predictions
# (ml_registry/lib_view/), not a primitive-conjunction library, so there is
# nothing for discover_patterns.py to contribute there.
DISCOVERED_DIR = None if ENGINE == "ml" else Path(
    os.environ.get("DASHBOARD_DISCOVERED_DIR", str(ROOT / "discovered_patterns"))
)

app = FastAPI(title=f"Gold Signals Dashboard ({'ML Challenger' if ENGINE == 'ml' else 'Rule-Based'})")

# ---- background job runner (for the dashboard's action buttons) -----------

JOBS: dict[str, dict] = {}

if ENGINE == "ml":
    JOB_SCRIPTS = {
        "refresh_live": [sys.executable, str(ML_SYSTEM / "ml_live_update.py"),
                          "--candles-dir", str(SHARED_DATA_DIR / "candles"),
                          "--news-data-dir", str(SHARED_DATA_DIR),
                          "--data-dir", str(DATA_DIR), "--registry-dir", str(ML_REGISTRY_DIR)],
        "rebuild_library": [sys.executable, str(ML_SYSTEM / "train.py"),
                             "--candles-dir", str(SHARED_DATA_DIR / "candles"),
                             "--data-dir", str(DATA_DIR), "--registry-dir", str(ML_REGISTRY_DIR)],
        "refresh_fundamentals": [sys.executable, str(SRC / "build_fundamentals.py"), "--data-dir", str(SHARED_DATA_DIR)],
        "refresh_news_calendar": [sys.executable, str(SRC / "news_calendar.py"), "--data-dir", str(SHARED_DATA_DIR)],
    }
else:
    JOB_SCRIPTS = {
        "refresh_live": [sys.executable, str(SRC / "live_update.py")],
        "rebuild_library": [sys.executable, str(SRC / "build_pattern_library.py")],
        "discover_patterns": [sys.executable, str(SRC / "discover_patterns.py"),
                               "--out-dir", str(DISCOVERED_DIR)],
        "refresh_fundamentals": [sys.executable, str(SRC / "build_fundamentals.py")],
        "refresh_news_calendar": [sys.executable, str(SRC / "news_calendar.py")],
    }


def _run_job(job_id: str, cmd: list[str]) -> None:
    job = JOBS[job_id]
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            job["output"].append(line.rstrip("\n"))
            job["output"] = job["output"][-500:]
        proc.wait()
        job["status"] = "done" if proc.returncode == 0 else "error"
        job["returncode"] = proc.returncode
    except Exception as e:
        job["status"] = "error"
        job["output"].append(f"launcher error: {e}")
    job["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/actions/import_mt_data")
def trigger_mt_import(export_dir: str, symbol: str):
    # NOTE: this route MUST be registered before /api/actions/{action}
    # below - FastAPI/Starlette matches routes in registration order, and
    # {action} would otherwise greedily capture "import_mt_data" as a
    # path param and 404 on it (JOB_SCRIPTS has no such key), never
    # reaching this handler. Verified this was a real bug, not a
    # hypothetical: it 404'd until this was moved above the generic route.
    action = "import_mt_data"
    if any(j["action"] == action and j["status"] == "running" for j in JOBS.values()):
        raise HTTPException(409, f"{action} is already running")
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "action": action, "status": "running", "output": [],
        "started_at": datetime.now(timezone.utc).isoformat(), "returncode": None,
    }
    cmd = [sys.executable, str(SRC / "mt_import.py"), "--export-dir", export_dir, "--symbol", symbol]
    threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/actions/{action}")
def trigger_action(action: str):
    if action not in JOB_SCRIPTS:
        raise HTTPException(404, f"unknown action {action}")
    if any(j["action"] == action and j["status"] == "running" for j in JOBS.values()):
        raise HTTPException(409, f"{action} is already running")
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "action": action, "status": "running", "output": [],
        "started_at": datetime.now(timezone.utc).isoformat(), "returncode": None,
    }
    threading.Thread(target=_run_job, args=(job_id, JOB_SCRIPTS[action]), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/actions/{job_id}/status")
def action_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    return job


@app.get("/api/actions")
def list_actions():
    return sorted(JOBS.items(), key=lambda kv: kv[1]["started_at"], reverse=True)[:20]


# ---- read-only data endpoints ----------------------------------------------

def _candle_paths() -> list[Path]:
    d = SHARED_DATA_DIR / "candles"
    return sorted(d.glob(f"{SYMBOL}_*.parquet")) if d.exists() else []


def _tf_from_path(p: Path) -> str:
    return p.stem.replace(f"{SYMBOL}_", "")


@app.get("/api/storage")
def storage():
    out = {}
    for p in _candle_paths():
        tf = _tf_from_path(p)
        df = pd.read_parquet(p)
        by_source = (
            df["source"].value_counts().to_dict() if "source" in df.columns
            else {"dukascopy": len(df)}  # pre-source-tracking files were Dukascopy-only
        )
        # Which feed the SINGLE NEWEST candle came from - a different
        # question from by_source's aggregate counts above ("mostly
        # Dukascopy, mostly broker") - this is specifically "is live data
        # flowing in right now, and from where," which /api/system_status
        # surfaces across all timeframes at once. Computed here (not a
        # second read of the same file in system_status()) since `df` is
        # already loaded for this exact purpose.
        latest_source = (
            df.loc[df["timestamp"].idxmax(), "source"] if "source" in df.columns and len(df) else
            ("dukascopy" if len(df) else None)
        )
        out[tf] = {
            "candles": len(df),
            "start": df["timestamp"].min().isoformat() if len(df) else None,
            "end": df["timestamp"].max().isoformat() if len(df) else None,
            "size_bytes": p.stat().st_size,
            "by_source": by_source,
            "latest_source": latest_source,
        }
    raw_dir = SHARED_DATA_DIR / "raw_bi5"
    raw_files = list(raw_dir.rglob("*.bi5")) if raw_dir.exists() else []
    out["_raw_cache"] = {
        "files": len(raw_files),
        "size_bytes": sum(f.stat().st_size for f in raw_files),
    }
    return out


def _system_status(storage_data: dict | None = None) -> dict:
    """`storage_data`: pass an already-computed storage() result when the
    caller has one (overview() below does, every poll) so this doesn't
    trigger a SECOND full read of every candle file just to look up
    latest_source - storage() already read them all for its own
    response. Standalone callers (the /api/system_status route itself)
    pass None and this computes it fresh."""
    if storage_data is None:
        storage_data = storage()
    now = datetime.now(timezone.utc)

    freshest_end = None
    freshest_tf = None
    for p in _candle_paths():
        # Only the timestamp column here - just to find WHICH timeframe
        # holds the overall-freshest candle without a full read of every
        # file; latest_source for the winning timeframe comes from
        # storage_data above, not a second read of that file either.
        ts_col = pd.read_parquet(p, columns=["timestamp"])
        if ts_col.empty:
            continue
        end = ts_col["timestamp"].max()
        if freshest_end is None or end > freshest_end:
            freshest_end, freshest_tf = end, _tf_from_path(p)

    freshest_source = None
    if freshest_tf is not None:
        freshest_source = storage_data.get(freshest_tf, {}).get("latest_source")

    return {
        "engine": ENGINE,
        "symbol": SYMBOL,
        "server_started_at": SERVER_START_TIME.isoformat(),
        "uptime_seconds": (now - SERVER_START_TIME).total_seconds(),
        "data_dir": str(DATA_DIR),
        "shared_data_dir": str(SHARED_DATA_DIR),
        "lib_dir": str(LIB_DIR),
        "freshest_candle": ({
            "timeframe": freshest_tf,
            "timestamp": freshest_end.isoformat(),
            "source": freshest_source,
        } if freshest_end is not None else None),
    }


@app.get("/api/system_status")
def system_status():
    """Broader health view than /api/pulse's 4 tracked jobs alone -
    this dashboard process's own uptime (proof of life beyond "the page
    loaded"), which engine/data directories this instance is actually
    configured against (easy to lose track of when running both engines
    side by side on different ports), and which timeframe holds the
    single freshest candle across the WHOLE system plus which feed it
    came from - the practical "is live data actually flowing right now,
    and from where" question that /api/storage's per-timeframe table
    otherwise makes you piece together yourself."""
    return _system_status()


def _dir_summary(path: Path, glob_pattern: str = "**/*") -> dict:
    """File count + total size for `path` - a single file, a directory
    (recursively globbed), or nonexistent. Nonexistent is a NORMAL state
    here, not an error: most of these categories only appear on disk
    after their owning script has run at least once (e.g. data/context/
    before build_fundamentals.py, ml_registry/ before any ML training)."""
    if not path.exists():
        return {"exists": False, "files": 0, "size_bytes": 0}
    if path.is_file():
        return {"exists": True, "files": 1, "size_bytes": path.stat().st_size}
    files = [f for f in path.glob(glob_pattern) if f.is_file()]
    return {"exists": True, "files": len(files), "size_bytes": sum(f.stat().st_size for f in files)}


@app.get("/api/files_overview")
def files_overview():
    """Full on-disk footprint of the pipeline - not just the per-
    timeframe candle breakdown /api/storage already shows, but every
    category of file this system writes: raw tick cache, pattern
    library, signal journal, fundamentals/context series, heartbeats,
    the data quality report, and (ML engine) the model registry.
    Deliberately NOT part of /api/overview's 8-second poll cycle - the
    raw tick cache alone can be tens of thousands of files after a full
    20-year backfill, and re-globbing that on every poll would be
    wasted, avoidable disk I/O for a number that barely changes minute
    to minute. The frontend fetches this on demand instead (page load
    plus a manual refresh button)."""
    sections = []

    def add(label: str, path: Path, pattern: str = "**/*"):
        sections.append({"label": label, "path": str(path), **_dir_summary(Path(path), pattern)})

    add("Candles (all timeframes)", SHARED_DATA_DIR / "candles", "*.parquet")
    add("Raw Dukascopy tick cache", SHARED_DATA_DIR / "raw_bi5", "**/*.bi5")
    add("Fundamentals & news calendar", SHARED_DATA_DIR / "events")
    add("Context series (DXY, real yields)", SHARED_DATA_DIR / "context")
    add("Pattern library (mined stats)", LIB_DIR, "*.json")
    if DISCOVERED_DIR is not None:
        add("Discovered patterns (self-learned)", DISCOVERED_DIR, "*.json")
    add("Signal journal", DATA_DIR / "signal_journal.parquet")
    add("Heartbeats / job status", DATA_DIR / "heartbeats.json")
    add("Data quality report", SHARED_DATA_DIR / "data_quality_report.json")
    if ENGINE == "ml":
        add("ML model registry", ML_REGISTRY_DIR)
    logs_dir = ROOT / "logs"
    if logs_dir.exists():
        add("Logs", logs_dir)

    return {
        "sections": sections,
        "total_files": sum(s["files"] for s in sections),
        "total_bytes": sum(s["size_bytes"] for s in sections),
    }


@app.get("/api/candles/{tf}")
def candles(tf: str, limit: int = 300):
    p = SHARED_DATA_DIR / "candles" / f"{SYMBOL}_{tf}.parquet"
    if not p.exists():
        raise HTTPException(404, f"no candle data for {tf}")
    df = pd.read_parquet(p).sort_values("timestamp").tail(limit)
    df["timestamp"] = df["timestamp"].astype(str)
    return json.loads(df.to_json(orient="records"))


def _pattern_category(name: str) -> str:
    if name.startswith("ml_model"):  # covers both the old fixed name and "ml_model_rr<tag>" tiers
        return "ml"
    if name.startswith("discovered__"):
        return "discovered"
    if name.startswith("combo__"):
        return "combo"
    if name.startswith("fundamental_"):
        return "fundamental"
    if name.startswith("session_"):
        return "session"
    if name.startswith("sr_"):
        return "support_resistance"
    if name.startswith("smc_"):
        return "smc"
    return "technical"


def _load_library(tf: str) -> dict:
    """`pattern_library/<symbol>_<tf>.json` merged with `discovered_patterns/
    <symbol>_<tf>.json` (if any) - the SAME merge signal_engine.load_inputs()
    does, so every dashboard view of "the pattern library" (overview,
    detail table, category counts, storage footprint) shows discovered
    patterns too, not just what build_pattern_library.py mined. Missing
    discovered_patterns/ entirely is normal (no discovery run yet), not
    an error."""
    path = LIB_DIR / f"{SYMBOL}_{tf}.json"
    lib = json.loads(path.read_text()) if path.exists() else {}
    if DISCOVERED_DIR is not None:
        discovered_path = DISCOVERED_DIR / f"{SYMBOL}_{tf}.json"
        if discovered_path.exists():
            lib.update(json.loads(discovered_path.read_text()))
    return lib


def _known_timeframes() -> list[str]:
    """Every timeframe with EITHER a pattern_library file OR a
    discovered_patterns file - a discovery run can in principle finish
    for a timeframe before/after build_pattern_library.py has, so neither
    directory alone is guaranteed to list every timeframe that actually
    has something to show."""
    tfs = {p.stem.replace(f"{SYMBOL}_", "") for p in LIB_DIR.glob(f"{SYMBOL}_*.json")}
    if DISCOVERED_DIR is not None:
        tfs |= {p.stem.replace(f"{SYMBOL}_", "") for p in DISCOVERED_DIR.glob(f"{SYMBOL}_*.json")}
    return sorted(tfs)


@app.get("/api/patterns")
def patterns_overview():
    out = {}
    for tf in _known_timeframes():
        lib = _load_library(tf)
        entries = {k: v for k, v in lib.items() if not k.startswith("_")}
        qualifying = []
        for name, entry in entries.items():
            if "stats" in entry and entry["stats"].get("qualifies"):
                qualifying.append(name)
            elif entry.get("as_long", {}).get("qualifies") or entry.get("as_short", {}).get("qualifies"):
                qualifying.append(name)
        by_category: dict[str, int] = {}
        for n in entries:
            cat = _pattern_category(n)
            by_category[cat] = by_category.get(cat, 0) + 1
        out[tf] = {
            "total_patterns": len(entries),
            "qualifying": qualifying,
            "qualifying_count": len(qualifying),
            "by_category": by_category,
        }
    return out


@app.get("/api/patterns/{tf}")
def patterns_detail(tf: str):
    lib = _load_library(tf)
    if not lib:
        raise HTTPException(404, f"no pattern library for {tf}")
    rows = []
    for name, entry in lib.items():
        if name.startswith("_"):
            continue
        variants = []
        if "stats" in entry:
            variants.append((entry["direction"], entry["stats"]))
        else:
            if "as_long" in entry:
                variants.append(("bullish", entry["as_long"]))
            if "as_short" in entry:
                variants.append(("bearish", entry["as_short"]))
        why_entry = entry.get("why")
        for direction, stats in variants:
            oos = stats.get("out_of_sample") or {}
            # `why_entry` mirrors `entry` itself: flat (has its own
            # "significant_factors") for a directional pattern, or
            # {"as_long":..., "as_short":...} for an ambiguous one - see
            # causal_autopsy.autopsy_pattern()'s output shape, which
            # scripts/event_autopsy.py --merge-into-library writes
            # verbatim into this same library entry under "why". Picks
            # the side matching THIS variant's own direction, same as
            # `stats` above does for win-rate.
            why = None
            if why_entry:
                if "significant_factors" in why_entry:
                    why = why_entry
                elif direction == "bullish":
                    why = why_entry.get("as_long")
                elif direction == "bearish":
                    why = why_entry.get("as_short")
            rows.append({
                "pattern": name,
                "category": _pattern_category(name),
                "combo_of": entry.get("combo_of"),
                "direction": direction,
                "win_rate": stats.get("win_rate"),
                "win_rate_wilson_lower": stats.get("win_rate_wilson_lower"),
                "resolved": stats.get("resolved", 0),
                "samples": stats.get("samples", 0),
                "expectancy_r": stats.get("expectancy_r"),
                "expectancy_r_after_costs": stats.get("expectancy_r_after_costs"),
                "oos_win_rate": oos.get("win_rate"),
                "oos_resolved": oos.get("resolved", 0),
                "oos_qualifies": oos.get("qualifies", False),
                "qualifies": stats.get("qualifies", False),
                "why": ({"n_significant": why["n_significant"],
                         "significant_factors": why["significant_factors"][:5]}
                        if why and why.get("n_significant") else None),
            })
    rows.sort(key=lambda r: (r["qualifies"], r["win_rate"] or 0), reverse=True)
    return {"timeframe": tf, "rr_ratio": RR_RATIO, "min_win_rate": MIN_WIN_RATE,
            "min_resolved_samples": MIN_RESOLVED_SAMPLES, "rows": rows}


# ---- Pattern Discovery Engine - its own dedicated section --------------------
#
# Everything below is scoped ONLY to discovered_patterns/ - the self-learned
# patterns discover_patterns.py mines from raw primitives, never the hand-
# picked pattern_library/ catalog. Deliberately treated as its own
# quasi-independent system with its own signal/performance/patterns views
# (same "run it separately, see how it does on its own" philosophy the ML
# challenger already gets its own dashboard for), while ALSO already being
# merged into the main blended /api/signal (see signal_engine.load_inputs) -
# this section exists to make that self-learned contribution visible and
# auditable on its own terms, not to replace the blended view.
#
# Rule-based engine only (ENGINE != "ml") - the ML challenger has no
# primitive library to discover patterns from (DISCOVERED_DIR is None
# there), so every route below reports {"available": false} rather than
# erroring, the same fail-open convention used throughout this dashboard.


def _discovery_patterns_raw(tf: str) -> dict:
    """Reads discovered_patterns/<symbol>_<tf>.json DIRECTLY - not merged
    with pattern_library/ (see _load_library) - so the full discovery_meta
    provenance (primitives, era scores, p-value, confirmation slice) is
    still there to show. Empty dict if no discovery run has happened yet
    for this timeframe, not an error."""
    if DISCOVERED_DIR is None:
        return {}
    path = DISCOVERED_DIR / f"{SYMBOL}_{tf}.json"
    return json.loads(path.read_text()) if path.exists() else {}


@app.get("/api/discovery/summary")
def discovery_summary():
    if DISCOVERED_DIR is None:
        return {"available": False, "reason": "no Pattern Discovery Engine on the ML challenger - "
                                                "its own mined view is model predictions, not a "
                                                "primitive-conjunction library"}
    by_tf = {}
    total, qualifying, best_win_rate = 0, 0, None
    for tf in _known_timeframes():
        raw = _discovery_patterns_raw(tf)
        n = len(raw)
        q = sum(1 for e in raw.values() if e.get("stats", {}).get("qualifies"))
        wrs = [e["stats"]["win_rate"] for e in raw.values() if e.get("stats", {}).get("win_rate") is not None]
        by_tf[tf] = {"total": n, "qualifying": q, "best_win_rate": max(wrs) if wrs else None}
        total += n
        qualifying += q
        if wrs:
            best_win_rate = max(best_win_rate or 0, max(wrs))
    return {
        "available": True, "by_timeframe": by_tf,
        "total_discovered": total, "total_qualifying": qualifying,
        "best_win_rate": best_win_rate,
    }


@app.get("/api/discovery/patterns")
def discovery_patterns():
    """Full detail listing (not the collapsed pattern_library-shaped rows
    /api/patterns/{tf} returns) - every discovered pattern's own
    provenance: which primitives/families built it, its worst-era score,
    per-era win rate/sample counts, the FDR p-value/threshold it cleared,
    and the blind confirmation-slice result that was its actual final
    exam. This is the "no human made, nothing - show your work" view."""
    if DISCOVERED_DIR is None:
        return {"available": False, "by_timeframe": {}}
    by_tf = {}
    for tf in _known_timeframes():
        raw = _discovery_patterns_raw(tf)
        rows = []
        for name, entry in raw.items():
            stats = entry.get("stats", {})
            meta = entry.get("discovery_meta", {})
            confirmation = meta.get("confirmation_slice", {})
            cross_tf = meta.get("cross_timeframe")
            synth_exprs = meta.get("synthesized_expressions", {})
            rows.append({
                "pattern": name,
                "direction": entry.get("direction"),
                "win_rate": stats.get("win_rate"),
                "win_rate_wilson_lower": stats.get("win_rate_wilson_lower"),
                "resolved": stats.get("resolved", 0),
                "expectancy_r": stats.get("expectancy_r"),
                "qualifies": stats.get("qualifies", False),
                "primitives": meta.get("primitives", []),
                "families": meta.get("families", []),
                "worst_era_score": meta.get("discovery_worst_era_score"),
                "era_scores": meta.get("discovery_era_scores", []),
                "era_samples": meta.get("discovery_era_samples", []),
                "p_value": meta.get("p_value"),
                "bh_threshold": meta.get("bh_threshold"),
                "n_tested_this_run": meta.get("n_tested_this_run"),
                "confirmation_win_rate": confirmation.get("win_rate"),
                "confirmation_resolved": confirmation.get("resolved", 0),
                "confirmation_qualifies": confirmation.get("qualifies", False),
                # Cross-timeframe confirmation (discover_patterns._cross_timeframe_confirm) -
                # None means "no sibling timeframe was available to check," NOT
                # "checked and failed" - see signal_engine._cross_timeframe_mismatch.
                "cross_timeframe": cross_tf,
                # Any component built by Layer 0's genetic synthesis
                # (discovery_synthesis.py) rather than the hand-designed
                # catalog - surfaced so "no human made, nothing" is
                # actually verifiable, not just claimed.
                "synthesized_primitives": list(synth_exprs.keys()),
                "synthesized_expressions": synth_exprs,
            })
        rows.sort(key=lambda r: (r["qualifies"], r["win_rate"] or 0), reverse=True)
        by_tf[tf] = rows
    return {"available": True, "by_timeframe": by_tf}


@app.get("/api/discovery/performance")
def discovery_performance():
    """Same self-assessment machinery /api/performance uses
    (signal_journal.pattern_scorecard/equity_curve), filtered to ONLY
    discovered__-prefixed patterns - how the self-learned patterns are
    ACTUALLY doing live, not just what they scored at discovery time.
    Equity curve is recomputed over the journal subset (not sliced from
    the blended curve) so cumulative R here means "just these patterns'
    own trades," matching what the scorecard rows above it show."""
    if DISCOVERED_DIR is None:
        return {"available": False, "scorecard": [], "equity_curve": []}
    from signal_journal import equity_curve, load_journal, pattern_scorecard
    j = load_journal(DATA_DIR)
    scorecard = [r for r in pattern_scorecard(j, LIB_DIR, SYMBOL, DISCOVERED_DIR)
                 if r["pattern"].startswith("discovered__")]
    j_discovered = j[j["pattern"].str.startswith("discovered__")] if not j.empty else j
    return {"available": True, "scorecard": scorecard, "equity_curve": equity_curve(j_discovered)}


@app.get("/api/discovery/signal")
def discovery_signal():
    """What the self-learned patterns ALONE would say right now - computed
    with a library restricted to ONLY discovered__ entries (the blended
    /api/signal already includes them alongside pattern_library/'s hand-
    picked patterns; this is the standalone read, same "run it separately,
    see how it does on its own" comparison the ML challenger already gets
    its own dashboard for)."""
    if DISCOVERED_DIR is None:
        return {"available": False, "direction": "UNAVAILABLE", "confidence": 0,
                "trade_plan": None, "freshness": None, "news_risk": None, "contributions": []}
    from circuit_breaker import check_circuit_breaker
    from signal_engine import compute_signal, load_inputs, load_suspended
    from signal_journal import load_journal
    candles_by_tf, library_by_tf, events, upcoming = load_inputs(
        SYMBOL, SHARED_DATA_DIR, LIB_DIR, discovered_dir=DISCOVERED_DIR,
    )
    library_by_tf = {
        tf: {name: entry for name, entry in lib.items() if name.startswith("discovered__")}
        for tf, lib in library_by_tf.items()
    }
    library_by_tf = {tf: lib for tf, lib in library_by_tf.items() if lib}
    if not candles_by_tf or not library_by_tf:
        return {"available": True, "direction": "UNAVAILABLE", "confidence": 0, "trade_plan": None,
                "freshness": None, "news_risk": None, "contributions": [],
                "reason": "no discovered patterns yet - run discover_patterns.py"}
    suspended = load_suspended(SHARED_DATA_DIR, LIB_DIR, SYMBOL, DISCOVERED_DIR)
    journal = load_journal(DATA_DIR)
    breaker = check_circuit_breaker(journal)
    sig = compute_signal(candles_by_tf, library_by_tf, events, upcoming,
                         suspended=suspended, circuit_breaker=breaker, journal=journal)
    return {"available": True, **sig.to_dict()}


# ---- Explored Setups (scripts/explore_setups.py) - raw research leads ------
#
# READ-ONLY section: scripts/explore_setups.py runs the same beam search as
# the Pattern Discovery Engine above but with the win-rate pruning opened up
# and no FDR/blind-confirmation gate, so it surfaces far more candidates -
# none of them validated, all of them written to explored_setups/ for a
# human to review (see that script's own module docstring for the full
# reasoning). Deliberately NOT wired into JOB_SCRIPTS/the Controls panel's
# trigger buttons - a 1min/5min run can take a long time and should be a
# deliberate foreground/supervised decision, not one click in a browser tab.
# This section only ever reads whatever explore_setups.py has already
# written to disk. Rule-based engine only, same reasoning as DISCOVERED_DIR
# above - explore_setups.py mines discovery_primitives.py's catalog, which
# has no ML-challenger equivalent.
EXPLORED_DIR = None if ENGINE == "ml" else Path(
    os.environ.get("DASHBOARD_EXPLORED_DIR", str(ROOT / "explored_setups"))
)


def _explored_raw(tf: str) -> "dict | None":
    if EXPLORED_DIR is None:
        return None
    path = EXPLORED_DIR / f"{SYMBOL}_{tf}.json"
    return json.loads(path.read_text()) if path.exists() else None


@app.get("/api/explored/summary")
def explored_summary():
    if EXPLORED_DIR is None:
        return {"available": False, "reason": "no explore_setups.py output on the ML challenger - "
                                                "it mines discovery_primitives.py's catalog, which has "
                                                "no ML-challenger equivalent"}
    by_tf = {}
    for path in sorted(EXPLORED_DIR.glob(f"{SYMBOL}_*.json")):
        tf = path.stem.replace(f"{SYMBOL}_", "")
        raw = _explored_raw(tf) or {}
        meta = raw.get("_meta", {})
        setups = raw.get("setups", [])
        by_tf[tf] = {
            "n_tested": meta.get("n_tested"),
            "n_conjunctions_with_enough_samples": meta.get("n_conjunctions_with_enough_samples"),
            "n_kept": len(setups),
            "technicals_only": meta.get("technicals_only", False),
            "exclude_news_window": meta.get("exclude_news_window", False),
            "rr_ratios": meta.get("rr_ratios", []),
            "best_expectancy_r_after_costs": setups[0]["best_expectancy_r_after_costs"] if setups else None,
        }
    return {"available": True, "by_timeframe": by_tf}


@app.get("/api/explored/setups/{tf}")
def explored_setups(tf: str):
    if EXPLORED_DIR is None:
        return {"available": False, "setups": [], "meta": {}}
    raw = _explored_raw(tf)
    if raw is None:
        return {"available": True, "setups": [], "meta": {},
                "reason": f"no explored_setups/{SYMBOL}_{tf}.json yet - run scripts/explore_setups.py"}
    return {"available": True, "setups": raw.get("setups", []), "meta": raw.get("_meta", {})}


# ---- Event Autopsy (scripts/event_autopsy.py) - win/loss factor analysis --
#
# READ-ONLY, same "this dashboard section only ever reads what the script
# already wrote to disk" convention as Explored Setups above. Unlike
# Explored Setups, NOT gated to the rule-based engine only - event_autopsy.py
# detects events with patterns.py/support_resistance.py's existing
# detectors but analyzes factors using ml_system/features.py's feature
# table, so it's a genuinely separate, engine-independent analysis
# category, available regardless of which ENGINE this dashboard instance
# is configured for.
EVENT_AUTOPSY_DIR = Path(os.environ.get("DASHBOARD_EVENT_AUTOPSY_DIR", str(ROOT / "event_autopsy")))


def _event_autopsy_raw(tf: str) -> "dict | None":
    path = EVENT_AUTOPSY_DIR / f"{SYMBOL}_{tf}.json"
    return json.loads(path.read_text()) if path.exists() else None


@app.get("/api/event_autopsy/summary")
def event_autopsy_summary():
    by_tf = {}
    for path in sorted(EVENT_AUTOPSY_DIR.glob(f"{SYMBOL}_*.json")):
        tf = path.stem.replace(f"{SYMBOL}_", "")
        raw = _event_autopsy_raw(tf) or {}
        events = raw.get("events", {})
        by_tf[tf] = {
            "rr_ratio": raw.get("rr_ratio"), "fdr_alpha": raw.get("fdr_alpha"),
            "events": {
                name: {
                    "direction": ev.get("direction"), "n_resolved": ev.get("n_resolved"),
                    "win_rate": ev.get("win_rate"), "n_significant": ev.get("n_significant"),
                    "n_features_tested": ev.get("n_features_tested"),
                    "top_factor": (ev.get("significant_factors") or [None])[0],
                }
                for name, ev in events.items()
            },
        }
    return {"available": bool(by_tf), "by_timeframe": by_tf}


@app.get("/api/event_autopsy/{tf}")
def event_autopsy_detail(tf: str):
    raw = _event_autopsy_raw(tf)
    if raw is None:
        return {"available": False, "events": {},
                "reason": f"no event_autopsy/{SYMBOL}_{tf}.json yet - run scripts/event_autopsy.py"}
    return {"available": True, **raw}


@app.get("/api/fundamentals")
def fundamentals():
    path = SHARED_DATA_DIR / "events" / "fundamentals.parquet"
    if not path.exists():
        return {"configured": False, "events": 0, "by_type": {}, "last_event": None}
    df = pd.read_parquet(path)
    return {
        "configured": True,
        "events": len(df),
        "by_type": df["event_type"].value_counts().to_dict(),
        "last_event": df["datetime_utc"].max().isoformat() if len(df) else None,
        "recent": json.loads(
            df.sort_values("datetime_utc").tail(10)
            .assign(datetime_utc=lambda d: d["datetime_utc"].astype(str))
            [["datetime_utc", "event_type", "value", "change"]]
            .to_json(orient="records")
        ),
    }


@app.get("/api/data_quality")
def data_quality_endpoint():
    report = data_quality.load_report(SHARED_DATA_DIR)
    if report is None:
        return {"available": False, "reason": "no data_quality_report.json yet - run build_history.py"}
    return {"available": True, **report}


@app.get("/api/pulse")
def pulse():
    heartbeats = read_heartbeats(DATA_DIR / "heartbeats.json")
    now = datetime.now(timezone.utc)
    out = {}
    for job, hb in heartbeats.items():
        ts = datetime.fromisoformat(hb["timestamp_utc"])
        out[job] = {**hb, "seconds_ago": (now - ts).total_seconds()}
    return out


def _ml_signals() -> list[dict]:
    """Every R:R tier's own independent signal this run - see
    live_signal.compute_ml_signal()'s module docstring for why this is a
    list, not one blended dict, now that multi-tier search (train.py's
    RR_GRID) exists. Empty list (not a single "UNAVAILABLE" dict) when
    there's no candle data yet - callers decide how to present that."""
    import live_signal
    from circuit_breaker import check_circuit_breaker
    from news_calendar import load_upcoming
    from signal_journal import load_journal

    candles_by_tf = {}
    for p in _candle_paths():
        tf = _tf_from_path(p)
        candles_by_tf[tf] = pd.read_parquet(p).sort_values("timestamp").tail(300).reset_index(drop=True)
    if not candles_by_tf:
        return []
    suspended = live_signal.load_suspended_ml(DATA_DIR, ML_REGISTRY_DIR, SYMBOL)
    journal = load_journal(DATA_DIR)
    breaker = check_circuit_breaker(journal)
    upcoming = load_upcoming(SHARED_DATA_DIR)
    return live_signal.compute_ml_signal(candles_by_tf, ML_REGISTRY_DIR, SYMBOL, upcoming=upcoming,
                                          suspended=suspended, circuit_breaker=breaker, journal=journal)


def _ml_signal() -> dict:
    """ONE representative signal for the hero card / /api/signal, which
    every existing renderer (renderHero, the Overview tab) expects as a
    single dict - unchanged contract for them. Picks whichever tier is
    actually actionable with the HIGHEST confidence right now (a trader
    opening the dashboard wants to see the strongest live call, not an
    arbitrary tier); if nothing is actionable this run, prefers the
    standard 1:4 tier's HOLD (the most directly comparable to the
    rule-based system's own fixed-R:R signal) so the hero card still
    shows something familiar. The FULL per-tier breakdown (every tier,
    not just this one pick) is at /api/signal/tiers - see
    render_ml_tiers() in the frontend."""
    signals = _ml_signals()
    if not signals:
        return {"direction": "UNAVAILABLE", "confidence": 0, "trade_plan": None,
                "freshness": None, "news_risk": None,
                "contributions": [], "reason": "no candle data yet - run build_history.py"}
    actionable = [s for s in signals if s.get("direction") in ("BUY", "SELL")]
    if actionable:
        return max(actionable, key=lambda s: s.get("confidence") or 0)
    standard = next((s for s in signals if s.get("risk_reward") == "1:4"), None)
    return standard or signals[0]


@app.get("/api/signal/tiers")
def signal_tiers():
    """Every R:R tier's own independent signal, for a dashboard view that
    shows scalp/standard/swing tiers side by side instead of collapsing
    them into the single hero-card pick /api/signal makes. For the
    rule-based engine (ENGINE != "ml", which has no tier concept - it's
    fixed at 1:4) this is just a single-item list wrapping /api/signal's
    normal result, so the frontend can call this endpoint unconditionally
    regardless of which engine the dashboard is pointed at."""
    if ENGINE == "ml":
        signals = _ml_signals()
        return signals if signals else [{
            "direction": "UNAVAILABLE", "confidence": 0, "trade_plan": None,
            "freshness": None, "news_risk": None, "contributions": [],
            "reason": "no candle data yet - run build_history.py",
        }]
    return [signal()]


# ---- ML backtest report (ml_system/backtest_report.py) - built from the
# out-of-fold validation trades train.py persists per model version, NOT
# a re-simulation - see that module's own docstring. ML challenger only
# (ENGINE == "ml"): the rule-based engine's equivalent "backtest" is
# already the mined pattern_library stats shown elsewhere in this
# dashboard, there's no separate walk-forward trade log to persist there.
@app.get("/api/ml_backtest/summary")
def ml_backtest_summary():
    if ENGINE != "ml":
        return {"available": False, "reason": "backtest reports are ML-challenger-only - "
                                                 "the rule-based engine's equivalent is the mined pattern library"}
    import backtest_report
    import model_registry

    by_tf: dict = {}
    symbol_dir = ML_REGISTRY_DIR / SYMBOL
    if not symbol_dir.exists():
        return {"available": True, "by_timeframe": {}}
    for tf_dir in sorted(p for p in symbol_dir.iterdir() if p.is_dir()):
        tf = tf_dir.name
        entries = []
        for direction in ("bullish", "bearish"):
            for rr in model_registry.list_rr_tiers(ML_REGISTRY_DIR, SYMBOL, tf, direction):
                active = model_registry.load_active(ML_REGISTRY_DIR, SYMBOL, tf, direction, rr)
                if active is None:
                    continue
                trades = model_registry.load_validation_trades(
                    ML_REGISTRY_DIR, SYMBOL, tf, direction, active["version_id"], rr,
                )
                stats = backtest_report.summary_stats(trades)
                entries.append({
                    "direction": direction, "rr_ratio": rr, "rr_tag": model_registry.rr_tag(rr),
                    "version_id": active["version_id"], **stats,
                })
        if entries:
            by_tf[tf] = entries
    return {"available": True, "by_timeframe": by_tf}


@app.get("/api/ml_backtest/{tf}/{direction}/{rr_tag}")
def ml_backtest_detail(tf: str, direction: str, rr_tag: str):
    if ENGINE != "ml":
        return {"available": False, "reason": "backtest reports are ML-challenger-only"}
    import backtest_report
    import model_registry

    try:
        rr_ratio = float(rr_tag)
    except ValueError:
        return {"available": False, "reason": f"invalid rr_tag '{rr_tag}'"}

    active = model_registry.load_active(ML_REGISTRY_DIR, SYMBOL, tf, direction, rr_ratio)
    if active is None:
        return {"available": False, "reason": f"no active model for {tf}/{direction}/rr_{rr_tag}"}
    trades = model_registry.load_validation_trades(
        ML_REGISTRY_DIR, SYMBOL, tf, direction, active["version_id"], rr_ratio,
    )
    if trades is None:
        return {"available": False,
                "reason": "this model version predates persisted validation trades - retrain to get a backtest report"}
    return {
        "available": True, "version_id": active["version_id"],
        **backtest_report.full_report(trades),
        # ml_system/explainability.py's two global reports - already
        # embedded in this version's own meta.json by train.py, so no
        # extra read/recompute needed here, just surfaced through.
        "shap_global": active["meta"].get("shap_global"),
        "distilled_rules": active["meta"].get("distilled_rules"),
    }


@app.get("/api/ml_backtest/compare/{tf}/{direction}/{rr_tag}")
def ml_backtest_compare(tf: str, direction: str, rr_tag: str, other_version_id: str, recent_candles: int = 1000):
    """Re-scores the CURRENTLY ACTIVE version against `other_version_id`
    (any version still on disk for this tier, active or not) over the
    SAME most-recent `recent_candles` - see
    ml_system.backtest_report.compare_versions()'s own docstring for why
    this is a genuinely different, apples-to-apples question from each
    version's own (differently-windowed) persisted backtest."""
    if ENGINE != "ml":
        return {"available": False, "reason": "backtest reports are ML-challenger-only"}
    import joblib
    import backtest_report
    import model_registry

    try:
        rr_ratio = float(rr_tag)
    except ValueError:
        return {"available": False, "reason": f"invalid rr_tag '{rr_tag}'"}

    active = model_registry.load_active(ML_REGISTRY_DIR, SYMBOL, tf, direction, rr_ratio)
    if active is None:
        return {"available": False, "reason": f"no active model for {tf}/{direction}/rr_{rr_tag}"}

    other_meta = model_registry.load_meta(ML_REGISTRY_DIR, SYMBOL, tf, direction, other_version_id, rr_ratio)
    if other_meta is None:
        return {"available": False, "reason": f"version '{other_version_id}' not found for {tf}/{direction}/rr_{rr_tag}"}
    other_model_path = ML_REGISTRY_DIR / SYMBOL / tf / direction / f"rr_{rr_tag}" / other_version_id / "model.joblib"
    other = {"version_id": other_version_id, "model": joblib.load(other_model_path), "meta": other_meta}

    p = SHARED_DATA_DIR / "candles" / f"{SYMBOL}_{tf}.parquet"
    if not p.exists():
        return {"available": False, "reason": f"no candle data for {tf}"}
    candles = pd.read_parquet(p).sort_values("timestamp").tail(recent_candles).reset_index(drop=True)

    direction_int = 1 if direction == "bullish" else -1
    comparison = backtest_report.compare_versions(candles, direction_int, rr_ratio, active, other)
    return {"available": True, "timeframe": tf, "direction": direction, "rr_ratio": rr_ratio,
            "recent_candles": len(candles), "active_version_id": active["version_id"], **comparison}


@app.get("/api/signal")
def signal():
    if ENGINE == "ml":
        return _ml_signal()

    from circuit_breaker import check_circuit_breaker
    from signal_engine import compute_signal, load_inputs, load_suspended
    from signal_journal import load_journal
    load_inputs_kwargs = {} if DISCOVERED_DIR is None else {"discovered_dir": DISCOVERED_DIR}
    candles_by_tf, library_by_tf, events, upcoming = load_inputs(SYMBOL, SHARED_DATA_DIR, LIB_DIR, **load_inputs_kwargs)
    if not candles_by_tf or not library_by_tf:
        return {"direction": "UNAVAILABLE", "confidence": 0, "trade_plan": None,
                "freshness": None, "news_risk": None,
                "contributions": [], "reason": "no candle data or pattern library yet"}
    # Self-healing: the dashboard's live signal must reflect the same
    # suspension a cron-driven live_update.py run would apply - otherwise
    # the dashboard could show a signal built from a pattern/direction
    # that's actually DECAYING live, contradicting its own Self-
    # Assessment panel one section down. Same for the circuit breaker -
    # checked against THIS dashboard instance's own journal (DATA_DIR).
    suspended = load_suspended(SHARED_DATA_DIR, LIB_DIR, SYMBOL, DISCOVERED_DIR)
    journal = load_journal(DATA_DIR)
    breaker = check_circuit_breaker(journal)
    sig = compute_signal(candles_by_tf, library_by_tf, events, upcoming,
                         suspended=suspended, circuit_breaker=breaker, journal=journal)
    return sig.to_dict()


@app.get("/api/journal")
def journal(limit: int = 50):
    from signal_journal import detect_drift, load_journal, summary
    j = load_journal(DATA_DIR)
    if j.empty:
        return {"summary": summary(j), "rows": [], "drift": []}
    # Computed over the FULL journal, before it gets truncated to `limit`
    # rows below - summary() must reflect every logged signal, not just
    # the most recent page. (Previously re-read the journal from disk a
    # second time to get this, an unnecessary extra I/O on every request
    # - `j` already held the full journal right here, before truncation.)
    full_summary = summary(j)
    drift = detect_drift(j, LIB_DIR, SYMBOL, DISCOVERED_DIR)
    j = j.sort_values("logged_at_utc", ascending=False).head(limit).copy()
    for col in ("logged_at_utc", "entry_candle_timestamp", "resolved_at_utc"):
        j[col] = j[col].apply(lambda v: v.isoformat() if pd.notna(v) else None)
    return {"summary": full_summary, "rows": json.loads(j.to_json(orient="records")),
            "drift": drift}


@app.get("/api/journal/export")
def journal_export():
    """The full signal journal as a downloadable CSV - every column,
    every row, not just the /api/journal page-limited view the table on
    screen shows. For anyone who wants to pull this into Excel/a
    notebook for their own analysis rather than reading it through the
    dashboard's own tables."""
    from signal_journal import load_journal
    j = load_journal(DATA_DIR)
    buf = io.StringIO()
    j.to_csv(buf, index=False)
    return Response(
        content=buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={SYMBOL}_signal_journal.csv"},
    )


@app.get("/api/performance")
def performance():
    """The self-assessment view: how the system is ACTUALLY doing, live,
    per-pattern credibility, and the realized-R equity curve - everything
    needed to judge whether it's working without having to risk real
    money to find out. Pure computation over the existing signal journal
    (signal_journal.py) - nothing here is a new data source."""
    from signal_journal import context_scorecard, equity_curve, load_journal, overall_scorecard, pattern_scorecard
    j = load_journal(DATA_DIR)
    return {
        "scorecard": overall_scorecard(j),
        "patterns": pattern_scorecard(j, LIB_DIR, SYMBOL, DISCOVERED_DIR),
        "equity_curve": equity_curve(j),
        # Loss/win ATTRIBUTION - not "is this pattern working" (patterns,
        # above), but "WITHIN this pattern's own live trades, does the
        # confluence count / trading session it fired in make a
        # statistically real difference" - see signal_journal.
        # context_scorecard()'s own docstring for the full methodology.
        "context_attribution": context_scorecard(j),
    }


@app.get("/api/position_size")
def position_size_endpoint(account_size: float, risk_pct: float, contract_size: float = 100.0):
    from position_sizing import position_size
    sig = signal()
    trade_plan = sig.get("trade_plan")
    if not trade_plan:
        raise HTTPException(400, "no active trade plan to size right now")
    try:
        return position_size(account_size, risk_pct, trade_plan["risk"], contract_size)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/upcoming_news")
def upcoming_news():
    from news_calendar import load_upcoming
    df = load_upcoming(SHARED_DATA_DIR)
    if df.empty:
        return []
    df = df.sort_values("datetime_utc").head(20).copy()
    df["datetime_utc"] = df["datetime_utc"].astype(str)
    return json.loads(df.to_json(orient="records"))


@app.get("/api/overview")
def overview():
    try:
        sig = signal()
    except Exception as e:
        sig = {"direction": "ERROR", "confidence": 0, "trade_plan": None,
               "contributions": [], "reason": str(e)}
    storage_data = storage()
    return {
        "symbol": SYMBOL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "storage": storage_data,
        "patterns": patterns_overview(),
        "fundamentals": fundamentals(),
        "pulse": pulse(),
        "system_status": _system_status(storage_data),
        "signal": sig,
        "journal": journal(),
        "upcoming_news": upcoming_news(),
        "data_quality": data_quality_endpoint(),
        "performance": performance(),
    }


# ---- static frontend --------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    # Configurable so the rule-based and ML-challenger dashboards (see
    # ml_system/README.md's "Dashboard" section) can actually run as two
    # instances on the same machine, as documented - port 8000 was
    # hardcoded here, so a second `python dashboard/server.py` for the
    # other engine would just fail to bind instead of doing what the
    # README says it does.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DASHBOARD_PORT", 8000)))
