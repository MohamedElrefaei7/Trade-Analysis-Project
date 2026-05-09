"""
scheduler.py — Polling loop / orchestration for the maritime data pipeline.

Uses Prefect 3.x (serve + deployments) for all scheduled batch jobs.
The AIS WebSocket stream runs as a persistent daemon thread alongside Prefect,
with a watchdog thread that auto-restarts it on failure.

── Schedule at a glance ──────────────────────────────────────────────────────
  Source                 Cadence              Prefect flow name
  ─────────────────────  ───────────────────  ────────────────────────────────
  AIS positions          continuous (daemon)  n/a — background thread
  Port call refresh      every 2 hours        port-call-refresh
  OpenSky air freight    daily  06:00 UTC     opensky-daily
  FRED macro data        daily  07:00 UTC     fred-daily
  BDI scraper            daily  18:30 UTC     bdi-daily (post market close)
  Drewry WCI rates       weekly Fri 09:00     wci-weekly
  UN Comtrade trade      monthly 15th 08:00   comtrade-monthly
  Port of LA TEU         monthly 16th 08:00   port-la-monthly
  Normalizer (Step 7)    daily  23:30 UTC     normalizer-nightly
  Targets builder        daily  23:45 UTC     targets-nightly
  Signals sweep          daily  23:55 UTC     signals-nightly
  Model trainer          daily  00:05 UTC     models-nightly
  Alerter                daily  00:15 UTC     alerts-nightly
──────────────────────────────────────────────────────────────────────────────

Quick start:
    # Terminal 1 — Prefect API server (required, NOT optional)
    prefect server start

    # Terminal 2 — run the pipeline (must see PREFECT_API_URL in .env)
    python scheduler.py

    # UI at http://127.0.0.1:4200

If PREFECT_API_URL is unset, `serve()` would silently spawn an ephemeral
in-process server, detach the scheduler from the persistent run history,
and lose every flow result on restart. We fail fast at startup instead.

Environment variables required (.env):
    DATABASE_URL, PREFECT_API_URL, AISSTREAM_API_KEY, FRED_API_KEY,
    OPENSKY_USER, OPENSKY_PASS, COMTRADE_SUBSCRIPTION_KEY
"""

import asyncio
import logging
import threading
import time
from datetime import timedelta

from prefect import flow, serve
from prefect.logging import get_run_logger
from prefect.schedules import Cron, Interval
from sqlalchemy import text

from clients.base import Session, logger as pipeline_logger
from clients.aisstream import stream as _ais_stream
from clients.fred import run as _fred_run
from clients.comtrade import run as _comtrade_run
from clients.opensky import run as _opensky_run
from clients.scraper import bdi_scraper, wci_scraper, port_la_scraper
from normalizer import run_all as _normalizer_run
from targets import run_all as _targets_run
from signals import run_all as _signals_run
from models import run_all as _models_run
from alerts import run_all as _alerts_run

# ── AIS stream daemon + watchdog ──────────────────────────────────────────────

_ais_thread: threading.Thread | None = None
_ais_lock = threading.Lock()


def _start_ais_thread() -> None:
    """Spin up the AIS WebSocket stream in its own thread with its own event loop."""
    global _ais_thread
    with _ais_lock:
        t = threading.Thread(
            target=lambda: asyncio.run(_ais_stream()),
            daemon=True,
            name="ais-stream",
        )
        t.start()
        _ais_thread = t
    pipeline_logger.info("AIS stream thread started (tid=%s)", t.ident)


def _ais_watchdog() -> None:
    """
    Checks every 60 s whether the AIS thread is alive.
    Restarts it automatically if it has died (e.g. unhandled exception, OOM).
    Runs as a daemon thread — exits when the main process exits.
    """
    while True:
        time.sleep(60)
        if _ais_thread is not None and not _ais_thread.is_alive():
            pipeline_logger.warning(
                "AIS stream thread (tid=%s) is dead — restarting", _ais_thread.ident
            )
            _start_ais_thread()


def _start_watchdog() -> None:
    t = threading.Thread(target=_ais_watchdog, daemon=True, name="ais-watchdog")
    t.start()
    pipeline_logger.info("AIS watchdog started")


# ══════════════════════════════════════════════════════════════════════════════
# Prefect flows
# ══════════════════════════════════════════════════════════════════════════════

# ── Port call refresh — every 2 hours ─────────────────────────────────────────

@flow(name="port-call-refresh", log_prints=True)
def port_call_refresh_flow() -> dict:
    """
    Runs every 2 hours. Does three things:

    1. Checks whether the AIS stream daemon thread is alive.
    2. Summarises open port calls (arrived, not yet departed) by port.
    3. Flags calls open > 48 h — these are likely stale (missed departure event).

    This flow is purely observational — it does not write to the DB.
    Use the Prefect UI to see trends in open-call counts over time.
    """
    log = get_run_logger()

    # ── Thread health ────────────────────────────────────────────────────────
    if _ais_thread is None:
        log.warning("AIS thread: not started (AISSTREAM_API_KEY may be missing)")
    elif _ais_thread.is_alive():
        log.info("AIS thread: alive (tid=%s)", _ais_thread.ident)
    else:
        log.warning("AIS thread: DEAD (tid=%s) — watchdog will restart it", _ais_thread.ident)

    results: dict = {}

    with Session() as session:
        # ── Open calls by port ────────────────────────────────────────────────
        open_rows = session.execute(
            text(
                """
                SELECT pc.port_unlocode,
                       COUNT(*)                                                  AS open_calls,
                       AVG(EXTRACT(EPOCH FROM NOW() - pc.arrived_at) / 3600)::int AS avg_dwell_h
                FROM port_calls pc
                WHERE pc.departed_at IS NULL
                GROUP BY pc.port_unlocode
                ORDER BY open_calls DESC
                """
            )
        ).fetchall()

        log.info("Open port calls — %d ports active", len(open_rows))
        for row in open_rows:
            log.info(
                "  %-6s  open=%d  avg_dwell=%dh",
                row.port_unlocode, row.open_calls, row.avg_dwell_h or 0,
            )
        results["open_ports"] = len(open_rows)

        # ── Stale calls: no departure after 48 h ─────────────────────────────
        stale_rows = session.execute(
            text(
                """
                SELECT v.mmsi,
                       pc.port_unlocode,
                       (EXTRACT(EPOCH FROM NOW() - pc.arrived_at) / 3600)::int AS hours
                FROM port_calls pc
                JOIN vessels v ON pc.vessel_id = v.vessel_id
                WHERE pc.departed_at IS NULL
                  AND pc.arrived_at < NOW() - INTERVAL '48 hours'
                ORDER BY hours DESC
                LIMIT 25
                """
            )
        ).fetchall()

        if stale_rows:
            log.warning("Stale open port calls (>48h without departure):")
            for row in stale_rows:
                log.warning(
                    "  MMSI=%-12s  port=%-6s  dwell=%dh",
                    row.mmsi, row.port_unlocode, row.hours,
                )
        results["stale_calls"] = len(stale_rows)

        # ── Position freshness ────────────────────────────────────────────────
        pos_last_hour = session.execute(
            text("SELECT COUNT(*) FROM positions WHERE ts > NOW() - INTERVAL '1 hour'")
        ).scalar() or 0

        results["positions_last_hour"] = pos_last_hour
        if pos_last_hour == 0:
            log.warning("No positions written in the last hour — AIS stream may be stalled")
        else:
            log.info("Positions written in last hour: %d", pos_last_hour)

    return results


# ── OpenSky — daily 06:00 UTC ────────────────────────────────────────────────

@flow(name="opensky-daily", log_prints=True)
def opensky_flow() -> None:
    """
    Poll all configured bounding boxes on OpenSky for large / heavy aircraft.
    Cargo-operator callsigns get cargo_flag=True in flight_events.
    """
    log = get_run_logger()
    log.info("Starting OpenSky air freight poll…")
    _opensky_run()
    log.info("OpenSky poll complete")


# ── FRED macro — daily 07:00 UTC ─────────────────────────────────────────────

@flow(name="fred-daily", log_prints=True)
def fred_flow() -> None:
    """
    Fetch the five configured FRED series (GDP, trade balance, AUD/USD,
    CNY/USD, import price index) and append new observations to
    economic_benchmarks with series_id prefix FRED:.
    """
    log = get_run_logger()
    log.info("Starting FRED macro ingest…")
    _fred_run()
    log.info("FRED ingest complete")


# ── BDI scraper — daily 18:30 UTC (post market close) ───────────────────────

@flow(name="bdi-daily", log_prints=True)
def bdi_flow() -> None:
    """
    Scrape the Baltic Dry Index closing value from Investing.com.
    Runs after 18:30 UTC so the day's close is published.
    Uses Playwright (headless Chromium) — JS-rendered page.
    """
    log = get_run_logger()
    log.info("Starting BDI scraper…")
    n = bdi_scraper()
    log.info("BDI: %d rows inserted", n)


# ── WCI scraper — weekly Fridays 09:00 UTC ───────────────────────────────────

@flow(name="wci-weekly", log_prints=True)
def wci_flow() -> None:
    """
    Extract Drewry World Container Index spot rates (composite + 5 lanes) from
    Hellenic Shipping News' weekly Drewry commentary. Runs Friday mornings
    because Drewry publishes the index Thursday afternoon UTC and HSN re-posts
    it shortly after.
    """
    log = get_run_logger()
    log.info("Starting WCI scraper…")
    n = wci_scraper()
    log.info("WCI: %d rows inserted", n)


# ── UN Comtrade — monthly 15th 08:00 UTC ─────────────────────────────────────

@flow(name="comtrade-monthly", log_prints=True)
def comtrade_flow() -> None:
    """
    Fetch UN Comtrade bilateral trade flow data for configured country-commodity
    pairs (CN→US electronics/fuels, AU→CN iron ore, BR→CN soybeans).
    Comtrade data lags 4–6 weeks; running on the 15th catches the previous
    month's release.
    """
    log = get_run_logger()
    log.info("Starting Comtrade ingest…")
    _comtrade_run()
    log.info("Comtrade ingest complete")


# ── Port of LA TEU — monthly 16th 08:00 UTC ─────────────────────────────────

@flow(name="port-la-monthly", log_prints=True)
def port_la_flow() -> None:
    """
    Scrape Port of LA monthly container throughput (TEUs, imports, exports)
    from their historical statistics pages. Port of LA publishes ~2 weeks
    after month end; running on the 16th reliably catches the prior month.
    Upserts into port_daily_summary (date = first of month).
    """
    log = get_run_logger()
    log.info("Starting Port of LA scraper…")
    n = port_la_scraper(years_back=2)
    log.info("Port LA: %d rows upserted", n)


# ── Normalizer (Step 7) — daily 23:30 UTC ───────────────────────────────────

@flow(name="normalizer-nightly", log_prints=True)
def normalizer_flow() -> dict:
    """
    Nightly transformation pass. Runs last in the day's schedule so it
    sees the output of every upstream ingest. Executes in fixed order:

        port_resolver → vessel_normalizer → feature_builder

    Writes to `features`; never mutates raw tables. The feature rows
    produced here are what the backtest layer consumes exclusively.
    """
    log = get_run_logger()
    log.info("Starting nightly normalizer run…")
    summary = _normalizer_run()
    log.info("Normalizer complete — %s", summary)
    return summary


# ── Targets — daily 23:45 UTC (after normalizer) ─────────────────────────────

@flow(name="targets-nightly", log_prints=True)
def targets_flow() -> dict:
    """
    Build the prediction targets table from the daily-aligned features.
    Runs 15 minutes after normalizer-nightly so it sees the day's features.
    """
    log = get_run_logger()
    log.info("Starting nightly targets build…")
    summary = _targets_run()
    log.info("Targets complete — %s", summary)
    return summary


# ── Signals — daily 23:55 UTC (after targets) ────────────────────────────────

@flow(name="signals-nightly", log_prints=True)
def signals_flow() -> dict:
    """
    Sweep every (feature × target × window) for the strongest lead-lag
    relationship and store the top-N into the `signals` table. Runs 10 minutes
    after targets-nightly so it sees the same day's features and targets.
    """
    log = get_run_logger()
    log.info("Starting nightly signals sweep…")
    summary = _signals_run()
    log.info("Signals complete — %s", summary)
    return summary


# ── Models — daily 00:05 UTC (after signals, next-day rollover) ──────────────

@flow(name="models-nightly", log_prints=True)
def models_flow() -> dict:
    """
    Train one ElasticNet per (target, horizon) on lag-adjusted feature
    z-scores and write predictions (walk-forward OOS for the realised period
    plus live forecasts beyond). Runs 10 minutes after signals-nightly.
    """
    log = get_run_logger()
    log.info("Starting nightly model training…")
    summary = _models_run()
    log.info("Models complete — %s", summary)
    return summary


# ── Alerter — daily 00:15 UTC (after models) ─────────────────────────────────

@flow(name="alerts-nightly", log_prints=True)
def alerts_flow() -> dict:
    """
    Three edge-triggered checks against today's normalized data:
      (a) feature z-score crossed ±2σ AND has a significant lead-lag signal
      (b) prediction_z crossed ±2σ
      (c) signals row just became significant (regime change)
    Writes to the `alerts` table. Posts a digest to Slack if SLACK_WEBHOOK_URL
    is set.
    """
    log = get_run_logger()
    log.info("Starting nightly alerter…")
    summary = _alerts_run()
    log.info("Alerts complete — %s", summary)
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Deployment registry
# ══════════════════════════════════════════════════════════════════════════════

def _build_deployments() -> list:
    """
    Returns a list of RunnerDeployment objects ready to pass to serve().
    All times are UTC. Tags enable filtering in the Prefect UI.
    """
    return [
        port_call_refresh_flow.to_deployment(
            name="port-call-refresh",
            schedule=Interval(timedelta(hours=2)),
            description="Audit open port calls and monitor AIS stream health.",
            tags=["ais", "monitoring"],
        ),
        opensky_flow.to_deployment(
            name="opensky-daily",
            schedule=Cron("0 6 * * *", timezone="UTC"),
            description="Scrape ADS-B states for large/heavy aircraft.",
            tags=["air", "opensky", "daily"],
        ),
        fred_flow.to_deployment(
            name="fred-daily",
            schedule=Cron("0 7 * * *", timezone="UTC"),
            description="Fetch FRED macro: GDP, trade balance, FX rates.",
            tags=["macro", "fred", "daily"],
        ),
        bdi_flow.to_deployment(
            name="bdi-daily",
            schedule=Cron("30 18 * * *", timezone="UTC"),
            description="Scrape Baltic Dry Index from Investing.com.",
            tags=["shipping", "scraper", "daily"],
        ),
        wci_flow.to_deployment(
            name="wci-weekly",
            schedule=Cron("0 9 * * 5", timezone="UTC"),
            description="Drewry World Container Index spot rates (composite + 5 lanes).",
            tags=["shipping", "scraper", "weekly"],
        ),
        comtrade_flow.to_deployment(
            name="comtrade-monthly",
            schedule=Cron("0 8 15 * *", timezone="UTC"),
            description="UN Comtrade bilateral trade flows.",
            tags=["macro", "comtrade", "monthly"],
        ),
        port_la_flow.to_deployment(
            name="port-la-monthly",
            schedule=Cron("0 8 16 * *", timezone="UTC"),
            description="Port of LA monthly TEU throughput.",
            tags=["port", "scraper", "monthly"],
        ),
        normalizer_flow.to_deployment(
            name="normalizer-nightly",
            schedule=Cron("30 23 * * *", timezone="UTC"),
            description="Step 7: resolve ports, smooth AIS, build features table.",
            tags=["normalizer", "daily"],
        ),
        targets_flow.to_deployment(
            name="targets-nightly",
            schedule=Cron("45 23 * * *", timezone="UTC"),
            description="Build prediction targets (forward returns) from features.",
            tags=["targets", "daily"],
        ),
        signals_flow.to_deployment(
            name="signals-nightly",
            schedule=Cron("55 23 * * *", timezone="UTC"),
            description="Sweep lead-lag relationships across (feature × target × window).",
            tags=["signals", "daily"],
        ),
        models_flow.to_deployment(
            name="models-nightly",
            schedule=Cron("5 0 * * *", timezone="UTC"),
            description="Train ElasticNet per (target, horizon); write predictions.",
            tags=["models", "daily"],
        ),
        alerts_flow.to_deployment(
            name="alerts-nightly",
            schedule=Cron("15 0 * * *", timezone="UTC"),
            description="Edge-triggered alerts on extreme features, predictions, and regime changes.",
            tags=["alerts", "daily"],
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _require_prefect_api() -> None:
    """
    Fail fast if PREFECT_API_URL isn't pointing at a reachable server.

    Without this, `serve()` silently spawns an in-process ephemeral server on
    a random port. The scheduler appears healthy but is detached from the
    persistent run history — every flow result vanishes on restart, the UI
    at :4200 shows nothing, and a failing `models-nightly` would go
    unnoticed for days.
    """
    import os
    from urllib import error as _urlerror, request as _urlrequest

    api_url = os.environ.get("PREFECT_API_URL")
    if not api_url:
        raise SystemExit(
            "PREFECT_API_URL is unset.\n"
            "  Add to .env:    PREFECT_API_URL=http://127.0.0.1:4200/api\n"
            "  Then start:     prefect server start    (in a separate terminal)\n"
            "Refusing to start — see scheduler.py docstring."
        )

    try:
        with _urlrequest.urlopen(f"{api_url.rstrip('/')}/health", timeout=3) as r:
            if not (200 <= r.status < 300):
                raise SystemExit(f"PREFECT_API_URL={api_url} returned HTTP {r.status}")
    except _urlerror.URLError as exc:
        raise SystemExit(
            f"Cannot reach PREFECT_API_URL={api_url} ({exc.reason}).\n"
            "Start the server with:  prefect server start"
        )

    pipeline_logger.info("Prefect API: %s (reachable)", api_url)


def main() -> None:
    pipeline_logger.info("=" * 60)
    pipeline_logger.info("Maritime Analyzer — scheduler starting")
    pipeline_logger.info("=" * 60)

    # ── 0. Make sure we're talking to the persistent Prefect server ──────────
    _require_prefect_api()

    # ── 1. Start AIS continuous stream ───────────────────────────────────────
    import os
    if os.environ.get("AISSTREAM_API_KEY"):
        _start_ais_thread()
        _start_watchdog()
    else:
        pipeline_logger.warning(
            "AISSTREAM_API_KEY not set — AIS stream disabled. "
            "Set the key in .env to enable real-time vessel tracking."
        )

    # ── 2. Print schedule summary ─────────────────────────────────────────────
    pipeline_logger.info("")
    pipeline_logger.info("Scheduled flows:")
    pipeline_logger.info("  %-28s  every 2 hours", "port-call-refresh")
    pipeline_logger.info("  %-28s  daily  06:00 UTC", "opensky-daily")
    pipeline_logger.info("  %-28s  daily  07:00 UTC", "fred-daily")
    pipeline_logger.info("  %-28s  daily  18:30 UTC", "bdi-daily")
    pipeline_logger.info("  %-28s  weekly Fri 09:00 UTC", "wci-weekly")
    pipeline_logger.info("  %-28s  monthly 15th 08:00 UTC", "comtrade-monthly")
    pipeline_logger.info("  %-28s  monthly 16th 08:00 UTC", "port-la-monthly")
    pipeline_logger.info("  %-28s  daily  23:30 UTC", "normalizer-nightly")
    pipeline_logger.info("  %-28s  daily  23:45 UTC", "targets-nightly")
    pipeline_logger.info("  %-28s  daily  23:55 UTC", "signals-nightly")
    pipeline_logger.info("  %-28s  daily  00:05 UTC", "models-nightly")
    pipeline_logger.info("  %-28s  daily  00:15 UTC", "alerts-nightly")
    pipeline_logger.info("")
    pipeline_logger.info(
        "Prefect UI: run `prefect server start` in a separate terminal, "
        "then open http://127.0.0.1:4200"
    )
    pipeline_logger.info("")

    # ── 3. Hand off to Prefect (blocking) ────────────────────────────────────
    serve(*_build_deployments())


if __name__ == "__main__":
    main()
