"""
tasks.py — the nine scheduled jobs, each a @job-decorated, zero-argument
function returning the number of rows it wrote (`heartbeat` is the one
exception — see its own docstring). Business logic stays in clients/,
normalizer/, targets/, signals/, models/, alerts/, and orchestration/
(heartbeat.py) — these are thin wrappers, one per unit the old Prefect
schedule ran independently (see scheduler.py's docstring for the cadence
this replaces; Commit 3 builds the scheduler that actually calls these
via the JOBS registry below).

Job names are the former Prefect deployment names, unchanged, character
for character, with one addition (`heartbeat`, Phase 3 Commit 6 — it has
no former Prefect deployment, since it didn't exist under Prefect) —
they're the join key for Commit 3's cadence configuration and the
heartbeat's own per-job overdue check. See CLAUDE.md's Jobs contract for
the full set of rules this file follows.
"""

from __future__ import annotations

from typing import Callable

from sqlalchemy import text

from clients.base import Session, logger
from clients.scraper import bdi_scraper, wci_scraper
from normalizer import run_all as _normalizer_run_all
from targets import run_all as _targets_run_all
from signals import run_all as _signals_run_all
from models import run_all as _models_run_all
from alerts import run_all as _alerts_run_all

from .heartbeat import run_all as _heartbeat_run_all
from .jobs import job


def _count_stale_port_calls() -> int:
    """
    Summarizes open port calls by port and flags calls open more than 48h
    as stale. Purely observational — writes nothing to the database.
    Returns the count of stale (open > 48h) port calls found. Split out
    from port_call_refresh() so it's a single stubbable dependency, same
    as every other job's underlying call.
    """
    with Session() as session:
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

        logger.info("port-call-refresh: %d ports with open calls", len(open_rows))
        for row in open_rows:
            logger.info(
                "  %-6s  open=%d  avg_dwell=%dh",
                row.port_unlocode, row.open_calls, row.avg_dwell_h or 0,
            )

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
                """
            )
        ).fetchall()

    if stale_rows:
        logger.warning("port-call-refresh: %d stale open port calls (>48h)", len(stale_rows))
        for row in stale_rows[:25]:
            logger.warning(
                "  MMSI=%-12s  port=%-6s  dwell=%dh",
                row.mmsi, row.port_unlocode, row.hours,
            )

    return len(stale_rows)


@job("port-call-refresh")
def port_call_refresh() -> int:
    """
    Every 2 hours: reports open/stale port calls via
    _count_stale_port_calls(). Returns the count of stale (open > 48h)
    port calls found.

    Previously also health-checked the AIS stream daemon thread; that
    half is dropped here because it's dead on arrival — Commit 4 moves
    the AIS daemon into its own container, where Docker's `restart:
    unless-stopped` supervises it, a stronger guarantee than one Python
    thread polling a sibling thread that dies along with the very process
    that thread lives in.
    """
    return _count_stale_port_calls()


@job("bdi-daily")
def bdi_daily() -> int:
    """Scrape the Baltic Dry Index daily close from Hellenic Shipping
    News. Returns the number of new economic_benchmarks rows inserted —
    0 is the correct, expected result when nothing new has published
    since the last run, not an error."""
    return bdi_scraper()


@job("wci-weekly")
def wci_weekly() -> int:
    """Scrape Drewry World Container Index spot rates from Hellenic
    Shipping News. Returns the number of new economic_benchmarks rows
    inserted across all six WCI:* series — 0 is the correct, expected
    result outside the weekly publication window, not an error."""
    return wci_scraper()


@job("normalizer-nightly")
def normalizer_nightly() -> int:
    """Full Step 7 chain — port_resolver -> vessel_normalizer ->
    port_summary_builder -> time_aligner -> seasonal_adjuster ->
    feature_builder — run as a single scheduled unit; no sub-step gets
    its own @job (see CLAUDE.md's Jobs contract for why). Returns the
    number of `features` rows upserted by the chain's final step."""
    return _normalizer_run_all()


@job("targets-nightly")
def targets_nightly() -> int:
    """Build forward-return prediction targets from the daily features
    table. Returns the number of `targets` rows upserted (rows affected —
    inserts and updates both count, since ON CONFLICT DO UPDATE doesn't
    distinguish them)."""
    return _targets_run_all()


@job("signals-nightly")
def signals_nightly() -> int:
    """Sweep lead-lag relationships across (feature x target x window).
    Returns the number of `signals` rows upserted (rows affected —
    inserts and updates both count)."""
    return _signals_run_all()


@job("models-nightly")
def models_nightly() -> int:
    """Train one ElasticNet per (target, horizon) and write predictions.
    Returns the number of `predictions` rows written (rows affected — OOS
    and live rows both count)."""
    return _models_run_all()


@job("alerts-nightly")
def alerts_nightly() -> int:
    """Run the three edge-triggered alert checks. Returns the number of
    NEW alerts inserted this run — not the number of candidates examined,
    and not existing rows re-affirmed via the upsert's UPDATE path. This
    is the one job where the number means newly_inserted rather than
    rows-affected; see alerts/builder.py::run_all()."""
    return _alerts_run_all()


@job("heartbeat")
def heartbeat() -> int:
    """Hourly: checks every job in worker/cadences.py::CADENCES for
    overdue/never-succeeded/stale-running last-success state, and
    separately checks whether the AIS feed itself (MAX(ts) FROM
    positions) is still advancing. Posts a Slack digest if anything's
    wrong and SLACK_WEBHOOK_URL is set; a Slack failure never fails this
    job. Returns the number of problems found — overdue jobs +
    never-succeeded jobs + stale-running jobs + 1 if AIS is stale — not
    rows written to any table; this is the second job, after
    alerts-nightly, where the number isn't a row count. Zero is healthy.

    Being itself a @job means it checks its own pulse too: if the
    heartbeat stops running, its own last success ages out and the next
    run that does happen reports it. That doesn't cover the heartbeat
    never running again at all — see orchestration/heartbeat.py's
    docstring for why, and Phase 11's UptimeRobot /api/health check for
    the piece that closes that gap."""
    return _heartbeat_run_all()


JOBS: dict[str, Callable[[], int]] = {
    "port-call-refresh": port_call_refresh,
    "bdi-daily": bdi_daily,
    "wci-weekly": wci_weekly,
    "normalizer-nightly": normalizer_nightly,
    "targets-nightly": targets_nightly,
    "signals-nightly": signals_nightly,
    "models-nightly": models_nightly,
    "alerts-nightly": alerts_nightly,
    "heartbeat": heartbeat,
}
