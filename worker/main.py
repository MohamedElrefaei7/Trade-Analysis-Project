"""
main.py — the scheduler process. Runs every job in
orchestration/tasks.py::JOBS on the cadence defined in worker/cadences.py.

Run with:
    python -m worker.main

BlockingScheduler, not BackgroundScheduler: this process *is* the
container's foreground process (Docker captures stdout for logs), and
BackgroundScheduler plus a sleep loop is the version that exits silently
once the main thread ends.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from typing import Callable

import psycopg2
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.blocking import BlockingScheduler

from orchestration.tasks import JOBS
from worker.cadences import CADENCES, UTC, Cadence

logger = logging.getLogger("maritime")


def _check_cadences_match_jobs(cadences: dict[str, Cadence], jobs: dict[str, Callable]) -> None:
    """
    Raise immediately, naming the offending entries, if `cadences` and
    `jobs` disagree. A job registered with no cadence never runs; a
    cadence naming no job crashes at fire time, hours later, in a log —
    both must fail at startup instead, loudly, before the scheduler
    blocks.
    """
    cadence_names = set(cadences)
    job_names = set(jobs)
    jobs_without_cadence = job_names - cadence_names
    cadences_without_job = cadence_names - job_names
    if jobs_without_cadence or cadences_without_job:
        raise RuntimeError(
            "worker: CADENCES and JOBS disagree — "
            f"jobs with no cadence entry: {sorted(jobs_without_cadence) or 'none'}; "
            f"cadence entries naming no job: {sorted(cadences_without_job) or 'none'}"
        )


def _connect():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    return conn


def _record_missed(event) -> None:
    """
    EVENT_JOB_MISSED listener — the only way a misfire becomes visible.
    The @job decorator never ran (that's what "missed" means: the fire
    time passed outside misfire_grace_time and APScheduler's executor
    skipped the call entirely), so this writes job_runs directly instead
    of going through it. started_at and finished_at are both the slot's
    scheduled_run_time, not "now" — this is a record of a moment that was
    missed, not an execution with its own timeline, and setting
    finished_at (rather than leaving it NULL) keeps a `missed` row from
    reading like an in-progress `running` one when scanning for stuck jobs.
    """
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO job_runs "
                "(run_id, job_name, started_at, finished_at, status, error_message) "
                "VALUES (%s, %s, %s, %s, 'missed', %s)",
                (
                    str(uuid.uuid4()),
                    event.job_id,
                    event.scheduled_run_time,
                    event.scheduled_run_time,
                    "misfire: scheduled run time exceeded misfire_grace_time",
                ),
            )
    finally:
        conn.close()
    logger.warning(
        "worker: job %r missed its slot scheduled for %s", event.job_id, event.scheduled_run_time
    )


def _log_error(event) -> None:
    """
    EVENT_JOB_ERROR listener — logs only, writes nothing. The @job
    decorator already recorded the `failed` row before re-raising; a
    listener writing a second one here would double-count every failure
    in the health view.
    """
    logger.error(
        "worker: job %r raised during execution: %s", event.job_id, event.exception,
        exc_info=(type(event.exception), event.exception, None) if event.exception else None,
    )


def _already_persisted_job_ids(database_url: str, candidate_ids: set[str]) -> set[str]:
    """
    Which of `candidate_ids` already have a row in the SQLAlchemyJobStore's
    table (apscheduler_jobs). A direct query, deliberately not going
    through APScheduler's own Job/JobStore objects — this only needs to
    answer one question (does this job already exist from a prior
    process?), and asking it directly avoids depending on internals of
    how APScheduler represents a not-yet-started scheduler's job state.
    Returns the empty set on a fresh database (table doesn't exist yet).
    """
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'apscheduler_jobs')"
            )
            if not cur.fetchone()[0]:
                return set()
            cur.execute(
                "SELECT id FROM apscheduler_jobs WHERE id = ANY(%s)",
                (list(candidate_ids),),
            )
            return {row[0] for row in cur.fetchall()}
    finally:
        conn.close()


def build_scheduler(
    cadences: dict[str, Cadence] = CADENCES,
    jobs: dict[str, Callable] = JOBS,
) -> BlockingScheduler:
    """
    Construct and fully configure a BlockingScheduler — jobs registered,
    listeners attached — without starting it. Split out from main() so
    tests can inspect the configured scheduler without blocking forever
    in .start().

    Jobs are persisted in a SQLAlchemyJobStore backed by the same
    Postgres database (not the library default, in-memory job store).
    This is what decisions 2/3 (coalesce + proportional misfire grace)
    actually depend on: with an in-memory store, a fresh process has no
    memory of a job's previous next_run_time, so on restart it computes
    a brand-new "now + interval" and simply never notices anything was
    missed — coalesce and grace have nothing to act on. Discovered live
    (see CONTEXT.md, 2026-08-08): the first version of this file used the
    default in-memory store and a restart-recovery verification produced
    zero catch-up rows, not one.

    A job whose id already has a persisted row is deliberately NOT
    re-added — add_job(..., replace_existing=True) computes a fresh
    next_run_time via trigger.get_next_fire_time(None, now) on every
    call, which would silently overwrite the persisted value (and the
    fact that a slot was missed) on every single restart. Skipping
    already-persisted jobs leaves their stored next_run_time alone;
    APScheduler's own _process_jobs() queries the jobstore for due jobs
    directly, so an already-persisted job is picked up and correctly
    coalesced/misfired against its *real* last-known schedule the moment
    the scheduler starts, with no re-registration needed. Trade-off,
    stated plainly: changing a job's cadence in code has no effect on an
    already-persisted job until its stored row is cleared — this design
    favors restart correctness over live config reload, which is what
    this commit's done-condition is actually about.
    """
    _check_cadences_match_jobs(cadences, jobs)

    database_url = os.environ["DATABASE_URL"]
    jobstore = SQLAlchemyJobStore(url=database_url)
    scheduler = BlockingScheduler(jobstores={"default": jobstore}, timezone=UTC)

    already_persisted = _already_persisted_job_ids(database_url, set(cadences))
    for name, cadence in cadences.items():
        if name in already_persisted:
            logger.info("worker: %s already scheduled (persisted) — leaving it alone", name)
            continue
        scheduler.add_job(
            jobs[name],
            trigger=cadence.trigger,
            id=name,
            name=name,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=cadence.misfire_grace_time,
        )
        logger.info("worker: registered %s (first run)", name)

    scheduler.add_listener(_record_missed, EVENT_JOB_MISSED)
    scheduler.add_listener(_log_error, EVENT_JOB_ERROR)
    return scheduler


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )

    scheduler = build_scheduler()
    logger.info("worker: starting — %d jobs scheduled", len(CADENCES))
    scheduler.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
