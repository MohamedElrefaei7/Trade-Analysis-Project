"""
test_worker.py — enforcement tests for worker/cadences.py and
worker/main.py.

Tests that touch job_runs run against a disposable scratch database
(migrations 0001 + 0002 applied via orchestration.migrate, DATABASE_URL
monkeypatched to it) — never `mydb`, the frozen Phase 2 rollback volume.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg2
import pytest
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED, JobExecutionEvent
from apscheduler.triggers.interval import IntervalTrigger
from psycopg2 import sql

from orchestration.migrate import run_migrations
from orchestration.tasks import JOBS
from worker.cadences import CADENCES, Cadence
from worker.main import _check_cadences_match_jobs, _log_error, _record_missed, build_scheduler

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_MIGRATIONS_DIR = REPO_ROOT / "migrations"

_FEW_MINUTES = timedelta(minutes=5)


def _admin_database_url() -> str:
    return os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql://admin:password@localhost:5432/postgres",
    )


@pytest.fixture
def scratch_db(monkeypatch):
    admin_url = _admin_database_url()
    db_name = f"orch_worker_test_{uuid.uuid4().hex[:12]}"

    admin_conn = psycopg2.connect(admin_url)
    admin_conn.autocommit = True
    with admin_conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    admin_conn.close()

    parts = urlsplit(admin_url)
    scratch_url = urlunsplit(parts._replace(path=f"/{db_name}"))
    run_migrations(scratch_url, migrations_dir=REAL_MIGRATIONS_DIR)
    monkeypatch.setenv("DATABASE_URL", scratch_url)

    try:
        yield scratch_url
    finally:
        admin_conn = psycopg2.connect(admin_url)
        admin_conn.autocommit = True
        with admin_conn.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (db_name,),
            )
            cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))
        admin_conn.close()


def _fetch_all(database_url: str, query: str, params=None):
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------


def test_cadences_and_jobs_are_the_same_set():
    assert set(CADENCES.keys()) == set(JOBS.keys())


def test_worker_raises_on_cadence_without_job():
    bad_cadences = dict(CADENCES)
    bad_cadences["totally-unregistered-job"] = bad_cadences["bdi-daily"]

    with pytest.raises(RuntimeError):
        _check_cadences_match_jobs(bad_cadences, JOBS)


def test_all_triggers_are_utc():
    for name, cadence in CADENCES.items():
        assert cadence.trigger.timezone == timezone.utc, f"{name}'s trigger is not UTC"


def test_no_job_uses_default_misfire_grace():
    for name, cadence in CADENCES.items():
        assert cadence.misfire_grace_time is not None, f"{name} has no misfire_grace_time"
        assert cadence.misfire_grace_time != 1, f"{name} left misfire_grace_time at the library default"


def test_misfire_grace_is_bounded_by_interval():
    for name, cadence in CADENCES.items():
        grace = cadence.misfire_grace_time
        assert grace > _FEW_MINUTES.total_seconds(), f"{name}'s grace is too short to matter: {grace}s"
        assert grace < cadence.interval.total_seconds(), (
            f"{name}'s grace ({grace}s) reaches or exceeds its own interval "
            f"({cadence.interval.total_seconds()}s) — would fire redundantly near its next slot"
        )


def test_all_jobs_coalesce_and_max_instances_one(scratch_db):
    scheduler = build_scheduler()
    jobs = scheduler.get_jobs()
    assert len(jobs) == 8
    for apjob in jobs:
        assert apjob.coalesce is True, f"{apjob.id} does not coalesce"
        assert apjob.max_instances == 1, f"{apjob.id} allows more than one concurrent instance"


def test_worker_preserves_persisted_next_run_time_across_restart(scratch_db):
    """
    The property decisions 2+3 actually depend on: a fresh scheduler
    process ("a restart") must not discard a previously-persisted
    next_run_time, or coalesce/misfire_grace_time have nothing to act on.
    Regression test for a real bug caught during live verification (see
    CONTEXT.md, 2026-08-08): the first version of build_scheduler() used
    APScheduler's default in-memory job store with
    add_job(..., replace_existing=True), which recomputes next_run_time
    fresh on every call — a restart silently never noticed anything had
    been missed, producing zero catch-up runs instead of one.

    A 60-second interval and sub-second sleeps keep this deterministic:
    neither scheduler instance runs long enough to actually fire the job,
    so this isolates persistence of scheduling state from job execution.
    """
    override_cadences = {
        "port-call-refresh": Cadence(
            trigger=IntervalTrigger(seconds=60, timezone=timezone.utc),
            interval=timedelta(seconds=60),
            max_age=timedelta(seconds=180),
            misfire_grace_time=300,
        ),
    }
    override_jobs = {"port-call-refresh": JOBS["port-call-refresh"]}

    scheduler1 = build_scheduler(cadences=override_cadences, jobs=override_jobs)
    thread1 = threading.Thread(target=scheduler1.start, daemon=True)
    thread1.start()
    time.sleep(0.5)
    first_next_run = scheduler1.get_job("port-call-refresh").next_run_time
    scheduler1.shutdown(wait=False)
    thread1.join(timeout=5)

    assert first_next_run is not None

    scheduler2 = build_scheduler(cadences=override_cadences, jobs=override_jobs)
    # The already-persisted job must not be re-registered this run — a
    # skipped job is absent from the pre-start pending list.
    assert scheduler2.get_jobs() == []

    thread2 = threading.Thread(target=scheduler2.start, daemon=True)
    thread2.start()
    time.sleep(0.3)
    resumed_next_run = scheduler2.get_job("port-call-refresh").next_run_time
    scheduler2.shutdown(wait=False)
    thread2.join(timeout=5)

    assert resumed_next_run == first_next_run


def test_max_age_exceeds_interval_for_every_job():
    for name, cadence in CADENCES.items():
        assert cadence.max_age > cadence.interval, (
            f"{name}'s max_age ({cadence.max_age}) does not exceed its interval "
            f"({cadence.interval}) — the heartbeat threshold must not be tighter than the schedule"
        )


def test_missed_event_writes_missed_row(scratch_db):
    scheduled_run_time = datetime(2026, 8, 8, 18, 30, tzinfo=timezone.utc)
    event = JobExecutionEvent(EVENT_JOB_MISSED, "bdi-daily", "default", scheduled_run_time)

    _record_missed(event)

    rows = _fetch_all(
        scratch_db, "SELECT job_name, status FROM job_runs WHERE job_name = %s", ("bdi-daily",)
    )
    assert rows == [("bdi-daily", "missed")]


def test_error_event_does_not_write_second_row(scratch_db, monkeypatch):
    def boom():
        raise RuntimeError("scraper exploded")

    monkeypatch.setattr("orchestration.tasks.wci_scraper", boom)

    with pytest.raises(RuntimeError):
        JOBS["wci-weekly"]()

    # The @job decorator already wrote the failed row. Now simulate
    # APScheduler's own EVENT_JOB_ERROR firing for the same job.
    event = JobExecutionEvent(
        EVENT_JOB_ERROR,
        "wci-weekly",
        "default",
        datetime.now(timezone.utc),
        exception=RuntimeError("scraper exploded"),
        traceback="<fake traceback>",
    )
    _log_error(event)

    rows = _fetch_all(
        scratch_db, "SELECT status FROM job_runs WHERE job_name = %s", ("wci-weekly",)
    )
    assert rows == [("failed",)]


def test_status_check_accepts_missed_and_rejects_unknown(scratch_db):
    conn = psycopg2.connect(scratch_db)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO job_runs (run_id, job_name, started_at, status) "
                "VALUES (gen_random_uuid(), 'x', now(), 'missed')"
            )
        conn.commit()

        with pytest.raises(psycopg2.IntegrityError):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO job_runs (run_id, job_name, started_at, status) "
                    "VALUES (gen_random_uuid(), 'y', now(), 'skipped')"
                )
    finally:
        conn.rollback()
        conn.close()
