"""
test_heartbeat.py — enforcement tests for orchestration/heartbeat.py.

Tests run against a disposable scratch database (migrations 0001 + 0002
applied via orchestration.migrate, plus a minimal test-local `positions`
table this file creates itself — the real one is a TimescaleDB hypertable
defined in schema.sql, which is out of scope for this commit and not
needed for a MAX(ts) query) — never `mydb`, the frozen Phase 2 rollback
volume.

Two families of test here, deliberately:
  - The check_jobs()/check_ais() tests run a real SQLAlchemy session
    against the scratch database directly (bypassing clients.base.Session,
    which — per test_tasks.py's own docstring — stays bound to `mydb` for
    the lifetime of the test process). This is the only way to point
    heartbeat's actual queries at disposable data.
  - The run_all()/JOBS["heartbeat"] tests stub check_jobs/check_ais (same
    pattern test_tasks.py uses for every other job's business logic) so
    they exercise the orchestration — Slack posting, return value,
    job_runs recording — without needing clients.base.Session to see the
    scratch database too.

No test calls a real Slack webhook.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg2
import pytest
from psycopg2 import sql
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from orchestration import heartbeat
from orchestration.heartbeat import AisHealth, JobHealth
from orchestration.migrate import run_migrations
from orchestration.tasks import JOBS
from worker.cadences import CADENCES

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_MIGRATIONS_DIR = REPO_ROOT / "migrations"


def _admin_database_url() -> str:
    return os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql://admin:password@localhost:5432/postgres",
    )


@pytest.fixture
def scratch_db(monkeypatch):
    admin_url = _admin_database_url()
    db_name = f"orch_heartbeat_test_{uuid.uuid4().hex[:12]}"

    admin_conn = psycopg2.connect(admin_url)
    admin_conn.autocommit = True
    with admin_conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    admin_conn.close()

    parts = urlsplit(admin_url)
    scratch_url = urlunsplit(parts._replace(path=f"/{db_name}"))
    run_migrations(scratch_url, migrations_dir=REAL_MIGRATIONS_DIR)

    setup_conn = psycopg2.connect(scratch_url)
    setup_conn.autocommit = True
    with setup_conn.cursor() as cur:
        cur.execute("CREATE TABLE positions (ts TIMESTAMPTZ NOT NULL)")
    setup_conn.close()

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


@pytest.fixture
def scratch_session(scratch_db):
    engine = create_engine(scratch_db)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _fetch_all(database_url: str, query: str, params=None):
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()
    finally:
        conn.close()


def _insert_job_run(database_url, job_name, status, started_at, finished_at=None):
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO job_runs (run_id, job_name, started_at, finished_at, status) "
                "VALUES (%s, %s, %s, %s, %s)",
                (str(uuid.uuid4()), job_name, started_at, finished_at, status),
            )
        conn.commit()
    finally:
        conn.close()


def _insert_position(database_url, ts):
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO positions (ts) VALUES (%s)", (ts,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Part 1 — check_jobs()
# ---------------------------------------------------------------------------


def test_uses_max_age_from_cadences_not_local_constants(scratch_db, scratch_session, monkeypatch):
    now = datetime.now(timezone.utc)
    # port-call-refresh's real max_age is 2.5h — a 3h-old success is overdue.
    _insert_job_run(
        scratch_db, "port-call-refresh", "success",
        now - timedelta(hours=3, minutes=5), now - timedelta(hours=3),
    )

    health = heartbeat.check_jobs(scratch_session)
    assert any(name == "port-call-refresh" for name, _ in health.overdue), (
        "expected port-call-refresh to be overdue against its real 2.5h max_age"
    )

    widened = replace(CADENCES["port-call-refresh"], max_age=timedelta(hours=10))
    monkeypatch.setitem(CADENCES, "port-call-refresh", widened)

    health2 = heartbeat.check_jobs(scratch_session)
    assert not any(name == "port-call-refresh" for name, _ in health2.overdue), (
        "heartbeat did not pick up the monkeypatched CADENCES max_age — "
        "it must read the threshold from worker.cadences.CADENCES, not a local constant"
    )


def test_recent_failure_does_not_count_as_recent_success(scratch_db, scratch_session):
    now = datetime.now(timezone.utc)
    # bdi-daily max_age is 30h. Old success (well past it) + a recent failure.
    _insert_job_run(
        scratch_db, "bdi-daily", "success",
        now - timedelta(hours=40, minutes=5), now - timedelta(hours=40),
    )
    _insert_job_run(
        scratch_db, "bdi-daily", "failed",
        now - timedelta(hours=1, minutes=5), now - timedelta(hours=1),
    )

    health = heartbeat.check_jobs(scratch_session)

    assert any(name == "bdi-daily" for name, _ in health.overdue), (
        "a recent failed row must not be mistaken for a recent success — "
        "last-success must come from status='success' rows only"
    )
    assert "bdi-daily" not in health.never_succeeded
    assert not any(name == "bdi-daily" for name, _ in health.stale_running)


def test_running_row_does_not_count_as_success(scratch_db, scratch_session):
    now = datetime.now(timezone.utc)
    _insert_job_run(
        scratch_db, "bdi-daily", "success",
        now - timedelta(hours=40, minutes=5), now - timedelta(hours=40),
    )
    # Recent (not stale) running row — job is presently mid-run.
    _insert_job_run(scratch_db, "bdi-daily", "running", now - timedelta(minutes=10), None)

    health = heartbeat.check_jobs(scratch_session)

    assert any(name == "bdi-daily" for name, _ in health.overdue), (
        "a recent running row must not count as a recent success"
    )
    assert not any(name == "bdi-daily" for name, _ in health.stale_running), (
        "the running row is only 10 minutes old — nowhere near bdi-daily's 30h max_age"
    )


def test_job_with_no_runs_is_reported_as_never_succeeded(scratch_db, scratch_session):
    # Untouched scratch DB — no job_runs rows for anything.
    health = heartbeat.check_jobs(scratch_session)

    assert "models-nightly" in health.never_succeeded
    assert not any(name == "models-nightly" for name, _ in health.overdue), (
        "never-succeeded must not be folded into overdue with a fabricated age"
    )
    assert not any(name == "models-nightly" for name, _ in health.stale_running)


def test_stale_running_row_reported_separately(scratch_db, scratch_session):
    now = datetime.now(timezone.utc)
    # bdi-daily max_age is 30h. Success is old (would independently read
    # overdue) AND the most recent attempt is a running row that's also
    # far past max_age — the process died mid-run.
    _insert_job_run(
        scratch_db, "bdi-daily", "success",
        now - timedelta(hours=60, minutes=5), now - timedelta(hours=60),
    )
    _insert_job_run(scratch_db, "bdi-daily", "running", now - timedelta(hours=40), None)

    health = heartbeat.check_jobs(scratch_session)

    assert any(name == "bdi-daily" for name, _ in health.stale_running)
    assert not any(name == "bdi-daily" for name, _ in health.overdue), (
        "a stale running row must be reported under stale_running only, "
        "not double-counted under overdue too"
    )


# ---------------------------------------------------------------------------
# Part 2 — check_ais()
# ---------------------------------------------------------------------------


def test_ais_freshness_alerts_on_stale_max_ts(scratch_db, scratch_session):
    now = datetime.now(timezone.utc)
    _insert_position(scratch_db, now - timedelta(hours=7))

    stale_health = heartbeat.check_ais(scratch_session)
    assert stale_health.stale is True

    _insert_position(scratch_db, now - timedelta(hours=1))

    fresh_health = heartbeat.check_ais(scratch_session)
    assert fresh_health.stale is False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_heartbeat_is_registered_in_jobs_and_cadences():
    assert "heartbeat" in JOBS
    assert "heartbeat" in CADENCES
    assert set(JOBS.keys()) == set(CADENCES.keys())


# ---------------------------------------------------------------------------
# Orchestration — Slack posting, return value, job_runs recording
# ---------------------------------------------------------------------------


def test_slack_failure_does_not_raise(scratch_db, monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/fake")
    monkeypatch.setattr(
        heartbeat, "check_jobs",
        lambda session: JobHealth(
            overdue=[("bdi-daily", timedelta(hours=40))],
            never_succeeded=[], stale_running=[],
        ),
    )
    monkeypatch.setattr(
        heartbeat, "check_ais",
        lambda session: AisHealth(stale=False, age=timedelta(hours=1), max_ts=datetime.now(timezone.utc)),
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("slack webhook exploded")

    monkeypatch.setattr(heartbeat, "_maybe_post_slack", _boom)

    result = JOBS["heartbeat"]()

    assert result == 1
    rows = _fetch_all(scratch_db, "SELECT status FROM job_runs WHERE job_name = %s", ("heartbeat",))
    assert rows == [("success",)], (
        "a Slack posting failure must never turn the heartbeat's own job_runs row into 'failed' — "
        "that would make the monitoring's own failure indistinguishable from what it monitors"
    )


def test_no_problems_posts_nothing(scratch_db, monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/fake")
    monkeypatch.setattr(
        heartbeat, "check_jobs",
        lambda session: JobHealth(overdue=[], never_succeeded=[], stale_running=[]),
    )
    monkeypatch.setattr(
        heartbeat, "check_ais",
        lambda session: AisHealth(stale=False, age=timedelta(hours=1), max_ts=datetime.now(timezone.utc)),
    )
    calls = []
    monkeypatch.setattr(heartbeat, "_maybe_post_slack", lambda *a, **k: calls.append((a, k)))

    result = JOBS["heartbeat"]()

    assert result == 0
    assert calls == [], "no problems found must mean zero Slack webhook calls"
