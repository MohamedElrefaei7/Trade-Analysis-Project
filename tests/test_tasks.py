"""
test_tasks.py — enforcement tests for orchestration/tasks.py's nine
@job-decorated functions and the JOBS registry.

Each test that touches job_runs runs against a disposable scratch database
on the local Postgres server (migration 0001 applied via
orchestration.migrate, DATABASE_URL monkeypatched to it) — never `mydb`,
the frozen Phase 2 rollback volume (see CONTEXT.md). Every underlying
business-logic dependency (clients/scraper.py, normalizer/*, targets/,
signals/, models/, alerts/) is stubbed via monkeypatch, so none of these
tests touch a real external service or exercise clients.base's engine,
which stays bound to `mydb` for the lifetime of the test process.
"""

from __future__ import annotations

import inspect
import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg2
import pytest
from psycopg2 import sql

from orchestration import tasks
from orchestration.migrate import run_migrations
from orchestration.tasks import JOBS

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_MIGRATIONS_DIR = REPO_ROOT / "migrations"

EXPECTED_JOB_NAMES = {
    "port-call-refresh",
    "bdi-daily",
    "wci-weekly",
    "normalizer-nightly",
    "targets-nightly",
    "signals-nightly",
    "models-nightly",
    "alerts-nightly",
    "heartbeat",
}


def _admin_database_url() -> str:
    return os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql://admin:password@localhost:5432/postgres",
    )


@pytest.fixture
def scratch_db(monkeypatch):
    admin_url = _admin_database_url()
    db_name = f"orch_tasks_test_{uuid.uuid4().hex[:12]}"

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


def _stub_every_underlying_dependency(monkeypatch, **overrides) -> None:
    """Stub all nine underlying calls tasks.py's jobs delegate to, so a
    full run of every JOBS entry never touches a real external service or
    a real database beyond job_runs itself. `overrides` replaces specific
    stub return values by dependency attribute name."""
    defaults = {
        "_count_stale_port_calls": lambda: 0,
        "bdi_scraper": lambda: 0,
        "wci_scraper": lambda: 0,
        "_normalizer_run_all": lambda: 0,
        "_targets_run_all": lambda: 0,
        "_signals_run_all": lambda: 0,
        "_models_run_all": lambda: 0,
        "_alerts_run_all": lambda: 0,
        "_heartbeat_run_all": lambda: 0,
    }
    defaults.update(overrides)
    for attr, stub in defaults.items():
        monkeypatch.setattr(f"orchestration.tasks.{attr}", stub)


# ---------------------------------------------------------------------------


def test_jobs_registry_contains_exactly_the_nine_expected_names():
    assert set(JOBS.keys()) == EXPECTED_JOB_NAMES


def test_all_jobs_take_no_arguments():
    for name, fn in JOBS.items():
        params = inspect.signature(fn).parameters
        assert params == {}, f"{name} takes arguments: {list(params)}"


def test_all_jobs_return_int(scratch_db, monkeypatch):
    _stub_every_underlying_dependency(
        monkeypatch,
        _count_stale_port_calls=lambda: 1,
        bdi_scraper=lambda: 2,
        wci_scraper=lambda: 3,
        _normalizer_run_all=lambda: 4,
        _targets_run_all=lambda: 5,
        _signals_run_all=lambda: 6,
        _models_run_all=lambda: 7,
        _alerts_run_all=lambda: 8,
        _heartbeat_run_all=lambda: 9,
    )

    for name, fn in JOBS.items():
        result = fn()
        assert isinstance(result, int) and not isinstance(result, bool), (
            f"{name} returned {result!r} ({type(result)}), not a plain int"
        )


def test_normalizer_job_writes_exactly_one_job_run_row(scratch_db, monkeypatch):
    # Stub the real sub-steps normalizer.run_all() calls internally — not
    # run_all() itself — so the real orchestration in feature_builder.py
    # executes for real. If a future change decorates any of these
    # sub-steps with @job, this test exercises that decoration and must
    # go red.
    monkeypatch.setattr("normalizer.port_resolver.run", lambda: {"resolved": 0})
    monkeypatch.setattr("normalizer.vessel_normalizer.run", lambda: {"smoothed": 0})
    monkeypatch.setattr("normalizer.port_summary_builder.run", lambda: 0)
    monkeypatch.setattr("normalizer.feature_builder.build", lambda: 3)

    result = JOBS["normalizer-nightly"]()

    assert result == 3
    rows = _fetch_all(scratch_db, "SELECT job_name FROM job_runs")
    assert rows == [("normalizer-nightly",)]


def test_rows_written_reflects_rows_written_not_examined(scratch_db, monkeypatch):
    # Simulates a scraper that examined 50 candidate posts but only wrote
    # 3 new rows — the wrapper must record 3, never 50.
    examined = 50
    written = 3
    assert examined != written

    monkeypatch.setattr("orchestration.tasks.bdi_scraper", lambda: written)

    JOBS["bdi-daily"]()

    rows = _fetch_all(
        scratch_db, "SELECT rows_written FROM job_runs WHERE job_name = %s", ("bdi-daily",)
    )
    assert rows == [(written,)]


def test_port_call_refresh_has_no_ais_thread_reference():
    source = (
        inspect.getsource(tasks.port_call_refresh)
        + inspect.getsource(tasks._count_stale_port_calls)
    ).lower()
    for forbidden in ("ais_thread", "watchdog", "ais stream thread", "ais daemon thread"):
        assert forbidden not in source, f"found forbidden reference {forbidden!r}"


def test_zero_rows_written_is_recorded_as_zero_not_null(scratch_db, monkeypatch):
    monkeypatch.setattr("orchestration.tasks.bdi_scraper", lambda: 0)

    JOBS["bdi-daily"]()

    rows = _fetch_all(
        scratch_db, "SELECT rows_written FROM job_runs WHERE job_name = %s", ("bdi-daily",)
    )
    assert rows == [(0,)]
    assert rows[0][0] is not None


def test_failing_job_records_failed_row_with_job_name(scratch_db, monkeypatch):
    def boom():
        raise RuntimeError("scraper exploded")

    monkeypatch.setattr("orchestration.tasks.wci_scraper", boom)

    with pytest.raises(RuntimeError):
        JOBS["wci-weekly"]()

    rows = _fetch_all(
        scratch_db,
        "SELECT job_name, status FROM job_runs WHERE job_name = %s",
        ("wci-weekly",),
    )
    assert rows == [("wci-weekly", "failed")]
