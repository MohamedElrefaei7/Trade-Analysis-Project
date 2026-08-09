"""
test_jobs.py — enforcement tests for orchestration/jobs.py's @job decorator.

Each test runs against a disposable scratch database on the local Postgres
server, created and dropped per test (with migration 0001 applied via
orchestration.migrate to create job_runs) — never `mydb`, the frozen
Phase 2 rollback volume (see CONTEXT.md).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg2
import pytest
from psycopg2 import sql

from orchestration.jobs import job
from orchestration.migrate import run_migrations

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
    db_name = f"orch_jobs_test_{uuid.uuid4().hex[:12]}"

    admin_conn = psycopg2.connect(admin_url)
    admin_conn.autocommit = True
    with admin_conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    admin_conn.close()

    parts = urlsplit(admin_url)
    scratch_url = urlunsplit(parts._replace(path=f"/{db_name}"))

    # migrations/0003_port_calls_derived.sql FKs to vessels(vessel_id) —
    # not created by any migration (it's part of schema.sql), so a
    # minimal stand-in is required before run_migrations() gets there.
    setup_conn = psycopg2.connect(scratch_url)
    setup_conn.autocommit = True
    with setup_conn.cursor() as cur:
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        cur.execute("CREATE TABLE vessels (vessel_id UUID PRIMARY KEY DEFAULT uuid_generate_v4())")
    setup_conn.close()

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


def test_success_writes_success_row_with_rows_written(scratch_db):
    @job("test_success_job")
    def do_work():
        return 42

    result = do_work()

    assert result == 42
    rows = _fetch_all(
        scratch_db,
        "SELECT status, rows_written, finished_at FROM job_runs WHERE job_name = %s",
        ("test_success_job",),
    )
    assert len(rows) == 1
    status, rows_written, finished_at = rows[0]
    assert status == "success"
    assert rows_written == 42
    assert finished_at is not None


def test_exception_writes_failed_row(scratch_db):
    @job("test_failure_job")
    def do_work():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        do_work()

    rows = _fetch_all(
        scratch_db,
        "SELECT status, error_message FROM job_runs WHERE job_name = %s",
        ("test_failure_job",),
    )
    assert len(rows) == 1
    status, error_message = rows[0]
    assert status == "failed"
    assert "ValueError" in error_message


def test_exception_propagates(scratch_db):
    @job("test_propagate_job")
    def do_work():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        do_work()


def test_job_row_survives_wrapped_transaction_rollback(scratch_db):
    setup_conn = psycopg2.connect(scratch_db)
    setup_conn.autocommit = True
    with setup_conn.cursor() as cur:
        cur.execute("CREATE TABLE rollback_marker (id INTEGER)")
    setup_conn.close()

    @job("test_rollback_job")
    def do_work():
        work_conn = psycopg2.connect(scratch_db)
        try:
            with work_conn.cursor() as cur:
                cur.execute("INSERT INTO rollback_marker (id) VALUES (1)")
            raise RuntimeError("simulated failure mid-transaction")
        finally:
            # Closing without commit() implicitly rolls back this
            # connection's transaction — the row above never persists.
            work_conn.close()

    with pytest.raises(RuntimeError):
        do_work()

    marker_rows = _fetch_all(scratch_db, "SELECT * FROM rollback_marker")
    assert marker_rows == []

    job_rows = _fetch_all(
        scratch_db, "SELECT status FROM job_runs WHERE job_name = %s", ("test_rollback_job",)
    )
    assert job_rows == [("failed",)]


def test_running_row_exists_during_execution(scratch_db):
    seen = {}

    @job("test_running_visibility_job")
    def do_work():
        check_conn = psycopg2.connect(scratch_db)
        try:
            with check_conn.cursor() as cur:
                cur.execute(
                    "SELECT status FROM job_runs WHERE job_name = %s",
                    ("test_running_visibility_job",),
                )
                seen["rows"] = cur.fetchall()
        finally:
            check_conn.close()
        return 1

    do_work()

    assert seen["rows"] == [("running",)]


def test_status_check_constraint_rejects_unknown_value(scratch_db):
    conn = psycopg2.connect(scratch_db)
    try:
        with pytest.raises(psycopg2.IntegrityError):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO job_runs (run_id, job_name, started_at, status) "
                    "VALUES (gen_random_uuid(), 'x', now(), 'succeeded')"
                )
    finally:
        conn.rollback()
        conn.close()


def test_non_int_return_records_null_rows_written(scratch_db):
    @job("test_non_int_return_job")
    def do_work():
        return {"wrote": 5}

    do_work()

    rows = _fetch_all(
        scratch_db,
        "SELECT rows_written FROM job_runs WHERE job_name = %s",
        ("test_non_int_return_job",),
    )
    assert rows == [(None,)]
