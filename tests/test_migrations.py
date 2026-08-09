"""
test_migrations.py — enforcement tests for orchestration/migrate.py.

Each test runs against a disposable scratch database on the local Postgres
server, created and dropped per test — never `mydb`, the frozen Phase 2
rollback volume (see CONTEXT.md).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg2
import pytest
from psycopg2 import sql

from orchestration.migrate import MigrationError, run_migrations

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_MIGRATIONS_DIR = REPO_ROOT / "migrations"


def _admin_database_url() -> str:
    return os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql://admin:password@localhost:5432/postgres",
    )


@pytest.fixture
def scratch_db():
    admin_url = _admin_database_url()
    db_name = f"orch_migrate_test_{uuid.uuid4().hex[:12]}"

    admin_conn = psycopg2.connect(admin_url)
    admin_conn.autocommit = True
    with admin_conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    admin_conn.close()

    parts = urlsplit(admin_url)
    scratch_url = urlunsplit(parts._replace(path=f"/{db_name}"))

    # migrations/0003_port_calls_derived.sql FKs to vessels(vessel_id) —
    # not created by any migration (it's part of schema.sql), so a
    # minimal stand-in is required before any test's own run_migrations()
    # call against REAL_MIGRATIONS_DIR reaches it.
    setup_conn = psycopg2.connect(scratch_url)
    setup_conn.autocommit = True
    with setup_conn.cursor() as cur:
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        cur.execute("CREATE TABLE vessels (vessel_id UUID PRIMARY KEY DEFAULT uuid_generate_v4())")
    setup_conn.close()

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


def _table_exists(database_url: str, table_name: str) -> bool:
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                (table_name,),
            )
            return cur.fetchone()[0]
    finally:
        conn.close()


def _fetch_all(database_url: str, query: str):
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
    finally:
        conn.close()


def _write_migration(migrations_dir: Path, filename: str, sql_text: str) -> Path:
    path = migrations_dir / filename
    path.write_text(sql_text)
    return path


# ---------------------------------------------------------------------------


def _real_migration_filenames() -> list[str]:
    """The actual migrations/ directory's .sql files, in application order —
    computed rather than hardcoded so this file doesn't need editing every
    time a new real migration is added (0002, 0003, ...)."""
    return sorted(p.name for p in REAL_MIGRATIONS_DIR.iterdir() if p.suffix == ".sql")


def test_runner_applies_pending_and_records_version(scratch_db):
    expected = _real_migration_filenames()

    applied = run_migrations(scratch_db, migrations_dir=REAL_MIGRATIONS_DIR)

    assert applied == expected
    assert _table_exists(scratch_db, "job_runs")
    rows = _fetch_all(scratch_db, "SELECT filename FROM schema_migrations ORDER BY version")
    assert [r[0] for r in rows] == expected


def test_runner_is_idempotent(scratch_db):
    expected_count = len(_real_migration_filenames())
    run_migrations(scratch_db, migrations_dir=REAL_MIGRATIONS_DIR)
    second_run = run_migrations(scratch_db, migrations_dir=REAL_MIGRATIONS_DIR)

    assert second_run == []
    rows = _fetch_all(scratch_db, "SELECT version FROM schema_migrations")
    assert len(rows) == expected_count


def test_runner_aborts_on_modified_applied_migration(scratch_db, tmp_path):
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    _write_migration(migrations_dir, "0001_thing.sql", "CREATE TABLE thing (id INTEGER);")

    run_migrations(scratch_db, migrations_dir=migrations_dir)

    # Mutate the already-applied file's bytes, and add a new pending one.
    _write_migration(
        migrations_dir, "0001_thing.sql", "CREATE TABLE thing (id INTEGER, extra TEXT);"
    )
    _write_migration(migrations_dir, "0002_other.sql", "CREATE TABLE other (id INTEGER);")

    with pytest.raises(MigrationError):
        run_migrations(scratch_db, migrations_dir=migrations_dir)

    assert not _table_exists(scratch_db, "other")
    rows = _fetch_all(scratch_db, "SELECT version FROM schema_migrations")
    assert len(rows) == 1


def test_runner_refuses_schema_sql(scratch_db, tmp_path):
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    _write_migration(
        migrations_dir, "schema.sql", "CREATE TABLE should_never_exist (id INTEGER);"
    )

    with pytest.raises(MigrationError):
        run_migrations(scratch_db, migrations_dir=migrations_dir)

    assert not _table_exists(scratch_db, "should_never_exist")


def test_failed_migration_records_no_version(scratch_db, tmp_path):
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    _write_migration(migrations_dir, "0001_broken.sql", "THIS IS NOT VALID SQL AT ALL;")

    with pytest.raises(MigrationError):
        run_migrations(scratch_db, migrations_dir=migrations_dir)

    rows = _fetch_all(scratch_db, "SELECT version FROM schema_migrations")
    assert rows == []


def test_migrations_apply_in_zero_padded_order(scratch_db, tmp_path):
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    # Written to disk out of numeric order — a runner that sorted by mtime
    # or directory-listing order instead of the zero-padded filename would
    # apply 0010 first, which fails outright (order_log doesn't exist yet).
    _write_migration(
        migrations_dir,
        "0010_second.sql",
        "INSERT INTO order_log (name) VALUES ('0010_second.sql');",
    )
    _write_migration(
        migrations_dir,
        "0002_first.sql",
        "CREATE TABLE order_log (id SERIAL PRIMARY KEY, name TEXT); "
        "INSERT INTO order_log (name) VALUES ('0002_first.sql');",
    )

    applied = run_migrations(scratch_db, migrations_dir=migrations_dir)

    assert applied == ["0002_first.sql", "0010_second.sql"]
    rows = _fetch_all(scratch_db, "SELECT name FROM order_log ORDER BY id")
    assert [r[0] for r in rows] == ["0002_first.sql", "0010_second.sql"]
