"""
test_rederive.py — enforcement tests for normalizer/rederive.py and
migrations/0003_port_calls_derived.sql.

Runs against a disposable scratch database: the real migrations
(0001-0003) applied via orchestration.migrate, plus a minimal test-local
`vessels` / `ports` / `positions` / `port_calls` schema this file creates
itself — the real versions are defined in schema.sql, out of scope for
this commit and unnecessary for these tests (no TimescaleDB hypertable
semantics, no vessel_type/nav_status enums; nav_status is plain TEXT
here, which is all normalizer.vessel_normalizer._smoothed_transitions
ever compares against).

Every rederive.run_all()/classify()/phantom_spike_overlap() call in this
file passes an explicit session bound to the scratch engine — see
rederive.run_all()'s own docstring for why: clients.base.Session binds to
whatever DATABASE_URL was set at its first import anywhere in this
pytest process, not necessarily this file's own scratch database, so
skipping that and using clients.base.Session directly would silently
break test isolation. No test in this file touches real production data.
"""

from __future__ import annotations

import ast
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg2
import pytest
from psycopg2 import sql
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from normalizer import rederive
from normalizer import vessel_normalizer
from orchestration.migrate import run_migrations

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_MIGRATIONS_DIR = REPO_ROOT / "migrations"

UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)

PORT_CODE = "TESTP"
PORT_LAT, PORT_LON = 0.0, 0.0
SEA_LAT, SEA_LON = 20.0, 20.0  # ~2,700km from the port — well outside PROXIMITY_KM


def _admin_database_url() -> str:
    return os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql://admin:password@localhost:5432/postgres",
    )


@pytest.fixture
def scratch_db(monkeypatch):
    admin_url = _admin_database_url()
    db_name = f"rederive_test_{uuid.uuid4().hex[:12]}"

    admin_conn = psycopg2.connect(admin_url)
    admin_conn.autocommit = True
    with admin_conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    admin_conn.close()

    parts = urlsplit(admin_url)
    scratch_url = urlunsplit(parts._replace(path=f"/{db_name}"))

    setup_conn = psycopg2.connect(scratch_url)
    setup_conn.autocommit = True
    with setup_conn.cursor() as cur:
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        cur.execute(
            """
            CREATE TABLE vessels (
                vessel_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                mmsi TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE ports (
                unlocode CHAR(5) PRIMARY KEY,
                lat DOUBLE PRECISION,
                lon DOUBLE PRECISION
            )
            """
        )
    setup_conn.close()

    # 0003 needs `vessels` to exist first (its FK) — the real migration
    # is applied here exactly as it would be against the real database.
    run_migrations(scratch_url, migrations_dir=REAL_MIGRATIONS_DIR)

    setup_conn = psycopg2.connect(scratch_url)
    setup_conn.autocommit = True
    with setup_conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE positions (
                position_id UUID NOT NULL DEFAULT uuid_generate_v4(),
                vessel_id   UUID NOT NULL REFERENCES vessels(vessel_id),
                ts          TIMESTAMPTZ NOT NULL,
                lat         DOUBLE PRECISION NOT NULL,
                lon         DOUBLE PRECISION NOT NULL,
                nav_status  TEXT,
                source      TEXT NOT NULL DEFAULT 'test',
                PRIMARY KEY (position_id, ts)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE port_calls (
                call_id       UUID NOT NULL DEFAULT uuid_generate_v4(),
                vessel_id     UUID NOT NULL REFERENCES vessels(vessel_id),
                port_unlocode CHAR(5) NOT NULL,
                arrived_at    TIMESTAMPTZ NOT NULL,
                departed_at   TIMESTAMPTZ,
                source        TEXT NOT NULL,
                PRIMARY KEY (call_id, arrived_at)
            )
            """
        )
        cur.execute(
            "INSERT INTO ports (unlocode, lat, lon) VALUES (%s, %s, %s)",
            (PORT_CODE, PORT_LAT, PORT_LON),
        )
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


def _insert_vessel(database_url: str) -> str:
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO vessels (mmsi) VALUES (%s) RETURNING vessel_id",
                (f"MMSI{uuid.uuid4().hex[:9]}",),
            )
            vessel_id = cur.fetchone()[0]
        conn.commit()
        return str(vessel_id)
    finally:
        conn.close()


def _insert_positions(database_url: str, vessel_id: str, pings: list[tuple]) -> None:
    """pings: list of (ts, lat, lon, nav_status)."""
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            for ts, lat, lon, nav_status in pings:
                cur.execute(
                    "INSERT INTO positions (vessel_id, ts, lat, lon, nav_status) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (vessel_id, ts, lat, lon, nav_status),
                )
        conn.commit()
    finally:
        conn.close()


def _insert_port_call(database_url, vessel_id, port_unlocode, arrived_at, departed_at=None, source="aisstream"):
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO port_calls (vessel_id, port_unlocode, arrived_at, departed_at, source) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING call_id",
                (vessel_id, port_unlocode, arrived_at, departed_at, source),
            )
            call_id = cur.fetchone()[0]
        conn.commit()
        return str(call_id)
    finally:
        conn.close()


def _insert_port_calls_derived(database_url, run_id, vessel_id, port_unlocode, arrived_at, supporting_ping_count=6):
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO port_calls_derived "
                "(vessel_id, port_unlocode, arrived_at, source, derivation_run_id, supporting_ping_count) "
                "VALUES (%s, %s, %s, 'rederive', %s, %s)",
                (vessel_id, port_unlocode, arrived_at, str(run_id), supporting_ping_count),
            )
        conn.commit()
    finally:
        conn.close()


def _underway_ping(ts):
    return (ts, SEA_LAT, SEA_LON, "underway_engine")


def _moored_ping(ts):
    return (ts, PORT_LAT, PORT_LON, "moored")


def _port_calls_snapshot(database_url):
    rows = _fetch_all(
        database_url,
        "SELECT call_id, vessel_id, port_unlocode, arrived_at, departed_at, source "
        "FROM port_calls ORDER BY call_id",
    )
    return sorted(rows)


# ---------------------------------------------------------------------------
# Part 1/2 — derivation
# ---------------------------------------------------------------------------


def test_no_write_touches_port_calls(scratch_db, scratch_session):
    vessel_id = _insert_vessel(scratch_db)
    _insert_port_call(scratch_db, vessel_id, PORT_CODE, BASE, BASE + timedelta(hours=5))
    _insert_port_call(scratch_db, vessel_id, PORT_CODE, BASE + timedelta(days=2), None)

    # Synthetic positions producing real derived activity, so the
    # derivation actually does something — a no-op run proves nothing.
    pings = [_underway_ping(BASE + timedelta(minutes=20 * i)) for i in range(6)]
    pings += [_moored_ping(BASE + timedelta(minutes=20 * i)) for i in range(6, 12)]
    _insert_positions(scratch_db, vessel_id, pings)

    before = _port_calls_snapshot(scratch_db)

    result = rederive.run_all(session=scratch_session)
    assert result["arrivals"] + result["departures"] > 0, "test setup produced no activity to guard against"

    after = _port_calls_snapshot(scratch_db)
    assert before == after, "port_calls must be byte-identical before and after a rederive run"


def test_already_moored_vessel_at_window_start_emits_no_arrival(scratch_db, scratch_session):
    vessel_id = _insert_vessel(scratch_db)
    # Stopped from the very first ping onward — no underway pings at all.
    pings = [_moored_ping(BASE + timedelta(minutes=20 * i)) for i in range(20)]
    _insert_positions(scratch_db, vessel_id, pings)

    result = rederive.run_all(session=scratch_session)

    assert result["arrivals"] == 0
    rows = _fetch_all(
        scratch_db, "SELECT COUNT(*) FROM port_calls_derived WHERE vessel_id = %s", (vessel_id,)
    )
    assert rows[0][0] == 0


def test_transition_spanning_window_boundary_emits_one_arrival(scratch_db, scratch_session):
    vessel_id = _insert_vessel(scratch_db)

    # 10 underway pings, continuously 20 minutes apart, ending 20 minutes
    # before the moored sequence starts — establishes prev_state=False
    # with margin, but *without* opening a >6h gap (GAP_THRESHOLD_HOURS)
    # that would make this test exercise gap-handling instead of the
    # boundary-carry behavior it's actually meant to isolate.
    boundary = BASE + timedelta(hours=24)
    moored_start = boundary - timedelta(minutes=10)
    pings = [_underway_ping(moored_start - timedelta(minutes=20 * (10 - i))) for i in range(10)]
    # 6 moored pings straddling the boundary: one before, five after. The
    # majority vote flips on the 4th moored ping in the window, which
    # lands after the boundary — this only gets detected correctly if the
    # carried tail from chunk 1 survives into chunk 2.
    pings += [_moored_ping(moored_start + timedelta(minutes=20 * i)) for i in range(6)]
    _insert_positions(scratch_db, vessel_id, pings)

    result = rederive.run_all(chunk_days=1, session=scratch_session)

    assert result["arrivals"] == 1
    rows = _fetch_all(
        scratch_db,
        "SELECT arrived_at FROM port_calls_derived WHERE vessel_id = %s",
        (vessel_id,),
    )
    assert len(rows) == 1
    (arrived_at,) = rows[0]
    assert arrived_at > boundary, "the transition should land after the chunk boundary, per this test's construction"


def test_gap_exceeding_threshold_yields_unknown_not_carried_state(scratch_db, scratch_session):
    vessel_id = _insert_vessel(scratch_db)

    # Underway -> stopped: a legitimate arrival.
    pings = [_underway_ping(BASE + timedelta(minutes=20 * i)) for i in range(6)]
    pings += [_moored_ping(BASE + timedelta(minutes=20 * i)) for i in range(6, 12)]

    # A gap well past GAP_THRESHOLD_HOURS (6h), then fresh underway pings.
    # Without gap handling this reads as a departure the moment the feed
    # resumes; with it, the prior state is unknown and no departure fires.
    resume = BASE + timedelta(minutes=20 * 11) + timedelta(hours=rederive.GAP_THRESHOLD_HOURS + 2)
    pings += [_underway_ping(resume + timedelta(minutes=20 * i)) for i in range(8)]
    _insert_positions(scratch_db, vessel_id, pings)

    result = rederive.run_all(session=scratch_session)

    assert result["arrivals"] == 1
    assert result["departures"] == 0, "a transition spanning the data gap must not be emitted as a departure"
    assert result["gaps"] >= 1


# ---------------------------------------------------------------------------
# Part 2 — idempotency
# ---------------------------------------------------------------------------


def test_rerun_same_run_id_is_idempotent(scratch_db, scratch_session):
    vessel_id = _insert_vessel(scratch_db)
    pings = [_underway_ping(BASE + timedelta(minutes=20 * i)) for i in range(6)]
    pings += [_moored_ping(BASE + timedelta(minutes=20 * i)) for i in range(6, 12)]
    pings += [_underway_ping(BASE + timedelta(minutes=20 * i)) for i in range(12, 18)]
    _insert_positions(scratch_db, vessel_id, pings)

    run_id = uuid.uuid4()
    rederive.run_all(derivation_run_id=run_id, session=scratch_session)
    first_count = _fetch_all(scratch_db, "SELECT COUNT(*) FROM port_calls_derived")[0][0]
    assert first_count > 0

    rederive.run_all(derivation_run_id=run_id, session=scratch_session)
    second_count = _fetch_all(scratch_db, "SELECT COUNT(*) FROM port_calls_derived")[0][0]

    assert second_count == first_count


# ---------------------------------------------------------------------------
# Part 3 — comparison
# ---------------------------------------------------------------------------


def test_comparison_classifies_all_three_categories(scratch_db, scratch_session):
    run_id = uuid.uuid4()

    # stored-and-derivable: a stored call with a matching derived row.
    v1 = _insert_vessel(scratch_db)
    _insert_port_call(scratch_db, v1, PORT_CODE, BASE)
    _insert_port_calls_derived(scratch_db, run_id, v1, PORT_CODE, BASE + timedelta(minutes=30))

    # stored-but-not-derivable: a phantom, no derived counterpart at all.
    v2 = _insert_vessel(scratch_db)
    _insert_port_call(scratch_db, v2, PORT_CODE, BASE + timedelta(days=1))

    # derivable-but-not-stored: a derived row with no stored counterpart.
    v3 = _insert_vessel(scratch_db)
    _insert_port_calls_derived(scratch_db, run_id, v3, PORT_CODE, BASE + timedelta(days=2))

    result = rederive.classify(scratch_session, run_id)

    assert result["stored_and_derivable"] == 1
    assert result["stored_but_not_derivable"] == 1
    assert result["derivable_but_not_stored"] == 1

    matched = {
        r[0] for r in _fetch_all(
            scratch_db,
            "SELECT matched_port_call_id FROM port_calls_derived "
            "WHERE derivation_run_id = %s AND matched_port_call_id IS NOT NULL",
            (str(run_id),),
        )
    }
    unmatched_stored = {
        r[0] for r in _fetch_all(
            scratch_db,
            "SELECT call_id FROM port_calls WHERE call_id::text != ALL(%s)",
            (list(str(m) for m in matched) or ["00000000-0000-0000-0000-000000000000"],),
        )
    }
    assert not (matched & unmatched_stored), "no port_calls row may land in both categories"


def test_tolerance_boundary(scratch_db, scratch_session):
    run_id = uuid.uuid4()
    tol = timedelta(hours=rederive.ARRIVAL_TOLERANCE_HOURS)

    v_within = _insert_vessel(scratch_db)
    stored_at = BASE
    _insert_port_call(scratch_db, v_within, PORT_CODE, stored_at)
    _insert_port_calls_derived(scratch_db, run_id, v_within, PORT_CODE, stored_at + tol - timedelta(minutes=1))

    v_outside = _insert_vessel(scratch_db)
    stored_at2 = BASE + timedelta(days=1)
    _insert_port_call(scratch_db, v_outside, PORT_CODE, stored_at2)
    _insert_port_calls_derived(scratch_db, run_id, v_outside, PORT_CODE, stored_at2 + tol + timedelta(minutes=1))

    result = rederive.classify(scratch_session, run_id)

    assert result["stored_and_derivable"] == 1
    assert result["stored_but_not_derivable"] == 1
    assert result["derivable_but_not_stored"] == 1


# ---------------------------------------------------------------------------
# Reuse enforcement
# ---------------------------------------------------------------------------


def test_imports_vessel_normalizer_rather_than_reimplementing():
    assert rederive._smoothed_transitions is vessel_normalizer._smoothed_transitions
    assert rederive.WINDOW_SIZE == vessel_normalizer.WINDOW_SIZE
    assert rederive.PROXIMITY_KM == vessel_normalizer.PROXIMITY_KM
    assert rederive.DEDUP_HOURS == vessel_normalizer.DEDUP_HOURS
    assert rederive.MIN_DWELL_HOURS == vessel_normalizer.MIN_DWELL_HOURS

    source = Path(rederive.__file__).read_text()
    tree = ast.parse(source, filename=rederive.__file__)

    imported_from_vessel_normalizer = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "normalizer.vessel_normalizer":
            imported_from_vessel_normalizer.update(alias.name for alias in node.names)

    assert "_smoothed_transitions" in imported_from_vessel_normalizer, (
        "rederive.py must import the majority-vote logic from vessel_normalizer, not reimplement it"
    )

    local_function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "_smoothed_transitions" not in local_function_names, (
        "rederive.py defines its own _smoothed_transitions — this shadows/duplicates "
        "vessel_normalizer's instead of reusing the imported one"
    )
