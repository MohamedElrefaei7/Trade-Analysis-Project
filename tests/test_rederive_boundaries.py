"""
test_rederive_boundaries.py — chunk-boundary regression tests for
normalizer/rederive.py at the real CHUNK_DAYS=1, written for the Part 3
validation described in the 2026-08-10 rederivation-audit prompt.

test_rederive.py's own `test_transition_spanning_window_boundary_emits_one_arrival`
was written when CHUNK_DAYS was still 7 — one boundary per week. At
CHUNK_DAYS=1, the derivation walks a boundary every single day, so the
carried-state handoff (_VesselState.tail / .prev_state, carried in
normalizer/rederive.py's `vessel_states` dict across chunks) is exercised
roughly 30x more often over the same history. These three tests extend
that existing case: a stop landing solidly *before* a boundary and staying
stopped past it (no arrival should be duplicated when the following
chunk re-observes the same steady state), the same carried-state path
repeated across two consecutive boundaries instead of one, and the
mirror-image departure transition straddling a boundary.

Same disposable-scratch-database approach as test_rederive.py, duplicated
here rather than imported — this project's convention (see
test_rederive.py's own docstring, and CONTEXT.md's 2026-08-09 entry on
why five other scratch_db fixtures were each given their own minimal
schema rather than a shared one) is that each test file creates exactly
the schema its own tests need, nothing shared.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg2
import pytest
from psycopg2 import sql
from sqlalchemy import create_engine

from normalizer import rederive
from orchestration.migrate import run_migrations

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_MIGRATIONS_DIR = REPO_ROOT / "migrations"

UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)  # midnight UTC, so BASE + N days lands exactly on a CHUNK_DAYS=1 boundary

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
    db_name = f"rederive_boundary_test_{uuid.uuid4().hex[:12]}"

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
    setup_conn.close()

    run_migrations(scratch_url, migrations_dir=REAL_MIGRATIONS_DIR)

    setup_conn = psycopg2.connect(scratch_url)
    setup_conn.autocommit = True
    with setup_conn.cursor() as cur:
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
    from sqlalchemy.orm import sessionmaker

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


def _underway_ping(ts):
    return (ts, SEA_LAT, SEA_LON, "underway_engine")


def _moored_ping(ts):
    return (ts, PORT_LAT, PORT_LON, "moored")


# ---------------------------------------------------------------------------
# Part 3 — CHUNK_DAYS=1 boundary regressions
# ---------------------------------------------------------------------------


def test_stop_before_midnight_still_stopped_after_emits_one_arrival(scratch_db, scratch_session):
    """A vessel that starts going moored at 23:40 — one ping before
    midnight, the rest after, so the majority vote's 6-ping window itself
    straddles the boundary — and then stays moored well into the next
    chunk must emit exactly one arrival: not zero (the flip getting lost
    because the carried tail from chunk 1 didn't survive into chunk 2)
    and not two (the following chunk mistaking "still moored" for a
    fresh transition because prev_state wasn't carried in as True).
    Only a flip whose window itself spans the boundary is actually at
    risk here — see test_vessel_stopped_across_three_consecutive_chunks'
    docstring for why a flip that completes with margin before the
    boundary doesn't exercise this at all."""
    vessel_id = _insert_vessel(scratch_db)

    boundary = BASE + timedelta(days=1)  # midnight, the CHUNK_DAYS=1 boundary
    moored_start = boundary - timedelta(minutes=20)  # 23:40

    # 10 underway warm-up pings, ending exactly where the moored run starts.
    pings = [_underway_ping(moored_start - timedelta(minutes=20 * (10 - i))) for i in range(10)]
    # 30 moored pings, 20 minutes apart: the first (23:40) lands in this
    # chunk, the rest land after midnight. The majority vote flips true
    # on the 6th one (01:20, in the next chunk) — only detected correctly
    # if the carried tail survives the handoff — then the vessel stays
    # moored for another ~8 hours, well past the flip, to prove nothing
    # fires a second time once it does.
    pings += [_moored_ping(moored_start + timedelta(minutes=20 * i)) for i in range(30)]
    _insert_positions(scratch_db, vessel_id, pings)

    result = rederive.run_all(chunk_days=1, session=scratch_session)

    assert result["arrivals"] == 1
    assert result["departures"] == 0
    rows = _fetch_all(
        scratch_db,
        "SELECT arrived_at FROM port_calls_derived WHERE vessel_id = %s",
        (vessel_id,),
    )
    assert len(rows) == 1
    (arrived_at,) = rows[0]
    assert arrived_at > boundary, "the flip only completes once the window has enough post-boundary pings"


def test_vessel_stopped_across_three_consecutive_chunks_emits_one_arrival(scratch_db, scratch_session):
    """The same boundary-straddling flip as the test above — the risk is
    entirely in getting that one flip right, since a majority vote that
    finishes with margin before a boundary needs nothing carried into the
    next chunk at all, and a state that's already settled into "moored"
    survives an ordinary chunk handoff whether or not the carry is
    dropped (prev_state resets to None, which the cold-start rule already
    treats as "no transition" either way — the same rule Commit 4 added
    for the daemon). What CHUNK_DAYS=1 changes isn't that risk, it's how
    often a vessel's dwell happens to straddle a boundary at all — the
    routine case now, not the rare one — so this extends the same flip
    across two consecutive boundaries (three chunks) instead of one, to
    prove the carry keeps working the same way the second time, not just
    the first."""
    vessel_id = _insert_vessel(scratch_db)

    boundary1 = BASE + timedelta(days=1)
    moored_start = boundary1 - timedelta(minutes=20)  # 23:40, same flip as the test above

    pings = [_underway_ping(moored_start - timedelta(minutes=20 * (10 - i))) for i in range(10)]
    # 110 moored pings, 20 minutes apart (~36.7 hours): the flip straddles
    # boundary1 exactly as above, and the dwell continues on past
    # boundary2 (day 2->3) into the third chunk.
    pings += [_moored_ping(moored_start + timedelta(minutes=20 * i)) for i in range(110)]
    _insert_positions(scratch_db, vessel_id, pings)

    result = rederive.run_all(chunk_days=1, session=scratch_session)

    assert result["arrivals"] == 1
    assert result["departures"] == 0
    assert result["gaps"] == 0, "continuous 20-minute pings must never register as a data gap"
    rows = _fetch_all(
        scratch_db,
        "SELECT arrived_at FROM port_calls_derived WHERE vessel_id = %s",
        (vessel_id,),
    )
    assert len(rows) == 1


def test_departure_spanning_boundary_stamps_one_departure(scratch_db, scratch_session):
    """The underway-transition mirror of
    test_transition_spanning_window_boundary_emits_one_arrival: a vessel
    solidly moored well before the boundary, then departing (underway)
    in a run of pings straddling the boundary — one before, five after,
    so the majority vote only flips once the window has enough
    post-boundary pings. Detecting this correctly, instead of missing it
    or splitting it, requires the moored state (not just the arrival
    row) and the pre-boundary ping tail both to survive the chunk
    handoff."""
    vessel_id = _insert_vessel(scratch_db)

    boundary = BASE + timedelta(days=1)
    warmup_end = boundary - timedelta(hours=8)

    # Underway warm-up, ending where the moored run begins.
    pings = [_underway_ping(warmup_end - timedelta(minutes=20 * (10 - i))) for i in range(10)]
    # Moored for 6 hours (18 pings, 20 min apart) — firmly established,
    # with margin, well before the boundary-straddling underway run below.
    pings += [_moored_ping(warmup_end + timedelta(minutes=20 * i)) for i in range(18)]

    # 6 underway pings straddling the boundary: one before, five after —
    # same construction as the existing arrival-boundary test, mirrored
    # for a departure. The majority vote flips on the 4th underway ping
    # in the window, which lands after the boundary.
    underway_start = boundary - timedelta(minutes=10)
    pings += [_underway_ping(underway_start + timedelta(minutes=20 * i)) for i in range(6)]
    _insert_positions(scratch_db, vessel_id, pings)

    result = rederive.run_all(chunk_days=1, session=scratch_session)

    assert result["arrivals"] == 1
    assert result["departures"] == 1
    rows = _fetch_all(
        scratch_db,
        "SELECT arrived_at, departed_at FROM port_calls_derived WHERE vessel_id = %s",
        (vessel_id,),
    )
    assert len(rows) == 1
    arrived_at, departed_at = rows[0]
    assert departed_at is not None
    assert departed_at > boundary, "the departure flip should land after the chunk boundary, per this test's construction"
