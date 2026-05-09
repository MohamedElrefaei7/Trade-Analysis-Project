"""
vessel_normalizer.py — smooth raw AIS pings into clean port-call events.

Pipeline per vessel:

    1. Pull the last N hours of positions ordered by timestamp.
    2. Collapse nav_status into a binary "stopped" signal (anchored OR
       moored). Apply a majority-vote filter over a sliding 6-ping
       window — a single underway flicker does not flip the state back.
    3. When the smoothed signal transitions underway → stopped and the
       vessel is within PROXIMITY_KM of a known port, write an arrival
       row to port_calls (skipping if an open row already exists within
       the 24-hour dedup window).
    4. When the smoothed signal transitions stopped → underway, stamp
       `departed_at` on the vessel's most-recent open port_call.

`duration_hours` is a generated column in the schema, so we do not write
it directly. Calls shorter than MIN_DWELL_HOURS are flagged in the log
as likely false positives.
"""

from collections import deque
from datetime import datetime, timedelta

from sqlalchemy import text

from clients.base import Session, logger
from clients.geo import Port, load_ports, nearest_port

# ── Tunables ─────────────────────────────────────────────────────────────────

WINDOW_SIZE = 6             # pings considered in the majority vote
STOPPED_VOTES_REQUIRED = 4  # ≥4-of-6 stopped = truly stopped
PROXIMITY_KM = 30.0         # vessel within this radius of a port counts as arrived
DEDUP_HOURS = 24            # don't open a second call for the same (vessel, port) inside this window
MIN_DWELL_HOURS = 2.0       # calls shorter than this are flagged as drive-bys
LOOKBACK_HOURS = 48         # how far back to scan positions on each run

STOPPED_STATUSES = {"anchored", "moored"}


# ── Core ─────────────────────────────────────────────────────────────────────

def _smoothed_transitions(pings):
    """
    Yield (ts, lat, lon, is_stopped) rows with the smoothed binary state,
    filtering out noise via a sliding majority vote.
    """
    window: deque[bool] = deque(maxlen=WINDOW_SIZE)
    for p in pings:
        window.append(p.nav_status in STOPPED_STATUSES)
        if len(window) < WINDOW_SIZE:
            continue  # not enough history to decide yet
        is_stopped = sum(window) >= STOPPED_VOTES_REQUIRED
        yield p.ts, p.lat, p.lon, is_stopped


def _has_recent_open_call(session, vessel_id, port_code: str, arrived_at: datetime) -> bool:
    hit = session.execute(
        text(
            """
            SELECT 1 FROM port_calls
             WHERE vessel_id = :vid
               AND port_unlocode = :port
               AND arrived_at > :cutoff
             LIMIT 1
            """
        ),
        {
            "vid": vessel_id,
            "port": port_code,
            "cutoff": arrived_at - timedelta(hours=DEDUP_HOURS),
        },
    ).first()
    return hit is not None


def _process_vessel(session, vessel_id, ports: list[Port]) -> dict:
    stats = {"arrivals": 0, "departures": 0, "skipped_dupe": 0, "short_dwell": 0}

    pings = session.execute(
        text(
            """
            SELECT ts, lat, lon, nav_status
            FROM positions
            WHERE vessel_id = :vid
              AND ts > NOW() - make_interval(hours => :hours)
            ORDER BY ts
            """
        ),
        {"vid": vessel_id, "hours": LOOKBACK_HOURS},
    ).fetchall()

    if len(pings) < WINDOW_SIZE:
        return stats

    prev_state: bool | None = None
    for ts, lat, lon, is_stopped in _smoothed_transitions(pings):
        if prev_state is None:
            prev_state = is_stopped
            continue

        # underway → stopped : candidate arrival
        if not prev_state and is_stopped:
            port_code, _dist = nearest_port(lat, lon, ports, max_km=PROXIMITY_KM)
            if port_code and not _has_recent_open_call(session, vessel_id, port_code, ts):
                session.execute(
                    text(
                        """
                        INSERT INTO port_calls
                            (vessel_id, port_unlocode, arrived_at, source)
                        VALUES (:vid, :port, :ts, 'vessel_normalizer')
                        """
                    ),
                    {"vid": vessel_id, "port": port_code, "ts": ts},
                )
                stats["arrivals"] += 1
            elif port_code:
                stats["skipped_dupe"] += 1

        # stopped → underway : close the most-recent open call
        if prev_state and not is_stopped:
            updated = session.execute(
                text(
                    """
                    UPDATE port_calls
                       SET departed_at = :ts
                     WHERE call_id = (
                        SELECT call_id
                        FROM port_calls
                        WHERE vessel_id = :vid
                          AND departed_at IS NULL
                        ORDER BY arrived_at DESC
                        LIMIT 1
                     )
                     RETURNING EXTRACT(EPOCH FROM (:ts - arrived_at)) / 3600.0 AS dwell
                    """
                ),
                {"vid": vessel_id, "ts": ts},
            ).first()
            if updated is not None:
                stats["departures"] += 1
                if updated.dwell is not None and updated.dwell < MIN_DWELL_HOURS:
                    stats["short_dwell"] += 1

        prev_state = is_stopped

    return stats


# ── Public API ───────────────────────────────────────────────────────────────

def run() -> dict:
    """Normalize AIS status for every vessel with recent activity."""
    totals = {"vessels": 0, "arrivals": 0, "departures": 0,
              "skipped_dupe": 0, "short_dwell": 0}

    with Session() as s:
        ports = load_ports(s)
        vessels = s.execute(
            text(
                """
                SELECT DISTINCT vessel_id
                FROM positions
                WHERE ts > NOW() - make_interval(hours => :hours)
                """
            ),
            {"hours": LOOKBACK_HOURS},
        ).fetchall()

        for row in vessels:
            partial = _process_vessel(s, row.vessel_id, ports)
            totals["vessels"] += 1
            for k in ("arrivals", "departures", "skipped_dupe", "short_dwell"):
                totals[k] += partial[k]
        s.commit()

    logger.info(
        "vessel_normalizer: vessels=%d arrivals=%d departures=%d "
        "skipped_dupe=%d short_dwell=%d",
        totals["vessels"], totals["arrivals"], totals["departures"],
        totals["skipped_dupe"], totals["short_dwell"],
    )
    return totals
