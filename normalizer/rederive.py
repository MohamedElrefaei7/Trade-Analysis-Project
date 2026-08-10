"""
rederive.py — historical re-derivation of port_calls from positions, for
the phantom-arrival measurement `port_calls_derived` (migrations/0003)
exists for.

Context: port_calls holds 30,771 rows, up to 13,665 of them in hours
showing restart-spike signatures (thousands of arrivals in a single hour
against a normal tail of ~160). The cause — clients/aisstream.py's
_update_port_call() once treated "no cached prior vessel state" as "was
underway," so every daemon restart registered a fresh arrival for every
vessel already sitting in port — was found and fixed in Phase 3 Commit 4.
This module answers the question that fix couldn't answer on its own:
how many of the 30,771 rows already in the table are phantoms from before
the fix landed.

Three parts, matching the three functions below:

  1. derive_transitions() / run_all() — re-derive arrival/departure
     transitions from positions over the full history, writing them to
     port_calls_derived (never port_calls — see "Never touches
     port_calls" below).
  2. classify() — three-way comparison against the stored port_calls,
     backfilling port_calls_derived.matched_port_call_id.
  3. phantom_spike_overlap() — cross-tabulates the "stored but not
     derivable" category against the restart-spike hours, which is the
     actual finding this whole exercise exists to produce.

Never touches port_calls (decision 1 of the commit this shipped in)
------------------------------------------------------------------
port_calls has two legitimate writers by design (clients/aisstream.py
live, normalizer/vessel_normalizer.py nightly — see CLAUDE.md § Hard
invariants), and it's the only record of arrivals whose supporting
position pings may predate what's still in `positions`. Overwriting it
with a re-derivation would destroy exactly the rows this measurement
exists to identify, before anyone had seen the numbers. Every function in
this module only ever SELECTs from port_calls; every INSERT/UPDATE here
targets port_calls_derived.

Reuses vessel_normalizer.py rather than reimplementing it (decision 2)
------------------------------------------------------------------
The 6-ping sliding-window majority vote (_smoothed_transitions), the
30 km port-proximity rule (PROXIMITY_KM), the 24-hour arrival dedup
window (DEDUP_HOURS), and the sub-2-hour drive-by flag (MIN_DWELL_HOURS)
are imported from normalizer.vessel_normalizer, not copied — two
implementations of "what counts as an arrival" would make the comparison
below measure the gap between two of this project's own functions
instead of between derived and stored reality.

One piece of vessel_normalizer.py genuinely cannot be reused as-is:
_process_vessel()'s positions query is hardcoded to
`ts > NOW() - make_interval(hours => :hours)` — a rolling lookback from
the current time, not an arbitrary historical [start, end) window — and
it writes directly to port_calls, which this module must never do.
Per this commit's own instruction, that's reported here rather than
edited into vessel_normalizer.py: parameterizing its query is a
legitimate future change, but it belongs in its own commit with its own
regression test, since the nightly job depends on the current behavior.
rederive.py therefore runs its own positions query per chunk (below),
feeding the results through the imported, unmodified _smoothed_transitions.

Chunked processing with carried per-vessel state (decision 3)
------------------------------------------------------------------
38.5M position rows — peaking at over 17M in a single week — will not
fit in memory on a 2 GiB box. The full history is walked in CHUNK_DAYS
(1) windows, streamed from Postgres (execution_options(yield_per=...))
rather than fetched whole. Memory that persists across chunks is bounded
to one _VesselState per vessel seen: up to WINDOW_SIZE-1 raw pings (the
majority-vote window's warm-up tail) plus a single prev_state bool — a
few hundred bytes per vessel, not per row. A vessel that goes stopped on
a Sunday night and stays stopped into Monday's chunk must not register as
a fresh arrival on Monday; carrying the tail buffer and prev_state across
the chunk boundary and feeding (carry + new pings) through
_smoothed_transitions as one continuous sequence is what prevents that —
the same class of bug Commit 4 fixed for daemon restarts, reproduced here
across chunk boundaries if state isn't carried.

CHUNK_DAYS was 7 (calendar week) through 2026-08-09; changed to 1 after
that value was measured, live, to be the actual blocker (see CONTEXT.md's
2026-08-10 entry). A held server-side cursor (stream_results=True,
required to bound client memory the same way) runs its underlying plan
without parallel workers — confirmed directly via pg_stat_activity during
an active FETCH, no `parallel worker` backend rows appear — and pays a
roughly constant, large per-FETCH cost that does not shrink with a bigger
yield_per (measured: batch sizes of 2,000/20,000/50,000 against the same
15.6M-row chunk showed no meaningful improvement; three consecutive
50,000-row FETCHes from the same open cursor cost ~17-20s *each*, not
"slow first FETCH, fast the rest," ruling out a one-time
materialize-before-first-row explanation). The same week processed as six
CHUNK_DAYS=1 windows — same query, same cursor, same yield_per, only a
smaller per-window row count — completed in 443s total with no window
individually slow; the single busiest day in that week (3.66M rows) alone
took 88s. Root cause is not fully isolated beyond "the per-fetch cost
scales worse than linearly with the single query's total working set on
this box's ~600MB of available memory," but the fix that empirically
closes the gap needed no change to the fetch/cursor mechanics at all —
only to how much any single one of them is asked to hold.

Data gaps are a third state, never silently "no transition" (decision 4)
------------------------------------------------------------------
The history has a real ten-week hole (no positions chunk between
2026-05-21 and 2026-07-30; weekly volume degraded from multi-GB to 148 MB
before going silent). Carrying smoothed state across a gap like that
would manufacture a ten-week voyage the instant the feed returns. Any
gap between consecutive raw pings for a vessel — whether within one
chunk's data or spanning a chunk boundary — exceeding GAP_THRESHOLD_HOURS
(6h) splits that vessel's ping sequence into independent runs: prev_state
resets to None (unknown, matching Commit 4's cold-start rule — no
transition emitted) at the start of every run after the first, and each
gap is counted in the returned stats rather than silently absorbed.

Idempotent and resumable, keyed on derivation_run_id (decision 5)
------------------------------------------------------------------
Every write is scoped to `derivation_run_id` and keyed for idempotency:
arrivals upsert via ON CONFLICT (derivation_run_id, vessel_id,
arrived_at) DO NOTHING, and a departure-close only affects a row that's
still open, so re-running the exact same run_id after an interruption
recomputes identical, already-written rows as safe no-ops rather than
duplicates (see test_rerun_same_run_id_is_idempotent). Concretely,
"resumable" here means "safe to invoke again to completion with the same
run_id" — a rerun replays the full chunk walk from the start rather than
skipping to a persisted checkpoint; positions data doesn't change
between runs, so the replay is deterministic and its only cost is
recomputation, not double-writing. A new run_id always starts a fully
independent set of rows, coexisting with any prior run's in the same
table.

This is deliberately not a @job (orchestration/tasks.py): it's a one-off
historical analysis, not a scheduled unit, and registering it would put
it in worker/cadences.py::CADENCES and the heartbeat's per-job overdue
check, where a multi-hour one-time backfill does not belong.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from clients.base import Session, logger
from clients.geo import load_ports, nearest_port
from normalizer.vessel_normalizer import (
    DEDUP_HOURS,
    MIN_DWELL_HOURS,
    PROXIMITY_KM,
    WINDOW_SIZE,
    _smoothed_transitions,
)

# ── Tunables ─────────────────────────────────────────────────────────────────

CHUNK_DAYS = 1                   # was 7; measured to be the client-side wait-bound bottleneck — see module docstring
GAP_THRESHOLD_HOURS = 6.0       # a ping gap wider than this makes the prior state unknown, not carried
QUERY_BATCH_SIZE = 2000         # server-side cursor batch size for the per-chunk positions query
SOURCE_LABEL = "rederive"

# Why 2 hours: the live daemon stamps arrival at the ping where its
# majority vote flips; re-deriving over different window/chunk boundaries
# lands within a couple of hours of that, not exactly on it. A tolerance
# of zero would classify every real arrival as both phantom and missed.
ARRIVAL_TOLERANCE_HOURS = 2.0

# Matches the spec query this whole measurement is built to audit:
# hours with more than this many arrivals are restart-spike candidates.
RESTART_SPIKE_MIN_COUNT = 150


# ── Per-vessel carried state ─────────────────────────────────────────────────

@dataclass
class _VesselState:
    tail: list = field(default_factory=list)  # raw ping rows, always < WINDOW_SIZE long
    prev_state: bool | None = None


def _split_into_runs(pings: list, gap_threshold: timedelta) -> list[list]:
    """Split a time-ordered ping list into contiguous runs, breaking
    wherever the gap between consecutive pings exceeds gap_threshold.
    A gap between the carried tail's last ping and the first new ping of
    a chunk is just another internal gap in this combined list — handled
    uniformly, no special-casing needed."""
    if not pings:
        return []
    runs = [[pings[0]]]
    for prev, cur in zip(pings, pings[1:]):
        if (cur.ts - prev.ts) > gap_threshold:
            runs.append([])
        runs[-1].append(cur)
    return runs


# ── port_calls_derived I/O (never port_calls) ───────────────────────────────

def _has_recent_open_derived_call(session, run_id, vessel_id, port_code: str, arrived_at: datetime) -> bool:
    hit = session.execute(
        text(
            """
            SELECT 1 FROM port_calls_derived
             WHERE derivation_run_id = :rid
               AND vessel_id = :vid
               AND port_unlocode = :port
               AND arrived_at > :cutoff
             LIMIT 1
            """
        ),
        {
            "rid": str(run_id),
            "vid": vessel_id,
            "port": port_code,
            "cutoff": arrived_at - timedelta(hours=DEDUP_HOURS),
        },
    ).first()
    return hit is not None


def _insert_derived_arrival(session, run_id, vessel_id, port_code: str, ts: datetime, supporting_ping_count: int) -> bool:
    """Returns True if a new row was actually inserted (False on an
    idempotent conflict, e.g. a rerun of the same run_id)."""
    result = session.execute(
        text(
            """
            INSERT INTO port_calls_derived
                (vessel_id, port_unlocode, arrived_at, source, derivation_run_id, supporting_ping_count)
            VALUES (:vid, :port, :ts, :source, :rid, :spc)
            ON CONFLICT (derivation_run_id, vessel_id, arrived_at) DO NOTHING
            """
        ),
        {
            "vid": vessel_id, "port": port_code, "ts": ts,
            "source": SOURCE_LABEL, "rid": str(run_id), "spc": supporting_ping_count,
        },
    )
    return result.rowcount == 1


def _close_open_derived_call(session, run_id, vessel_id, ts: datetime) -> float | None:
    """Closes this vessel's most recent open derived call for this run.
    Returns the dwell in hours, or None if there was nothing open to
    close (including the idempotent rerun case — already closed)."""
    row = session.execute(
        text(
            """
            UPDATE port_calls_derived
               SET departed_at = :ts
             WHERE call_id = (
                SELECT call_id
                FROM port_calls_derived
                WHERE derivation_run_id = :rid
                  AND vessel_id = :vid
                  AND departed_at IS NULL
                ORDER BY arrived_at DESC
                LIMIT 1
             )
             RETURNING EXTRACT(EPOCH FROM (:ts - arrived_at)) / 3600.0 AS dwell
            """
        ),
        {"rid": str(run_id), "vid": vessel_id, "ts": ts},
    ).first()
    return row.dwell if row is not None else None


# ── Per-vessel, per-chunk processing ─────────────────────────────────────────

def _process_vessel_chunk(session, run_id, vessel_id, state: _VesselState, new_pings: list, ports) -> dict:
    stats = {"arrivals": 0, "departures": 0, "skipped_dupe": 0, "short_dwell": 0, "gaps": 0}

    combined = state.tail + new_pings
    if not combined:
        return stats

    runs = _split_into_runs(combined, timedelta(hours=GAP_THRESHOLD_HOURS))
    stats["gaps"] = max(0, len(runs) - 1)

    for i, run in enumerate(runs):
        # Only the first run continues the state carried from the
        # previous chunk; every run after an internal gap is a cold
        # start (decision 4) — no transition until _smoothed_transitions
        # has rebuilt enough context to judge a state at all.
        prev_state = state.prev_state if i == 0 else None

        for ts, lat, lon, is_stopped in _smoothed_transitions(run):
            if prev_state is None:
                prev_state = is_stopped
                continue

            if not prev_state and is_stopped:
                port_code, _dist = nearest_port(lat, lon, ports, max_km=PROXIMITY_KM)
                if port_code and not _has_recent_open_derived_call(session, run_id, vessel_id, port_code, ts):
                    if _insert_derived_arrival(session, run_id, vessel_id, port_code, ts, WINDOW_SIZE):
                        stats["arrivals"] += 1
                elif port_code:
                    stats["skipped_dupe"] += 1

            if prev_state and not is_stopped:
                dwell = _close_open_derived_call(session, run_id, vessel_id, ts)
                if dwell is not None:
                    stats["departures"] += 1
                    if dwell < MIN_DWELL_HOURS:
                        stats["short_dwell"] += 1

            prev_state = is_stopped

        state.prev_state = prev_state

    # Carry forward only the last run's tail — a run boundary is a
    # detected gap, so pings from before it must not bleed into the next
    # chunk's window as if they were adjacent.
    last_run = runs[-1] if runs else []
    state.tail = last_run[-(WINDOW_SIZE - 1):] if WINDOW_SIZE > 1 else []

    return stats


# ── Part 2: full-history re-derivation ───────────────────────────────────────

def run_all(
    derivation_run_id: uuid.UUID | str | None = None,
    chunk_days: int = CHUNK_DAYS,
    session=None,
) -> dict:
    """
    Re-derives arrival/departure transitions over the full positions
    history in chunk_days-wide, time-ordered chunks, writing only to
    port_calls_derived. Returns a summary dict; "arrivals"/"departures"
    count genuinely new rows written this call (idempotent re-inserts on
    a rerun of the same run_id are not double-counted — see
    _insert_derived_arrival/_close_open_derived_call).

    `session` is normally left as None, in which case this opens its own
    session against clients.base's engine — the real invocation path.
    Tests pass an explicit session bound to a scratch database instead:
    clients.base.Session is bound to whatever DATABASE_URL was set at
    that module's *first* import anywhere in the process (see
    test_tasks.py's own docstring), which — in a full pytest run where
    some earlier test file already imported it — is not guaranteed to be
    the scratch database a given test just created. Accepting an
    explicit session is what makes test_no_write_touches_port_calls
    actually prove isolation instead of silently validating against
    whatever database clients.base happened to bind to first.
    """
    run_id = uuid.UUID(str(derivation_run_id)) if derivation_run_id else uuid.uuid4()

    if session is not None:
        return _run_all_with_session(session, run_id, chunk_days)
    with Session() as owned_session:
        return _run_all_with_session(owned_session, run_id, chunk_days)


def _run_all_with_session(session, run_id: uuid.UUID, chunk_days: int) -> dict:
    totals = {
        "derivation_run_id": str(run_id),
        "chunks": 0, "vessels_seen": 0,
        "arrivals": 0, "departures": 0, "skipped_dupe": 0, "short_dwell": 0, "gaps": 0,
    }
    vessel_states: dict = {}

    ports = load_ports(session)
    bounds = session.execute(text("SELECT MIN(ts), MAX(ts) FROM positions")).first()
    if not bounds or bounds[0] is None:
        logger.info("rederive: positions is empty — nothing to do")
        return totals

    chunk_start = bounds[0].replace(hour=0, minute=0, second=0, microsecond=0)
    overall_end = bounds[1]
    step = timedelta(days=chunk_days)

    stmt = text(
        """
        SELECT vessel_id, ts, lat, lon, nav_status
        FROM positions
        WHERE ts >= :start AND ts < :end
        ORDER BY vessel_id, ts
        """
    ).execution_options(stream_results=True, yield_per=QUERY_BATCH_SIZE)

    while chunk_start <= overall_end:
        chunk_end = chunk_start + step
        totals["chunks"] += 1

        result = session.execute(stmt, {"start": chunk_start, "end": chunk_end})
        for vessel_id, group in itertools.groupby(result, key=lambda r: r.vessel_id):
            state = vessel_states.setdefault(vessel_id, _VesselState())
            new_pings = list(group)
            partial = _process_vessel_chunk(session, run_id, vessel_id, state, new_pings, ports)
            for k in ("arrivals", "departures", "skipped_dupe", "short_dwell", "gaps"):
                totals[k] += partial[k]

        session.commit()
        logger.info(
            "rederive: run_id=%s chunk %s..%s done (running totals: %s)",
            run_id, chunk_start, chunk_end, totals,
        )
        chunk_start = chunk_end

    totals["vessels_seen"] = len(vessel_states)
    logger.info("rederive: run_id=%s complete — %s", run_id, totals)
    return totals


# ── Part 3: three-way comparison ─────────────────────────────────────────────

def classify(session, derivation_run_id: uuid.UUID | str) -> dict:
    """
    Three-way comparison between port_calls (stored) and
    port_calls_derived for this run (derived), matching on
    (vessel_id, port_unlocode, arrived_at within ARRIVAL_TOLERANCE_HOURS).
    Backfills port_calls_derived.matched_port_call_id (never touches
    port_calls). Every port_calls row lands in exactly one of
    stored_and_derivable / stored_but_not_derivable; every
    port_calls_derived row for this run lands in exactly one of
    stored_and_derivable's matched set / derivable_but_not_stored.

      - stored_and_derivable: a real arrival — port_calls has a matching
        derived transition.
      - stored_but_not_derivable: the phantoms — a stored row with no
        supporting transition anywhere in positions.
      - derivable_but_not_stored: a real transition the live daemon
        missed. Expected to be nonzero: Commit 4's cold-start fix
        deliberately skips transitions with no prior state, and
        vessel_normalizer's nightly re-derivation is what was supposed
        to recover them.
    """
    rid = str(derivation_run_id)
    tol_seconds = ARRIVAL_TOLERANCE_HOURS * 3600

    session.execute(
        text(
            """
            WITH best_match AS (
                SELECT DISTINCT ON (pcd.call_id, pcd.arrived_at)
                    pcd.call_id AS derived_call_id,
                    pcd.arrived_at AS derived_arrived_at,
                    pc.call_id AS stored_call_id
                FROM port_calls_derived pcd
                JOIN port_calls pc
                  ON pc.vessel_id = pcd.vessel_id
                 AND pc.port_unlocode = pcd.port_unlocode
                 AND ABS(EXTRACT(EPOCH FROM (pcd.arrived_at - pc.arrived_at))) <= :tol
                WHERE pcd.derivation_run_id = :rid
                ORDER BY pcd.call_id, pcd.arrived_at,
                         ABS(EXTRACT(EPOCH FROM (pcd.arrived_at - pc.arrived_at)))
            )
            UPDATE port_calls_derived pcd
               SET matched_port_call_id = bm.stored_call_id
              FROM best_match bm
             WHERE pcd.call_id = bm.derived_call_id
               AND pcd.arrived_at = bm.derived_arrived_at
            """
        ),
        {"rid": rid, "tol": tol_seconds},
    )

    stored_and_derivable = session.execute(
        text(
            """
            SELECT COUNT(*) FROM port_calls pc
             WHERE EXISTS (
                SELECT 1 FROM port_calls_derived pcd
                 WHERE pcd.derivation_run_id = :rid AND pcd.matched_port_call_id = pc.call_id
             )
            """
        ),
        {"rid": rid},
    ).scalar_one()

    stored_but_not_derivable = session.execute(
        text(
            """
            SELECT COUNT(*) FROM port_calls pc
             WHERE NOT EXISTS (
                SELECT 1 FROM port_calls_derived pcd
                 WHERE pcd.derivation_run_id = :rid AND pcd.matched_port_call_id = pc.call_id
             )
            """
        ),
        {"rid": rid},
    ).scalar_one()

    derivable_but_not_stored = session.execute(
        text(
            """
            SELECT COUNT(*) FROM port_calls_derived
             WHERE derivation_run_id = :rid AND matched_port_call_id IS NULL
            """
        ),
        {"rid": rid},
    ).scalar_one()

    session.commit()

    return {
        "stored_and_derivable": stored_and_derivable,
        "stored_but_not_derivable": stored_but_not_derivable,
        "derivable_but_not_stored": derivable_but_not_stored,
    }


def phantom_spike_overlap(session, derivation_run_id: uuid.UUID | str) -> dict:
    """
    Cross-tabulates stored_but_not_derivable (the phantoms) against hours
    showing restart-spike signatures (> RESTART_SPIKE_MIN_COUNT arrivals
    in that hour). Call after classify() has backfilled
    matched_port_call_id. The prediction is that phantoms concentrate in
    spike hours; if they're spread evenly instead, the restart-bug
    explanation is wrong and phantoms_outside_spike_hours is the number
    that says so — report it, don't bury it under the total.
    """
    rid = str(derivation_run_id)

    spike_hours = {
        row.hr
        for row in session.execute(
            text(
                """
                SELECT date_trunc('hour', arrived_at) AS hr
                FROM port_calls
                GROUP BY 1
                HAVING COUNT(*) > :thr
                """
            ),
            {"thr": RESTART_SPIKE_MIN_COUNT},
        ).fetchall()
    }

    phantom_hours = session.execute(
        text(
            """
            SELECT date_trunc('hour', pc.arrived_at) AS hr, COUNT(*) AS n
            FROM port_calls pc
            WHERE NOT EXISTS (
                SELECT 1 FROM port_calls_derived pcd
                 WHERE pcd.derivation_run_id = :rid AND pcd.matched_port_call_id = pc.call_id
            )
            GROUP BY 1
            """
        ),
        {"rid": rid},
    ).fetchall()

    total_phantoms = sum(r.n for r in phantom_hours)
    in_spike = sum(r.n for r in phantom_hours if r.hr in spike_hours)

    return {
        "spike_hour_count": len(spike_hours),
        "total_phantoms": total_phantoms,
        "phantoms_in_spike_hours": in_spike,
        "phantoms_outside_spike_hours": total_phantoms - in_spike,
        "overlap_fraction": (in_spike / total_phantoms) if total_phantoms else None,
    }


if __name__ == "__main__":
    summary = run_all()
    print(summary)
    with Session() as s:
        print(classify(s, summary["derivation_run_id"]))
        print(phantom_spike_overlap(s, summary["derivation_run_id"]))
