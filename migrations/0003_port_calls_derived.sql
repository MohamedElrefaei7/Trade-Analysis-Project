-- port_calls_derived: a parallel table for the port_calls re-derivation
-- (normalizer/rederive.py). port_calls itself is never written, updated,
-- or deleted by this migration or by rederive.py — see CLAUDE.md § 9 and
-- rederive.py's own module docstring for why. This table mirrors
-- port_calls's columns exactly (same generated duration_hours, same
-- origin/dest columns, deliberately unpopulated by rederive.py — kept
-- only for shape parity with port_calls) plus three provenance columns:
--
--   derivation_run_id     — so multiple re-derivation runs can coexist
--                           and be compared against each other, not just
--                           against port_calls
--   supporting_ping_count — how many pings backed the transition
--   matched_port_call_id  — the port_calls row this derived row
--                           corresponds to within tolerance, backfilled
--                           by the comparison step (Part 3); NULL until
--                           then, and NULL permanently for a transition
--                           the live daemon genuinely missed. No FK to
--                           port_calls(call_id): port_calls's primary
--                           key is the composite (call_id, arrived_at),
--                           so call_id alone isn't unique and can't be
--                           an FK target — the tolerance match is an
--                           application-level join, not a referential
--                           constraint, deliberately.
--
-- CREATE EXTENSION IF NOT EXISTS below (not just relying on schema.sql)
-- because this migration must also succeed against disposable
-- test-scratch databases that were created via CREATE DATABASE + this
-- migrations/ directory only, never schema.sql — idempotent no-ops on
-- the real server, where both extensions already exist.
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE port_calls_derived (
    call_id                 UUID NOT NULL DEFAULT uuid_generate_v4(),
    vessel_id               UUID NOT NULL REFERENCES vessels(vessel_id),
    port_unlocode           CHAR(5) NOT NULL,
    arrived_at              TIMESTAMPTZ NOT NULL,
    departed_at             TIMESTAMPTZ,
    duration_hours          FLOAT                        -- derived, populated on departure
        GENERATED ALWAYS AS (
            EXTRACT(EPOCH FROM (departed_at - arrived_at)) / 3600.0
        ) STORED,
    origin_unlocode         CHAR(5),
    dest_unlocode           CHAR(5),
    source                  TEXT NOT NULL,
    derivation_run_id       UUID NOT NULL,
    supporting_ping_count   INTEGER,
    matched_port_call_id    UUID,
    PRIMARY KEY (call_id, arrived_at),
    -- The idempotency key: a rerun of the same derivation_run_id
    -- recomputes identical (vessel, arrived_at) transitions from
    -- unchanged positions data, so ON CONFLICT (...) DO NOTHING makes a
    -- rerun a safe no-op instead of a duplicate insert.
    UNIQUE (derivation_run_id, vessel_id, arrived_at)
);

-- Matches port_calls's own chunk_time_interval (schema.sql) for
-- consistent comparison-query performance characteristics — not
-- positions's 1-week interval, which is a different table's tuning.
SELECT create_hypertable(
    'port_calls_derived', 'arrived_at',
    chunk_time_interval => INTERVAL '1 month',
    migrate_data => true
);

CREATE INDEX idx_port_calls_derived_run    ON port_calls_derived(derivation_run_id);
CREATE INDEX idx_port_calls_derived_vessel ON port_calls_derived(vessel_id, arrived_at DESC);
CREATE INDEX idx_port_calls_derived_match  ON port_calls_derived(matched_port_call_id);
