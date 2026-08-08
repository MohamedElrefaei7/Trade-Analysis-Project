-- job_runs: written by orchestration.jobs.job(). status is constrained
-- here, not just by application code, so a typo'd status string fails
-- loudly instead of making the eventual Phase 11 heartbeat's
-- `WHERE status = 'success'` query silently stop matching. Plain table,
-- not a hypertable — a few thousand rows a year doesn't need chunking.

CREATE TABLE job_runs (
    run_id        UUID PRIMARY KEY,
    job_name      TEXT NOT NULL,
    started_at    TIMESTAMPTZ NOT NULL,
    finished_at   TIMESTAMPTZ,
    status        TEXT NOT NULL CHECK (status IN ('running','success','failed')),
    rows_written  INTEGER,
    error_message TEXT
);

CREATE INDEX ON job_runs (job_name, started_at DESC);
