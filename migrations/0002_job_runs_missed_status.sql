-- Extends job_runs.status to accept 'missed': a misfired job never
-- invokes the @job-decorated function at all, so nothing gets written by
-- the normal running/success/failed path — the job is invisible in
-- job_runs, indistinguishable from "never scheduled." worker/main.py's
-- EVENT_JOB_MISSED listener writes this row directly, outside the
-- decorator, precisely to make that outcome visible and distinguishable
-- from both a real failure and no run at all. See CLAUDE.md § Scheduling.
--
-- The existing CHECK constraint's name is looked up rather than assumed —
-- it's Postgres's auto-generated name from an unnamed inline CHECK in
-- 0001, and this migration should not depend on remembering that detail
-- correctly by hand.

DO $$
DECLARE
    existing_constraint text;
BEGIN
    SELECT conname INTO existing_constraint
    FROM pg_constraint
    WHERE conrelid = 'job_runs'::regclass
      AND contype = 'c'
      AND pg_get_constraintdef(oid) LIKE '%status%';

    IF existing_constraint IS NULL THEN
        RAISE EXCEPTION
            'job_runs: no existing CHECK constraint on status found — refusing to proceed blind';
    END IF;

    EXECUTE format('ALTER TABLE job_runs DROP CONSTRAINT %I', existing_constraint);
END $$;

ALTER TABLE job_runs ADD CONSTRAINT job_runs_status_check
    CHECK (status IN ('running', 'success', 'failed', 'missed'));
