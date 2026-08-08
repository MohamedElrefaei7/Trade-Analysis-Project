"""
jobs.py — the @job(name) decorator: records every invocation of a
scheduled function into job_runs (see migrations/0001_job_runs.sql).
"""

from __future__ import annotations

import functools
import os
import uuid
from datetime import datetime, timezone

import psycopg2

_MAX_ERROR_MESSAGE_LENGTH = 2000


def job(name: str):
    """
    Decorate a job function so every call records a job_runs row: a
    'running' row is committed before the wrapped function starts, then
    updated to 'success' or 'failed' on completion — a hung or killed job
    leaves a stale, alertable 'running' row instead of no row at all. The
    running/finished writes use their own database connection, committed
    independently of whatever the wrapped function does with its own
    session, so a rollback inside the wrapped function's own transaction
    can never erase the record of its own failure. Exceptions are always
    re-raised after being recorded — never swallowed.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            run_id = uuid.uuid4()
            started_at = datetime.now(timezone.utc)
            _record_start(name, run_id, started_at)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                _record_finish(run_id, "failed", None, _format_error(exc))
                raise
            rows_written = result if isinstance(result, int) and not isinstance(result, bool) else None
            _record_finish(run_id, "success", rows_written, None)
            return result
        return wrapper
    return decorator


def _connect():
    database_url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    return conn


def _record_start(job_name: str, run_id: uuid.UUID, started_at: datetime) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO job_runs (run_id, job_name, started_at, status) "
                "VALUES (%s, %s, %s, 'running')",
                (str(run_id), job_name, started_at),
            )
    finally:
        conn.close()


def _record_finish(
    run_id: uuid.UUID,
    status: str,
    rows_written: int | None,
    error_message: str | None,
) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE job_runs SET status = %s, finished_at = %s, rows_written = %s, "
                "error_message = %s WHERE run_id = %s",
                (status, datetime.now(timezone.utc), rows_written, error_message, str(run_id)),
            )
    finally:
        conn.close()


def _format_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_MESSAGE_LENGTH]
