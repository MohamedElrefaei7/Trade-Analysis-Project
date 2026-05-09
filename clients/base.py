"""
base.py — shared foundation for all API clients.

Provides:
  - SQLAlchemy engine + Session factory (reads DATABASE_URL from .env)
  - retry() decorator: exponential back-off, configurable attempts
  - latest_ts() helper: most recent ts for a (table, column, value) triple
  - module-level logger (level driven by LOG_LEVEL env var, default INFO)
"""

import functools
import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(name)-20s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("maritime")

# ── Database ──────────────────────────────────────────────────────────────────

_DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(
    _DATABASE_URL,
    pool_pre_ping=True,   # reconnect silently after idle-timeout drops
    pool_size=5,
    max_overflow=2,
)

# Use as:  with Session() as session:  session.execute(...); session.commit()
Session = sessionmaker(bind=engine)


# ── Query helpers ─────────────────────────────────────────────────────────────

def latest_ts(
    session,
    table: str,
    where_col: str,
    where_val,
    ts_col: str = "ts",
) -> datetime | None:
    """
    Return MAX(ts_col) from `table` filtered on a single (col = val) predicate.

    Used by ingest clients to skip rows already stored:
        latest_ts(session, "economic_benchmarks", "series_id", "FRED:GDP")
    """
    row = session.execute(
        text(f"SELECT MAX({ts_col}) FROM {table} WHERE {where_col} = :v"),
        {"v": where_val},
    ).fetchone()
    return row[0] if row and row[0] else None


# ── Retry decorator ───────────────────────────────────────────────────────────

def retry(max_attempts: int = 3, backoff_base: float = 2.0):
    """
    Retry a function on any exception with exponential back-off.

    Attempt delays: 1 s, 2 s, 4 s … (backoff_base ** attempt_index).
    Raises the final exception if all attempts are exhausted.

    Usage:
        @retry(max_attempts=3)
        def fetch_data(): ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        delay = backoff_base ** attempt
                        logger.warning(
                            "%s attempt %d/%d failed — %s. Retrying in %.0fs.",
                            fn.__qualname__, attempt + 1, max_attempts, exc, delay,
                        )
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
