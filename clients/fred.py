"""
fred.py — FRED (Federal Reserve Economic Data) client.

Fetches time-series observations and writes them to economic_benchmarks
with series_id prefixed as "FRED:<series_id>" (e.g. "FRED:GDP").

Endpoint used:
    GET https://api.stlouisfed.org/fred/series/observations
        ?series_id=<id>&api_key=<key>&file_type=json&sort_order=desc&limit=<n>

Usage:
    from clients.fred import run
    run()                       # fetch all configured series
    run(["GDP", "DEXUSAL"])     # fetch specific series
"""

import os
from datetime import datetime, timezone

import requests
from sqlalchemy import text

from .base import Session, latest_ts, logger, retry

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Series to collect at startup. Extend this list freely.
# lag_days: known publication delay (days after observation date).
SERIES_CONFIG: dict[str, dict] = {
    "GDP": {
        "frequency": "quarterly",
        "unit": "billions of chained 2012 USD",
        "lag_days": 28,       # BEA advance estimate ~4 weeks after quarter end
    },
    "BOPGSTB": {
        "frequency": "monthly",
        "unit": "millions of USD",
        "lag_days": 35,       # BEA trade-in-goods release ~5 weeks after month end
    },
    "DEXUSAL": {
        "frequency": "daily",
        "unit": "AUD per USD",
        "lag_days": 1,
    },
    "DEXCHUS": {
        "frequency": "daily",
        "unit": "CNY per USD",
        "lag_days": 1,
    },
    "WPUSI0200": {
        "frequency": "monthly",
        "unit": "index 2000=100",
        "lag_days": 30,
    },
}


@retry(max_attempts=3)
def _fetch_observations(series_id: str, limit: int = 200) -> list[dict]:
    """Call the FRED observations endpoint and return raw observation dicts."""
    resp = requests.get(
        _BASE_URL,
        params={
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "observations" not in data:
        raise ValueError(f"Unexpected FRED response for {series_id}: {data}")
    return data["observations"]


def ingest_series(series_id: str) -> int:
    """
    Fetch and store one FRED series. Returns the number of rows inserted.
    Only inserts observations newer than what's already in the DB.
    """
    cfg = SERIES_CONFIG.get(series_id, {})
    frequency = cfg.get("frequency", "daily")
    unit = cfg.get("unit", "")
    lag_days = cfg.get("lag_days", 0)
    series_id_full = f"FRED:{series_id}"

    observations = _fetch_observations(series_id)

    with Session() as session:
        latest = latest_ts(session, "economic_benchmarks", "series_id", series_id_full)

        rows = []
        for obs in observations:
            # FRED uses "." as a placeholder for missing data
            if obs.get("value") == ".":
                continue

            obs_date = datetime.fromisoformat(obs["date"]).replace(
                tzinfo=timezone.utc
            )

            # Skip rows we already have
            if latest and obs_date <= latest:
                continue

            rows.append(
                {
                    "series_id": series_id_full,
                    "source": "fred",
                    "ts": obs_date,
                    "value": float(obs["value"]),
                    "unit": unit,
                    "frequency": frequency,
                    "lag_days": lag_days,
                }
            )

        if rows:
            session.execute(
                text(
                    """
                    INSERT INTO economic_benchmarks
                        (series_id, source, ts, value, unit, frequency, lag_days)
                    VALUES
                        (:series_id, :source, :ts, :value, :unit,
                         CAST(:frequency AS data_frequency), :lag_days)
                    """
                ),
                rows,
            )
            session.commit()

    logger.info("FRED:%-12s  inserted %d rows", series_id, len(rows))
    return len(rows)


def run(series_ids: list[str] | None = None) -> None:
    """Fetch all configured series (or a subset) and store them."""
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY not set — skipping FRED ingest")
        return

    targets = series_ids if series_ids is not None else list(SERIES_CONFIG)
    total = 0
    for sid in targets:
        try:
            total += ingest_series(sid)
        except Exception as exc:
            logger.error("FRED:%s ingest failed — %s", sid, exc)
    logger.info("FRED ingest complete — %d total rows inserted", total)


if __name__ == "__main__":
    run()
