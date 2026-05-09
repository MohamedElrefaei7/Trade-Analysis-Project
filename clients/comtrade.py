"""
comtrade.py — UN Comtrade API client.

Fetches monthly bilateral trade flow data and writes to economic_benchmarks
with series_id formatted as "COMTRADE:<reporter>-<partner>-<cmd>"
e.g. "COMTRADE:CN-US-85" (China exporting electronics to the US).

Endpoint used:
    GET https://comtradeapi.un.org/data/v1/get/C/M/HS
        ?reporterCode=<iso3>&partnerCode=<iso3>&cmdCode=<hs>&flowCode=X
    Header: Ocp-Apim-Subscription-Key: <key>

Docs: https://comtradeapi.un.org/

Usage:
    from clients.comtrade import run
    run()                   # fetch all configured flows, last 24 months
    run(months_back=12)     # fetch last 12 months
"""

import os
import time
from datetime import date, datetime, timezone
from dateutil.relativedelta import relativedelta

import requests
from sqlalchemy import text

from .base import Session, latest_ts, logger, retry

COMTRADE_KEY = os.environ.get("COMTRADE_SUBSCRIPTION_KEY", "")
_BASE_URL = "https://comtradeapi.un.org/data/v1/get/C/M/HS"

# ISO numeric codes used by Comtrade
_COUNTRY_CODE = {
    "CN": "156",   # China
    "US": "842",   # United States
    "AU": "036",   # Australia
    "BR": "076",   # Brazil
}

# (reporter_alpha2, partner_alpha2, hs_cmd, label_suffix)
TRADE_FLOWS: list[tuple[str, str, str]] = [
    ("CN", "US", "85"),   # China → US  : electronics
    ("CN", "US", "27"),   # China → US  : mineral fuels / oil
    ("AU", "CN", "26"),   # Australia → China : iron ore
    ("BR", "CN", "12"),   # Brazil → China : soybeans
]

_REQUEST_DELAY = 1.5   # seconds between API calls (rate-limit courtesy)


def _series_id(reporter: str, partner: str, cmd: str) -> str:
    return f"COMTRADE:{reporter}-{partner}-{cmd}"


def _period_range(months_back: int) -> list[str]:
    """Return YYYYMM strings for the last N months (most-recent first)."""
    today = date.today()
    periods = []
    for i in range(months_back):
        d = today - relativedelta(months=i + 1)   # offset by 1: last full month
        periods.append(d.strftime("%Y%m"))
    return periods


@retry(max_attempts=3)
def _fetch_flow(
    reporter: str, partner: str, cmd: str, period: str
) -> list[dict]:
    """
    Fetch one month of trade data for a reporter→partner→commodity combination.
    Returns a (possibly empty) list of data-row dicts.
    """
    resp = requests.get(
        _BASE_URL,
        headers={"Ocp-Apim-Subscription-Key": COMTRADE_KEY},
        params={
            "reporterCode": _COUNTRY_CODE[reporter],
            "partnerCode": _COUNTRY_CODE[partner],
            "cmdCode": cmd,
            "flowCode": "X",      # exports from reporter's perspective
            "period": period,
        },
        timeout=30,
    )
    if resp.status_code == 404:
        return []   # no data for this period, not an error
    resp.raise_for_status()
    return resp.json().get("data", [])


def ingest_flow(
    reporter: str, partner: str, cmd: str, months_back: int = 24
) -> int:
    """
    Fetch and store one trade flow for the last N months.
    Skips periods already present in the DB.
    Returns rows inserted.
    """
    sid = _series_id(reporter, partner, cmd)
    inserted = 0

    with Session() as session:
        latest = latest_ts(session, "economic_benchmarks", "series_id", sid)

    for period in _period_range(months_back):
        # period = YYYYMM → first day of that month as TS
        period_ts = datetime(
            int(period[:4]), int(period[4:]), 1, tzinfo=timezone.utc
        )
        if latest and period_ts <= latest:
            continue   # already stored

        data = _fetch_flow(reporter, partner, cmd, period)
        time.sleep(_REQUEST_DELAY)

        if not data:
            continue

        rows = []
        for row in data:
            value = row.get("primaryValue")
            if value is None:
                continue
            rows.append(
                {
                    "series_id": sid,
                    "source": "comtrade",
                    "ts": period_ts,
                    "value": float(value),
                    "unit": "USD",
                    "frequency": "monthly",
                    "lag_days": 45,   # Comtrade data typically 4-6 weeks late
                }
            )

        if rows:
            with Session() as session:
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
            inserted += len(rows)

    logger.info(
        "Comtrade %s-%s cmd%s  inserted %d rows",
        reporter, partner, cmd, inserted,
    )
    return inserted


def run(months_back: int = 24) -> None:
    """Fetch all configured trade flows and store them."""
    if not COMTRADE_KEY:
        logger.warning(
            "COMTRADE_SUBSCRIPTION_KEY not set — skipping Comtrade ingest"
        )
        return

    total = 0
    for reporter, partner, cmd in TRADE_FLOWS:
        try:
            total += ingest_flow(reporter, partner, cmd, months_back=months_back)
        except Exception as exc:
            logger.error(
                "Comtrade %s→%s cmd=%s failed — %s",
                reporter, partner, cmd, exc,
            )
    logger.info("Comtrade ingest complete — %d total rows inserted", total)


if __name__ == "__main__":
    run()
