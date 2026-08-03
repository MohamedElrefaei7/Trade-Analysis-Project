"""
port_summary_builder.py — aggregate raw port_calls into port_daily_summary.

`port_calls` holds one row per vessel arrival/departure event, per port,
across every port AIS tracks. `port_daily_summary` is the daily-per-port
grain feature_builder._fetch_port_daily() expects. This module is the
missing link between the two:

    vessels_in_port  — distinct vessels with an open or overlapping call
                        on that calendar day
    avg_wait_hours   — mean dwell time (duration_hours) of calls that
                        departed on that day
    container/bulk/tanker_count — arrivals that day, broken out by
                        vessels.vessel_type
    arrivals/departures — raw daily counts

Only emits rows for (port, date) combinations derived from port_calls, so
this never collides with rows written by other sources — e.g.
clients/scraper.py's Port of LA TEU scraper, which owns USLAX rows
exclusively (a real, separate published-statistics feed, not AIS-derived).
"""

from sqlalchemy import text

from clients.base import Session, logger

_AGGREGATE_SQL = text(
    """
    WITH bounds AS (
        SELECT port_unlocode,
               MIN(arrived_at)::date AS start_date,
               GREATEST(
                   MAX(arrived_at),
                   MAX(COALESCE(departed_at, arrived_at)),
                   NOW()
               )::date AS end_date
        FROM port_calls
        GROUP BY port_unlocode
    ),
    days AS (
        SELECT b.port_unlocode, gs::date AS date
        FROM bounds b
        CROSS JOIN LATERAL generate_series(b.start_date, b.end_date, INTERVAL '1 day') gs
    ),
    in_port AS (
        SELECT d.port_unlocode, d.date, COUNT(DISTINCT pc.vessel_id) AS vessels_in_port
        FROM days d
        JOIN port_calls pc
          ON pc.port_unlocode = d.port_unlocode
         AND pc.arrived_at < d.date + INTERVAL '1 day'
         AND (pc.departed_at IS NULL OR pc.departed_at >= d.date)
        GROUP BY d.port_unlocode, d.date
    ),
    arrivals AS (
        SELECT pc.port_unlocode, DATE(pc.arrived_at) AS date,
               COUNT(*)                                                AS arrivals,
               COUNT(*) FILTER (WHERE v.vessel_type = 'container')     AS container_count,
               COUNT(*) FILTER (WHERE v.vessel_type = 'bulk_carrier')  AS bulk_count,
               COUNT(*) FILTER (WHERE v.vessel_type = 'tanker')        AS tanker_count
        FROM port_calls pc
        JOIN vessels v ON v.vessel_id = pc.vessel_id
        GROUP BY pc.port_unlocode, DATE(pc.arrived_at)
    ),
    departures AS (
        SELECT port_unlocode, DATE(departed_at) AS date,
               COUNT(*)            AS departures,
               AVG(duration_hours) AS avg_wait_hours
        FROM port_calls
        WHERE departed_at IS NOT NULL
        GROUP BY port_unlocode, DATE(departed_at)
    )
    SELECT d.port_unlocode, d.date,
           COALESCE(ip.vessels_in_port, 0)  AS vessels_in_port,
           dep.avg_wait_hours,
           COALESCE(arr.container_count, 0) AS container_count,
           COALESCE(arr.bulk_count, 0)      AS bulk_count,
           COALESCE(arr.tanker_count, 0)    AS tanker_count,
           COALESCE(arr.arrivals, 0)        AS arrivals,
           COALESCE(dep.departures, 0)      AS departures
    FROM days d
    LEFT JOIN in_port    ip  ON ip.port_unlocode  = d.port_unlocode AND ip.date  = d.date
    LEFT JOIN arrivals   arr ON arr.port_unlocode = d.port_unlocode AND arr.date = d.date
    LEFT JOIN departures dep ON dep.port_unlocode = d.port_unlocode AND dep.date = d.date
    ORDER BY d.port_unlocode, d.date
    """
)

_UPSERT_SQL = text(
    """
    INSERT INTO port_daily_summary
        (port_unlocode, date, vessels_in_port, avg_wait_hours,
         container_count, bulk_count, tanker_count, arrivals, departures)
    VALUES
        (:port_unlocode, :date, :vessels_in_port, :avg_wait_hours,
         :container_count, :bulk_count, :tanker_count, :arrivals, :departures)
    ON CONFLICT (port_unlocode, date) DO UPDATE SET
        vessels_in_port = EXCLUDED.vessels_in_port,
        avg_wait_hours  = EXCLUDED.avg_wait_hours,
        container_count = EXCLUDED.container_count,
        bulk_count      = EXCLUDED.bulk_count,
        tanker_count    = EXCLUDED.tanker_count,
        arrivals        = EXCLUDED.arrivals,
        departures      = EXCLUDED.departures
    """
)


def run() -> int:
    """Roll port_calls up into port_daily_summary. Returns rows upserted."""
    with Session() as session:
        rows = [dict(r) for r in session.execute(_AGGREGATE_SQL).mappings().all()]
        if not rows:
            logger.info("port_summary_builder: no port_calls to aggregate")
            return 0
        session.execute(_UPSERT_SQL, rows)
        session.commit()

    logger.info("port_summary_builder: %d port-day rows upserted", len(rows))
    return len(rows)
