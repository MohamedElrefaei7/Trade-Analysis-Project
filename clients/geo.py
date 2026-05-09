"""
geo.py — shared geospatial utilities.

One source of truth for great-circle distance, the ports cache, and
nearest-port lookups. Imported by clients.aisstream, normalizer.vessel_normalizer,
and normalizer.port_resolver.
"""

import math
from typing import NamedTuple

from sqlalchemy import text


class Port(NamedTuple):
    unlocode: str
    lat: float
    lon: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_ports(session) -> list[Port]:
    """Return every port with valid coordinates from the ports table."""
    rows = session.execute(
        text(
            "SELECT unlocode, lat, lon FROM ports "
            "WHERE lat IS NOT NULL AND lon IS NOT NULL"
        )
    ).fetchall()
    return [Port(r.unlocode, r.lat, r.lon) for r in rows]


def nearest_port(
    lat: float,
    lon: float,
    ports: list[Port],
    max_km: float | None = None,
) -> tuple[str | None, float]:
    """
    Find the closest port to (lat, lon).

    Returns (unlocode, distance_km). When `max_km` is set and no port is
    within that radius, returns (None, distance_to_closest).
    """
    best_code: str | None = None
    best_dist = float("inf")
    for p in ports:
        d = haversine_km(lat, lon, p.lat, p.lon)
        if d < best_dist:
            best_dist, best_code = d, p.unlocode
    if max_km is not None and best_dist > max_km:
        return None, best_dist
    return best_code, best_dist
