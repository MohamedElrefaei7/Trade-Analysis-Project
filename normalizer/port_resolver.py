"""
port_resolver.py — map raw port strings + coordinates to UN/LOCODE.

Three-layer resolution:
    1. Exact lookup of hand-curated aliases (case-insensitive).
    2. Fuzzy match against the ports table using rapidfuzz.
    3. Coordinate snap: if a (lat, lon) sits within SNAP_RADIUS_KM of a
       port in the registry, return that port's code regardless of what
       the raw name field said — AIS destination strings are entered by
       hand on the bridge and are frequently wrong.

Returns None when no layer resolves — better a NULL than a wrong port.

Batch entry point `run()` backfills port_calls rows whose port_unlocode
or origin_unlocode is NULL or ill-formed.
"""

from functools import lru_cache

from rapidfuzz import fuzz, process
from sqlalchemy import text

from clients.base import Session, logger
from clients.geo import Port, nearest_port

# ── Tunables ─────────────────────────────────────────────────────────────────

SNAP_RADIUS_KM = 50.0    # coordinate snap radius
FUZZY_THRESHOLD = 85     # rapidfuzz score 0-100; below this, reject the match

# Hand-curated aliases built up over time as the ingest reveals new variants.
# Keys are normalised (upper, stripped). Extend as new mismatches are spotted.
EXACT_ALIASES: dict[str, str] = {
    "ROTTERDAM": "NLRTM",
    "NL RTM": "NLRTM",
    "PORT OF ROTTERDAM": "NLRTM",
    "SHANGHAI": "CNSHA",
    "CN SHA": "CNSHA",
    "PORT OF SHANGHAI": "CNSHA",
    "SINGAPORE": "SGSIN",
    "SG SIN": "SGSIN",
    "LOS ANGELES": "USLAX",
    "PORT OF LOS ANGELES": "USLAX",
    "LA": "USLAX",
    "LONG BEACH": "USLGB",
    "PORT OF LONG BEACH": "USLGB",
    "HAMBURG": "DEHAM",
    "BUSAN": "KRPUS",
    "PUSAN": "KRPUS",
    "YOKOHAMA": "JPYOK",
    "FELIXSTOWE": "GBFXT",
    "DUBAI": "AEDXB",
    "JEBEL ALI": "AEDXB",
    "PORT OF JEBEL ALI": "AEDXB",
}


# ── Registry access ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_registry() -> list[tuple[str, str, float | None, float | None]]:
    """Return [(unlocode, name, lat, lon), …] from the ports table."""
    with Session() as s:
        rows = s.execute(
            text("SELECT unlocode, name, lat, lon FROM ports")
        ).fetchall()
    return [(r.unlocode, r.name, r.lat, r.lon) for r in rows]


# ── Resolution layers ────────────────────────────────────────────────────────

def _exact(raw: str) -> str | None:
    key = raw.strip().upper()
    if len(key) == 5 and key.isalpha():
        # already a LOCODE — trust it if it exists in the registry
        if any(u == key for u, *_ in _load_registry()):
            return key
    return EXACT_ALIASES.get(key)


def _fuzzy(raw: str) -> str | None:
    registry = _load_registry()
    choices = {unlocode: name.upper() for unlocode, name, *_ in registry}
    if not choices:
        return None
    match = process.extractOne(
        raw.strip().upper(), choices, scorer=fuzz.WRatio
    )
    if match is None:
        return None
    _name, score, unlocode = match
    return unlocode if score >= FUZZY_THRESHOLD else None


def _coord_snap(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None:
        return None
    ports = [
        Port(unlocode, plat, plon)
        for unlocode, _name, plat, plon in _load_registry()
        if plat is not None and plon is not None
    ]
    code, _dist = nearest_port(lat, lon, ports, max_km=SNAP_RADIUS_KM)
    return code


# ── Public API ───────────────────────────────────────────────────────────────

def resolve(
    raw: str | None,
    lat: float | None = None,
    lon: float | None = None,
) -> str | None:
    """
    Resolve a raw port string (and optional coordinates) to a UN/LOCODE.
    Returns None when nothing crosses the confidence threshold.
    """
    if raw:
        hit = _exact(raw) or _fuzzy(raw)
        if hit:
            return hit
    return _coord_snap(lat, lon)


def run() -> dict:
    """
    Backfill port_calls rows whose port_unlocode is NULL or not a 5-letter
    LOCODE. Writes directly; safe to re-run.
    """
    stats = {"scanned": 0, "updated": 0, "still_unresolved": 0}

    with Session() as s:
        rows = s.execute(
            text(
                """
                SELECT pc.call_id, pc.arrived_at,
                       pc.port_unlocode  AS raw_port,
                       p.lat AS vlat, p.lon AS vlon
                FROM port_calls pc
                LEFT JOIN LATERAL (
                    SELECT lat, lon
                    FROM positions
                    WHERE vessel_id = pc.vessel_id
                      AND ts BETWEEN pc.arrived_at - INTERVAL '2 hours'
                                 AND pc.arrived_at + INTERVAL '2 hours'
                    ORDER BY ABS(EXTRACT(EPOCH FROM (ts - pc.arrived_at)))
                    LIMIT 1
                ) p ON TRUE
                WHERE pc.port_unlocode IS NULL
                   OR LENGTH(pc.port_unlocode) <> 5
                """
            )
        ).fetchall()

        stats["scanned"] = len(rows)
        for r in rows:
            code = resolve(r.raw_port, r.vlat, r.vlon)
            if code is None:
                stats["still_unresolved"] += 1
                continue
            s.execute(
                text(
                    """
                    UPDATE port_calls
                       SET port_unlocode = :code
                     WHERE call_id = :cid
                       AND arrived_at = :ts
                    """
                ),
                {"code": code, "cid": r.call_id, "ts": r.arrived_at},
            )
            stats["updated"] += 1
        s.commit()

    logger.info(
        "port_resolver: scanned=%d updated=%d unresolved=%d",
        stats["scanned"], stats["updated"], stats["still_unresolved"],
    )
    return stats
