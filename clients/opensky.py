"""
opensky.py — OpenSky Network REST client.

Polls the /states/all endpoint for each bounding box, filters for cargo /
heavy aircraft, and writes rows to flight_events. For each aircraft that
passes the filter, makes a second call to /flights/aircraft to resolve
its origin/destination airport — /states/all only carries live position
state, never flight-plan data, so this enrichment call is unavoidable.

Endpoints:
    GET https://opensky-network.org/api/states/all
        ?lamin=&lomin=&lamax=&lomax=

    GET https://opensky-network.org/api/flights/aircraft
        ?icao24=&begin=&end=
    Returns estDepartureAirport / estArrivalAirport as ICAO airport codes
    (4 letters, e.g. 'EHAM') — stored as-is in origin_icao/dest_icao.
    dest_icao is frequently null even on a successful lookup: OpenSky can
    only estimate the arrival airport once the aircraft has actually
    landed near one, and these lookups fire the moment we sight the
    aircraft mid-flight, so the landing generally hasn't happened yet.
    The [begin, end] window is capped at 30 days by the API; we use a
    narrow +/-12h window around the sighting. Enrichment is best-effort:
    a failed or empty lookup leaves origin_icao/dest_icao NULL rather
    than failing the whole poll.

State vector field order (index → meaning):
    0  icao24          hex transponder address
    1  callsign        padded to 8 chars
    2  origin_country
    3  time_position   unix timestamp of last position fix
    4  last_contact    unix timestamp of last ADS-B message
    5  longitude
    6  latitude
    7  baro_altitude   metres
    8  on_ground       bool
    9  velocity        m/s
    10 true_track      degrees from north
    11 vertical_rate   m/s
    12 sensors         (can be None)
    13 geo_altitude    metres
    14 squawk
    15 spi             special purpose indicator
    16 position_source  0=ADS-B, 1=ASTERIX, 2=MLAT
    17 category        ADS-B emitter category (4=large, 5=high-vortex, 6=heavy)

Only categories 4, 5, 6 (large/heavy aircraft) are written.
cargo_flag=True is set when the callsign matches a known all-cargo airline prefix.

Usage:
    from clients.opensky import run
    run()                   # poll all regions once
    run(["east_asia"])      # poll a single region
"""

import os
from datetime import datetime, timedelta, timezone

import requests
from sqlalchemy import text

from .base import Session, logger, retry

OPENSKY_USER = os.environ.get("OPENSKY_USER", "")
OPENSKY_PASS = os.environ.get("OPENSKY_PASS", "")
_BASE_URL = "https://opensky-network.org/api/states/all"
_FLIGHTS_URL = "https://opensky-network.org/api/flights/aircraft"
_ROUTE_LOOKUP_WINDOW = timedelta(hours=12)

# Bounding boxes: (min_lat, min_lon, max_lat, max_lon)
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "east_asia":      (20.0, 100.0,  50.0, 145.0),
    "north_atlantic": (35.0, -70.0,  65.0,  10.0),
    "indian_ocean":   (-10.0, 40.0,  30.0,  90.0),
    "trans_pacific":  (15.0, 130.0,  55.0, 180.0),
    "australia":      (-40.0, 110.0, -10.0, 160.0),
}

# ADS-B emitter categories that include large commercial / cargo aircraft
_CARGO_CATEGORIES: frozenset[int] = frozenset({4, 5, 6})

# ICAO 3-letter prefixes for confirmed all-cargo operators
_CARGO_CALLSIGN_PREFIXES: frozenset[str] = frozenset(
    {
        "UPS",  # UPS Airlines
        "FDX",  # FedEx Express
        "GTI",  # Atlas Air
        "CLX",  # Cargolux
        "ABX",  # ABX Air
        "CPA",  # Cathay Pacific Cargo
        "KZR",  # Kalitta Air
        "RCK",  # Gemini Air Cargo
        "CAL",  # China Airlines Cargo
        "AIC",  # Air India Cargo
        "PAC",  # Polar Air Cargo
        "MPH",  # Martinair Cargo
        "AHY",  # Azerbaijan Airlines (cargo ops)
        "TGX",  # TNT Airways
        "TAY",  # ASL Airlines Belgium (TNT)
    }
)


def _is_cargo(category: int | None, callsign: str | None) -> bool:
    """True if the aircraft is a probable cargo flight."""
    prefix = (callsign or "").strip()[:3].upper()
    return prefix in _CARGO_CALLSIGN_PREFIXES


def _auth() -> tuple[str, str] | None:
    if OPENSKY_USER and OPENSKY_PASS:
        return (OPENSKY_USER, OPENSKY_PASS)
    return None


@retry(max_attempts=3)
def _fetch_states(
    lamin: float, lomin: float, lamax: float, lomax: float
) -> list[list]:
    """
    Call /states/all for the given bounding box.
    Returns list of state vectors (each a list of mixed types).
    """
    resp = requests.get(
        _BASE_URL,
        auth=_auth(),
        params={
            "lamin": lamin,
            "lomin": lomin,
            "lamax": lamax,
            "lomax": lomax,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("states") or []


@retry(max_attempts=2)
def _fetch_route(icao24: str, around: datetime) -> tuple[str | None, str | None]:
    """
    Resolve (origin_icao, dest_icao) for one aircraft via /flights/aircraft.

    Queries a +/-12h window around `around` (the sighting time) and picks
    the flight segment whose [firstSeen, lastSeen] covers that instant. If
    none covers it exactly, falls back to the most recent segment that had
    already started. Returns (None, None) if no candidate flight is found
    or OpenSky didn't estimate one/both airports.
    """
    begin = int((around - _ROUTE_LOOKUP_WINDOW).timestamp())
    end = int((around + _ROUTE_LOOKUP_WINDOW).timestamp())
    resp = requests.get(
        _FLIGHTS_URL,
        auth=_auth(),
        params={"icao24": icao24, "begin": begin, "end": end},
        timeout=30,
    )
    if resp.status_code == 404:
        return None, None
    resp.raise_for_status()
    flights = resp.json() or []

    around_ts = around.timestamp()
    for f in flights:
        first_seen, last_seen = f.get("firstSeen"), f.get("lastSeen")
        if first_seen is not None and last_seen is not None and first_seen <= around_ts <= last_seen:
            return f.get("estDepartureAirport"), f.get("estArrivalAirport")

    started = [f for f in flights if f.get("firstSeen") is not None and f["firstSeen"] <= around_ts]
    if started:
        best = max(started, key=lambda f: f["firstSeen"])
        return best.get("estDepartureAirport"), best.get("estArrivalAirport")

    return None, None


def _already_seen_today(session, icao24s: list[str], today) -> set[str]:
    """icao24s already written to flight_events today — skip route lookups for these."""
    if not icao24s:
        return set()
    rows = session.execute(
        text(
            """
            SELECT DISTINCT icao24
            FROM flight_events
            WHERE icao24 = ANY(:icao24s)
              AND (departed_at AT TIME ZONE 'UTC')::date = :today
            """
        ),
        {"icao24s": icao24s, "today": today},
    ).fetchall()
    return {r.icao24 for r in rows}


def _store_states(states: list[list], region: str) -> int:
    """Filter and write a batch of state vectors to flight_events."""
    candidates = []
    now = datetime.now(timezone.utc)

    for sv in states:
        # Skip if we can't place it geographically or it's on the ground
        if sv[5] is None or sv[6] is None:
            continue
        if sv[8]:   # on_ground = True
            continue

        category = sv[17] if len(sv) > 17 else None
        icao24: str = (sv[0] or "").strip()
        callsign: str = (sv[1] or "").strip()

        # When ADS-B category is populated, keep only large/heavy aircraft.
        # When it's missing (common on /states/all), fall back to matching
        # the callsign against the known cargo-operator list.
        if category is not None:
            if category not in _CARGO_CATEGORIES:
                continue
        else:
            if not _is_cargo(category, callsign):
                continue

        candidates.append({"icao24": icao24, "callsign": callsign, "category": category})

    if not candidates:
        return 0

    with Session() as session:
        seen = _already_seen_today(session, [c["icao24"] for c in candidates], now.date())

    new_candidates = [c for c in candidates if c["icao24"] not in seen]
    if not new_candidates:
        return 0

    rows = []
    for c in new_candidates:
        try:
            origin, dest = _fetch_route(c["icao24"], now)
        except Exception as exc:
            logger.warning("OpenSky route lookup failed for icao24=%s: %s", c["icao24"], exc)
            origin, dest = None, None

        rows.append(
            {
                "icao24": c["icao24"],
                "callsign": c["callsign"] or None,
                "aircraft_type": None,   # not available from /states/all
                "origin_icao": origin,
                "dest_icao": dest,
                "departed_at": now,
                "arrived_at": None,
                "cargo_flag": _is_cargo(c["category"], c["callsign"]),
                "source": "opensky",
            }
        )

    with Session() as session:
        # Belt-and-suspenders: re-check dedup at insert time in case another
        # poll wrote the same (icao24, UTC date) since the check above.
        session.execute(
            text(
                """
                INSERT INTO flight_events
                    (icao24, callsign, aircraft_type, origin_icao, dest_icao,
                     departed_at, arrived_at, cargo_flag, source)
                SELECT :icao24, :callsign, :aircraft_type, :origin_icao,
                       :dest_icao, :departed_at, :arrived_at, :cargo_flag, :source
                WHERE NOT EXISTS (
                    SELECT 1 FROM flight_events fe
                    WHERE fe.icao24 = :icao24
                      AND (fe.departed_at AT TIME ZONE 'UTC')::date
                        = (CAST(:departed_at AS timestamptz) AT TIME ZONE 'UTC')::date
                )
                """
            ),
            rows,
        )
        session.commit()

    return len(rows)


def run(regions: list[str] | None = None) -> None:
    """Poll the given regions (or all) once and store large aircraft sightings."""
    targets = regions if regions is not None else list(REGIONS)
    total = 0
    for name in targets:
        bbox = REGIONS[name]
        states = _fetch_states(*bbox)
        inserted = _store_states(states, name)
        logger.info(
            "OpenSky %-16s  %d states fetched, %d rows inserted",
            name, len(states), inserted,
        )
        total += inserted
    logger.info("OpenSky poll complete — %d total rows inserted", total)


if __name__ == "__main__":
    run()
