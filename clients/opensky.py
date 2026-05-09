"""
opensky.py — OpenSky Network REST client.

Polls the /states/all endpoint for each bounding box, filters for cargo /
heavy aircraft, and writes rows to flight_events.

Endpoint:
    GET https://opensky-network.org/api/states/all
        ?lamin=&lomin=&lamax=&lomax=

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
from datetime import datetime, timezone

import requests
from sqlalchemy import text

from .base import Session, logger, retry

OPENSKY_USER = os.environ.get("OPENSKY_USER", "")
OPENSKY_PASS = os.environ.get("OPENSKY_PASS", "")
_BASE_URL = "https://opensky-network.org/api/states/all"

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


def _store_states(states: list[list], region: str) -> int:
    """Filter and write a batch of state vectors to flight_events."""
    rows = []
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

        rows.append(
            {
                "icao24": icao24,
                "callsign": callsign or None,
                "aircraft_type": None,   # not available from /states/all
                "origin_iata": None,
                "dest_iata": None,
                "departed_at": now,
                "arrived_at": None,
                "cargo_flag": _is_cargo(category, callsign),
                "source": "opensky",
            }
        )

    if not rows:
        return 0

    with Session() as session:
        # Skip aircraft we already wrote today — dedup on (icao24, UTC date).
        session.execute(
            text(
                """
                INSERT INTO flight_events
                    (icao24, callsign, aircraft_type, origin_iata, dest_iata,
                     departed_at, arrived_at, cargo_flag, source)
                SELECT :icao24, :callsign, :aircraft_type, :origin_iata,
                       :dest_iata, :departed_at, :arrived_at, :cargo_flag, :source
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
