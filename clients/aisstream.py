"""
aisstream.py — AISStream WebSocket client.

Connects to the AISStream real-time AIS feed and writes:
  - positions      : every PositionReport message
  - vessels        : new vessel rows on first sight of an unseen MMSI
  - port_calls     : arrival when a vessel moors/anchors near a known port;
                     departure when it gets underway again

AISStream uses a persistent WebSocket, not REST polling. This module runs as
a long-lived async process. On disconnect it reconnects automatically.

WebSocket URL:  wss://stream.aisstream.io/v0/stream
Subscribe msg format:
    {
        "APIKey": "...",
        "BoundingBoxes": [[[min_lat, min_lon], [max_lat, max_lon]], ...],
        "FilterMessageTypes": ["PositionReport"]
    }

Usage:
    import asyncio
    from clients.aisstream import stream
    asyncio.run(stream())
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import websockets
from sqlalchemy import text

from .base import Session, logger
from .geo import Port, load_ports, nearest_port

AISSTREAM_API_KEY = os.environ.get("AISSTREAM_API_KEY", "")
_WS_URL = "wss://stream.aisstream.io/v0/stream"

# Bounding boxes [[min_lat, min_lon], [max_lat, max_lon]]
BOUNDING_BOXES: list[list[list[float]]] = [
    [[20.0, 100.0], [50.0, 145.0]],    # East Asia
    [[35.0, -70.0], [65.0,  10.0]],    # North Atlantic / Europe
    [[-40.0, 110.0], [-10.0, 160.0]],  # Australia
    [[20.0,  50.0], [30.0,  60.0]],    # Arabian Gulf
    [[-35.0, -55.0], [5.0,  -35.0]],   # South Atlantic (Brazil routes)
]

# AIS numerical nav status → our enum values
_NAV_STATUS_MAP: dict[int, str] = {
    0: "underway_engine",
    1: "anchored",
    2: "not_under_command",
    3: "restricted_maneuverability",
    5: "moored",
}
_STATIONARY: frozenset[str] = frozenset({"moored", "anchored"})

# Vessel arrives at a port if within this radius (km)
_ARRIVAL_RADIUS_KM = 30.0

# AIS ship type codes → vessel_type enum
def _map_vessel_type(ship_type: int) -> str:
    if 70 <= ship_type <= 79:
        return "container"
    if 80 <= ship_type <= 89:
        return "tanker"
    if ship_type in (71, 72, 73, 74, 1022, 1023, 1024):
        return "bulk_carrier"
    if ship_type in (50, 51, 52, 53, 54, 55):
        return "general_cargo"
    if ship_type == 60:
        return "roro"
    return "other"


# ── In-memory caches (per-process) ───────────────────────────────────────────

# Loaded once at startup via _load_port_cache().
_ports: list[Port] = []

# mmsi → {vessel_id, nav_status, call_id?, arrived_at?, port_unlocode?}
_vessel_state: dict[str, dict] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_port_cache(session) -> None:
    """Populate the module-level port cache from the ports table."""
    _ports[:] = load_ports(session)
    logger.info("Port cache loaded — %d ports", len(_ports))


def _ensure_vessel(session, mmsi: str, meta: dict, report: dict) -> str:
    """
    Return vessel_id for this MMSI, inserting a new row if it's the first time
    we've seen it. Uses the in-memory state cache to avoid per-message DB reads.
    """
    state = _vessel_state.get(mmsi, {})
    if "vessel_id" in state:
        return state["vessel_id"]

    # Check DB (handles restarts)
    row = session.execute(
        text("SELECT vessel_id FROM vessels WHERE mmsi = :mmsi"),
        {"mmsi": mmsi},
    ).fetchone()

    if row:
        vessel_id = str(row.vessel_id)
    else:
        vessel_id = str(uuid4())
        ship_type = report.get("ShipType", 0) or 0
        name = (meta.get("ShipName") or "").strip() or None

        session.execute(
            text(
                """
                INSERT INTO vessels
                    (vessel_id, mmsi, name, vessel_type, created_at, updated_at)
                VALUES
                    (CAST(:vid AS uuid), :mmsi, :name,
                     CAST(:vtype AS vessel_type), NOW(), NOW())
                ON CONFLICT (mmsi) DO NOTHING
                """
            ),
            {
                "vid": vessel_id,
                "mmsi": mmsi,
                "name": name,
                "vtype": _map_vessel_type(ship_type),
            },
        )
        session.commit()
        logger.debug("New vessel  MMSI=%s  name=%s", mmsi, name)

    _vessel_state.setdefault(mmsi, {})["vessel_id"] = vessel_id
    return vessel_id


def _write_position(session, vessel_id: str, lat: float, lon: float,
                    speed, heading, nav_status_str: str, ts: datetime) -> None:
    session.execute(
        text(
            """
            INSERT INTO positions
                (vessel_id, ts, lat, lon, speed_knots, heading, nav_status, source)
            VALUES
                (CAST(:vid AS uuid), :ts, :lat, :lon, :spd, :hdg,
                 CAST(:ns AS nav_status), 'aisstream')
            """
        ),
        {
            "vid": vessel_id,
            "ts": ts,
            "lat": lat,
            "lon": lon,
            # AIS SOG sentinel: 1023 (→ 102.3 knots) means "not available".
            # AIS COG sentinel: 360 means "not available".
            "spd": (float(speed) if speed is not None and float(speed) < 102.2
                    else None),
            "hdg": (int(heading) if heading is not None and int(heading) < 360
                    else None),
            "ns": nav_status_str,
        },
    )


def _update_port_call(session, mmsi: str, vessel_id: str,
                      lat: float, lon: float,
                      nav_status_str: str, ts: datetime) -> None:
    """
    Detect arrival / departure transitions and write to port_calls.

    Arrival  : vessel enters STATIONARY state within ARRIVAL_RADIUS_KM of a port.
    Departure: vessel leaves STATIONARY state when a prior call_id is tracked.
    """
    state = _vessel_state.get(mmsi, {})
    prev_status = state.get("nav_status", "unknown")

    # ── Arrival ───────────────────────────────────────────────────────────────
    if nav_status_str in _STATIONARY and prev_status not in _STATIONARY:
        port_code, dist_km = nearest_port(lat, lon, _ports, max_km=_ARRIVAL_RADIUS_KM)
        if port_code:
            call_id = str(uuid4())
            session.execute(
                text(
                    """
                    INSERT INTO port_calls
                        (call_id, vessel_id, port_unlocode, arrived_at, source)
                    VALUES
                        (CAST(:cid AS uuid), CAST(:vid AS uuid),
                         :port, :arrived_at, 'aisstream')
                    """
                ),
                {
                    "cid": call_id,
                    "vid": vessel_id,
                    "port": port_code,
                    "arrived_at": ts,
                },
            )
            session.commit()
            _vessel_state[mmsi].update(
                {
                    "nav_status": nav_status_str,
                    "call_id": call_id,
                    "arrived_at": ts,
                    "port_unlocode": port_code,
                }
            )
            logger.info(
                "ARRIVAL  MMSI=%-12s  port=%-5s  dist=%.1f km",
                mmsi, port_code, dist_km,
            )
            return

    # ── Departure ─────────────────────────────────────────────────────────────
    if prev_status in _STATIONARY and nav_status_str not in _STATIONARY:
        call_id = state.get("call_id")
        arrived_at = state.get("arrived_at")
        if call_id and arrived_at:
            session.execute(
                text(
                    """
                    UPDATE port_calls
                    SET departed_at = :dep
                    WHERE call_id = CAST(:cid AS uuid) AND arrived_at = :arr
                    """
                ),
                {"dep": ts, "cid": call_id, "arr": arrived_at},
            )
            session.commit()
            logger.info(
                "DEPARTURE  MMSI=%-12s  port=%-5s",
                mmsi, state.get("port_unlocode", "?"),
            )

    _vessel_state.setdefault(mmsi, {})["nav_status"] = nav_status_str


def _handle_message(session, msg: dict) -> None:
    """Process one PositionReport message from AISStream."""
    meta = msg.get("MetaData", {})
    report = msg.get("Message", {}).get("PositionReport", {})

    mmsi = str(meta.get("MMSI", "")).strip()
    if not mmsi:
        return

    lat = report.get("Latitude") or meta.get("latitude")
    lon = report.get("Longitude") or meta.get("longitude")
    if lat is None or lon is None:
        return

    ts = datetime.now(timezone.utc)
    speed = report.get("Sog")
    heading = report.get("TrueHeading")
    ais_status_code = report.get("NavigationalStatus", 15)
    nav_status_str = _NAV_STATUS_MAP.get(ais_status_code, "unknown")

    vessel_id = _ensure_vessel(session, mmsi, meta, report)
    _write_position(session, vessel_id, float(lat), float(lon),
                    speed, heading, nav_status_str, ts)
    session.commit()

    _update_port_call(session, mmsi, vessel_id,
                      float(lat), float(lon), nav_status_str, ts)


# ── Main async loop ───────────────────────────────────────────────────────────

async def stream() -> None:
    """
    Connect to AISStream, subscribe to all configured bounding boxes,
    and process messages indefinitely. Reconnects automatically on disconnect.
    """
    if not AISSTREAM_API_KEY:
        logger.error("AISSTREAM_API_KEY not set — cannot start AIS stream")
        return

    # Load port cache once at startup
    with Session() as session:
        _load_port_cache(session)

    subscribe_payload = json.dumps(
        {
            "APIKey": AISSTREAM_API_KEY,
            "BoundingBoxes": BOUNDING_BOXES,
            "FilterMessageTypes": ["PositionReport"],
        }
    )

    while True:
        try:
            async with websockets.connect(_WS_URL, ping_interval=30) as ws:
                await ws.send(subscribe_payload)
                logger.info("AISStream connected — listening for position reports")

                with Session() as session:
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                            if msg.get("MessageType") == "PositionReport":
                                _handle_message(session, msg)
                        except Exception as exc:
                            # Don't crash the loop on a single bad message
                            logger.warning("Message handling error: %s", exc)

        except websockets.ConnectionClosed as exc:
            logger.warning("AISStream disconnected (%s) — reconnecting in 5s", exc)
        except Exception as exc:
            logger.error("AISStream error: %s — reconnecting in 5s", exc)

        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(stream())
