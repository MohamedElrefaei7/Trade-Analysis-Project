"""
feature_builder.py — final step of the nightly normalizer.

Reads from the raw ingest tables, applies every transformation the other
normalizer modules expose, computes rolling z-scores, and writes one row
per (date, feature_name) to the `features` table. This is the only table
the backtest layer should query.

Execution contract (see also `normalizer/__init__.py`):

    run_all()  → port_resolver → vessel_normalizer → build()

The scheduler's nightly deployment calls run_all(). Running build()
directly (e.g. from the REPL) skips the upstream cleaning steps — useful
while iterating on feature definitions.

Feature names follow a namespaced dotted form so the backtest layer can
filter with simple LIKE patterns:

    port.<UNLOCODE>.<metric>
    BDI.<metric>
    WCI.<metric>
    FRED.<metric>[.lag_adjusted]
    COMTRADE.<flow>
    air.cargo_flights.<origin>_<dest>

z_score is computed over a rolling 90-day window so all features land on
a comparable scale regardless of their native units.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from clients.base import Session, logger

from . import port_resolver, vessel_normalizer
from .lag_adjuster import apply_lag
from .seasonal_adjuster import deseasonalize
from .time_aligner import to_daily

# ── Configuration ────────────────────────────────────────────────────────────

ZSCORE_WINDOW = 90     # days — rolling window for standardisation

# Map FRED series_id → (feature-name stem, deseasonalize?)
FRED_FEATURES: dict[str, tuple[str, bool]] = {
    "FRED:GDP":       ("FRED.GDP",            False),
    "FRED:BOPGSTB":   ("FRED.trade_balance",  False),
    "FRED:DEXUSAL":   ("FRED.AUDUSD",         False),
    "FRED:DEXCHUS":   ("FRED.CNYUSD",         False),
    "FRED:WPUSI0200": ("FRED.import_price_index", False),
}

# Map Comtrade series_id → friendly feature name.
COMTRADE_FEATURES: dict[str, str] = {
    "COMTRADE:CN-US-85": "COMTRADE.CN_US_electronics",
    "COMTRADE:CN-US-27": "COMTRADE.CN_US_fuels",
    "COMTRADE:AU-CN-26": "COMTRADE.AU_CN_iron_ore",
    "COMTRADE:BR-CN-12": "COMTRADE.BR_CN_soybeans",
}


# ── Fetch helpers ────────────────────────────────────────────────────────────

def _fetch_bench(session, series_id: str) -> pd.DataFrame:
    rows = session.execute(
        text(
            """
            SELECT ts, value, lag_days
            FROM economic_benchmarks
            WHERE series_id = :sid
            ORDER BY ts
            """
        ),
        {"sid": series_id},
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["ts", "value", "lag_days"])
    return pd.DataFrame(rows, columns=["ts", "value", "lag_days"])


def _fetch_port_daily(session) -> pd.DataFrame:
    rows = session.execute(
        text(
            """
            SELECT port_unlocode, date,
                   vessels_in_port, avg_wait_hours,
                   container_count, bulk_count, tanker_count,
                   arrivals, departures
            FROM port_daily_summary
            ORDER BY date
            """
        )
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=[
        "port_unlocode", "date",
        "vessels_in_port", "avg_wait_hours",
        "container_count", "bulk_count", "tanker_count",
        "arrivals", "departures",
    ])


def _fetch_cargo_routes(session) -> pd.DataFrame:
    """Daily cargo-flight count per (origin_iata, dest_iata) pair."""
    rows = session.execute(
        text(
            """
            SELECT DATE(departed_at) AS date,
                   origin_iata, dest_iata,
                   COUNT(*)          AS flights
            FROM flight_events
            WHERE cargo_flag IS TRUE
              AND origin_iata IS NOT NULL
              AND dest_iata   IS NOT NULL
            GROUP BY 1, 2, 3
            ORDER BY 1
            """
        )
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["date", "origin_iata", "dest_iata", "flights"])
    return pd.DataFrame(rows, columns=["date", "origin_iata", "dest_iata", "flights"])


# ── Feature assembly ─────────────────────────────────────────────────────────

def _add_feature(
    bag: list[dict],
    name: str,
    daily: pd.DataFrame,
    *,
    lag_adjusted: bool = False,
    deseasonalized: bool = False,
) -> None:
    """Append every (date, value) row to the insert bag, with a z-score."""
    if daily.empty:
        return
    values = daily["value"].astype(float)
    mean = values.rolling(ZSCORE_WINDOW, min_periods=10).mean()
    std  = values.rolling(ZSCORE_WINDOW, min_periods=10).std(ddof=0)
    z = (values - mean) / std.replace(0, pd.NA)

    for date, v, zv in zip(daily.index, values, z):
        if pd.isna(v):
            continue
        bag.append({
            "date": date.date(),
            "feature_name": name,
            "value": float(v),
            "z_score": None if pd.isna(zv) else float(zv),
            "lag_adjusted": lag_adjusted,
            "deseasonalized": deseasonalized,
        })


def _build_port_features(session, bag: list[dict]) -> None:
    df = _fetch_port_daily(session)
    if df.empty:
        return

    metric_cols = {
        "vessels_in_port": "mean",
        "avg_wait_hours":  "mean",
        "container_count": "sum",
        "arrivals":        "sum",
        "departures":      "sum",
    }

    for unlocode, sub in df.groupby("port_unlocode"):
        sub = sub.rename(columns={"date": "ts"})
        for metric, agg in metric_cols.items():
            if metric not in sub.columns:
                continue
            single = sub[["ts", metric]].rename(columns={metric: "value"})
            daily = to_daily(single, agg=agg)
            name = f"port.{unlocode}.{metric}"

            # Count-type series get STL deseasonalization; wait hours doesn't.
            do_deseason = metric in {"vessels_in_port", "container_count",
                                     "arrivals", "departures"}
            if do_deseason:
                adjusted, was = deseasonalize(daily)
                _add_feature(bag, name, adjusted, deseasonalized=was)
            else:
                _add_feature(bag, name, daily)


def _build_bench_features(session, bag: list[dict]) -> None:
    # Baltic Dry Index — level + 5-day momentum.
    bdi = _fetch_bench(session, "BDI:INDEX")
    if not bdi.empty:
        daily = to_daily(bdi, agg="last", is_market=True)
        _add_feature(bag, "BDI.daily_close", daily)

        momentum = (daily["value"] / daily["value"].shift(5) - 1).to_frame("value")
        _add_feature(bag, "BDI.5d_momentum", momentum.dropna())

    # Drewry WCI — composite (Drewry-published 8-lane average) + per-lane spot
    # rates. Composite comes straight from Drewry's commentary, so we trust it
    # over a re-derived mean. Lane features are stored independently for
    # downstream feature engineering.
    wci_lanes = {
        "WCI:COMPOSITE": "WCI.composite",
        "WCI:SH-GEN":    "WCI.sh_genoa",
        "WCI:SH-RTM":    "WCI.sh_rotterdam",
        "WCI:SH-LA":     "WCI.sh_la",
        "WCI:SH-NY":     "WCI.sh_ny",
        "WCI:TRANS":     "WCI.transatlantic",
    }
    for sid, name in wci_lanes.items():
        raw = _fetch_bench(session, sid)
        if raw.empty:
            continue
        daily = to_daily(raw, agg="last", is_market=True)
        _add_feature(bag, name, daily)

    # FRED macro — lag-adjusted where lag_days > 0.
    for series_id, (stem, _) in FRED_FEATURES.items():
        raw = _fetch_bench(session, series_id)
        if raw.empty:
            continue
        lag = int(raw["lag_days"].iloc[0] or 0)
        daily = to_daily(raw, agg="last", is_market=True)
        adjusted, was_shifted = apply_lag(daily, lag)
        name = f"{stem}.lag_adjusted" if was_shifted else stem
        _add_feature(bag, name, adjusted, lag_adjusted=was_shifted)

    # UN Comtrade — always lag-adjusted (4–6 week release lag baked in).
    for series_id, name in COMTRADE_FEATURES.items():
        raw = _fetch_bench(session, series_id)
        if raw.empty:
            continue
        lag = int(raw["lag_days"].iloc[0] or 0)
        daily = to_daily(raw, agg="last")
        adjusted, was_shifted = apply_lag(daily, lag)
        _add_feature(bag, name, adjusted, lag_adjusted=was_shifted)


def _build_air_features(session, bag: list[dict]) -> None:
    df = _fetch_cargo_routes(session)
    if df.empty:
        return
    for (origin, dest), sub in df.groupby(["origin_iata", "dest_iata"]):
        single = sub[["date", "flights"]].rename(
            columns={"date": "ts", "flights": "value"}
        )
        daily = to_daily(single, agg="sum")
        name = f"air.cargo_flights.{origin}_{dest}"
        adjusted, was = deseasonalize(daily)
        _add_feature(bag, name, adjusted, deseasonalized=was)


# ── Persistence ──────────────────────────────────────────────────────────────

_UPSERT_SQL = text(
    """
    INSERT INTO features
        (date, feature_name, value, z_score, lag_adjusted, deseasonalized)
    VALUES
        (:date, :feature_name, :value, :z_score, :lag_adjusted, :deseasonalized)
    ON CONFLICT (date, feature_name) DO UPDATE SET
        value          = EXCLUDED.value,
        z_score        = EXCLUDED.z_score,
        lag_adjusted   = EXCLUDED.lag_adjusted,
        deseasonalized = EXCLUDED.deseasonalized
    """
)


def _write(bag: list[dict]) -> int:
    if not bag:
        return 0
    with Session() as s:
        s.execute(_UPSERT_SQL, bag)
        s.commit()
    return len(bag)


# ── Public API ───────────────────────────────────────────────────────────────

def build() -> int:
    """
    Build the features table from current raw data. Returns the number
    of rows upserted. Safe to call repeatedly — the (date, feature_name)
    unique constraint drives the upsert.
    """
    bag: list[dict] = []
    with Session() as s:
        _build_port_features(s, bag)
        _build_bench_features(s, bag)
        _build_air_features(s, bag)
    n = _write(bag)
    logger.info("feature_builder: %d feature rows upserted", n)
    return n


def run_all() -> dict:
    """
    Full Step 7 pipeline: resolve port names, smooth AIS into port calls,
    then assemble the features table. This is what the nightly scheduler
    calls.
    """
    logger.info("normalizer: starting nightly run")
    resolver_stats = port_resolver.run()
    vessel_stats = vessel_normalizer.run()
    n = build()
    summary = {
        "port_resolver": resolver_stats,
        "vessel_normalizer": vessel_stats,
        "features_upserted": n,
    }
    logger.info("normalizer: done — %s", summary)
    return summary
