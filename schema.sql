-- ============================================================
-- Shipping & Macro Data Pipeline — Schema
-- Requires: PostgreSQL 15+ with TimescaleDB extension
-- ============================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── ENUMS ────────────────────────────────────────────────────

CREATE TYPE vessel_type AS ENUM (
    'container', 
    'bulk_carrier',
    'tanker',
    'general_cargo',
    'roro',       -- Roll-on/Roll-off
    'other'
);

CREATE TYPE nav_status AS ENUM (
    'underway_engine',
    'anchored',
    'moored',
    'restricted_maneuverability',
    'not_under_command',
    'unknown'
);

CREATE TYPE data_frequency AS ENUM (
    'tick',       -- real-time / sub-minute
    'daily',
    'weekly',
    'monthly',
    'quarterly'
);

-- ── CORE TABLES ──────────────────────────────────────────────

-- Static vessel registry — one row per vessel, updated rarely
CREATE TABLE vessels (
    vessel_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mmsi            VARCHAR(9) UNIQUE NOT NULL,   -- Maritime Mobile Service Identity
    imo             VARCHAR(10),                  -- IMO number (more stable than MMSI)
    name            TEXT,
    vessel_type     vessel_type NOT NULL,
    flag_country    CHAR(2),                      -- ISO 3166-1 alpha-2
    dwt             INTEGER,                      -- Deadweight tonnage (capacity proxy)
    cargo_class     TEXT,                         -- e.g. 'crude_oil', 'iron_ore', 'containers'
    year_built      SMALLINT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_vessels_mmsi ON vessels(mmsi);
CREATE INDEX idx_vessels_type ON vessels(vessel_type);

-- ── TIME-SERIES: AIS position pings ──────────────────────────
-- TimescaleDB hypertable — chunked by week
CREATE TABLE positions (
    position_id     UUID NOT NULL DEFAULT uuid_generate_v4(),
    vessel_id       UUID NOT NULL REFERENCES vessels(vessel_id),
    ts              TIMESTAMPTZ NOT NULL,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    speed_knots     FLOAT,                        -- SOG: speed over ground
    heading         SMALLINT,                     -- 0–359 degrees
    nav_status      nav_status DEFAULT 'unknown',
    source          TEXT NOT NULL,                -- 'marinetraffic', 'aishub', 'self_hosted'
    PRIMARY KEY (position_id, ts)
);

SELECT create_hypertable('positions', 'ts', chunk_time_interval => INTERVAL '1 week');

CREATE INDEX idx_positions_vessel_ts ON positions(vessel_id, ts DESC);
-- Spatial index for bounding box queries (port area lookups)
CREATE INDEX idx_positions_lat_lon ON positions(lat, lon);

-- ── TIME-SERIES: Port arrival / departure events ──────────────
CREATE TABLE port_calls (
    call_id          UUID NOT NULL DEFAULT uuid_generate_v4(),
    vessel_id        UUID NOT NULL REFERENCES vessels(vessel_id),
    port_unlocode    CHAR(5) NOT NULL,            -- UN/LOCODE e.g. 'CNSHA' = Shanghai
    arrived_at       TIMESTAMPTZ NOT NULL,
    departed_at      TIMESTAMPTZ,
    PRIMARY KEY (call_id, arrived_at),
    duration_hours   FLOAT                        -- derived, populated on departure
        GENERATED ALWAYS AS (
            EXTRACT(EPOCH FROM (departed_at - arrived_at)) / 3600.0
        ) STORED,
    origin_unlocode  CHAR(5),
    dest_unlocode    CHAR(5),
    source           TEXT NOT NULL
);

SELECT create_hypertable(
    'port_calls', 'arrived_at',
    chunk_time_interval => INTERVAL '1 month',
    migrate_data => true
);

CREATE INDEX idx_port_calls_port_ts   ON port_calls(port_unlocode, arrived_at DESC);
CREATE INDEX idx_port_calls_vessel_ts ON port_calls(vessel_id, arrived_at DESC);

-- ── AIR FREIGHT EVENTS ───────────────────────────────────────
-- One row per cargo flight departure (from OpenSky / FlightAware)
CREATE TABLE flight_events (
    event_id        UUID NOT NULL DEFAULT uuid_generate_v4(),
    icao24          TEXT NOT NULL,                -- Aircraft transponder hex ID
    callsign        TEXT,
    aircraft_type   TEXT,                         -- e.g. '747F', '777F'
    origin_iata     CHAR(4),
    dest_iata       CHAR(4),
    departed_at     TIMESTAMPTZ NOT NULL,
    arrived_at      TIMESTAMPTZ,
    cargo_flag      BOOLEAN NOT NULL DEFAULT FALSE,
    source          TEXT NOT NULL,                -- 'opensky', 'flightaware'
    PRIMARY KEY (event_id, departed_at)
);

SELECT create_hypertable('flight_events', 'departed_at',
    chunk_time_interval => INTERVAL '1 month');

CREATE INDEX idx_flights_route_ts ON flight_events(origin_iata, dest_iata, departed_at DESC);
CREATE INDEX idx_flights_cargo    ON flight_events(cargo_flag, departed_at DESC);

-- ── DERIVED: Daily port-level aggregates ─────────────────────
-- Populated nightly by an aggregation job (not live)
CREATE TABLE port_daily_summary (
    summary_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    port_unlocode     CHAR(5) NOT NULL,
    date              DATE NOT NULL,
    vessels_in_port   INTEGER,
    avg_wait_hours    FLOAT,
    container_count   INTEGER,
    bulk_count        INTEGER,
    tanker_count      INTEGER,
    arrivals          INTEGER,
    departures        INTEGER,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (port_unlocode, date)
);

CREATE INDEX idx_summary_port_date ON port_daily_summary(port_unlocode, date DESC);

-- ── DERIVED: Economic benchmarks ─────────────────────────────
-- Generic key-value time-series for FRED, BDI, FX, equities
-- series_id examples: 'FRED:GDP', 'BDI:INDEX', 'FX:AUDUSD', 'EQ:ZIM'
CREATE TABLE economic_benchmarks (
    bench_id    UUID NOT NULL DEFAULT uuid_generate_v4(),
    series_id   TEXT NOT NULL,                    -- namespaced identifier
    source      TEXT NOT NULL,                    -- 'fred', 'bdi', 'yahoo', 'ecb'
    ts          TIMESTAMPTZ NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    unit        TEXT,                             -- 'USD', 'index', 'percent', etc.
    frequency   data_frequency NOT NULL,
    lag_days    SMALLINT DEFAULT 0,               -- known publication lag (for ALFRED)
    PRIMARY KEY (bench_id, ts)
);

SELECT create_hypertable('economic_benchmarks', 'ts',
    chunk_time_interval => INTERVAL '1 month');

CREATE INDEX idx_bench_series_ts ON economic_benchmarks(series_id, ts DESC);

-- ── REFERENCE: Port registry ──────────────────────────────────
-- Static lookup: maps UN/LOCODE to human names + coordinates
CREATE TABLE ports (
    unlocode        CHAR(5) PRIMARY KEY,          -- e.g. 'CNSHA'
    country         CHAR(2) NOT NULL,
    name            TEXT NOT NULL,
    lat             DOUBLE PRECISION,
    lon             DOUBLE PRECISION,
    tier            SMALLINT DEFAULT 3            -- 1=megaport, 2=major, 3=regional
);

-- Pre-load the key ports you're tracking
INSERT INTO ports (unlocode, country, name, lat, lon, tier) VALUES
    ('CNSHA', 'CN', 'Shanghai',          31.2304,  121.4737, 1),
    ('NLRTM', 'NL', 'Rotterdam',         51.9225,    4.4792, 1),
    ('SGSIN', 'SG', 'Singapore',          1.3521,  103.8198, 1),
    ('USLAX', 'US', 'Los Angeles',       33.7290, -118.2620, 1),
    ('USLGB', 'US', 'Long Beach',        33.7676, -118.1956, 1),
    ('DEHAM', 'DE', 'Hamburg',           53.5753,    9.9300, 1),
    ('KRPUS', 'KR', 'Busan',             35.1028,  129.0403, 1),
    ('JPYOK', 'JP', 'Yokohama',          35.4437,  139.6380, 2),
    ('GBFXT', 'GB', 'Felixstowe',        51.9642,    1.3419, 2),
    ('AEDXB', 'AE', 'Dubai (Jebel Ali)', 24.9857,   55.0272, 1);

-- ── DERIVED: Normalized feature table (Step 7 output) ────────────────────────
-- Single wide-ish (date, feature_name, value) store consumed by the backtest
-- layer. Produced nightly by the `normalizer` package. Raw tables are never
-- mutated; everything here is re-derivable by re-running the normalizer.
CREATE TABLE features (
    feature_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date            DATE NOT NULL,
    feature_name    TEXT NOT NULL,
    value           DOUBLE PRECISION,
    z_score         DOUBLE PRECISION,    -- rolling 90-day standardised value
    lag_adjusted    BOOLEAN DEFAULT FALSE,
    deseasonalized  BOOLEAN DEFAULT FALSE,
    UNIQUE (date, feature_name)
);

CREATE INDEX idx_features_name_date ON features(feature_name, date DESC);
CREATE INDEX idx_features_date      ON features(date DESC);

-- ── DERIVED: Prediction targets ──────────────────────────────────────────────
-- Same shape as `features`, plus `horizon_days`. The modeling layer pairs each
-- (date, target_name) with feature rows on the same date — features observed at
-- t are joined to target value spanning t → t + horizon_days.
CREATE TABLE targets (
    target_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date          DATE NOT NULL,
    target_name   TEXT NOT NULL,
    value         DOUBLE PRECISION,
    horizon_days  SMALLINT NOT NULL,
    UNIQUE (date, target_name, horizon_days)
);

CREATE INDEX idx_targets_name_date ON targets(target_name, date DESC);
CREATE INDEX idx_targets_date      ON targets(date DESC);

-- ── DERIVED: Lead-lag signals ────────────────────────────────────────────────
-- Nightly snapshot of the strongest lead-lag relationships found between every
-- feature and every target across rolling windows (180d, 720d). One row per
-- (as_of_date, feature_name, target_name, window_days) — `lag_days` stores the
-- shift at which `|pearson_r|` was maximal. Positive `lag_days` means the
-- feature LEADS the target by that many days.
--
-- Granger p-value is computed only for the rows that pass the top-N filter, so
-- a NULL granger_p just means the row didn't survive the strength cutoff.
-- This is the "what's working right now" memory the alerter reads from.
CREATE TABLE signals (
    signal_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    as_of_date    DATE NOT NULL,
    feature_name  TEXT NOT NULL,
    target_name   TEXT NOT NULL,
    window_days   SMALLINT NOT NULL,
    lag_days      SMALLINT NOT NULL,
    pearson_r     DOUBLE PRECISION,
    spearman_r    DOUBLE PRECISION,
    granger_p     DOUBLE PRECISION,
    sample_size   INTEGER NOT NULL,
    UNIQUE (as_of_date, feature_name, target_name, window_days)
);

CREATE INDEX idx_signals_asof   ON signals(as_of_date DESC);
CREATE INDEX idx_signals_target ON signals(target_name, as_of_date DESC);
CREATE INDEX idx_signals_pair   ON signals(feature_name, target_name, as_of_date DESC);

-- ── DERIVED: Model predictions ───────────────────────────────────────────────
-- One row per (date, target_name, horizon_days, model_version). Historical
-- rows are out-of-sample predictions from walk-forward CV; rows whose
-- horizon hasn't elapsed yet are forecasts from the latest fit on all
-- realized data. The alerter thresholds against `predicted_z` to surface
-- "abnormally bullish/bearish" forecasts without re-deriving them.
CREATE TABLE predictions (
    prediction_id    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date             DATE NOT NULL,
    target_name      TEXT NOT NULL,
    horizon_days     SMALLINT NOT NULL,
    predicted_value  DOUBLE PRECISION,
    predicted_z      DOUBLE PRECISION,
    model_version    TEXT NOT NULL,
    UNIQUE (date, target_name, horizon_days, model_version)
);

CREATE INDEX idx_predictions_target ON predictions(target_name, horizon_days, date DESC);
CREATE INDEX idx_predictions_date   ON predictions(date DESC);

-- ── DERIVED: Alerts ──────────────────────────────────────────────────────────
-- Edge-triggered events surfaced by the nightly alerter. Three flavours:
--   feature_extreme    — feature z-score crossed ±2σ AND has a significant
--                        lead-lag relationship with at least one target.
--   prediction_extreme — model predicted_z crossed ±2σ.
--   regime_change      — signals row just became significant (|pearson_r|≥0.20
--                        and granger_p<0.05) after being insignificant within
--                        the last 14 days.
-- The UNIQUE constraint dedupes a re-run of the same data date — so the
-- alerter is idempotent. NULLS NOT DISTINCT (Postgres 15+) lets the same
-- key work for prediction alerts (feature_name=NULL) and feature alerts
-- (horizon_days=NULL) without colliding.
CREATE TABLE alerts (
    alert_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    triggered_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    as_of_date    DATE NOT NULL,
    alert_type    TEXT NOT NULL,
    severity      TEXT NOT NULL,            -- 'info' | 'warning' | 'critical'
    subject       TEXT NOT NULL,
    message       TEXT NOT NULL,
    feature_name  TEXT,
    target_name   TEXT,
    horizon_days  SMALLINT,
    window_days   SMALLINT,
    z_score       DOUBLE PRECISION,
    pearson_r     DOUBLE PRECISION,
    UNIQUE NULLS NOT DISTINCT
        (as_of_date, alert_type, feature_name, target_name, horizon_days, window_days)
);

CREATE INDEX idx_alerts_asof ON alerts(as_of_date DESC);
CREATE INDEX idx_alerts_type ON alerts(alert_type, as_of_date DESC);
