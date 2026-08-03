# Trade Analysis Project — Contract

This file describes the pipeline **as it exists in the code today**. It does
not describe planned work — forward-looking plans belong in `CONTEXT.md` §
Up Next. If something here stops being true, fix this file in the same
commit that changes the code; a contract nobody updates is worse than no
contract.

For current row counts, freshness, schedule status, and known-broken things,
see `CONTEXT.md` — that's the running log. This file is the stable part.

---

## 1. What this project is

Trade_Analysis_Project ingests live vessel positions, port calls, air-cargo
sightings, shipping indices, and macro benchmarks, then normalizes them into
a single daily `features` table used to discover lead-lag relationships and
forecast shipping-index returns. A nightly ElasticNet model per
(target, horizon) turns those features into predictions, and an
edge-triggered alerter plus a Streamlit dashboard surface what changed today.
The system exists to answer one question — "what should I pay attention to
right now?" — not to be a general-purpose market-data warehouse.

---

## 2. Layer contract

Pipeline order: **ingest → normalize → targets → signals → models → alerts →
dashboard**. Each stage reads only what's listed below — wiring a shortcut
between two layers that aren't supposed to touch is the most common way this
pipeline breaks silently.

| Stage | Module | Reads | Writes |
|---|---|---|---|
| Ingest | `clients/aisstream.py` | AISStream WebSocket (external) | `vessels`, `positions`, `port_calls` (rough, real-time arrivals/departures) |
| Ingest | `clients/fred.py`, `clients/comtrade.py`, `clients/scraper.py` (`bdi_scraper`, `wci_scraper`) | external HTTP / scraped pages | `economic_benchmarks` |
| Ingest | `clients/scraper.py` (`port_la_scraper`) | Port of LA website | `port_daily_summary` (`USLAX` rows only) |
| Normalize | `normalizer/port_resolver.py` | `port_calls`, `ports`, `positions` | `port_calls` (backfills `port_unlocode`) |
| Normalize | `normalizer/vessel_normalizer.py` | `positions`, `ports` | `port_calls` (re-smoothed arrivals, closes departures) |
| Normalize | `normalizer/port_summary_builder.py` | `port_calls`, `vessels` | `port_daily_summary` (AIS-tracked ports only — never `USLAX`) |
| Normalize | `normalizer/feature_builder.py` | `port_daily_summary`, `economic_benchmarks`, `flight_events` | `features` |
| Targets | `targets/builder.py` | `features` | `targets` |
| Signals | `signals/builder.py` | `features`, `targets` | `signals` |
| Models | `models/trainer.py` | `targets`, `features` | `predictions` |
| Alerts | `alerts/builder.py` | `features`, `signals`, `predictions` | `alerts` (+ optional Slack post) |
| Dashboard | `dashboard/streamlit_app.py` | `features`, `targets`, `signals`, `predictions`, `alerts` | nothing — read-only |
| Dashboard | `dashboard/conclusions.py` | in-memory DataFrames passed in by the caller | `Conclusion` objects — no DB access at all |

---

## 3. Hard invariants

### The modeling and alerting layers only ever read `features` (+ its derived tables)

`models/trainer.py` and `alerts/builder.py` never query `positions`,
`port_calls`, `flight_events`, or `economic_benchmarks` directly — only
`features`, `targets`, `signals`, and `predictions`. This is what makes
re-running the normalizer safe: everything downstream is re-derivable from
`features`.

### `port_calls` has two writers — this is deliberate, not a violation

`positions`, `flight_events`, and `economic_benchmarks` are written only by
their owning `clients/` ingest module and never touched again. `port_calls`
is the one exception: `clients/aisstream.py` inserts rough arrival/departure
rows in real time, and the normalizer layer — `normalizer/vessel_normalizer.py`
(6-ping majority-vote re-smoothing, arrivals/departures) and
`normalizer/port_resolver.py` (backfills bad/missing `port_unlocode`) — both
mutate those same rows overnight. So "the normalizer never mutates raw
tables" is **not** a true blanket statement; `port_calls` is a shared table
by design. What *is* true without exception: nothing outside `normalizer/`
writes to `features`, and `features` is what everything downstream reads.

### Every derived table is idempotent via upsert — unique keys (from `schema.sql`)

| Table | Unique key |
|---|---|
| `features` | `(date, feature_name)` |
| `targets` | `(date, target_name, horizon_days)` |
| `signals` | `(as_of_date, feature_name, target_name, window_days)` |
| `predictions` | `(date, target_name, horizon_days, model_version)` |
| `alerts` | `(as_of_date, alert_type, feature_name, target_name, horizon_days, window_days)` |
| `port_daily_summary` | `(port_unlocode, date)` |

`alerts` additionally uses `NULLS NOT DISTINCT` (Postgres 15+) so a
`feature_extreme` row (`target_name=NULL`) and a `prediction_extreme` row
(`feature_name=NULL`) both dedupe correctly against re-runs on the same day.

### `models/trainer.py` uses `TimeSeriesSplit(gap=horizon_days)`

Both the outer walk-forward CV (`_walk_forward_oos`) and the inner
alpha/l1_ratio search (`_new_estimator`) pass a non-zero `gap` equal to the
target's horizon. Targets are forward log-returns spanning `t → t+horizon`,
so a training row ending at `t` and a test row starting at `t+1` would still
overlap in what they "see" for up to `horizon` days without the gap — the
model would be scored on data that leaked into its own training window. This
is the single most damaging revert available in this codebase: removing the
gap doesn't crash anything, it just makes out-of-sample R² look better than
it is.

### Alerts are edge-triggered, not level-triggered

All three detectors in `alerts/builder.py` (`feature_extreme`,
`prediction_extreme`, `regime_change`) require the "today crosses, yesterday
didn't" transition — a feature parked at z=2.5 for a week fires once, on the
day it crossed 2.0, not every day it stays elevated.

### `dashboard/conclusions.py` is a pure module

No SQL, no Streamlit imports — the whole module is dataclasses + `numpy` +
`pandas` over DataFrames the caller passes in. That's what lets its four
detectors be unit-tested with synthetic frames instead of a live database.

---

## 4. Conventions

### SQL parameter casts: `CAST(:p AS type)`, never `:p::type`

SQLAlchemy 2.0's bind-parameter parser breaks on `:param::type` — the `::`
gets parsed as part of the token. Every cast on a *bound parameter* in this
codebase goes through `CAST(:x AS sometype)` (see `clients/aisstream.py`,
`clients/fred.py`, `clients/comtrade.py`). Plain `expression::type` on a
column or function result (not a bind param) is fine and already used in a
few places (`normalizer/port_summary_builder.py`, `clients/opensky.py`,
`scheduler.py`) — the rule is specifically about parameters, not all casts.
Someone will "tidy" a `CAST(:p AS type)` back into `:p::type`; it will parse
fine in the editor and break at runtime, not at import time.

### Lag convention: positive `lag_days` means the feature LEADS the target

Verified in `signals/builder.py`'s `_scan_best_lag` (`f.shift(lag)` puts
`feature[t-lag]` at index `t`, so a positive lag shifts the feature backward
in time relative to the target — i.e. the feature leads) and in the matching
comment on the `signals` table in `schema.sql`. Negative `lag_days` means the
feature lags the target by `|lag_days|` days.

### Feature namespacing

| Prefix | Example | Source |
|---|---|---|
| `port.<UNLOCODE>.<metric>` | `port.NLRTM.vessels_in_port` | `port_daily_summary` |
| `BDI.<metric>` | `BDI.daily_close` | `economic_benchmarks` |
| `WCI.<metric>` | `WCI.composite` | `economic_benchmarks` |
| `FRED.<metric>[.lag_adjusted]` | `FRED.trade_balance.lag_adjusted` | `economic_benchmarks` |
| `COMTRADE.<flow>` | `COMTRADE.AU_CN_iron_ore` | `economic_benchmarks` |
| `air.cargo_flights.<origin>_<dest>` | `air.cargo_flights.RJTT_KLAX` | `flight_events` |

`.lag_adjusted` is only appended to the name when `apply_lag()` actually
shifted the series (`lag_days > 0`); a zero-lag series keeps the bare stem.

### AIS sentinel clamping (`clients/aisstream.py::_write_position`)

- `speed_knots` (SOG): stored only if `< 102.2` knots; otherwise `NULL`. AIS
  encodes "not available" as raw value 1023 (→ 102.3 kn); the code's cutoff
  is 102.2, one tenth of a knot below the sentinel.
- `heading`: stored only if `< 360` degrees; otherwise `NULL`. AIS encodes
  "not available" as 360.

### Severity scores share one 0–100 scale

Every conclusion type in `dashboard/conclusions.py`
(`_score_threshold_breach`, `_score_regime_change`, `_score_model_extreme`,
and the fixed `60.0` for stale data) produces a score meant to be compared
directly against every other type, so `generate_conclusions()` can rank and
cap across detectors without per-type priority rules.

---

## 5. Enums (`schema.sql`)

- `vessel_type`: `container`, `bulk_carrier`, `tanker`, `general_cargo`, `roro`, `other`
- `nav_status`: `underway_engine`, `anchored`, `moored`, `restricted_maneuverability`, `not_under_command`, `unknown`
- `data_frequency`: `tick`, `daily`, `weekly`, `monthly`, `quarterly`

---

## 6. Environment variables

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | SQLAlchemy connection string for the TimescaleDB instance |
| `FRED_API_KEY` | FRED (Federal Reserve Economic Data) API key |
| `AISSTREAM_API_KEY` | AISStream WebSocket API key |
| `OPENSKY_USER` / `OPENSKY_PASS` | OpenSky Network credentials (optional — anonymous access has lower rate limits) |
| `COMTRADE_SUBSCRIPTION_KEY` | UN Comtrade API subscription key |
| `PREFECT_API_URL` | URL of a running `prefect server start` instance — `scheduler.py` fails fast if this is unset or unreachable |
| `GRAFANA_PASSWORD` | Grafana admin password (Docker Compose) |
| `SLACK_WEBHOOK_URL` | Optional — enables the alerter's Slack digest post |
| `LOG_LEVEL` | Python `logging` level for the shared `"maritime"` logger (default `INFO`) |

---

## 7. Operational rules

### Never run `schema.sql` against a populated database

`schema.sql` contains **no `DROP` statements** — it's a pure `CREATE`
script, so it won't silently wipe existing tables the way a drop-and-recreate
migration would. But every `CREATE TABLE` and `CREATE TYPE` lacks
`IF NOT EXISTS`, so re-running it against a database that already has these
objects fails loudly with "already exists" errors partway through the script
— it does not run cleanly to completion, and can leave the schema
half-applied. More importantly: this database holds live-streamed AIS
history that cannot be re-collected, so there is no acceptable reason to
re-apply this file against a database that already has data in it.

### Exactly one `scheduler.py` process at a time

Running two instances causes both to re-register Prefect deployments on
startup, which can reset scheduled trigger times. Daily flows self-heal
within 24h (they just re-trigger on their next scheduled time); weekly and
monthly flows do not — a reset trigger time can mean `wci-weekly` or
`comtrade-monthly` silently never fires again until someone notices.

### Prefect server must be running before `scheduler.py` starts

`scheduler.py`'s `_require_prefect_api()` checks that `PREFECT_API_URL` is
set and reachable (`GET {url}/health`) before doing anything else, and
raises `SystemExit` if not. Without this check, Prefect's `serve()` would
silently spawn an ephemeral in-process server on a random port — the
scheduler would appear to start fine, but every flow run would be detached
from persistent history and vanish on restart.
