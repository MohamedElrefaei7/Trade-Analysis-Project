# Trade Analysis Project

A maritime and trade data pipeline that ingests real-time vessel positions, air freight events, shipping indices, and macroeconomic data — then normalizes, models, and surfaces actionable signals in a daily monitoring dashboard.

---

## What It Does

1. **Ingests** data from multiple sources continuously and on schedule
2. **Normalizes** everything into a daily feature table (lag-adjusted, deseasonalized, z-scored)
3. **Discovers** lead-lag relationships between features and shipping index returns
4. **Forecasts** BDI and WCI forward returns using walk-forward ElasticNet models
5. **Alerts** on extreme feature moves, extreme model predictions, and regime changes
6. **Visualizes** conclusions, signals, and predictions in a Streamlit dashboard

---

## Data Sources

| Source | Data | Cadence |
|---|---|---|
| [AISStream](https://aisstream.io) | Real-time vessel positions, port arrivals/departures | Continuous |
| [OpenSky Network](https://opensky-network.org) | Air cargo flight events | Daily |
| [FRED](https://fred.stlouisfed.org) | GDP, trade balance, AUD/USD, CNY/USD, import price index | Daily |
| [UN Comtrade](https://comtradeapi.un.org) | Bilateral trade flows (CN→US electronics/fuels, AU→CN iron ore, BR→CN soybeans) | Monthly |
| Hellenic Shipping News | Baltic Dry Index closing values | Daily (scraped) |
| Hellenic Shipping News | Drewry World Container Index (composite + 5 lanes) | Weekly (scraped) |
| Port of Los Angeles | Monthly TEU throughput | Monthly (scraped) |

---

## Stack

- **Database** — PostgreSQL 16 + TimescaleDB (hypertables for positions, port calls, flight events, benchmarks)
- **Orchestration** — Prefect 3.x (12 scheduled flows + AIS daemon thread)
- **ML** — scikit-learn `ElasticNetCV` with walk-forward time-series cross-validation
- **Statistics** — Pearson/Spearman correlations, Granger causality (statsmodels)
- **Dashboard** — Streamlit + Plotly
- **Scraping** — Playwright (headless Chromium) + BeautifulSoup
- **Infrastructure** — Docker Compose (TimescaleDB + Grafana)

---

## Requirements

- Python 3.12+
- Docker + Docker Compose
- Playwright Chromium (`playwright install chromium`)
- API keys for AISStream, FRED, OpenSky, and UN Comtrade

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd Trade_Analysis_Project

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required variables:

```env
DATABASE_URL=postgresql://admin:password@localhost:5432/mydb
FRED_API_KEY=...
AISSTREAM_API_KEY=...
OPENSKY_USER=...
OPENSKY_PASS=...
COMTRADE_SUBSCRIPTION_KEY=...
PREFECT_API_URL=http://127.0.0.1:4200/api

# Optional
SLACK_WEBHOOK_URL=...
GRAFANA_PASSWORD=changeme_grafana
```

### 3. Start the database

```bash
docker-compose up -d
```

### 4. Apply the schema

```bash
psql $DATABASE_URL -f schema.sql
```

### 5. Start Prefect server (terminal 1)

```bash
prefect server start
# UI available at http://127.0.0.1:4200
```

### 6. Start the pipeline (terminal 2)

```bash
source venv/bin/activate
python scheduler.py
```

### 7. Start the dashboard (terminal 3)

```bash
source venv/bin/activate
streamlit run dashboard/streamlit_app.py
# Available at http://localhost:8501
```

---

## Schedule

| Flow | When | Description |
|---|---|---|
| AIS stream | Continuous | Real-time vessel position + port call detection |
| `port-call-refresh` | Every 2 hours | AIS thread health check; stale call audit |
| `opensky-daily` | 06:00 UTC | Air cargo flight snapshot |
| `fred-daily` | 07:00 UTC | FRED macro series |
| `bdi-daily` | 18:30 UTC | Baltic Dry Index close |
| `wci-weekly` | Fridays 09:00 UTC | Drewry WCI spot rates |
| `comtrade-monthly` | 15th 08:00 UTC | UN Comtrade bilateral trade flows |
| `port-la-monthly` | 16th 08:00 UTC | Port of LA TEU throughput |
| `normalizer-nightly` | 23:30 UTC | Feature table build (lag-adjust, deseasonalize, z-score) |
| `targets-nightly` | 23:45 UTC | Forward log-return targets (BDI 5d/20d, WCI 20d) |
| `signals-nightly` | 23:55 UTC | Lead-lag sweep across all (feature × target × window) pairs |
| `models-nightly` | 00:05 UTC | Walk-forward ElasticNet training + live forecasts |
| `alerts-nightly` | 00:15 UTC | Edge-triggered alerter + optional Slack digest |

---

## Dashboard

Five tabs, in priority order:

- **Today** — Up to 5 ranked conclusions in plain English, target outlook cards, and a "what changed since yesterday" delta table
- **Signals** — Filterable lead-lag grid with evidence plots and sub-window stability indicator
- **Predictions** — Actual vs. walk-forward prediction overlay, residual MAE health badge, feature contribution proxy
- **Explore** — Ad-hoc correlation heatmap, lag sweep, and interactive Granger causality test
- **Health** — Feature freshness, recent alerts, model retraining audit

---

## Project Structure

```
├── scheduler.py          # Prefect orchestrator + AIS daemon
├── schema.sql            # Full database schema
├── requirements.txt
│
├── clients/              # Data ingest
│   ├── base.py           # Shared DB session, retry decorator
│   ├── aisstream.py      # AIS WebSocket → positions + port_calls
│   ├── fred.py           # FRED REST API
│   ├── comtrade.py       # UN Comtrade API
│   ├── opensky.py        # OpenSky Network API
│   ├── scraper.py        # BDI, WCI, Port of LA scrapers
│   └── geo.py            # Haversine distance + port lookup
│
├── normalizer/           # Nightly transformation pipeline
│   ├── feature_builder.py
│   ├── port_resolver.py
│   ├── vessel_normalizer.py
│   ├── time_aligner.py
│   ├── lag_adjuster.py
│   └── seasonal_adjuster.py
│
├── targets/builder.py    # Forward log-return target construction
├── signals/builder.py    # Lead-lag signal discovery
├── models/trainer.py     # ElasticNet walk-forward training
├── alerts/builder.py     # Edge-triggered alerter + Slack digest
│
├── dashboard/
│   ├── streamlit_app.py  # Main 5-tab dashboard
│   ├── conclusions.py    # Detector logic (testable, no Streamlit imports)
│   └── correlation.py    # Standalone correlation explorer
│
└── grafana/              # Auto-provisioned Grafana dashboards
```

---

## Grafana

Available at `http://localhost:3000` (default login: `admin` / value of `GRAFANA_PASSWORD`).

Two pre-provisioned dashboards:
- **Pipeline Health** — ingest freshness and flow run status
- **Signal Monitor** — lead-lag signal strength over time
