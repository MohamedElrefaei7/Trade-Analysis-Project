# Trade Analysis Project

A maritime data pipeline that ingests real-time vessel positions and shipping indices — then normalizes, models, and surfaces actionable signals in a daily monitoring dashboard.

---

## What It Does

1. **Ingests** data from multiple sources continuously and on schedule
2. **Normalizes** everything into a daily feature table (deseasonalized, z-scored)
3. **Discovers** lead-lag relationships between features and shipping index returns
4. **Forecasts** BDI and WCI forward returns using walk-forward ElasticNet models
5. **Alerts** on extreme feature moves, extreme model predictions, and regime changes
6. **Visualizes** conclusions, signals, and predictions in a Streamlit dashboard

---

## Data Sources

| Source | Data | Cadence |
|---|---|---|
| [AISStream](https://aisstream.io) | Real-time vessel positions, port arrivals/departures | Continuous |
| Hellenic Shipping News | Baltic Dry Index closing values | Daily (scraped) |
| Hellenic Shipping News | Drewry World Container Index (composite + 5 lanes) | Weekly (scraped) |

---

## Stack

- **Database** — PostgreSQL 16 + TimescaleDB (hypertables for positions, port calls, benchmarks)
- **Orchestration** — APScheduler (`worker/main.py`, 8 scheduled jobs) + a standalone AIS daemon (`ais/main.py`)
- **ML** — scikit-learn `ElasticNetCV` with walk-forward time-series cross-validation
- **Statistics** — Pearson/Spearman correlations, Granger causality (statsmodels)
- **Dashboard** — Streamlit + Plotly
- **Scraping** — BeautifulSoup (Hellenic Shipping News WordPress REST API)
- **Infrastructure** — Docker Compose (TimescaleDB + Grafana)

---

## Requirements

- Python 3.12+
- Docker + Docker Compose
- API key for AISStream

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd Trade_Analysis_Project

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required variables:

```env
DATABASE_URL=postgresql://admin:password@localhost:5432/mydb
AISSTREAM_API_KEY=...

# Optional
SLACK_WEBHOOK_URL=...
GRAFANA_PASSWORD=changeme_grafana
```

### 3. Start the database

```bash
docker-compose up -d
```

### 4. Restore from a dump

`schema.sql` is a historical artifact and is never re-applied against a
populated database — see `CLAUDE.md` § Hard invariants. A fresh database
is provisioned by restoring the latest verified dump:

```bash
pg_restore -d $DATABASE_URL /path/to/latest.dump
```

### 5. Start the AIS daemon (terminal 1)

```bash
source venv/bin/activate
python -m ais.main
```

### 6. Start the scheduled-job worker (terminal 2)

```bash
source venv/bin/activate
python -m worker.main
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
| AIS daemon (`ais/main.py`) | Continuous, standalone process | Real-time vessel position + port call detection |
| `port-call-refresh` | Every 2 hours | Stale open-call audit |
| `bdi-daily` | 18:30 UTC | Baltic Dry Index close |
| `wci-weekly` | Fridays 09:00 UTC | Drewry WCI spot rates |
| `normalizer-nightly` | 23:30 UTC | Feature table build (deseasonalize, z-score) |
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

## Project Architecture

```
├── schema.sql            # Full database schema (historical artifact, never re-applied)
├── requirements.txt
│
├── ais/main.py           # Standalone AIS daemon entrypoint — connection lifecycle only
│
├── worker/               # APScheduler process running the eight scheduled jobs
│   ├── main.py
│   └── cadences.py
│
├── orchestration/        # migrate.py (numbered-SQL runner), jobs.py (@job decorator), tasks.py (JOBS registry)
│
├── clients/              # Data ingest
│   ├── base.py           # Shared DB session, retry decorator
│   ├── aisstream.py      # AIS WebSocket message handling → positions + port_calls
│   ├── scraper.py        # BDI, WCI scrapers
│   └── geo.py            # Haversine distance + port lookup
│
├── normalizer/           # Nightly transformation pipeline
│   ├── feature_builder.py
│   ├── port_resolver.py
│   ├── vessel_normalizer.py
│   ├── time_aligner.py
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
