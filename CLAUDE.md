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

Trade_Analysis_Project ingests live vessel positions, port calls, and
shipping indices, then normalizes them into a single daily `features` table
used to discover lead-lag relationships and forecast shipping-index returns.
A nightly ElasticNet model per
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
| Ingest | `clients/scraper.py` (`bdi_scraper`, `wci_scraper`) | Hellenic Shipping News (WP REST API) | `economic_benchmarks` |
| Normalize | `normalizer/port_resolver.py` | `port_calls`, `ports`, `positions` | `port_calls` (backfills `port_unlocode`) |
| Normalize | `normalizer/vessel_normalizer.py` | `positions`, `ports` | `port_calls` (re-smoothed arrivals, closes departures) |
| Normalize | `normalizer/port_summary_builder.py` | `port_calls`, `vessels` | `port_daily_summary` (AIS-tracked ports only — never `USLAX`) |
| Normalize | `normalizer/feature_builder.py` | `port_daily_summary`, `economic_benchmarks` | `features` |
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
`port_calls`, or `economic_benchmarks` directly — only `features`, `targets`,
`signals`, and `predictions`. This is what makes re-running the normalizer
safe: everything downstream is re-derivable from `features`.

### `port_calls` has two writers — this is deliberate, not a violation

`positions` and `economic_benchmarks` are written only by
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

### `schema.sql` is a historical artifact — it is never re-applied

Schema changes ship as additive migration files, not as edits to
`schema.sql` followed by a re-run. `schema.sql` records the `CREATE`
statements as they were run once, at project start, and is kept for
reference — what the base schema looked like — not as an install script. A
fresh database is provisioned by restoring a dump, never by piping
`schema.sql` into `psql`. See § 7 for why re-running it against a populated
database is actively unsafe, not just redundant.

### A dump is verified by `pg_restore -f /dev/null`, not `--list` or a checksum

A database dump is verified by `pg_restore -f /dev/null <file>` completing
with no stderr output. `pg_restore --list` is not verification — it reads
only the archive's table of contents, which sits at the head of the file.
On 2026-08-07 a dump that was one-third its correct size, with unreadable
data blocks, passed `--list` cleanly, matched its own SHA-256 across three
machines, and failed on restore. Any script that produces or consumes a
dump — including the Phase 11 nightly backup and its monthly restore test
— verifies with `-f /dev/null` or it has not verified anything. Full
diagnostic sequence in `CONTEXT.md`'s dated log entry.

---

## 4. Conventions

### SQL parameter casts: `CAST(:p AS type)`, never `:p::type`

SQLAlchemy 2.0's bind-parameter parser breaks on `:param::type` — the `::`
gets parsed as part of the token. Every cast on a *bound parameter* in this
codebase goes through `CAST(:x AS sometype)` (see `clients/aisstream.py`,
`clients/scraper.py`). Plain `expression::type` on a
column or function result (not a bind param) is fine and already used in a
few places (`normalizer/port_summary_builder.py`, `scheduler.py`) —
the rule is specifically about parameters, not all casts.
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

`features.lag_adjusted` is always written as a literal `False` — nothing left
in the pipeline has a publication lag to correct for (see `CONTEXT.md` for
what was removed and why).

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
| `AISSTREAM_API_KEY` | AISStream WebSocket API key |
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
within 24h (they just re-trigger on their next scheduled time); `wci-weekly`
— the only non-daily flow left — does not: a reset trigger time can mean it
silently never fires again until someone notices.

### Prefect server must be running before `scheduler.py` starts

`scheduler.py`'s `_require_prefect_api()` checks that `PREFECT_API_URL` is
set and reachable (`GET {url}/health`) before doing anything else, and
raises `SystemExit` if not. Without this check, Prefect's `serve()` would
silently spawn an ephemeral in-process server on a random port — the
scheduler would appear to start fine, but every flow run would be detached
from persistent history and vanish on restart.

---

## 8. Infrastructure (`infra/terraform/`)

The AWS footprint (EC2 instance, security group, Elastic IP, persistent
data EBS volume) is defined in `infra/terraform/` — see that directory's
`README.md` for apply order and state-backup instructions. As a contract:

- **The data EBS volume is never destroyed by tooling.** It holds AIS
  position history that cannot be re-collected. It carries
  `lifecycle { prevent_destroy = true }`; only a human, deliberately,
  detaches or deletes it — never an automated script, never a routine
  `terraform apply`.
- **The instance is replaceable.** Unlike the data volume, the instance
  itself has no `prevent_destroy` — it's meant to be rebuildable from the
  Terraform config plus the (separately maintained) provisioning scripts.
- **`terraform destroy` is never run against this configuration.** If
  deprovisioning is ever needed, it's a manual, resource-by-resource
  decision, not a single command.
- **The instance's SSM access is defined in Terraform (`iam.tf`), not a
  manual console step.**

### OS provisioning (`provision/`)

Idempotent scripts, run by the operator via `provision/install.sh`, turn
the Terraform-defined instance into a hardened Docker host — see
`provision/README.md` for run order and done-condition checks. As a
contract:

- **The data volume is referenced by UUID everywhere** — `/etc/fstab`,
  operator documentation, everything — never by device path (`/dev/sdf`,
  `/dev/nvme1n1`). Nitro device enumeration isn't guaranteed stable across
  reboots or instance-type changes; the UUID, read via `blkid` after
  formatting, is the only stable handle.
- **Finding the volume in the first place is also identity-based, not
  positional.** `provision/02-mount-data-volume.sh` requires
  `DATA_VOLUME_ID` (the real `aws_ebs_volume.data` ID, from
  `terraform output -raw data_volume_id`) and matches it against the NVMe
  controller's serial number — Nitro exposes the actual EBS volume ID
  there, readable from sysfs with no AWS API access needed. There is no
  fallback to "whichever disk isn't root": that was tried and removed —
  it only ever got exercised in the anomalous cases (an extra volume
  attached, a snapshot restore in progress) where a topology-based guess
  is most likely to be wrong.
- **ufw governs 22/tcp (scoped to `ADMIN_CIDR`), 80/tcp, and 443/tcp.**
  The security group and ufw are both gates in series, not redundant
  alternatives — traffic must clear both to reach the instance. ufw's
  `default deny incoming` applies independently of the security group, so
  with no ufw rule for 22, SSH is unreachable at the OS level the moment
  ufw activates, regardless of what the security group permits. This was
  discovered live: an earlier version of `provision/00-harden.sh` opened
  only 80/tcp and 443/tcp on the theory that the security group already
  handled port 22, and the first real `install.sh` run locked out SSH
  immediately, recovered only via SSM Session Manager — see CONTEXT.md.
  `00-harden.sh` now requires an `ADMIN_CIDR` env var (no default, same
  pattern as `DATA_VOLUME_ID`) sourced from the same Terraform
  `admin_cidr` value the security group rule already uses, and scopes
  ufw's SSH rule to it — never a bare `ufw allow 22`, which would open
  SSH to the world regardless of what the security group restricts.
- **`iptables -L DOCKER-USER -n` — not `ufw status` — is the source of
  truth for what's actually reachable through Docker.** Docker
  manipulates iptables directly, ahead of ufw's own rules, so a container
  that publishes a port is reachable regardless of what ufw reports.
  `provision/03-docker-user-chain.sh` installs an explicit default-deny in
  `DOCKER-USER`, reinstalled on every boot via systemd.

---

## 9. Schema migrations (`migrations/`)

Schema changes ship as additive, numbered SQL files under `migrations/`
(`NNNN_description.sql`), applied only by `orchestration/migrate.py` — see
`migrations/README.md` for the runner's full behavior and rationale (why
not Alembic, per-file transactions, zero-padded lexicographic ordering).

- **An applied migration is never edited.** The runner records a sha256
  checksum per applied file in `schema_migrations` and recomputes it on
  every run; a mismatch aborts the run and names the file. A schema
  change is always a new numbered file, never an edit to an old one.
- **`schema.sql` is a historical artifact.** The runner explicitly
  refuses to execute any file named `schema.sql`, anywhere under
  `migrations/` — see § 7's "Never run `schema.sql` against a populated
  database."
- **`job_runs.status` is constrained to exactly `running` / `success` /
  `failed`** by a `CHECK` constraint (`migrations/0001_job_runs.sql`),
  not just application-level convention. The eventual Phase 11 heartbeat
  queries `WHERE status = 'success'`; a typo'd status string fails at the
  database instead of silently making every job look overdue — or,
  worse, none.
- **No code path deletes from `job_runs`.** `orchestration/jobs.py`'s
  `@job` decorator only inserts a row once (`running`) and updates it in
  place to its terminal status — application code never issues a
  `DELETE FROM job_runs`. It is the audit trail for whether and how a job
  actually ran; a silently-deleted row is indistinguishable from a job
  that never ran at all, which is exactly the ambiguity the
  `running`-row-first design (see `orchestration/jobs.py`'s docstring)
  exists to rule out. Any future retention/cleanup need is an archival
  policy (e.g. move rows older than N to a separate table), not a delete
  in the application's own code path.

  This is a rule about automated code paths, not a claim the table is
  physically immutable. A human can still run a one-off `DELETE` by hand
  for a stated, specific reason — e.g. rows known to be fabricated rather
  than real job history (see CONTEXT.md, 2026-08-08) — same as `git
  push --force` being forbidden by default doesn't mean no human ever
  runs it. What's never acceptable is a script or a routine reaching for
  `DELETE` to make the table match some other belief about what "should"
  be true, or synthesizing a replacement row to paper over a gap instead
  of just recording that the gap exists.

---

## 10. Jobs (`orchestration/tasks.py`)

Every scheduled unit of work is a plain, zero-argument, `-> int` function
decorated `@job("name")` (`orchestration/jobs.py`) and registered in
`orchestration/tasks.py::JOBS`, a `dict[str, Callable[[], int]]`. Business
logic stays in `clients/`, `normalizer/`, `targets/`, `signals/`,
`models/`, and `alerts/` — `tasks.py` holds thin wrappers only, so a
mechanical orchestration change can't hide a behavioral regression inside
a large diff.

- **Exactly one `@job` per scheduled unit — never nested.** A multi-step
  chain (e.g. `normalizer-nightly`'s `port_resolver → vessel_normalizer →
  port_summary_builder → time_aligner → seasonal_adjuster →
  feature_builder`) runs as a single decorated call; no sub-step gets its
  own `@job`. A sub-step failure would otherwise write a `failed` row for
  itself *and* a `failed` row for the parent — double-counting one real
  failure in the health view — and a partial success would leave a mix of
  `success`/`failed` rows for what the schedule only ever knew as one
  unit, with no way to tell which outcome actually happened.
- **Job names are stable identifiers, not Python-convention renames.**
  The eight names (`port-call-refresh`, `bdi-daily`, `wci-weekly`,
  `normalizer-nightly`, `targets-nightly`, `signals-nightly`,
  `models-nightly`, `alerts-nightly`) are exactly the former Prefect
  deployment names, hyphens and all. They are the join key across
  CONTEXT.md's schedule table, Commit 3's cadence configuration, and the
  Phase 11 heartbeat's per-job overdue check — renaming one to match
  Python naming conventions orphans every one of those silently.
- **`rows_written` means rows written to the database, never rows
  examined or processed.** A job that parsed 50 candidates and wrote 3
  reports 3. A processed-count that reads nonzero while nothing landed
  downstream is this project's recurring failure theme reproduced inside
  its own monitoring. The one documented exception:
  `alerts/builder.py::run_all()` returns newly-inserted alerts only (not
  rows-affected by the upsert), since edge-triggered alerts are only
  meaningful the day they first fire — every job's docstring states in
  one sentence what its number counts, and the convention is never mixed
  silently between functions.
- **`0` and `NULL` are distinguishable and both meaningful.** `0` means
  the job ran and legitimately wrote nothing (e.g. a scraper with no new
  data to insert). `NULL` means the job didn't report a count at all.
  Collapsing either into the other loses a real, alertable distinction.
