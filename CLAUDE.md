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
few places (`normalizer/port_summary_builder.py`) —
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
  not just application-level convention. The `heartbeat` job (§ 13)
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
- **`port_calls` is never rewritten in bulk.** A suspected data-quality
  problem in `port_calls` — e.g. the restart-spike phantom arrivals found
  and measured via `port_calls_derived`/`normalizer/rederive.py` — gets
  corrected by deriving a parallel table and comparing, never by an
  UPDATE/DELETE against `port_calls` itself as part of the measurement.
  `port_calls` has two legitimate writers by design (§ 2's layer
  contract) and is the only record of arrivals whose supporting position
  pings may predate what's still in `positions`; a bulk rewrite driven by
  a re-derivation would destroy exactly the rows the measurement exists
  to identify, before anyone had seen the numbers. Any decision to
  actually alter the stored rows is a separate, human-approved step taken
  after the numbers are in hand — not a step inside the measurement
  itself.

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

---

## 11. Scheduling (`worker/cadences.py`, `worker/main.py`)

`worker/cadences.py::CADENCES` is the single source of truth for both how
each job in `orchestration/tasks.py::JOBS` is triggered and how long it
may go without a successful run before it's overdue — one `Cadence` entry
per job carries both `trigger` and `max_age`. The eventual Phase 11
heartbeat imports this module and reads `max_age` from it; it never
restates an expected age of its own. This project already has the worked
example of what happens when those two facts live in separate places:
`wci-weekly` read as stale for two and a half months because a freshness
check that didn't know WCI's own cadence couldn't tell "overdue" from
"its Friday hasn't come yet." `max_age` is set well above its job's
interval (roughly 1.25×) specifically so a job that's merely mid-run
isn't flagged overdue.

- **The scheduler uses a `SQLAlchemyJobStore` (same Postgres database),
  never APScheduler's default in-memory job store.** A restart is a new
  process; an in-memory store remembers nothing about the previous
  process's schedule, so on restart it computes a brand-new "now +
  interval" `next_run_time` and never notices anything was missed at
  all — coalesce and `misfire_grace_time` have nothing to act on if the
  scheduler doesn't know a slot passed. This is not a hypothetical: the
  first version of `worker/main.py` used the in-memory default and a live
  restart-recovery verification produced *zero* catch-up runs, not one
  (see CONTEXT.md, 2026-08-08). `build_scheduler()` also deliberately does
  **not** re-`add_job()` a job that already has a persisted row —
  `add_job(..., replace_existing=True)` recomputes `next_run_time` fresh
  on every call, which would silently overwrite the persisted value (and
  the fact that a slot was missed) on every restart; APScheduler's own
  job-processing loop queries the jobstore for due jobs directly, so an
  already-persisted job needs no re-registration to be picked up
  correctly. Trade-off, stated plainly: a `cadences.py` change has no
  effect on an already-persisted job until its stored row is cleared —
  restart correctness was the point, not live config reload.
- **Every job sets `coalesce=True` and `misfire_grace_time` explicitly —
  never the library default.** APScheduler's own default grace is one
  second; under that default, a fire time that passes while the process
  is down is silently skipped and forgotten, which is the exact failure
  this whole migration/worker effort exists to eliminate, re-created
  inside the tool meant to fix it. Combined with the persistent job store
  above, `coalesce=True` (collapse any backlog of missed fires into one)
  and a grace roughly proportional to each job's own interval produce the
  property that matters: **a restart after an outage catches every
  overdue job up once, promptly — it does not wait for the job's next
  naturally scheduled slot.** This is correct specifically because every
  job in this pipeline recomputes from current database state rather than
  consuming a per-slot increment; a future job that processes one
  discrete slot per fire would silently lose work under `coalesce=True`
  and needs its own judgment call, not this default.
- **Every job sets `max_instances=1`.** A run that overruns into its own
  next slot must queue, not overlap — two concurrent writers racing the
  same upsert key is a race with no acceptable winner.
- **All triggers, and the scheduler itself, are explicitly UTC.**
  `CronTrigger`/`IntervalTrigger` default to the local timezone; the
  container happens to run UTC today, but that's incidental, not
  contracted — a base-image change or a stray `TZ` env var would silently
  shift every schedule with no error anywhere. `timezone=` is passed
  explicitly everywhere a trigger or scheduler is constructed.
- **`job_runs.status` is one of `running` / `success` / `failed` /
  `missed`** (`migrations/0002_job_runs_missed_status.sql` extended the
  `CHECK`). `missed` means the fire time passed outside
  `misfire_grace_time` and the `@job`-decorated function never ran at
  all — APScheduler's executor skips the call entirely, so nothing in the
  normal decorator path ever executes. `worker/main.py`'s
  `EVENT_JOB_MISSED` listener is the only thing that makes this outcome
  visible; without it, a misfire looks identical to "never scheduled."
  `missed` is deliberately distinct from `failed` (the job didn't error —
  it never ran) and must stay distinguishable from both.
- **`worker/main.py` refuses to start if `CADENCES` and `JOBS` disagree.**
  Checked at startup via a symmetric-difference comparison, before the
  scheduler blocks — a job with no cadence would otherwise simply never
  run, and a cadence naming no job would crash hours later at fire time,
  in a log nobody's watching.
- **`apscheduler_jobs` is created and owned entirely by APScheduler's
  `SQLAlchemyJobStore`** — not by any migration in `migrations/`, and
  deliberately absent from `schema_migrations`. It holds pickled Python
  state (each row's `job_state` column), not plain relational data, so it
  is never hand-modified and never hand-migrated: a schema change to a
  `@job`-decorated function's signature or to `Cadence` itself is not
  something you edit this table to reflect. Its contents are disposable —
  deleting every row costs exactly one scheduling cycle, since
  `worker/main.py` re-registers every job from `CADENCES` on startup (see
  above: it does not re-`add_job()` an already-persisted job, but an
  *empty* jobstore has nothing persisted to skip, so a fresh start from
  zero rows re-adds all nine cleanly). Phase 11's backup job must exclude
  it (`pg_dump --exclude-table=apscheduler_jobs`): restoring pickled
  scheduler state from a month-old dump resurrects stale fire times and
  may not even unpickle cleanly against a newer APScheduler version than
  the one that wrote it — the disposability above is exactly why excluding
  it from the backup costs nothing.

---

## 12. AIS daemon (`ais/main.py`, `clients/aisstream.py`)

The AIS ingest daemon splits across two files, each owning exactly one
concern: `clients/aisstream.py` owns subscription construction, message
parsing, the `_vessel_state` cache, and the `vessels`/`positions`/
`port_calls` writes; `ais/main.py` owns the connection lifecycle —
connect, hand messages to the client, decide whether a dropped
connection is routine or fatal, and respond to signals. Run standalone
with `python -m ais.main`.

As contract:

- **Transient WebSocket disconnects are handled in-process**, with
  capped exponential backoff (`ais/main.py::run_forever()` — starts near
  5s, caps near 60s). This is what stops a routine network blip from
  recycling the whole container.
- **Sustained failure escalates — it is never absorbed.** Past
  `MAX_CONSECUTIVE_FAILURES` (10) consecutive failed connection
  attempts, the process raises `ConnectionFailureLimitExceeded` and
  exits non-zero, handing recovery to Docker's `restart: unless-stopped`
  policy. A successful connection, however brief, resets the failure
  count to zero. The process never decides on its own that persistent
  failure is survivable — a daemon that swallows every exception and
  retries forever looks, to Docker, like a healthy running process while
  ingesting nothing, and nothing downstream would ever know.
- **Unknown prior vessel state at cold start records no transition.**
  `_vessel_state` is in-memory and empty on every restart;
  `clients/aisstream.py::_update_port_call()` treats "no cached prior
  nav_status" as exactly that — not as "underway" — and just records the
  observed state instead of writing a spurious arrival for every vessel
  already sitting in port. `normalizer/vessel_normalizer.py`'s nightly
  48-hour lookback over `positions` is the recovery path for whatever
  this misses; the guard is only safe because that recovery exists.
- **Exit codes are deliberate, not incidental.** `SIGTERM` (Docker's stop
  signal, sent before `SIGKILL`) triggers a clean shutdown — close the
  socket, let any open session unwind — and exits `0`, so
  `docker compose down` never reads as a crash in the logs. Hitting the
  failure ceiling exits non-zero, so it does.
- **This daemon writes nothing to `job_runs`.** It's a continuous
  process, not a scheduled unit that returns (see § 10's `@job`
  contract) — a process reporting its own liveness is the same pattern
  that let Prefect flows record "Completed" through a dead stack. AIS
  liveness must be observed from outside, by something that checks
  whether `positions.ts` is actually still advancing, not by asking the
  daemon whether it feels fine.

---

## 13. Monitoring (`orchestration/heartbeat.py`)

The `heartbeat` job (hourly, Phase 3 Commit 6) is two independent freshness
checks — one for every job in `JOBS`/`CADENCES`, one for the AIS feed
itself — that post a Slack digest when something's overdue or stale.

- **`max_age` comes from `worker/cadences.py::CADENCES` only —
  `heartbeat.py` defines no thresholds of its own.** `wci-weekly` once
  read as stale for two and a half months because a separate freshness
  check kept its own copy of "should run weekly" out of sync with the
  real cadence (§ 11). Every job in `CADENCES` is checked; there is no
  opt-out list.
- **"Last success" means the most recent `job_runs` row with
  `status = 'success'`, by `finished_at`** — never the most recent row of
  any status. A job that has failed every night for a week has recent
  `job_runs` rows and no recent successes; keying off the latest row
  regardless of status would read that as healthy.
- **`never_succeeded` and `stale_running` are distinct reported states,
  never folded into `overdue`.** A job with no success row ever
  (`never_succeeded`) is the most alarming state a job can be in — a
  naive `LEFT JOIN` silently drops it, so it's surfaced explicitly
  instead, with no fabricated age. A job whose most recent `running` row
  is older than its own `max_age` (`stale_running`) means the process
  died mid-run, distinct from both `overdue` and `failed`; when a job has
  a stale running row, it's reported only under `stale_running`, not
  double-counted under `overdue` too, even if its last real success is
  also old.
- **AIS liveness is measured from `MAX(ts) FROM positions`, never from
  the AIS daemon's own status.** The daemon writes nothing to `job_runs`
  by design (§ 12) — a process reporting its own liveness is the same
  failure mode Prefect had. `aisstream.io` went into silent failure on
  2026-08-05 (connection accepted, subscription accepted, zero messages
  delivered) and it was found by hand on 2026-08-08; every layer above
  the data itself reported healthy. `AIS_STALE_THRESHOLD` (6 hours) is a
  provider-outage detector, not a data-volume check — a feed running at a
  fraction of normal volume still advances `MAX(ts)` and passes cleanly.
- **A Slack posting failure never fails the heartbeat job.** If
  `SLACK_WEBHOOK_URL` is unset, the findings are logged at `WARNING` and
  the job still records `success`. If the post itself raises, it's caught
  and logged, never re-raised — a Slack outage must not make the
  monitoring's own failure state indistinguishable from the thing it
  monitors failing. Reuses `alerts/builder.py::_maybe_post_slack()`
  rather than a second webhook poster.
- **`heartbeat` is itself a `@job`, registered in `JOBS`/`CADENCES` like
  every other one — deliberately, not incidentally.** It writes its own
  `job_runs` rows, so it checks its own pulse: if it stops running, its
  own last success ages out and the next run that does happen reports
  it. That does not cover the heartbeat never running again at all —
  nothing internal to the scheduler can detect that; Phase 11's
  UptimeRobot check against `/api/health` is what closes that gap.
- **Its `rows_written` is the count of problems found, not a row
  count** — overdue jobs + never-succeeded jobs + stale-running jobs,
  plus one if AIS is stale. Zero is the healthy state. This is the
  second job, after `alerts-nightly`, where the number isn't rows written
  to a table (§ 10).

---

## 14. Deployment (`Dockerfile`, `docker-compose.yml`)

`worker` and `ais` run as containers, both built from the same root
`Dockerfile` and both under `restart: unless-stopped`, alongside
`timescaledb` and `grafana`. As contract:

- **One image serves both `worker` and `ais`, differentiated only by
  `command:`.** They import the same `clients/`/`orchestration/` tree and
  the same `requirements.txt`; two Dockerfiles would mean two dependency
  installs free to drift apart, surfacing as one service having a library
  version the other doesn't, in production, months later.
- **The deployed Python runtime is pinned at 3.12 (`python:3.12-slim`),
  while local development runs 3.14.** This divergence is deliberate, not
  an oversight — the server's system Python and the venv the live
  verifications actually ran under are both 3.12. Base images are always
  pinned to a specific minor version, never `latest` or a bare major tag
  — `timescale/timescaledb:latest-pg16` resolving to two different
  TimescaleDB versions three months apart already cost a debugging
  session over exactly this.
- **All image tags are pinned, never `latest`.** Applies to every service
  in `docker-compose.yml`, not just the application image.
- **Migrations are an explicit operator step, never run on container
  start.** `docker compose run --rm worker python -m orchestration.migrate`,
  by hand, before `docker compose up -d` — see `provision/README.md`. A
  migration is a deliberate act against a database holding irreplaceable
  data; coupling it to container restart turns a restart loop into a
  migration loop.
- **No service publishes a port except `caddy`, in a later phase.**
  Nothing outside the Compose network talks to `worker` or `ais`
  directly, and `timescaledb`/`grafana` stay bound to `127.0.0.1`. The
  `DOCKER-USER` iptables chain (§ 8) is the backstop, not the primary
  defense — not publishing the port is.
- **`worker` and `ais` run as a non-root user inside the container.** Root
  inside a container isn't root on the host, but it's one
  container-escape away from it, and neither process needs it.
- **`depends_on: timescaledb` only waits for the container to start, not
  for Postgres to accept connections.** A cold boot's first connection
  attempt from `worker`/`ais` can fail and get restarted by Docker — that
  is the supervision model working, not a fault. A
  healthcheck-gated `depends_on: condition: service_healthy` is a
  reasonable later refinement, deliberately not done yet — it adds a
  failure mode (a healthcheck that never passes blocks startup silently)
  that should not be introduced before the plain retry path has been
  exercised for real.
