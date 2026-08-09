# migrations/

Numbered, reviewable SQL migrations, applied only by `orchestration/migrate.py`
— never by hand, never by re-running `schema.sql` (see `CLAUDE.md` §
Hard invariants for why `schema.sql` is a historical artifact, not an
install script).

## Why not Alembic

Alembic's autogenerate diffs SQLAlchemy models against the live database.
Against TimescaleDB it sees hypertable chunks under `_timescaledb_internal`
as unmanaged tables it doesn't recognize — the generated migration would
contain `DROP TABLE` statements for chunks holding AIS history that cannot
be re-collected. A migration tool whose headline feature can emit
destructive SQL against this schema is a liability, not a convenience.
Plain numbered SQL is reviewable as text, which matters more than
autogeneration on a project with one developer and one database.

## Naming convention

`NNNN_description.sql` — a zero-padded four-digit version prefix, applied
in lexicographic (== numeric, because of the padding) order. Non-padded
prefixes, where `2_foo.sql` sorts after `10_foo.sql`, are exactly the bug
this convention exists to rule out.

## Running

```bash
python -m orchestration.migrate
```

Reads `DATABASE_URL` from the environment — no default, so a migration
meant for the server can never silently land on a laptop. Applies every
pending migration, one file at a time, each in its own transaction: if a
migration's DDL fails, nothing from that file is left half-applied, and
its `schema_migrations` row is never written. A failure at file five never
touches files one through four's already-committed state.

Rerunning once everything is applied is a no-op.

## Migrations that can't run inside a transaction

Some statements — `CREATE INDEX CONCURRENTLY` is the one that comes up —
refuse to run inside a transaction block at all, which the normal
per-file-transaction path above can't accommodate no matter how the file
is written. A migration whose **first line** is exactly

```sql
-- migrate:no-transaction
```

runs with autocommit on instead — no surrounding transaction (see
`orchestration/migrate.py::_apply_migration_no_transaction`). Two
consequences worth knowing before reaching for this:

- **The file must contain exactly one statement** after the marker line.
  PostgreSQL's simple query protocol implicitly wraps a multi-statement
  string in its own transaction regardless of the connection's autocommit
  setting, which would silently reintroduce the transaction block this
  marker exists to avoid.
- **The all-or-nothing guarantee is gone for that file.** A
  `CREATE INDEX CONCURRENTLY` that fails partway can leave an `INVALID`
  index object behind under the index's name, and a later
  `IF NOT EXISTS` rerun will skip right past it instead of repairing it —
  the name already exists, even though the index itself is unusable.
  Check `pg_index.indisvalid` and `DROP INDEX` before retrying a marked
  migration that failed once. See `0004_positions_vessel_ts_index.sql`
  for a worked example, including this exact recovery note inline.

Reach for a plain `CREATE INDEX` (inside the normal transaction path,
no marker) whenever the table is small enough that the brief lock is a
non-issue — this marker is for tables where `CONCURRENTLY`'s
no-blocking-writes property actually matters, not a default.

## An applied migration is never edited

`schema_migrations` records a sha256 checksum of each applied file
alongside its version, filename, and `applied_at`. Every run recomputes
the checksum of every already-applied file and aborts — naming the file —
the moment one doesn't match what was recorded. If a migration needs to
change, write a new numbered file; don't touch an old one. The runner
enforces this so "the migration history says X" and "the database
actually has X" can never quietly diverge.

## `schema.sql` is refused outright

The runner refuses to execute anything named `schema.sql`, anywhere under
this directory, with an explicit error. `schema.sql` is a historical
artifact — see `CLAUDE.md` § Hard invariants — and this is the cheapest
possible guard against it ever being fed to the runner by accident.

## Not a hypertable reflex

Not every table with a timestamp column needs to be a TimescaleDB
hypertable. `job_runs` (`0001_job_runs.sql`) will hold a few thousand rows
a year — chunking it would add catalog complexity to the table the health
endpoint queries most, for no benefit.
