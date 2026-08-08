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
