"""
migrate.py — numbered-SQL migration runner.

Not Alembic: Alembic's autogenerate diffs SQLAlchemy models against the
live database, and against TimescaleDB it sees hypertable chunks under
_timescaledb_internal as unmanaged tables — the generated migration would
contain DROP TABLE statements for chunks holding AIS history that cannot
be re-collected. Plain numbered SQL files are reviewable as text and carry
no such risk. See migrations/README.md for the full rationale.

Usage:
    python -m orchestration.migrate
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import psycopg2

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
MIGRATION_FILENAME_RE = re.compile(r"^\d{4}_[A-Za-z0-9_]+\.sql$")

# A migration whose first line is exactly this marker runs with
# autocommit on, outside any transaction block — see
# _apply_migration_no_transaction()'s docstring for why this exists and
# what it gives up to get it.
NO_TRANSACTION_MARKER = "-- migrate:no-transaction"


class MigrationError(Exception):
    """Raised for any condition that must stop a migration run cold:
    schema.sql present, a checksum mismatch, a malformed filename, or a
    migration's DDL failing."""


def run_migrations(database_url: str, migrations_dir: Path | str = MIGRATIONS_DIR) -> list[str]:
    """
    Apply every pending migration in migrations_dir against database_url,
    one file at a time, each in its own transaction — except a file whose
    first line is exactly NO_TRANSACTION_MARKER, which runs with
    autocommit on instead (see _apply_migration_no_transaction() for why:
    statements like CREATE INDEX CONCURRENTLY refuse to run inside a
    transaction block at all, so no amount of per-file transaction
    wrapping can support them). Returns the filenames applied, in
    application order. A no-op if everything is already applied. Raises
    MigrationError without applying anything further if schema.sql is
    present, an already-applied file's checksum no longer matches, or a
    migration's DDL fails.
    """
    migrations_dir = Path(migrations_dir)
    _refuse_schema_sql(migrations_dir)
    files = _discover_migration_files(migrations_dir)

    conn = psycopg2.connect(database_url)
    conn.autocommit = False
    try:
        _ensure_schema_migrations_table(conn)
        applied = _get_applied_versions(conn)
        _verify_checksums(files, applied)

        newly_applied = []
        for path in files:
            version = _version_of(path)
            if version in applied:
                continue
            _apply_migration(conn, path, version)
            newly_applied.append(path.name)
        return newly_applied
    finally:
        conn.close()


def _refuse_schema_sql(migrations_dir: Path) -> None:
    matches = sorted(migrations_dir.rglob("schema.sql"))
    if matches:
        raise MigrationError(
            f"Refusing to run: found schema.sql at {matches[0]}. schema.sql "
            "is a historical artifact and is never applied by this runner "
            "— see CLAUDE.md § Hard invariants."
        )


def _discover_migration_files(migrations_dir: Path) -> list[Path]:
    files = sorted(p for p in migrations_dir.iterdir() if p.is_file() and p.suffix == ".sql")
    seen: dict[int, Path] = {}
    for path in files:
        if not MIGRATION_FILENAME_RE.match(path.name):
            raise MigrationError(
                f"Migration filename does not match the NNNN_name.sql convention: {path.name}"
            )
        version = _version_of(path)
        if version in seen:
            raise MigrationError(
                f"Duplicate migration version {version:04d}: {seen[version].name} and {path.name}"
            )
        seen[version] = path
    return files


def _ensure_schema_migrations_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    INTEGER PRIMARY KEY,
                filename   TEXT NOT NULL UNIQUE,
                checksum   TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
    conn.commit()


def _get_applied_versions(conn) -> dict[int, tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT version, filename, checksum FROM schema_migrations")
        return {version: (filename, checksum) for version, filename, checksum in cur.fetchall()}


def _verify_checksums(files: list[Path], applied: dict[int, tuple[str, str]]) -> None:
    by_version = {_version_of(path): path for path in files}
    for version, (filename, recorded_checksum) in applied.items():
        path = by_version.get(version)
        if path is None:
            raise MigrationError(
                f"schema_migrations records version {version:04d} ({filename}) as "
                "applied, but that file no longer exists in the migrations directory."
            )
        current_checksum = _sha256_bytes(path.read_bytes())
        if current_checksum != recorded_checksum:
            raise MigrationError(
                f"{path.name} has changed on disk since it was applied "
                f"(recorded checksum {recorded_checksum}, current {current_checksum}). "
                "An applied migration must never be edited — add a new migration instead."
            )


def _requires_no_transaction(sql_text: str) -> bool:
    lines = sql_text.splitlines()
    return bool(lines) and lines[0].strip() == NO_TRANSACTION_MARKER


def _apply_migration(conn, path: Path, version: int) -> None:
    sql_text = path.read_text()
    checksum = _sha256_bytes(path.read_bytes())

    if _requires_no_transaction(sql_text):
        _apply_migration_no_transaction(conn, path, version, sql_text, checksum)
        return

    try:
        with conn.cursor() as cur:
            cur.execute(sql_text)
            cur.execute(
                "INSERT INTO schema_migrations (version, filename, checksum, applied_at) "
                "VALUES (%s, %s, %s, now())",
                (version, path.name, checksum),
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise MigrationError(f"Migration {path.name} failed: {exc}") from exc


def _apply_migration_no_transaction(conn, path: Path, version: int, sql_text: str, checksum: str) -> None:
    """
    Applies a `-- migrate:no-transaction`-marked migration with autocommit
    on, so its SQL runs with no surrounding transaction block. This
    exists for statements PostgreSQL refuses to run inside one at all —
    CREATE INDEX CONCURRENTLY chief among them — which the ordinary path
    (every file wrapped in one transaction, per this module's docstring)
    cannot support no matter how the file is written.

    A marked file must contain exactly one statement after the marker
    line. The simple query protocol implicitly wraps a multi-statement
    string in its own BEGIN/COMMIT regardless of the connection's
    autocommit setting, which would silently reintroduce the same
    transaction block this exists to avoid.

    This trades away the ordinary path's all-or-nothing guarantee,
    deliberately, because there's no alternative that keeps it: a
    CONCURRENTLY build that fails partway can leave an INVALID index
    object behind, and — a known PostgreSQL sharp edge, not a bug here —
    a later `IF NOT EXISTS` rerun will skip right over it rather than
    repairing it, because the name already exists even though the index
    itself is unusable. An operator hitting that needs to check
    `pg_index.indisvalid` and `DROP INDEX` it before retrying; this
    runner has no way to detect or fix that for them. The
    schema_migrations row is still only written after the DDL succeeds —
    same contract as every other migration — it's just that "succeeds"
    no longer means "atomically, together with that row."
    """
    # psycopg2 refuses to toggle autocommit while a transaction is open
    # (e.g. left open by _get_applied_versions()'s/_verify_checksums()'s
    # reads, or a prior ordinary migration's implicit transaction) — close
    # it out first. Nothing here needs rolling back; those were all reads
    # or already-committed writes.
    conn.commit()
    original_autocommit = conn.autocommit
    conn.autocommit = True
    try:
        try:
            with conn.cursor() as cur:
                cur.execute(sql_text)
        except Exception as exc:
            raise MigrationError(f"Migration {path.name} failed: {exc}") from exc
    finally:
        conn.autocommit = original_autocommit

    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO schema_migrations (version, filename, checksum, applied_at) "
                "VALUES (%s, %s, %s, now())",
                (version, path.name, checksum),
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise MigrationError(
            f"Migration {path.name}'s DDL succeeded but recording it in "
            f"schema_migrations failed: {exc}"
        ) from exc


def _version_of(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    try:
        database_url = os.environ["DATABASE_URL"]
    except KeyError:
        print("DATABASE_URL is not set — refusing to guess a target database.", file=sys.stderr)
        return 1

    try:
        applied = run_migrations(database_url)
    except MigrationError as exc:
        print(f"Migration run failed: {exc}", file=sys.stderr)
        return 1

    if applied:
        print(f"Applied {len(applied)} migration(s):")
        for name in applied:
            print(f"  {name}")
    else:
        print("Up to date — no pending migrations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
