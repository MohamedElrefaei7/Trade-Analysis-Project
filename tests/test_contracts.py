"""
test_contracts.py — enforcement tests for the conventions documented in
CLAUDE.md. Each test here should go red the moment the thing it guards is
reverted; a documented convention with no test is just a comment.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_EXCLUDED_DIR_PARTS = {"venv", ".venv", ".git", "__pycache__", "node_modules"}


def _repo_py_files() -> list[Path]:
    out = []
    for path in REPO_ROOT.rglob("*.py"):
        if _EXCLUDED_DIR_PARTS & set(path.parts):
            continue
        out.append(path)
    return out


# ---------------------------------------------------------------------------
# test_no_postgres_cast_shorthand
# ---------------------------------------------------------------------------
#
# The convention (CLAUDE.md § Conventions) is narrow: a *bound parameter*
# must never be cast with Postgres's `:param::type` shorthand, because
# SQLAlchemy 2.0's bind-parser breaks on it. Plain `expression::type` on a
# column or function result (not a bind param) is valid Postgres and is
# already used in this codebase (e.g. `gs::date` in
# normalizer/port_summary_builder.py) — so a blanket "no `::` anywhere"
# check would false-positive on those.
#
# To scope this reliably (as asked), we parse each file with `ast`, find
# every call to a function named `text` (i.e. `sqlalchemy.text(...)`), pull
# out any string-literal / f-string argument, and search *only* inside that
# extracted SQL text for the `:identifier::` pattern — a bind parameter
# immediately followed by a Postgres cast. This can't fire on Python type
# annotations or comments, since we never look outside `text()` call
# arguments.

_BIND_PARAM_CAST_RE = re.compile(r":[A-Za-z_][A-Za-z0-9_]*::")


def _string_literal_value(node: ast.AST) -> str | None:
    """Best-effort extraction of the string content of a text() argument,
    including f-strings (JoinedStr) by concatenating their constant parts.
    Interpolated {expressions} are dropped — we only need the literal SQL
    skeleton, not the runtime values."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = [
            v.value for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        ]
        return "".join(parts) if parts else None
    return None


def _text_call_sql_strings(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "text":
            continue
        for arg in node.args:
            sql = _string_literal_value(arg)
            if sql is not None:
                yield sql


def test_no_postgres_cast_shorthand():
    violations: list[str] = []
    for py_file in _repo_py_files():
        try:
            source = py_file.read_text()
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for sql in _text_call_sql_strings(tree):
            for m in _BIND_PARAM_CAST_RE.finditer(sql):
                violations.append(
                    f"{py_file.relative_to(REPO_ROOT)}: found {m.group(0)!r} — "
                    f"use CAST({m.group(0)[:-2]} AS type) instead"
                )

    assert not violations, (
        "Postgres `:param::type` bind-parameter cast shorthand found "
        "(SQLAlchemy 2.0's bind parser breaks on this — use "
        "CAST(:param AS type) instead):\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# test_conclusions_module_is_pure
# ---------------------------------------------------------------------------
#
# dashboard/conclusions.py must import neither streamlit nor sqlalchemy —
# directly or transitively — so it stays unit-testable against synthetic
# DataFrames without a live Streamlit session or database. A direct-import
# check via `ast` would miss a transitive import (e.g. a detector adding
# `from clients.base import Session`, which itself imports sqlalchemy), so
# we import the module in a clean subprocess and inspect sys.modules
# afterward — that catches transitive imports regardless of what else this
# test session has already imported.

def test_conclusions_module_is_pure():
    script = (
        "import sys\n"
        "import dashboard.conclusions\n"
        "leaked = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m == 'streamlit' or m == 'sqlalchemy'\n"
        "    or m.startswith('streamlit.') or m.startswith('sqlalchemy.')\n"
        ")\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"importing dashboard.conclusions failed:\n{result.stderr}"
    )
    leaked = [m for m in result.stdout.strip().split(",") if m]
    assert not leaked, (
        f"dashboard/conclusions.py pulled in {leaked!r} (directly or "
        "transitively) at import time — it must stay a pure module "
        "(dataclasses/numpy/pandas only) so detectors are unit-testable "
        "against synthetic DataFrames without Streamlit or a database."
    )


# ---------------------------------------------------------------------------
# test_trainer_uses_gap
# ---------------------------------------------------------------------------
#
# models/trainer.py must construct every TimeSeriesSplit with a non-zero
# `gap`. Without it, overlapping forward-return targets leak into training
# and out-of-sample scores look better than they are — the single most
# damaging revert available in this codebase, because it fails invisibly.

def test_trainer_uses_gap():
    trainer_path = REPO_ROOT / "models" / "trainer.py"
    tree = ast.parse(trainer_path.read_text(), filename=str(trainer_path))

    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "TimeSeriesSplit")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "TimeSeriesSplit")
        )
    ]
    assert calls, "models/trainer.py no longer constructs any TimeSeriesSplit — expected at least one"

    for call in calls:
        gap_kwargs = [kw for kw in call.keywords if kw.arg == "gap"]
        assert gap_kwargs, (
            f"TimeSeriesSplit(...) call at {trainer_path}:{call.lineno} is missing "
            "the `gap` argument — without it, overlapping forward-return targets "
            "leak into training and OOS scores are silently inflated."
        )
        gap_value = gap_kwargs[0].value
        is_literal_zero = isinstance(gap_value, ast.Constant) and gap_value.value == 0
        assert not is_literal_zero, (
            f"TimeSeriesSplit(...) call at {trainer_path}:{call.lineno} sets gap=0 — "
            "this reintroduces target leakage between train and test folds."
        )


# ---------------------------------------------------------------------------
# test_unique_keys_match_documented_contract
# ---------------------------------------------------------------------------
#
# Parses the unique-key table out of CLAUDE.md § Hard invariants and checks
# it against the live database's actual UNIQUE constraints. Read-only:
# only ever issues SELECTs against information_schema. Skips (not fails)
# if DATABASE_URL is unset or the database is unreachable, so the suite
# still runs cleanly on a fresh clone with no database up.

_CLAUDE_MD_TABLE_ROW_RE = re.compile(
    r"^\|\s*`(?P<table>[a-z_]+)`\s*\|\s*`\((?P<cols>[^)]+)\)`\s*\|\s*$"
)


def _parse_documented_unique_keys() -> dict[str, list[str]]:
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text().splitlines()
    documented: dict[str, list[str]] = {}
    for line in claude_md:
        m = _CLAUDE_MD_TABLE_ROW_RE.match(line.strip())
        if not m:
            continue
        table = m.group("table")
        cols = [c.strip() for c in m.group("cols").split(",")]
        documented[table] = cols
    return documented


def test_unique_keys_match_documented_contract():
    documented = _parse_documented_unique_keys()
    assert documented, "Could not parse any unique-key rows out of CLAUDE.md — did the table format change?"

    try:
        from dotenv import load_dotenv
        load_dotenv()
        from sqlalchemy import create_engine, text
    except ImportError:
        pytest.skip("sqlalchemy/dotenv not installed — skipping live-schema contract check")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set (checked process env and .env) — skipping live-schema contract check")
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT tc.table_name,
                           string_agg(kcu.column_name, ',' ORDER BY kcu.ordinal_position) AS cols
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema    = kcu.table_schema
                    WHERE tc.table_schema = 'public'
                      AND tc.constraint_type = 'UNIQUE'
                      AND tc.table_name = ANY(:tables)
                    GROUP BY tc.table_name, tc.constraint_name
                    """
                ),
                {"tables": list(documented.keys())},
            ).fetchall()
    except Exception as exc:
        pytest.skip(f"Database unreachable at DATABASE_URL — skipping ({exc})")

    actual: dict[str, list[str]] = {}
    for table_name, cols in rows:
        actual.setdefault(table_name, []).append([c.strip() for c in cols.split(",")])

    mismatches = []
    for table, doc_cols in documented.items():
        db_variants = actual.get(table)
        if not db_variants:
            mismatches.append(f"{table}: CLAUDE.md documents {doc_cols!r} but no UNIQUE constraint exists in the database")
            continue
        if not any(sorted(variant) == sorted(doc_cols) for variant in db_variants):
            mismatches.append(
                f"{table}: CLAUDE.md documents {doc_cols!r}, "
                f"database UNIQUE constraint(s) are {db_variants!r}"
            )

    assert not mismatches, (
        "CLAUDE.md's documented unique keys no longer match the live schema:\n"
        + "\n".join(mismatches)
    )


# ---------------------------------------------------------------------------
# test_regime_change_score_never_exceeds_70 / test_regime_change_score_still_scales
# ---------------------------------------------------------------------------
#
# _score_regime_change is documented (dashboard/conclusions.py's own scoring
# comment, and the general "0-100 scale" contract in CLAUDE.md) as capping out
# around 70. The implementation multiplies an already-capped `base` by
# `(current_abs_r / 0.5)`, which is > 1 whenever current_abs_r > 0.5 — a common
# value, not an edge case — pushing the product well past 70. These two tests
# pin the fix: the ceiling must hold, and it must hold without flattening the
# ranking the scoring system exists for.

from dashboard.conclusions import _score_regime_change  # noqa: E402


@pytest.mark.parametrize(
    "correlation_delta,current_abs_r",
    [
        (0.20, 0.50),   # boundary case — already respected the cap pre-fix
        (0.30, 0.70),
        (0.50, 0.90),   # exactly the values that broke the cap before the fix
        (0.50, 1.00),
        (0.99, 1.00),
    ],
)
def test_regime_change_score_never_exceeds_70(correlation_delta, current_abs_r):
    score = _score_regime_change(correlation_delta, current_abs_r)
    assert score <= 70.0, (
        f"_score_regime_change({correlation_delta}, {current_abs_r}) = {score} "
        "exceeds the documented 70 ceiling"
    )


def test_regime_change_score_still_scales():
    weak = _score_regime_change(correlation_delta=0.05, current_abs_r=0.10)
    strong = _score_regime_change(correlation_delta=0.15, current_abs_r=0.40)

    assert strong > weak, (
        "a stronger correlation shift must still score higher than a weaker "
        "one — a fix that clamps every result to 70 would pass the ceiling "
        "test while destroying this ranking"
    )
    assert strong <= 70.0
    assert weak <= 70.0
