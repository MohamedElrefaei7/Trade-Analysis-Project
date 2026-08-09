"""
test_ais.py — enforcement tests for ais/main.py's connection-lifecycle
supervision (retry/backoff, consecutive-failure ceiling, SIGTERM
handling) and for clients/aisstream.py's cold-start no-fabricated-arrival
guard.

The WebSocket is mocked throughout — no test opens a real connection or
needs an AISSTREAM_API_KEY. Also DB-free: the daemon never touches
job_runs (see ais/main.py's docstring), and ensure_port_cache_loaded()
is monkeypatched wherever serve()/main() runs end to end, so nothing
here needs a real Postgres connection beyond what importing clients.base
already requires (DATABASE_URL from .env).
"""

from __future__ import annotations

import ast
import asyncio
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import ais.main as ais_main
import clients.aisstream as aisstream

REPO_ROOT = Path(__file__).resolve().parent.parent
_EXCLUDED_DIR_PARTS = {"venv", ".venv", ".git", "__pycache__", "node_modules"}


def _repo_py_files() -> list[Path]:
    return [
        p for p in REPO_ROOT.rglob("*.py")
        if not (_EXCLUDED_DIR_PARTS & set(p.parts))
    ]


async def _no_sleep(_seconds: float) -> None:
    return None


# ---------------------------------------------------------------------------
# run_forever() — retry / backoff / failure ceiling (decision 3)
# ---------------------------------------------------------------------------


def test_exits_nonzero_after_consecutive_connection_failures():
    """Must go red if the retry loop is unbounded: a connect() that
    always raises has to eventually escalate, not loop forever."""
    async def always_fails():
        raise ConnectionRefusedError("no route to host")

    async def run():
        try:
            await ais_main.run_forever(
                always_fails,
                sleep=_no_sleep,
                max_consecutive_failures=10,
            )
        except ais_main.ConnectionFailureLimitExceeded:
            return
        raise AssertionError("run_forever did not raise after the failure ceiling")

    asyncio.run(run())


def test_transient_disconnect_does_not_exit():
    attempts = {"n": 0}

    async def fail_once_then_succeed():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionRefusedError("transient")
        return None

    async def run():
        # Two attempts: one failure, one success. If run_forever raised or
        # left the failure counter nonzero, this would either propagate an
        # exception here or fail the assertion below.
        remaining_failures = await ais_main.run_forever(
            fail_once_then_succeed,
            sleep=_no_sleep,
            max_consecutive_failures=10,
            max_attempts=2,
        )
        assert remaining_failures == 0, "failure counter did not reset after success"

    asyncio.run(run())


def test_backoff_grows_and_is_capped():
    recorded: list[float] = []

    async def record_sleep(seconds: float) -> None:
        recorded.append(seconds)

    async def always_fails():
        raise ConnectionRefusedError("still down")

    async def run():
        await ais_main.run_forever(
            always_fails,
            sleep=record_sleep,
            initial_backoff=5.0,
            max_backoff=60.0,
            max_consecutive_failures=1000,
            max_attempts=8,
        )

    asyncio.run(run())

    assert recorded == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0, 60.0], recorded
    assert max(recorded) <= 60.0


# ---------------------------------------------------------------------------
# Cold-start port_call guard (decision 4)
# ---------------------------------------------------------------------------


class _FakeResult:
    def first(self):
        return None

    def fetchone(self):
        return None


class _FakeSession:
    def __init__(self):
        self.executed: list[str] = []

    def execute(self, stmt, params=None):
        self.executed.append(str(stmt))
        return _FakeResult()

    def commit(self):
        pass


def test_cold_start_records_no_port_call_for_already_moored_vessel(monkeypatch):
    """Must go red if an unknown prior state is treated as underway: a
    vessel with no cached state must not generate a port_calls INSERT on
    its very first ping, even if that ping is already 'moored'."""
    monkeypatch.setattr(aisstream, "_vessel_state", {})
    monkeypatch.setattr(
        aisstream, "nearest_port", lambda lat, lon, ports, max_km: ("NLRTM", 1.0)
    )

    session = _FakeSession()
    ts = datetime.now(timezone.utc)
    aisstream._update_port_call(session, "123456789", "vessel-uuid", 51.9, 4.5, "moored", ts)

    assert not any("INSERT INTO port_calls" in s for s in session.executed), session.executed
    assert aisstream._vessel_state["123456789"]["nav_status"] == "moored"


# ---------------------------------------------------------------------------
# SIGTERM handling (decision 5)
# ---------------------------------------------------------------------------


def test_sigterm_exits_zero(monkeypatch):
    monkeypatch.setattr(ais_main, "ensure_port_cache_loaded", lambda: None)

    async def long_lived_connection():
        await asyncio.Event().wait()  # blocks until the task is cancelled

    real_serve = ais_main.serve
    monkeypatch.setattr(ais_main, "serve", lambda: real_serve(connect=long_lived_connection))

    def send_sigterm_soon():
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=send_sigterm_soon, daemon=True).start()

    exit_code = ais_main.main()

    assert exit_code == 0


# ---------------------------------------------------------------------------
# scheduler.py / prefect removal (Part 2)
# ---------------------------------------------------------------------------


def _find_top_level_import_violations(module_name: str) -> list[str]:
    violations = []
    for py_file in _repo_py_files():
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == module_name or alias.name.startswith(module_name + "."):
                        violations.append(f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == module_name or mod.startswith(module_name + "."):
                    violations.append(f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}")
    return violations


def test_no_module_imports_scheduler():
    violations = _find_top_level_import_violations("scheduler")
    assert not violations, f"scheduler still imported at: {violations}"


def test_no_module_imports_prefect():
    violations = _find_top_level_import_violations("prefect")
    assert not violations, f"prefect still imported at: {violations}"


def test_prefect_absent_from_requirements():
    text = (REPO_ROOT / "requirements.txt").read_text().lower()
    assert "prefect" not in text, "prefect is still listed in requirements.txt"
