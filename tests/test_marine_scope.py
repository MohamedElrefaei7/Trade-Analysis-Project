"""
test_marine_scope.py — enforcement tests for the marine/AIS-only scope
reduction (removal of opensky, FRED, Comtrade, Port of LA, and the
lag_adjuster normalizer step). Each test should go red if the corresponding
removal is ever silently reverted.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Part 1 — clients/ inventory
# ---------------------------------------------------------------------------

def test_clients_module_inventory():
    """clients/ must contain exactly {base, aisstream, geo, scraper} (+ __init__).
    Red the moment opensky.py, fred.py, or comtrade.py is ever re-added."""
    modules = {
        p.stem for p in (REPO_ROOT / "clients").glob("*.py")
        if p.stem != "__init__"
    }
    assert modules == {"base", "aisstream", "geo", "scraper"}, (
        f"clients/ module inventory drifted — found {modules!r}, "
        "expected exactly {'base', 'aisstream', 'geo', 'scraper'}"
    )


def test_scraper_exports():
    """clients.scraper must still export the two TARGET-source scrapers
    (BDI, WCI) and must NOT export port_la_scraper. Guards both directions:
    over-deletion (targets removed) and under-deletion (port_la left behind)."""
    from clients import scraper

    assert hasattr(scraper, "bdi_scraper"), "clients.scraper lost bdi_scraper — BDI is a prediction target, not an out-of-scope feature source"
    assert hasattr(scraper, "wci_scraper"), "clients.scraper lost wci_scraper — WCI is a prediction target, not an out-of-scope feature source"
    assert not hasattr(scraper, "port_la_scraper"), "clients.scraper still exports port_la_scraper — it should have been removed"


def test_no_playwright_import():
    """No file under clients/ should reference playwright — it was only a
    dependency for the now-removed Port of LA scraper."""
    violations = []
    for py_file in (REPO_ROOT / "clients").glob("*.py"):
        if "playwright" in py_file.read_text():
            violations.append(str(py_file.relative_to(REPO_ROOT)))
    assert not violations, f"playwright still referenced in: {violations}"


# ---------------------------------------------------------------------------
# Part 2 — lag_adjuster removed from the normalizer chain
# ---------------------------------------------------------------------------

def test_feature_builder_no_lag_adjuster_import():
    """normalizer/feature_builder.py must not import lag_adjuster or apply_lag —
    red if someone re-wires lag adjustment back into the chain."""
    fb_path = REPO_ROOT / "normalizer" / "feature_builder.py"
    tree = ast.parse(fb_path.read_text(), filename=str(fb_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "lag_adjuster" and (node.module or "") != ".lag_adjuster", (
                f"feature_builder.py imports from lag_adjuster at line {node.lineno}"
            )
            for alias in node.names:
                assert alias.name != "apply_lag", (
                    f"feature_builder.py imports apply_lag at line {node.lineno}"
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "lag_adjuster" not in alias.name, (
                    f"feature_builder.py imports lag_adjuster at line {node.lineno}"
                )

    assert not (REPO_ROOT / "normalizer" / "lag_adjuster.py").exists(), (
        "normalizer/lag_adjuster.py still exists on disk"
    )


def test_features_written_with_lag_adjusted_false():
    """Every feature row built by feature_builder._add_feature must carry
    lag_adjusted = False (a literal boolean), never None/NULL. Runs against
    a small synthetic frame — no database required."""
    import pandas as pd
    from normalizer import feature_builder

    bag: list[dict] = []
    daily = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(
            ["2026-01-01", "2026-01-02", "2026-01-03"], name="date", tz="UTC"
        ),
    )
    feature_builder._add_feature(bag, "TEST.feature", daily)

    assert bag, "no rows were produced by _add_feature on a non-empty frame"
    for row in bag:
        assert row["lag_adjusted"] is False, (
            f"expected lag_adjusted is False, got {row['lag_adjusted']!r} for {row}"
        )


# ---------------------------------------------------------------------------
# Part 3 — scheduler deployments
# ---------------------------------------------------------------------------

_EXPECTED_DEPLOYMENTS = {
    "port-call-refresh",
    "bdi-daily",
    "wci-weekly",
    "normalizer-nightly",
    "targets-nightly",
    "signals-nightly",
    "models-nightly",
    "alerts-nightly",
}


def test_deployment_names_exact():
    """The set of deployment names scheduler._build_deployments() returns
    must equal exactly the 8 surviving flows — not just the same count,
    so swapping one flow for another is also caught."""
    import scheduler

    names = {d.name for d in scheduler._build_deployments()}
    assert names == _EXPECTED_DEPLOYMENTS, (
        f"deployment set drifted — got {names!r}, expected {_EXPECTED_DEPLOYMENTS!r}"
    )


# ---------------------------------------------------------------------------
# Part 4 — dependencies
# ---------------------------------------------------------------------------

def test_requirements_excludes_playwright():
    """requirements.txt must not list playwright — a future
    `pip freeze > requirements.txt` would silently re-add it otherwise."""
    text = (REPO_ROOT / "requirements.txt").read_text().lower()
    assert "playwright" not in text, "playwright is still listed in requirements.txt"
