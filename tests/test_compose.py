"""
test_compose.py — structural enforcement tests for docker-compose.yml and
.dockerignore, covering the design decisions in Phase 3 Commit 5 (see
CONTEXT.md's dated entry and CLAUDE.md's Deployment subsection). Parses
docker-compose.yml with yaml.safe_load — no `docker compose` binary, no
running daemon required — and should go red the moment any of these
decisions is silently reverted.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_PATH = REPO_ROOT / "docker-compose.yml"
DOCKERIGNORE_PATH = REPO_ROOT / ".dockerignore"

_SECRET_KEY_RE = re.compile(r"(PASSWORD|SECRET|API_KEY|TOKEN)$", re.IGNORECASE)
_VAR_REF_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*(:-[^}]*)?\}")
_LOOSE_TAG_RE = re.compile(r":(latest)$")
_BARE_MAJOR_TAG_RE = re.compile(r":v?\d+(\.\d+)?$")  # e.g. ":16" or ":2" or ":1.0" — not a full pin


def _load_compose() -> dict[str, Any]:
    with COMPOSE_PATH.open() as f:
        return yaml.safe_load(f)


def _services() -> dict[str, dict]:
    return _load_compose()["services"]


def test_all_services_restart_unless_stopped():
    """Every service — including timescaledb and grafana, unchanged by this
    commit — must set restart: unless-stopped, so a `sudo reboot` brings the
    whole stack back with no human action."""
    services = _services()
    assert services, "no services found in docker-compose.yml"
    for name, cfg in services.items():
        assert cfg.get("restart") == "unless-stopped", (
            f"service {name!r} must set 'restart: unless-stopped'"
        )


def test_no_service_publishes_postgres_or_worker_ports():
    """No 'ports' entry maps 5432 to a non-loopback address, and worker/ais
    publish no ports at all — nothing outside the Compose network talks to
    them directly."""
    services = _services()

    for name in ("worker", "ais"):
        assert name in services, f"expected a {name!r} service"
        assert not services[name].get("ports"), (
            f"service {name!r} must not publish any ports"
        )

    for name, cfg in services.items():
        for mapping in cfg.get("ports", []):
            mapping_str = str(mapping)
            if ":5432:" in mapping_str or mapping_str.endswith(":5432"):
                assert mapping_str.startswith("127.0.0.1:"), (
                    f"service {name!r} publishes 5432 without binding to "
                    f"127.0.0.1: {mapping_str!r}"
                )


def test_database_url_uses_service_name_not_loopback():
    """worker and ais must reach Postgres via the Compose service name
    'timescaledb', not 127.0.0.1 — inside the network, loopback is the
    container itself, not the timescaledb container. Goes red on the most
    likely copy-paste error."""
    services = _services()
    for name in ("worker", "ais"):
        env = services[name].get("environment", {})
        database_url = env.get("DATABASE_URL", "")
        assert "@timescaledb:" in database_url, (
            f"service {name!r} DATABASE_URL must target '@timescaledb:', got {database_url!r}"
        )
        assert "127.0.0.1" not in database_url, (
            f"service {name!r} DATABASE_URL must not use 127.0.0.1: {database_url!r}"
        )


def test_no_hardcoded_password_in_compose():
    """No environment value contains a literal secret — every PASSWORD/
    SECRET/API_KEY/TOKEN-suffixed variable, and the password segment of
    DATABASE_URL, must be a ${VAR} reference, not a literal string baked
    into docker-compose.yml."""
    services = _services()
    for name, cfg in services.items():
        env = cfg.get("environment", {})
        if isinstance(env, list):
            # list form ("KEY=value") normalized to a dict for uniform checking
            env = dict(item.split("=", 1) for item in env)
        for key, value in env.items():
            value = str(value)
            if _SECRET_KEY_RE.search(key):
                assert _VAR_REF_RE.fullmatch(value), (
                    f"service {name!r} env {key!r} must be a ${{VAR}} reference, "
                    f"not a literal value: {value!r}"
                )
            if key == "DATABASE_URL":
                # postgresql://user:PASSWORD@host:port/db — the password
                # segment (between the first ':' after '//' and the '@')
                # must itself be a ${VAR} reference.
                match = re.search(r"//[^:@/]*:([^@]*)@", value)
                assert match and _VAR_REF_RE.fullmatch(match.group(1)), (
                    f"service {name!r} DATABASE_URL password segment must be "
                    f"a ${{VAR}} reference: {value!r}"
                )


def test_all_services_have_log_rotation():
    """Every service sets a bounded json-file max-size — the AIS daemon
    under a dead provider logs every few seconds, and Docker's default
    json-file driver has no size limit of its own."""
    services = _services()
    for name, cfg in services.items():
        logging_cfg = cfg.get("logging")
        assert logging_cfg, f"service {name!r} must set a logging driver with rotation"
        assert logging_cfg.get("driver") == "json-file", (
            f"service {name!r} must use the json-file logging driver"
        )
        assert logging_cfg.get("options", {}).get("max-size"), (
            f"service {name!r} logging options must set max-size"
        )


def test_images_are_pinned():
    """No service uses a 'latest' or bare-major image tag — a moving base
    tag already cost a debugging session (timescale/timescaledb
    :latest-pg16 resolving to two different versions three months apart).
    Services that build from a local Dockerfile (worker, ais) are exempt —
    'image' pinning doesn't apply to them."""
    services = _services()
    for name, cfg in services.items():
        image = cfg.get("image")
        if image is None:
            assert "build" in cfg, f"service {name!r} has neither 'image' nor 'build'"
            continue
        assert not _LOOSE_TAG_RE.search(image), f"service {name!r} uses a floating 'latest' tag: {image!r}"
        assert ":" in image, f"service {name!r} image has no tag at all: {image!r}"
        tag = image.rsplit(":", 1)[1]
        assert not _BARE_MAJOR_TAG_RE.match(f":{tag}"), (
            f"service {name!r} uses a bare major/minor tag, not a full pin: {image!r}"
        )


def test_dockerignore_excludes_secrets_and_dumps():
    """.env, *.dump, and backups/ must never be shipped into an image layer
    — .env because a secret baked into a layer persists even after a later
    layer deletes it, *.dump/backups/ because they're multi-gigabyte and
    have no business in the build context."""
    assert DOCKERIGNORE_PATH.exists(), ".dockerignore must exist"
    lines = {
        line.strip() for line in DOCKERIGNORE_PATH.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert ".env" in lines, ".dockerignore must exclude .env"
    assert any(entry in ("*.dump",) for entry in lines), ".dockerignore must exclude *.dump"
    assert any(entry.rstrip("/") == "backups" for entry in lines), ".dockerignore must exclude backups/"
