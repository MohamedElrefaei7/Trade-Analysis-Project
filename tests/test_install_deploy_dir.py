"""
test_install_deploy_dir.py — enforcement tests for provision/install.sh's
DEPLOY_DIR resolution and its .git-checkout refusal guard.

Runs install.sh with INSTALL_SH_RESOLVE_ONLY=1, a test-only hook that
prints the resolved DEPLOY_DIR and exits before the root check or any
side effect (rsync, systemd, apt) — so this needs no root, no
DATA_VOLUME_ID/ADMIN_CIDR, and touches nothing on the real system.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "provision" / "install.sh"


def _run(env_overrides: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "INSTALL_SH_RESOLVE_ONLY": "1"}
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(INSTALL_SH)],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_deploy_dir_defaults_to_opt_trade_signals(tmp_path):
    """No DEPLOY_DIR set, invoked from a directory that is not the deploy
    target: must resolve to the fixed /opt/trade-signals, not wherever the
    checkout happens to live. Goes red against the old
    DEPLOY_DIR="${DEPLOY_DIR:-$REPO_ROOT}" default, which would print
    REPO_ROOT (this repo's own checkout path) instead."""
    result = _run({}, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/opt/trade-signals"
    assert result.stdout.strip() != str(REPO_ROOT)


def test_deploy_dir_explicit_override_still_honored(tmp_path):
    """An operator-supplied DEPLOY_DIR (pointing somewhere with no .git)
    is respected as-is — the fixed default must not become a hardcoded
    value that can no longer be overridden."""
    target = tmp_path / "custom-deploy"
    result = _run({"DEPLOY_DIR": str(target)}, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(target)


def test_refuses_when_deploy_dir_is_a_git_checkout(tmp_path):
    """DEPLOY_DIR resolving to a path containing a .git directory must be
    refused outright — rsync -a --delete into a repository checkout would
    destroy it, so this must be impossible, not merely unlikely. Using
    this repo's own REPO_ROOT (which has a real .git/) as DEPLOY_DIR is
    exactly the dangerous case the guard exists for."""
    result = _run({"DEPLOY_DIR": str(REPO_ROOT)}, cwd=tmp_path)
    assert result.returncode != 0
    assert ".git" in result.stderr
    assert result.stdout.strip() == ""


def test_refuses_when_deploy_dir_has_nested_git_dir(tmp_path):
    """Same guard, constructed fresh rather than reusing the repo checkout
    — a plain directory that happens to contain a .git subdirectory must
    also be refused."""
    target = tmp_path / "looks-like-a-checkout"
    (target / ".git").mkdir(parents=True)
    result = _run({"DEPLOY_DIR": str(target)}, cwd=tmp_path)
    assert result.returncode != 0
    assert result.stdout.strip() == ""
