#!/usr/bin/env bash
# provision/install.sh
#
# Orchestrator: runs 00-harden.sh through 03-docker-user-chain.sh in
# order, then renders and installs the trade-signals systemd unit.
#
# Safe to rerun end-to-end. Every sub-script checks current state before
# acting (ufw rules via `ufw status | grep`, the swapfile via
# `swapon --show`, the deploy user via `id`, Docker via `docker --version`,
# the data volume via blkid), so rerunning after a partial failure
# completes the remaining steps instead of erroring out on — or, worse,
# destructively redoing — the ones already done.
#
# DEPLOY_DIR defaults to /opt/trade-signals, a fixed location independent
# of however this repo happened to get onto the instance (full `git
# clone`, hand-copied `provision/` scripts, whatever) — that variability
# is exactly what left `docker-compose.yml` undeployed on the instance
# used for Commit 2c's live verification, and made
# `test_systemd_unit_enabled_and_starts_clean` a standing documented
# exception instead of a real pass. This script copies the full build
# context — not just `docker-compose.yml` and `grafana/` — from REPO_ROOT
# into DEPLOY_DIR (see below). `worker` and `ais` both build with
# `build: .`, and Compose resolves that context relative to wherever
# docker-compose.yml lives, i.e. DEPLOY_DIR, not REPO_ROOT: copying only
# the compose file and grafana/ left the Dockerfile and the entire
# application source tree missing from that context, so `docker compose
# build`/`up -d --build` run from DEPLOY_DIR had nothing for `COPY . .`
# to copy from. `.dockerignore` already states exactly what the image
# doesn't need (`.git`, `venv/`, `tests/`, dumps, caches), so it doubles
# as the exclude list for this copy too.
set -euo pipefail

# Variable resolution happens before the root check and the hard-required
# env vars below, deliberately: it has no side effects (no writes, no
# subprocesses) and INSTALL_SH_RESOLVE_ONLY (further down) needs to reach
# it without requiring root or DATA_VOLUME_ID/ADMIN_CIDR, so
# tests/test_install_deploy_dir.py can exercise the default and the .git
# guard directly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_USER="${DEPLOY_USER:-deploy}"
DEPLOY_DIR="${DEPLOY_DIR:-/opt/trade-signals}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/trade-signals-data}"

# Refuse to run if DEPLOY_DIR resolves into a git checkout. rsync -a
# --delete below is destructive; an operator who overrides DEPLOY_DIR to
# (or forgets to override it away from) a repository checkout would sync
# --delete the checkout out of existence, not just deploy into it. A
# checkout is recognized by the presence of DEPLOY_DIR/.git — cheap,
# reliable, no ambiguity about what "is a checkout" means.
if [[ -d "$DEPLOY_DIR/.git" ]]; then
  echo "[install] refusing to run: DEPLOY_DIR=$DEPLOY_DIR contains a .git directory" >&2
  echo "[install] rsync -a --delete into a repository checkout would destroy it" >&2
  exit 1
fi

# Test hook only: prints the resolved DEPLOY_DIR and exits before the root
# check or any side effect, so the default and the guard above can be
# asserted without root, DATA_VOLUME_ID, or ADMIN_CIDR. Never set this in
# a real provisioning run.
if [[ "${INSTALL_SH_RESOLVE_ONLY:-0}" == "1" ]]; then
  echo "$DEPLOY_DIR"
  exit 0
fi

if [[ $EUID -ne 0 ]]; then
  echo "run as root (sudo provision/install.sh)" >&2
  exit 1
fi

# Hard-required, no default: the operator's own value from
#   terraform -chdir=infra/terraform output -raw data_volume_id
# Checked up front so a missing value halts before any step runs, rather
# than failing partway through 02-mount-data-volume.sh. See that script's
# header for why this replaced disk-position/count-based identification.
: "${DATA_VOLUME_ID:?DATA_VOLUME_ID is required — see provision/README.md}"

# Hard-required, no default: the same CIDR already used by Terraform's
#   var.admin_cidr (infra/terraform/network.tf's security group SSH rule)
# Checked up front, same as DATA_VOLUME_ID above, so a missing value halts
# before 00-harden.sh runs rather than completing "successfully" while
# leaving SSH unreachable — see that script's header for the incident this
# guards against.
: "${ADMIN_CIDR:?ADMIN_CIDR is required — see provision/README.md}"

echo "== 00-harden =="
ADMIN_CIDR="$ADMIN_CIDR" bash "$SCRIPT_DIR/00-harden.sh"

echo "== 01-docker =="
DEPLOY_USER="$DEPLOY_USER" bash "$SCRIPT_DIR/01-docker.sh"

echo "== 02-mount-data-volume =="
DEPLOY_USER="$DEPLOY_USER" MOUNT_POINT="$MOUNT_POINT" DATA_VOLUME_ID="$DATA_VOLUME_ID" bash "$SCRIPT_DIR/02-mount-data-volume.sh"

echo "== 03-docker-user-chain =="
bash "$SCRIPT_DIR/03-docker-user-chain.sh"

if ! command -v rsync >/dev/null 2>&1; then
  echo "[install] installing rsync..."
  apt-get update -qq
  apt-get install -y -qq rsync
fi

echo "== deploying build context to $DEPLOY_DIR =="
# Fixed deploy path, decoupled from REPO_ROOT (wherever this checkout
# happens to live) — trade-signals.service's WorkingDirectory, and
# docker-compose.yml's `build: .` context for `worker`/`ais`, both point
# here, not at the checkout. --exclude-from reuses .dockerignore (it's
# already the list of what the image doesn't need); .env/.env.* are
# excluded explicitly on top of that, since the operator's secrets file
# lives only in DEPLOY_DIR and must never be overwritten by, or deleted
# in favor of, whatever REPO_ROOT happens to have. --delete keeps
# DEPLOY_DIR from accumulating files a later commit removed from the
# repo — safe here specifically because rsync's default behavior is to
# never delete a path that the filter rules exclude, so the .env
# protection above doubles as delete-protection, not just copy-protection.
install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$DEPLOY_DIR"
rsync -a --delete \
  --exclude-from="$REPO_ROOT/.dockerignore" \
  --exclude ".env" --exclude ".env.*" \
  "$REPO_ROOT/" "$DEPLOY_DIR/"
chown -R "$DEPLOY_USER":"$DEPLOY_USER" "$DEPLOY_DIR"

# docker-compose.yml's timescale_data volume is a bind mount (driver_opts
# type: none, o: bind) pointed at this exact path — unlike a plain
# `volumes:` short-syntax bind, Docker's local driver does NOT create a
# missing device path for it, so `docker compose up -d` fails outright if
# it's absent. Owned by DEPLOY_USER, matching $MOUNT_POINT itself
# (chowned in 02-mount-data-volume.sh): the timescaledb container starts
# as root and its entrypoint chowns $PGDATA to its own postgres uid
# before dropping privileges, so host-side ownership only has to permit
# that initial chown, not match postgres's in-container uid exactly.
install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$MOUNT_POINT/timescale"

echo "== installing trade-signals.service (DEPLOY_DIR=$DEPLOY_DIR, DEPLOY_USER=$DEPLOY_USER) =="
sed -e "s#__DEPLOY_DIR__#${DEPLOY_DIR}#g" \
    -e "s#__DEPLOY_USER__#${DEPLOY_USER}#g" \
    "$SCRIPT_DIR/trade-signals.service" > /etc/systemd/system/trade-signals.service
systemctl daemon-reload
systemctl enable trade-signals >/dev/null

cat <<EOF

install.sh complete. Verify:
  ufw status verbose                             # Default: deny (incoming); ALLOW 22/tcp (from \$ADMIN_CIDR), 80/tcp, 443/tcp
  su - $DEPLOY_USER -c "docker compose version"  # no sudo needed
  systemctl is-enabled trade-signals             # enabled
  findmnt $MOUNT_POINT                           # data volume mounted
  iptables -L DOCKER-USER -n -v                  # inbound default-deny past 80/443/established, scoped to the external interface

Full checks: tests/test_provision.sh
EOF
