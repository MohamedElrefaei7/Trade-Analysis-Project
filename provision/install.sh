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
# exception instead of a real pass. This script now copies
# `docker-compose.yml` and `grafana/` from REPO_ROOT into DEPLOY_DIR
# itself (see below), so `trade-signals.service`'s WorkingDirectory always
# has what it needs, however the checkout arrived.
set -euo pipefail

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_USER="${DEPLOY_USER:-deploy}"
DEPLOY_DIR="${DEPLOY_DIR:-/opt/trade-signals}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/trade-signals-data}"

echo "== 00-harden =="
ADMIN_CIDR="$ADMIN_CIDR" bash "$SCRIPT_DIR/00-harden.sh"

echo "== 01-docker =="
DEPLOY_USER="$DEPLOY_USER" bash "$SCRIPT_DIR/01-docker.sh"

echo "== 02-mount-data-volume =="
DEPLOY_USER="$DEPLOY_USER" MOUNT_POINT="$MOUNT_POINT" DATA_VOLUME_ID="$DATA_VOLUME_ID" bash "$SCRIPT_DIR/02-mount-data-volume.sh"

echo "== 03-docker-user-chain =="
bash "$SCRIPT_DIR/03-docker-user-chain.sh"

echo "== deploying docker-compose.yml to $DEPLOY_DIR =="
# Fixed deploy path, decoupled from REPO_ROOT (wherever this checkout
# happens to live) — trade-signals.service's WorkingDirectory always
# points here, not at the checkout. docker-compose.yml's grafana/
# volumes (./grafana/provisioning, ./grafana/dashboards) are relative to
# the compose file, so grafana/ has to travel with it or those bind
# mounts silently come up as empty directories instead of the real
# dashboards. .env is deliberately NOT copied here — it's gitignored,
# operator-supplied secrets (POSTGRES_PASSWORD, GRAFANA_PASSWORD), placed
# directly at $DEPLOY_DIR/.env by the operator. See provision/README.md.
install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$DEPLOY_DIR"
cp "$REPO_ROOT/docker-compose.yml" "$DEPLOY_DIR/docker-compose.yml"
# rm before cp -r: on rerun, cp -r into an already-existing grafana/ would
# nest a second copy inside it (grafana/grafana/...) instead of
# overwriting — this isn't the persistent data volume, so a clean
# rm+recopy from REPO_ROOT on every run is correct, not destructive.
rm -rf "$DEPLOY_DIR/grafana"
cp -r "$REPO_ROOT/grafana" "$DEPLOY_DIR/grafana"
chown -R "$DEPLOY_USER":"$DEPLOY_USER" "$DEPLOY_DIR/docker-compose.yml" "$DEPLOY_DIR/grafana"

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
  iptables -L DOCKER-USER -n                     # default-deny past 80/443/established

Full checks: tests/test_provision.sh
EOF
