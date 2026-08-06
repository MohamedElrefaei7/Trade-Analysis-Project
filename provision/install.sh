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
# DEPLOY_DIR defaults to this repo's own checkout location: the expected
# operational flow is "clone this repo onto the instance (ideally as the
# deploy user, or somewhere deploy can read), then run
# `sudo provision/install.sh` from inside it." That checkout IS where
# docker-compose.yml lives, so trade-signals.service's WorkingDirectory
# points straight at it — no separate deploy directory or copy step.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_USER="${DEPLOY_USER:-deploy}"
DEPLOY_DIR="${DEPLOY_DIR:-$REPO_ROOT}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/trade-signals-data}"

echo "== 00-harden =="
bash "$SCRIPT_DIR/00-harden.sh"

echo "== 01-docker =="
DEPLOY_USER="$DEPLOY_USER" bash "$SCRIPT_DIR/01-docker.sh"

echo "== 02-mount-data-volume =="
DEPLOY_USER="$DEPLOY_USER" MOUNT_POINT="$MOUNT_POINT" DATA_VOLUME_ID="$DATA_VOLUME_ID" bash "$SCRIPT_DIR/02-mount-data-volume.sh"

echo "== 03-docker-user-chain =="
bash "$SCRIPT_DIR/03-docker-user-chain.sh"

echo "== installing trade-signals.service (DEPLOY_DIR=$DEPLOY_DIR, DEPLOY_USER=$DEPLOY_USER) =="
sed -e "s#__DEPLOY_DIR__#${DEPLOY_DIR}#g" \
    -e "s#__DEPLOY_USER__#${DEPLOY_USER}#g" \
    "$SCRIPT_DIR/trade-signals.service" > /etc/systemd/system/trade-signals.service
systemctl daemon-reload
systemctl enable trade-signals >/dev/null

cat <<EOF

install.sh complete. Verify:
  ufw status verbose                             # Default: deny (incoming); ALLOW 80/tcp, 443/tcp only
  su - $DEPLOY_USER -c "docker compose version"  # no sudo needed
  systemctl is-enabled trade-signals             # enabled
  findmnt $MOUNT_POINT                           # data volume mounted
  iptables -L DOCKER-USER -n                     # default-deny past 80/443/established

Full checks: tests/test_provision.sh
EOF
