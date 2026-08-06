#!/usr/bin/env bash
# provision/01-docker.sh
#
# Installs Docker Engine + the Compose plugin from Docker's official apt
# repo, and puts the deploy user in the docker group. Idempotent — skips
# the install entirely if `docker` is already on PATH.
#
# deploy is added to the docker group, NOT given passwordless sudo. Docker
# group membership is already root-equivalent (a known, accepted Docker
# footgun), so withholding sudo adds no additional safety — but it keeps
# the audit trail honest: actions taken as deploy show up as Docker
# operations, not arbitrary root commands, which matters if you're ever
# debugging what a compromised or misbehaving container did.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "run as root (sudo provision/01-docker.sh)" >&2
  exit 1
fi

DEPLOY_USER="${DEPLOY_USER:-deploy}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[01-docker] installing Docker Engine + Compose plugin..."
  apt-get update -qq
  apt-get install -y -qq ca-certificates curl gnupg

  install -m 0755 -d /etc/apt/keyrings
  if [[ ! -f /etc/apt/keyrings/docker.asc ]]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
  fi

  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ ! -f /etc/apt/sources.list.d/docker.list ]]; then
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
      > /etc/apt/sources.list.d/docker.list
  fi

  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
else
  echo "[01-docker] Docker already installed, skipping install."
fi

systemctl enable --now docker >/dev/null

if ! id -u "$DEPLOY_USER" >/dev/null 2>&1; then
  echo "[01-docker] creating $DEPLOY_USER user..."
  useradd -m -s /bin/bash "$DEPLOY_USER"
fi

if ! id -nG "$DEPLOY_USER" | grep -qw docker; then
  echo "[01-docker] adding $DEPLOY_USER to docker group..."
  usermod -aG docker "$DEPLOY_USER"
fi

echo "[01-docker] done."
