#!/usr/bin/env bash
# provision/00-harden.sh
#
# Baseline hardening for a fresh Ubuntu 24.04 trade-signals instance: ufw,
# fail2ban, unattended-upgrades, and a swapfile. Idempotent — every step
# checks current state before acting, so reruns only do the work that's
# still outstanding.
#
# ufw here governs 22/tcp, 80/tcp, and 443/tcp. The security group
# (infra/terraform/network.tf, scoped to var.admin_cidr) and ufw are both
# gates in series, not redundant alternatives to each other — traffic has
# to clear both to reach the instance. ufw's default-deny-incoming applies
# independently of the security group, so with no ufw rule for 22, SSH is
# unreachable at the OS level the moment the firewall activates, no matter
# what the security group permits. (Discovered live: a version of this
# script without the rule below locked out SSH the instant it ran,
# recovered only via SSM Session Manager. See CONTEXT.md.)
#
# The SSH rule is scoped to $ADMIN_CIDR, a required env var with no
# default (same pattern as DATA_VOLUME_ID in provision/install.sh) —
# sourced from the same Terraform admin_cidr value the security group
# rule already uses, never hardcoded here, so the two can't silently
# drift apart. See CLAUDE.md §8.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "run as root (sudo provision/00-harden.sh)" >&2
  exit 1
fi

: "${ADMIN_CIDR:?ADMIN_CIDR is required — see provision/README.md}"

echo "[00-harden] installing ufw, fail2ban, unattended-upgrades..."
apt-get update -qq
apt-get install -y -qq ufw fail2ban unattended-upgrades

echo "[00-harden] configuring ufw (22/tcp scoped to \$ADMIN_CIDR, 80/tcp, 443/tcp)..."

# IPv6 stays enabled in ufw. An earlier version of this script disabled it
# here so `ufw allow 80/tcp` wouldn't also produce a "(v6)" ALLOW line,
# purely to make a rule-count assertion in tests/test_provision.sh come
# out to two instead of four. That was fixing the environment to make an
# easy count pass rather than fixing the check to verify the actual
# invariant (80/443 only, nothing else) — and it cost real capability:
# Phase 10 needs IPv6 working at the OS level for Let's Encrypt and any
# future AAAA record. The test now filters to v4 before counting and
# separately checks v4/v6 parity, so ufw allowing both address families
# is expected and correct.
if [[ -f /etc/default/ufw ]] && grep -q '^IPV6=no' /etc/default/ufw; then
  sed -i 's/^IPV6=no/IPV6=yes/' /etc/default/ufw
fi

ufw default deny incoming
ufw default allow outgoing

# Scoped to ADMIN_CIDR, never `ufw allow 22/tcp` bare — an unscoped rule
# would open SSH to the world regardless of what the security group
# restricts, defeating the reason admin_cidr exists. Checked against the
# CIDR's host part specifically (not just "any 22/tcp ALLOW rule exists")
# so a rerun after ADMIN_CIDR changes adds the new rule instead of no-op'ing
# on a stale one scoped to the old value.
ADMIN_HOST="${ADMIN_CIDR%%/*}"
if ! ufw status | grep -E '^22/tcp[[:space:]]+ALLOW' | grep -v '(v6)' | grep -qF "$ADMIN_HOST"; then
  ufw allow from "$ADMIN_CIDR" to any port 22 proto tcp
fi
if ! ufw status | grep -qE '^80/tcp[[:space:]]+ALLOW'; then
  ufw allow 80/tcp
fi
if ! ufw status | grep -qE '^443/tcp[[:space:]]+ALLOW'; then
  ufw allow 443/tcp
fi

if ufw status | grep -q "Status: active"; then
  ufw reload
else
  ufw --force enable
fi

echo "[00-harden] enabling fail2ban..."
systemctl enable --now fail2ban >/dev/null

echo "[00-harden] enabling unattended-upgrades..."
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF
systemctl enable --now unattended-upgrades >/dev/null

echo "[00-harden] swapfile..."
if ! swapon --show | grep -q '/swapfile'; then
  if [[ ! -f /swapfile ]]; then
    fallocate -l 2G /swapfile
  fi
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
fi
if ! grep -q '^/swapfile ' /etc/fstab; then
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

install -d /etc/sysctl.d
cat > /etc/sysctl.d/60-swappiness.conf <<'EOF'
# Low swappiness (Ubuntu default is 60): this box runs Postgres, and
# aggressively swapping a database process is worse than running slightly
# closer to the memory ceiling. The swapfile is a safety margin for
# transient spikes (an ElasticNet training run overlapping a heavy ingest
# moment), not a substitute for the t4g.medium sizing decision already
# made in Terraform.
vm.swappiness=10
EOF
sysctl -p /etc/sysctl.d/60-swappiness.conf >/dev/null

echo "[00-harden] done."
