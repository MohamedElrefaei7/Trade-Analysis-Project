#!/usr/bin/env bash
# tests/test_provision.sh
#
# Bash test harness for provision/ — run on the instance itself, as root
# (sudo tests/test_provision.sh), after provision/install.sh. Asserts
# against live system state (ufw, iptables, systemd, mounts), which is
# exactly why this is a bash script an operator runs on the box rather
# than something CI can run.
#
# Run this again after `sudo reboot` too: a mount or firewall rule that
# only holds until the next reboot is not verified, only asserted. See
# provision/README.md's "Reboot verification" section.
set -uo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "run as root (sudo tests/test_provision.sh)" >&2
  exit 1
fi

# Required (no default): test_data_volume_not_reformatted_on_rerun reruns
# install.sh, which itself hard-requires this — see provision/README.md.
: "${DATA_VOLUME_ID:?DATA_VOLUME_ID is required — see provision/README.md}"

# Required (no default): test_ufw_allows_ssh_from_admin_cidr checks the ufw
# rule against this value, and test_data_volume_not_reformatted_on_rerun's
# install.sh rerun hard-requires it too — see provision/README.md.
: "${ADMIN_CIDR:?ADMIN_CIDR is required — see provision/README.md}"

MOUNT_POINT="${MOUNT_POINT:-/mnt/trade-signals-data}"
DEPLOY_USER="${DEPLOY_USER:-deploy}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PASS=0
FAIL=0

run_test() {
  local name="$1"
  if "$name"; then
    echo "PASS: $name"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $name"
    FAIL=$((FAIL + 1))
  fi
}

# ufw status verbose includes "Default: deny (incoming)". Fails if a
# future edit switches the default policy while adding rules piecemeal —
# a common way ufw configs drift open.
test_ufw_default_deny_incoming() {
  ufw status verbose | grep -q "Default: deny (incoming)"
}

# The actual invariant is "ufw allows only 22 (SSH, scoped to ADMIN_CIDR),
# 80, and 443, nothing wider" — not "ufw has exactly N ALLOW lines," which
# is merely a proxy that happens to equal a fixed count only when IPv6 is
# disabled. IPv6 stays enabled (see 00-harden.sh), so ufw is expected to
# list v6 mirrors of 80/443 — but NOT of 22, since ADMIN_CIDR (from
# Terraform's admin_cidr) is IPv4-only and no v6 SSH rule is ever added.
# This filters to v4-only ALLOW lines and asserts their ports are exactly
# {22, 80, 443}, then separately asserts the v6 ALLOW lines are exactly
# {80, 443} — that second half is what actually catches port 22 (or
# anything else) sneaking in via the IPv6 side specifically, which a
# v4-only filter alone would be blind to. This test only checks *which
# ports* are open, not *who* they're scoped to — the "not Anywhere" scoping
# check for port 22 lives in test_ufw_allows_ssh_from_admin_cidr below.
test_ufw_v4_allow_rules_are_22_80_and_443() {
  local v4_ports v6_ports
  v4_ports="$(ufw status | grep 'ALLOW' | grep -v '(v6)' | awk '{print $1}' | sed 's#/tcp##' | sort -n | tr '\n' ' ')"
  v6_ports="$(ufw status | grep 'ALLOW' | grep '(v6)' | awk '{print $1}' | sed 's#/tcp##' | sort -n | tr '\n' ' ')"
  [[ "$v4_ports" == "22 80 443 " ]] || return 1
  [[ "$v6_ports" == "80 443 " ]] || return 1
}

# ufw's default-deny-incoming applies in series with the security group,
# not as a redundant alternative to it — with no ufw rule for 22, SSH is
# unreachable at the OS level the instant the firewall activates,
# regardless of what the security group permits. This asserts a 22/tcp
# ALLOW rule exists scoped to $ADMIN_CIDR specifically (not "Anywhere"),
# since a wide-open ufw rule would defeat the whole reason the security
# group's admin_cidr scoping exists. See CLAUDE.md §8.
test_ufw_allows_ssh_from_admin_cidr() {
  local admin_host="${ADMIN_CIDR%%/*}"
  local v4_ssh_allow
  v4_ssh_allow="$(ufw status | grep -E '^22/tcp[[:space:]]+ALLOW' | grep -v '(v6)')"
  [[ -n "$v4_ssh_allow" ]] || return 1
  echo "$v4_ssh_allow" | grep -qF "$admin_host" || return 1
  ! echo "$v4_ssh_allow" | grep -q 'Anywhere'
}

# iptables -L DOCKER-USER -n shows a terminal DROP/REJECT rule for
# traffic not matching 80/443/established. This is the automated
# structural check only — it confirms the chain is built correctly, but
# NOT that it's actually unreachable from outside: intra-host traffic
# doesn't cross this chain the same way a real external connection does.
# For the case that must fail if a container publishes an unintended
# port, run this manually from a SECOND machine (never localhost):
#   docker run -d -p 5432:5432 --name test-exposure alpine sleep 30
#   nc -zv -w3 <instance-public-ip> 5432     # expect: connection refused
#   docker rm -f test-exposure
test_docker_user_chain_default_deny() {
  iptables -L DOCKER-USER -n 2>/dev/null | tail -1 | grep -qE '^(DROP|REJECT)'
}

# findmnt shows the mount active, and /etc/fstab's entry for that mount
# point uses UUID=, not a device path.
test_data_volume_mounted_by_uuid() {
  findmnt "$MOUNT_POINT" >/dev/null 2>&1 || return 1
  grep -E "[[:space:]]${MOUNT_POINT}[[:space:]]" /etc/fstab | grep -q '^UUID='
}

# Write a marker file to the mounted volume, rerun install.sh, confirm
# the marker survives. Must fail if the mkfs-only-when-no-filesystem
# guard in 02-mount-data-volume.sh is missing.
test_data_volume_not_reformatted_on_rerun() {
  local marker="$MOUNT_POINT/.provision_test_marker"
  local content="test-$(date +%s)"
  echo "$content" > "$marker" || return 1
  DATA_VOLUME_ID="$DATA_VOLUME_ID" bash "$REPO_ROOT/provision/install.sh" >/dev/null 2>&1 || return 1
  [[ -f "$marker" ]] && grep -q "$content" "$marker"
}

# su - deploy -c "docker ps" succeeds, exit 0, no sudo.
test_deploy_user_runs_docker_without_sudo() {
  su - "$DEPLOY_USER" -c "docker ps" >/dev/null 2>&1
}

# sysctl vm.swappiness returns 10, not the Ubuntu default of 60.
test_swappiness_is_low() {
  [[ "$(sysctl -n vm.swappiness)" -eq 10 ]]
}

# systemctl is-enabled trade-signals returns enabled; start it, confirm
# active; stop it again to leave the box as found.
test_systemd_unit_enabled_and_starts_clean() {
  [[ "$(systemctl is-enabled trade-signals 2>/dev/null)" == "enabled" ]] || return 1
  systemctl start trade-signals || return 1
  local active
  active="$(systemctl is-active trade-signals)"
  systemctl stop trade-signals
  [[ "$active" == "active" ]]
}

echo "=== provision test harness ==="
echo

for t in \
  test_ufw_default_deny_incoming \
  test_ufw_v4_allow_rules_are_22_80_and_443 \
  test_ufw_allows_ssh_from_admin_cidr \
  test_docker_user_chain_default_deny \
  test_data_volume_mounted_by_uuid \
  test_data_volume_not_reformatted_on_rerun \
  test_deploy_user_runs_docker_without_sudo \
  test_swappiness_is_low \
  test_systemd_unit_enabled_and_starts_clean \
; do
  run_test "$t"
done

echo
echo "$PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
