#!/usr/bin/env bash
# provision/03-docker-user-chain.sh
#
# ufw does not gate anything Docker publishes. Docker manipulates iptables
# directly via the DOCKER-USER chain, ahead of ufw's own rules, so a
# container that publishes a port is reachable regardless of what
# `ufw status` reports. This installs an explicit default-deny in
# DOCKER-USER for everything except 80/443/established-related, and a
# systemd unit that reinstalls it on every boot — iptables rules don't
# persist across reboots on their own without iptables-persistent or an
# equivalent, and this script is exactly what that boot-time unit runs.
#
# `ufw status` is NOT the source of truth for what's reachable through
# Docker; `iptables -L DOCKER-USER -n` is. Trusting the former was flagged
# explicitly as the failure this project's law already caught once, in
# application form — a layer reporting a state that the thing downstream
# doesn't actually honor.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "run as root (sudo provision/03-docker-user-chain.sh)" >&2
  exit 1
fi

CHAIN_SCRIPT=/usr/local/sbin/trade-signals-docker-user-chain.sh
SERVICE_FILE=/etc/systemd/system/trade-signals-docker-user-chain.service

install -d /usr/local/sbin
cat > "$CHAIN_SCRIPT" <<'EOF'
#!/usr/bin/env bash
# Rebuilt from scratch on every run so the chain's end state is
# deterministic regardless of prior contents — see
# provision/03-docker-user-chain.sh for the reasoning. This runs on every
# boot via trade-signals-docker-user-chain.service.
set -euo pipefail

iptables -N DOCKER-USER 2>/dev/null || true
iptables -F DOCKER-USER

iptables -A DOCKER-USER -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A DOCKER-USER -p tcp --dport 80 -j ACCEPT
iptables -A DOCKER-USER -p tcp --dport 443 -j ACCEPT
iptables -A DOCKER-USER -j DROP
EOF
chmod 755 "$CHAIN_SCRIPT"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Reinstall the DOCKER-USER default-deny iptables rules
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$CHAIN_SCRIPT

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable trade-signals-docker-user-chain >/dev/null
# restart (not start): RemainAfterExit=yes means a prior run already looks
# "active" to systemd, but a rerun of this script must actually reinstall
# the chain in case something (e.g. a Docker restart) reset it.
systemctl restart trade-signals-docker-user-chain

echo "[03-docker-user-chain] DOCKER-USER default-deny installed."
