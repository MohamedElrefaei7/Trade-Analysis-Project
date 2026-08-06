# provision/ — instance hardening and Docker install

Idempotent shell scripts that turn a fresh Ubuntu 24.04 instance (as
provisioned by `infra/terraform/`) into a hardened box with Docker, a
non-root deploy user, the persistent data volume mounted, and a systemd
unit stub ready for the real `docker-compose.yml` service definitions
that later commits add. Not `user_data` — see `infra/terraform/README.md`
and CLAUDE.md §8 for why: provisioning failures need to be visible and
rerunnable, not silent and one-shot.

## Prerequisites

- A running instance from `infra/terraform/` (Ubuntu 24.04 arm64,
  `t4g.medium`, with `aws_ebs_volume.data` attached).
- This repo checked out on the instance — ideally as (or readable by) the
  `deploy` user, e.g.:
  ```sh
  sudo -u deploy git clone <repo-url> /home/deploy/trade-signals
  ```
  (If `deploy` doesn't exist yet, clone anywhere `root` can read and rerun
  after `01-docker.sh` creates the user — `install.sh` doesn't move or
  chown the checkout, so it needs to already be somewhere `deploy` can
  read `docker-compose.yml` from before `trade-signals.service` will
  start cleanly.)
- SSH access via the keypair from `infra/terraform` (`admin_cidr`).
- `DATA_VOLUME_ID` — the `aws_ebs_volume.data` ID, required with no
  default. Get it with:
  ```sh
  terraform -chdir=infra/terraform output -raw data_volume_id
  ```
  This stays an operator-supplied value — `install.sh` never reaches into
  AWS credentials or calls the AWS API to look it up itself.

## What to run, in what order

```sh
cd trade-signals   # this checkout
export DATA_VOLUME_ID=vol-0123456789abcdef0   # from terraform output, see above
sudo -E provision/install.sh
```

(`sudo -E` — or otherwise arrange for `DATA_VOLUME_ID` to survive the
`sudo`, e.g. `sudo DATA_VOLUME_ID="$DATA_VOLUME_ID" provision/install.sh`.
`install.sh` fails immediately, before touching anything, if the variable
isn't set.)

`install.sh` runs, in order:

1. `00-harden.sh` — ufw (80/tcp + 443/tcp only, see below), fail2ban,
   unattended-upgrades, a 2 GB swapfile at `vm.swappiness=10`.
2. `01-docker.sh` — Docker Engine + Compose plugin, `deploy` user created
   and added to the `docker` group (not given passwordless sudo — see
   CLAUDE.md §8).
3. `02-mount-data-volume.sh` — identifies the data volume by matching
   `DATA_VOLUME_ID` against the NVMe controller serial number (the real
   EBS volume ID, readable from sysfs — see the script's header for why
   this isn't inferred from disk position/count), formats it only if it
   has no filesystem yet, and mounts it at `/mnt/trade-signals-data`,
   referenced in `/etc/fstab` by UUID, never by device path.
4. `03-docker-user-chain.sh` — installs a default-deny rule in the
   `DOCKER-USER` iptables chain (everything except 80/443/established),
   reinstalled on every boot via a systemd unit.
5. Renders `trade-signals.service` (substituting the checkout path and
   deploy user) into `/etc/systemd/system/trade-signals.service`, then
   `daemon-reload` + `enable` (not `start` — starting is left to the
   operator or `tests/test_provision.sh`).

Each numbered script can also be run standalone (`sudo provision/01-docker.sh`,
etc.) — `install.sh` is a thin, ordered wrapper, not the only entry point.
Rerunning `install.sh` in full is safe: every step checks current state
before acting.

## Two independent firewall layers — read this before troubleshooting connectivity

- **ufw governs 80/tcp and 443/tcp only.** SSH access is entirely a
  Terraform/security-group concern (`infra/terraform/network.tf`, scoped
  to `admin_cidr`) — `00-harden.sh` never adds an SSH rule to ufw.
  `ufw status` should show exactly two ALLOW rules, ports 80 and 443,
  nothing else. If you find yourself wanting to `ufw allow 22`, don't —
  that's how a security group correctly scoped to one IP ends up paired
  with a ufw rule open to the world, silently, because the security group
  is still doing its job and nobody checks ufw too.
- **`DOCKER-USER` is the source of truth for what Docker exposes.** Docker
  manipulates iptables directly, ahead of ufw's own rules, so a container
  publishing a port (`docker run -p 5432:5432 ...`) is reachable
  regardless of what `ufw status` reports. `iptables -L DOCKER-USER -n`
  is what actually governs Docker-published ports — check that, not ufw,
  when debugging exposure.

## Done-condition checks

After `install.sh` completes:

```sh
ufw status                                       # exactly 2 ALLOW rules: 80/tcp, 443/tcp
su - deploy -c "docker compose version"          # succeeds, no sudo
systemctl is-enabled trade-signals               # enabled
findmnt /mnt/trade-signals-data                  # mounted
grep /mnt/trade-signals-data /etc/fstab          # entry uses UUID=, not a device path
iptables -L DOCKER-USER -n                       # terminal DROP past 80/443/established
```

Run the full suite with `sudo -E tests/test_provision.sh` (`DATA_VOLUME_ID`
must still be exported — the harness reruns `install.sh` as part of
`test_data_volume_not_reformatted_on_rerun`). See that file — it's a
harness run on the instance itself, not something CI can run, since it
asserts against live system state.

## Reboot verification

A mount or firewall rule that only holds until the next reboot is not
verified, only asserted. After `install.sh` completes and the checks
above pass:

```sh
sudo reboot
# wait — SSH refuses briefly while the instance comes back
sudo tests/test_provision.sh
```

Specifically confirm `findmnt /mnt/trade-signals-data` and
`iptables -L DOCKER-USER -n` still show the expected state post-reboot —
neither persists on its own without the fstab entry (mount) and the
`trade-signals-docker-user-chain.service` unit (iptables rules) installed
by this commit.

## Out of scope here

`docker-compose.yml`'s real service definitions, the APScheduler worker,
AIS ingestion, S3 backups, Caddy/TLS. The `04-` numbering is left open for
those. See CLAUDE.md's `§ Up Next` / `CONTEXT.md`.
