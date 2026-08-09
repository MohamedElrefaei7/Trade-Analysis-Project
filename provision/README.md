# provision/ — instance hardening and Docker install

Idempotent shell scripts that turn a fresh Ubuntu 24.04 instance (as
provisioned by `infra/terraform/`) into a hardened box with Docker, a
non-root deploy user, the persistent data volume mounted, and the
`docker-compose.yml` stack (TimescaleDB + Grafana) deployed at a fixed
path and running under systemd. Not `user_data` — see
`infra/terraform/README.md` and CLAUDE.md §8 for why: provisioning
failures need to be visible and rerunnable, not silent and one-shot.

## Prerequisites

- A running instance from `infra/terraform/` (Ubuntu 24.04 arm64,
  `t4g.medium`, with `aws_ebs_volume.data` attached).
- This repo checked out somewhere `root` can read — how doesn't matter
  (`git clone`, hand-copied files, whatever): `install.sh` copies
  `docker-compose.yml` and `grafana/` out of the checkout into
  `DEPLOY_DIR` (`/opt/trade-signals` by default) itself, so the checkout
  location isn't load-bearing after `install.sh` runs. See "Deploy
  directory and `.env`" below.
- SSH access via the keypair from `infra/terraform` (`admin_cidr`).
- `DATA_VOLUME_ID` — the `aws_ebs_volume.data` ID, required with no
  default. Get it with:
  ```sh
  terraform -chdir=infra/terraform output -raw data_volume_id
  ```
  This stays an operator-supplied value — `install.sh` never reaches into
  AWS credentials or calls the AWS API to look it up itself.
- `ADMIN_CIDR` — the same CIDR as Terraform's `var.admin_cidr`, required
  with no default. It isn't a Terraform output (it's an input), so read it
  straight from `infra/terraform/terraform.tfvars`:
  ```sh
  grep admin_cidr infra/terraform/terraform.tfvars
  ```
  `00-harden.sh` uses this to scope ufw's SSH rule — see "Two independent
  firewall layers" below for why this can't default to anything, and
  especially can't default to open.

## What to run, in what order

```sh
cd trade-signals   # this checkout
export DATA_VOLUME_ID=vol-0123456789abcdef0   # from terraform output, see above
export ADMIN_CIDR=203.0.113.4/32              # from terraform.tfvars, see above
sudo -E provision/install.sh
```

(`sudo -E` — or otherwise arrange for `DATA_VOLUME_ID` and `ADMIN_CIDR` to
survive the `sudo`, e.g.
`sudo DATA_VOLUME_ID="$DATA_VOLUME_ID" ADMIN_CIDR="$ADMIN_CIDR" provision/install.sh`.
`install.sh` fails immediately, before touching anything, if either
variable isn't set.)

`install.sh` runs, in order:

1. `00-harden.sh` — ufw (22/tcp scoped to `ADMIN_CIDR`, 80/tcp, 443/tcp;
   see below), fail2ban, unattended-upgrades, a 2 GB swapfile at
   `vm.swappiness=10`.
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
5. Copies `docker-compose.yml` and `grafana/` from the checkout into
   `DEPLOY_DIR` (`/opt/trade-signals` by default), owned by `DEPLOY_USER`,
   and creates `$MOUNT_POINT/timescale` (the bind-mount source for
   TimescaleDB's data directory), also owned by `DEPLOY_USER`. See
   "Deploy directory and `.env`" below.

   **As of Phase 3 Commit 5, this step does not yet copy what
   `worker`/`ais` need to build.** Those two services build from the
   repo's `Dockerfile`, which needs the full application source
   (`requirements.txt`, `clients/`, `orchestration/`, `worker/`, `ais/`,
   etc.) as its build context — `install.sh` only stages
   `docker-compose.yml` and `grafana/`. Until that gap is closed (a later
   provisioning commit), get a full checkout onto the server yourself
   before `docker compose build` — `git clone`/`git pull` at `DEPLOY_DIR`,
   or `rsync`, either is fine, same as any other checkout-location
   flexibility this script already tolerates. See "Building and starting
   the application services" below.
6. Renders `trade-signals.service` (substituting `DEPLOY_DIR` and
   `DEPLOY_USER`) into `/etc/systemd/system/trade-signals.service`, then
   `daemon-reload` + `enable` (not `start` — starting is left to the
   operator or `tests/test_provision.sh`).

Each numbered script can also be run standalone (`sudo provision/01-docker.sh`,
etc.) — `install.sh` is a thin, ordered wrapper, not the only entry point.
Rerunning `install.sh` in full is safe: every step checks current state
before acting, including step 5 — `grafana/` is removed and recopied
fresh from the checkout each run (it's config, not data, so there's
nothing to lose), and `docker-compose.yml` is simply overwritten.

## Deploy directory and `.env`

`DEPLOY_DIR` (`/opt/trade-signals` by default, override by exporting
`DEPLOY_DIR` before `install.sh`) is where `trade-signals.service`
actually runs `docker compose` from — not the git checkout. `install.sh`
populates it with `docker-compose.yml` and `grafana/`, but deliberately
**never** touches `.env`: it's gitignored, holds real secrets
(`POSTGRES_PASSWORD`, optionally `GRAFANA_PASSWORD`, and — as of Phase 3
Commit 5 — `AISSTREAM_API_KEY`), and has to be placed there by the
operator directly, e.g.:

```sh
sudo tee /opt/trade-signals/.env >/dev/null <<'EOF'
POSTGRES_PASSWORD=<real password>
GRAFANA_PASSWORD=<real password>
AISSTREAM_API_KEY=<real key>
EOF
sudo chown deploy:deploy /opt/trade-signals/.env
sudo chmod 600 /opt/trade-signals/.env
```

`docker compose` reads `.env` automatically from its working directory —
no separate flag or step needed once it's in place. Without
`POSTGRES_PASSWORD`, TimescaleDB refuses to initialize; without
`AISSTREAM_API_KEY`, the `ais` service fails to start — deliberately, see
below, a silently keyless AIS daemon is worse than one that refuses to
run. Do this before `systemctl start trade-signals` (or
`tests/test_provision.sh`, which starts the unit as part of
`test_systemd_unit_enabled_and_starts_clean`).

## Building and starting the application services

`docker-compose.yml` defines four services: `timescaledb`, `grafana`
(unchanged since Provision 4), and — as of Phase 3 Commit 5 — `worker`
(the APScheduler process, CLAUDE.md § 11) and `ais` (the AIS WebSocket
daemon, CLAUDE.md § 12), both built from the repo's `Dockerfile` and both
running under `restart: unless-stopped`. From `DEPLOY_DIR`, with a full
checkout in place (see the note on step 5 above) and `.env` populated as
above:

```sh
cd /opt/trade-signals
sudo -u deploy docker compose build
```

**Then apply any pending schema migrations — before, and separately
from, `docker compose up -d`:**

```sh
sudo -u deploy docker compose run --rm worker python -m orchestration.migrate
```

This is a deliberate, explicit operator step, not something either
service's container command does on startup. A migration is an act
against a database holding irreplaceable AIS history — coupling it to
container start means a restart loop becomes a migration-attempt loop.
`docker compose run --rm worker ...` reuses the `worker` service's image
and environment (so it gets `DATABASE_URL` for free) without leaving a
long-running container behind. See `migrations/README.md` for what the
runner itself guarantees (checksummed, never-edit-after-apply,
`schema.sql` refused outright).

Only then:

```sh
sudo -u deploy docker compose up -d
sudo -u deploy docker ps
```

`depends_on` in `docker-compose.yml` only waits for the `timescaledb`
*container* to start, not for Postgres to actually accept connections —
so on a cold boot, `worker` and `ais` will likely fail their first
connection attempt and get restarted by Docker. That's the supervision
model working as designed, not a fault to chase; both should settle into
a stable `Up` state within a few restarts. See CLAUDE.md § Deployment.

## Two firewall layers, in series — read this before troubleshooting connectivity

- **The security group and ufw are both gates traffic must clear, not
  redundant alternatives to each other.** It's tempting to think "the
  security group already restricts port 22, so ufw doesn't need to" —
  that's wrong. ufw's `default deny incoming` applies independently of
  the security group, so with no ufw rule for 22, SSH is unreachable at
  the OS level the moment ufw activates, regardless of what the security
  group permits. This was discovered live: an earlier version of
  `00-harden.sh` opened only 80/tcp and 443/tcp, and the first real
  `install.sh` run on a fresh instance locked out SSH immediately,
  recovered only via SSM Session Manager. See CONTEXT.md for the dated
  log entry.
- **ufw governs 22/tcp (scoped to `ADMIN_CIDR`), 80/tcp, and 443/tcp.**
  `00-harden.sh` requires `ADMIN_CIDR` — the same value as Terraform's
  `var.admin_cidr` — and adds a ufw rule for port 22 scoped to exactly
  that CIDR, never bare `ufw allow 22` (which would open SSH to the world
  regardless of what the security group restricts, defeating the reason
  `admin_cidr` exists in the first place). `ufw status` should show a
  22/tcp ALLOW rule scoped to your admin IP, plus 80 and 443, nothing
  wider.
- **`DOCKER-USER` is the source of truth for what Docker exposes.** Docker
  manipulates iptables directly, ahead of ufw's own rules, so a container
  publishing a port (`docker run -p 5432:5432 ...`) is reachable
  regardless of what `ufw status` reports. `iptables -L DOCKER-USER -n`
  is what actually governs Docker-published ports — check that, not ufw,
  when debugging exposure.

## Done-condition checks

After `install.sh` completes (and `.env` is placed per above):

```sh
ufw status                                       # 22/tcp ALLOW scoped to $ADMIN_CIDR, plus 80/tcp, 443/tcp
su - deploy -c "docker compose version"          # succeeds, no sudo
systemctl is-enabled trade-signals               # enabled
systemctl start trade-signals && systemctl is-active trade-signals   # active
findmnt /mnt/trade-signals-data                  # mounted
grep /mnt/trade-signals-data /etc/fstab          # entry uses UUID=, not a device path
iptables -L DOCKER-USER -n                       # terminal DROP past 80/443/established
ls /opt/trade-signals                            # docker-compose.yml, grafana/, .env
```

Run the full suite with `sudo -E tests/test_provision.sh` (`DATA_VOLUME_ID`
and `ADMIN_CIDR` must still be exported — the harness reruns `install.sh`
as part of `test_data_volume_not_reformatted_on_rerun`, and checks the
ufw SSH rule against `ADMIN_CIDR` directly in
`test_ufw_allows_ssh_from_admin_cidr`). See that file — it's a harness run
on the instance itself, not something CI can run, since it asserts
against live system state.

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

Having `install.sh` itself stage a full application checkout (so
`docker compose build` needs nothing beyond what step 5 already copies),
S3 backups, Caddy/TLS, and healthcheck-gated `depends_on` ordering for
`worker`/`ais` against `timescaledb`. The `04-` numbering is left open for
these. See CLAUDE.md's `§ Deployment` and `§ Up Next` / `CONTEXT.md`.
