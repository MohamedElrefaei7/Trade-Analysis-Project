#!/usr/bin/env bash
# provision/02-mount-data-volume.sh
#
# Formats (if needed) and mounts the persistent data EBS volume
# (infra/terraform/storage.tf's aws_ebs_volume.data, requested at
# /dev/sdf) at a stable path, referenced in /etc/fstab by filesystem
# UUID — never by device path.
#
# Nitro device enumeration is not guaranteed stable across reboots or
# instance-type changes — the AWS-requested name (/dev/sdf) already
# didn't survive to become the OS-visible name (it shows up as an NVMe
# device instead). blkid after mkfs gives the UUID; that's what goes in
# fstab, with `nofail` so a missing volume doesn't hang boot. The tempting
# wrong version hardcodes /dev/nvme1n1 because that's what lsblk shows
# right now — it works until the next t4g.medium resize or instance
# replacement changes the enumeration, then fails silently at boot with
# the mount simply absent.
#
# The device is identified by matching DATA_VOLUME_ID (required, no
# fallback — see below) against the NVMe controller's serial number.
# Nitro exposes the actual EBS volume ID to the guest that way, readable
# straight from sysfs with no AWS API call or IAM permissions needed —
# `cat /sys/class/nvme/nvme1/serial` returns the volume ID with the
# hyphen stripped. That's checking the resource's real identity, not its
# position in `lsblk` — immune to how many other disks happen to be
# attached, which matters most in exactly the situations (a snapshot
# restore in progress, an extra volume attached for some unrelated
# reason) where guessing from disk count is most likely to guess wrong.
# An earlier version of this script identified the data disk by
# elimination — "the one disk that isn't root" — and fell back to asking
# the operator for a device path when that was ambiguous. That fallback
# path was never exercised except during the exact anomalous conditions
# where a wrong guess does the most damage, which makes it worse than no
# fallback: an untested code path biased toward firing when caution
# matters most. Matching by volume ID removes the guess entirely instead
# of demoting it to a rarer case.
#
# mkfs runs ONLY if the device has no existing filesystem (blkid TYPE
# empty). This volume is meant to outlive instance replacements — a rerun
# of this script (or install.sh) against a volume that already holds data
# must never reformat it. Same class of danger as schema.sql: an
# idempotent-looking script that is destructive on the one occasion it
# matters.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "run as root (sudo provision/02-mount-data-volume.sh)" >&2
  exit 1
fi

MOUNT_POINT="${MOUNT_POINT:-/mnt/trade-signals-data}"
DEPLOY_USER="${DEPLOY_USER:-deploy}"

# Hard-required, no default and no topology-based fallback. Get it with:
#   terraform -chdir=infra/terraform output -raw data_volume_id
: "${DATA_VOLUME_ID:?DATA_VOLUME_ID is required — set it to the aws_ebs_volume.data ID (e.g. vol-0123456789abcdef0) from 'terraform -chdir=infra/terraform output -raw data_volume_id'. See provision/README.md.}"

WANT_SERIAL="${DATA_VOLUME_ID//-/}"

MATCHES=()
for ctrl in /sys/class/nvme/nvme*; do
  [[ -e "$ctrl/serial" ]] || continue
  serial="$(tr -d '[:space:]' < "$ctrl/serial")"
  if [[ "$serial" == "$WANT_SERIAL" ]]; then
    MATCHES+=("$(basename "$ctrl")")
  fi
done

if [[ "${#MATCHES[@]}" -eq 0 ]]; then
  echo "[02-mount-data-volume] no NVMe controller with serial ${WANT_SERIAL} found (DATA_VOLUME_ID=${DATA_VOLUME_ID}) — is the volume attached?" >&2
  exit 1
elif [[ "${#MATCHES[@]}" -gt 1 ]]; then
  echo "[02-mount-data-volume] multiple NVMe controllers matched serial ${WANT_SERIAL} (${MATCHES[*]}) — refusing to guess." >&2
  exit 1
fi

DATA_DEV="/dev/${MATCHES[0]}n1"
if [[ ! -b "$DATA_DEV" ]]; then
  echo "[02-mount-data-volume] expected data device $DATA_DEV does not exist." >&2
  exit 1
fi

echo "[02-mount-data-volume] data device: $DATA_DEV (matched DATA_VOLUME_ID=${DATA_VOLUME_ID})"

EXISTING_FS="$(blkid -o value -s TYPE "$DATA_DEV" 2>/dev/null || true)"
if [[ -z "$EXISTING_FS" ]]; then
  echo "[02-mount-data-volume] no filesystem found on $DATA_DEV — formatting ext4."
  mkfs.ext4 -q "$DATA_DEV"
else
  echo "[02-mount-data-volume] existing filesystem ($EXISTING_FS) found on $DATA_DEV — not reformatting."
fi

DATA_UUID="$(blkid -o value -s UUID "$DATA_DEV")"

mkdir -p "$MOUNT_POINT"

if ! grep -q "^UUID=${DATA_UUID} " /etc/fstab; then
  echo "UUID=${DATA_UUID} ${MOUNT_POINT} ext4 defaults,nofail 0 2" >> /etc/fstab
fi

if ! mountpoint -q "$MOUNT_POINT"; then
  mount "$MOUNT_POINT"
fi

if id -u "$DEPLOY_USER" >/dev/null 2>&1; then
  chown "$DEPLOY_USER":"$DEPLOY_USER" "$MOUNT_POINT"
fi

echo "[02-mount-data-volume] mounted at $MOUNT_POINT (UUID=$DATA_UUID)."
