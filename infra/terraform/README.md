# infra/terraform — trade-signals AWS footprint

Defines the EC2 instance, its persistent data volume, the Elastic IP, and
the security group for the trade-signals deployment. Nothing else: OS
provisioning (Docker, ufw, fail2ban, the deploy user, the systemd unit),
the data volume's filesystem/fstab entry, backups, and DNS all live in
later commits. This config never writes `user_data` — provisioning stays in
idempotent scripts whose failures are visible, not in a block that runs
once, silently, at first boot.

## State is local — back it up

There is no remote backend configured. `terraform.tfstate` lives on
whichever machine runs `terraform apply`, and it is the only record of what
Terraform believes exists in AWS. **Copy `terraform.tfstate` (and
`terraform.tfstate.backup`) somewhere durable — off this machine — after
every apply.** Losing it doesn't destroy the AWS resources, but it does mean
Terraform loses track of them: the next `plan` would propose creating
duplicates instead of recognizing what's already running, and the
`prevent_destroy`-guarded data volume would no longer have management state
tracking it.

Both files are gitignored on purpose — Terraform state can contain
resource attributes that shouldn't sit in a repo. Back them up out of band
(e.g. `scp` to your workstation, or a private, encrypted location), not by
committing them.

## Prerequisites

- Terraform >= 1.6
- An existing VPC and subnet (this config does not create a VPC)
- AWS credentials in the environment (`AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`, or an `AWS_PROFILE`) —
  only needed from `plan` onward, not for `init` or `validate`

## Apply order

```sh
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: vpc_id, subnet_id, admin_cidr, ssh_public_key

terraform init
terraform validate
terraform fmt -check

terraform plan -out=tfplan
terraform show -json tfplan | jq '[.resource_changes[] | {addr: .address, action: .change.actions[0]}]'
```

Read the `jq` output before applying anything. **Expected output on a clean
first run: eight resources, every action `create`.**

```json
[
  { "addr": "aws_security_group.trade_signals", "action": "create" },
  { "addr": "aws_vpc_security_group_ingress_rule.ssh", "action": "create" },
  { "addr": "aws_key_pair.trade_signals", "action": "create" },
  { "addr": "aws_instance.trade_signals", "action": "create" },
  { "addr": "aws_eip.trade_signals", "action": "create" },
  { "addr": "aws_eip_association.trade_signals", "action": "create" },
  { "addr": "aws_ebs_volume.data", "action": "create" },
  { "addr": "aws_volume_attachment.data", "action": "create" }
]
```

(`data.aws_ami.ubuntu` also appears in the full `resource_changes` list with
action `read` — that's a data source lookup, not a managed resource, and
isn't counted in the eight above.)

If that list matches, apply:

```sh
terraform apply tfplan
```

**Any `delete` or `replace` action in a future `terraform plan` — on this or
any later run — is a stop-and-read-carefully signal.** In particular:

- A `replace` on `aws_ebs_volume.data` should be impossible
  (`prevent_destroy = true`); if you see one anyway, do not proceed —
  something is wrong with the plan or the state, and applying it risks the
  AIS position history that volume holds.
- A `replace` on `aws_instance.trade_signals` proposing to change the `ami`
  argument specifically should not happen either — `lifecycle.ignore_changes
  = [ami]` suppresses that. A `replace` for a different reason (instance
  type, subnet, etc.) is expected to happen occasionally and is fine to
  apply after you understand why.

## Files

| File | Contents |
|---|---|
| `versions.tf` | Terraform/provider version constraints |
| `variables.tf` | All input variables |
| `network.tf` | Security group + SSH ingress rule |
| `compute.tf` | AMI data source, key pair, instance, EIP, EIP association |
| `storage.tf` | Data EBS volume + attachment |
| `outputs.tf` | Instance ID, public IP, security group ID, data volume ID |
| `terraform.tfvars.example` | Placeholder values — copy to `terraform.tfvars` |

## Never run `terraform destroy` here

The data volume holds AIS position history that cannot be re-collected.
`terraform destroy` is never run against this configuration — see the
Infrastructure section of the repo's `CLAUDE.md`. Deprovisioning, if it's
ever needed, is a manual, deliberate, resource-by-resource decision, not a
single command.
