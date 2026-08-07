"""
test_infra.py — enforcement tests for the design decisions documented in
infra/terraform/README.md and the Commit 1 spec. Each test parses the raw
.tf files with hcl2 (no `terraform` binary, no AWS credentials required)
and should go red the moment the thing it guards is reverted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hcl2
import pytest
from hcl2.utils import SerializationOptions

REPO_ROOT = Path(__file__).resolve().parent.parent
TF_DIR = REPO_ROOT / "infra" / "terraform"

_OPTS = SerializationOptions(
    with_comments=False,
    explicit_blocks=False,
    strip_string_quotes=True,
)


def _load_all_tf() -> dict[str, list[Any]]:
    merged: dict[str, list[Any]] = {}
    for tf_file in sorted(TF_DIR.glob("*.tf")):
        with tf_file.open() as f:
            parsed = hcl2.load(f, serialization_options=_OPTS)
        for block_type, blocks in parsed.items():
            merged.setdefault(block_type, [])
            merged[block_type].extend(blocks)
    return merged


def _resources(tf: dict, resource_type: str) -> dict[str, dict]:
    """{resource_name: attrs} for every resource of the given type, across all files."""
    out: dict[str, dict] = {}
    for block in tf.get("resource", []):
        if resource_type in block:
            out.update(block[resource_type])
    return out


def _variables(tf: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for block in tf.get("variable", []):
        out.update(block)
    return out


def _lifecycle(resource_attrs: dict) -> dict:
    """lifecycle {} sub-blocks come back as a one-element list of dicts."""
    lifecycle = resource_attrs.get("lifecycle", [{}])
    return lifecycle[0] if isinstance(lifecycle, list) else lifecycle


@pytest.fixture(scope="module")
def tf() -> dict:
    return _load_all_tf()


# ---------------------------------------------------------------------------
# test_no_ingress_rule_open_to_world_except_http_https
# ---------------------------------------------------------------------------


def test_no_ingress_rule_open_to_world_except_http_https(tf):
    rules = _resources(tf, "aws_vpc_security_group_ingress_rule")
    assert rules, "expected at least one aws_vpc_security_group_ingress_rule resource"

    world_open = [
        (name, attrs) for name, attrs in rules.items() if attrs.get("cidr_ipv4") == "0.0.0.0/0"
    ]
    for name, attrs in world_open:
        assert attrs.get("from_port") in (80, 443), (
            f"ingress rule {name!r} is open to 0.0.0.0/0 with from_port="
            f"{attrs.get('from_port')!r} — only 80/443 may ever be opened to the world"
        )


# ---------------------------------------------------------------------------
# test_no_ingress_rule_for_postgres
# ---------------------------------------------------------------------------


def test_no_ingress_rule_for_postgres(tf):
    rules = _resources(tf, "aws_vpc_security_group_ingress_rule")
    for name, attrs in rules.items():
        assert attrs.get("from_port") != 5432, f"ingress rule {name!r} exposes Postgres (from_port)"
        assert attrs.get("to_port") != 5432, f"ingress rule {name!r} exposes Postgres (to_port)"


# ---------------------------------------------------------------------------
# test_admin_cidr_has_no_default
# ---------------------------------------------------------------------------


def test_admin_cidr_has_no_default(tf):
    variables = _variables(tf)
    assert "admin_cidr" in variables, "expected a var.admin_cidr declaration"
    assert "default" not in variables["admin_cidr"], (
        "var.admin_cidr must have no default — a missing value should halt the plan, "
        "not silently fall back to something world-open"
    )


# ---------------------------------------------------------------------------
# test_data_volume_prevented_from_destruction
# ---------------------------------------------------------------------------


def test_data_volume_prevented_from_destruction(tf):
    volumes = _resources(tf, "aws_ebs_volume")
    assert "data" in volumes, "expected an aws_ebs_volume.data resource"
    assert _lifecycle(volumes["data"]).get("prevent_destroy") is True


# ---------------------------------------------------------------------------
# test_instance_ignores_ami_changes
# ---------------------------------------------------------------------------


def test_instance_ignores_ami_changes(tf):
    instances = _resources(tf, "aws_instance")
    assert len(instances) == 1, f"expected exactly one aws_instance resource, found {list(instances)}"
    (instance,) = instances.values()
    ignore_changes = _lifecycle(instance).get("ignore_changes", [])
    assert "ami" in ignore_changes


# ---------------------------------------------------------------------------
# test_instance_type_is_graviton
# ---------------------------------------------------------------------------


def test_instance_type_is_graviton(tf):
    variables = _variables(tf)
    assert "instance_type" in variables
    default = variables["instance_type"].get("default")
    assert isinstance(default, str) and default.startswith("t4g."), (
        f"var.instance_type default is {default!r} — must stay a t4g.* Graviton type, "
        "the AMI data source is pinned to arm64"
    )


# ---------------------------------------------------------------------------
# test_default_instance_type_is_medium
# ---------------------------------------------------------------------------
#
# Distinct from test_instance_type_is_graviton above: that test guards the
# *family* (must be t4g.*, since the AMI is arm64) and t4g.small is a
# legitimate value for it — a valid override for someone who wants a
# smaller box. This test guards the *size* — the box runs Postgres, four
# Python processes, and periodic ElasticNet training concurrently, and
# t4g.small (2 GiB) needs aggressive swap under that load. The two tests
# must be able to fail independently: this one goes red the moment the
# default reverts to t4g.small (or anything but t4g.medium) even though
# t4g.small would still satisfy the Graviton check.


def test_default_instance_type_is_medium(tf):
    variables = _variables(tf)
    assert "instance_type" in variables
    default = variables["instance_type"].get("default")
    assert default == "t4g.medium", (
        f"var.instance_type default is {default!r}, expected exactly 't4g.medium' — "
        "t4g.small needs aggressive swap under this box's Postgres + 4 Python "
        "processes + ElasticNet training load"
    )


# ---------------------------------------------------------------------------
# test_root_volume_smaller_than_data_volume
# ---------------------------------------------------------------------------


def test_root_volume_smaller_than_data_volume(tf):
    variables = _variables(tf)
    root_size = variables["root_volume_size_gb"]["default"]
    data_size = variables["data_volume_size_gb"]["default"]
    assert root_size < data_size, (
        f"root_volume_size_gb default ({root_size}) must stay smaller than "
        f"data_volume_size_gb default ({data_size}) — consolidating onto a single, "
        "larger root volume is the failure mode this guards against"
    )


# ---------------------------------------------------------------------------
# test_instance_has_ssm_instance_profile
# ---------------------------------------------------------------------------


def test_instance_has_ssm_instance_profile(tf):
    instances = _resources(tf, "aws_instance")
    (instance,) = instances.values()
    profile_ref = instance.get("iam_instance_profile")
    assert profile_ref, "aws_instance.trade_signals is missing iam_instance_profile"

    profiles = _resources(tf, "aws_iam_instance_profile")
    assert profiles, "expected an aws_iam_instance_profile resource in iam.tf"
    (profile_name,) = profiles.keys()
    assert profile_ref == f"${{aws_iam_instance_profile.{profile_name}.name}}", (
        f"aws_instance.trade_signals.iam_instance_profile ({profile_ref!r}) must "
        f"reference aws_iam_instance_profile.{profile_name} by name, not a literal "
        "string — otherwise Terraform can't detect drift between the two"
    )


# ---------------------------------------------------------------------------
# test_ssm_role_uses_managed_policy
# ---------------------------------------------------------------------------


def test_ssm_role_uses_managed_policy(tf):
    managed_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"

    attachments = _resources(tf, "aws_iam_role_policy_attachment")
    matches = [attrs for attrs in attachments.values() if attrs.get("policy_arn") == managed_arn]
    assert matches, (
        f"expected an aws_iam_role_policy_attachment referencing {managed_arn!r} — "
        "AWS maintains this policy's permissions as SSM's requirements evolve; an "
        "inline copy would freeze them at whatever existed the day it was written"
    )

    inline_policies = _resources(tf, "aws_iam_role_policy")
    for name, attrs in inline_policies.items():
        policy_text = str(attrs.get("policy", ""))
        assert "AmazonSSMManagedInstanceCore" not in policy_text, (
            f"aws_iam_role_policy.{name!r} embeds AmazonSSMManagedInstanceCore inline "
            "— it must be an aws_iam_role_policy_attachment instead"
        )
