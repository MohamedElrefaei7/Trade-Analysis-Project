# SSM instance profile for the trade-signals instance. This is what made
# the SSH-lockout recovery in CONTEXT.md possible (Provision 2c) — it was
# set up as a manual console click-through at launch, with nothing here to
# reconstruct it if the instance is ever replaced. This commit brings it
# under Terraform so the SSM path never again depends on a step someone has
# to remember to redo by hand.

# Scoped to ec2.amazonaws.com specifically via a data source rather than a
# hand-written JSON heredoc, so the trust policy's structure is checked at
# `terraform validate` time instead of being an opaque string someone could
# widen by accident later (e.g. into a wildcard principal) while debugging
# something unrelated.
data "aws_iam_policy_document" "ssm_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ssm" {
  name               = "trade-signals-ssm-role"
  assume_role_policy = data.aws_iam_policy_document.ssm_assume_role.json
}

# AWS-managed policy, not an inline copy — AWS maintains and updates this
# policy's permissions as SSM's own requirements evolve. An inline policy
# would freeze this instance's SSM capability at whatever permissions
# existed the day it was written, silently falling behind AWS's own service
# requirements.
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.ssm.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ssm" {
  name = "trade-signals-ssm-role"
  role = aws_iam_role.ssm.name
}
