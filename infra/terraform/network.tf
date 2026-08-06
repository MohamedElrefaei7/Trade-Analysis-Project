# Security group for the trade-signals instance. Ingress is expressed as
# separate aws_vpc_security_group_ingress_rule resources rather than inline
# `ingress {}` blocks on this resource: inline blocks take exclusive
# ownership of the rule set and silently remove anything added out of band,
# and mixing inline + standalone rule resources fights itself on every
# apply.
resource "aws_security_group" "trade_signals" {
  name        = "${var.project_name}-sg"
  description = "Security group for the trade-signals EC2 instance"
  vpc_id      = var.vpc_id

  tags = {
    Name = "${var.project_name}-sg"
  }
}

# SSH, scoped to the operator's admin CIDR only. var.admin_cidr has no
# default (see variables.tf), so this can never silently widen to
# 0.0.0.0/0.
resource "aws_vpc_security_group_ingress_rule" "ssh" {
  security_group_id = aws_security_group.trade_signals.id
  description       = "SSH from admin CIDR"
  cidr_ipv4         = var.admin_cidr
  from_port         = 22
  to_port           = 22
  ip_protocol       = "tcp"
}

# No HTTP/HTTPS ingress rule exists yet — nothing listens on this instance
# until a later commit provisions nginx/Docker. When that lands, any rule
# it opens to the world (0.0.0.0/0) must use port 80 or 443; nothing else
# is ever allowed in this security group (test_infra.py enforces this as a
# guardrail against, e.g., accidentally exposing the Streamlit default
# port or Postgres).

resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.trade_signals.id
  cidr_ipv4          = "0.0.0.0/0"
  ip_protocol         = "-1"  # all protocols, all ports
  description         = "Allow all outbound"
}