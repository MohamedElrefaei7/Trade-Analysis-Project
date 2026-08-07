data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["arm64"]
  }
}

resource "aws_key_pair" "trade_signals" {
  key_name   = "${var.project_name}-key"
  public_key = var.ssh_public_key
}

resource "aws_instance" "trade_signals" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [aws_security_group.trade_signals.id]
  key_name               = aws_key_pair.trade_signals.key_name
  iam_instance_profile   = aws_iam_instance_profile.ssm.name

  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  # IMDSv2-only. Free, and it closes the SSRF-to-instance-credentials path
  # — which matters more once the Phase 8 API is fetching anything.
  metadata_options {
    http_tokens   = "required"
    http_endpoint = "enabled"
  }

  tags = {
    Name = var.project_name
  }

  # Deliberate AMI upgrades: remove this lifecycle block for exactly one
  # `terraform apply`, then restore it. Without ignore_changes, the next
  # Canonical release makes an unrelated `terraform plan` silently propose
  # replacing the running instance.
  lifecycle {
    ignore_changes = [ami]
  }
}

# domain = "vpc" plus a separate association below — associating the EIP
# inside the instance resource would tie re-association to instance
# lifecycle (e.g. a replace-triggered AMI change would drop the address).
resource "aws_eip" "trade_signals" {
  domain = "vpc"

  tags = {
    Name = var.project_name
  }
}

resource "aws_eip_association" "trade_signals" {
  instance_id   = aws_instance.trade_signals.id
  allocation_id = aws_eip.trade_signals.id
}
