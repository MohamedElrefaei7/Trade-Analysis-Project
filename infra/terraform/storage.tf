# Persistent data volume, carrying AIS position history that cannot be
# re-collected if lost. Kept as its own aws_ebs_volume — not part of the
# instance's root_block_device — specifically so instance-level changes
# (AMI swaps, instance replacement) never touch this data.
resource "aws_ebs_volume" "data" {
  availability_zone = aws_instance.trade_signals.availability_zone
  size              = var.data_volume_size_gb
  type              = "gp3"
  encrypted         = true

  tags = {
    Name = "${var.project_name}-data"
  }

  # The instance is replaceable; this volume is not. prevent_destroy lives
  # here only — not on the instance — so ordinary instance-level operations
  # never require a manual state edit to proceed.
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_volume_attachment" "data" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.data.id
  instance_id = aws_instance.trade_signals.id
}
