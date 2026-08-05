output "instance_id" {
  description = "ID of the trade-signals EC2 instance."
  value       = aws_instance.trade_signals.id
}

output "public_ip" {
  description = "Elastic IP address of the trade-signals instance."
  value       = aws_eip.trade_signals.public_ip
}

output "security_group_id" {
  description = "ID of the trade-signals security group."
  value       = aws_security_group.trade_signals.id
}

output "data_volume_id" {
  description = "ID of the persistent data EBS volume."
  value       = aws_ebs_volume.data.id
}
