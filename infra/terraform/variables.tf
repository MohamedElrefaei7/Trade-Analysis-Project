variable "aws_region" {
  description = "AWS region to deploy the trade-signals footprint into."
  type        = string
  default     = "us-east-1"
}

variable "vpc_id" {
  description = "VPC to launch the instance and security group into."
  type        = string
}

variable "subnet_id" {
  description = "Subnet (within var.vpc_id) to launch the instance into."
  type        = string
}

variable "admin_cidr" {
  description = <<-EOT
    CIDR block allowed to SSH into the instance (e.g. "203.0.113.4/32").
    Deliberately has no default: a missing value must halt the plan rather
    than silently falling back to something world-open.
  EOT
  type        = string
}

variable "ssh_public_key" {
  description = "Public key material (e.g. contents of an id_ed25519.pub) for the operator keypair. Never the private key."
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type. Must stay a Graviton (t4g.*) family — the AMI data source in compute.tf is pinned to arm64. Defaults to t4g.medium (4 GiB): the box runs Postgres, four Python processes, and periodic ElasticNet training concurrently, and t4g.small (2 GiB) needs aggressive swap under that load."
  type        = string
  default     = "t4g.medium"
}

variable "root_volume_size_gb" {
  description = "Size of the disposable root volume, in GB."
  type        = number
  default     = 20
}

variable "data_volume_size_gb" {
  description = "Size of the persistent data volume (AIS position history), in GB."
  type        = number
  default     = 30
}

variable "project_name" {
  description = "Name prefix applied to resource names and tags."
  type        = string
  default     = "trade-signals"
}
