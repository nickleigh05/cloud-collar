variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "api_key" {
  description = "Shared secret sent by the edge device in the x-api-key header"
  type        = string
  sensitive   = true
}
