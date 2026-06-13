output "api_invoke_url" {
  description = "The HTTPS endpoint to put in CLOUD_COLLAR_API_URL on the edge device"
  value       = "${aws_apigatewayv2_stage.default.invoke_url}/sessions"
}
