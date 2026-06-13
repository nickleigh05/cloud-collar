terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── DynamoDB ──────────────────────────────────────────────────────────────────

resource "aws_dynamodb_table" "sessions" {
  name         = "cloud-collar-sessions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"
  range_key    = "person_id"

  attribute {
    name = "run_id"
    type = "S"
  }

  attribute {
    name = "person_id"
    type = "N"
  }
}

# ── IAM ───────────────────────────────────────────────────────────────────────

resource "aws_iam_role" "lambda_exec" {
  name = "cloud-collar-lambda-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "cloud-collar-lambda-policy"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # write session records
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem", "dynamodb:BatchWriteItem"]
        Resource = aws_dynamodb_table.sessions.arn
      },
      {
        # basic CloudWatch logging
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# ── Lambda ────────────────────────────────────────────────────────────────────

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/../cloud/lambda/handler.py"
  output_path = "${path.module}/build/handler.zip"
}

resource "aws_lambda_function" "handler" {
  function_name    = "cloud-collar-handler"
  role             = aws_iam_role.lambda_exec.arn
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  runtime          = "python3.12"
  handler          = "handler.lambda_handler"
  timeout          = 10

  environment {
    variables = {
      DYNAMODB_TABLE = aws_dynamodb_table.sessions.name
      API_KEY        = var.api_key
    }
  }
}

# ── API Gateway ───────────────────────────────────────────────────────────────

resource "aws_apigatewayv2_api" "http_api" {
  name          = "cloud-collar-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.http_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.handler.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "post_sessions" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "POST /sessions"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http_api.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.handler.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http_api.execution_arn}/*/*"
}
