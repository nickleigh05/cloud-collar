import json
import os
from decimal import Decimal

import boto3

TABLE_NAME = os.environ["DYNAMODB_TABLE"]
API_KEY    = os.environ["API_KEY"]

dynamodb = boto3.resource("dynamodb")
table    = dynamodb.Table(TABLE_NAME)


def lambda_handler(event, context):

    # --- auth ---
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    if headers.get("x-api-key") != API_KEY:
        return {"statusCode": 401, "body": "Unauthorized"}

    # --- parse body ---
    try:
        body = json.loads(event.get("body") or "{}")
        run_id  = str(body["run_id"])
        persons = list(body["persons"])
        if not run_id or not isinstance(persons, list):
            raise ValueError
    except (KeyError, ValueError, TypeError):
        return {"statusCode": 400, "body": "Bad Request"}

    # --- upsert one DynamoDB item per person ---
    # DynamoDB rejects Python floats — everything numeric goes through Decimal
    with table.batch_writer() as batch:
        for p in persons:
            batch.put_item(Item={
                "run_id":          run_id,
                "person_id":       int(p["person_id"]),
                "on_floor_seconds": Decimal(p["on_floor_seconds"]),
                "phone_sightings":  int(p["phone_sightings"]),
                "phone_seconds":    Decimal(p["phone_seconds"]),
                "idle_seconds":     Decimal(p["idle_seconds"]),
                "away_seconds":     Decimal(p["away_seconds"]),
                "timestamp":        int(body.get("timestamp", 0)),
            })

    return {"statusCode": 200, "body": json.dumps({"saved": len(persons)})}
