import json
import os
import time
import urllib.error
import urllib.request
import uuid

RUN_ID = str(uuid.uuid4()) # groups all uploads from this run together


def build_payload(tracker) -> dict:

    now = time.time()
    persons = []

    for person in tracker.persons.values():

        phone_time = person.phone_time
        if person.phone_visible and person.phone_start:
            phone_time += now - person.phone_start

        away_time = person.away_time
        if person.away_start:
            away_time += now - person.away_start

        persons.append({
            "person_id": person.person_id,
            "on_floor_seconds": round(person.total_time),
            "phone_sightings": person.phone_sightings,
            "phone_seconds": round(phone_time),
            "idle_seconds": round(person.idle_time),
            "away_seconds": round(away_time),
        })

    return {
        "run_id": RUN_ID,
        "timestamp": int(now),
        "persons": persons,
    }


def upload(tracker) -> bool:

    api_url = os.environ.get("CLOUD_COLLAR_API_URL", "")
    api_key = os.environ.get("CLOUD_COLLAR_API_KEY", "")

    if not api_url:
        print("[uploader] CLOUD_COLLAR_API_URL not set — skipping upload")
        return False

    payload = build_payload(tracker)
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        api_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"[uploader] uploaded {len(payload['persons'])} persons — {resp.status}")
            return True
    except urllib.error.HTTPError as e:
        print(f"[uploader] HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        print(f"[uploader] network error: {e.reason}")
    except Exception as e:
        print(f"[uploader] unexpected error: {e}")

    return False
