# Cloud Collar

![CI](https://github.com/nickleigh05/cloud-collar/actions/workflows/ci.yml/badge.svg)

A real-time employee productivity analytics system that tracks behavior — time on the floor, phone usage, idle time — using a camera feed processed entirely on-device. No facial recognition. No video leaves the machine.

---

## Demo

<video src="https://github.com/user-attachments/assets/73dc262c-4522-4b2c-8d1e-4272f53b9319" width="800" autoplay loop muted playsinline></video>

---

## Why I built this

I wanted a tool that could actually measure employee productivity in a real workplace — not just headcount, but behavior. How long is someone on the floor? How often are they picking up their phone? Are they idle?

Every existing solution I looked at was either way too invasive (facial recognition, full video streamed to a cloud server) or too shallow to be useful. I wanted something in between: rich behavioral data, with a privacy model I could actually defend.

The core insight is that you don't need to know *who* someone is to track how they behave. Cloud Collar gives every person an anonymous ID based on their appearance and tracks their session from there. The cloud never sees anything except numbers.

---

## How it works

```
[Camera / Raspberry Pi]
        |
        v
  YOLOv8 detects people each frame
  ByteTrack assigns short-lived track IDs
  ResNet18 re-ID matches returning people to stable person IDs
  Tracker accumulates per-person session data
  Uploader POSTs anonymous JSON snapshots every 30s
        |
        | HTTPS  (JSON only — never video)
        v
  AWS API Gateway → Lambda → DynamoDB
```

### Re-identification — the hard part

Standard trackers like ByteTrack assign a track ID to each visible person. When someone steps out of frame, the track dies — and when they walk back in, they're a brand new ID. That's fine for object counting, but useless for session-level analytics.

Cloud Collar layers appearance-based re-identification on top to maintain continuity:

- Each person crop is run through a **ResNet18** backbone (pretrained on ImageNet, classifier removed) to produce a 512-d L2-normalized embedding — a fingerprint of how that person looks.
- New tracks buffer a few frames before a decision is made, so one blurry frame can't cause a misidentification.
- The averaged embedding is compared against all known persons using cosine similarity. Above the match threshold, the track is merged into the existing record. Below it, a new person ID is created.
- People currently in frame are excluded from matching — one person can't be in two places at once.
- Embeddings are refreshed as a running average every 15 frames, adapting to lighting and pose changes over time.

### What gets tracked per person

| Metric | Description |
|---|---|
| Time on floor | Cumulative seconds the person has been visible |
| Phone usage | Number of sightings + total seconds holding a phone |
| Idle time | Seconds stationary beyond the movement threshold |
| Away time | Seconds a known person was absent from the frame |

---

## Tech stack — and why

**YOLOv8s** — Fast enough for real-time inference on edge hardware without a dedicated GPU. The small variant hits the right balance of speed and accuracy for person detection.

**ByteTrack** — Built into Ultralytics, handles short-lived track assignment with no extra configuration. Paired with the re-ID layer, it becomes a robust full-session tracker.

**ResNet18** — Lightweight enough to run on a Raspberry Pi. With the classification head removed, the final feature layer produces appearance embeddings that generalize well to unseen people without any fine-tuning.

**AWS Lambda** — Serverless, so the backend scales to zero when the system isn't running. No instances to manage, and the cost for a single-camera deployment is negligible.

**DynamoDB** — Pay-per-request and schemaless. Session records have a variable number of persons, so a fixed relational schema would just add friction.

**Terraform** — One `terraform apply` stands up the entire backend in about 30 seconds. `terraform destroy` tears it down cleanly. The whole infrastructure is version-controlled and reproducible.

---

## Setup

### Prerequisites

- Python 3.12+
- A webcam or video file
- AWS account + AWS CLI configured (`aws configure`)
- Terraform installed

### 1 — Clone and install

```bash
git clone https://github.com/nickleigh05/cloud-collar.git
cd cloud-collar
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2 — Deploy the AWS backend

```bash
cd infra
terraform init
terraform apply -var="api_key=pick-any-secret-string"
```

This creates the DynamoDB table, Lambda function, and API Gateway in about 30 seconds. It prints an `api_invoke_url` when done.

To tear everything down: `terraform destroy -var="api_key=same-secret"`

### 3 — Configure the edge device

```bash
cp .env.example .env
# edit .env — paste in the api_invoke_url and the same api_key
```

### 4 — Run

```bash
cd edge
python main.py
```

Press `q` to quit. Set `VIDEO_PATH = 0` in `main.py` for a live webcam, or point it at a video file.

---

## Repository structure

```
cloud-collar/
├── edge/
│   ├── main.py          # detection + tracking loop
│   ├── tracker.py       # Person and Tracker classes
│   ├── reid.py          # ResNet18 embedding extractor
│   └── uploader.py      # batches and POSTs session snapshots
├── cloud/
│   └── lambda/
│       └── handler.py   # Lambda: auth, validation, DynamoDB upsert
├── infra/
│   ├── main.tf          # DynamoDB, Lambda, API Gateway, IAM
│   ├── variables.tf
│   └── outputs.tf
├── tests/
│   └── test_tracker.py  # tracker + re-ID logic (no camera or GPU needed)
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

---

## Privacy

This was a first-class design constraint, not an afterthought.

- Raw video never leaves the device. All inference — detection, tracking, re-ID — runs locally.
- No facial recognition. Embeddings describe whole-body appearance and are never uploaded.
- The cloud receives only anonymous numeric IDs, timestamps, and durations.
- Embeddings live only in memory for the duration of a run and are discarded on exit.
- Any real deployment requires informed consent and visible signage.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests use fake embeddings — no camera or GPU needed. CI runs on every push.
