# Cloud Collar

![CI](https://github.com/nickleigh05/cloud-collar/actions/workflows/ci.yml/badge.svg)

Edge-based presence analytics pipeline. A camera feed is processed entirely on a local device — person detection, tracking, and re-identification all run on-device — and only anonymous structured session data is sent to a serverless AWS backend. Raw video never leaves the machine.

Built as a portfolio project demonstrating a full-stack pipeline: edge CV inference, appearance-based re-identification, infrastructure as code, and a serverless backend.

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

### Re-identification

ByteTrack IDs are short-lived: when someone leaves the frame their track dies, and they come back as a new ID. Cloud Collar layers appearance-based re-ID on top:

- Each person crop is run through **ResNet18** (pretrained on ImageNet, classifier removed) to produce a 512-d L2-normalized embedding — a fingerprint of how the person looks.
- A new track buffers `PENDING_FRAMES` samples before being judged, so one blurry frame can't cause a misidentification.
- The averaged embedding is compared (cosine similarity) against all known persons; above `SIMILARITY_THRESHOLD` the track is merged, otherwise a new person ID is created.
- People already visible are excluded from matching — one person can't be in two places at once.
- Embeddings are refreshed as a running average every `EMBED_EVERY` frames, adapting to lighting and pose changes over time.

### Metrics tracked per person

| Metric | Description |
|---|---|
| Time on floor | Cumulative seconds the person has been visible |
| Phone usage | Number of sightings + total seconds holding a phone |
| Idle time | Seconds stationary beyond the movement threshold |
| Away time | Seconds a known person was absent from the frame |

---

## Stack

| Layer | Technology |
|---|---|
| Detection | YOLOv8s + ByteTrack (Ultralytics) |
| Re-ID | ResNet18 (torchvision) |
| Video I/O | OpenCV |
| Cloud | AWS Lambda + DynamoDB + API Gateway |
| Infrastructure | Terraform |
| CI | GitHub Actions + pytest + ruff |

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

- Raw video never leaves the device. All inference is local.
- No facial recognition. Embeddings describe whole-body appearance, live only in memory for the duration of a run, and are never uploaded.
- The cloud stores only anonymous numeric IDs, timestamps, and durations.
- Any real deployment requires informed consent and visible signage.

---

## Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Tests use fake embeddings — no camera or GPU required. CI runs on every push.
