# Deployment — YouTube comment labeling

Separate package for serving DEEP-Dari and labeling **real public YouTube comments**. Training code stays under `scripts/` and `augmentations/`.

## Setup

From the repo root (after the main project requirements are installed):

```bash
pip install -r deployment/requirements.txt
```

Get a [YouTube Data API v3](https://console.cloud.google.com/) key, then:

```bash
# Windows PowerShell
$env:YOUTUBE_API_KEY = "your-key"

# Linux / macOS
export YOUTUBE_API_KEY=your-key
```

Or copy `deployment/.env.example` to `deployment/.env` and load it yourself (the scripts read the process environment).

Model resolution order (first existing checkpoint wins):

1. `outputs/ablation/A4`
2. `Models/parsbert_emotion_incremental`
3. `Models/parsbert_emotion`

Override with `--model-path` or `DEPLOYMENT_MODEL_PATH`.

## Others abstention

If the top-class confidence is **below** `--min-confidence` (default **0.50**), the final **`label` is `Others`**. The argmax class is still stored as `raw_emotion`. Unusable / non-Persian short text is also `Others` (`abstain_reason=unusable_text`).

Fear/Anger review queue entries are created only when the **final** label is Fear or Anger (never for Others).

## Batch-label video comments

```bash
python -m deployment.youtube_batch --video-id VIDEO_ID --max-comments 500
python -m deployment.youtube_batch --channel-id CHANNEL_ID --max-videos 5 --max-comments 1000 --min-confidence 0.50
```

Outputs under `outputs/deployment/youtube_batch_<timestamp>/`:

| File | Content |
|------|---------|
| `labeled_comments.csv` / `.sqlite` | All rows with `label`, `raw_emotion`, `confidence`, `abstain` |
| `summary.json` | Counts for 8 emotions + Others |
| `review_candidates.csv` | High-confidence Fear/Anger |
| `others_sample.csv` | Sample of abstained comments |

Global queue: `outputs/deployment/review_queue.jsonl`.

## API

```bash
uvicorn deployment.api:app --host 0.0.0.0 --port 8000
```

- `GET /health`
- `POST /predict` body: `{"text": "..."}` or `{"texts": ["..."]}` optional `min_confidence`

## Optional live chat

```bash
python -m deployment.youtube_monitor --video-id LIVE_VIDEO_ID --poll-seconds 20
```

Requires an **active** live broadcast (`activeLiveChatId`). Prefer batch mode for thesis demos (quota-friendly).

## Ethics

Public comments only, minimal storage, research/demo use. Alerts notify a human review queue — **no auto-action**.
