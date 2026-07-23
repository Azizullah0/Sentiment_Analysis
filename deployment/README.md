# Deployment — YouTube labeling + DEEP-Dari dashboard

Separate package for serving DEEP-Dari, labeling **real public YouTube comments**, and browsing results in a bilingual **EN / FA (RTL)** dashboard. Training code stays under `scripts/` and `augmentations/`.

## Setup

From the repo root (after the main project requirements are installed):

```bash
pip install -r deployment/requirements.txt
```

Set a [YouTube Data API v3](https://console.cloud.google.com/) key:

```bash
# Linux / macOS / DGX
export YOUTUBE_API_KEY=your-key

# Windows PowerShell
$env:YOUTUBE_API_KEY = "your-key"
```

Model resolution order (first existing checkpoint wins):

1. `outputs/ablation/A4`
2. `Models/parsbert_emotion_incremental`
3. `Models/parsbert_emotion`

Override with `--model-path` or `DEPLOYMENT_MODEL_PATH`.

## Others abstention

If top-class confidence is **below** `--min-confidence` (default **0.50**), the final **`label` is `Others`**. Unusable / non-Persian short text is also `Others`.

## CLI batch labeling

```bash
python -m deployment.youtube_batch --video-id VIDEO_ID --max-comments 500
python -m deployment.youtube_batch --channel-id CHANNEL_ID --max-videos 5 --max-comments 1000
```

Outputs: `outputs/deployment/youtube_batch_<timestamp>/`

## Professional dashboard (EN / FA)

Build the React UI once (needs Node.js 18+):

```bash
cd deployment/dashboard
npm install
npm run build
cd ../..
```

Run API + dashboard together:

```bash
export YOUTUBE_API_KEY=your-key   # required to start jobs from the UI
uvicorn deployment.api:app --host 0.0.0.0 --port 8000
```

Open **http://SERVER:8000**

| Page | Features |
|------|----------|
| Runs | Past batch folders with KPIs |
| Run detail | Emotion charts, filters, RTL comment table, Fear/Anger review marks |
| New job | Start video/channel labeling; live logs |
| Global queue | `review_queue.jsonl` |

Language toggle **EN / FA** in the header (Persian uses Vazirmatn + RTL).

### Dev mode (hot reload)

Terminal 1:

```bash
uvicorn deployment.api:app --reload --port 8000
```

Terminal 2:

```bash
cd deployment/dashboard && npm run dev
```

Open http://127.0.0.1:5173 (Vite proxies `/api` → 8000).

## API (selected)

- `GET /api/health` — model + YouTube key status
- `GET /api/runs` — list batches
- `GET /api/runs/{id}` — summary
- `GET /api/runs/{id}/comments` — paginated filters
- `GET /api/runs/{id}/review` / `POST` — review annotations
- `POST /api/jobs` — start batch job
- `GET /api/jobs/{id}` — job status + logs
- `POST /api/predict` — single/batch text predict

## Optional live chat

```bash
python -m deployment.youtube_monitor --video-id LIVE_VIDEO_ID --poll-seconds 20
```

## Ethics

Public comments only, minimal storage, research/demo use. Alerts notify a human review queue — **no auto-action**.
