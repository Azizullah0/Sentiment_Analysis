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

## Filter-then-classify (Excluded vs Others)

Pipeline:

1. **Clean** — strip URLs, `@mentions`, emojis; unify ي/ی and ك/ک  
2. **Gate** — drop comments that are empty, emoji-only, too short (&lt; 4 tokens), or mostly non-Persian/Dari (Persian-script ratio &lt; 0.5) → **`Excluded`** (no model call)  
3. **Predict** — ParsBERT on remaining text  
4. **Abstain** — if confidence &lt; `--min-confidence` (default **0.50**) → **`Others`**

So **Others = model unsure on usable Dari/Persian**, not emoji/Latin noise.

`summary.json` reports both views:

- `exclusion_rate`, `n_excluded`  
- `others_rate_among_usable`, `label_counts_usable`  
- `label_counts_all` (includes Excluded)

Dashboard **Run detail** has a defense toggle: **Usable only** (default) vs **All comments**.

Refresh old runs (remap legacy `Others`+`unusable_text` → `Excluded`):

```bash
python -m deployment.reaggregate
python -m deployment.reaggregate --run-id youtube_batch_20260727T035253Z
```

## CLI batch labeling

```bash
python -m deployment.youtube_batch --video-id VIDEO_ID --max-comments 500
python -m deployment.youtube_batch --channel-id CHANNEL_ID --max-videos 5 --max-comments 1000
```

Video and channel arguments accept **raw IDs or full YouTube links**. The CLI and New Job form normalize them the same way:

| Paste | Resolved to |
|-------|-------------|
| `https://www.youtube.com/watch?v=…`, `youtu.be/…`, `/shorts/…`, `/embed/…` | 11-char video id |
| Bare 11-char id | unchanged |
| `https://www.youtube.com/channel/UC…` or bare `UC…` | channel id |
| `https://www.youtube.com/@Handle` or `@Handle` | `UC…` via YouTube API (`channels.list(forHandle=…)`) |
| `/c/…` or `/user/…` | best-effort resolve via API search |

Examples:

```bash
python -m deployment.youtube_batch --video-id 'https://www.youtube.com/watch?v=16e75OffBTA'
python -m deployment.youtube_batch --channel-id '@SomeHandle' --max-videos 5
```

Invalid pastes return a clear error before the job starts.

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
