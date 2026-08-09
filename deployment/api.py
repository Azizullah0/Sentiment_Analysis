"""FastAPI: predict + dashboard API + static SPA for DEEP-Dari."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths import PATHS  # noqa: E402

from deployment.batch_service import (  # noqa: E402
    get_job,
    list_jobs,
    list_runs,
    load_review_candidates,
    load_review_queue,
    load_run_detail,
    query_comments,
    run_dir,
    save_review_annotations,
    start_batch_job,
)
from deployment.model_registry import (  # noqa: E402
    DEFAULT_MODEL_ID,
    list_models,
)
from deployment.predictor import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    EmotionPredictor,
    append_review_queue,
)

app = FastAPI(
    title="DEEP-Dari Emotion API",
    description="Label Dari/Persian text; dashboard for YouTube batch runs.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Optional[EmotionPredictor] = None
_default_min_confidence = float(
    os.environ.get("DEPLOYMENT_MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)
)
_review_queue = os.path.join(
    PATHS.get("deployment_outputs", "outputs/deployment"),
    "review_queue.jsonl",
)


class PredictRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    min_confidence: Optional[float] = None


class PredictItem(BaseModel):
    text: Optional[str]
    label: str
    raw_emotion: str
    confidence: float
    abstain: bool
    abstain_reason: Optional[str] = None
    all_probabilities: dict = Field(default_factory=dict)


class JobRequest(BaseModel):
    video_ids: List[str] = Field(default_factory=list)
    channel_id: Optional[str] = None
    max_videos: int = 5
    max_comments: int = 500
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    model_id: str = DEFAULT_MODEL_ID


class ReviewUpdate(BaseModel):
    comment_id: str
    checked: bool = True
    note: Optional[str] = None


class ReviewBulkUpdate(BaseModel):
    items: List[ReviewUpdate]


def get_predictor() -> EmotionPredictor:
    global _predictor
    if _predictor is None:
        model_path = os.environ.get("DEPLOYMENT_MODEL_PATH") or None
        _predictor = EmotionPredictor(
            model_path=model_path,
            min_confidence=_default_min_confidence,
        )
    return _predictor


@app.on_event("startup")
def startup() -> None:
    try:
        get_predictor()
        app.state.model_error = None
    except FileNotFoundError as exc:
        app.state.model_error = str(exc)


@app.get("/health")
@app.get("/api/health")
def health():
    err = getattr(app.state, "model_error", None)
    pred = None
    try:
        pred = get_predictor()
        err = None
        app.state.model_error = None
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
    return {
        "status": "ok" if err is None else "model_unavailable",
        "error": err,
        "device": str(pred.device) if pred else None,
        "model_path": pred.model_path if pred else None,
        "min_confidence": _default_min_confidence,
        "youtube_key_set": bool(os.environ.get("YOUTUBE_API_KEY", "").strip()),
        "review_queue": _review_queue,
        "n_runs": len(list_runs()),
    }


@app.post("/predict")
@app.post("/api/predict")
def predict(req: PredictRequest) -> Union[PredictItem, List[PredictItem]]:
    if not req.text and not req.texts:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'texts'.")
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    threshold = (
        req.min_confidence
        if req.min_confidence is not None
        else _default_min_confidence
    )
    payload = req.text if req.text is not None else req.texts
    result = predictor.predict(payload, min_confidence=threshold)

    def to_item(item: dict) -> PredictItem:
        append_review_queue(item, _review_queue)
        return PredictItem(
            text=item.get("text"),
            label=item["label"],
            raw_emotion=item.get("raw_emotion") or "",
            confidence=float(item["confidence"]),
            abstain=bool(item["abstain"]),
            abstain_reason=item.get("abstain_reason"),
            all_probabilities=item.get("all_probabilities") or {},
        )

    if isinstance(result, list):
        return [to_item(r) for r in result]
    return to_item(result)


@app.get("/api/runs")
def api_list_runs():
    return {"runs": list_runs()}


@app.get("/api/runs/{run_id}")
def api_run_detail(run_id: str):
    try:
        return load_run_detail(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/comments")
def api_run_comments(
    run_id: str,
    label: Optional[str] = None,
    video_id: Optional[str] = None,
    abstain: Optional[bool] = None,
    usable_only: bool = False,
    q: Optional[str] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
):
    try:
        return query_comments(
            run_id,
            label=label,
            video_id=video_id,
            abstain=abstain,
            usable_only=usable_only,
            q=q,
            offset=offset,
            limit=limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/review")
def api_run_review(run_id: str):
    try:
        return {"items": load_review_candidates(run_id)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/review")
def api_save_review(run_id: str, body: ReviewBulkUpdate):
    try:
        updates = {
            item.comment_id: {"checked": item.checked, "note": item.note}
            for item in body.items
        }
        return {"annotations": save_review_annotations(run_id, updates)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/export.csv")
def api_export_csv(run_id: str):
    try:
        path = os.path.join(run_dir(run_id), "labeled_comments.csv")
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="CSV not found")
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"{run_id}_labeled_comments.csv",
    )


@app.get("/api/review-queue")
def api_review_queue(limit: int = Query(200, ge=1, le=2000)):
    return {"items": load_review_queue(limit=limit)}


@app.get("/api/models")
def api_list_models():
    return {"models": list_models(), "default": DEFAULT_MODEL_ID}


@app.post("/api/jobs")
def api_start_job(body: JobRequest):
    try:
        job = start_batch_job(
            video_ids=body.video_ids or None,
            channel_id=body.channel_id,
            max_videos=body.max_videos,
            max_comments=body.max_comments,
            min_confidence=body.min_confidence,
            model_id=body.model_id or DEFAULT_MODEL_ID,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job


@app.get("/api/jobs")
def api_list_jobs():
    return {"jobs": list_jobs()}


@app.get("/api/jobs/{job_id}")
def api_get_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# Static SPA (built with npm run build in deployment/dashboard)
_DASHBOARD_DIST = os.path.join(os.path.dirname(__file__), "dashboard", "dist")


def _mount_spa() -> None:
    if not os.path.isdir(_DASHBOARD_DIST):
        return
    assets = os.path.join(_DASHBOARD_DIST, "assets")
    if os.path.isdir(assets):
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def spa_index():
        index = os.path.join(_DASHBOARD_DIST, "index.html")
        return FileResponse(index)

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        # Let API routes win; this is registered last for SPA client routes
        if full_path.startswith("api/") or full_path in (
            "health",
            "predict",
            "docs",
            "openapi.json",
            "redoc",
        ):
            raise HTTPException(status_code=404)
        candidate = os.path.join(_DASHBOARD_DIST, full_path)
        if os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_DASHBOARD_DIST, "index.html"))


_mount_spa()
