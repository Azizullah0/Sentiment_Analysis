"""FastAPI serve endpoint for DEEP-Dari deployment (Others abstention)."""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths import PATHS  # noqa: E402

from deployment.predictor import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    EmotionPredictor,
    append_review_queue,
)

app = FastAPI(
    title="DEEP-Dari Emotion API",
    description="Label Dari/Persian text; low-confidence predictions become Others.",
    version="1.0.0",
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
    min_confidence: Optional[float] = Field(
        default=None,
        description="Override abstention threshold (default 0.50).",
    )


class PredictItem(BaseModel):
    text: Optional[str]
    label: str
    raw_emotion: str
    confidence: float
    abstain: bool
    abstain_reason: Optional[str] = None
    all_probabilities: dict = Field(default_factory=dict)


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
    except FileNotFoundError as exc:
        # Allow process to start; /health reports error until model is available
        app.state.model_error = str(exc)
    else:
        app.state.model_error = None


@app.get("/health")
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
        "review_queue": _review_queue,
    }


@app.post("/predict")
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
