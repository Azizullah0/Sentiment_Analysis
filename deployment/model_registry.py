"""Ablation checkpoints available for deployment batch jobs (A0 / A1 / A4)."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths import PATHS  # noqa: E402

DEFAULT_MODEL_ID = "A4"

# Stable catalog for the dashboard / CLI (same 8-class label space).
MODEL_CATALOG: Dict[str, Dict[str, str]] = {
    "A4": {
        "label": "A4 — full template stack (recommended)",
        "description": "Fear + Surprise + Anger + Disgust templates. Best Macro-F1 / Fear F1.",
    },
    "A0": {
        "label": "A0 — no templates",
        "description": "Pseudo-labels + class weights only. Weak Fear (baseline).",
    },
    "A1": {
        "label": "A1 — Fear templates",
        "description": "A0 + 9,000 Fear templates. Large Fear F1 jump.",
    },
}


def ablation_dir() -> str:
    return PATHS.get("ablation_outputs") or os.path.join(
        PATHS.get("outputs", "outputs"), "ablation"
    )


def model_path_for_id(model_id: str) -> str:
    mid = (model_id or "").strip().upper()
    if mid not in MODEL_CATALOG:
        raise ValueError(
            f"Unknown model_id {model_id!r}. Choose one of: {', '.join(MODEL_CATALOG)}"
        )
    return os.path.join(ablation_dir(), mid)


def is_checkpoint_available(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def list_models() -> List[Dict[str, Any]]:
    """Return catalog entries with availability (no absolute paths required by UI)."""
    items: List[Dict[str, Any]] = []
    for mid, meta in MODEL_CATALOG.items():
        path = model_path_for_id(mid)
        items.append(
            {
                "id": mid,
                "label": meta["label"],
                "description": meta["description"],
                "available": is_checkpoint_available(path),
            }
        )
    return items


def resolve_model_id(model_id: Optional[str] = None) -> Dict[str, str]:
    """
    Resolve a model_id to a checkpoint path.
    Default A4; raises ValueError if unknown or missing on disk.
    """
    mid = (model_id or DEFAULT_MODEL_ID).strip().upper() or DEFAULT_MODEL_ID
    if mid not in MODEL_CATALOG:
        raise ValueError(
            f"Unknown model_id {model_id!r}. Choose one of: {', '.join(MODEL_CATALOG)}"
        )
    path = model_path_for_id(mid)
    if not is_checkpoint_available(path):
        raise ValueError(
            f"Model {mid} not found at {path}. "
            f"Train or copy the checkpoint under outputs/ablation/{mid}/."
        )
    return {"model_id": mid, "model_path": path}
