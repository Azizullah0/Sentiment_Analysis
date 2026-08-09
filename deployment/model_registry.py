"""Ablation checkpoints available for deployment batch jobs (A0 / A1 / A4)."""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths import PATHS  # noqa: E402

DEFAULT_MODEL_ID = "A4"
# Preferred multi-seed folder when A4 root has no config.json (DGX layout).
PREFERRED_SEED_DIR = "seed_42"

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

_SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


def ablation_dir() -> str:
    return PATHS.get("ablation_outputs") or os.path.join(
        PATHS.get("outputs", "outputs"), "ablation"
    )


def model_path_for_id(model_id: str) -> str:
    """Nominal run folder (outputs/ablation/<id>), not necessarily the load path."""
    mid = (model_id or "").strip().upper()
    if mid not in MODEL_CATALOG:
        raise ValueError(
            f"Unknown model_id {model_id!r}. Choose one of: {', '.join(MODEL_CATALOG)}"
        )
    return os.path.join(ablation_dir(), mid)


def is_checkpoint_available(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and os.path.isfile(
        os.path.join(path, "config.json")
    )


def find_checkpoint_dir(run_dir: str) -> Optional[str]:
    """
    Resolve a loadable HF checkpoint directory under an ablation run folder.

    Order:
    1. run_dir itself (config.json at top level, e.g. A0)
    2. run_dir/seed_42 (multi-seed A4 layout on DGX)
    3. any other run_dir/seed_<N> with config.json (lowest seed number)
    """
    if not run_dir or not os.path.isdir(run_dir):
        return None
    if is_checkpoint_available(run_dir):
        return run_dir

    preferred = os.path.join(run_dir, PREFERRED_SEED_DIR)
    if is_checkpoint_available(preferred):
        return preferred

    seed_candidates: List[tuple[int, str]] = []
    try:
        names = os.listdir(run_dir)
    except OSError:
        return None
    for name in names:
        m = _SEED_DIR_RE.match(name)
        if not m:
            continue
        candidate = os.path.join(run_dir, name)
        if is_checkpoint_available(candidate):
            seed_candidates.append((int(m.group(1)), candidate))
    if seed_candidates:
        seed_candidates.sort(key=lambda x: x[0])
        return seed_candidates[0][1]
    return None


def list_models() -> List[Dict[str, Any]]:
    """Return catalog entries with availability (no absolute paths required by UI)."""
    items: List[Dict[str, Any]] = []
    for mid, meta in MODEL_CATALOG.items():
        run_dir = model_path_for_id(mid)
        ckpt = find_checkpoint_dir(run_dir)
        items.append(
            {
                "id": mid,
                "label": meta["label"],
                "description": meta["description"],
                "available": ckpt is not None,
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
    run_dir = model_path_for_id(mid)
    path = find_checkpoint_dir(run_dir)
    if not path:
        raise ValueError(
            f"Model {mid} not found under {run_dir}. "
            f"Expected config.json at the run root or in seed_42/ (multi-seed layout)."
        )
    return {"model_id": mid, "model_path": path}
