"""Ablation checkpoints available for deployment batch jobs (A0 / A1 / A4).

Resolves loadable HuggingFace dirs without manual symlinks when possible:

* A0 — ``outputs/ablation/A0`` (top-level config.json)
* A4 — ``outputs/ablation/A4`` or ``A4/seed_42`` (multi-seed), else
  ``Models/parsbert_emotion_incremental`` if trained on allAug
* A1 — ``outputs/ablation/A1`` (or seed_/checkpoint-*), else legacy
  ``outputs/full_8label_aug_*`` whose metadata points at fearAug (not allAug)
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths import PATHS  # noqa: E402

DEFAULT_MODEL_ID = "A4"
PREFERRED_SEED_DIR = "seed_42"

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
_CKPT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
_FEAR_AUG_MARKER = "Combined_Labeled_Dataset_with_fearAug.csv"
_ALL_AUG_MARKER = "Combined_Labeled_Dataset_with_allAug.csv"


def ablation_dir() -> str:
    return PATHS.get("ablation_outputs") or os.path.join(
        PATHS.get("outputs", "outputs"), "ablation"
    )


def outputs_dir() -> str:
    return PATHS.get("outputs") or os.path.join(
        PATHS.get("storage_root") or os.path.dirname(ablation_dir()), "outputs"
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


def _best_checkpoint_subdir(run_dir: str) -> Optional[str]:
    """Prefer highest checkpoint-<step> that has config.json."""
    best: Optional[Tuple[int, str]] = None
    try:
        names = os.listdir(run_dir)
    except OSError:
        return None
    for name in names:
        m = _CKPT_DIR_RE.match(name)
        if not m:
            continue
        candidate = os.path.join(run_dir, name)
        if not is_checkpoint_available(candidate):
            continue
        step = int(m.group(1))
        if best is None or step > best[0]:
            best = (step, candidate)
    return best[1] if best else None


def find_checkpoint_dir(run_dir: str) -> Optional[str]:
    """
    Resolve a loadable HF checkpoint under an ablation (or legacy) run folder.

    Order: run root → seed_42 → other seed_* → highest checkpoint-*.
    """
    if not run_dir or not os.path.isdir(run_dir):
        return None
    if is_checkpoint_available(run_dir):
        return run_dir

    preferred = os.path.join(run_dir, PREFERRED_SEED_DIR)
    if is_checkpoint_available(preferred):
        return preferred

    seed_candidates: List[Tuple[int, str]] = []
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

    return _best_checkpoint_subdir(run_dir)


def _read_training_metadata(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, "training_metadata.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _dataset_basename(meta: Dict[str, Any]) -> str:
    ds = (meta.get("dataset_path") or "").replace("\\", "/")
    return ds.rsplit("/", 1)[-1] if ds else ""


def find_legacy_a1_checkpoint() -> Optional[str]:
    """
    Locate Fear-only (thesis A1 / Phase 4) weights under outputs/full_8label_aug_*.

    Skips runs whose metadata points at allAug (misnamed Jul-2026 runs).
    """
    root = outputs_dir()
    if not os.path.isdir(root):
        return None
    ranked: List[Tuple[str, int, str]] = []
    try:
        names = os.listdir(root)
    except OSError:
        return None
    for name in names:
        if not name.startswith("full_8label_aug_"):
            continue
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue
        meta = _read_training_metadata(run_dir)
        if not meta:
            continue
        base = _dataset_basename(meta)
        if base != _FEAR_AUG_MARKER:
            continue
        ckpt = find_checkpoint_dir(run_dir)
        if not ckpt:
            continue
        size = int(meta.get("dataset_size") or 0)
        # Prefer canonical A1 size, then newer folder name
        ranked.append((name, size, ckpt))
    if not ranked:
        return None
    ranked.sort(
        key=lambda t: (
            0 if t[1] == 400691 else 1,
            t[0],  # timestamp in folder name; last wins when reverse
        )
    )
    # After sort, put preferred size first; among those, take lexicographically last name
    preferred = [t for t in ranked if t[1] == 400691] or ranked
    preferred.sort(key=lambda t: t[0])
    return preferred[-1][2]


def find_legacy_a4_checkpoint() -> Optional[str]:
    """Fallback: Models/parsbert_emotion_incremental if trained on allAug."""
    path = PATHS.get("incremental_finetuned_model")
    if not path or not is_checkpoint_available(path):
        return None
    meta = _read_training_metadata(path)
    if meta is not None:
        base = _dataset_basename(meta)
        if base and base != _ALL_AUG_MARKER and "with_allAug" not in base:
            return None
    return path


def resolve_checkpoint_path(model_id: str) -> Optional[str]:
    """Full resolution chain for a catalog id."""
    mid = (model_id or "").strip().upper()
    if mid not in MODEL_CATALOG:
        return None
    primary = find_checkpoint_dir(model_path_for_id(mid))
    if primary:
        return primary
    if mid == "A1":
        return find_legacy_a1_checkpoint()
    if mid == "A4":
        return find_legacy_a4_checkpoint()
    return None


def list_models() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for mid, meta in MODEL_CATALOG.items():
        ckpt = resolve_checkpoint_path(mid)
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
    mid = (model_id or DEFAULT_MODEL_ID).strip().upper() or DEFAULT_MODEL_ID
    if mid not in MODEL_CATALOG:
        raise ValueError(
            f"Unknown model_id {model_id!r}. Choose one of: {', '.join(MODEL_CATALOG)}"
        )
    path = resolve_checkpoint_path(mid)
    if not path:
        raise ValueError(
            f"Model {mid} not found. For A1 expected outputs/ablation/A1 or "
            f"outputs/full_8label_aug_* trained on {_FEAR_AUG_MARKER}; "
            f"for A4 expected outputs/ablation/A4 (or seed_42/) "
            f"or Models/parsbert_emotion_incremental (allAug)."
        )
    return {"model_id": mid, "model_path": path}
