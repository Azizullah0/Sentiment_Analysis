"""Deployment EmotionPredictor: Excluded (pre-filter) vs Others (abstain)."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths import PATHS  # noqa: E402

from deployment.preprocess import classify_text  # noqa: E402

LABEL_MAP = {
    0: "Hope",
    1: "Happy",
    2: "Neutral",
    3: "Surprise",
    4: "Disgust",
    5: "Sad",
    6: "Anger",
    7: "Fear",
}

DEFAULT_MIN_CONFIDENCE = 0.50
DEFAULT_MAX_LENGTH = 256
OTHERS_LABEL = "Others"
EXCLUDED_LABEL = "Excluded"


def resolve_model_path(explicit: Optional[str] = None) -> str:
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model path does not exist: {path}")
        return path

    candidates = [
        PATHS.get("ablation_a4_model"),
        PATHS.get("incremental_finetuned_model"),
        PATHS.get("fine_tuned_model"),
        PATHS.get("parsbert_emotion"),
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            if os.path.isfile(os.path.join(candidate, "config.json")):
                return candidate
    for candidate in candidates:
        if candidate:
            return candidate
    raise FileNotFoundError("No model path configured in PATHS.")


def excluded_result(
    text: Optional[str],
    cleaned: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "text": text,
        "text_clean": cleaned,
        "label": EXCLUDED_LABEL,
        "raw_emotion": "",
        "confidence": 0.0,
        "abstain": True,
        "abstain_reason": reason,
        "all_probabilities": {},
    }


def apply_confidence_gate(
    raw_emotion: str,
    confidence: float,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    all_probabilities: Optional[Dict[str, float]] = None,
    text: Optional[str] = None,
) -> Dict[str, Any]:
    """Map low-confidence predictions to Others (usable text only)."""
    abstain = confidence < float(min_confidence)
    label = OTHERS_LABEL if abstain else raw_emotion
    return {
        "text": text,
        "label": label,
        "raw_emotion": raw_emotion,
        "confidence": float(confidence),
        "abstain": abstain,
        "abstain_reason": "low_confidence" if abstain else None,
        "all_probabilities": all_probabilities or {},
    }


class EmotionPredictor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        self.model_path = resolve_model_path(model_path)
        self.max_length = max_length
        self.min_confidence = float(min_confidence)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"Model not found at: {self.model_path}\n"
                "Train A4 / place weights under Models/parsbert_emotion_incremental "
                "or outputs/ablation/A4, or pass --model-path."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        ).to(self.device)
        self.model.eval()

    def _predict_raw_batch(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)

        preds = probs.argmax(dim=1).cpu().numpy()
        results = []
        for i, original in enumerate(texts):
            idx = int(preds[i])
            conf = float(probs[i].max().cpu())
            raw_emotion = LABEL_MAP.get(idx, f"Label_{idx}")
            all_probs = {
                LABEL_MAP.get(j, f"Label_{j}"): float(probs[i][j])
                for j in range(self.model.config.num_labels)
            }
            results.append(
                {
                    "text": original,
                    "raw_emotion": raw_emotion,
                    "confidence": conf,
                    "all_probabilities": all_probs,
                }
            )
        return results

    def predict(
        self,
        text: Union[str, Sequence[str]],
        min_confidence: Optional[float] = None,
        apply_preprocess: bool = True,
        skip_unusable: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predict emotion(s).
        - Excluded: non-Persian / emoji-only / too short (no model call)
        - Others: usable text with confidence below threshold
        """
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        threshold = (
            self.min_confidence if min_confidence is None else float(min_confidence)
        )

        meta: List[Dict[str, Any]] = []
        for t in texts:
            if apply_preprocess:
                cleaned, usable, reason = classify_text(t)
            else:
                cleaned = (t or "").strip()
                usable = bool(cleaned)
                reason = None if usable else "empty"
            meta.append(
                {
                    "original": t,
                    "cleaned": cleaned,
                    "usable": usable,
                    "reason": reason,
                }
            )

        usable_indices = [i for i, m in enumerate(meta) if m["usable"]]
        usable_texts = [meta[i]["cleaned"] for i in usable_indices]
        raw_by_index: Dict[int, Dict[str, Any]] = {}
        if usable_texts:
            raw_batch = self._predict_raw_batch(usable_texts)
            for idx, raw in zip(usable_indices, raw_batch):
                raw_by_index[idx] = raw

        outputs: List[Dict[str, Any]] = []
        for i, m in enumerate(meta):
            if not m["usable"]:
                if skip_unusable:
                    continue
                outputs.append(
                    excluded_result(
                        m["original"], m["cleaned"], m["reason"] or "unusable_text"
                    )
                )
                continue

            raw = raw_by_index[i]
            gated = apply_confidence_gate(
                raw_emotion=raw["raw_emotion"],
                confidence=raw["confidence"],
                min_confidence=threshold,
                all_probabilities=raw["all_probabilities"],
                text=m["original"],
            )
            gated["text_clean"] = m["cleaned"]
            outputs.append(gated)

        return outputs[0] if single else outputs


def append_review_queue(
    record: Dict[str, Any],
    queue_path: str,
    emotions=("Fear", "Anger"),
) -> bool:
    import json

    label = record.get("label")
    if label not in emotions:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(queue_path)) or ".", exist_ok=True)
    with open(queue_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True
