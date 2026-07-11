"""
Back-translation augmentation for minority emotion classes in Afghan Persian (Dari).

Round-trip translation (Dari -> English -> Dari) using NLLB adds lexical paraphrase
diversity complementary to template augmentation. Source texts come from the 4K gold
labeled dataset, not from generated templates.

Output schema matches emotion_augmenter.py / fear_augmenter.py:
    channelId, publishedAt, clean, token_count, Label, label_id

Usage:
    python augmentations/back_translation_augmenter.py --emotion surprise --n 100 --spot-check-only
    python augmentations/back_translation_augmenter.py --emotions surprise,anger --n 2000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from augmentations.emotion_augmenter import distinct_n, normalize_for_dedup
from config.paths import PATHS, augmented_bt_path

MODEL_ID = "facebook/nllb-200-distilled-600M"
SRC_LANG = "pes_Arab"
TGT_LANG = "eng_Latn"

LABEL_IDS = {
    "Surprise": 3,
    "Disgust": 4,
    "Sad": 5,
    "Anger": 6,
    "Fear": 7,
}

EMOTION_ALIASES = {
    "surprise": "Surprise",
    "disgust": "Disgust",
    "sad": "Sad",
    "anger": "Anger",
    "fear": "Fear",
}

MIN_TOKENS = 3
MIN_LENGTH_RATIO = 0.5
MAX_LENGTH_RATIO = 2.0
MAX_CHAR_OVERLAP = 0.95
DEFAULT_BATCH_SIZE = 8
SPOTCHECK_COUNT = 100


@dataclass
class FilterStats:
    too_short: int = 0
    exact_duplicate: int = 0
    length_ratio: int = 0
    char_overlap: int = 0
    empty_translation: int = 0
    accepted: int = 0

    def to_dict(self) -> dict:
        return {
            "too_short": self.too_short,
            "exact_duplicate": self.exact_duplicate,
            "length_ratio": self.length_ratio,
            "char_overlap": self.char_overlap,
            "empty_translation": self.empty_translation,
            "accepted": self.accepted,
        }


def char_overlap_ratio(source: str, candidate: str) -> float:
    source_chars = set(source.replace(" ", ""))
    candidate_chars = set(candidate.replace(" ", ""))
    if not source_chars or not candidate_chars:
        return 0.0
    return len(source_chars & candidate_chars) / max(len(source_chars), len(candidate_chars))


def token_count(text: str) -> int:
    return len(text.split())


def passes_filters(source: str, back_translated: str, stats: FilterStats) -> bool:
    back_translated = back_translated.strip()
    if not back_translated:
        stats.empty_translation += 1
        return False

    if token_count(back_translated) < MIN_TOKENS:
        stats.too_short += 1
        return False

    if normalize_for_dedup(source) == normalize_for_dedup(back_translated):
        stats.exact_duplicate += 1
        return False

    source_len = max(len(source), 1)
    ratio = len(back_translated) / source_len
    if ratio < MIN_LENGTH_RATIO or ratio > MAX_LENGTH_RATIO:
        stats.length_ratio += 1
        return False

    if char_overlap_ratio(source, back_translated) > MAX_CHAR_OVERLAP:
        stats.char_overlap += 1
        return False

    stats.accepted += 1
    return True


def to_schema(rows: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [
        {
            "channelId": f"bt_{row['label'].lower()}",
            "publishedAt": now,
            "clean": row["text"],
            "token_count": token_count(row["text"]),
            "Label": row["label"],
            "label_id": row["label_id"],
            "source_text": row.get("source_text", ""),
        }
        for row in rows
    ]


def load_source_pool(source_dataset: str, label: str) -> list[str]:
    path = Path(source_dataset)
    if not path.exists():
        raise FileNotFoundError(f"Source dataset not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "clean" not in df.columns or "Label" not in df.columns:
        raise ValueError(f"{path} must contain 'clean' and 'Label' columns")

    pool = (
        df.loc[df["Label"].astype(str).str.strip() == label, "clean"]
        .dropna()
        .astype(str)
        .map(str.strip)
        .tolist()
    )
    pool = [text for text in pool if text]
    if not pool:
        raise ValueError(f"No source rows found for label '{label}' in {path}")
    return pool


class BackTranslationEngine:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=256,
                num_beams=4,
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def back_translate_batch(self, texts: list[str]) -> list[str]:
        english = self._translate_batch(texts, SRC_LANG, TGT_LANG)
        return self._translate_batch(english, TGT_LANG, SRC_LANG)


def sample_sources(pool: list[str], n: int, rng: random.Random) -> tuple[list[str], bool]:
    if len(pool) >= n:
        return rng.sample(pool, n), False
    return [rng.choice(pool) for _ in range(n)], True


def generate_for_emotion(
    emotion_key: str,
    n: int,
    seed: int,
    source_dataset: str,
    output: str | None,
    spot_check_only: bool,
    batch_size: int,
    engine: BackTranslationEngine | None = None,
) -> Path:
    label = EMOTION_ALIASES[emotion_key]
    label_id = LABEL_IDS[label]
    target_n = SPOTCHECK_COUNT if spot_check_only else n

    rng = random.Random(seed)
    pool = load_source_pool(source_dataset, label)
    sources, upsampled = sample_sources(pool, target_n, rng)

    owned_engine = engine is None
    if owned_engine:
        engine = BackTranslationEngine()

    accepted_rows: list[dict] = []
    seen_keys: set[str] = set()
    stats = FilterStats()
    attempts = 0
    max_attempts = target_n * 4

    print(
        f"\nBack-translating {target_n} {label} samples "
        f"(pool={len(pool)}, upsampled={upsampled}, spot_check={spot_check_only})..."
    )

    source_index = 0
    while len(accepted_rows) < target_n and attempts < max_attempts:
        batch_sources = []
        while len(batch_sources) < batch_size and source_index < len(sources):
            batch_sources.append(sources[source_index])
            source_index += 1
        if not batch_sources:
            extra, _ = sample_sources(pool, batch_size, rng)
            batch_sources = extra

        attempts += len(batch_sources)
        back_translated = engine.back_translate_batch(batch_sources)

        for source_text, bt_text in zip(batch_sources, back_translated):
            if len(accepted_rows) >= target_n:
                break
            if not passes_filters(source_text, bt_text, stats):
                continue
            dedup_key = normalize_for_dedup(bt_text)
            if dedup_key in seen_keys:
                stats.exact_duplicate += 1
                continue
            seen_keys.add(dedup_key)
            accepted_rows.append(
                {
                    "text": bt_text.strip(),
                    "label": label,
                    "label_id": label_id,
                    "source_text": source_text,
                }
            )

    if len(accepted_rows) < target_n:
        print(
            f"Warning: accepted {len(accepted_rows)} samples (target {target_n}). "
            "Consider lowering filter thresholds or enlarging the source pool."
        )

    texts = [row["text"] for row in accepted_rows]
    d1, d2 = distinct_n(texts, 1), distinct_n(texts, 2)
    print(f"Accepted: {len(accepted_rows)} | distinct-1={d1:.3f} distinct-2={d2:.3f}")
    print(f"Filter stats: {stats.to_dict()}")

    if spot_check_only:
        out_path = Path(
            output
            or os.path.join(
                PATHS["bt_spotcheck_dir"],
                f"bt_spotcheck_{emotion_key}.csv",
            )
        )
    else:
        out_path = Path(output or augmented_bt_path(emotion_key, n))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(to_schema(accepted_rows)).to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved to {out_path}")

    metadata = {
        "generator": "back_translation_augmenter.py",
        "model_id": MODEL_ID,
        "src_lang": SRC_LANG,
        "tgt_lang": TGT_LANG,
        "label": label,
        "label_id": label_id,
        "emotion_key": emotion_key,
        "n_requested": target_n,
        "n_accepted": len(accepted_rows),
        "seed": seed,
        "source_dataset": str(source_dataset),
        "source_pool_size": len(pool),
        "upsampled": upsampled,
        "spot_check_only": spot_check_only,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "filter_stats": stats.to_dict(),
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "filters": {
            "min_tokens": MIN_TOKENS,
            "min_length_ratio": MIN_LENGTH_RATIO,
            "max_length_ratio": MAX_LENGTH_RATIO,
            "max_char_overlap": MAX_CHAR_OVERLAP,
        },
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Metadata sidecar: {meta_path}")

    print(f"\nSample back-translated {label} texts:\n" + "-" * 60)
    for row in accepted_rows[:5]:
        print("-", row["text"])
    return out_path


def parse_emotions(value: str) -> list[str]:
    emotions = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(emotions) - set(EMOTION_ALIASES))
    if invalid:
        raise ValueError(f"Invalid emotion(s): {invalid}. Valid: {sorted(EMOTION_ALIASES)}")
    return emotions


def run(
    emotions: list[str],
    n: int,
    seed: int,
    source_dataset: str,
    output: str | None,
    spot_check_only: bool,
    batch_size: int,
) -> list[Path]:
    engine = BackTranslationEngine()
    outputs = []
    for index, emotion in enumerate(emotions):
        emotion_seed = seed + index
        emotion_output = output if len(emotions) == 1 else None
        path = generate_for_emotion(
            emotion_key=emotion,
            n=n,
            seed=emotion_seed,
            source_dataset=source_dataset,
            output=emotion_output,
            spot_check_only=spot_check_only,
            batch_size=batch_size,
            engine=engine,
        )
        outputs.append(path)
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="NLLB back-translation augmentation from 4K gold seed texts."
    )
    parser.add_argument(
        "--emotion",
        choices=[*EMOTION_ALIASES.keys()],
        help="Single emotion to augment.",
    )
    parser.add_argument(
        "--emotions",
        type=str,
        default=None,
        help="Comma-separated emotions (surprise,anger,disgust,sad,fear).",
    )
    parser.add_argument("--n", type=int, default=2000, help="Target accepted samples per emotion.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--source-dataset",
        default=PATHS["Labeled_4K"],
        help="Gold labeled seed CSV (default: Labeled_4K.csv).",
    )
    parser.add_argument(
        "--spot-check-only",
        action="store_true",
        help=f"Generate only {SPOTCHECK_COUNT} samples per emotion for human review.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV (single emotion only).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Translation batch size.",
    )
    args = parser.parse_args()

    if args.emotion:
        emotions = [args.emotion]
    elif args.emotions:
        emotions = parse_emotions(args.emotions)
    else:
        emotions = list(EMOTION_ALIASES.keys())

    if args.output and len(emotions) > 1:
        parser.error("--output can only be used with a single --emotion")

    run(
        emotions=emotions,
        n=args.n,
        seed=args.seed,
        source_dataset=args.source_dataset,
        output=args.output,
        spot_check_only=args.spot_check_only,
        batch_size=args.batch_size,
    )
    print("\nDONE.\n")


if __name__ == "__main__":
    main()
