"""
Compute the confusion matrix of a trained ablation model on its own fixed
stratified test split (the same split train.py used: test_size=0.2,
random_state=42, stratified by label_id).

No retraining happens; the saved checkpoint is only evaluated.

Example:
    python scripts/evaluate_confusion_matrix.py --run-id A4
    python scripts/evaluate_confusion_matrix.py --model outputs/ablation/A4 \
        --dataset Data/processed/Combined_Labeled_Dataset_with_allAug.csv
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import PATHS


LABEL_NAMES = ["Hope", "Happy", "Neutral", "Surprise", "Disgust", "Sad", "Anger", "Fear"]

RUN_DATASETS = {
    "A0": PATHS["Combined_Labeled_Dataset"],
    "A1": PATHS["Combined_Labeled_Dataset_with_fearAug"],
    "A4": PATHS["Combined_Labeled_Dataset_with_allAug"],
    "A5": PATHS["Combined_Labeled_Dataset_with_allAug_bt"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Confusion matrix on the fixed ablation test split.")
    parser.add_argument("--run-id", default="A4", choices=sorted(RUN_DATASETS.keys()))
    parser.add_argument("--model", default=None, help="Model dir (default: outputs/ablation/<run-id>).")
    parser.add_argument("--dataset", default=None, help="Dataset CSV (default: the run's ablation dataset).")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output", default=None, help="JSON output (default: <model>/confusion_matrix.json).")
    return parser.parse_args()


def resolve_model_dir(path):
    """Accept a run folder, a checkpoint-<step> folder, or a seed_42 subfolder."""
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    checkpoints = sorted(
        glob.glob(os.path.join(path, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
    )
    for candidate in reversed(checkpoints):
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    seed_dir = os.path.join(path, "seed_42")
    if os.path.isdir(seed_dir):
        return resolve_model_dir(seed_dir)
    raise FileNotFoundError(f"No config.json found under: {path}")


def main():
    args = parse_args()
    dataset_path = args.dataset or RUN_DATASETS[args.run_id]
    model_root = args.model or os.path.join(PATHS["ablation_outputs"], args.run_id)
    model_dir = resolve_model_dir(os.path.abspath(os.path.expanduser(model_root)))
    output_path = args.output or os.path.join(model_root, "confusion_matrix.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_dir}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")

    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()
    text_col = "clean" if "clean" in df.columns else "text"
    df = df.dropna(subset=[text_col, "label_id"])
    df["label_id"] = df["label_id"].astype(int)

    # Identical call to train.py's split_train_eval: only the eval side is used.
    _, eval_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["label_id"],
        random_state=args.random_state,
    )
    print(f"Test split: {len(eval_df):,} rows (of {len(df):,})")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    texts = eval_df[text_col].astype(str).tolist()
    true_labels = eval_df["label_id"].tolist()
    preds = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        if (start // args.batch_size) % 100 == 0:
            print(f"  {start:,}/{len(texts):,}")

    cm_counts = confusion_matrix(true_labels, preds, labels=list(range(len(LABEL_NAMES))))
    cm_rownorm = cm_counts / cm_counts.sum(axis=1, keepdims=True)

    print(f"\nAccuracy: {accuracy_score(true_labels, preds):.4f}")
    print(f"Macro-F1: {f1_score(true_labels, preds, average='macro', zero_division=0):.4f}")
    header = " " * 10 + "".join(f"{n[:5]:>7}" for n in LABEL_NAMES)
    print("\nRow-normalized confusion matrix (rows = true label):")
    print(header)
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name:>9} " + "".join(f"{cm_rownorm[i, j]:7.2f}" for j in range(len(LABEL_NAMES))))

    result = {
        "run_id": args.run_id,
        "model_dir": model_dir,
        "dataset_path": dataset_path,
        "split": {"test_size": args.test_size, "random_state": args.random_state, "stratify": "label_id"},
        "eval_samples": len(eval_df),
        "accuracy": float(accuracy_score(true_labels, preds)),
        "f1_macro": float(f1_score(true_labels, preds, average="macro", zero_division=0)),
        "labels": LABEL_NAMES,
        "confusion_counts": cm_counts.tolist(),
        "confusion_rownorm": np.round(cm_rownorm, 4).tolist(),
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
