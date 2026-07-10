"""
Evaluate a trained model on the fixed original holdout set.

Use this to get valid metrics for confidence experiments (e.g. re-evaluate C1
without retraining after discovering filtered-split leakage).

Example:
    python scripts/evaluate_holdout.py --model outputs/confidence/C1
    python scripts/evaluate_holdout.py --model outputs/confidence/C1 --holdout Data/processed/eval_holdout_original.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import PATHS


LABEL_NAMES = {
    0: "Hope",
    1: "Happy",
    2: "Neutral",
    3: "Surprise",
    4: "Disgust",
    5: "Sad",
    6: "Anger",
    7: "Fear",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the fixed holdout CSV.")
    parser.add_argument("--model", required=True, help="Path to saved model directory.")
    parser.add_argument(
        "--holdout",
        default=PATHS["eval_holdout_original"],
        help="Fixed eval holdout CSV (unfiltered original 20%% split).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--text-column", default=None, help="Auto-detect clean or text.")
    parser.add_argument("--label-column", default="label_id")
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path (default: <model>/holdout_eval.json).",
    )
    return parser.parse_args()


def resolve_text_column(df, text_column):
    if text_column:
        return text_column
    if "text" in df.columns:
        return "text"
    if "clean" in df.columns:
        return "clean"
    raise ValueError("Holdout CSV must contain 'text' or 'clean' column.")


def load_holdout(path, text_column, label_column):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    text_col = resolve_text_column(df, text_column)
    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column}")
    df = df.dropna(subset=[text_col, label_column])
    return df, text_col


def predict_batches(model, tokenizer, texts, device, batch_size, max_length):
    all_preds = []
    all_probs = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
        all_preds.extend(probs.argmax(dim=1).cpu().tolist())
        all_probs.extend(probs.max(dim=1).values.cpu().tolist())

    return all_preds, all_probs


def main():
    args = parse_args()
    model_path = os.path.abspath(os.path.expanduser(args.model))
    holdout_path = os.path.abspath(os.path.expanduser(args.holdout))
    output_path = args.output or os.path.join(model_path, "holdout_eval.json")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(
            f"Holdout not found: {holdout_path}\n"
            "Run prepare_confidence_splits.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_path}")
    print(f"Holdout: {holdout_path}")
    print(f"Device: {device}")

    df, text_col = load_holdout(holdout_path, args.text_column, args.label_column)
    texts = df[text_col].fillna("").astype(str).tolist()
    true_labels = df[args.label_column].astype(int).tolist()
    print(f"Evaluating {len(texts):,} holdout samples...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    pred_labels, confidences = predict_batches(
        model, tokenizer, texts, device, args.batch_size, args.max_length
    )

    metrics = {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "f1_macro": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0)),
        "precision": float(precision_score(true_labels, pred_labels, average="weighted", zero_division=0)),
        "recall": float(recall_score(true_labels, pred_labels, average="weighted", zero_division=0)),
    }
    per_class_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)

    print("\nHoldout results (valid — unfiltered original test set):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("Per-class F1:")
    for i, score in enumerate(per_class_f1):
        print(f"  {LABEL_NAMES.get(i, i)}: {score:.4f}")

    result = {
        "model_path": model_path,
        "holdout_path": holdout_path,
        "eval_protocol": "fixed_holdout_original",
        "eval_samples": len(df),
        "metrics": metrics,
        "per_class_f1": {LABEL_NAMES.get(i, str(i)): float(score) for i, score in enumerate(per_class_f1)},
        "mean_confidence": float(np.mean(confidences)),
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
