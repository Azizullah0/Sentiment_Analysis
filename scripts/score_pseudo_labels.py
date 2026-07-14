"""
Score existing pseudo-labels with the 4K seed model (parsbert_emotion).

Adds pseudo_confidence, pseudo_predicted_id, and label_agrees columns so you
can review label quality before filtering or retraining.

Example:
    python scripts/score_pseudo_labels.py
    python scripts/score_pseudo_labels.py --batch-size 64 --max-rows 10000
"""

import argparse
import os
import sys

import pandas as pd
import torch
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
    parser = argparse.ArgumentParser(description="Score pseudo-labels with the seed model.")
    parser.add_argument(
        "--input",
        default=PATHS["Combined_Labeled_Dataset"],
        help="Pseudo-labeled CSV to score.",
    )
    parser.add_argument(
        "--output",
        default=PATHS["Combined_Labeled_Dataset_scored"],
        help="Path for scored output CSV.",
    )
    parser.add_argument(
        "--model-path",
        default=PATHS["parsbert_emotion"],
        help="Seed model used for scoring (default: 4K parsbert_emotion).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer max length.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for quick smoke tests.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Text column name (auto-detected: clean or text).",
    )
    parser.add_argument(
        "--label-column",
        default="label_id",
        help="Existing pseudo-label column.",
    )
    return parser.parse_args()


def resolve_text_column(df, text_column):
    if text_column:
        if text_column not in df.columns:
            raise ValueError(f"Text column not found: {text_column}")
        return text_column
    if "clean" in df.columns:
        return "clean"
    if "text" in df.columns:
        return "text"
    raise ValueError("Dataset must contain a 'clean' or 'text' column.")


def load_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train baseline_4k first or pass --model-path."
        )

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return tokenizer, model


def score_batches(texts, tokenizer, model, device, batch_size, max_length):
    confidences = []
    predicted_ids = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            batch_conf, batch_pred = probs.max(dim=1)

        confidences.extend(batch_conf.cpu().tolist())
        predicted_ids.extend(batch_pred.cpu().tolist())

        if (start // batch_size) % 100 == 0 and start > 0:
            print(f"  Scored {start}/{len(texts)} rows...")

    return confidences, predicted_ids


def print_diagnostics(df, label_column):
    total = len(df)
    agreement = df["label_agrees"].mean()
    print(f"\n{'=' * 60}")
    print("Scoring diagnostics")
    print(f"{'=' * 60}")
    print(f"Total rows: {total:,}")
    print(f"Overall agreement (predicted == {label_column}): {agreement:.2%}")

    print("\nConfidence summary:")
    print(df["pseudo_confidence"].describe().to_string())

    print("\nAgreement by existing label:")
    by_label = (
        df.groupby(label_column)
        .agg(
            count=("label_agrees", "size"),
            agreement_rate=("label_agrees", "mean"),
            mean_confidence=("pseudo_confidence", "mean"),
        )
        .reset_index()
    )
    for _, row in by_label.iterrows():
        label_id = int(row[label_column])
        label_name = LABEL_NAMES.get(label_id, f"Label_{label_id}")
        print(
            f"  {label_name} ({label_id}): "
            f"agreement={row['agreement_rate']:.2%}, "
            f"mean_conf={row['mean_confidence']:.3f}, "
            f"n={int(row['count']):,}"
        )

    print("\nSuggested threshold preview (agreement + confidence filter):")
    for threshold in (0.7, 0.8, 0.9):
        mask = (df["pseudo_confidence"] >= threshold) & df["label_agrees"]
        retained = mask.sum()
        print(f"  conf >= {threshold} & agrees: {retained:,} rows ({retained / total:.1%})")

    print(f"\n{'=' * 60}")
    print("Review the summary above before running prepare_confidence_splits.py.")
    print(f"{'=' * 60}")


def main():
    args = parse_args()
    input_path = os.path.abspath(os.path.expanduser(args.input))
    output_path = os.path.abspath(os.path.expanduser(args.output))

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()

    if args.label_column not in df.columns:
        raise ValueError(f"Label column not found: {args.label_column}")

    text_column = resolve_text_column(df, args.text_column)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
        print(f"Limited to first {args.max_rows} rows for smoke test.")

    texts = df[text_column].fillna("").astype(str).tolist()
    print(f"Scoring {len(texts):,} rows from: {input_path}")

    tokenizer, model = load_model(args.model_path, device)
    confidences, predicted_ids = score_batches(
        texts, tokenizer, model, device, args.batch_size, args.max_length
    )

    df["pseudo_confidence"] = confidences
    df["pseudo_predicted_id"] = predicted_ids
    df["label_agrees"] = df["pseudo_predicted_id"] == df[args.label_column]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved scored dataset: {output_path}")

    print_diagnostics(df, args.label_column)


if __name__ == "__main__":
    main()
