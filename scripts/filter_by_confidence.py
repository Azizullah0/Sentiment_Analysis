"""
Filter a scored pseudo-labeled dataset by confidence threshold.

Run score_pseudo_labels.py first and review the printed diagnostics before
choosing thresholds.

Example:
    python scripts/filter_by_confidence.py --threshold 0.8
    python scripts/filter_by_confidence.py --threshold 0.7 0.8 0.9
    python scripts/filter_by_confidence.py --threshold 0.8 --no-require-agreement
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import PATHS, confidence_filtered_path


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
    parser = argparse.ArgumentParser(description="Filter scored pseudo-labels by confidence.")
    parser.add_argument(
        "--input",
        default=PATHS["Combined_Labeled_Dataset_scored"],
        help="Scored CSV from score_pseudo_labels.py.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=[0.7, 0.8, 0.9],
        help="One or more confidence thresholds (default: 0.7 0.8 0.9).",
    )
    parser.add_argument(
        "--require-agreement",
        action="store_true",
        default=True,
        help="Keep rows where pseudo_predicted_id matches label_id (default: on).",
    )
    parser.add_argument(
        "--no-require-agreement",
        dest="require_agreement",
        action="store_false",
        help="Filter by confidence only, ignore label agreement.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PATHS["data_root"], "Data/processed"),
        help="Directory for filtered CSV outputs.",
    )
    parser.add_argument(
        "--label-column",
        default="label_id",
        help="Label column to preserve in output.",
    )
    return parser.parse_args()


def filter_dataframe(df, threshold, require_agreement):
    mask = df["pseudo_confidence"] >= threshold
    if require_agreement:
        mask &= df["label_agrees"]
    return df[mask].copy()


def drop_scoring_columns(df):
    scoring_columns = ["pseudo_confidence", "pseudo_predicted_id", "label_agrees"]
    return df.drop(columns=[col for col in scoring_columns if col in df.columns])


def print_filter_summary(original_df, filtered_df, threshold, require_agreement, output_path, label_column):
    total = len(original_df)
    retained = len(filtered_df)
    print(f"\nThreshold >= {threshold} (require_agreement={require_agreement})")
    print(f"  Retained: {retained:,} / {total:,} ({retained / total:.1%})")
    print(f"  Output: {output_path}")

    if retained == 0:
        print("  WARNING: no rows retained at this threshold.")
        return

    if label_column in filtered_df.columns:
        print("  Label distribution (retained):")
        counts = filtered_df[label_column].value_counts().sort_index()
        for label_id, count in counts.items():
            label_name = LABEL_NAMES.get(int(label_id), f"Label_{label_id}")
            print(f"    {label_name} ({int(label_id)}): {count:,}")


def main():
    args = parse_args()
    input_path = os.path.abspath(os.path.expanduser(args.input))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Scored dataset not found: {input_path}\n"
            "Run score_pseudo_labels.py first."
        )

    required = {"pseudo_confidence", "pseudo_predicted_id", "label_agrees", args.label_column}
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Scored CSV missing columns: {sorted(missing)}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Input: {input_path} ({len(df):,} rows)")
    print(f"Output directory: {output_dir}")

    for threshold in args.threshold:
        filtered = filter_dataframe(df, threshold, args.require_agreement)
        training_df = drop_scoring_columns(filtered)
        output_path = confidence_filtered_path(threshold, output_dir)
        training_df.to_csv(output_path, index=False, encoding="utf-8")
        print_filter_summary(df, filtered, threshold, args.require_agreement, output_path, args.label_column)

    print("\nReview retained counts above, then train with run_confidence_experiments.py.")


if __name__ == "__main__":
    main()
