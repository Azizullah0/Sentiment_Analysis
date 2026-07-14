"""
Prepare fixed holdout and filtered training splits for valid confidence experiments.

Splits the original 400K dataset once (seed=42, test_size=0.2), matching the
ablation protocol. Confidence filtering is applied ONLY to the 80% training pool;
the 20% eval holdout stays unfiltered.

Requires Combined_Labeled_Dataset_scored.csv from score_pseudo_labels.py.

Example:
    python scripts/prepare_confidence_splits.py
    python scripts/prepare_confidence_splits.py --threshold 0.7 0.8 0.9
"""

import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import PATHS, train_filtered_confidence_path


def filter_dataframe(df, threshold, require_agreement):
    mask = df["pseudo_confidence"] >= threshold
    if require_agreement:
        mask &= df["label_agrees"]
    return df[mask].copy()


def drop_scoring_columns(df):
    scoring_columns = ["pseudo_confidence", "pseudo_predicted_id", "label_agrees"]
    return df.drop(columns=[col for col in scoring_columns if col in df.columns])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create fixed eval holdout and filtered train splits for confidence experiments."
    )
    parser.add_argument(
        "--original",
        default=PATHS["Combined_Labeled_Dataset"],
        help="Unfiltered pseudo-labeled dataset.",
    )
    parser.add_argument(
        "--scored",
        default=PATHS["Combined_Labeled_Dataset_scored"],
        help="Scored dataset from score_pseudo_labels.py.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        nargs="+",
        default=[0.7, 0.8, 0.9],
        help="Confidence thresholds for filtered train splits.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Eval holdout fraction (default 0.2, matches train.py).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Split seed (default 42, matches train.py).",
    )
    parser.add_argument(
        "--require-agreement",
        action="store_true",
        default=True,
        help="Filter by label agreement (default: on).",
    )
    parser.add_argument(
        "--no-require-agreement",
        dest="require_agreement",
        action="store_false",
        help="Filter by confidence only.",
    )
    parser.add_argument(
        "--label-column",
        default="label_id",
        help="Label column for stratified split.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PATHS["data_root"], "Data/processed"),
        help="Directory for output CSVs.",
    )
    return parser.parse_args()


def align_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "clean" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"clean": "text"})
    return df


def main():
    args = parse_args()
    original_path = os.path.abspath(os.path.expanduser(args.original))
    scored_path = os.path.abspath(os.path.expanduser(args.scored))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    for path in (original_path, scored_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    original = align_columns(pd.read_csv(original_path))
    scored = align_columns(pd.read_csv(scored_path))

    if args.label_column not in original.columns:
        raise ValueError(f"Original dataset missing label column: {args.label_column}")
    if len(original) != len(scored):
        raise ValueError(
            f"Row count mismatch: original={len(original):,}, scored={len(scored):,}. "
            "Re-run score_pseudo_labels.py on the same Combined_Labeled_Dataset.csv."
        )

    original = original.dropna(subset=[args.label_column])
    scored = scored.dropna(subset=[args.label_column])

    train_pool, eval_holdout = train_test_split(
        original,
        test_size=args.test_size,
        stratify=original[args.label_column],
        random_state=args.random_state,
    )
    scored_train, scored_eval = train_test_split(
        scored,
        test_size=args.test_size,
        stratify=scored[args.label_column],
        random_state=args.random_state,
    )

    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, "eval_holdout_original.csv")
    train_pool_path = os.path.join(output_dir, "train_pool_original.csv")

    eval_holdout.to_csv(eval_path, index=False, encoding="utf-8")
    train_pool.to_csv(train_pool_path, index=False, encoding="utf-8")

    print("Fixed split created (same protocol as ablation runs):")
    print(f"  Original rows: {len(original):,}")
    print(f"  Train pool:    {len(train_pool):,} -> {train_pool_path}")
    print(f"  Eval holdout:  {len(eval_holdout):,} -> {eval_path}")
    print(f"  Seed: {args.random_state} | test_size: {args.test_size}")

    for threshold in args.threshold:
        filtered_train = filter_dataframe(scored_train, threshold, args.require_agreement)
        training_df = drop_scoring_columns(filtered_train)
        out_path = train_filtered_confidence_path(threshold, output_dir)
        training_df.to_csv(out_path, index=False, encoding="utf-8")
        retained = len(training_df)
        print(
            f"\nThreshold >= {threshold} (train pool only, require_agreement={args.require_agreement})"
        )
        print(f"  Retained: {retained:,} / {len(scored_train):,} ({retained / len(scored_train):.1%})")
        print(f"  Output: {out_path}")

    print("\nTrain with fixed holdout:")
    print("  python scripts/run_confidence_experiments.py --run-id C1")


if __name__ == "__main__":
    main()
