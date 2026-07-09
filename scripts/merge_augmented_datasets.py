"""Merge fear-augmented and newly generated emotion augmentation datasets."""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import PATHS


REQUIRED_COLUMNS = {"clean", "Label", "label_id"}

DEFAULT_AUGMENTED_EMOTIONS = ["surprise", "disgust", "anger"]
VALID_EMOTIONS = set(DEFAULT_AUGMENTED_EMOTIONS)


def parse_emotions(value):
    if not value:
        return list(DEFAULT_AUGMENTED_EMOTIONS)

    emotions = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(emotions) - VALID_EMOTIONS)
    if invalid:
        raise ValueError(
            f"Invalid emotion(s): {invalid}. Valid options: {sorted(VALID_EMOTIONS)}"
        )
    return emotions


def augmented_files_for_emotions(emotions, n):
    return [processed_path(f"augmented_afghan_{emotion}_{n}.csv") for emotion in emotions]


def processed_path(filename):
    return os.path.join(PATHS["data_root"], "Data/processed", filename)


def default_augmented_files(n, emotions=None):
    return augmented_files_for_emotions(emotions or DEFAULT_AUGMENTED_EMOTIONS, n)


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    text = (
        text.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("ۀ", "ه")
        .replace("\u200c", " ")
        .replace("،", ",")
    )
    return " ".join(text.split())


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Required dataset not found: "
            f"{path}\n"
            "Set SENTIMENT_DATA_ROOT/SENTIMENT_BASE_PATH to your Google Drive "
            "Sentiment_Analysis folder, or pass explicit --base-dataset and "
            "--augmented-file paths."
        )


def load_dataset(path):
    require_file(path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return df


def align_to_base_schema(df, base_columns):
    aligned = df.copy()

    if "token_count" in base_columns:
        aligned["token_count"] = aligned["clean"].fillna("").astype(str).str.split().str.len()

    for column in base_columns:
        if column not in aligned.columns:
            aligned[column] = pd.NA

    return aligned[base_columns]


def print_distribution(df, label):
    print(f"\n{label}")
    print(f"Rows: {len(df)}")
    if {"Label", "label_id"}.issubset(df.columns):
        distribution = (
            df.groupby(["label_id", "Label"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["label_id", "Label"])
        )
        print(distribution.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge Combined_Labeled_Dataset_with_fearAug.csv with the generated "
            "Surprise, Disgust, and Anger augmentation CSVs."
        )
    )
    parser.add_argument(
        "--base-dataset",
        default=PATHS["Combined_Labeled_Dataset_with_fearAug"],
        help=(
            "Path to the latest fear-augmented combined dataset. Defaults to "
            "PATHS['Combined_Labeled_Dataset_with_fearAug']."
        ),
    )
    parser.add_argument(
        "--output",
        default=PATHS["Combined_Labeled_Dataset_with_allAug"],
        help=(
            "Path for the merged all-augmentation CSV. Defaults to "
            "PATHS['Combined_Labeled_Dataset_with_allAug']."
        ),
    )
    parser.add_argument(
        "--emotions",
        default=",".join(DEFAULT_AUGMENTED_EMOTIONS),
        help=(
            "Comma-separated subset of augmentation emotions to merge "
            "(surprise, disgust, anger). Example: --emotions surprise"
        ),
    )
    parser.add_argument(
        "--augmented-file",
        action="append",
        dest="augmented_files",
        help=(
            "Extra/generated augmentation CSV to merge. Repeat this option to "
            "override the default Surprise, Disgust, and Anger files."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=9000,
        help="Sample count used in default generated augmentation filenames.",
    )
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        help="Generate missing default Surprise, Disgust, and Anger CSVs before merging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when --generate-missing creates augmentation CSVs.",
    )
    return parser.parse_args()


def maybe_generate_missing(paths, n, seed, emotions):
    missing = [path for path in paths if not os.path.exists(path)]
    if not missing:
        return

    from augmentations.emotion_augmenter import run

    missing_by_emotion = {
        emotion: processed_path(f"augmented_afghan_{emotion}_{n}.csv") for emotion in emotions
    }
    for emotion, path in missing_by_emotion.items():
        if path in missing:
            run(emotion=emotion, n=n, seed=seed, output=path)


def main():
    args = parse_args()
    emotions = parse_emotions(args.emotions)
    base_path = os.path.abspath(os.path.expanduser(args.base_dataset))
    output_path = os.path.abspath(os.path.expanduser(args.output))
    augmented_files = [
        os.path.abspath(os.path.expanduser(path))
        for path in (args.augmented_files or default_augmented_files(args.n, emotions))
    ]

    if args.generate_missing and not args.augmented_files:
        maybe_generate_missing(augmented_files, n=args.n, seed=args.seed, emotions=emotions)

    print(f"Base dataset: {base_path}")
    print(f"Output dataset: {output_path}")
    print(f"Emotions to merge: {', '.join(emotions)}")
    print("Augmented datasets:")
    for path in augmented_files:
        print(f" - {path}")

    base_df = load_dataset(base_path)
    print_distribution(base_df, "Base distribution")

    base_columns = list(base_df.columns)
    frames = [base_df]

    for path in augmented_files:
        augmented_df = load_dataset(path)
        aligned_df = align_to_base_schema(augmented_df, base_columns)
        print_distribution(aligned_df, f"Adding {os.path.basename(path)}")
        frames.append(aligned_df)

    combined = pd.concat(frames, ignore_index=True)
    before_dedup = len(combined)
    combined["_dedup_key"] = combined["clean"].map(normalize_text)
    combined = combined[combined["_dedup_key"] != ""]
    combined = combined.drop_duplicates(subset="_dedup_key", keep="first")
    combined = combined.drop(columns=["_dedup_key"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8")

    print_distribution(combined, "Final merged distribution")
    print(f"\nRemoved duplicates/empty rows: {before_dedup - len(combined)}")
    print(f"Saved merged dataset: {output_path}")


if __name__ == "__main__":
    main()
