import argparse
import os
import pandas as pd

DEFAULT_COMBINED = "/content/drive/MyDrive/Sentiment_Analysis/Data/processed/Combined_Labeled_Dataset.csv"
DEFAULT_FEAR_AUG = "/content/drive/MyDrive/Sentiment_Analysis/Data/processed/augmented_afghan_fear_9000.csv"
DEFAULT_OUTPUT   = "/content/drive/MyDrive/Sentiment_Analysis/Data/processed/Combined_Labeled_Dataset_with_fearAug.csv"

REQUIRED_COLS = ["channelId", "publishedAt", "clean", "token_count", "Label", "label_id"]

def load_and_normalize(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    df = pd.read_csv(path)
    # common rename if main file still has 'text'
    if "text" in df.columns and "clean" not in df.columns:
        df = df.rename(columns={"text": "clean"})
    # enforce required columns (add if missing)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    # keep only required
    df = df[REQUIRED_COLS]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", default=DEFAULT_COMBINED, help="Path to main Combined_Labeled_Dataset.csv")
    ap.add_argument("--fear_aug", default=DEFAULT_FEAR_AUG, help="Path to augmented fear CSV")
    ap.add_argument("--out", default=DEFAULT_OUTPUT, help="Output path for merged CSV")
    # Use parse_known_args to ignore arguments passed by the IPython kernel
    args, unknown = ap.parse_known_args()

    df_main = load_and_normalize(args.combined, "Combined dataset")
    df_fear = load_and_normalize(args.fear_aug, "Augmented fear dataset")

    merged = pd.concat([df_main, df_fear], ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Merged dataset saved: {args.out}")
    print(f"Rows: {len(merged):,} (main: {len(df_main):,}, fear_aug: {len(df_fear):,})")

if __name__ == "__main__":
    main()
