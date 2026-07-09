"""
Confidence threshold training experiments.

Run one experiment at a time after reviewing score_pseudo_labels.py output
and creating filtered datasets with filter_by_confidence.py.

Example:
    python scripts/run_confidence_experiments.py --list
    python scripts/run_confidence_experiments.py --dry-run
    python scripts/run_confidence_experiments.py --run-id C0
    python scripts/run_confidence_experiments.py --run-id C2 --build-filtered --threshold 0.8
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import PATHS, confidence_filtered_path


TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train.py")
FILTER_SCRIPT = os.path.join(os.path.dirname(__file__), "filter_by_confidence.py")
SCORE_SCRIPT = os.path.join(os.path.dirname(__file__), "score_pseudo_labels.py")

TRAIN_HYPERPARAMS = [
    "--use-dynamic-padding",
    "--batch-size",
    "16",
    "--max-length",
    "256",
    "--num-train-epochs",
    "4",
]

DEFAULT_THRESHOLDS = {
    "C1": 0.7,
    "C2": 0.8,
    "C3": 0.9,
}

CONFIDENCE_RUNS = {
    "C0": {
        "description": "Unfiltered pseudo-labeled 400K (baseline reference)",
        "dataset_path": PATHS["Combined_Labeled_Dataset"],
        "threshold": None,
    },
    "C1": {
        "description": "Confidence >= 0.7 with label agreement",
        "threshold": 0.7,
    },
    "C2": {
        "description": "Confidence >= 0.8 with label agreement",
        "threshold": 0.8,
    },
    "C3": {
        "description": "Confidence >= 0.9 with label agreement",
        "threshold": 0.9,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run confidence threshold training experiments one at a time."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--run-id",
        choices=sorted(CONFIDENCE_RUNS.keys()),
        help="Run a single confidence experiment (e.g. C0, C2).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all confidence experiments.",
    )
    parser.add_argument(
        "--build-filtered",
        action="store_true",
        help="For C1/C2/C3, run filter_by_confidence.py before training.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override threshold for a filtered run (e.g. 0.75 for a custom C2).",
    )
    parser.add_argument(
        "--build-scored",
        action="store_true",
        help="Run score_pseudo_labels.py before filtering (slow on full 400K).",
    )
    return parser.parse_args()


def confidence_output_dir(run_id):
    return os.path.join(PATHS["confidence_outputs"], run_id)


def dataset_path_for_run(run_config, threshold_override=None):
    threshold = threshold_override if threshold_override is not None else run_config.get("threshold")
    if threshold is None:
        return run_config["dataset_path"]
    return confidence_filtered_path(threshold)


def build_score_command():
    return [sys.executable, SCORE_SCRIPT]


def build_filter_command(threshold):
    return [sys.executable, FILTER_SCRIPT, "--threshold", str(threshold)]


def build_train_command(run_id, dataset_path):
    output_dir = confidence_output_dir(run_id)
    return [
        sys.executable,
        TRAIN_SCRIPT,
        "--mode",
        "full_8label",
        "--dataset-path",
        dataset_path,
        "--output-dir",
        output_dir,
        "--final-model-dir",
        output_dir,
        *TRAIN_HYPERPARAMS,
    ]


def print_command(label, command):
    rendered = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"\n{label}\n{rendered}")


def run_command(command):
    print("\nExecuting:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def print_run_summary(run_id, run_config, dataset_path):
    print(f"\n{'=' * 60}")
    print(f"Run ID: {run_id}")
    print(f"Description: {run_config['description']}")
    print(f"Dataset: {dataset_path}")
    print(f"Output dir: {confidence_output_dir(run_id)}")
    print(f"{'=' * 60}")


def show_metadata_hint(run_id):
    metadata_path = os.path.join(confidence_output_dir(run_id), "training_metadata.json")
    print(f"\nReview results in: {metadata_path}")
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        metrics = metadata.get("final_metrics", {})
        print("Final metrics:")
        for key, value in metrics.items():
            if key.startswith("eval_"):
                print(f"  {key}: {value:.4f}")
        print(f"  dataset_size: {metadata.get('dataset_size', 'n/a')}")


def list_runs():
    print("\nConfidence threshold experiments:\n")
    for run_id, run_config in CONFIDENCE_RUNS.items():
        path = dataset_path_for_run(run_config)
        print(f"  {run_id}: {run_config['description']}")
        print(f"       dataset: {path}")
    print("\nWorkflow:")
    print("  1. python scripts/score_pseudo_labels.py")
    print("  2. Review diagnostics, adjust thresholds if needed")
    print("  3. python scripts/filter_by_confidence.py --threshold 0.7 0.8 0.9")
    print("  4. python scripts/run_confidence_experiments.py --run-id C0")


def execute_run(run_id, build_filtered, build_scored, threshold_override, dry_run=False):
    run_config = CONFIDENCE_RUNS[run_id]
    threshold = threshold_override if threshold_override is not None else run_config.get("threshold")
    dataset_path = dataset_path_for_run(run_config, threshold)

    print_run_summary(run_id, run_config, dataset_path)

    if build_scored:
        score_command = build_score_command()
        print_command("Score command:", score_command)
        if not dry_run:
            run_command(score_command)

    if threshold is not None and not dry_run:
        scored_path = PATHS["Combined_Labeled_Dataset_scored"]
        if not os.path.exists(scored_path):
            raise FileNotFoundError(
                f"Scored dataset not found: {scored_path}\n"
                "Run score_pseudo_labels.py first (or pass --build-scored)."
            )

        if build_filtered or not os.path.exists(dataset_path):
            filter_command = build_filter_command(threshold)
            print_command("Filter command:", filter_command)
            run_command(filter_command)

    elif threshold is not None:
        filter_command = build_filter_command(threshold)
        print_command("Filter command:", filter_command)

    if not dry_run and not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run filter_by_confidence.py first or pass --build-filtered."
        )

    train_command = build_train_command(run_id, dataset_path)
    print_command("Train command:", train_command)

    if dry_run:
        print("\nDry run only — command not executed.")
        return

    os.makedirs(confidence_output_dir(run_id), exist_ok=True)
    run_command(train_command)
    show_metadata_hint(run_id)
    print("\nReview training_metadata.json before running the next experiment.")


def main():
    args = parse_args()

    if args.list:
        list_runs()
        return

    if args.run_id:
        execute_run(
            args.run_id,
            build_filtered=args.build_filtered,
            build_scored=args.build_scored,
            threshold_override=args.threshold,
            dry_run=args.dry_run,
        )
        return

    print("Confidence experiment dry-run:\n")
    for run_id, run_config in CONFIDENCE_RUNS.items():
        dataset_path = dataset_path_for_run(run_config)
        print_run_summary(run_id, run_config, dataset_path)
        if run_config.get("threshold") is not None:
            print_command("Filter command:", build_filter_command(run_config["threshold"]))
        print_command("Train command:", build_train_command(run_id, dataset_path))

    print("\nWorkflow:")
    print("  python scripts/score_pseudo_labels.py")
    print("  python scripts/run_confidence_experiments.py --run-id C0")
    print("  python scripts/run_confidence_experiments.py --run-id C2 --build-filtered")


if __name__ == "__main__":
    main()
