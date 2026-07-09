"""
Ablation experiment runner for augmentation studies.

Runs one experiment at a time so you can review training_metadata.json
before continuing. Default behaviour is dry-run (print commands only).

Example:
    python scripts/run_ablation.py --dry-run
    python scripts/run_ablation.py --run-id A0
    python scripts/run_ablation.py --run-id A2 --build-datasets
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import PATHS


TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train.py")
MERGE_SCRIPT = os.path.join(os.path.dirname(__file__), "merge_augmented_datasets.py")

ABLATION_HYPERPARAMS = [
    "--use-dynamic-padding",
    "--batch-size",
    "16",
    "--max-length",
    "256",
    "--num-train-epochs",
    "4",
]

# Anchor runs (A1, A4) have known results — skip unless --force is passed.
ABLATION_RUNS = {
    "A0": {
        "description": "Pseudo-labeled 400K only (no template augmentation)",
        "mode": "full_8label",
        "dataset_path": PATHS["Combined_Labeled_Dataset"],
        "anchor": False,
    },
    "A1": {
        "description": "Fear augmentation only",
        "mode": "full_8label",
        "dataset_path": PATHS["Combined_Labeled_Dataset_with_fearAug"],
        "anchor": True,
        "anchor_metrics": {"accuracy": 0.8575, "f1_macro": 0.836},
    },
    "A2": {
        "description": "Fear + Surprise augmentation",
        "mode": "full_8label",
        "dataset_path": PATHS["Combined_Labeled_Dataset_with_fearAug_surprise"],
        "anchor": False,
        "build_merge": {
            "emotions": "surprise",
            "output": PATHS["Combined_Labeled_Dataset_with_fearAug_surprise"],
        },
    },
    "A3": {
        "description": "Fear + Surprise + Anger augmentation",
        "mode": "full_8label",
        "dataset_path": PATHS["Combined_Labeled_Dataset_with_fearAug_surprise_anger"],
        "anchor": False,
        "build_merge": {
            "emotions": "surprise,anger",
            "output": PATHS["Combined_Labeled_Dataset_with_fearAug_surprise_anger"],
        },
    },
    "A4": {
        "description": "Full stack: Fear + Surprise + Anger + Disgust",
        "mode": "full_8label_all_aug",
        "dataset_path": PATHS["Combined_Labeled_Dataset_with_allAug"],
        "anchor": True,
        "anchor_metrics": {"accuracy": 0.8612, "f1_macro": 0.857},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run augmentation ablation experiments one at a time."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing (default when --run-id is omitted).",
    )
    parser.add_argument(
        "--run-id",
        choices=sorted(ABLATION_RUNS.keys()),
        help="Run a single ablation experiment (e.g. A0, A2).",
    )
    parser.add_argument(
        "--build-datasets",
        action="store_true",
        help="For A2/A3, build partial merged CSVs before training.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run anchor experiments A1/A4 even though results are already known.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all ablation runs and their status.",
    )
    return parser.parse_args()


def ablation_output_dir(run_id):
    return os.path.join(PATHS["ablation_outputs"], run_id)


def build_merge_command(build_config):
    return [
        sys.executable,
        MERGE_SCRIPT,
        "--base-dataset",
        PATHS["Combined_Labeled_Dataset_with_fearAug"],
        "--output",
        build_config["output"],
        "--emotions",
        build_config["emotions"],
    ]


def build_train_command(run_id, run_config):
    output_dir = ablation_output_dir(run_id)
    command = [
        sys.executable,
        TRAIN_SCRIPT,
        "--mode",
        run_config["mode"],
        "--dataset-path",
        run_config["dataset_path"],
        "--output-dir",
        output_dir,
        "--final-model-dir",
        output_dir,
        *ABLATION_HYPERPARAMS,
    ]
    return command


def print_command(label, command):
    print(f"\n{label}")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))


def run_command(command):
    print("\nExecuting:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def print_run_summary(run_id, run_config):
    print(f"\n{'=' * 60}")
    print(f"Run ID: {run_id}")
    print(f"Description: {run_config['description']}")
    print(f"Mode: {run_config['mode']}")
    print(f"Dataset: {run_config['dataset_path']}")
    if run_config.get("anchor"):
        metrics = run_config.get("anchor_metrics", {})
        print(
            "Anchor (known result): "
            f"accuracy={metrics.get('accuracy', 'n/a')}, "
            f"f1_macro={metrics.get('f1_macro', 'n/a')}"
        )
    print(f"Output dir: {ablation_output_dir(run_id)}")
    print(f"{'=' * 60}")


def maybe_skip_anchor(run_id, run_config, force):
    if run_config.get("anchor") and not force:
        metrics = run_config.get("anchor_metrics", {})
        print(
            f"\nSkipping {run_id} (anchor run with known results). "
            f"accuracy={metrics.get('accuracy')}, f1_macro={metrics.get('f1_macro')}. "
            "Pass --force to re-run."
        )
        return True
    return False


def show_metadata_hint(run_id):
    metadata_path = os.path.join(ablation_output_dir(run_id), "training_metadata.json")
    print(f"\nReview results in: {metadata_path}")
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        metrics = metadata.get("final_metrics", {})
        print("Final metrics:")
        for key, value in metrics.items():
            if key.startswith("eval_"):
                print(f"  {key}: {value:.4f}")


def list_runs():
    print("\nAblation experiment matrix:\n")
    for run_id, run_config in ABLATION_RUNS.items():
        status = "ANCHOR (known)" if run_config.get("anchor") else "needs run"
        print(f"  {run_id}: {run_config['description']} [{status}]")
    print("\nRun one experiment:")
    print("  python scripts/run_ablation.py --run-id A0")
    print("\nPreview all commands:")
    print("  python scripts/run_ablation.py --dry-run")


def execute_run(run_id, build_datasets, force, dry_run=False):
    run_config = ABLATION_RUNS[run_id]
    print_run_summary(run_id, run_config)

    if maybe_skip_anchor(run_id, run_config, force):
        return

    if build_datasets and run_config.get("build_merge"):
        merge_command = build_merge_command(run_config["build_merge"])
        print_command("Merge command:", merge_command)
        if not dry_run:
            run_command(merge_command)

    if not os.path.exists(run_config["dataset_path"]):
        hint = ""
        if run_config.get("build_merge"):
            hint = f" Run with --build-datasets to create it, or merge manually."
        raise FileNotFoundError(f"Dataset not found: {run_config['dataset_path']}.{hint}")

    train_command = build_train_command(run_id, run_config)
    print_command("Train command:", train_command)

    if dry_run:
        print("\nDry run only — command not executed.")
        return

    os.makedirs(ablation_output_dir(run_id), exist_ok=True)
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
            build_datasets=args.build_datasets,
            force=args.force,
            dry_run=args.dry_run,
        )
        return

    # Default: dry-run all non-anchor runs
    print("Ablation dry-run (non-anchor experiments):\n")
    for run_id, run_config in ABLATION_RUNS.items():
        if run_config.get("anchor") and not args.force:
            print(f"\n--- {run_id} (anchor, skipped) ---")
            continue
        print_run_summary(run_id, run_config)
        if run_config.get("build_merge"):
            print_command("Merge command:", build_merge_command(run_config["build_merge"]))
        print_command("Train command:", build_train_command(run_id, run_config))

    print("\nTo execute one run:")
    print("  python scripts/run_ablation.py --run-id A0")
    print("For A2/A3, add --build-datasets to create partial merged CSVs first.")


if __name__ == "__main__":
    main()
