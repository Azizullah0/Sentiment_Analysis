"""
Confidence threshold training experiments.

Valid evaluation (default): train on filtered train-pool only, evaluate on fixed
unfiltered holdout from the original 400K (same seed=42 split as ablation).

When train_pool / eval_holdout / train_filtered_conf*.csv already exist (see
audit_datasets.py or docs/thesis_dataset_notes.md), train directly without
--prepare-splits so splits stay aligned with completed ablation work.

Example:
    python scripts/run_confidence_experiments.py --run-id C0 --valid-eval
    python scripts/run_confidence_experiments.py --run-id C1 --valid-eval
    python scripts/evaluate_holdout.py --model outputs/confidence/C1
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import PATHS, train_filtered_confidence_path


TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train.py")
FILTER_SCRIPT = os.path.join(os.path.dirname(__file__), "filter_by_confidence.py")
SCORE_SCRIPT = os.path.join(os.path.dirname(__file__), "score_pseudo_labels.py")
PREPARE_SCRIPT = os.path.join(os.path.dirname(__file__), "prepare_confidence_splits.py")
EVAL_HOLDOUT_SCRIPT = os.path.join(os.path.dirname(__file__), "evaluate_holdout.py")

TRAIN_HYPERPARAMS = [
    "--use-dynamic-padding",
    "--batch-size",
    "16",
    "--max-length",
    "256",
    "--num-train-epochs",
    "4",
]

CONFIDENCE_RUNS = {
    "C0": {
        "description": "Unfiltered train pool (80% of original) — baseline with fixed holdout",
        "threshold": None,
        "train_path_key": "train_pool_original",
    },
    "C1": {
        "description": "Filtered train pool: confidence >= 0.7 + agreement",
        "threshold": 0.7,
    },
    "C2": {
        "description": "Filtered train pool: confidence >= 0.8 + agreement",
        "threshold": 0.8,
    },
    "C3": {
        "description": "Filtered train pool: confidence >= 0.9 + agreement",
        "threshold": 0.9,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run confidence threshold training experiments one at a time."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--run-id", choices=sorted(CONFIDENCE_RUNS.keys()), help="Run one experiment.")
    parser.add_argument("--list", action="store_true", help="List all confidence experiments.")
    parser.add_argument(
        "--valid-eval",
        action="store_true",
        default=True,
        help="Use fixed holdout eval (default: on). Train on filtered train pool only.",
    )
    parser.add_argument(
        "--legacy-eval",
        dest="valid_eval",
        action="store_false",
        help="Old behaviour: random split inside filtered CSV (inflated metrics).",
    )
    parser.add_argument(
        "--prepare-splits",
        action="store_true",
        help="Run prepare_confidence_splits.py before training (requires scored CSV).",
    )
    parser.add_argument(
        "--build-scored",
        action="store_true",
        help="Run score_pseudo_labels.py before preparing splits.",
    )
    parser.add_argument("--threshold", type=float, help="Override threshold for C1/C2/C3.")
    return parser.parse_args()


def confidence_output_dir(run_id):
    return os.path.join(PATHS["confidence_outputs"], run_id)


def train_dataset_path(run_config, threshold_override=None, valid_eval=True):
    threshold = threshold_override if threshold_override is not None else run_config.get("threshold")
    if valid_eval:
        if threshold is None:
            return PATHS["train_pool_original"]
        return train_filtered_confidence_path(threshold)
    if threshold is None:
        return PATHS["Combined_Labeled_Dataset"]
    from config.paths import confidence_filtered_path

    return confidence_filtered_path(threshold)


def build_score_command():
    return [sys.executable, SCORE_SCRIPT]


def build_prepare_command(thresholds):
    cmd = [sys.executable, PREPARE_SCRIPT]
    for threshold in thresholds:
        cmd.extend(["--threshold", str(threshold)])
    return cmd


def build_train_command(run_id, train_path, valid_eval):
    output_dir = confidence_output_dir(run_id)
    command = [
        sys.executable,
        TRAIN_SCRIPT,
        "--mode",
        "full_8label",
        "--dataset-path",
        train_path,
        "--output-dir",
        output_dir,
        "--final-model-dir",
        output_dir,
        *TRAIN_HYPERPARAMS,
    ]
    if valid_eval:
        command.extend(["--eval-dataset-path", PATHS["eval_holdout_original"]])
    return command


def build_holdout_eval_command(run_id):
    return [sys.executable, EVAL_HOLDOUT_SCRIPT, "--model", confidence_output_dir(run_id)]


def print_command(label, command):
    rendered = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"\n{label}\n{rendered}")


def run_command(command):
    print("\nExecuting:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def print_run_summary(run_id, run_config, train_path, valid_eval):
    print(f"\n{'=' * 60}")
    print(f"Run ID: {run_id}")
    print(f"Description: {run_config['description']}")
    print(f"Train dataset: {train_path}")
    if valid_eval:
        print(f"Eval dataset:  {PATHS['eval_holdout_original']} (fixed unfiltered holdout)")
    else:
        print("Eval: random split inside train CSV (legacy — metrics may be inflated)")
    print(f"Output dir: {confidence_output_dir(run_id)}")
    print(f"{'=' * 60}")


def show_metadata_hint(run_id):
    output_dir = confidence_output_dir(run_id)
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    holdout_path = os.path.join(output_dir, "holdout_eval.json")
    print(f"\nReview: {metadata_path}")
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        metrics = metadata.get("final_metrics", {})
        print("Training eval metrics:")
        for key, value in metrics.items():
            if key.startswith("eval_"):
                print(f"  {key}: {value:.4f}")
        if metadata.get("eval_protocol") == "fixed_holdout":
            print("  (valid — fixed holdout)")
    if os.path.exists(holdout_path):
        print(f"Holdout eval: {holdout_path}")


def list_runs():
    print("\nConfidence threshold experiments (valid-eval default):\n")
    for run_id, run_config in CONFIDENCE_RUNS.items():
        train_path = train_dataset_path(run_config, valid_eval=True)
        print(f"  {run_id}: {run_config['description']}")
        print(f"       train: {train_path}")
        print(f"       eval:  {PATHS['eval_holdout_original']}")
    print("\nWorkflow:")
    print("  1. python scripts/score_pseudo_labels.py")
    print("  2. python scripts/prepare_confidence_splits.py")
    print("  3. python scripts/run_confidence_experiments.py --run-id C1 --valid-eval")
    print("\nRe-evaluate existing model without retraining:")
    print("  python scripts/evaluate_holdout.py --model outputs/confidence/C1")


def thresholds_for_prepare(threshold_override, run_id):
    if threshold_override is not None:
        return [threshold_override]
    if run_id and CONFIDENCE_RUNS[run_id].get("threshold") is not None:
        return [CONFIDENCE_RUNS[run_id]["threshold"]]
    return [0.7, 0.8, 0.9]


def execute_run(run_id, valid_eval, prepare_splits, build_scored, threshold_override, dry_run=False):
    run_config = CONFIDENCE_RUNS[run_id]
    train_path = train_dataset_path(run_config, threshold_override, valid_eval)

    print_run_summary(run_id, run_config, train_path, valid_eval)

    if valid_eval and (prepare_splits or build_scored):
        if build_scored:
            score_command = build_score_command()
            print_command("Score command:", score_command)
            if not dry_run:
                run_command(score_command)

        prepare_command = build_prepare_command(thresholds_for_prepare(threshold_override, run_id))
        print_command("Prepare splits command:", prepare_command)
        if not dry_run:
            run_command(prepare_command)

    if not dry_run and not os.path.exists(train_path):
        hint = " Run with --prepare-splits after score_pseudo_labels.py."
        if not valid_eval and run_config.get("threshold"):
            hint = " Run filter_by_confidence.py first."
        raise FileNotFoundError(f"Train dataset not found: {train_path}.{hint}")

    if valid_eval and not dry_run and not os.path.exists(PATHS["eval_holdout_original"]):
        raise FileNotFoundError(
            f"Eval holdout not found: {PATHS['eval_holdout_original']}\n"
            "Run prepare_confidence_splits.py first (or pass --prepare-splits)."
        )

    train_command = build_train_command(run_id, train_path, valid_eval)
    print_command("Train command:", train_command)

    if dry_run:
        if valid_eval:
            print_command("Optional holdout re-eval:", build_holdout_eval_command(run_id))
        print("\nDry run only — commands not executed.")
        return

    os.makedirs(confidence_output_dir(run_id), exist_ok=True)
    run_command(train_command)
    show_metadata_hint(run_id)
    print("\nReview training_metadata.json before running the next experiment.")
    print("To re-evaluate on holdout only: python scripts/evaluate_holdout.py --model", confidence_output_dir(run_id))


def main():
    args = parse_args()

    if args.list:
        list_runs()
        return

    if args.run_id:
        execute_run(
            args.run_id,
            valid_eval=args.valid_eval,
            prepare_splits=args.prepare_splits,
            build_scored=args.build_scored,
            threshold_override=args.threshold,
            dry_run=args.dry_run,
        )
        return

    print("Confidence experiment dry-run (valid-eval):\n")
    for run_id, run_config in CONFIDENCE_RUNS.items():
        train_path = train_dataset_path(run_config, valid_eval=args.valid_eval)
        print_run_summary(run_id, run_config, train_path, args.valid_eval)
        if args.valid_eval:
            print_command("Prepare splits:", build_prepare_command([0.7, 0.8, 0.9]))
        print_command("Train command:", build_train_command(run_id, train_path, args.valid_eval))

    print("\nStart:")
    print("  python scripts/score_pseudo_labels.py")
    print("  python scripts/prepare_confidence_splits.py")
    print("  python scripts/run_confidence_experiments.py --run-id C1 --valid-eval")


if __name__ == "__main__":
    main()
