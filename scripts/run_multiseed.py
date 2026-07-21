"""
Multi-seed stability runner for ablation experiments.

Trains the same ablation run multiple times with different training seeds
while keeping the train/test split fixed (random_state=42 by default).
Per-seed outputs go under outputs/ablation/<run_id>/seed_<N>/ so the anchor
model at outputs/ablation/<run_id>/ is not overwritten.

Example:
    python scripts/run_multiseed.py --run-id A4 --force --dry-run
    python scripts/run_multiseed.py --run-id A4 --force
    python scripts/run_multiseed.py --run-id A4 --seeds 41 42 43 --force
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from config.paths import PATHS

from run_ablation import ABLATION_HYPERPARAMS, ABLATION_RUNS, TRAIN_SCRIPT


DEFAULT_SEEDS = [41, 42, 43, 44, 45]
DEFAULT_SPLIT_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-seed stability checks for an ablation experiment."
    )
    parser.add_argument(
        "--run-id",
        choices=sorted(ABLATION_RUNS.keys()),
        default="A4",
        help="Ablation run to repeat (default: A4).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"Training seeds to loop (default: {' '.join(map(str, DEFAULT_SEEDS))}).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f"Fixed split seed passed as --random-state (default: {DEFAULT_SPLIT_SEED}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Required for anchor runs (A1, A4) which are otherwise skipped in run_ablation.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training; only read existing seed_*/training_metadata.json and write summary.",
    )
    return parser.parse_args()


def ablation_output_dir(run_id):
    return os.path.join(PATHS["ablation_outputs"], run_id)


def seed_output_dir(run_id, seed):
    return os.path.join(ablation_output_dir(run_id), f"seed_{seed}")


def build_train_command(run_id, run_config, seed, split_seed):
    output_dir = seed_output_dir(run_id, seed)
    return [
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
        "--seed",
        str(seed),
        "--random-state",
        str(split_seed),
        *ABLATION_HYPERPARAMS,
    ]


def print_command(label, command):
    print(f"\n{label}")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))


def run_command(command):
    print("\nExecuting:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def load_seed_metrics(run_id, seed):
    metadata_path = os.path.join(seed_output_dir(run_id, seed), "training_metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    final_metrics = metadata.get("final_metrics", {})
    per_class_f1 = metadata.get("per_class_f1", {})
    return {
        "seed": seed,
        "metadata_path": metadata_path,
        "accuracy": final_metrics.get("eval_accuracy"),
        "f1_macro": final_metrics.get("eval_f1_macro"),
        "f1_weighted": final_metrics.get("eval_f1_weighted"),
        "fear_f1": per_class_f1.get("Fear"),
        "per_class_f1": per_class_f1,
        "split_random_state": metadata.get("split_random_state"),
        "training_seed": metadata.get("training_seed"),
    }


def aggregate_results(run_id, seeds, split_seed):
    per_seed = []
    missing = []
    for seed in seeds:
        row = load_seed_metrics(run_id, seed)
        if row is None:
            missing.append(seed)
            continue
        per_seed.append(row)

    if not per_seed:
        raise FileNotFoundError(
            f"No training_metadata.json found under {ablation_output_dir(run_id)}/seed_*/"
        )

    def collect(key):
        values = [row[key] for row in per_seed if row.get(key) is not None]
        return values

    def summary_stats(key):
        values = collect(key)
        if not values:
            return None
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    summary = {
        "run_id": run_id,
        "split_seed": split_seed,
        "seeds_requested": seeds,
        "seeds_completed": [row["seed"] for row in per_seed],
        "seeds_missing": missing,
        "per_seed": per_seed,
        "aggregate": {
            "accuracy": summary_stats("accuracy"),
            "f1_macro": summary_stats("f1_macro"),
            "f1_weighted": summary_stats("f1_weighted"),
            "fear_f1": summary_stats("fear_f1"),
        },
    }

    summary_path = os.path.join(ablation_output_dir(run_id), "multiseed_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary, summary_path


def print_summary_table(summary):
    print("\n" + "=" * 72)
    print(f"Multi-seed summary: {summary['run_id']} (split_seed={summary['split_seed']})")
    print("=" * 72)
    print(f"{'Seed':>6}  {'Accuracy':>10}  {'Macro-F1':>10}  {'Fear F1':>10}")
    print("-" * 72)
    for row in summary["per_seed"]:
        acc = row.get("accuracy")
        macro = row.get("f1_macro")
        fear = row.get("fear_f1")
        acc_s = f"{acc:.4f}" if acc is not None else "n/a"
        macro_s = f"{macro:.4f}" if macro is not None else "n/a"
        fear_s = f"{fear:.4f}" if fear is not None else "n/a"
        print(f"{row['seed']:>6}  {acc_s:>10}  {macro_s:>10}  {fear_s:>10}")

    agg = summary.get("aggregate", {})
    print("-" * 72)
    for label, key in [("Accuracy", "accuracy"), ("Macro-F1", "f1_macro"), ("Fear F1", "fear_f1")]:
        stats = agg.get(key)
        if stats:
            print(
                f"{label + ' mean±std':>28}: "
                f"{stats['mean']:.4f} ± {stats['std']:.4f}  "
                f"(min {stats['min']:.4f}, max {stats['max']:.4f})"
            )
    print("=" * 72)


def execute_multiseed(run_id, seeds, split_seed, force, dry_run, aggregate_only):
    run_config = ABLATION_RUNS[run_id]

    if run_config.get("anchor") and not force:
        print(
            f"\n{run_id} is an anchor run. Pass --force to run multi-seed stability checks."
        )
        return

    if not os.path.exists(run_config["dataset_path"]) and not dry_run:
        raise FileNotFoundError(f"Dataset not found: {run_config['dataset_path']}")

    print(f"\nMulti-seed run: {run_id}")
    print(f"Description: {run_config['description']}")
    print(f"Dataset: {run_config['dataset_path']}")
    print(f"Training seeds: {seeds}")
    print(f"Split seed (random_state): {split_seed}")
    print(f"Base output: {ablation_output_dir(run_id)}")

    if not aggregate_only:
        for seed in seeds:
            command = build_train_command(run_id, run_config, seed, split_seed)
            print_command(f"Seed {seed}:", command)
            if dry_run:
                continue
            os.makedirs(seed_output_dir(run_id, seed), exist_ok=True)
            run_command(command)

        if dry_run:
            print("\nDry run only — no training executed, no summary written.")
            return

    summary, summary_path = aggregate_results(run_id, seeds, split_seed)
    print_summary_table(summary)
    print(f"\nSummary written to: {summary_path}")
    if summary.get("seeds_missing"):
        print(f"Warning: missing seeds (no metadata): {summary['seeds_missing']}")


def main():
    args = parse_args()
    execute_multiseed(
        run_id=args.run_id,
        seeds=args.seeds,
        split_seed=args.split_seed,
        force=args.force,
        dry_run=args.dry_run,
        aggregate_only=args.aggregate_only,
    )


if __name__ == "__main__":
    main()
