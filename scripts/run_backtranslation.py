"""
Back-translation experiment runner (Phase 3).

Runs one step at a time with human review gates between spot-check and full
generation. Default behaviour is dry-run (print commands only).

Example:
    python scripts/run_backtranslation.py --step spotcheck --dry-run
    python scripts/run_backtranslation.py --step spotcheck
    python scripts/run_backtranslation.py --step generate
    python scripts/run_backtranslation.py --step merge
    python scripts/run_backtranslation.py --step train
    python scripts/run_ablation.py --run-id A5
"""

import argparse
import os
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import PATHS, augmented_bt_path


BT_AUGMENTER = os.path.join(
    os.path.dirname(__file__), "..", "augmentations", "back_translation_augmenter.py"
)
MERGE_SCRIPT = os.path.join(os.path.dirname(__file__), "merge_augmented_datasets.py")
ABLATION_SCRIPT = os.path.join(os.path.dirname(__file__), "run_ablation.py")

BT_EMOTIONS = {
    "surprise": {"n": 2000},
    "anger": {"n": 2000},
    "disgust": {"n": 2000},
    "sad": {"n": 1500},
    "fear": {"n": 1000},
}

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run back-translation augmentation steps one at a time."
    )
    parser.add_argument(
        "--step",
        choices=["spotcheck", "generate", "merge", "train"],
        help="Pipeline step to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Translation batch size for the augmenter.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List pipeline steps and target sample counts.",
    )
    return parser.parse_args()


def emotion_list() -> str:
    return ",".join(BT_EMOTIONS.keys())


def bt_output_paths() -> list[str]:
    return [augmented_bt_path(emotion, config["n"]) for emotion, config in BT_EMOTIONS.items()]


def build_spotcheck_command(seed: int, batch_size: int) -> list[str]:
    return [
        sys.executable,
        BT_AUGMENTER,
        "--emotions",
        emotion_list(),
        "--spot-check-only",
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
        "--source-dataset",
        PATHS["Labeled_4K"],
    ]


def build_generate_commands(seed: int, batch_size: int) -> list[list[str]]:
    commands = []
    for index, (emotion, config) in enumerate(BT_EMOTIONS.items()):
        commands.append(
            [
                sys.executable,
                BT_AUGMENTER,
                "--emotion",
                emotion,
                "--n",
                str(config["n"]),
                "--seed",
                str(seed + index),
                "--batch-size",
                str(batch_size),
                "--source-dataset",
                PATHS["Labeled_4K"],
                "--output",
                augmented_bt_path(emotion, config["n"]),
            ]
        )
    return commands


def build_merge_command() -> list[str]:
    command = [
        sys.executable,
        MERGE_SCRIPT,
        "--base-dataset",
        PATHS["Combined_Labeled_Dataset_with_allAug"],
        "--output",
        PATHS["Combined_Labeled_Dataset_with_allAug_bt"],
        "--no-template-aug",
    ]
    for path in bt_output_paths():
        command.extend(["--extra-augmented-file", path])
    return command


def build_train_command() -> list[str]:
    return [
        sys.executable,
        ABLATION_SCRIPT,
        "--run-id",
        "A5",
    ]


def print_command(label: str, command: list[str]) -> None:
    rendered = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"\n{label}")
    print(rendered)


def run_command(command: list[str]) -> None:
    print("\nExecuting:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def list_steps():
    print("\nBack-translation pipeline (Phase 3):\n")
    print("  spotcheck  — 100 samples/class for human review (~30-60 min GPU)")
    print("  generate   — full BT generation from 4K seed (~6-12 h GPU)")
    print("  merge      — append BT CSVs onto allAug base")
    print("  train      — run ablation A5 (same protocol as A4)\n")
    print("Target counts per emotion:")
    for emotion, config in BT_EMOTIONS.items():
        print(f"  {emotion}: {config['n']}")
    print("\nStop rule: abort generate if spot-check has <80% label-preserving samples/class.")
    print("\nExample:")
    print("  python scripts/run_backtranslation.py --step spotcheck")
    print("  python scripts/run_backtranslation.py --step generate")
    print("  python scripts/run_backtranslation.py --step merge")
    print("  python scripts/run_backtranslation.py --step train")


def execute_step(step: str, dry_run: bool, seed: int, batch_size: int) -> None:
    if step == "spotcheck":
        command = build_spotcheck_command(seed, batch_size)
        print_command("Spot-check command:", command)
        if not dry_run:
            run_command(command)
            print(
                "\nReview spot-check CSVs in Data/processed/bt_spotcheck_*.csv "
                "before running --step generate."
            )
        return

    if step == "generate":
        commands = build_generate_commands(seed, batch_size)
        for command in commands:
            print_command("Generate command:", command)
            if not dry_run:
                run_command(command)
        if dry_run:
            print("\nDry run only — commands not executed.")
        else:
            print("\nBack-translation generation complete. Run --step merge next.")
        return

    if step == "merge":
        command = build_merge_command()
        print_command("Merge command:", command)
        if dry_run:
            print("\nDry run only — command not executed.")
            return
        run_command(command)
        print(f"\nMerged dataset: {PATHS['Combined_Labeled_Dataset_with_allAug_bt']}")
        return

    if step == "train":
        command = build_train_command()
        print_command("Train command (delegates to run_ablation.py --run-id A5):", command)
        if dry_run:
            print("\nDry run only — command not executed.")
            return
        run_command(command)
        print("\nReview outputs/ablation/A5/training_metadata.json and compare to A4 anchor.")


def main():
    args = parse_args()

    if args.list:
        list_steps()
        return

    if not args.step:
        print("Back-translation dry-run (all steps):\n")
        execute_step("spotcheck", dry_run=True, seed=args.seed, batch_size=args.batch_size)
        execute_step("generate", dry_run=True, seed=args.seed, batch_size=args.batch_size)
        execute_step("merge", dry_run=True, seed=args.seed, batch_size=args.batch_size)
        execute_step("train", dry_run=True, seed=args.seed, batch_size=args.batch_size)
        print("\nTo execute one step:")
        print("  python scripts/run_backtranslation.py --step spotcheck")
        return

    execute_step(args.step, dry_run=args.dry_run, seed=args.seed, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
