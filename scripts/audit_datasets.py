"""
Audit experiment-related datasets: row counts, label distributions, duplicates,
and merge/split cross-checks. Terminal output for thesis table verification.

Example:
    python scripts/audit_datasets.py
    python scripts/audit_datasets.py --group ablation --strict
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import PATHS, augmented_bt_path, train_filtered_confidence_path

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

BT_EMOTIONS = {
    "surprise": 2000,
    "anger": 2000,
    "disgust": 2000,
    "sad": 1500,
    "fear": 1000,
}

TEMPLATE_EMOTIONS = ["fear", "surprise", "disgust", "anger"]


def normalize_text(value) -> str:
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


@dataclass
class DatasetEntry:
    entry_id: str
    path: str
    group: str
    description: str = ""
    show_label_distribution: bool = True


@dataclass
class DatasetStats:
    entry_id: str
    path: str
    exists: bool
    rows: int = 0
    dupes: int = 0
    text_nan: int = 0
    label_nan: int = 0
    text_column: str | None = None
    label_column: str | None = None
    size_mb: float = 0.0
    label_counts: dict[int, int] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ValidationResult:
    name: str
    status: str  # PASS, WARN, EXPECTED, FAIL
    detail: str

    @property
    def passed(self) -> bool:
        return self.status != "FAIL"


KNOWN_DUPE_DATASETS = frozenset({"A0", "A1", "eval_holdout", "train_pool", "C1_train"})
DEDUPED_ABLATION_IDS = ["A2", "A3", "A4", "A5"]


def ablation_entries() -> list[DatasetEntry]:
    runs = {
        "A0": (PATHS["Combined_Labeled_Dataset"], "Pseudo-labeled 400K only"),
        "A1": (PATHS["Combined_Labeled_Dataset_with_fearAug"], "Fear augmentation only"),
        "A2": (
            PATHS["Combined_Labeled_Dataset_with_fearAug_surprise"],
            "Fear + Surprise",
        ),
        "A3": (
            PATHS["Combined_Labeled_Dataset_with_fearAug_surprise_anger"],
            "Fear + Surprise + Anger",
        ),
        "A4": (PATHS["Combined_Labeled_Dataset_with_allAug"], "Full template stack"),
        "A5": (
            PATHS["Combined_Labeled_Dataset_with_allAug_bt"],
            "A4 + back-translation",
        ),
    }
    return [
        DatasetEntry(entry_id=run_id, path=path, group="ablation", description=desc)
        for run_id, (path, desc) in runs.items()
    ]


def confidence_entries() -> list[DatasetEntry]:
    entries = [
        DatasetEntry(
            "eval_holdout",
            PATHS["eval_holdout_original"],
            "confidence",
            "Fixed 20% unfiltered holdout (all C runs)",
        ),
        DatasetEntry(
            "train_pool",
            PATHS["train_pool_original"],
            "confidence",
            "80% unfiltered train pool (C0)",
        ),
    ]
    for run_id, threshold in [("C1_train", 0.7), ("C2_train", 0.8), ("C3_train", 0.9)]:
        entries.append(
            DatasetEntry(
                run_id,
                train_filtered_confidence_path(threshold),
                "confidence",
                f"Filtered train pool conf>={threshold}",
            )
        )
    entries.append(
        DatasetEntry(
            "scored",
            PATHS["Combined_Labeled_Dataset_scored"],
            "confidence",
            "Scored pseudo-labels (optional input)",
            show_label_distribution=False,
        )
    )
    return entries


def augmentation_entries() -> list[DatasetEntry]:
    processed = os.path.join(PATHS["data_root"], "Data", "processed")
    entries = [
        DatasetEntry(
            "Labeled_4K",
            PATHS["Labeled_4K"],
            "augmentation",
            "Gold seed for baseline + BT",
        ),
        DatasetEntry(
            "fear_template",
            PATHS["augmented_afghan_fear_9000"],
            "augmentation",
            "Template Fear augmentation",
            show_label_distribution=False,
        ),
    ]
    for emotion in TEMPLATE_EMOTIONS[1:]:
        entries.append(
            DatasetEntry(
                f"template_{emotion}",
                os.path.join(processed, f"augmented_afghan_{emotion}_9000.csv"),
                "augmentation",
                f"Template {emotion.title()} augmentation",
                show_label_distribution=False,
            )
        )
    for emotion, n in BT_EMOTIONS.items():
        entries.append(
            DatasetEntry(
                f"bt_{emotion}",
                augmented_bt_path(emotion, n),
                "augmentation",
                f"Back-translation {emotion.title()}",
                show_label_distribution=False,
            )
        )
    for path in sorted(glob.glob(os.path.join(processed, "bt_spotcheck_*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        entries.append(
            DatasetEntry(
                name,
                path,
                "augmentation",
                "BT spot-check sample",
                show_label_distribution=False,
            )
        )
    return entries


def resolve_text_column(df: pd.DataFrame) -> str | None:
    if "clean" in df.columns:
        return "clean"
    if "text" in df.columns:
        return "text"
    return None


def resolve_label_column(df: pd.DataFrame) -> str | None:
    for column in ("label_id", "labels", "Label"):
        if column in df.columns:
            return column
    return None


def label_id_series(df: pd.DataFrame, label_column: str) -> pd.Series:
    if label_column == "label_id":
        return pd.to_numeric(df["label_id"], errors="coerce")
    if label_column == "labels":
        return pd.to_numeric(df["labels"], errors="coerce")
    mapping = {
        "Hope": 0,
        "Happy": 1,
        "Neutral": 2,
        "Surprise": 3,
        "Suprise": 3,
        "Disgust": 4,
        "Sad": 5,
        "Anger": 6,
        "Fear": 7,
    }
    labels = df["Label"].astype(str).str.strip().replace("Suprise", "Surprise")
    return labels.map(mapping)


def analyze_entry(entry: DatasetEntry) -> DatasetStats:
    stats = DatasetStats(entry_id=entry.entry_id, path=entry.path, exists=os.path.exists(entry.path))
    if not stats.exists:
        return stats

    try:
        stats.size_mb = os.path.getsize(entry.path) / (1024 * 1024)
        df = pd.read_csv(entry.path)
        df.columns = df.columns.str.strip()
        stats.rows = len(df)

        text_col = resolve_text_column(df)
        label_col = resolve_label_column(df)
        stats.text_column = text_col
        stats.label_column = label_col

        if text_col:
            stats.text_nan = int(df[text_col].isna().sum())
            keys = df[text_col].map(normalize_text)
            stats.dupes = int(stats.rows - keys.nunique())

        if label_col:
            stats.label_nan = int(df[label_col].isna().sum())
            if entry.show_label_distribution:
                ids = label_id_series(df, label_col).dropna().astype(int)
                stats.label_counts = ids.value_counts().sort_index().to_dict()
    except Exception as exc:
        stats.error = str(exc)

    return stats


def print_header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print("=" * 72)


def print_summary_table(entries: list[DatasetEntry], stats_map: dict[str, DatasetStats]) -> None:
    print(f"\n{'ID':<16} {'Status':<8} {'Rows':>10} {'Dupes':>8} {'SizeMB':>8}  Path")
    print("-" * 72)
    for entry in entries:
        stats = stats_map[entry.entry_id]
        if not stats.exists:
            print(f"{entry.entry_id:<16} {'MISSING':<8} {'—':>10} {'—':>8} {'—':>8}  {entry.path}")
            continue
        if stats.error:
            print(
                f"{entry.entry_id:<16} {'ERROR':<8} {'—':>10} {'—':>8} {'—':>8}  "
                f"{stats.error}"
            )
            continue
        print(
            f"{entry.entry_id:<16} {'OK':<8} {stats.rows:>10,} {stats.dupes:>8,} "
            f"{stats.size_mb:>8.2f}  {entry.path}"
        )
        if entry.description:
            print(f"{'':16} ({entry.description})")


def print_label_distribution(stats: DatasetStats) -> None:
    if not stats.label_counts or stats.error:
        return
    total = sum(stats.label_counts.values())
    print(f"\n  Label distribution for {stats.entry_id} (n={total:,}):")
    for label_id in sorted(stats.label_counts):
        count = stats.label_counts[label_id]
        name = LABEL_NAMES.get(label_id, f"Label_{label_id}")
        pct = 100.0 * count / total if total else 0.0
        print(f"    {name:<10} ({label_id}): {count:>8,} ({pct:5.2f}%)")


def rows_for(stats_map: dict[str, DatasetStats], entry_id: str) -> int | None:
    stats = stats_map.get(entry_id)
    if not stats or not stats.exists or stats.error:
        return None
    return stats.rows


def run_validations(
    ablation_stats: dict[str, DatasetStats],
    confidence_stats: dict[str, DatasetStats],
    augmentation_stats: dict[str, DatasetStats],
) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    ablation_ids = ["A0", "A1", "A2", "A3", "A4", "A5"]
    ablation_rows = {run_id: rows_for(ablation_stats, run_id) for run_id in ablation_ids}

    # A5 merge arithmetic
    a4 = ablation_rows.get("A4")
    a5 = ablation_rows.get("A5")
    bt_total = 0
    bt_found = 0
    for emotion in BT_EMOTIONS:
        bt_stats = augmentation_stats.get(f"bt_{emotion}")
        if bt_stats and bt_stats.exists and not bt_stats.error:
            bt_found += 1
            bt_total += bt_stats.rows
    if a4 is not None and a5 is not None and bt_found > 0:
        delta = a5 - a4
        dedup_removed = bt_total - delta
        merge_ok = 0 <= dedup_removed <= bt_total
        results.append(
            ValidationResult(
                "A5 merge (A5 - A4 vs BT sum)",
                "PASS" if merge_ok else "FAIL",
                f"A4={a4:,}, A5={a5:,}, delta={delta:+,}, BT files sum={bt_total:,}, "
                f"dedup removed={dedup_removed:,}",
            )
        )
    else:
        results.append(
            ValidationResult(
                "A5 merge (A5 - A4 vs BT sum)",
                "PASS",
                "Skipped — missing A4, A5, or BT files",
            )
        )

    # Ablation monotonicity from A2 onward (A1→A2 shrink is expected after dedup merge)
    mono_issues = []
    expected_shrink = []
    prev_id = None
    prev_rows = None
    for run_id in ablation_ids:
        current = ablation_rows.get(run_id)
        if current is None:
            continue
        if prev_rows is not None and current < prev_rows:
            a2_stats = ablation_stats.get("A2")
            if (
                run_id == "A2"
                and prev_id == "A1"
                and a2_stats
                and a2_stats.exists
                and not a2_stats.error
                and a2_stats.dupes == 0
            ):
                expected_shrink.append(
                    f"A2 ({current:,}) < A1 ({prev_rows:,}) after dedup merge (expected)"
                )
            else:
                mono_issues.append(f"{run_id} ({current:,}) < {prev_id} ({prev_rows:,})")
        prev_id = run_id
        prev_rows = current

    if expected_shrink and not mono_issues:
        mono_status = "EXPECTED"
        mono_detail = "; ".join(expected_shrink)
    elif mono_issues:
        mono_status = "FAIL"
        mono_detail = "; ".join(mono_issues)
        if expected_shrink:
            mono_detail = f"{mono_detail}; {'; '.join(expected_shrink)}"
    else:
        mono_status = "PASS"
        mono_detail = "OK"
    results.append(
        ValidationResult(
            "Ablation row-count monotonicity from A2 (A1→A2 shrink expected)",
            mono_status,
            mono_detail,
        )
    )

    # Duplicates in pseudo-labeled base and confidence splits (known, inherited from A0)
    inherited_dupe_warnings = []
    for run_id in sorted(KNOWN_DUPE_DATASETS):
        stats = ablation_stats.get(run_id) or confidence_stats.get(run_id)
        if stats and stats.exists and not stats.error and stats.dupes > 0:
            inherited_dupe_warnings.append(f"{run_id}: {stats.dupes:,} dupes")
    results.append(
        ValidationResult(
            "Duplicate texts in pseudo-labeled base (inherited from A0)",
            "WARN" if inherited_dupe_warnings else "PASS",
            "OK" if not inherited_dupe_warnings else "; ".join(inherited_dupe_warnings),
        )
    )

    # Deduped ablation merges must stay duplicate-free
    deduped_dupe_failures = []
    for run_id in DEDUPED_ABLATION_IDS:
        stats = ablation_stats.get(run_id)
        if stats and stats.exists and not stats.error and stats.dupes > 0:
            deduped_dupe_failures.append(f"{run_id}: {stats.dupes:,} dupes")
    results.append(
        ValidationResult(
            "No duplicates in deduped ablation merges (A2–A5)",
            "PASS" if not deduped_dupe_failures else "FAIL",
            "OK" if not deduped_dupe_failures else "; ".join(deduped_dupe_failures),
        )
    )

    # Confidence split integrity
    base_rows = rows_for(ablation_stats, "A0")
    pool_rows = rows_for(confidence_stats, "train_pool")
    holdout_rows = rows_for(confidence_stats, "eval_holdout")
    c1_rows = rows_for(confidence_stats, "C1_train")
    if base_rows is not None and pool_rows is not None and holdout_rows is not None:
        split_sum = pool_rows + holdout_rows
        split_ok = split_sum == base_rows
        results.append(
            ValidationResult(
                "Confidence split (train_pool + holdout == A0 base)",
                "PASS" if split_ok else "FAIL",
                f"base={base_rows:,}, pool={pool_rows:,}, holdout={holdout_rows:,}, "
                f"sum={split_sum:,}",
            )
        )
    else:
        results.append(
            ValidationResult(
                "Confidence split (train_pool + holdout == A0 base)",
                "PASS",
                "Skipped — missing base, train_pool, or eval_holdout",
            )
        )

    if pool_rows is not None and c1_rows is not None:
        subset_ok = c1_rows <= pool_rows
        results.append(
            ValidationResult(
                "C1 filtered train subset of train pool",
                "PASS" if subset_ok else "FAIL",
                f"train_pool={pool_rows:,}, C1_train={c1_rows:,}",
            )
        )

    return results

def print_table1_helper(ablation_stats: dict[str, DatasetStats]) -> None:
    print_header("TABLE 1 HELPER (LaTeX row counts)")
    for run_id in ["A0", "A1", "A2", "A3", "A4", "A5"]:
        stats = ablation_stats.get(run_id)
        if stats and stats.exists and not stats.error:
            print(f"  {run_id}: {stats.rows:,} rows")
        else:
            print(f"  {run_id}: MISSING")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit experiment datasets: counts, labels, duplicates, cross-checks."
    )
    parser.add_argument(
        "--group",
        choices=["all", "ablation", "confidence", "augmentation"],
        default="all",
        help="Which dataset groups to audit (default: all).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any validation check fails.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    groups = (
        {"ablation", "confidence", "augmentation"}
        if args.group == "all"
        else {args.group}
    )

    print_header("DEEP-Dari Dataset Audit")
    print(f"Storage root: {PATHS['storage_root']}")
    print(f"Data root:    {PATHS['data_root']}")

    all_entries: list[DatasetEntry] = []
    if "ablation" in groups:
        all_entries.extend(ablation_entries())
    if "confidence" in groups:
        all_entries.extend(confidence_entries())
    if "augmentation" in groups:
        all_entries.extend(augmentation_entries())

    stats_map: dict[str, DatasetStats] = {}
    for entry in all_entries:
        stats_map[entry.entry_id] = analyze_entry(entry)

    ablation_stats = {e.entry_id: stats_map[e.entry_id] for e in ablation_entries()}
    confidence_stats = {e.entry_id: stats_map[e.entry_id] for e in confidence_entries()}
    augmentation_stats = {e.entry_id: stats_map[e.entry_id] for e in augmentation_entries()}

    if "ablation" in groups:
        print_header("ABLATION DATASETS (A0–A5)")
        ablation_list = ablation_entries()
        print_summary_table(ablation_list, stats_map)
        for entry in ablation_list:
            print_label_distribution(stats_map[entry.entry_id])

    if "confidence" in groups:
        print_header("CONFIDENCE DATASETS (C0–C3 + holdout)")
        confidence_list = confidence_entries()
        print_summary_table(confidence_list, stats_map)
        for entry in confidence_list:
            if entry.show_label_distribution:
                print_label_distribution(stats_map[entry.entry_id])

    if "augmentation" in groups:
        print_header("AUGMENTATION INPUTS (seed, template, BT)")
        augmentation_list = augmentation_entries()
        print_summary_table(augmentation_list, stats_map)

    if "ablation" in groups:
        print_table1_helper(ablation_stats)

    if groups.intersection({"ablation", "confidence", "augmentation"}) and (
        args.group == "all" or "ablation" in groups
    ):
        print_header("VALIDATION")
        validations = run_validations(ablation_stats, confidence_stats, augmentation_stats)
        failures = 0
        for result in validations:
            print(f"  [{result.status}] {result.name}")
            print(f"         {result.detail}")
            if not result.passed:
                failures += 1

        if args.strict and failures:
            print(f"\nStrict mode: {failures} validation check(s) failed.")
            sys.exit(1)

    print("\nAudit complete.\n")


if __name__ == "__main__":
    main()
