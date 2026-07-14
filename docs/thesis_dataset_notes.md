# Thesis Dataset Notes

Copy-ready text for thesis tables and limitations. Row counts verified by `python scripts/audit_datasets.py` on the server dataset (July 2026).

## Table 1 — Ablation dataset sizes

Use these row counts as-is in the thesis. Do not assume monotonic growth across steps.

| Run | Description | Rows |
|-----|-------------|-----:|
| A0 | Pseudo-labeled 400K only | 391,691 |
| A1 | Fear augmentation only | 400,691 |
| A2 | Fear + Surprise | 394,476 |
| A3 | Fear + Surprise + Anger | 403,476 |
| A4 | Full template stack | 412,476 |
| A5 | A4 + back-translation | 413,914 |

Anchor results (unchanged): A1 — 85.75% accuracy, 0.836 Macro-F1; A4 — 86.12% accuracy, 0.857 Macro-F1.

## Data quality and limitations (Methods or Limitations)

The pseudo-labeled base dataset (A0) contains 15,215 duplicate normalized texts (3.9% of rows). These duplicates are inherited by A1 and by the confidence experiment train/holdout splits derived from A0. Duplicate texts may slightly overweight repeated samples during training on A0, A1, and confidence runs C0–C3. Merges for A2–A5 apply deduplication on normalized text when combining template and back-translation augmentations, so those datasets are duplicate-free.

Ablation row counts are not strictly monotonic: A1 (400,691 rows) exceeds A2 (394,476 rows) because the A2 merge step removes 6,215 overlapping rows when Surprise augmentation is added. Report actual dataset sizes per run rather than assuming each augmentation step only adds rows.

Back-translation generation did not reach the target counts per emotion class. The net contribution to A5 was +1,438 rows over A4 (1,439 generated, 1 removed as a duplicate on merge). The A5 versus A4 comparison remains valid using the merged dataset as trained.

For confidence-threshold experiments, filtering the training pool by pseudo-label confidence increases class imbalance: the Happy share rises from 31.5% in the unfiltered train pool to 41.5% (conf >= 0.7), 46.6% (conf >= 0.8), and 59.9% (conf >= 0.9), while rare classes such as Anger become sparser at high thresholds. **C1** was trained and validated on the fixed unfiltered holdout (75.56% accuracy, 0.648 Macro-F1, 0.248 Fear F1). **C2 and C3** were not trained after C1's catastrophic holdout performance.

## Confidence experiment splits (ready to use)

Do not rebuild splits if these files already exist on disk:

| File | Rows | Role |
|------|-----:|------|
| `train_pool_original.csv` | 313,352 | C0 training |
| `eval_holdout_original.csv` | 78,339 | Fixed holdout (all C runs) |
| `train_filtered_conf07.csv` | 148,924 | C1 training (trained) |
| `train_filtered_conf08.csv` | 114,702 | C2 (prepared, not trained) |
| `train_filtered_conf09.csv` | 65,488 | C3 (prepared, not trained) |

C1 was trained and evaluated on the fixed unfiltered holdout. C2/C3 were not run after C1 failed (75.56% acc, 0.648 Macro-F1, 0.248 Fear F1 vs A4: 86.12%, 0.857, 0.950).

```bash
python scripts/run_confidence_experiments.py --run-id C1 --valid-eval
python scripts/evaluate_holdout.py --model outputs/confidence/C1
```

C2/C3 commands exist but were not executed in this thesis.

Omit `--prepare-splits` unless you intentionally regenerate splits from scratch (would require re-scoring and is not needed for the current experiment).
