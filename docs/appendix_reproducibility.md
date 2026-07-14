# Appendix: Reproducibility Notes

Supporting material for the thesis experiments chapter. Script names and commands are listed here rather than in the chapter body.

## Environment

- Model: `HooshvareLab/bert-base-parsbert-uncased` (ParsBERT)
- Data root: set `SENTIMENT_DATA_ROOT` or `SENTIMENT_BASE_PATH` to the folder containing `Data/processed/`
- Storage root: set `SENTIMENT_STORAGE_ROOT` for models and `outputs/`

## Augmentation generation

| Emotion | Samples | Script |
|---------|--------:|--------|
| Fear | 9,000 | `augmentations/fear_augmenter.py` |
| Surprise | 9,000 | `augmentations/emotion_augmenter.py --emotion surprise` |
| Anger | 9,000 | `augmentations/emotion_augmenter.py --emotion anger` |
| Disgust | 9,000 | `augmentations/emotion_augmenter.py --emotion disgust` |

Default seed: 42. Each run writes a JSON sidecar with frame counts and distinct-n metrics.

## Dataset merge and audit

```bash
python scripts/merge_augmented_datasets.py --emotions surprise --output Data/processed/Combined_Labeled_Dataset_with_fearAug_surprise.csv
python scripts/audit_datasets.py
python scripts/audit_datasets.py --strict
```

## Back-translation pipeline

```bash
python scripts/run_backtranslation.py --step spotcheck
python scripts/run_backtranslation.py --step generate
python scripts/run_backtranslation.py --step merge
python scripts/run_ablation.py --run-id A5
```

Stop rule: abort full generation if spot-check shows fewer than 80% label-preserving samples per class.

## Ablation training

```bash
python scripts/run_ablation.py --list
python scripts/run_ablation.py --run-id A0
python scripts/run_ablation.py --run-id A2 --build-datasets
```

## Multi-seed stability (A4)

Fixed split (`--random-state 42`); vary training seed (`--seed`). Does not overwrite `outputs/ablation/A4/`.

```bash
python scripts/run_multiseed.py --run-id A4 --force --dry-run
python scripts/run_multiseed.py --run-id A4 --force
python scripts/run_multiseed.py --run-id A4 --aggregate-only
```

Per-seed: `outputs/ablation/A4/seed_<N>/training_metadata.json`  
Summary: `outputs/ablation/A4/multiseed_summary.json`

## Confidence experiments

```bash
python scripts/score_pseudo_labels.py
python scripts/prepare_confidence_splits.py --threshold 0.7 0.8 0.9
python scripts/run_confidence_experiments.py --run-id C1 --valid-eval
python scripts/evaluate_holdout.py --model outputs/confidence/C1
```

C1 was trained and validated on the fixed unfiltered holdout. C2/C3 were not run after C1's catastrophic holdout performance.

Use `--valid-eval` (default). Do not evaluate on a filtered test set.

## Output locations

- Ablation: `outputs/ablation/<run_id>/training_metadata.json`
- Multi-seed A4: `outputs/ablation/A4/seed_<N>/`, summary `outputs/ablation/A4/multiseed_summary.json`
- Confidence: `outputs/confidence/<run_id>/training_metadata.json`, optional `holdout_eval.json`
