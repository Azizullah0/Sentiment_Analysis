# Persian Emotion Classification with ParsBERT

This repository contains the training and inference code for a Persian/Dari emotion classification project built on top of `HooshvareLab/bert-base-parsbert-uncased`. The model is designed to classify social media text into eight emotion categories: `Hope`, `Happy`, `Neutral`, `Surprise`, `Disgust`, `Sad`, `Anger`, and `Fear`.

The codebase has been simplified to use a single training script, `scripts/train.py`, which covers all supported experiment settings through command-line presets.

## Repository Overview

- `scripts/train.py`: main training entrypoint
- `scripts/run_ablation.py`: ablation experiment runner (one run at a time)
- `scripts/run_backtranslation.py`: back-translation pipeline (spot-check → generate → merge → A5)
- `scripts/score_pseudo_labels.py`: score pseudo-labels with seed model confidence
- `scripts/filter_by_confidence.py`: filter scored dataset by confidence threshold
- `scripts/run_confidence_experiments.py`: confidence threshold training experiments
- `scripts/predict.py`: inference script for loading a trained model and running predictions
- `config/paths.py`: path configuration for datasets, models, and output directories
- `augmentations/fear_augmenter.py`: utilities related to fear-class augmentation
- `augmentations/back_translation_augmenter.py`: NLLB round-trip augmentation from 4K gold seed
- `utils/dataset_utils.py`: helper functions for dataset preparation

## Requirements

The project requires Python 3.8 or later and the packages listed in `requirements.txt`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data and Directory Configuration

Datasets are not included in the repository. Path resolution is handled in `config/paths.py`, with support for the following environment variables:

- `SENTIMENT_STORAGE_ROOT`: location for saved models, checkpoints, and experiment outputs
- `SENTIMENT_DATA_ROOT`: location of the project datasets
- `SENTIMENT_BASE_PATH`: backward-compatible fallback used by older setups

The training script expects the following processed files when using the default configuration:

- `Data/processed/Labeled_4K.csv`
- `Data/processed/Combined_Labeled_Dataset.csv`
- `Data/processed/Combined_Labeled_Dataset_with_fearAug.csv`
- `Data/processed/Combined_Labeled_Dataset_with_allAug.csv`
- `Data/processed/Combined_Labeled_Dataset_with_allAug_bt.csv` (after Phase 3 merge)

## Training

All training runs are handled through `scripts/train.py`.

Basic example:

```bash
python scripts/train.py --mode baseline_4k
```

Supported modes:

- `baseline_4k`: 8-label training on the 4K labeled dataset
- `full_8label`: training on the full labeled dataset with all eight classes
- `full_7label`: training on the full labeled dataset after removing the `Fear` class
- `full_8label_aug`: training on the fear-augmented full dataset
- `full_8label_all_aug`: training on the full dataset with all template augmentations (Fear + Surprise + Anger + Disgust)
- `full_8label_all_aug_bt`: training on template augmentations plus back-translation from 4K gold seed (A5)

Example commands:

```bash
python scripts/train.py --mode full_8label
python scripts/train.py --mode full_7label
python scripts/train.py --mode full_8label_aug
python scripts/train.py --mode full_8label_all_aug --batch-size 16 --max-length 256 --use-dynamic-padding
python scripts/train.py --mode full_8label_aug --batch-size 16 --max-length 256 --use-dynamic-padding
python scripts/train.py --mode full_8label_aug --batch-size 8 --num-train-epochs 2
```

Padding strategy used in this project:

- `baseline_4k` is run with static padding
- `full_8label`, `full_7label`, `full_8label_aug`, and `full_8label_all_aug` are run with dynamic padding

Common optional arguments:

- `--dataset-path`
- `--base-model`
- `--output-dir`
- `--final-model-dir`
- `--batch-size`
- `--num-train-epochs`
- `--learning-rate`
- `--max-length`
- `--use-dynamic-padding`
- `--fp16`
- `--no-fp16`

Recommended commands:

Baseline run with static padding:

```bash
python scripts/train.py --mode baseline_4k
```

Full-dataset run with all augmentations (best known setup: 86.12% accuracy, 0.857 Macro-F1):

```bash
python scripts/train.py --mode full_8label_all_aug --batch-size 16 --max-length 256 --use-dynamic-padding
```

## Outputs

Trained models are stored under `Models/`, while experiment-specific runs are written to `outputs/`. Each run saves metadata and evaluation results to make comparisons between experiments easier.

## Ablation Study

Run augmentation ablation experiments one at a time. Review `training_metadata.json` after each run before continuing.

List all runs:

```bash
python scripts/run_ablation.py --list
```

Preview commands (dry-run):

```bash
python scripts/run_ablation.py --dry-run
```

Run a single experiment:

```bash
python scripts/run_ablation.py --run-id A0
python scripts/run_ablation.py --run-id A2 --build-datasets
python scripts/run_ablation.py --run-id A3 --build-datasets
```

| Run ID | Dataset | Status |
|--------|---------|--------|
| A0 | Pseudo-labeled 400K only | Needs run |
| A1 | Fear augmentation only | Anchor: 85.75% acc, 0.836 Macro-F1 |
| A2 | Fear + Surprise | Needs run (use `--build-datasets`) |
| A3 | Fear + Surprise + Anger | Needs run (use `--build-datasets`) |
| A4 | Full stack (all augmentations) | Anchor: 86.12% acc, 0.857 Macro-F1 |
| A5 | A4 + back-translation from 4K seed | Needs run (after Phase 3 merge) |

Results are saved under `outputs/ablation/<run_id>/`. Anchor runs A1 and A4 are skipped unless you pass `--force`.

## Back-Translation (Phase 3)

Complements template augmentation with lexical paraphrase diversity from real 4K gold sentences via NLLB round-trip (`pes_Arab` ↔ `eng_Latn`). Compare **A5** against anchor **A4**.

**Stop rule:** If spot-check shows fewer than 80% label-preserving / natural samples per class, skip full generation and keep A4 as the best model.

**Step 1 — Spot-check** (review before continuing, ~30–60 min GPU):

```bash
python scripts/run_backtranslation.py --step spotcheck --dry-run
python scripts/run_backtranslation.py --step spotcheck
```

Review `Data/processed/bt_spotcheck_*.csv`: label still correct? Dari natural? No English leakage?

**Step 2 — Full generation** (overnight, ~6–12 h GPU):

```bash
python scripts/run_backtranslation.py --step generate
```

Target counts: Surprise 2000, Anger 2000, Disgust 2000, Sad 1500, Fear 1000 (from 4K originals only).

**Step 3 — Merge** onto existing allAug dataset:

```bash
python scripts/run_backtranslation.py --step merge
```

Writes `Data/processed/Combined_Labeled_Dataset_with_allAug_bt.csv`.

**Step 4 — Train A5**:

```bash
python scripts/run_backtranslation.py --step train
# or: python scripts/run_ablation.py --run-id A5
```

| Run ID | Dataset | Compare to |
|--------|---------|------------|
| A4 (anchor) | `Combined_Labeled_Dataset_with_allAug.csv` | 86.12% acc, 0.857 Macro-F1 |
| A5 | `Combined_Labeled_Dataset_with_allAug_bt.csv` | Win if Macro-F1 > 0.857 |

Single-class generation (manual):

```bash
python augmentations/back_translation_augmenter.py --emotion surprise --n 2000 --seed 42
python augmentations/back_translation_augmenter.py --emotion surprise --spot-check-only
```

## Confidence Threshold Experiments

Validate self-training quality by scoring the 400K pseudo-labeled dataset, filtering the **training pool only**, and evaluating on a **fixed unfiltered holdout** (same 20% split as ablation, seed=42).

**Important:** Do not train and test on the same filtered CSV — that inflates metrics (~98% accuracy). Use `--valid-eval` (default).

**Step 1 — Score** (review printed diagnostics before continuing):

```bash
python scripts/score_pseudo_labels.py
```

**Step 2 — Prepare splits** (fixed holdout + filtered train pools):

```bash
python scripts/prepare_confidence_splits.py --threshold 0.7 0.8 0.9
```

Creates:
- `Data/processed/eval_holdout_original.csv` — fixed 20% test (unfiltered)
- `Data/processed/train_pool_original.csv` — 80% train pool (unfiltered, for C0)
- `Data/processed/train_filtered_conf07.csv` etc. — filtered train pools only

**Step 3 — Train** (one experiment at a time, valid eval by default):

```bash
python scripts/run_confidence_experiments.py --list
python scripts/run_confidence_experiments.py --run-id C1 --valid-eval --prepare-splits
python scripts/run_confidence_experiments.py --run-id C2 --valid-eval
```

**Re-evaluate existing C1 model** (no retrain needed):

```bash
python scripts/evaluate_holdout.py --model outputs/confidence/C1
```

| Run ID | Train data | Eval data | Purpose |
|--------|------------|-----------|---------|
| C0 | `train_pool_original.csv` | `eval_holdout_original.csv` | Baseline on same holdout |
| C1 | `train_filtered_conf07.csv` | fixed holdout | conf >= 0.7 |
| C2 | `train_filtered_conf08.csv` | fixed holdout | conf >= 0.8 |
| C3 | `train_filtered_conf09.csv` | fixed holdout | conf >= 0.9 |

Results: `outputs/confidence/<run_id>/training_metadata.json` and optional `holdout_eval.json`.

Legacy (invalid) mode: `--legacy-eval` — random split inside filtered CSV; do not report in thesis.

## Merging Augmented Datasets

To create the full augmented training CSV from the latest fear-augmented dataset plus the generated Surprise, Anger, and Disgust datasets:

```bash
python scripts/merge_augmented_datasets.py --generate-missing
```

On a server where the data lives in Google Drive, point the project at that folder first:

```bash
export SENTIMENT_DATA_ROOT=/content/drive/MyDrive/Sentiment_Analysis
python scripts/merge_augmented_datasets.py --generate-missing
```

The script reads `Data/processed/Combined_Labeled_Dataset_with_fearAug.csv` and writes `Data/processed/Combined_Labeled_Dataset_with_allAug.csv`.

Build partial datasets for ablation (A2, A3):

```bash
python scripts/merge_augmented_datasets.py --emotions surprise --output Data/processed/Combined_Labeled_Dataset_with_fearAug_surprise.csv
python scripts/merge_augmented_datasets.py --emotions surprise,anger --output Data/processed/Combined_Labeled_Dataset_with_fearAug_surprise_anger.csv
```

Merge back-translation files onto allAug (also done by `run_backtranslation.py --step merge`):

```bash
python scripts/merge_augmented_datasets.py \
  --base-dataset Data/processed/Combined_Labeled_Dataset_with_allAug.csv \
  --no-template-aug \
  --extra-augmented-file Data/processed/augmented_bt_surprise_2000.csv \
  --extra-augmented-file Data/processed/augmented_bt_anger_2000.csv \
  --output Data/processed/Combined_Labeled_Dataset_with_allAug_bt.csv
```

## Inference

To run prediction with a trained model:

```bash
python scripts/predict.py
```

By default, `scripts/predict.py` loads the model from `PATHS["fine_tuned_model"]`.

## License

Academic use only.
