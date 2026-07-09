# Persian Emotion Classification with ParsBERT

This repository contains the training and inference code for a Persian/Dari emotion classification project built on top of `HooshvareLab/bert-base-parsbert-uncased`. The model is designed to classify social media text into eight emotion categories: `Hope`, `Happy`, `Neutral`, `Surprise`, `Disgust`, `Sad`, `Anger`, and `Fear`.

The codebase has been simplified to use a single training script, `scripts/train.py`, which covers all supported experiment settings through command-line presets.

## Repository Overview

- `scripts/train.py`: main training entrypoint
- `scripts/run_ablation.py`: ablation experiment runner (one run at a time)
- `scripts/score_pseudo_labels.py`: score pseudo-labels with seed model confidence
- `scripts/filter_by_confidence.py`: filter scored dataset by confidence threshold
- `scripts/run_confidence_experiments.py`: confidence threshold training experiments
- `scripts/predict.py`: inference script for loading a trained model and running predictions
- `config/paths.py`: path configuration for datasets, models, and output directories
- `augmentations/fear_augmenter.py`: utilities related to fear-class augmentation
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

Results are saved under `outputs/ablation/<run_id>/`. Anchor runs A1 and A4 are skipped unless you pass `--force`.

## Confidence Threshold Experiments

Validate self-training quality by scoring the 400K pseudo-labeled dataset with the 4K seed model, then training on confidence-filtered subsets.

**Step 1 — Score** (review printed diagnostics before continuing):

```bash
python scripts/score_pseudo_labels.py
```

**Step 2 — Filter** (adjust thresholds based on your review):

```bash
python scripts/filter_by_confidence.py --threshold 0.7 0.8 0.9
```

**Step 3 — Train** (one experiment at a time):

```bash
python scripts/run_confidence_experiments.py --list
python scripts/run_confidence_experiments.py --run-id C0
python scripts/run_confidence_experiments.py --run-id C2 --build-filtered
```

| Run ID | Dataset | Purpose |
|--------|---------|---------|
| C0 | Unfiltered 400K | Baseline reference |
| C1 | conf >= 0.7 + agreement | Quality vs. size trade-off |
| C2 | conf >= 0.8 + agreement | Likely sweet spot |
| C3 | conf >= 0.9 + agreement | Highest precision, smallest set |

Results are saved under `outputs/confidence/<run_id>/`. Review `training_metadata.json` after each run before continuing.

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

## Inference

To run prediction with a trained model:

```bash
python scripts/predict.py
```

By default, `scripts/predict.py` loads the model from `PATHS["fine_tuned_model"]`.

## License

Academic use only.
