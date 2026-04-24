# Persian Emotion Classification with ParsBERT

This repository contains the training and inference code for a Persian/Dari emotion classification project built on top of `HooshvareLab/bert-base-parsbert-uncased`. The model is designed to classify social media text into eight emotion categories: `Hope`, `Happy`, `Neutral`, `Surprise`, `Disgust`, `Sad`, `Anger`, and `Fear`.

The codebase has been simplified to use a single training script, `scripts/train.py`, which covers all supported experiment settings through command-line presets.

## Repository Overview

- `scripts/train.py`: main training entrypoint
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
- `full_8label_aug`: training on the augmented full dataset

Example commands:

```bash
python scripts/train.py --mode full_8label
python scripts/train.py --mode full_7label
python scripts/train.py --mode full_8label_aug
python scripts/train.py --mode full_8label_aug --batch-size 16 --max-length 256 --use-dynamic-padding
python scripts/train.py --mode full_8label_aug --batch-size 8 --num-train-epochs 2
```

Padding strategy used in this project:

- `baseline_4k` is run with static padding
- `full_8label`, `full_7label`, and `full_8label_aug` are run with dynamic padding

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

Full-dataset run with dynamic padding:

```bash
python scripts/train.py --mode full_8label_aug --batch-size 16 --max-length 256 --use-dynamic-padding
```

## Outputs

Trained models are stored under `Models/`, while experiment-specific runs are written to `outputs/`. Each run saves metadata and evaluation results to make comparisons between experiments easier.

## Inference

To run prediction with a trained model:

```bash
python scripts/predict.py
```

By default, `scripts/predict.py` loads the model from `PATHS["fine_tuned_model"]`.

## License

Academic use only.
