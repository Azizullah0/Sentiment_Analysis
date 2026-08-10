# DEEP-Dari — Persian/Dari Emotion Classification

Eight-class emotion classification for Dari/Persian social media (`Hope`, `Happy`, `Neutral`, `Surprise`, `Disgust`, `Sad`, `Anger`, `Fear`) on top of [`HooshvareLab/bert-base-parsbert-uncased`](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased), with culturally grounded minority-class augmentation and a YouTube monitoring deployment.

| Best model (A4) | Value |
|-----------------|------:|
| Accuracy | **86.12%** |
| Macro-F1 | **0.857** |
| Fear F1 | **0.950** |

**Cite:** see [`CITATION.cff`](CITATION.cff) (conference paper under review; manuscript not in this repo)

Training is driven by `scripts/train.py` and ablation runners; live labeling lives under [`deployment/`](deployment/README.md) (FastAPI + bilingual EN/FA dashboard).

## Repository Overview

There is **one** training script (`scripts/train.py`). The other `scripts/*.py` files prepare data, call `train.py` for named experiments, or evaluate / predict.

**Train**
- `scripts/train.py` — ParsBERT fine-tuning (used by all experiment runners)

**Experiment runners** (invoke `train.py`)
- `scripts/run_ablation.py` — A0–A5 ablation (one run at a time)
- `scripts/run_multiseed.py` — multi-seed A4 stability (seeds 41–45)
- `scripts/run_confidence_experiments.py` — C0–C3 confidence-threshold runs
- `scripts/run_backtranslation.py` — back-translation pipeline → A5

**Data prep / checks**
- `scripts/merge_augmented_datasets.py` — merge template/BT rows into ablation CSVs
- `scripts/score_pseudo_labels.py` — score ~400k pseudo-labels with seed-model confidence
- `scripts/prepare_confidence_splits.py` — fixed holdout + filtered train pools
- `scripts/audit_datasets.py` — dataset inventory and split/merge checks

**Eval / inference**
- `scripts/evaluate_holdout.py` — score a saved checkpoint on the fixed holdout
- `scripts/predict.py` — run predictions on new text

**Shared code**
- `config/paths.py` — dataset / model / output paths
- `augmentations/` — Fear, Surprise/Anger/Disgust templates; NLLB back-translation
- `deployment/` — YouTube labeling API + bilingual EN/FA dashboard ([deployment/README.md](deployment/README.md))

## Requirements

The project requires Python 3.8 or later and the packages listed in `requirements.txt`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

For YouTube labeling / API serving, also install:

```bash
pip install -r deployment/requirements.txt
```

Dashboard (after `npm install && npm run build` in `deployment/dashboard`):

```bash
export YOUTUBE_API_KEY=your-key
uvicorn deployment.api:app --host 0.0.0.0 --port 8000
# open http://localhost:8000  — EN/FA toggle in the header
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

## Dataset Audit

Before updating thesis tables, verify row counts, label distributions, duplicates, and merge integrity:

```bash
python scripts/audit_datasets.py
python scripts/audit_datasets.py --group ablation
python scripts/audit_datasets.py --strict
```

The script prints a **Table 1 helper** with A0–A5 row counts from disk and validation checks (e.g. A5 = A4 + BT files, confidence split sums).

**Interpreting validation results:** Two outcomes that previously showed as `FAIL` are expected for this project and do not require dataset rebuilds:

- **`[EXPECTED] Ablation row-count monotonicity from A2`** — A2 (394,476) can be smaller than A1 (400,691) because `merge_augmented_datasets.py` deduplicates when adding Surprise; A2–A5 should stay duplicate-free.
- **`[WARN] Duplicate texts in pseudo-labeled base`** — A0/A1 and confidence splits inherit ~15,215 duplicate normalized texts from the pseudo-labeled base. Document in the thesis; do not dedupe and re-train completed ablation runs.

Verified Table 1 row counts live in local notes (not published); use `scripts/audit_datasets.py` for on-disk checks.

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

| Run ID | Dataset | Rows | Status |
|--------|---------|-----:|--------|
| A0 | Pseudo-labeled 400K only | 391,691 | Complete |
| A1 | Fear augmentation only | 400,691 | Anchor: 85.75% acc, 0.836 Macro-F1 |
| A2 | Fear + Surprise | 394,476 | Complete |
| A3 | Fear + Surprise + Anger | 403,476 | Complete |
| A4 | Full stack (all augmentations) | 412,476 | Anchor: 86.12% acc, 0.857 Macro-F1 |
| A5 | A4 + back-translation from 4K seed | 413,914 | Complete |

Results are saved under `outputs/ablation/<run_id>/`. Anchor runs A1 and A4 are skipped unless you pass `--force`.

### Multi-seed stability (A4)

Train A4 multiple times with different **training seeds** while keeping the **split seed** fixed (`random_state=42`). Per-seed outputs go to `outputs/ablation/A4/seed_<N>/` so the anchor model at `outputs/ablation/A4/` is not overwritten.

```bash
python scripts/run_multiseed.py --run-id A4 --force --dry-run
python scripts/run_multiseed.py --run-id A4 --force
python scripts/run_multiseed.py --run-id A4 --aggregate-only   # re-summarize existing runs
```

Summary table: `outputs/ablation/A4/multiseed_summary.json` (mean ± std for accuracy, Macro-F1, Fear F1).

**Completed run (July 2026):** mean accuracy 86.16% ± 0.05 pp, mean Macro-F1 0.857 ± 0.001, mean Fear F1 0.947 ± 0.003 (seeds 41–45). Anchor A4 (86.12%, 0.857) falls within the observed range.

Single manual re-run with a custom seed:

```bash
python scripts/run_ablation.py --run-id A4 --force --seed 43 --split-seed 42 \
  --output-dir outputs/ablation/A4/seed_43
```

`train.py` flags: `--seed` (training RNG), `--random-state` (stratified split only).

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

**Important:** Training always uses the fixed unfiltered holdout for evaluation. Do not train and test on the same filtered CSV — that inflates metrics (~98% accuracy).

**Step 1 — Score** (review printed diagnostics before continuing):

```bash
python scripts/score_pseudo_labels.py
```

**Step 2 — Prepare splits** (only if CSVs are missing; skip if audit shows OK):

```bash
python scripts/prepare_confidence_splits.py --threshold 0.7 0.8 0.9
```

Creates:
- `Data/processed/eval_holdout_original.csv` — fixed 20% test (unfiltered)
- `Data/processed/train_pool_original.csv` — 80% train pool (unfiltered, for C0)
- `Data/processed/train_filtered_conf07.csv` etc. — filtered train pools only

If these files already exist (verified by `audit_datasets.py`), **do not rebuild** — use them as-is so C0–C3 stay aligned with the ablation split protocol.

**Step 3 — Train** (C1 was run and validated; C2/C3 were not run after C1's holdout failure):

```bash
python scripts/run_confidence_experiments.py --list
python scripts/run_confidence_experiments.py --run-id C1
```

**Re-evaluate existing C1 model** (no retrain needed):

```bash
python scripts/evaluate_holdout.py --model outputs/confidence/C1
```

C2/C3 were not trained. Higher thresholds would retain even fewer minority-class samples.

| Run ID | Train data | Eval data | Status |
|--------|------------|-----------|--------|
| C0 | `train_pool_original.csv` | `eval_holdout_original.csv` | Optional baseline |
| C1 | `train_filtered_conf07.csv` | fixed holdout | **Trained** (75.56% acc, 0.648 Macro-F1) |
| C2 | `train_filtered_conf08.csv` | fixed holdout | Not trained |
| C3 | `train_filtered_conf09.csv` | fixed holdout | Not trained |

Results: `outputs/confidence/<run_id>/training_metadata.json` and optional `holdout_eval.json`.

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
