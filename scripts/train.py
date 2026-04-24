"""
Training entry point for the sentiment experiments in this repository.

Supported presets:

- baseline_4k: base 8-label training on the 4K labeled dataset
- full_8label: full dataset training with all 8 labels
- full_7label: full dataset training with Fear removed
- full_8label_aug: full dataset training on the augmented dataset
"""

import argparse
import datetime
import inspect
import json
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import MODEL_CONFIG, PATHS


logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
disable_progress_bar()


LABEL_NAMES_8 = {
    0: "Hope",
    1: "Happy",
    2: "Neutral",
    3: "Surprise",
    4: "Disgust",
    5: "Sad",
    6: "Anger",
    7: "Fear",
}

LABEL_NAMES_7 = {
    0: "Hope",
    1: "Happy",
    2: "Neutral",
    3: "Surprise",
    4: "Disgust",
    5: "Sad",
    6: "Anger",
}

BASE_FIXED_WEIGHTS = [11.09, 3.25, 4.83, 22.32, 14.07, 5.32, 15.04, 39.68]


class WeightedTrainer(Trainer):
    def __init__(self, model=None, class_weights=None, **kwargs):
        super().__init__(model=model, **kwargs)
        self.class_weights = class_weights
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        if labels is None:
            labels = inputs.pop("label", None)

        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

        if labels is None:
            loss = outputs.get("loss") if isinstance(outputs, dict) else getattr(outputs, "loss", None)
            if loss is None:
                raise ValueError(
                    "Expected a batch label tensor under 'labels' or 'label', but none was provided "
                    f"and the model output did not include a loss. Batch keys: {sorted(inputs.keys())}"
                )
            return (loss, outputs) if return_outputs else loss

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(description="Unified training script for sentiment experiments.")
    parser.add_argument(
        "--mode",
        choices=["baseline_4k", "full_8label", "full_7label", "full_8label_aug"],
        required=True,
        help="Experiment preset to run.",
    )
    parser.add_argument("--dataset-path", type=str, help="Override dataset CSV path.")
    parser.add_argument("--base-model", type=str, help="Override model/checkpoint used for initialization.")
    parser.add_argument("--output-dir", type=str, help="Override training output directory.")
    parser.add_argument("--final-model-dir", type=str, help="Override final saved model directory.")
    parser.add_argument("--batch-size", type=int, help="Override train/eval batch size.")
    parser.add_argument("--num-train-epochs", type=float, help="Override number of epochs.")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate.")
    parser.add_argument("--max-length", type=int, help="Override tokenizer max length.")
    parser.add_argument("--weight-decay", type=float, help="Override weight decay.")
    parser.add_argument("--warmup-steps", type=int, help="Override warmup steps.")
    parser.add_argument("--save-total-limit", type=int, help="Override number of checkpoints kept.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Train/test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument("--logging-steps", type=int, help="Override logging frequency.")
    parser.add_argument("--early-stopping-patience", type=int, help="Override early stopping patience.")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=None,
        help="Force fp16 on.",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Force fp16 off.",
    )
    parser.add_argument(
        "--use-dynamic-padding",
        action="store_true",
        help="Use dynamic padding instead of padding every sample to max_length.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show progress bars and more detailed trainer logs.",
    )
    return parser.parse_args()


def get_preset(mode):
    presets = {
        "baseline_4k": {
            "training_type": "base_8_label",
            "dataset_path": PATHS["Labeled_4K"],
            "base_model": PATHS.get("pretrained_base", "HooshvareLab/bert-base-parsbert-uncased"),
            "output_dir": PATHS["fine_tuned_model"],
            "final_model_dir": PATHS["fine_tuned_model"],
            "text_column": "clean",
            "label_column": "labels",
            "num_labels": 8,
            "label_names": LABEL_NAMES_8,
            "problem_type": "single_label_classification",
            "rename_text_to_clean": True,
            "derive_labels_from_label_text": True,
            "drop_fear": False,
            "compute_dynamic_class_weights": False,
            "fixed_class_weights": BASE_FIXED_WEIGHTS,
            "learning_rate": MODEL_CONFIG.get("learning_rate", 2e-5),
            "batch_size": MODEL_CONFIG.get("batch_size", 16),
            "num_train_epochs": MODEL_CONFIG.get("num_train_epochs", 5),
            "weight_decay": 0.02,
            "warmup_steps": None,
            "logging_steps": None,
            "logging_strategy": "epoch",
            "logging_dir": os.path.join(PATHS["fine_tuned_model"], "logs"),
            "metric_for_best_model": "f1_macro",
            "greater_is_better": True,
            "save_total_limit": 1,
            "fp16": True,
            "push_to_hub": False,
            "dataloader_drop_last": False,
            "dataloader_num_workers": 4,
            "logging_first_step": False,
            "early_stopping_patience": 2,
            "timestamp_output": False,
            "tokenizer_use_fast": False,
            "padding": "max_length",
            "description_lines": [
                "Base 8-label training from pretrained ParsBERT.",
            ],
        },
        "full_8label": {
            "training_type": "full_8_label",
            "dataset_path": PATHS["Combined_Labeled_Dataset"],
            "base_model": PATHS["fine_tuned_model"],
            "output_dir": os.path.join(PATHS["outputs"], f"full_8label_{timestamp_now()}"),
            "final_model_dir": None,
            "text_column": "text",
            "label_column": "label_id",
            "num_labels": 8,
            "label_names": LABEL_NAMES_8,
            "problem_type": None,
            "rename_text_to_clean": False,
            "derive_labels_from_label_text": False,
            "drop_fear": False,
            "compute_dynamic_class_weights": True,
            "fixed_class_weights": None,
            "learning_rate": 1e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "logging_steps": 100,
            "logging_strategy": "steps",
            "logging_dir": None,
            "metric_for_best_model": "f1_macro",
            "greater_is_better": True,
            "save_total_limit": 1,
            "fp16": True,
            "push_to_hub": False,
            "dataloader_drop_last": False,
            "dataloader_num_workers": None,
            "logging_first_step": True,
            "early_stopping_patience": 3,
            "timestamp_output": True,
            "output_prefix": "full_8label",
            "tokenizer_use_fast": None,
            "padding": "max_length",
            "description_lines": [
                "Full-dataset 8-label training.",
                "Uses the same full dataset as full_7label, without removing Fear.",
            ],
        },
        "full_7label": {
            "training_type": "full_7_label",
            "dataset_path": PATHS["Combined_Labeled_Dataset"],
            "base_model": PATHS["fine_tuned_model"],
            "output_dir": os.path.join(PATHS["outputs"], f"full_7label_{timestamp_now()}"),
            "final_model_dir": None,
            "text_column": "text",
            "label_column": "label_id",
            "num_labels": 7,
            "label_names": LABEL_NAMES_7,
            "problem_type": None,
            "rename_text_to_clean": False,
            "derive_labels_from_label_text": False,
            "drop_fear": True,
            "compute_dynamic_class_weights": True,
            "fixed_class_weights": None,
            "learning_rate": 1e-5,
            "batch_size": 16,
            "num_train_epochs": 4,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "logging_steps": 100,
            "logging_strategy": "steps",
            "logging_dir": None,
            "metric_for_best_model": "f1_macro",
            "greater_is_better": True,
            "save_total_limit": 1,
            "fp16": True,
            "push_to_hub": False,
            "dataloader_drop_last": False,
            "dataloader_num_workers": None,
            "logging_first_step": True,
            "early_stopping_patience": 3,
            "timestamp_output": True,
            "output_prefix": "full_7label",
            "tokenizer_use_fast": None,
            "padding": "max_length",
            "description_lines": [
                "Full-dataset 7-label training.",
                "Uses the same full dataset as full_8label, with Fear removed before training.",
            ],
        },
        "full_8label_aug": {
            "training_type": "full_8_label_augmented",
            "dataset_path": PATHS["Combined_Labeled_Dataset_with_fearAug"],
            "base_model": PATHS["parsbert_emotion"],
            "output_dir": os.path.join(PATHS["outputs"], f"full_8label_aug_{timestamp_now()}"),
            "final_model_dir": PATHS["incremental_finetuned_model"],
            "text_column": "clean",
            "label_column": "label_id",
            "num_labels": 8,
            "label_names": LABEL_NAMES_8,
            "problem_type": None,
            "rename_text_to_clean": False,
            "derive_labels_from_label_text": False,
            "drop_fear": False,
            "compute_dynamic_class_weights": True,
            "fixed_class_weights": None,
            "learning_rate": 1e-5,
            "batch_size": MODEL_CONFIG.get("batch_size", 16),
            "num_train_epochs": 1,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "logging_steps": 50,
            "logging_strategy": "steps",
            "logging_dir": None,
            "metric_for_best_model": "f1_macro",
            "greater_is_better": True,
            "save_total_limit": 2,
            "fp16": True,
            "push_to_hub": False,
            "dataloader_drop_last": True,
            "dataloader_num_workers": None,
            "logging_first_step": False,
            "early_stopping_patience": 3,
            "timestamp_output": True,
            "output_prefix": "full_8label_aug",
            "tokenizer_use_fast": None,
            "padding": "max_length",
            "description_lines": [
                "Full-dataset 8-label training on the augmented dataset.",
                "Uses dynamic class weights for the augmented imbalanced dataset.",
            ],
        },
    }
    return presets[mode]


def timestamp_now():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


def standardize_strategy_key(training_kwargs):
    parameter_names = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameter_names:
        training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    return training_kwargs


def resolve_runtime_config(args):
    config = get_preset(args.mode).copy()

    if args.dataset_path:
        config["dataset_path"] = os.path.abspath(os.path.expanduser(args.dataset_path))
    if args.base_model:
        config["base_model"] = os.path.abspath(os.path.expanduser(args.base_model))
    if args.output_dir:
        config["output_dir"] = os.path.abspath(os.path.expanduser(args.output_dir))
    if args.final_model_dir:
        config["final_model_dir"] = os.path.abspath(os.path.expanduser(args.final_model_dir))
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_train_epochs is not None:
        config["num_train_epochs"] = args.num_train_epochs
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.max_length is not None:
        config["max_length"] = args.max_length
    else:
        config["max_length"] = MODEL_CONFIG.get("max_length", 512)
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay
    if args.warmup_steps is not None:
        config["warmup_steps"] = args.warmup_steps
    if args.save_total_limit is not None:
        config["save_total_limit"] = args.save_total_limit
    if args.logging_steps is not None:
        config["logging_steps"] = args.logging_steps
    if args.early_stopping_patience is not None:
        config["early_stopping_patience"] = args.early_stopping_patience
    if args.fp16 is not None:
        config["fp16"] = args.fp16
    if args.use_dynamic_padding:
        config["padding"] = False

    return config


def load_dataframe(config):
    dataset_path = config["dataset_path"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip()

    if "Label" in df.columns:
        typo_count = (df["Label"] == "Suprise").sum()
        if typo_count:
            print(f"Found {typo_count} rows with 'Suprise'. Normalizing to 'Surprise'.")
            df["Label"] = df["Label"].replace("Suprise", "Surprise")

    if config["mode"] == "baseline_4k":
        label_mapping = {
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
        df["labels"] = df["Label"].map(label_mapping)
        initial_count = len(df)
        df = df.dropna(subset=["labels", "text"])
        print(f"Dropped {initial_count - len(df)} rows due to NaN labels or text.")
        df["labels"] = df["labels"].astype(int)
        df = df.rename(columns={"text": "clean"})
    else:
        if "clean" in df.columns and "text" not in df.columns and config["text_column"] == "text":
            df = df.rename(columns={"clean": "text"})

        required_columns = {config["text_column"], config["label_column"]}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Dataset must contain columns: {sorted(required_columns)}; missing: {sorted(missing)}")

        print("Validating dataset...")
        print(f"NaN in '{config['text_column']}': {df[config['text_column']].isna().sum()}")
        print(f"NaN in '{config['label_column']}': {df[config['label_column']].isna().sum()}")
        initial_size = len(df)
        df = df.dropna(subset=[config["text_column"], config["label_column"]])
        if initial_size - len(df) > 0:
            print(f"Removed {initial_size - len(df)} rows with NaN values")

        valid_max = 7 if not config["drop_fear"] else 6
        print(f"Unique labels: {sorted(df[config['label_column']].unique())}")
        df = df[df[config["label_column"]].between(0, 7)]
        if config["drop_fear"]:
            df = df[df[config["label_column"]] != 7]
            df = df[df[config["label_column"]].between(0, 6)]
            if df.empty:
                raise ValueError("Filtered dataset is empty after removing Fear.")

    return df


def analyze_distribution(df, config):
    label_counts = df[config["label_column"]].value_counts().sort_index()
    total = len(df)

    print("\nLabel distribution:")
    for label_id, count in label_counts.items():
        percentage = (count / total) * 100
        label_name = config["label_names"].get(label_id, f"Label_{label_id}")
        print(f"  {label_name} ({label_id}): {count} ({percentage:.2f}%)")

    if len(label_counts) > 0:
        max_count = label_counts.max()
        min_count = label_counts.min()
        print("\nImbalance Analysis:")
        print(f"  Majority class: {max_count} samples")
        print(f"  Minority class: {min_count} samples")
        print(f"  Imbalance ratio: {max_count / min_count:.1f}:1")

    return label_counts


def create_datasets(df, config, args, tokenizer):
    train_df, eval_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df[config["label_column"]],
        random_state=args.random_state,
    )
    print(f"\nTrain size: {len(train_df)} | Eval size: {len(eval_df)}")

    train_ds = Dataset.from_pandas(train_df[[config["text_column"], config["label_column"]]])
    eval_ds = Dataset.from_pandas(eval_df[[config["text_column"], config["label_column"]]])

    def tokenize_batch(batch):
        tokenized = tokenizer(
            batch[config["text_column"]],
            truncation=True,
            padding=config["padding"],
            max_length=config["max_length"],
        )
        tokenized["labels"] = batch[config["label_column"]]
        return tokenized

    print("\nTokenizing dataset...")
    train_ds = train_ds.map(tokenize_batch, batched=True)
    eval_ds = eval_ds.map(tokenize_batch, batched=True)

    columns_to_remove = [config["text_column"]]
    if config["label_column"] != "labels":
        columns_to_remove.append(config["label_column"])
    train_ds = train_ds.remove_columns(columns_to_remove)
    eval_ds = eval_ds.remove_columns(columns_to_remove)
    train_ds.set_format("torch")
    eval_ds.set_format("torch")
    return train_df, eval_df, train_ds, eval_ds


def build_model_and_tokenizer(config):
    tokenizer_kwargs = {}
    if config["tokenizer_use_fast"] is not None:
        tokenizer_kwargs["use_fast"] = config["tokenizer_use_fast"]

    print(f"Loading tokenizer from: {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], **tokenizer_kwargs)

    model_kwargs = {"num_labels": config["num_labels"]}
    if config["problem_type"]:
        model_kwargs["problem_type"] = config["problem_type"]
    if config["mode"] == "full_7label":
        model_kwargs["ignore_mismatched_sizes"] = True

    print(f"Loading model from: {config['base_model']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        **model_kwargs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    return tokenizer, model, device


def build_class_weights(config, label_counts, device):
    if config["compute_dynamic_class_weights"]:
        total = sum(label_counts.values)
        weights = [total / label_counts.get(i, 1) for i in range(config["num_labels"])]
    elif config["fixed_class_weights"] is not None:
        weights = config["fixed_class_weights"]
    else:
        return None, None

    print("\nClass weights:")
    for i, weight in enumerate(weights):
        label_name = config["label_names"].get(i, f"Label_{i}")
        print(f"  {label_name}: {weight:.2f}x")
    return weights, torch.tensor(weights, dtype=torch.float).to(device)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
    }


def build_training_args(config):
    training_kwargs = {
        "output_dir": config["output_dir"],
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": config["learning_rate"],
        "per_device_train_batch_size": config["batch_size"],
        "per_device_eval_batch_size": config["batch_size"],
        "num_train_epochs": config["num_train_epochs"],
        "weight_decay": config["weight_decay"],
        "load_best_model_at_end": True,
        "metric_for_best_model": config["metric_for_best_model"],
        "greater_is_better": config["greater_is_better"],
        "report_to": "none",
        "fp16": config["fp16"],
        "save_total_limit": config["save_total_limit"],
        "push_to_hub": config["push_to_hub"],
        "dataloader_drop_last": config["dataloader_drop_last"],
        "disable_tqdm": not config.get("verbose", False),
    }

    if config["logging_dir"]:
        training_kwargs["logging_dir"] = config["logging_dir"]
    if config["logging_strategy"]:
        training_kwargs["logging_strategy"] = config["logging_strategy"]
    if config["logging_steps"] is not None:
        training_kwargs["logging_steps"] = config["logging_steps"]
    if config["logging_first_step"]:
        training_kwargs["logging_first_step"] = True
    if config["warmup_steps"] is not None:
        training_kwargs["warmup_steps"] = config["warmup_steps"]
    if config["dataloader_num_workers"] is not None:
        training_kwargs["dataloader_num_workers"] = config["dataloader_num_workers"]

    return TrainingArguments(**standardize_strategy_key(training_kwargs))


def ensure_output_paths(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    if config["final_model_dir"]:
        os.makedirs(config["final_model_dir"], exist_ok=True)


def save_metadata(config, metadata, destination_dir):
    with open(os.path.join(destination_dir, "training_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)


def main():
    args = parse_args()
    config = resolve_runtime_config(args)
    config["mode"] = args.mode
    config["verbose"] = args.verbose

    if args.verbose:
        print("Verbose logging enabled.")
    else:
        print("Quiet logging enabled. Progress bars are hidden; only key training info will be shown.")

    print(f"Running mode: {args.mode}")
    for line in config["description_lines"]:
        print(line)
    print(f"Storage root: {PATHS['storage_root']}")
    print(f"Data root: {PATHS['data_root']}")
    print(f"Dataset path: {config['dataset_path']}")
    print(f"Base model: {config['base_model']}")

    ensure_output_paths(config)
    df = load_dataframe(config)
    print(f"Loaded dataset: {len(df)} rows")
    label_counts = analyze_distribution(df, config)
    tokenizer, model, device = build_model_and_tokenizer(config)
    train_df, eval_df, train_ds, eval_ds = create_datasets(df, config, args, tokenizer)
    weights, weights_tensor = build_class_weights(config, label_counts, device)
    training_args = build_training_args(config)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=weights_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])],
    )

    print(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Train batches: {len(trainer.get_train_dataloader())}")
    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating...")
    evaluation_results = trainer.evaluate()
    predictions = trainer.predict(eval_ds)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    per_class_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)

    print("\nResults:")
    for metric_name, metric_value in evaluation_results.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print("Per-class F1:")
    for i, score in enumerate(per_class_f1):
        print(f"  {config['label_names'].get(i, i)}: {score:.4f}")

    final_model_dir = config["final_model_dir"] or config["output_dir"]
    print(f"\nSaving model to: {final_model_dir}")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    metadata = {
        "training_type": config["training_type"],
        "mode": args.mode,
        "base_model": config["base_model"],
        "training_date": datetime.datetime.now().isoformat(),
        "dataset_path": config["dataset_path"],
        "dataset_size": int(len(df)),
        "train_samples": int(len(train_df)),
        "eval_samples": int(len(eval_df)),
        "label_distribution": {int(k): int(v) for k, v in label_counts.to_dict().items()},
        "class_weights": None if weights is None else {int(i): float(w) for i, w in enumerate(weights)},
        "final_metrics": {k: float(v) for k, v in evaluation_results.items()},
        "per_class_f1": {
            config["label_names"].get(i, str(i)): float(score) for i, score in enumerate(per_class_f1)
        },
        "output_dir": config["output_dir"],
        "final_model_dir": final_model_dir,
        "arguments": vars(args),
        "resolved_config": {
            "batch_size": config["batch_size"],
            "num_train_epochs": config["num_train_epochs"],
            "learning_rate": config["learning_rate"],
            "max_length": config["max_length"],
            "weight_decay": config["weight_decay"],
            "fp16": config["fp16"],
            "padding": "dynamic" if config["padding"] is False else config["padding"],
        },
    }
    save_metadata(config, metadata, final_model_dir)

    if final_model_dir != config["output_dir"]:
        save_metadata(config, metadata, config["output_dir"])

    print("Done.")


if __name__ == "__main__":
    main()
