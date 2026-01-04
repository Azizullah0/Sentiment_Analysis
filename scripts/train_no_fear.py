"""
Train a 7-label model (Fear removed) starting from an existing 8-label checkpoint.
Adjust DATA_PATH, BASE_MODEL, OUTPUT_ROOT to your environment.
"""

# Paths to adjust for your environment
DATA_PATH = "/content/drive/MyDrive/Sentiment_Analysis/Data/processed/Combined_Labeled_Dataset.csv"
BASE_MODEL = "/content/drive/MyDrive/Sentiment_Analysis/Models/parsbert_emotion"  # 8-label fine-tuned checkpoint
OUTPUT_ROOT = "/content/drive/MyDrive/Sentiment_Analysis/outputs"

import os
import json
import datetime
import warnings
import logging

logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)


# ------- Data loading -------
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "clean" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"clean": "text"})
    if not {"text", "label_id"}.issubset(df.columns):
        raise ValueError("Dataset must have columns: text, label_id")

    df = df.dropna(subset=["text", "label_id"])
    df = df[df["label_id"] != 7]          # drop Fear
    df = df[df["label_id"].between(0, 6)] # keep 0-6
    if df.empty:
        raise ValueError("Filtered dataset is empty after removing Fear.")
    return df


def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
    }


class WeightedTrainer(Trainer):
    def __init__(self, model=None, class_weights=None, **kwargs):
        super().__init__(model=model, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    df = load_data()
    print(f"Loaded dataset: {len(df)} rows")

    label_counts = df["label_id"].value_counts().sort_index()
    total = len(df)
    label_names = {0: "Hope", 1: "Happy", 2: "Neutral", 3: "Suprise", 4: "Disgust", 5: "Sad", 6: "Anger"}
    print("\nLabel distribution (Fear removed):")
    for lid, cnt in label_counts.items():
        print(f"  {label_names.get(lid, lid)} ({lid}): {cnt} ({100*cnt/total:.2f}%)")

    train_df, eval_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)
    print(f"\nTrain size: {len(train_df)} | Eval size: {len(eval_df)}")

    # Model / Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=7,
        ignore_mismatched_sizes=True,  # reinit head for 7 classes
    )

    # Datasets
    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]])
    eval_ds = Dataset.from_pandas(eval_df[["text", "label_id"]])

    def tok_fn(batch):
        tok = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        tok["labels"] = batch["label_id"]
        return tok

    train_ds = train_ds.map(tok_fn, batched=True).remove_columns(["text", "label_id"])
    eval_ds = eval_ds.map(tok_fn, batched=True).remove_columns(["text", "label_id"])
    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    # Class weights
    weights = [total / label_counts.get(i, 1) for i in range(7)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    model.to(device)
    print("\nClass weights:")
    for i, w in enumerate(weights):
        print(f"  {label_names.get(i, i)}: {w:.2f}x")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.join(OUTPUT_ROOT, f"no_fear_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",
        fp16=True,
        logging_steps=100,
        logging_first_step=True,
        save_total_limit=1,
        push_to_hub=False,
        warmup_steps=100,
        dataloader_drop_last=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=weights_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"\ntrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"train batches: {len(trainer.get_train_dataloader())}")

    print("\nStarting 7-label training...")
    trainer.train()

    print("\nEvaluating...")
    eval_metrics = trainer.evaluate()
    preds = trainer.predict(eval_ds)
    pred_labels = np.argmax(preds.predictions, axis=1)
    true_labels = preds.label_ids
    per_class_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)

    print("\nResults (Fear removed):")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Per-class F1:")
    for i, f1_cls in enumerate(per_class_f1):
        print(f"  {label_names.get(i, i)}: {f1_cls:.4f}")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "training_type": "7_label_no_fear",
        "base_model": BASE_MODEL,
        "training_date": datetime.datetime.now().isoformat(),
        "dataset_size": len(df),
        "train_samples": len(train_df),
        "eval_samples": len(eval_df),
        "label_distribution": label_counts.to_dict(),
        "class_weights": {i: float(w) for i, w in enumerate(weights)},
        "final_metrics": eval_metrics,
        "per_class_f1": {label_names[i]: float(score) for i, score in enumerate(per_class_f1)},
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved 7-label model to {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
