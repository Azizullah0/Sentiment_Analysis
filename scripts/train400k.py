import sys
import os
import logging
import warnings
from transformers import AutoTokenizer
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.dataset_utils import load_dataset, split_dataset, tokenize_datasets


# ================================
# 1. Load Dataset (+ optional Fear augmentation)
# ================================
df = load_dataset('datasets/Training_Ready_Labeled.csv', label_col='label_id')

# If augmented Fear file exists, merge it
aug_path = 'datasets/fear_augmented.csv'
if os.path.exists(aug_path):
    aug_df = pd.read_csv(aug_path)
    df = pd.concat([df, aug_df], ignore_index=True)
    print(f"✅ Augmented Fear samples added: {len(aug_df)} rows")

train_df, test_df = split_dataset(df)

train_dataset, test_dataset, tokenizer =AutoTokenizer.from_pretrained("/content/drive/MyDrive/parsbert_emotion")


# ================================
# 2. Model and Data Collator
# ================================
num_labels = len(df['labels'].unique())
model = AutoModelForSequenceClassification.from_pretrained(
    "/content/drive/MyDrive/parsbert_emotion",
    num_labels=num_labels
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ================================
# 3. Compute Class Weights
# ================================
label_counts = df['label_id'].value_counts().sort_index().values
total = sum(label_counts)
weights = [total / c for c in label_counts]
class_weights = torch.tensor(weights, dtype=torch.float)
if torch.cuda.is_available():
    class_weights = class_weights.cuda()


# ================================
# 4. Custom Trainer with Weighted Loss
# ================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ================================
# 5. Metrics: Use Macro-F1
# ================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro"),
        "recall": recall_score(labels, preds, average="macro"),
    }


# ================================
# 6. Training Args
# ================================
training_args = TrainingArguments(
    output_dir="models/parsbert_emotion",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",   # use macro_f1 instead of accuracy
    logging_dir='models/logs',
    logging_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    disable_tqdm=True,
)


# ================================
# 7. Train & Evaluate
# ================================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

results = trainer.evaluate()
print("Final evaluation metrics:", results)


# ================================
# 8. Save Model
# ================================
save_path = "/content/drive/MyDrive/parsbert400k_emotion"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
