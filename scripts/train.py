import sys
import os
import logging
import warnings


logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.dataset_utils import load_dataset, split_dataset, tokenize_datasets


df = load_dataset('/content/drive/MyDrive/ColabFoulder/Labeled_400K_with_emotions.csv', label_col='label_id')
train_df, test_df = split_dataset(df)


train_dataset, test_dataset, tokenizer = tokenize_datasets(
    train_df, test_df,
    model_name="HooshvareLab/bert-base-parsbert-uncased",
    max_length=128
)


num_labels = len(df['labels'].unique())
model = AutoModelForSequenceClassification.from_pretrained(
    "HooshvareLab/bert-base-parsbert-uncased",
    num_labels=num_labels
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }


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
    metric_for_best_model="accuracy",
    logging_dir='models/logs',
    logging_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    disable_tqdm=True,         
)


trainer = Trainer(
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


save_path = "/content/drive/MyDrive/parsbert400_emotion"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)