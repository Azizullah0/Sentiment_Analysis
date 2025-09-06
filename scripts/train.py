import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

from config.paths import PATHS, MODEL_CONFIG
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch

df = pd.read_csv(PATHS["labeled_data"])
df = df.rename(columns={'text': 'clean', 'label': 'label'})

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = "HooshvareLab/bert-base-parsbert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)

def tokenize(batch):
    return tokenizer(
        batch["clean"], 
        padding="max_length", 
        truncation=True, 
        max_length=MODEL_CONFIG["max_length"]
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

columns = ["input_ids", "attention_mask", "labels"]
train_dataset.set_format("torch", columns=columns)
test_dataset.set_format("torch", columns=columns)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=PATHS["finetuned_model"],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=MODEL_CONFIG["learning_rate"],
    per_device_train_batch_size=MODEL_CONFIG["batch_size"],
    per_device_eval_batch_size=MODEL_CONFIG["batch_size"],
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir=os.path.join(PATHS["finetuned_model"], "logs"),
    logging_strategy="epoch",
    report_to="none"
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()