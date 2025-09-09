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

print(f"Loading data from: {PATHS['labeled_data']}")
df = pd.read_csv(PATHS["labeled_data"])
print(f"Columns in CSV: {df.columns.tolist()}")

# Label mapping
label_mapping = {
    "Hope": 0,
    "Happy": 1, 
    "Neutral": 2,
    "Suprise": 3,
    "Disgust": 4,
    "Sad": 5,
    "Anger": 6,
    "Fear": 7
}

df['labels'] = df['Label '].map(label_mapping)

# Remove NaN labels
if df['labels'].isna().sum() > 0:
    df = df.dropna(subset=['labels'])

df['labels'] = df['labels'].astype(int)

# Use 'text' column for training
df = df.rename(columns={'text': 'clean'})

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)

train_dataset = Dataset.from_pandas(train_df[['clean', 'labels']])
test_dataset = Dataset.from_pandas(test_df[['clean', 'labels']])

# Load tokenizer and model
model_name = "HooshvareLab/bert-base-parsbert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=8,
    problem_type="single_label_classification"
)

# Tokenization
def tokenize(batch):
    return tokenizer(
        batch["clean"], 
        padding="max_length", 
        truncation=True, 
        max_length=MODEL_CONFIG["max_length"]
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Convert labels to torch.long
def convert_labels_to_long(example):
    example['labels'] = torch.tensor(example['labels'], dtype=torch.long)
    return example

train_dataset = train_dataset.map(convert_labels_to_long)
test_dataset = test_dataset.map(convert_labels_to_long)

columns = ["input_ids", "attention_mask", "labels"]
train_dataset.set_format("torch", columns=columns)
test_dataset.set_format("torch", columns=columns)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments (save only final model in stable folder)
training_args = TrainingArguments(
    output_dir=PATHS["base_model"],  # stable folder
    evaluation_strategy="epoch",
    save_strategy="no",  # no intermediate checkpoints
    learning_rate=MODEL_CONFIG["learning_rate"],
    per_device_train_batch_size=MODEL_CONFIG["batch_size"],
    per_device_eval_batch_size=MODEL_CONFIG["batch_size"],
    num_train_epochs=5,
    weight_decay=0.02,
    load_best_model_at_end=False,
    metric_for_best_model="f1",
    logging_dir=os.path.join(PATHS["base_model"], "logs"),
    logging_strategy="epoch",
    report_to="none"
)

# Metrics
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

# Train
print("Starting training...")
trainer.train()
print("Training complete!")

# Save only final model and tokenizer in stable folder
trainer.save_model(PATHS["base_model"])
tokenizer.save_pretrained(PATHS["base_model"])
print(f"Model and tokenizer fully saved to: {PATHS['base_model']}")
