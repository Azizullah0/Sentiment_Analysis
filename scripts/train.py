import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import load_dataset, split_dataset, tokenize_datasets

# 1. Load and prepare data
df = load_dataset('datasets/Training_Ready_Labeled.csv', label_col='label_id')  # Use your actual CSV and label column
train_df, test_df = split_dataset(df)

# 2. Tokenize
train_dataset, test_dataset, tokenizer = tokenize_datasets(train_df, test_df, model_name="HooshvareLab/bert-base-parsbert-uncased", max_length=128)

# 3. Model
num_labels = len(df['labels'].unique())
model = AutoModelForSequenceClassification.from_pretrained(
    "HooshvareLab/bert-base-parsbert-uncased",
    num_labels=num_labels
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
    }

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="models/parsbert_emotion",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='models/logs',
    logging_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

# 6. Trainer
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

# 7. Train
trainer.train()

# 8. Save model and tokenizer
trainer.save_model("models/parsbert_emotion")
tokenizer.save_pretrained("models/parsbert_emotion")