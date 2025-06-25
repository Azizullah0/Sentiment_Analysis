
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
from utils.dataset_utils import load_dataset, split_dataset, tokenize_datasets

# Load and prepare data
df = load_dataset('datasets/Labeled_Dataset.csv')  # convert .xlsx to .csv if not yet done
train_df, test_df = split_dataset(df)
train_dataset, test_dataset, tokenizer = tokenize_datasets(train_df, test_df)

# Load model
model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", num_labels=8)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Training args
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
    logging_strategy="epoch"
)

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
model.save_pretrained("models/parsbert_emotion")
tokenizer.save_pretrained("models/parsbert_emotion")
