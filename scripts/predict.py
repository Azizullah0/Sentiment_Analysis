import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np

# Define the same label mapping used in training
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

# Reverse mapping for prediction
id2label = {v: k for k, v in label_mapping.items()}

# Use the BEST checkpoint (usually the last one)
checkpoint_path = "/content/drive/MyDrive/Sentiment_Analysis/Models/fine_tuned_model_20250906_1241/checkpoint-591"

print(f"Loading model from checkpoint: {checkpoint_path}")

# Load tokenizer and model from checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Using device: {device}")
print(f"Model label mapping: {model.config.id2label}")

# Load data
df = pd.read_csv("datasets/Cleaned_Dataset.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Ensure we have the correct column name (same as training)
if 'text' in df.columns and 'clean' not in df.columns:
    df = df.rename(columns={'text': 'clean'})
    print("Renamed 'text' column to 'clean'")

# Check for missing values
print(f"Missing values in 'clean' column: {df['clean'].isna().sum()}")
if df['clean'].isna().sum() > 0:
    print("Dropping rows with missing text...")
    df = df.dropna(subset=['clean'])

# Create dataset
dataset = Dataset.from_pandas(df)

# Tokenization function (same as training)
def tokenize(batch):
    return tokenizer(
        batch["clean"], 
        padding=True,  # Use dynamic padding for efficiency
        truncation=True, 
        max_length=512
    )

# Tokenize the dataset
dataset = dataset.map(tokenize, batched=True)

# Create Trainer for easy prediction
trainer = Trainer(model=model)

# Predict
print("Making predictions...")
predictions = trainer.predict(dataset)
preds = np.argmax(predictions.predictions, axis=1)

# Add predictions to dataframe
df["predicted_label"] = preds
df["predicted_emotion"] = [id2label[pred] for pred in preds]

# Get confidence scores
probabilities = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
df["confidence"] = [np.max(prob) for prob in probabilities]

# Save results
output_path = "datasets/Cleaned_Dataset_Labeled.csv"
df.to_csv(output_path, index=False)

print(f"Saved predictions to {output_path}")
print("\nPrediction distribution:")
print(df["predicted_emotion"].value_counts())
print(f"\nAverage confidence: {df['confidence'].mean():.3f}")

# Show sample predictions
print("\nSample predictions:")
for i in range(min(5, len(df))):
    print(f"Text: {df['clean'].iloc[i][:50]}...")
    print(f"Predicted: {df['predicted_emotion'].iloc[i]} (confidence: {df['confidence'].iloc[i]:.3f})")
    print("-" * 50)