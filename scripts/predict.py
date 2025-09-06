import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

# Model path (should be the same as your trained model directory)
model_path = "/content/drive/MyDrive/parsbert_emotion"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=8,
    problem_type="single_label_classification"
)

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
        padding="max_length", 
        truncation=True, 
        max_length=512  # Use the same as your training max_length
    )

# Tokenize the dataset
dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# Create DataLoader
loader = DataLoader(dataset, batch_size=32)

# Prediction
preds = []
probabilities = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get predictions
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        preds.extend(batch_preds)
        
        # Get probabilities for analysis
        batch_probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
        probabilities.extend(batch_probs)

# Add predictions to dataframe
df["predicted_label"] = preds
df["predicted_emotion"] = [id2label[pred] for pred in preds]

# Add confidence scores (max probability for each prediction)
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