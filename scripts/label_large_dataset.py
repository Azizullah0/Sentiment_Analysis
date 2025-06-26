import pandas as pd
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Load trained model and tokenizer
model_path = "/content/drive/MyDrive/parsbert_emotion"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Load cleaned dataset
df = pd.read_csv('/content/drive/MyDrive/ColabFoulder/Cleaned_Dataset.csv')

# 3. Set device (A100 will be automatically used if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. Initialize pipeline with batching and truncation
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512,
    batch_size=32  # adjust based on RAM if needed
)

# 5. Clean and filter input texts
texts = df["clean"].fillna("").astype(str).tolist()

# 6. Run predictions in batch
predictions = clf(texts)

# 7. Extract predicted labels
df["predicted_label"] = [pred["label"] for pred in predictions]

# 8. Save results
df.to_csv('/content/drive/MyDrive/ColabFoulder/Labeled_400K.csv', index=False)

print("✅ Labeling completed and saved.")
