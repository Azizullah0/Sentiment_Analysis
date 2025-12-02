# label_large_dataset.py

import sys
import os
import pandas as pd
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# Ensure parent folder is on sys.path BEFORE import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.paths import PATHS  # now works
# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# 1. Load fine-tuned model from PATHS
model_path = PATHS["finetuned_model"]
print(f" Loading model from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# 2. Load cleaned dataset
df = pd.read_csv(PATHS["raw_data"])
texts = df["clean"].fillna("").astype(str).tolist()
print(f" Loaded dataset with {len(texts)} rows")
# 3. Initialize pipeline
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512,
    batch_size=32  # adjust based on your RAM
)
# 4. Run predictions
print(" Running predictions on dataset...")
predictions = clf(texts)
# 5. Extract predicted labels
df["predicted_label"] = [pred["label"] for pred in predictions]
# 6. Save results
df.to_csv(PATHS["output_labeled"], index=False)
print(f" Labeling completed and saved to {PATHS['output_labeled']}")
