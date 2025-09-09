import pandas as pd
import torch
import warnings
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config.paths import PATHS  # <--- use your paths config
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# 1. Automatically pick latest fine-tuned model from PATHS
# -----------------------------
# Load final fine-tuned model

model_path = PATHS["finetuned_model"]
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# -----------------------------
# 2. Load cleaned dataset
# -----------------------------
df = pd.read_csv(PATHS["raw_data"])
texts = df["clean"].fillna("").astype(str).tolist()

# -----------------------------
# 3. Set device (GPU if available)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -----------------------------
# 4. Initialize pipeline
# -----------------------------
clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512,
    batch_size=32  # adjust based on RAM
)

# -----------------------------
# 5. Run predictions
# -----------------------------
predictions = clf(texts)

# -----------------------------
# 6. Extract predicted labels
# -----------------------------
df["predicted_label"] = [pred["label"] for pred in predictions]

# -----------------------------
# 7. Save results
# -----------------------------
df.to_csv(PATHS["output_labeled"], index=False)
print(f"✅ Labeling completed and saved to {PATHS['output_labeled']}")
