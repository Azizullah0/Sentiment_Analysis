import pandas as pd
import torch
import warnings
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from config.paths import PATHS  # <--- use your paths config

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# 1. Automatically pick latest fine-tuned model from PATHS
# -----------------------------
MODELS_DIR = PATHS["finetuned_model"].rsplit("/", 1)[0]  # parent folder
model_folders = [
    os.path.join(MODELS_DIR, d)
    for d in os.listdir(MODELS_DIR)
    if os.path.isdir(os.path.join(MODELS_DIR, d))
]
latest_model = max(model_folders, key=os.path.getmtime)
print(f"✅ Using latest model: {latest_model}")

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(latest_model, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(latest_model, local_files_only=True)

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
