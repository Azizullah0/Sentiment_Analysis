# predict.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config.paths import PATHS, MODEL_CONFIG   # ✅ now works

# Label mapping (same as training)
LABEL_MAP = {
    0: "Hope",
    1: "Happy",
    2: "Neutral",
    3: "Suprise",
    4: "Disgust",
    5: "Sad",
    6: "Anger",
    7: "Fear"
}

# Load final fine-tuned model
model_path = PATHS["finetuned_model"]
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(texts):
    """
    Predict emotion labels for given texts.
    Args:
        texts (list[str]): List of input sentences.
    Returns:
        List of predicted labels (str).
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MODEL_CONFIG["max_length"],
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    return [LABEL_MAP[int(p)] for p in predictions.cpu().numpy()]


if __name__ == "__main__":
    sample_texts = [
        "من امروز خیلی خوشحالم",      # I am very happy today
        "احساس ناراحتی می‌کنم",        # I feel sad
        "از امتحان فردا می‌ترسم"       # I am afraid of tomorrow’s exam
    ]
    preds = predict(sample_texts)
    for text, label in zip(sample_texts, preds):
        print(f"Text: {text}\n Predicted Emotion: {label}\n")
