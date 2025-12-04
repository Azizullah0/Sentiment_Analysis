# predict.py
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- FIXED MODEL PATH ---
MODEL_PATH = r"/content/drive/MyDrive/Sentiment_Analysis/outputs/incremental_imbalanced_20251029_1321/checkpoint-39168"

LABEL_MAP = {
    0: "Hope",
    1: "Happy",
    2: "Neutral",
    3: "Surprise",
    4: "Disgust",
    5: "Sad",
    6: "Anger",
    7: "Fear"
}

class EmotionPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        print(f"Loading model from: {self.model_path}")

        try:
            # IMPORTANT: Use local_files_only=False (safe, avoids HF Hub confusion)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True
            ).to(self.device)

            self.model.eval()

            print("Model loaded successfully!")
            print(f"Number of labels: {self.model.config.num_labels}")

        except Exception as e:
            print("❌ ERROR while loading model:")
            print(str(e))
            raise

    def predict(self, text, return_probabilities=True):
        if isinstance(text, str):
            text = [text]

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)

        preds = probs.argmax(dim=1).cpu().numpy()

        results = []
        for i, t in enumerate(text):
            result = {
                "text": t,
                "emotion": LABEL_MAP[int(preds[i])],
                "confidence": float(probs[i].max().cpu()),
                "all_probabilities": {LABEL_MAP[j]: float(probs[i][j]) for j in range(8)}
            }
            results.append(result)

        return results if len(results) > 1 else results[0]


if __name__ == "__main__":
    predictor = EmotionPredictor()

    test_samples = [
        "من امروز خیلی خوشحالم",
        "احساس ناراحتی می‌کنم",
        "از امتحان فردا می‌ترسم",
        "نگران آینده هستم",
        "این خبر مرا عصبانی کرد"
    ]

    output = predictor.predict(test_samples)
    for r in output:
        print("Text:", r["text"])
        print("Prediction:", r["emotion"], "| Confidence:", r["confidence"])
        print("Top 3:", sorted(r["all_probabilities"].items(), key=lambda x: -x[1])[:3])
        print("-" * 40)
