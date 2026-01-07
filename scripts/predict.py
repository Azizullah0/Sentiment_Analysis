# predict.py
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Update this to the folder that contains config.json/pytorch_model.bin/tokenizer.* (e.g., your saved run)
MODEL_PATH = r"/content/drive/MyDrive/Sentiment_Analysis/outputs/incremental_imbalanced_20260106_1346/checkpoint-40068"

# Label map must match the number/order of labels in the loaded model
LABEL_MAP = {
    0: "Hope",
    1: "Happy",
    2: "Neutral",
    3: "Surprise",
    4: "Disgust",
    5: "Sad",
    6: "Anger",
    7: "Fear",
  
}


class EmotionPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        print(f"Loading model from: {self.model_path}")
        try:
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
                "emotion": LABEL_MAP.get(int(preds[i]), f"Label_{int(preds[i])}"),
                "confidence": float(probs[i].max().cpu()),
                "all_probabilities": {LABEL_MAP.get(j, f"Label_{j}"): float(probs[i][j]) for j in range(self.model.config.num_labels)}
            }
            results.append(result)

        return results if len(results) > 1 else results[0]


if __name__ == "__main__":
    predictor = EmotionPredictor()
    test_samples = [
        "من امروز خیلی خوشحالم",
        "بعد از مدت‌ها کار پیدا کردم، دلم خوش است",
        "احساس ناراحتی می‌کنم",
        "دلم گرفته، هیچ چیز خوشحالم نمی‌کند",
        "از دیشب تا حالا فقط گریه کردم",
        "از امتحان فردا می‌ترسم",
        "این خبر مرا عصبانی کرد",
        "از این همه ظلم واقعاً خشم دارند",
        "حرف‌هایشان واقعاً خونم را به جوش آورد",
        "از انفجارهای اخیر وحشت دارم",
        "نگران آینده هستم که مبادا همه چیز خراب شود",
        "از این فساد و دروغ خسته شدم",
        "این رفتارشان واقعاً چندش‌آور بود",
        "!باورم نمی‌شود تیم ما برنده شد",
        "چه خبر عجیبی، اصلاً انتظارش را نداشتم",
        "هوا امروز معمولی بود، چیز خاصی نشد",
        "امیدوارم اوضاع بهتر شود و کار پیدا کنم",
        "انشاالله روزهای خوب در راه است",
        "یک کم نگران امتحانم هستم، شاید خوب شود",
        "فعلاً صبر می‌کنیم، شاید اوضاع بهتر شد",
        "نگران آینده هستم"
    ]
    output = predictor.predict(test_samples)
    for r in output:
        print("Text:", r["text"])
        print("Prediction:", r["emotion"], "| Confidence:", r["confidence"])
        print("Top 3:", sorted(r["all_probabilities"].items(), key=lambda x: -x[1])[:3])
        print("-" * 40)
