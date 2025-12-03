# predict.py
import sys
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Optional: Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.paths import PATHS, MODEL_CONFIG

# Label mapping (match your training labels)
LABEL_MAP = {
    0: "Hope",
    1: "Happy",
    2: "Neutral",
    3: "Surprise",   # fixed typo
    4: "Disgust",
    5: "Sad",
    6: "Anger",
    7: "Fear"
}

class EmotionPredictor:
    def __init__(self, model_path=None):
        # Use new trained model path or default
        self.model_path = model_path or PATHS["incremental_finetuned_model"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        """Load tokenizer and model with proper device allocation"""
        print(f"Loading model from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded! Number of labels: {self.model.config.num_labels}")

    def predict(self, texts, return_probabilities=False):
        """Predict emotions for single or multiple texts"""
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MODEL_CONFIG.get("max_length", 128),
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

        pred_labels = [LABEL_MAP[int(p)] for p in preds.cpu().numpy()]
        conf_scores = probs.cpu().numpy()

        if return_probabilities:
            results = []
            for text, label, prob_array in zip(texts, pred_labels, conf_scores):
                result = {
                    "text": text,
                    "emotion": label,
                    "confidence": float(np.max(prob_array)),
                    "all_probabilities": {LABEL_MAP[i]: float(prob) for i, prob in enumerate(prob_array)}
                }
                results.append(result)
            return results if len(results) > 1 else results[0]
        else:
            return pred_labels if len(pred_labels) > 1 else pred_labels[0]

    def predict_batch(self, texts, batch_size=32, return_probabilities=False):
        """Batch prediction"""
        all_results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            res = self.predict(batch, return_probabilities)
            # Ensure it's always a list
            if isinstance(res, dict):
                res = [res]
            all_results.extend(res)
            if i % (batch_size * 10) == 0:
                print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} texts")
        return all_results

# Global instance
_predictor = None

def get_predictor(model_path=None):
    global _predictor
    if _predictor is None:
        _predictor = EmotionPredictor(model_path)
    return _predictor

def predict(texts, return_probabilities=False):
    return get_predictor().predict(texts, return_probabilities)

def predict_batch(texts, batch_size=32, return_probabilities=False):
    return get_predictor().predict_batch(texts, batch_size, return_probabilities)

# Test script
if __name__ == "__main__":
    predictor = EmotionPredictor(model_path=PATHS["incremental_imbalanced_20251203_1541/checkpoint-40068"])
    
    sample_texts = [
        "من امروز خیلی خوشحالم",
        "احساس ناراحتی می‌کنم",
        "از امتحان فردا می‌ترسم",
        "این فیلم واقعا تعجب آور بود",
        "از این غذا متنفرم",
        "امیدوارم فردا روز بهتری باشد",
        "این خبر مرا عصبانی کرد",
        "هوا خوب است",
        "نگران آینده هستم",
        "چه اتفاق غیرمنتظره ای",
    ]

    print("Testing Emotion Prediction\n" + "="*50)
    results = predictor.predict(sample_texts, return_probabilities=True)
    for r in results:
        print(f"{r['text']}")
        print(f"→ {r['emotion']} (Confidence: {r['confidence']:.3f})")
        top3 = sorted(r['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        print("Top 3:", ", ".join([f"{e}({p:.3f})" for e, p in top3]))
        print("-"*40)
