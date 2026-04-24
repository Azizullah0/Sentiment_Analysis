# predict.py
import sys
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.paths import PATHS

# Update this to your model path on DGX Spark.
# If your model is stored in the project or on gdrive, PATHS will resolve it.
MODEL_PATH = PATHS["fine_tuned_model"]
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
        print(f"Using device: {self.device}")
        self.load_model()

    def load_model(self):
        print(f"Loading model from: {self.model_path}")
        
        # Check if model path exists
        if not os.path.exists(self.model_path):
            print(f"❌ ERROR: Model path does not exist: {self.model_path}")
            print("Please check the path or train the model first.")
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        try:
            # Load tokenizer (try local first, then download if needed)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=False  # Allow download if files missing
            )
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=False
            ).to(self.device)
            
            self.model.eval()
            print("✅ Model loaded successfully!")
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
    print("=" * 50)
    print("Emotion Predictor - DGX Spark")
    print("=" * 50)
    
    predictor = EmotionPredictor()
    
    test_samples = [
        "من امروز خیلی خوشحالم",  # Happy
        "احساس ناراحتی می‌کنم",    # Sad
        "از امتحان فردا می‌ترسم",  # Fear
        "این خبر مرا عصبانی کرد",  # Anger
        "امیدوارم اوضاع بهتر شود",  # Hope
        "هوا امروز معمولی بود",    # Neutral
        "واقعا شگفت‌زده شدم",      # Surprise
        "این رفتار چندش‌آور بود",  # Disgust
    ]
    
    print(f"\n🔮 Running predictions on {len(test_samples)} samples...\n")
    
    output = predictor.predict(test_samples)
    
    for i, r in enumerate(output, 1):
        print(f"{i}. Text: {r['text']}")
        print(f"   🎭 Prediction: {r['emotion']} | Confidence: {r['confidence']:.4f}")
        print(f"   📊 Top 3: {sorted(r['all_probabilities'].items(), key=lambda x: -x[1])[:3]}")
        print("-" * 50)