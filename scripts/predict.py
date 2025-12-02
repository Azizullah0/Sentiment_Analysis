# predict.py
import sys
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.paths import PATHS, MODEL_CONFIG

# Label mapping (consistent with your training)
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

class EmotionPredictor:
    def __init__(self, model_path=None):
        """
        Initialize the emotion predictor with fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned model. If None, uses PATHS["incremental_finetuned_model"]
        """
        self.model_path = model_path or PATHS["incremental_finetuned_model"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer with error handling"""
        try:
            print(f" Loading model from: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path, 
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            print(" Model loaded successfully!")
            
            # Verify model configuration
            print(f"Model configured for {self.model.config.num_labels} classes")
            
        except Exception as e:
            print(f" Error loading model: {e}")
            raise
    
    def predict(self, texts, return_probabilities=False):
        """
        Predict emotion labels for given Persian/Dari texts.
        
        Args:
            texts (str or list): Input text or list of texts
            return_probabilities (bool): Whether to return confidence scores
            
        Returns:
            dict or list: Predictions with emotions and optionally probabilities
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize with same parameters as training
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MODEL_CONFIG["max_length"],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        # Convert to numpy
        pred_labels = [LABEL_MAP[int(p)] for p in predictions.cpu().numpy()]
        conf_scores = probabilities.cpu().numpy()
        
        if return_probabilities:
            results = []
            for text, label, probs in zip(texts, pred_labels, conf_scores):
                result = {
                    "text": text,
                    "emotion": label,
                    "confidence": float(np.max(probs)),
                    "all_probabilities": {
                        LABEL_MAP[i]: float(prob) for i, prob in enumerate(probs)
                    }
                }
                results.append(result)
            return results if len(results) > 1 else results[0]
        else:
            return pred_labels if len(pred_labels) > 1 else pred_labels[0]
    
    def predict_batch(self, texts, batch_size=32, return_probabilities=False):
        """
        Predict emotions for large batches of texts.
        
        Args:
            texts (list): List of input texts
            batch_size (int): Batch size for processing
            return_probabilities (bool): Whether to return confidence scores
            
        Returns:
            list: List of predictions
        """
        all_predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_predictions = self.predict(batch_texts, return_probabilities)
            all_predictions.extend(batch_predictions)
            
            if i % (batch_size * 10) == 0:  # Progress indicator
                print(f" Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        return all_predictions

# Global predictor instance for easy import
_predictor = None

def get_predictor():
    """Get or create global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = EmotionPredictor()
    return _predictor

def predict(texts, return_probabilities=False):
    """
    Convenience function for quick predictions.
    
    Args:
        texts (str or list): Input text(s)
        return_probabilities (bool): Whether to return confidence scores
        
    Returns:
        Predictions in requested format
    """
    predictor = get_predictor()
    return predictor.predict(texts, return_probabilities)

def predict_batch(texts, batch_size=32, return_probabilities=False):
    """
    Convenience function for batch predictions.
    
    Args:
        texts (list): List of input texts
        batch_size (int): Batch size for processing
        return_probabilities (bool): Whether to return confidence scores
        
    Returns:
        list: List of predictions
    """
    predictor = get_predictor()
    return predictor.predict_batch(texts, batch_size, return_probabilities)

if __name__ == "__main__":
    # Initialize predictor
    predictor = EmotionPredictor()
    
    # Test samples covering different emotions
    sample_texts = [
        "من امروز خیلی خوشحالم",              # I am very happy today
        "احساس ناراحتی می‌کنم",              # I feel sad
        "از امتحان فردا می‌ترسم",             # I am afraid of tomorrow's exam
        "این فیلم واقعا تعجب آور بود",        # This movie was really surprising
        "از این غذا متنفرم",                  # I hate this food (Disgust)
        "امیدوارم فردا روز بهتری باشد",       # I hope tomorrow is a better day
        "این خبر مرا عصبانی کرد",             # This news made me angry
        "هوا خوب است",                        # The weather is good (Neutral)
        "نگران آینده هستم",                   # I'm worried about the future (Fear)
        "چه اتفاق غیرمنتظره ای" ،            # What an unexpected event (Surprise)
        "بد  نکن"،
        "رهایم کن"
    ]
    
    print(" Testing Emotion Prediction")
    print("=" * 50)
    
    # Test with confidence scores
    predictions = predictor.predict(sample_texts, return_probabilities=True)
    
    for result in predictions:
        print(f": {result['text']}")
        print(f": {result['emotion']} (Confidence: {result['confidence']:.3f})")
        
        # Show top 3 emotions by probability
        top_emotions = sorted(
            result['all_probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        print(" Top 3:", ", ".join([f"{e}({p:.3f})" for e, p in top_emotions]))
        print("-" * 40)
    
    # Quick test without probabilities
    print("\n Quick predictions (without probabilities):")
    quick_preds = predictor.predict(sample_texts[:3])
    for text, emotion in zip(sample_texts[:3], quick_preds):
        print(f"  '{text}' → {emotion}")
