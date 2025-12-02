# hybrid_fear_pipeline.py
from transformers import pipeline
import torch

class HybridFearEmotionClassifier:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Main emotion classifier
        self.emotion_classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=0 if self.device == "cuda" else -1,
            top_k=None
        )
        
        # Specialized fear detector
        self.context_detector = ContextAwareFearDetector()
        
        # Fear synonyms for final adjustment
        self.fear_synonyms = {
            "ترس", "هراس", "وحشت", "دلهره", "اضطراب",
            "نگرانی", "تشویش", "پریشانی", "هراسانی"
        }
    
    def predict_with_fear_enhancement(self, texts, threshold=0.7):
        """Predict emotions with enhanced fear detection"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        for text in texts:
            # Get base prediction
            base_preds = self.emotion_classifier(text)[0]
            
            # Find top prediction
            top_pred = max(base_preds, key=lambda x: x['score'])
            emotion = top_pred['label']
            confidence = top_pred['score']
            
            # Check if text contains fear indicators
            fear_context_score = self.context_detector.detect_fear_context(text)
            
            # Contains fear synonyms?
            has_fear_words = any(syn in text for syn in self.fear_synonyms)
            
            # Enhancement logic
            if fear_context_score > threshold or has_fear_words:
                if emotion != "Fear":
                    # Fear should be at least second choice
                    fear_pred = next((p for p in base_preds if p['label'] == "Fear"), None)
                    if fear_pred:
                        # If fear score is close to top prediction
                        if confidence - fear_pred['score'] < 0.2:
                            emotion = "Fear"
                            confidence = max(confidence, fear_pred['score'] * 1.3)
                        else:
                            # Still boost fear probability
                            fear_boosted_score = min(1.0, fear_pred['score'] * 1.5)
                            if fear_boosted_score > confidence:
                                emotion = "Fear"
                                confidence = fear_boosted_score
            
            results.append({
                'text': text,
                'emotion': emotion,
                'confidence': float(confidence),
                'fear_context_score': float(fear_context_score),
                'has_fear_words': has_fear_words,
                'all_predictions': base_preds
            })
        
        return results if len(results) > 1 else results[0]

# Usage
enhanced_classifier = HybridFearEmotionClassifier(
    model_path="/content/drive/MyDrive/Sentiment_Analysis/outputs/incremental_imbalanced_20251029_1321/checkpoint-39168"
)

# Test with problematic cases
problematic_texts = [
    "از امتحان فردا می‌ترسم",  # Should be Fear
    "نگران آینده هستم",        # Should be Fear
    "این خبر مرا عصبانی کرد",  # Should be Anger
]

results = enhanced_classifier.predict_with_fear_enhancement(problematic_texts)
for r in results:
    print(f"Text: {r['text']}")
    print(f"Prediction: {r['emotion']} (Conf: {r['confidence']:.3f})")
    print(f"Fear context score: {r['fear_context_score']:.3f}")
    print(f"Has fear words: {r['has_fear_words']}")
    print("-" * 50)