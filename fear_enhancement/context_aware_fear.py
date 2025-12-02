# context_aware_fear.py
import re
from typing import List, Tuple

class ContextAwareFearDetector:
    def __init__(self):
        # Fear patterns in Dari/Persian
        self.fear_patterns = [
            # Explicit fear
            (r"می‌ترسم (از|که)", 0.9),
            (r"نگران (هستم|ام|م)", 0.8),
            (r"هراس (دارم|از)", 0.85),
            (r"وحشت (دارم|از)", 0.88),
            (r"دلهره (دارم|از)", 0.75),
            (r"اضطراب (دارم|داشتن)", 0.7),
            
            # Physical symptoms
            (r"دل (شور|غصه) می‌زند", 0.65),
            (r"دستم (می‌لرزد|میلرزد)", 0.6),
            (r"قلبم (تند|سریع) می‌زند", 0.55),
            
            # Future-oriented fear
            (r"اگر (.*) شود چه", 0.7),
            (r"نگران (آینده|فردا|گذشته)", 0.8),
            (r"چه خواهد شد اگر", 0.75),
            
            # Threat expressions
            (r"خطر (.*) وجود دارد", 0.85),
            (r"ممکن است (.*) بشود", 0.6),
            (r"بترس از", 0.9),
            
            # Intensifiers
            (r"خیلی (می‌ترسم|نگرانم)", 1.0),
            (r"واقعا (هراس دارم|وحشت دارم)", 0.95),
            (r"کاملا (مضطرب|پریشان)", 0.85)
        ]
        
        # Fear contexts
        self.fear_contexts = [
            "آینده", "امتحان", "تاریکی", "مرگ", "بیماری",
            "جنگ", "تنهایی", "فقر", "طرد شدن", "شکست"
        ]
    
    def detect_fear_context(self, text: str, context_window: int = 3) -> float:
        """Detect fear based on context in conversation"""
        score = 0.0
        
        # Check for explicit fear patterns
        for pattern, pattern_score in self.fear_patterns:
            if re.search(pattern, text):
                score = max(score, pattern_score)
        
        # Contextual clues
        words = text.split()
        for i, word in enumerate(words):
            if word in self.fear_contexts:
                # Check surrounding words
                start = max(0, i - context_window)
                end = min(len(words), i + context_window + 1)
                context = " ".join(words[start:end])
                
                # Look for fear indicators in context
                for pattern, pattern_score in self.fear_patterns:
                    if re.search(pattern, context):
                        score = max(score, pattern_score * 0.8)
        
        return score
    
    def enhance_fear_prediction(self, text: str, 
                                model_prediction: str, 
                                model_confidence: float) -> Tuple[str, float]:
        """Enhance model prediction with context-aware fear detection"""
        
        fear_score = self.detect_fear_context(text)
        
        # If context strongly indicates fear
        if fear_score > 0.8:
            if model_prediction != "Fear":
                # Boost confidence for fear
                enhanced_confidence = max(model_confidence, fear_score * 0.9)
                return "Fear", enhanced_confidence
            else:
                # Already fear, boost confidence
                enhanced_confidence = min(1.0, model_confidence * 1.2)
                return "Fear", enhanced_confidence
        
        return model_prediction, model_confidence