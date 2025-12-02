# quick_fix.py - SUPER SIMPLE
def enhance_fear_detection(text, current_emotion, confidence):
    """
    Simple rule-based fear enhancement
    Returns: (new_emotion, new_confidence)
    """
    
    fear_keywords = {
        "می‌ترسم": 0.9,
        "نگران": 0.8, 
        "هراس": 0.85,
        "وحشت": 0.88,
        "دلهره": 0.75,
        "اضطراب": 0.7
    }
    
    # Check for fear keywords
    fear_score = 0
    for keyword, weight in fear_keywords.items():
        if keyword in text:
            fear_score = max(fear_score, weight)
    
    # If text contains strong fear indicators
    if fear_score > 0.7 and current_emotion != "Fear":
        return "Fear", max(confidence, fear_score)
    
    return current_emotion, confidence

# Usage in your predictor:
# In your predict() function, add:
# emotion, confidence = enhance_fear_detection(text, emotion, confidence)
