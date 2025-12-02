# quick_fear_fix.py - Implement this right now

def quick_fear_enhancement(predictor):
    """Quick enhancement to existing predictor"""
    
    fear_keywords = {
        "می‌ترسم": 0.9,
        "نگران": 0.8,
        "هراس": 0.85,
        "وحشت": 0.88,
        "دلهره": 0.75,
        "اضطراب": 0.7,
        "دل شور": 0.65,
        "دست میلرزد": 0.6,
    }
    
    # Monkey patch the predict method
    original_predict = predictor.predict
    
    def enhanced_predict(texts, return_probabilities=False):
        results = original_predict(texts, return_probabilities)
        
        if return_probabilities:
            if isinstance(results, dict):
                results = [results]
            
            for result in results:
                text = result['text']
                
                # Check for fear keywords
                fear_score = 0
                for keyword, weight in fear_keywords.items():
                    if keyword in text:
                        fear_score = max(fear_score, weight)
                
                # If strong fear indicator, adjust prediction
                if fear_score > 0.7 and result['emotion'] != 'Fear':
                    fear_prob = result['all_probabilities'].get('Fear', 0)
                    boosted_fear = min(1.0, fear_prob * (1 + fear_score))
                    
                    # If boosted fear is now highest
                    if boosted_fear > result['confidence']:
                        result['emotion'] = 'Fear'
                        result['confidence'] = boosted_fear
                        result['all_probabilities']['Fear'] = boosted_fear
        
        return results
    
    predictor.predict = enhanced_predict
    print("✅ Quick fear enhancement applied to predictor!")
    return predictor

# Apply to your existing predictor
enhanced_predictor = quick_fear_enhancement(predictor)

# Test again
test_texts = ["از امتحان فردا می‌ترسم", "نگران آینده هستم"]
results = enhanced_predictor.predict(test_texts, return_probabilities=True)