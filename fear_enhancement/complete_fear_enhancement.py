# complete_fear_enhancement.py
import pandas as pd
from datetime import datetime

class CompleteFearEnhancement:
    def __init__(self, model_path):
        print("🚀 Initializing Fear Enhancement Pipeline...")
        
        # 1. Load base model
        self.base_model = self._load_base_model(model_path)
        
        # 2. Initialize components
        self.augmenter = FearDataAugmenter()
        self.context_detector = ContextAwareFearDetector()
        self.hybrid_classifier = HybridFearEmotionClassifier(model_path)
        
        print("✅ Fear Enhancement Pipeline ready!")
    
    def run_enhancement_pipeline(self, dataset_path, output_dir):
        """Complete fear enhancement workflow"""
        
        # Step 1: Analyze current fear performance
        print("\n📊 Step 1: Analyzing current fear detection...")
        fear_analysis = self.analyze_fear_performance(dataset_path)
        
        # Step 2: Augment fear data
        print("\n🔄 Step 2: Augmenting fear samples...")
        augmented_data = self.augment_fear_data(dataset_path)
        
        # Step 3: Fine-tune with enhanced loss
        print("\n🎯 Step 3: Fine-tuning with fear-focused loss...")
        enhanced_model = self.fine_tune_with_fear_focus(augmented_data)
        
        # Step 4: Implement hybrid prediction
        print("\n🤝 Step 4: Implementing hybrid prediction...")
        self.test_enhanced_pipeline()
        
        # Step 5: Generate report
        print("\n📈 Step 5: Generating enhancement report...")
        report = self.generate_enhancement_report(fear_analysis)
        
        return enhanced_model, report
    
    def test_enhanced_pipeline(self):
        """Test the enhanced pipeline"""
        test_cases = [
            ("از امتحان فردا می‌ترسم", "Fear"),
            ("نگران آینده هستم", "Fear"),
            ("دلهره مرگ مرا فرا گرفته", "Fear"),
            ("وحشت از تاریکی", "Fear"),
            ("اضطراب دارم", "Fear"),
            ("من امروز خوشحالم", "Happy"),  # Control case
            ("عصبانی هستم", "Anger"),       # Control case
        ]
        
        print("\n🧪 Testing Enhanced Fear Detection:")
        print("=" * 60)
        
        correct = 0
        total = len(test_cases)
        
        for text, expected in test_cases:
            result = self.hybrid_classifier.predict_with_fear_enhancement(text)
            prediction = result['emotion'] if isinstance(result, dict) else result[0]['emotion']
            
            is_correct = prediction == expected
            if is_correct:
                correct += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} '{text}'")
            print(f"   Expected: {expected}, Got: {prediction}")
            if isinstance(result, dict):
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Fear context score: {result['fear_context_score']:.3f}")
            print()
        
        accuracy = correct / total * 100
        print(f"📊 Enhanced Accuracy: {accuracy:.1f}% ({correct}/{total})")
        
        return accuracy