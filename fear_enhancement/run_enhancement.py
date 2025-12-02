# run_enhancement.py - ONE FILE TO RUN EVERYTHING
"""
Fear Enhancement - Complete Solution
Run this script end-to-end
"""

import pandas as pd
import pickle

print("=" * 60)
print("🚀 SIMPLE FEAR ENHANCEMENT")
print("=" * 60)

# Step 1: Load your data
print("\n📥 STEP 1: Loading your data...")
try:
    # UPDATE THIS PATH to your actual data
    data_path = "/content/drive/MyDrive/Sentiment_Analysis/data/your_dataset.csv"
    df = pd.read_csv(data_path, encoding='utf-8')
    
    fear_samples = df[df['label'] == 'Fear']
    print(f"   Found {len(fear_samples)} fear samples")
    print(f"   Total samples: {len(df)}")
    
    # Show examples
    print("\n   Example fear texts:")
    for i, text in enumerate(fear_samples['text'].head(3).tolist(), 1):
        print(f"   {i}. {text[:50]}...")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   Please update the data_path variable with your actual path")

# Step 2: Generate more fear data
print("\n🔄 STEP 2: Generating more fear examples...")
from fear_augmenter import SimpleFearAugmenter

augmenter = SimpleFearAugmenter()
new_fear_texts = augmenter.generate_fear_samples(300)

print(f"   Generated {len(new_fear_texts)} new fear examples")
print("\n   Sample generated texts:")
for i, text in enumerate(new_fear_texts[:5], 1):
    print(f"   {i}. {text}")

# Step 3: Save augmented data
print("\n💾 STEP 3: Saving augmented data...")
augmented_df = pd.DataFrame({
    'text': new_fear_texts,
    'label': 'Fear',
    'source': 'augmented'
})

save_path = "/content/augmented_fear_samples.csv"
augmented_df.to_csv(save_path, index=False, encoding='utf-8')
print(f"   Saved to: {save_path}")

# Step 4: Create fear keyword detector
print("\n🔧 STEP 4: Creating fear keyword detector...")
fear_detector_rules = {
    "می‌ترسم": 0.9,
    "نگران": 0.8, 
    "هراس": 0.85,
    "وحشت": 0.88,
    "دلهره": 0.75,
    "اضطراب": 0.7,
    "دل شور": 0.65,
    "ترس از": 0.9
}

print(f"   Created detector with {len(fear_detector_rules)} rules")

# Step 5: Test on problematic cases
print("\n🧪 STEP 5: Testing enhancement...")
test_cases = [
    ("از امتحان فردا می‌ترسم", "Fear"),
    ("نگران آینده هستم", "Fear"),
    ("دلهره مرگ مرا فرا گرفته", "Fear"),
    ("من امروز خوشحالم", "Happy"),  # Should NOT be fear
    ("عصبانی هستم", "Anger")        # Should NOT be fear
]

from quick_fix import enhance_fear_detection

print("\n   Test Results:")
print("   " + "-" * 50)

for text, expected in test_cases:
    # Simulate original model prediction (usually wrong for fear)
    if expected == "Fear":
        original_emotion = "Sad"  # Model often predicts Sad for fear
        original_confidence = 0.6
    else:
        original_emotion = expected
        original_confidence = 0.9
    
    # Apply enhancement
    new_emotion, new_conf = enhance_fear_detection(
        text, original_emotion, original_confidence
    )
    
    is_correct = new_emotion == expected
    status = "✅" if is_correct else "❌"
    
    print(f"   {status} '{text[:20]}...'")
    print(f"      Before: {original_emotion} ({original_confidence:.2f})")
    print(f"      After:  {new_emotion} ({new_conf:.2f})")
    print(f"      Expected: {expected}")
    print()

print("=" * 60)
print("🎯 ENHANCEMENT COMPLETE!")
print("\nNext steps:")
print("1. Use the augmented data to retrain your model")
print("2. Add enhance_fear_detection() to your predict.py")
print("3. Test with real predictions")
print("=" * 60)
