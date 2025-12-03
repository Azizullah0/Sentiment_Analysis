# fear_augmenter.py - CLEAN VERSION
import random
from typing import List

class FearDataAugmenter:
    """Simple fear data augmentation"""
    
    def __init__(self):
        # Dari fear expressions
        self.fear_templates = [
            "می‌ترسم که {trigger}",
            "نگران {trigger} هستم",
            "از {trigger} هراس دارم",
            "دلهره {trigger} را دارم",
            "وحشت از {trigger} مرا گرفته",
            "ترس از {trigger} آزارم می‌دهد",
            "اضطراب {trigger} مرا اذیت می‌کند"
        ]
        
        self.fear_triggers = [
            "آینده", "امتحان", "تاریکی", "تنهایی", "مرگ",
            "بیماری", "فقر", "جنگ", "سقوط", "گم شدن",
            "شکست", "طرد شدن", "بلایای طبیعی"
        ]
    
    def generate_fear_samples(self, n_samples: int = 500) -> List[str]:
        """Generate new fear samples"""
        samples = []
        
        for _ in range(n_samples):
            template = random.choice(self.fear_templates)
            trigger = random.choice(self.fear_triggers)
            text = template.format(trigger=trigger)
            samples.append(text)
        
        return samples
    
    def augment_existing_fear(self, existing_texts: List[str], n_samples: int = 300) -> List[str]:
        """Augment existing fear samples with variations"""
        augmented = []
        
        for text in existing_texts[:50]:  # Use first 50 samples
            # Simple variations
            variations = [
                text,
                text + " و نمی‌دانم چه کنم",
                "همیشه " + text,
                "این روزها " + text,
                text.replace("می‌ترسم", "هراس دارم"),
                text.replace("نگران", "مضطرب"),
            ]
            augmented.extend(variations)
        
        # Add some completely new samples
        augmented.extend(self.generate_fear_samples(n_samples // 2))
        
        return list(set(augmented))[:n_samples]

# === NO CODE OUTSIDE THE CLASS ===
# This file should ONLY contain the class definition
