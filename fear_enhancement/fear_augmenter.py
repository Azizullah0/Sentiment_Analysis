# fear_augmentation.py
import pandas as pd
import random
from typing import List

class FearDataAugmenter:
    def __init__(self):
        # Dari fear expressions and patterns
        self.fear_templates = [
            "می‌ترسم که {trigger}",
            "نگران {trigger} هستم",
            "از {trigger} هراس دارم",
            "دلهره {trigger} را دارم",
            "وحشت از {trigger} مرا گرفته",
            "ترس از {trigger} آزارم می‌دهد",
            "اضطراب {trigger} مرا اذیت می‌کند",
            "دلم برای {trigger} شور می‌زند",
            "از {trigger} می‌لرزم",
            "هراسان از {trigger}",
            "نگرانی {trigger} خوابم را برده",
            "ترس {trigger} وجودم را فرا گرفته",
            "دچار ترس {trigger} شده‌ام",
            "از {trigger} به خود می‌لرزم",
            "وحشت {trigger} مرا فراگرفته"
        ]
        
        self.fear_triggers = [
            "آینده", "امتحان", "تاریکی", "تنهایی", "مرگ",
            "بیماری", "فقر", "جنگ", "سقوط", "گم شدن",
            "شکست", "طرد شدن", "بلایای طبیعی", "تصادف",
            "از دست دادن", "ناشناخته", "ارتفاع", "حیوانات",
            "توفان", "زلزله", "سیل", "آتش", "دزد",
            "پلیس", "دولت", "قضاوت دیگران", "شکست عاطفی"
        ]
        
        self.fear_adjectives = [
            "وحشت‌ناک", "هراس‌انگیز", "ترسناک", "دلهره‌آور",
            "مخوف", "هولناک", "مهیب", "مرگبار", "مخاطره‌آمیز"
        ]
    
    def augment_fear_samples(self, existing_fear_texts: List[str], target_count: int = 1000):
        """Generate augmented fear samples"""
        augmented = []
        
        # 1. Template-based generation
        for _ in range(target_count // 2):
            template = random.choice(self.fear_templates)
            trigger = random.choice(self.fear_triggers)
            text = template.format(trigger=trigger)
            
            # Add variation
            if random.random() > 0.7:
                adj = random.choice(self.fear_adjectives)
                text = text.replace(trigger, f"{adj} {trigger}")
            
            augmented.append(text)
        
        # 2. Paraphrase existing samples
        for text in existing_fear_texts[:min(100, len(existing_fear_texts))]:
            for _ in range(5):  # 5 variations per sample
                paraphrased = self.paraphrase_fear(text)
                augmented.append(paraphrased)
        
        # 3. Mix with context (makes it more realistic)
        context_templates = [
            "دیشب خواب دیدم که {fear}",
            "همیشه فکر می‌کنم که {fear}",
            "نمی‌توانم جلوی فکر کردن به {fear} را بگیرم",
            "از بچگی {fear}",
            "این روزها مدام {fear}",
            "دلم می‌خواهد با کسی در مورد {fear} صحبت کنم"
        ]
        
        for _ in range(target_count // 4):
            context = random.choice(context_templates)
            fear_text = random.choice(augmented[:50])
            augmented.append(context.format(fear=fear_text))
        
        return list(set(augmented))[:target_count]
    
    def paraphrase_fear(self, text: str) -> str:
        """Create paraphrases of fear expressions"""
        replacements = {
            "می‌ترسم": ["هراس دارم", "وحشت دارم", "دچار ترس شده‌ام"],
            "نگران": ["مضطرب", "دلهره‌دار", "پریشان"],
            "هراس": ["وحشت", "ترس", "دلهره"],
            "اضطراب": ["نگرانی", "تشویش", "بی‌قراری"]
        }
        
        for orig, alts in replacements.items():
            if orig in text and random.random() > 0.5:
                text = text.replace(orig, random.choice(alts))
        
        return text

# Usage
augmenter = FearDataAugmenter()
# Get existing fear samples from your dataset
fear_samples = df[df['label'] == 'Fear']['text'].tolist()
augmented_fear = augmenter.augment_fear_samples(fear_samples, target_count=2000)