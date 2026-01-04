import random
import pandas as pd
from datetime import datetime

# ----------------------------------------------------
# AFGHAN FEAR AUGMENTOR (Professional Edition)
# ----------------------------------------------------
# Generates up to 9,000 unique fear-like samples
# Based strictly on Afghan social-media language patterns
# ----------------------------------------------------

class AfghanFearAugmentorPro:

    def __init__(self):

        # -------------------------------
        # 1. War / Political Subjects
        # -------------------------------
        self.war_groups = [
            "طالبان", "داعش", "پاکستان", "امریکا", "ایران", "حکومت",
            "مجاهدین", "کوچی‌ها", "نظام قبلی", "تروریستان", "جنگ‌سالاران",
            "شبکه حقانی", "طالب", "استخبارات پاکستان", "عناصر خارجی"
        ]

        # -------------------------------
        # 2. Hate–Fear Subjects (insults + group)
        # -------------------------------
        self.insults = [
            "وحشی‌ها", "کثیف‌ها", "قاتلان", "دزدان", "بی‌غیرت‌ها", 
            "نوکران پاکستان", "جاسوس‌ها", "احمق‌ها", "خائنان", "بی‌سوادها"
        ]

        # -------------------------------
        # 3. Social & Future Subjects
        # -------------------------------
        self.future_subjects = [
            "آیندهٔ کشور", "آیندهٔ کودکان", "مردم بیچاره ما",
            "نسل بعد", "افغانستان", "شهرهای ما", "جوان‌ها",
            "مهاجرین", "وضعیت اقتصادی", "ناامنی‌ها"
        ]

        # -------------------------------
        # 4. EVENTS — matched with subject type
        # -------------------------------

        # War events
        self.war_events = [
            "حمله کنند", "انفجار شود", "جنگ دوباره شروع شود",
            "مردم را بکشند", "ترور کنند", "درگیری شدید شود",
            "امنیت را نابود کنند", "شهر سقوط کند"
        ]

        # Hate-fear actions
        self.hate_events = [
            "باز مردم را آزار بدهند", "خیانت کنند", "باعث خاک‌ساری مردم شوند",
            "باز ظلم کنند", "آرامی را از بین ببرند", "کشور را نابود کنند"
        ]

        # Future/social fears
        self.future_events = [
            "بدتر شود", "تباه شود", "همه چیز خراب‌تر شود",
            "مردم ناامید شوند", "هیچ آینده‌ای باقی نماند",
            "زندگی سخت‌تر شود", "مهاجرت بیشتر شود"
        ]

        # -------------------------------
        # Afghan Style Expressions
        # -------------------------------
        self.emotional_prefixes = [
            "به خدا قسم", "والله که", "این روزها واقعاً", "گاهی شب‌ها",
            "قسم به قرآن", "به‌خداوند مهربان", "باور کنید",
            "دلم میلرزه وقتی میبینم", "ای خدا رحم کن",
        ]

        self.emotional_endings = [
            "خدا رحم کنه", "انشاالله خیر باشد", 
            "خدا نکنه همچی شود", "وضع خیلی خراب است",
            "همه در ترس زندگی میکنیم", "خدا به مردم رحم کند",
        ]

        # -------------------------------
        # Text patterns: 4 MODULES
        # -------------------------------

        # Module 1: War/Political Fear
        self.patterns_war = [
            "می‌ترسم {group} دوباره {event}",
            "{prefix} {group} هر لحظه ممکن است {event}، {ending}",
            "از دست {group} وحشت دارم که باز {event}",
            "هر روز با ترس زندگی میکنیم که {group} {event}",
            "هراس دارم اگر {group} {event}، مردم بدبخت می‌شوند",
            "{group} اگر همینطور ادامه دهد، ممکن است {event}"
        ]

        # Module 2: Hate + Fear mixture (common in your dataset)
        self.patterns_hate = [
            "{prefix} این {insult} ممکن است دوباره {event}",
            "لعنت به این {insult} که هر لحظه ممکن است {event}",
            "{insult} اگر باز {event}، مردم نابود می‌شوند",
            "از دست این {insult} هیچ آرامی نمانده، میترسم {event}"
        ]

        # Module 3: Social / Future anxiety
        self.patterns_future = [
            "نگران {subject} هستم که {event}",
            "{prefix} {subject} ممکن است {event}",
            "از آیندهٔ {subject} میترسم، شاید {event}",
            "همه میگویند {subject} قرار است {event}، {ending}",
            "گاهی فکر میکنم اگر {subject} {event} چه خواهد شد"
        ]

        # Module 4: Mild / generic fear (realistic Afghan tone)
        self.patterns_soft = [
            "این روزها از همه چیز میترسم، مخصوصاً اینکه {event}",
            "گاهی نمیفهمم چی میشه، فقط حس میکنم {event}",
            "{prefix} احساس بدی دارم که شاید {event}",
            "دل‌آشوب میشم وقتی فکر میکنم {event}"
        ]

    # -------------------------------------------------------------
    # Core generator for a single sentence
    # -------------------------------------------------------------
    def generate_sentence(self):
        r = random.random()

        # Decide which module to use
        if r < 0.55:   # 55% War fear
            pattern = random.choice(self.patterns_war)
            return pattern.format(
                group=random.choice(self.war_groups),
                event=random.choice(self.war_events),
                prefix=random.choice(self.emotional_prefixes),
                ending=random.choice(self.emotional_endings)
            )

        elif r < 0.80:  # 25% Hate fear
            pattern = random.choice(self.patterns_hate)
            return pattern.format(
                insult=random.choice(self.insults),
                event=random.choice(self.hate_events),
                prefix=random.choice(self.emotional_prefixes),
                ending=random.choice(self.emotional_endings)
            )

        elif r < 0.93:  # 13% Social/future
            pattern = random.choice(self.patterns_future)
            return pattern.format(
                subject=random.choice(self.future_subjects),
                event=random.choice(self.future_events),
                prefix=random.choice(self.emotional_prefixes),
                ending=random.choice(self.emotional_endings)
            )

        else:  # 7% mild/anxiety
            pattern = random.choice(self.patterns_soft)
            return pattern.format(
                event=random.choice(self.future_events),
                prefix=random.choice(self.emotional_prefixes)
            )

    # -------------------------------------------------------------
    # Generate Batch
    # -------------------------------------------------------------
    def _normalize_text(self, text: str) -> str:
        # Simple canonicalization to catch near-duplicates
        return " ".join(text.replace("،", ",").replace("  ", " ").strip().split())

    def generate_batch(self, n=9000, min_chars=12):
        """
        Generate up to n unique fear-like samples with light normalization to avoid duplicates.
        min_chars filters out very short artefacts.
        """
        samples = set()
        attempts = 0
        max_attempts = max(n * 50, 2000)  # cap runaway loops

        while len(samples) < n and attempts < max_attempts:
            attempts += 1
            sentence = self.generate_sentence()

            # Add occasional 2-sentence paragraphs
            if random.random() > 0.92:
                extra = random.choice(self.emotional_endings)
                sentence = sentence + "، " + extra

            norm = self._normalize_text(sentence)
            if len(norm) < min_chars:
                continue
            samples.add(norm)

        if len(samples) < n:
            print(f"Warning: generated {len(samples)} unique samples (target {n}); increase sources or attempts if needed.")

        return list(samples)

def to_schema(samples, channel_id="generated_fear", label="Fear", label_id=7):
    """
    Convert generated samples to the target schema:
    channelId, publishedAt, clean, token_count, Label, label_id.
    publishedAt uses current date/time; token_count is word count.
    """
    from datetime import datetime
    rows = []
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for s in samples:
        token_count = len(s.split())
        rows.append({
            "channelId": channel_id,
            "publishedAt": now,
            "clean": s,
            "token_count": token_count,
            "Label": label,
            "label_id": label_id
        })
    return rows


# -------------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------------
def main():

    augmenter = AfghanFearAugmentorPro()

    target_n = 9000
    print(f"\nGenerating {target_n} Afghan Fear samples...")

    samples = augmenter.generate_batch(target_n)

    print(f"Generated unique samples: {len(samples)}")

    df = pd.DataFrame({
        "text": samples,
        "label": "Fear",
        "source": "afghan_augmented_pro",
        "language": "fa",
        "region": "afghanistan",
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Save
    output_csv = "/content/drive/MyDrive/Sentiment_Analysis/Data/processed/augmented_afghan_fear_9000.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\nSaved to {output_csv}\n")

    # Show sample preview
    print("\nSample Generated Texts:\n" + "-"*60)
    for s in samples[:20]:
        print("-", s)

    print("\nDONE.\n")


if __name__ == "__main__":
    main()
