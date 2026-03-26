import argparse
import random
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

# ----------------------------------------------------
# AFGHAN FEAR AUGMENTOR (Improved)
# ----------------------------------------------------
# - Stronger uniqueness with normalization
# - Reproducible (seedable)
# - Balanced module sampling
# - Writes required schema: channelId, publishedAt, clean, token_count, Label, label_id
# ----------------------------------------------------

OUTPUT_CSV = "/content/drive/MyDrive/Sentiment_Analysis/Data/processed/augmented_afghan_fear_9000.csv"


class AfghanFearAugmentor:
    def __init__(self, rng: random.Random):
        self.rng = rng

        self.war_groups = [
            "طالبان", "داعش", "پاکستان", "امریکا", "ایران", "حکومت",
            "مجاهدین", "کوچی‌ها", "نظام قبلی", "تروریستان", "جنگ‌سالاران",
            "شبکه حقانی", "طالب", "استخبارات پاکستان", "عناصر خارجی"
        ]

        self.insults = [
            "وحشی‌ها", "کثیف‌ها", "قاتلان", "دزدان", "بی‌غیرت‌ها",
            "نوکران پاکستان", "جاسوس‌ها", "احمق‌ها", "خائنان", "بی‌سوادها"
        ]

        self.future_subjects = [
            "آیندهٔ کشور", "آیندهٔ کودکان", "مردم بیچاره ما",
            "نسل بعد", "افغانستان", "شهرهای ما", "جوان‌ها",
            "مهاجرین", "وضعیت اقتصادی", "ناامنی‌ها"
        ]

        self.war_events = [
            "حمله کنند", "انفجار شود", "جنگ دوباره شروع شود",
            "مردم را بکشند", "ترور کنند", "درگیری شدید شود",
            "امنیت را نابود کنند", "شهر سقوط کند"
        ]
        self.hate_events = [
            "باز مردم را آزار بدهند", "خیانت کنند", "باعث خاک‌ساری مردم شوند",
            "باز ظلم کنند", "آرامی را از بین ببرند", "کشور را نابود کنند"
        ]
        self.future_events = [
            "بدتر شود", "تباه شود", "همه چیز خراب‌تر شود",
            "مردم ناامید شوند", "هیچ آینده‌ای باقی نماند",
            "زندگی سخت‌تر شود", "مهاجرت بیشتر شود"
        ]

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

        # Patterns
        self.patterns_war = [
            "می‌ترسم {group} دوباره {event}",
            "{prefix} {group} هر لحظه ممکن است {event}، {ending}",
            "از دست {group} وحشت دارم که باز {event}",
            "هر روز با ترس زندگی میکنیم که {group} {event}",
            "هراس دارم اگر {group} {event}، مردم بدبخت می‌شوند",
            "{group} اگر همینطور ادامه دهد، ممکن است {event}"
        ]
        self.patterns_hate = [
            "{prefix} این {insult} ممکن است دوباره {event}",
            "لعنت به این {insult} که هر لحظه ممکن است {event}",
            "{insult} اگر باز {event}، مردم نابود می‌شوند",
            "از دست این {insult} هیچ آرامی نمانده، میترسم {event}"
        ]
        self.patterns_future = [
            "نگران {subject} هستم که {event}",
            "{prefix} {subject} ممکن است {event}",
            "از آیندهٔ {subject} میترسم، شاید {event}",
            "همه میگویند {subject} قرار است {event}، {ending}",
            "گاهی فکر میکنم اگر {subject} {event} چه خواهد شد"
        ]
        self.patterns_soft = [
            "این روزها از همه چیز میترسم، مخصوصاً اینکه {event}",
            "گاهی نمیفهمم چی میشه، فقط حس میکنم {event}",
            "{prefix} احساس بدی دارم که شاید {event}",
            "دل‌آشوب میشم وقتی فکر میکنم {event}"
        ]

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.replace("،", ",").split())

    def generate_sentence(self) -> str:
        r = self.rng.random()
        if r < 0.55:
            pattern = self.rng.choice(self.patterns_war)
            return pattern.format(
                group=self.rng.choice(self.war_groups),
                event=self.rng.choice(self.war_events),
                prefix=self.rng.choice(self.emotional_prefixes),
                ending=self.rng.choice(self.emotional_endings),
            )
        elif r < 0.80:
            pattern = self.rng.choice(self.patterns_hate)
            return pattern.format(
                insult=self.rng.choice(self.insults),
                event=self.rng.choice(self.hate_events),
                prefix=self.rng.choice(self.emotional_prefixes),
                ending=self.rng.choice(self.emotional_endings),
            )
        elif r < 0.93:
            pattern = self.rng.choice(self.patterns_future)
            return pattern.format(
                subject=self.rng.choice(self.future_subjects),
                event=self.rng.choice(self.future_events),
                prefix=self.rng.choice(self.emotional_prefixes),
                ending=self.rng.choice(self.emotional_endings),
            )
        else:
            pattern = self.rng.choice(self.patterns_soft)
            return pattern.format(
                event=self.rng.choice(self.future_events),
                prefix=self.rng.choice(self.emotional_prefixes),
            )

    def generate_batch(self, n=9000, min_chars=12, two_sentence_prob=0.08):
        samples = set()
        attempts = 0
        max_attempts = max(n * 50, 2000)
        while len(samples) < n and attempts < max_attempts:
            attempts += 1
            sentence = self.generate_sentence()
            if self.rng.random() < two_sentence_prob:
                sentence = sentence + "، " + self.rng.choice(self.emotional_endings)
            norm = self._normalize_text(sentence)
            if len(norm) < min_chars:
                continue
            samples.add(norm)
        if len(samples) < n:
            print(f"Warning: generated {len(samples)} unique samples (target {n}); increase sources or attempts if needed.")
        return list(samples)


def to_schema(samples, channel_id="generated_fear", label="Fear", label_id=7):
    rows = []
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for s in samples:
        rows.append({
            "channelId": channel_id,
            "publishedAt": now,
            "clean": s,
            "token_count": len(s.split()),
            "Label": label,
            "label_id": label_id,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=9000, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    augmenter = AfghanFearAugmentor(rng)

    print(f"\nGenerating {args.n} Afghan Fear samples (seed={args.seed})...")
    samples = augmenter.generate_batch(n=args.n)
    print(f"Generated unique samples: {len(samples)}")

    df = pd.DataFrame(to_schema(samples))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved to {out_path}\n")

    print("\nSample Generated Texts:\n" + "-" * 60)
    for s in samples[:20]:
        print("-", s)
    print("\nDONE.\n")


if __name__ == "__main__":
    main()
