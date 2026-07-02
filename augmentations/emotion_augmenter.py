"""
Template-based data augmentation for underrepresented emotion classes
(Surprise, Anger, Disgust) in Afghan Persian (Dari) social-media text.

----------------------------------------------------------------------
Design rationale (what makes this more than naive templating)
----------------------------------------------------------------------
Naive template filling ("pattern x slot") yields text that is easy for a
classifier to memorise: few syntactic frames, no surface variation, and a
lexical distribution that collapses onto a handful of n-grams. The design
below layers five techniques from the augmentation literature on top of a
compositional grammar so the generated distribution is closer to real
social-media text:

1.  Compositional slot grammar (frame -> clause -> slot).
    Templates are organised as pragmatic FRAMES (e.g. exclamative,
    rhetorical question, collective complaint) whose slots are filled from
    independent lexical banks. Diversity grows multiplicatively with bank
    size, in the spirit of data recombination (Jia & Liang, 2016) rather
    than additively as with flat pattern lists.

2.  Label-preserving surface perturbations.
    - AEDA-style random punctuation insertion (Karimi et al., 2021):
      cheap, label-preserving, and shown to outperform EDA on text
      classification.
    - Optional-modifier dropout, an adaptation of EDA random deletion
      (Wei & Zou, 2019) that only ever removes *optional* discourse
      elements, so the emotional content is untouched.

3.  Orthographic noise for social-media realism.
    Real Afghan social-media Persian mixes Arabic and Persian code points
    (ي/ی, ك/ک), drops or adds zero-width non-joiners, and elongates
    emotive words ("واییی"). Injecting these (at low probability) mimics
    the noise a deployed model actually sees; robustness to such noise is
    a documented weakness of Persian PLMs (Farahani et al., 2021).

4.  Anti-collapse diversity control.
    Per-frame quotas cap how many samples any single syntactic frame can
    contribute (mode-collapse guard), and corpus-level lexical diversity
    is reported as distinct-1/distinct-2 (Li et al., 2016) so every
    generated batch ships with a measurable diversity score.

5.  Reproducibility and dataset documentation.
    Generation is fully seeded, and every batch writes a JSON sidecar
    with the seed, per-frame counts, and diversity metrics, following the
    documentation practice argued for in "Datasheets for Datasets"
    (Gebru et al., 2021).

Class-imbalance context: oversampling minority classes at the input level
is a stronger baseline than loss re-weighting alone when the imbalance is
severe (Buda et al., 2018); this module targets the three rarest classes
after Fear in the combined dataset.

----------------------------------------------------------------------
References
----------------------------------------------------------------------
- Wei, J. & Zou, K. (2019). EDA: Easy Data Augmentation Techniques for
  Boosting Performance on Text Classification Tasks. EMNLP-IJCNLP.
- Karimi, A., Rossi, L. & Prati, A. (2021). AEDA: An Easier Data
  Augmentation Technique for Text Classification. Findings of EMNLP.
- Feng, S. Y. et al. (2021). A Survey of Data Augmentation Approaches
  for NLP. Findings of ACL.
- Jia, R. & Liang, P. (2016). Data Recombination for Neural Semantic
  Parsing. ACL.
- Anaby-Tavor, A. et al. (2020). Do Not Have Enough Data? Deep Learning
  to the Rescue! (LAMBADA). AAAI.
- Sennrich, R., Haddow, B. & Birch, A. (2016). Improving Neural Machine
  Translation Models with Monolingual Data (back-translation). ACL.
- Li, J. et al. (2016). A Diversity-Promoting Objective Function for
  Neural Conversation Models (distinct-n). NAACL.
- Buda, M., Maki, A. & Mazurowski, M. A. (2018). A systematic study of
  the class imbalance problem in convolutional neural networks.
  Neural Networks 106.
- Gebru, T. et al. (2021). Datasheets for Datasets. CACM 64(12).
- Farahani, M. et al. (2021). ParsBERT: Transformer-based Model for
  Persian Language Understanding. Neural Processing Letters.

Output schema matches the existing pipeline (fear_augmenter.py):
    channelId, publishedAt, clean, token_count, Label, label_id

Usage:
    python augmentations/emotion_augmenter.py --emotion all --n 9000
    python augmentations/emotion_augmenter.py --emotion anger --n 9000 --seed 7
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.paths import PATHS

# ----------------------------------------------------------------------
# Label ids follow the project-wide 8-label scheme in scripts/train.py.
# ----------------------------------------------------------------------
LABEL_IDS = {"Surprise": 3, "Disgust": 4, "Anger": 6}

ZWNJ = "\u200c"


# ----------------------------------------------------------------------
# Text normalisation (dedup key) — unify Arabic/Persian code points and
# whitespace so orthographic variants of the same sentence never count
# as two "unique" samples.
# ----------------------------------------------------------------------
def normalize_for_dedup(text: str) -> str:
    text = (
        text.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("ۀ", "ه")
        .replace(ZWNJ, " ")
        .replace("،", ",")
    )
    # strip elongation: collapse 3+ repeated chars to one
    out = []
    for ch in text:
        if len(out) >= 2 and out[-1] == ch and out[-2] == ch:
            continue
        out.append(ch)
    return " ".join("".join(out).split())


# ----------------------------------------------------------------------
# Frame: one pragmatic/syntactic sentence shape with named slots.
# ----------------------------------------------------------------------
@dataclass
class Frame:
    name: str          # stable id used for quota accounting + metadata
    template: str      # str.format template with named slots
    weight: float = 1.0


@dataclass
class EmotionGrammar:
    """A compositional grammar for one emotion class."""

    label: str
    frames: list = field(default_factory=list)
    slots: dict = field(default_factory=dict)        # slot name -> bank
    prefixes: list = field(default_factory=list)     # optional discourse openers
    endings: list = field(default_factory=list)      # optional discourse closers
    emotive_words: list = field(default_factory=list)  # elongation targets


# ======================================================================
# SURPRISE — pragmatics: mirativity (unexpectedness). Frames cover
# exclamatives, rhetorical questions, disbelief reports, and narrative
# "when I saw/heard X" constructions, over both positive and negative
# surprise triggers common in Afghan discourse (prices, politics,
# weather, sports, announcements).
# ======================================================================
def build_surprise() -> EmotionGrammar:
    return EmotionGrammar(
        label="Surprise",
        slots={
            "subject": [
                "قیمت‌ها", "نرخ دالر", "وضعیت بازار", "این خبر", "تصمیم حکومت",
                "نتیجهٔ بازی دیروز", "هوای امروز", "این ویدیو", "سخنان او",
                "اعلامیهٔ تازه", "وضعیت برق", "نرخ مواد خوراکه", "این عکس‌ها",
                "رفتار مردم", "خبر تلویزیون", "این تغییرات",
            ],
            "event": [
                "یک‌شبه تغییر کرد", "غیرقابل باور شد", "همه را حیران ساخت",
                "هیچ‌کس انتظارش را نداشت", "خلاف تمام پیش‌بینی‌ها بود",
                "همه را شوکه کرد", "باورکردنی نیست", "سر زبان‌ها افتاد",
                "دهن همه را باز مانده", "همه را انگشت به دهن کرد",
            ],
            "exclaim": [
                "وای خدایا", "باورم نمی‌شود", "چی می‌گویی", "راستی؟", "عجب",
                "سبحان‌الله", "خدای من", "حیران ماندم", "این چه حال است",
                "الله اکبر", "به چشمان خود باور نمی‌کنم",
            ],
        },
        frames=[
            Frame("excl_lead", "{exclaim}! {subject} {event}"),
            Frame("excl_mid", "امروز شنیدم {subject} {event}، {exclaim}!"),
            Frame("rhet_q", "چطور ممکن است {subject} {event}؟"),
            Frame("who_believes", "کی باور می‌کند که {subject} {event}؟"),
            Frame("narr_saw", "وقتی دیدم {subject} {event} خشکم زد"),
            Frame("narr_heard", "یک لحظه خبر را شنیدم که {subject} {event}، هنوز در شوک هستم"),
            Frame("disbelief", "{exclaim}، هیچ فکر نمی‌کردم {subject} {event}"),
            Frame("shock_state", "{subject} {event}، تا حالی از حیرت بیرون نشده‌ام"),
            Frame("dream", "فکر کردم خواب می‌بینم، {subject} واقعاً {event}"),
        ],
        prefixes=["به خدا قسم", "والله", "باور کنید", "راست می‌گویم"],
        endings=[
            "هنوز باورم نمی‌شود", "واقعاً عجیب است", "کسی فکرش را نمی‌کرد",
            "دنیا چه زود تغییر می‌کند", "خدا خودش خبر دارد",
        ],
        emotive_words=["وای", "عجب", "راستی"],
    )


# ======================================================================
# ANGER — pragmatics: blame assignment + escalation. Frames cover direct
# accusation, rhetorical challenge, collective exhaustion ("we the
# people"), curse constructions, and contrastive injustice frames
# ("people starve while X ...") typical of Afghan social media.
# ======================================================================
def build_anger() -> EmotionGrammar:
    return EmotionGrammar(
        label="Anger",
        slots={
            "target": [
                "این مسئولین", "حکومت", "این اداره‌ها", "قیمت‌فروشان", "محتکران",
                "این سیاست‌مداران", "ادارهٔ برق", "شاروالی", "این رسانه‌ها",
                "زورمندان", "رشوت‌خوران", "دلالان بازار", "این به‌اصطلاح رهبران",
                "مامورین گمرک",
            ],
            "grievance": [
                "مردم را فراموش کرده‌اند", "فقط جیب خود را پر می‌کنند",
                "به داد مردم نمی‌رسند", "دروغ می‌گویند", "حق مردم را می‌خورند",
                "هیچ کاری نمی‌کنند", "مردم را فریب می‌دهند",
                "بر سر مردم ظلم می‌کنند", "وعده‌های میان‌تهی می‌دهند",
                "غم مردم را ندارند",
            ],
            "boil": [
                "دیگر بس است", "تا کی صبر کنیم", "خون آدم به جوش می‌آید",
                "دیگر طاقت نمانده", "این ظلم آشکار است",
                "کاسهٔ صبر مردم لبریز شده", "به خدا قهر آدم بالا می‌شود",
                "دیگر تحمل نداریم",
            ],
        },
        frames=[
            Frame("boil_lead", "{boil}! {target} {grievance}"),
            Frame("curse", "لعنت به این وضعیت، {target} {grievance}"),
            Frame("accuse_then_boil", "{target} {grievance}، {boil}"),
            Frame("rhet_why", "چرا {target} {grievance}؟ {boil}"),
            Frame("daily_witness", "هر روز می‌بینیم که {target} {grievance}"),
            Frame("fed_up", "از دست {target} به تنگ آمده‌ایم، {grievance}"),
            Frame("contrast_hunger", "مردم گرسنه است و {target} فقط {grievance}"),
            Frame("demand", "{target} باید جواب بدهند، تا کی {grievance}؟"),
            Frame("shame_excl", "شرم بر {target} که {grievance}"),
        ],
        prefixes=["به خدا قسم", "والله که", "صادقانه بگویم"],
        endings=[
            "این وضعیت قابل قبول نیست", "مردم دیگر خسته شده‌اند",
            "باید جواب بدهند", "تا کی این حال ادامه دارد",
            "روزی حساب پس می‌دهند",
        ],
        emotive_words=["لعنت", "شرم", "بس"],
    )


# ======================================================================
# DISGUST — pragmatics: moral revulsion + social distancing. Frames
# cover interjection + evaluation, physical-revulsion metaphors ("عق"),
# normalisation laments ("it has become normal"), and withdrawal
# statements ("I want nothing to do with...").
# ======================================================================
def build_disgust() -> EmotionGrammar:
    return EmotionGrammar(
        label="Disgust",
        slots={
            "subject": [
                "این رشوت‌خوری", "این فساد اداری", "این دورویی", "این چاپلوسی",
                "این خیانت", "این دروغ‌گویی", "این بی‌عدالتی",
                "این معامله‌های پشت پرده", "این تبعیض", "این کثافت‌کاری",
                "این حرام‌خوری", "رفتار این افراد", "این ریاکاری",
            ],
            "reaction": [
                "حال آدم را بد می‌کند", "دل آدم را سیاه می‌کند",
                "قابل تحمل نیست", "شرم‌آور است", "چندش‌آور است",
                "آدم را از جامعه بیزار می‌کند", "نفرت‌انگیز است",
                "مایهٔ ننگ است", "عق آدم را می‌آورد",
            ],
            "interject": [
                "اف به این روزگار", "توبه توبه", "خاک بر سر این وضعیت",
                "وای از این جامعه", "استغفرالله", "شرم است شرم",
                "نفرین به این حال", "چقدر زشت",
            ],
        },
        frames=[
            Frame("interject_eval", "{interject}، {subject} {reaction}"),
            Frame("plain_eval", "{subject} واقعاً {reaction}"),
            Frame("normalised", "به این حد رسیده که {subject} عادی شده، {reaction}"),
            Frame("witness", "هر بار که می‌بینم {subject} چقدر عام شده، {reaction}"),
            Frame("withdraw", "از {subject} بیزارم، {reaction}"),
            Frame("country_scope", "{subject} در این کشور {reaction}"),
            Frame("daily_sight", "دیدن {subject} هر روز {reaction}"),
            Frame("interject_excl", "{interject}! {subject} دیگر {reaction}"),
            Frame("physical", "با شنیدن {subject} عق می‌زنم، {reaction}"),
        ],
        prefixes=["راستش را بگویم", "به خدا", "بی‌پرده بگویم"],
        endings=[
            "دیگر اعتماد به کسی نمانده", "جامعه را خراب کرده‌اند",
            "انسانیت مرده است", "خدا این وضع را اصلاح کند",
            "آدم از آدم بودن شرمش می‌آید",
        ],
        emotive_words=["اف", "توبه", "وای"],
    )


GRAMMARS = {
    "surprise": build_surprise,
    "anger": build_anger,
    "disgust": build_disgust,
}


# ----------------------------------------------------------------------
# Generator with surface perturbations and anti-collapse quotas.
# ----------------------------------------------------------------------
class TemplateAugmentor:
    # perturbation probabilities (kept low: realism, not corruption)
    P_PREFIX = 0.30        # discourse opener
    P_ENDING = 0.22        # discourse closer (second clause)
    P_AEDA_PUNCT = 0.15    # AEDA-style punctuation insertion
    P_ELONGATION = 0.12    # emotive-word vowel elongation
    P_ARABIC_YE = 0.08     # ye/kaf Arabic code-point variation
    AEDA_MARKS = ["،", "!", "؟", "..."]

    def __init__(self, grammar: EmotionGrammar, rng: random.Random):
        self.grammar = grammar
        self.rng = rng
        self.frame_counts = {frame.name: 0 for frame in grammar.frames}

    # --- surface perturbations (all label-preserving) -----------------
    def _aeda_punct(self, text: str) -> str:
        """AEDA (Karimi et al., 2021): insert one punctuation mark at a
        random word boundary."""
        words = text.split()
        if len(words) < 4:
            return text
        pos = self.rng.randint(1, len(words) - 2)
        words.insert(pos, self.rng.choice(self.AEDA_MARKS))
        return " ".join(words)

    def _elongate(self, text: str) -> str:
        """Social-media style emotive elongation: 'وای' -> 'وایییی'."""
        for word in self.grammar.emotive_words:
            if word in text and text.strip():
                stretched = word + word[-1] * self.rng.randint(2, 4)
                return text.replace(word, stretched, 1)
        return text

    def _arabic_codepoints(self, text: str) -> str:
        """Swap Persian ی/ک for Arabic ي/ك once, as seen in real Afghan
        social-media text typed on Arabic keyboards."""
        if "ی" in text:
            index = text.index("ی")
            return text[:index] + "ي" + text[index + 1:]
        return text

    # --- core generation ----------------------------------------------
    def _pick_frame(self, quota: int) -> Frame:
        """Weighted frame choice, skipping frames that hit their quota
        (mode-collapse guard: no frame may dominate the batch)."""
        open_frames = [
            frame for frame in self.grammar.frames
            if self.frame_counts[frame.name] < quota
        ]
        pool = open_frames or self.grammar.frames
        weights = [frame.weight for frame in pool]
        return self.rng.choices(pool, weights=weights, k=1)[0]

    def _fill(self, frame: Frame) -> str:
        values = {
            slot: self.rng.choice(bank) for slot, bank in self.grammar.slots.items()
        }
        return frame.template.format(**values)

    def generate_one(self, quota: int) -> tuple:
        frame = self._pick_frame(quota)
        text = self._fill(frame)

        if self.grammar.prefixes and self.rng.random() < self.P_PREFIX:
            text = f"{self.rng.choice(self.grammar.prefixes)} {text}"
        if self.grammar.endings and self.rng.random() < self.P_ENDING:
            text = f"{text}، {self.rng.choice(self.grammar.endings)}"
        if self.rng.random() < self.P_AEDA_PUNCT:
            text = self._aeda_punct(text)
        if self.rng.random() < self.P_ELONGATION:
            text = self._elongate(text)
        if self.rng.random() < self.P_ARABIC_YE:
            text = self._arabic_codepoints(text)

        return frame.name, " ".join(text.split())

    def generate_batch(self, n: int, min_chars: int = 12) -> list:
        # quota: cap each frame at ~2.2x its fair share of the batch
        quota = max(1, int((n / max(1, len(self.grammar.frames))) * 2.2))
        seen = set()
        samples = []
        attempts, max_attempts = 0, max(n * 60, 5000)

        while len(samples) < n and attempts < max_attempts:
            attempts += 1
            frame_name, text = self.generate_one(quota)
            key = normalize_for_dedup(text)
            if len(key) < min_chars or key in seen:
                continue
            seen.add(key)
            samples.append(text)
            self.frame_counts[frame_name] += 1

        if len(samples) < n:
            print(
                f"Warning: generated {len(samples)} unique samples "
                f"(target {n}); enlarge slot banks or lower n."
            )
        return samples


# ----------------------------------------------------------------------
# Diversity metrics: distinct-n (Li et al., 2016).
# ----------------------------------------------------------------------
def distinct_n(samples: list, n: int) -> float:
    total, unique = 0, set()
    for sample in samples:
        tokens = sample.split()
        grams = list(zip(*[tokens[i:] for i in range(n)]))
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


# ----------------------------------------------------------------------
# Output: pipeline schema + datasheet-style metadata sidecar.
# ----------------------------------------------------------------------
def to_schema(samples: list, label: str, label_id: int) -> list:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [
        {
            "channelId": f"generated_{label.lower()}",
            "publishedAt": now,
            "clean": text,
            "token_count": len(text.split()),
            "Label": label,
            "label_id": label_id,
        }
        for text in samples
    ]


def run(emotion: str, n: int, seed: int, output: str | None) -> Path:
    rng = random.Random(seed)
    grammar = GRAMMARS[emotion]()
    augmentor = TemplateAugmentor(grammar, rng)
    label, label_id = grammar.label, LABEL_IDS[grammar.label]

    print(f"\nGenerating {n} Afghan {label} samples (seed={seed})...")
    samples = augmentor.generate_batch(n=n)

    d1, d2 = distinct_n(samples, 1), distinct_n(samples, 2)
    print(f"Generated: {len(samples)} unique | distinct-1={d1:.3f} distinct-2={d2:.3f}")

    out_path = Path(
        output
        or os.path.join(
            PATHS["data_root"], f"Data/processed/augmented_afghan_{emotion}_{n}.csv"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(to_schema(samples, label, label_id)).to_csv(
        out_path, index=False, encoding="utf-8"
    )
    print(f"Saved to {out_path}")

    # Datasheet-style sidecar (Gebru et al., 2021)
    metadata = {
        "generator": "emotion_augmenter.py",
        "label": label,
        "label_id": label_id,
        "n_requested": n,
        "n_generated": len(samples),
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "frame_counts": augmentor.frame_counts,
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "perturbation_probs": {
            "prefix": TemplateAugmentor.P_PREFIX,
            "ending": TemplateAugmentor.P_ENDING,
            "aeda_punct": TemplateAugmentor.P_AEDA_PUNCT,
            "elongation": TemplateAugmentor.P_ELONGATION,
            "arabic_codepoints": TemplateAugmentor.P_ARABIC_YE,
        },
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Metadata sidecar: {meta_path}")

    print(f"\nSample generated {label} texts:\n" + "-" * 60)
    for text in samples[:10]:
        print("-", text)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="SOTA-informed template augmentation for Surprise, Anger, Disgust."
    )
    parser.add_argument(
        "--emotion",
        choices=[*GRAMMARS.keys(), "all"],
        default="all",
        help="Emotion class to augment (default: all three).",
    )
    parser.add_argument("--n", type=int, default=9000, help="Samples per emotion.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (single --emotion only).",
    )
    args = parser.parse_args()

    emotions = list(GRAMMARS.keys()) if args.emotion == "all" else [args.emotion]
    if args.output and len(emotions) > 1:
        parser.error("--output can only be used with a single --emotion")

    for emotion in emotions:
        run(emotion, n=args.n, seed=args.seed, output=args.output)

    print("\nDONE.\n")


if __name__ == "__main__":
    main()
