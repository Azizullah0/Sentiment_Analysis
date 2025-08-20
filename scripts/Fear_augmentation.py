import pandas as pd
from transformers import MarianMTModel, MarianTokenizer

# ================================
# 1. Load Fear samples
# ================================
data_path = "/content/drive/MyDrive/ColabFoulder/Labeled_400K.csv"
df = pd.read_csv(data_path)

fear_df = df[df["predicted_label"] == "LABEL_7"].copy()
print(f"Found {len(fear_df)} Fear samples")

# ================================
# 2. Load MarianMT translation models
# Persian ↔ English
# ================================
src_lang = "fa"  # Persian/Dari
tgt_lang = "en"

# Persian → English
model_name_fa_en = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
tokenizer_fa_en = MarianTokenizer.from_pretrained(model_name_fa_en)
model_fa_en = MarianMTModel.from_pretrained(model_name_fa_en)

# English → Persian
model_name_en_fa = f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}"
tokenizer_en_fa = MarianTokenizer.from_pretrained(model_name_en_fa)
model_en_fa = MarianMTModel.from_pretrained(model_name_en_fa)

def translate(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# ================================
# 3. Back-translate (Fa → En → Fa)
# ================================
augmented_texts = []
for txt in fear_df["clean"].tolist():
    try:
        # Persian → English
        en_text = translate([txt], tokenizer_fa_en, model_fa_en)[0]
        # English → Persian
        fa_text = translate([en_text], tokenizer_en_fa, model_en_fa)[0]
        if fa_text != txt:  # keep only if changed
            augmented_texts.append(fa_text)
    except Exception as e:
        print(f"⚠️ Skipped a sample due to error: {e}")

print(f"Generated {len(augmented_texts)} new Fear samples")

# ================================
# 4. Save augmented dataset
# ================================
aug_df = pd.DataFrame({
    "clean": augmented_texts,
    "label_id": [7] * len(augmented_texts),   # Fear label
    "emotion": ["Fear"] * len(augmented_texts)
})

aug_save_path = "/content/drive/MyDrive/ColabFoulder/fear_augmented.csv"
aug_df.to_csv(aug_save_path, index=False, encoding="utf-8")
print(f"✅ Augmented Fear dataset saved to {aug_save_path}")
