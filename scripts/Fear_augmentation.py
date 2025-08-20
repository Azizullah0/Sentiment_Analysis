import pandas as pd
from optimumEasyNMT import EasyNMT

# 1. Load dataset and extract Fear samples
data_path = "/content/drive/MyDrive/ColabFoulder/Labeled_400K.csv"
df = pd.read_csv(data_path)
fear_df = df[df["predicted_label"] == "LABEL_7"].copy()
print(f"Found {len(fear_df)} Fear samples")

# 2. Initialize translator using EasyNMT
translator = EasyNMT('opus-mt')

augmented_texts = []
for txt in fear_df["clean"].tolist():
    try:
        en = translator.translate([txt], target_lang='en')[0]
        fa = translator.translate([en], target_lang='fa')[0]
        if fa != txt:
            augmented_texts.append(fa)
    except Exception as e:
        print(f"⚠️ Error translating: {e}")

print(f"Generated {len(augmented_texts)} augmented Fear samples")

# 3. Save augmented data
aug_df = pd.DataFrame({
    "clean": augmented_texts,
    "label_id": [7] * len(augmented_texts)
})
aug_save_path = "/content/drive/MyDrive/ColabFoulder/fear_augmented.csv"
aug_df.to_csv(aug_save_path, index=False, encoding="utf-8")
print(f"Saved augmented examples to {aug_save_path}")
