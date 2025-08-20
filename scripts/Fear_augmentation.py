import pandas as pd
from easynmt import EasyNMT

# Load dataset
data_path = "/content/drive/MyDrive/ColabFoulder/Labeled_400K.csv"
df = pd.read_csv(data_path)

# Filter Fear samples
fear_df = df[df["predicted_label"] == "LABEL_7"].copy()
print(f"Found {len(fear_df)} Fear samples")

# Init translator
translator = EasyNMT('opus-mt')  # supports fa <-> en

augmented_texts = []
for txt in fear_df["clean"].tolist():
    try:
        en = translator.translate(txt, target_lang='en')   # fa -> en
        fa = translator.translate(en, target_lang='fa')   # en -> fa
        if fa != txt:
            augmented_texts.append(fa)
    except Exception as e:
        print(f"⚠️ Error translating: {e}")

# Save augmented Fear samples
aug_df = pd.DataFrame({
    "clean": augmented_texts,
    "label_id": [7] * len(augmented_texts)
})
aug_save_path = "/content/drive/MyDrive/ColabFoulder/fear_augmented.csv"
aug_df.to_csv(aug_save_path, index=False, encoding="utf-8")
print(f"✅ Saved augmented examples to {aug_save_path}")
