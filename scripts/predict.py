import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset


model_path = "/content/drive/MyDrive/parsbert400_emotion"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


df = pd.read_csv("/content/drive/MyDrive/ColabFoulder/PeopleMedia.csv")
dataset = Dataset.from_pandas(df)


def tokenize(batch):
    return tokenizer(batch["clean"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask"])


loader = DataLoader(dataset, batch_size=32)
preds = []
with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        preds.extend(batch_preds)

df["predicted_label"] = preds
df.to_csv("datasets/peopleLabeled.csv", index=False)
print("Saved predictions to datasets/peopleLabeled.csv")