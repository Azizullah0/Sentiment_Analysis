predict_script_code = """
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.dataset_utils import tokenize_datasets
from datasets import Dataset

# Load model and tokenizer
model_path = "models/parsbert_emotion"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load data
df = pd.read_csv("datasets/Cleaned_Dataset.csv")  # or any unlabeled data
dataset = Dataset.from_pandas(df)

# Tokenize
def tokenize(batch):
    return tokenizer(batch["clean"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# Predict
preds = []
for item in dataset:
    input_ids = item["input_ids"].unsqueeze(0)
    attention_mask = item["attention_mask"].unsqueeze(0)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    pred = torch.argmax(output.logits, dim=1).item()
    preds.append(pred)

df["predicted_label"] = preds
df.to_csv("datasets/Cleaned_Dataset_Labeled.csv", index=False)
print("Saved predictions to Cleaned_Dataset_Labeled.csv")
"""

import pandas as pd
from ace_tools import display_dataframe_to_user

scripts_df = pd.DataFrame({
    "Filename": ["utils/dataset_utils.py", "scripts/train.py", "scripts/predict.py"],
    "Purpose": ["Loading, splitting, and tokenizing datasets",
                "Training the model using Trainer API",
                "Predicting and labeling new data with trained model"]
})

display_dataframe_to_user("Sentiment Analysis Project Scripts", scripts_df)

(train_script_code, predict_script_code, dataset_utils_code)