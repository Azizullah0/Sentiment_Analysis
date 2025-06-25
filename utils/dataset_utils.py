import os

# File: utils/dataset_utils.py
dataset_utils_code = """
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer

def load_dataset(csv_path, label_col='label_id'):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={label_col: 'labels'})
    return df

def split_dataset(df, test_size=0.2, seed=42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['labels'], random_state=seed)
    return train_df, test_df

def tokenize_datasets(train_df, test_df, model_name="HooshvareLab/bert-base-parsbert-uncased", max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    def tokenize(batch):
        return tokenizer(batch["clean"], padding="max_length", truncation=True, max_length=max_length)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, test_dataset, tokenizer
"""