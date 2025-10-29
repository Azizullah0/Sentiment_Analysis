# config/paths.py
import os

BASE_PATH = '/content/drive/MyDrive/Sentiment_Analysis'

PATHS = {
    "base_model": "HooshvareLab/bert-base-parsbert-uncased",
    "parsbert_emotion": "/content/drive/MyDrive/Sentiment_Analysis/Models/parsbert_emotion",  # Your 4K model
    "incremental_finetuned_model": "/content/drive/MyDrive/Sentiment_Analysis/Models/parsbert_incremental_400k",
    "Combined_Labeled_Dataset": "/content/drive/MyDrive/Sentiment_Analysis/Data/processed/Combined_Labeled_Dataset.csv",
    "augmented_data": "/content/drive/MyDrive/Sentiment_Analysis/Data/processed/fear_augmented.csv",
    "datasets": "/content/drive/MyDrive/Sentiment_Analysis/datasets",
    "outputs": "/content/drive/MyDrive/Sentiment_Analysis/outputs",
}

MODEL_CONFIG = {
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,  # Base LR, we use lower for incremental
    "num_train_epochs": 5,
}