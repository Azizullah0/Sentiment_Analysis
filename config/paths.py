import os
import datetime

# Simple path configuration - no drive mounting here
BASE_PATH = '/content/drive/MyDrive/Sentiment_Analysis'

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")


PATHS = {
    "finetuned_model": "/content/drive/MyDrive/Sentiment_Analysis/Models/parsbert_emotion",
    "datasets": "/content/drive/MyDrive/Sentiment_Analysis/datasets",
    "outputs": "/content/drive/MyDrive/Sentiment_Analysis/outputs",
    "raw_data": "/content/drive/MyDrive/Sentiment_Analysis/datasets/Cleaned_Dataset.csv",
    "output_labeled": "/content/drive/MyDrive/Sentiment_Analysis/outputs/Labeled_400K.csv"
}

MODEL_CONFIG = {
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
}