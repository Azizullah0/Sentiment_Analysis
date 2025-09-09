import os
import datetime

# Simple path configuration - no drive mounting here
BASE_PATH = '/content/drive/MyDrive/Sentiment_Analysis'

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")


PATHS = {
    "raw_data": os.path.join(BASE_PATH, "Data", "raw", "sampled_for_labeling_cleaned.csv"),
    "labeled_data": os.path.join(BASE_PATH, "Data", "processed", "Labeled_4K.csv"),
    "base_model": os.path.join(BASE_PATH, "Models", "parsbert_emotion"),  # <--- use this for predict
    "finetuned_model": os.path.join(BASE_PATH, "Models", "parsbert_emotion"),  # overwrite to match
}

MODEL_CONFIG = {
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
}