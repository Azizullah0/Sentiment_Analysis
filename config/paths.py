import os
import datetime

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

IN_COLAB = is_colab()

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = '/content/drive/MyDrive/Sentiment_Analysis'
else:
    BASE_PATH = './data'

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

PATHS = {
    "raw_data": os.path.join(BASE_PATH, "Data", "raw", "sampled_for_labeling_cleaned.csv"),
    "labeled_data": os.path.join(BASE_PATH, "Data", "processed", "Labeled_4K.csv"),
    "base_model": os.path.join(BASE_PATH, "Models", "parsbert_emotion"),
    "finetuned_model": os.path.join(BASE_PATH, "Models", f"fine_tuned_model_{timestamp}"),
}

MODEL_CONFIG = {
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
}