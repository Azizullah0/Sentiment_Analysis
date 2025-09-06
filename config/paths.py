import os
import datetime
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    drive.mount('/content/drive')
    BASE_PATH = '/content/drive/MyDrive/ColabFoulder'
else:
    BASE_PATH = './data'

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

PATHS = {
    "raw_data": os.path.join(BASE_PATH, "sampled_for_labeling_cleaned.csv"),
    "labeled_data": os.path.join(BASE_PATH, "Labeled_4K.csv"),
    "base_model": os.path.join(BASE_PATH, "parsbert_emotion"),
    "finetuned_model": os.path.join(BASE_PATH, f"fine_tuned_model_{timestamp}"),
}
MODEL_CONFIG = {
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
}