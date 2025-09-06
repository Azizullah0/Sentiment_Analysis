import os
import datetime

IN_COLAB = 'google.colab' in str(get_ipython()) if hasattr(__builtins__, '__IPYTHON__') else False

if IN_COLAB:
    from google.colab import drive
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