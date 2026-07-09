import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _expand(path_value):
    return os.path.abspath(os.path.expanduser(path_value))


def _existing_dir(path_value):
    if not path_value:
        return None
    candidate = _expand(path_value)
    return candidate if os.path.isdir(candidate) else None


def _prefer_existing(candidates):
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def get_storage_root():
    """
    Resolve where heavy artifacts should be stored.

    Priority:
    1. SENTIMENT_STORAGE_ROOT
    2. SENTIMENT_BASE_PATH (backward compatibility)
    3. Local project directory
    4. Legacy Google Drive path, only if it already exists
    """
    env_storage = _existing_dir(os.environ.get("SENTIMENT_STORAGE_ROOT"))
    if env_storage:
        return env_storage

    env_base = _existing_dir(os.environ.get("SENTIMENT_BASE_PATH"))
    if env_base:
        return env_base

    fallback = _prefer_existing(
        [
            PROJECT_ROOT,
            _existing_dir("~/Sentiment_Analysis"),
            _existing_dir("~/gdrive/Sentiment_Analysis"),
        ]
    )
    return fallback or PROJECT_ROOT


def get_data_root():
    """
    Resolve where datasets and processed CSV files live.

    This is separate from the storage root so models/checkpoints can move off
    Google Drive without forcing the datasets to move at the same time.
    """
    env_data = _existing_dir(os.environ.get("SENTIMENT_DATA_ROOT"))
    if env_data:
        return env_data

    env_base = _existing_dir(os.environ.get("SENTIMENT_BASE_PATH"))
    if env_base:
        return env_base

    for candidate in [
        PROJECT_ROOT,
        _existing_dir("~/Sentiment_Analysis"),
        _existing_dir("~/gdrive/Sentiment_Analysis"),
        "/content/drive/MyDrive/Sentiment_Analysis",
    ]:
        if candidate and os.path.isdir(os.path.join(candidate, "Data")):
            return candidate

    return PROJECT_ROOT


STORAGE_ROOT = get_storage_root()
DATA_ROOT = get_data_root()
BASE_PATH = STORAGE_ROOT

for path in [
    os.path.join(STORAGE_ROOT, "Models"),
    os.path.join(STORAGE_ROOT, "outputs"),
    os.path.join(DATA_ROOT, "datasets"),
    os.path.join(DATA_ROOT, "Data/processed"),
]:
    os.makedirs(path, exist_ok=True)

local_pretrained_candidates = [
    os.path.join(STORAGE_ROOT, "Models/bert-base-parsbert-uncased"),
    os.path.join(DATA_ROOT, "Models/bert-base-parsbert-uncased"),
]
local_pretrained_path = next(
    (path for path in local_pretrained_candidates if os.path.exists(path)),
    local_pretrained_candidates[0],
)
pretrained_base = (
    local_pretrained_path
    if os.path.exists(local_pretrained_path)
    else "HooshvareLab/bert-base-parsbert-uncased"
)

PATHS = {
    "base_path": BASE_PATH,
    "project_root": PROJECT_ROOT,
    "storage_root": STORAGE_ROOT,
    "data_root": DATA_ROOT,
    "pretrained_base": pretrained_base,
    "fine_tuned_model": os.path.join(STORAGE_ROOT, "Models/parsbert_emotion"),
    "base_model": os.path.join(STORAGE_ROOT, "Models/parsbert_emotion"),
    "parsbert_emotion": os.path.join(STORAGE_ROOT, "Models/parsbert_emotion"),
    "incremental_finetuned_model": os.path.join(STORAGE_ROOT, "Models/parsbert_emotion_incremental"),
    "Combined_Labeled_Dataset": os.path.join(DATA_ROOT, "Data/processed/Combined_Labeled_Dataset.csv"),
    "Combined_Labeled_Dataset_with_fearAug": os.path.join(
        DATA_ROOT, "Data/processed/Combined_Labeled_Dataset_with_fearAug.csv"
    ),
    "Combined_Labeled_Dataset_with_allAug": os.path.join(
        DATA_ROOT, "Data/processed/Combined_Labeled_Dataset_with_allAug.csv"
    ),
    "Combined_Labeled_Dataset_with_fearAug_surprise": os.path.join(
        DATA_ROOT, "Data/processed/Combined_Labeled_Dataset_with_fearAug_surprise.csv"
    ),
    "Combined_Labeled_Dataset_with_fearAug_surprise_anger": os.path.join(
        DATA_ROOT, "Data/processed/Combined_Labeled_Dataset_with_fearAug_surprise_anger.csv"
    ),
    "ablation_outputs": os.path.join(STORAGE_ROOT, "outputs/ablation"),
    "confidence_outputs": os.path.join(STORAGE_ROOT, "outputs/confidence"),
    "Combined_Labeled_Dataset_scored": os.path.join(
        DATA_ROOT, "Data/processed/Combined_Labeled_Dataset_scored.csv"
    ),
    "augmented_data": os.path.join(DATA_ROOT, "Data/processed/fear_augmented.csv"),
    "Labeled_4K": os.path.join(DATA_ROOT, "Data/processed/Labeled_4K.csv"),
    "augmented_afghan_fear_9000": os.path.join(DATA_ROOT, "Data/processed/augmented_afghan_fear_9000.csv"),
    "datasets": os.path.join(DATA_ROOT, "datasets"),
    "outputs": os.path.join(STORAGE_ROOT, "outputs"),
}

MODEL_CONFIG = {
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_train_epochs": 5,
}


def confidence_filtered_path(threshold, output_dir=None):
    """Return path for a confidence-filtered dataset CSV."""
    directory = output_dir or os.path.join(DATA_ROOT, "Data/processed")
    threshold_label = str(threshold).replace(".", "")
    filename = f"Combined_Labeled_Dataset_conf{threshold_label}.csv"
    return os.path.join(directory, filename)
