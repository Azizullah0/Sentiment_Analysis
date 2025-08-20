import sys
import os
import logging
import warnings
from transformers import AutoTokenizer
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Remove sys.path append related to utils to avoid potential issues
# if 'Sentiment_Analysis/utils' in sys.path:
#     sys.path.remove('Sentiment_Analysis/utils')


import torch
import torch.nn as nn
import numpy as np
import pandas as pd # Corrected syntax here
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split # Import train_test_split

# Remove imports from utils.dataset_utils
# from utils.dataset_utils import load_dataset, split_dataset, tokenize_datasets


from datasets import Dataset # Import Dataset from the datasets library

# Define tokenize_function directly in the notebook (if not already defined similarly)
def tokenize_function(examples, tokenizer):
    # Ensure 'clean' and 'label_id' are in examples dictionary
    if 'clean' not in examples or 'label_id' not in examples:
        print("Error: 'clean' or 'label_id' not found in examples dictionary for tokenization.")
        print("Available keys:", examples.keys())
        raise KeyError("Missing required columns in examples dictionary for tokenization.")

    # Assuming the text is in the 'clean' column and labels are in 'label_id'
    # Corrected: Explicitly set max_length for padding and truncation
    tokenized_examples = tokenizer(examples["clean"], truncation=True, padding="max_length", max_length=512)
    # Rename 'label_id' to 'labels' as expected by the Trainer
    tokenized_examples["labels"] = examples["label_id"]
    return tokenized_examples


# ================================
# 1. Load Dataset (+ optional Fear augmentation)
# ================================
# Load the 400K labeled dataset using pandas
file_path = '/content/drive/MyDrive/ColabFoulder/Labeled_400K.csv'
try:
    df = pd.read_csv(file_path)
    print(f"✅ Successfully read {file_path}")

    # Ensure 'label_id' column exists and is correctly mapped if necessary
    if 'label_id' not in df.columns:
         label_map = {
            0: "Hope",
            1: "Happy",
            2: "Neutral",
            3: "Surprise",
            4: "Disgust",
            5: "Sad",
            6: "Anger",
            7: "Fear"
        }
         if 'predicted_label' in df.columns:
             df["label_id"] = df["predicted_label"].str.replace("LABEL_", "").astype(int)
             df["emotion"] = df["label_id"].map(label_map)
         else:
             print("❌ Error: 'predicted_label' column not found. Cannot create 'label_id'.")
             # Handle this error appropriately, maybe exit or raise exception
             sys.exit("Required column 'predicted_label' not found.")


    # If augmented Fear file exists, merge it
    aug_path = '/content/drive/MyDrive/ColabFoulder/fear_augmented.csv' # Assuming augmented file is in the same ColabFoulder
    if os.path.exists(aug_path):
        try:
            aug_df = pd.read_csv(aug_path)
            if not aug_df.empty: # Check if the DataFrame is not empty
                 # Ensure augmented data has the same columns and label_id structure
                 if 'label_id' not in aug_df.columns and 'predicted_label' in aug_df.columns:
                      label_map = {
                         0: "Hope",
                         1: "Happy",
                         2: "Neutral",
                         3: "Surprise",
                         4: "Disgust",
                         5: "Sad",
                         6: "Anger",
                         7: "Fear"
                     }
                      aug_df["label_id"] = aug_df["predicted_label"].str.replace("LABEL_", "").astype(int)

                 df = pd.concat([df, aug_df], ignore_index=True)
                 print(f"✅ Augmented Fear samples added: {len(aug_df)} rows")
            else:
                print(f"ℹ️ {aug_path} exists but is empty. Skipping augmentation.")
        except pd.errors.EmptyDataError:
             print(f"⚠️ {aug_path} exists but is empty or improperly formatted. Skipping augmentation.")
        except Exception as e:
            print(f"❌ An error occurred while reading {aug_path}: {e}. Skipping augmentation.")

    print(f"Total dataset size after augmentation: {len(df)} rows")
    print("Label distribution before splitting:")
    print(df['label_id'].value_counts().sort_index())

    # Split the dataset using train_test_split
    # Ensure 'label_id' column is used for stratification
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_id'], random_state=42)


    print(f"Train dataset size: {len(train_df)} rows")
    print(f"Test dataset size: {len(test_df)} rows")


    # ================================
    # 2. Tokenization
    # ================================
    # Load the tokenizer from the fine-tuned model directory
    tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/parsbert_emotion")

    # Convert pandas DataFrames to Hugging Face Dataset objects
    # Select only the necessary columns before conversion
    train_dataset = Dataset.from_pandas(train_df[['clean', 'label_id']])
    test_dataset = Dataset.from_pandas(test_df[['clean', 'label_id']])


    # Apply the tokenization function to the datasets
    # Use a lambda function to pass the tokenizer to tokenize_function
    tokenized_train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    # Remove the original 'clean' and 'label_id' columns after they are used
    # Keep 'labels' for training
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["clean", "label_id"])
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(["clean", "label_id"])

    # Set format for PyTorch
    tokenized_train_dataset.set_format("torch")
    tokenized_test_dataset.set_format("torch")


    # ================================
    # 3. Model and Data Collator
    # ================================
    num_labels = df['label_id'].nunique() # Use nunique for number of unique labels
    model = AutoModelForSequenceClassification.from_pretrained(
        "/content/drive/MyDrive/parsbert_emotion",
        num_labels=num_labels
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Set device for the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")


    # ================================
    # 4. Compute Class Weights
    # ================================
    label_counts = df['label_id'].value_counts().sort_index().values
    total = sum(label_counts)
    # Calculate inverse frequency weights
    weights = [total / count for count in label_counts]
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    # Assign weights to the model's loss function if it supports it (e.g., CrossEntropyLoss)
    # AutoModelForSequenceClassification uses CrossEntropyLoss by default, which accepts weights.
    # We can pass these weights to the model's forward pass or modify the Trainer.
    # A common way is to pass them during model initialization if the model architecture supports it directly,
    # or handle it within a custom Trainer or loss function.
    # For simplicity with Trainer, we can often rely on the default loss and observe weighted metrics,
    # or if needed, customize the Trainer's compute_loss method.
    # Let's print the weights for now and consider how to apply them if necessary for Trainer.
    print("Computed class weights:", weights)

    # Note: Applying class weights directly to the model's loss function via Trainer
    # requires either a custom Trainer or ensuring the model's forward method
    # accepts a `weight` parameter for its loss calculation, which is standard for CrossEntropyLoss.
    # The default Trainer with AutoModelForSequenceClassification usually handles this correctly
    # if weights are passed to the model's loss function call within its forward method.
    # No direct argument in TrainingArguments for class weights.

    # ================================
    # 5. Define Metrics
    # ================================
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


    # ================================
    # 6. Training Arguments
    # ================================
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",       # Save checkpoint at the end of each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3, # Adjust number of epochs as needed
        weight_decay=0.01,
        load_best_model_at_end=True, # Load the best model based on evaluation metric
        metric_for_best_model="f1",  # Metric to monitor for early stopping and best model
        greater_is_better=True,      # Higher f1 is better
        report_to="none", # Set to "none" or configure logging to "tensorboard", "wandb", etc.
        fp16=True # Enable mixed precision training
    )


    # ================================
    # 7. Trainer
    # ================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset, # Use the tokenized dataset
        eval_dataset=tokenized_test_dataset,   # Use the tokenized dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # Add EarlyStoppingCallback
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Stop if metric doesn't improve for 3 evaluations
    )

    # ================================
    # 8. Train Model
    # ================================
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # ================================
    # 9. Evaluate and Save
    # ================================
    print("Evaluating model...")
    evaluation_results = trainer.evaluate()
    print("Evaluation results:", evaluation_results)

    # Save the fine-tuned model
    output_model_dir = "/content/drive/MyDrive/ColabFoulder/fine_tuned_parsbert_400k"
    trainer.save_model(output_model_dir)
    print(f"Fine-tuned model saved to {output_model_dir}")

    # You can also save the tokenizer if needed, although it's loaded from the same directory
    # tokenizer.save_pretrained(output_model_dir)

except FileNotFoundError:
    print(f"❌ Error: File not found at {file_path}")
except Exception as e:
    print(f"❌ An error occurred: {e}")