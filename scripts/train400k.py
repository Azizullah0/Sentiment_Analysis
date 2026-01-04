
import sys
import os
import logging
import warnings
import datetime
import json
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Import your custom configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.paths import PATHS, MODEL_CONFIG
# Force the correct path
PATHS["Combined_Labeled_Dataset"] = "/content/drive/MyDrive/Sentiment_Analysis/Data/processed/Combined_Labeled_Dataset.csv"
print(f"Overriding path to: {PATHS['Combined_Labeled_Dataset']}")
# Custom Trainer with class weights
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def tokenize_function(examples, tokenizer):
    try:
        # Validate input
        if 'text' not in examples or 'label_id' not in examples:
            missing = [k for k in ['text', 'label_id'] if k not in examples]
            raise KeyError(f"Missing columns: {missing}")
        
        # Tokenize
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=MODEL_CONFIG["max_length"],
            return_tensors=None
        )
        
        # Add labels
        tokenized["labels"] = examples["label_id"]
        return tokenized
        
    except Exception as e:
        print(f"Tokenization error: {e}")
        print(f"Sample data: {examples['text'][:2] if 'text' in examples else 'No text text'}")
        raise

def compute_metrics(p):
    """Compute comprehensive metrics for evaluation"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    f1_macro = f1_score(labels, predictions, average='macro')  # Added macro F1 for imbalance
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        "accuracy": accuracy, 
        "f1_weighted": f1,
        "f1_macro": f1_macro, 
        "precision": precision, 
        "recall": recall
    }

def validate_dataset(df):
    """Validate dataset quality and integrity"""
    print("Validating dataset...")
    
    # Check for required columns
    required_columns = ['text', 'label_id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for NaN values
    print(f"NaN in 'text': {df['text'].isna().sum()}")
    print(f"NaN in 'label_id': {df['label_id'].isna().sum()}")
    
    # Remove rows with NaN in critical columns
    initial_size = len(df)
    df = df.dropna(subset=['text', 'label_id'])
    if initial_size - len(df) > 0:
        print(f"Removed {initial_size - len(df)} rows with NaN values")
    
    # Validate label range (0-7 for your 8 emotions)
    print(f"Unique labels: {sorted(df['label_id'].unique())}")
    invalid_labels = df[~df['label_id'].between(0, 7)]
    if not invalid_labels.empty:
        print(f" Found {len(invalid_labels)} rows with invalid labels. Removing them.")
        df = df[df['label_id'].between(0, 7)]
    
    return df

def analyze_class_imbalance(df):
    """Analyze and report class imbalance"""
    print("\n ANALYZING CLASS IMBALANCE")
    print("=" * 40)
    
    label_counts = df['label_id'].value_counts().sort_index()
    total_samples = len(df)
    
    label_names = {
        0: "Hope",
        1: "Happy", 
        2: "Neutral",
        3: "Suprise",
        4: "Disgust",
        5: "Sad",
        6: "Anger",
        7: "Fear"
    }
    
    print("Label Distribution:")
    for label_id, count in label_counts.items():
        percentage = (count / total_samples) * 100
        label_name = label_names.get(label_id, f"Label_{label_id}")
        print(f"  {label_name} ({label_id}): {count:6d} samples ({percentage:5.2f}%)")
    
    # Calculate imbalance ratios
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\nImbalance Analysis:")
    print(f"  Majority class: {max_count} samples")
    print(f"  Minority class: {min_count} samples") 
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    return label_counts

def main():
    """Main training function for incremental fine-tuning"""
    try:
        print(" Starting Incremental Fine-tuning on Persian/Dari Dataset")
        print(" Continuing from previously fine-tuned model (4K dataset)")
        print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
      
        # 1. Load and Validate Dataset
      
        print("\n Loading new dataset for incremental training...")
        file_path = PATHS["Combined_Labeled_Dataset"]
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        df = pd.read_csv(file_path)
        print(f" New dataset loaded: {len(df)} rows")
        
        # Validate dataset
        df = validate_dataset(df)
        
        # Analyze class imbalance
        label_counts = analyze_class_imbalance(df)
        print(f" Final training dataset size: {len(df)} rows")
     
        # 2. Train-Test Split (Stratified)
     
        print("\n Splitting dataset with stratification...")
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df['label_id'], 
            random_state=42
        )
        
        print(f"Train samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        
       
        # 3. Load Previously Fine-tuned Model & Tokenizer
      
        print("\n Loading previously fine-tuned model and tokenizer...")
        
        # Load from your previously fine-tuned model directory
        previous_model_path = PATHS["parsbert_emotion"]
        tokenizer = AutoTokenizer.from_pretrained(previous_model_path)
        
        print(f" Loading model from: {previous_model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            previous_model_path,
            num_labels=8  # Same number of labels as before
        )
        
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Using device: {device}")
        
        
        # 4. Tokenization
     
        print("\n Tokenizing dataset...")
        
        # Convert to Hugging Face datasets
        train_dataset = Dataset.from_pandas(train_df[['text', 'label_id']])
        test_dataset = Dataset.from_pandas(test_df[['text', 'label_id']])
        
        # Tokenize
        tokenized_train = train_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer), 
            batched=True
        )
        tokenized_test = test_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer), 
            batched=True
        )
        
        # Remove original columns and set format
        columns_to_remove = ["text", "label_id"]
        tokenized_train = tokenized_train.remove_columns(columns_to_remove)
        tokenized_test = tokenized_test.remove_columns(columns_to_remove)
        
        tokenized_train.set_format("torch")
        tokenized_test.set_format("torch")
        
      
        # 5. Class Weight Calculation for Imbalance
      
        print("\n Calculating class weights for imbalance handling...")
        total = sum(label_counts.values)
        weights = [total / count for count in label_counts.values]
        weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        
        print("Class weights for imbalance:")
        label_names = {0: "Hope", 1: "Happy", 2: "Neutral", 3: "Suprise", 
                      4: "Disgust", 5: "Sad", 6: "Anger", 7: "Fear"}
        for i, (label_id, weight) in enumerate(zip(label_counts.index, weights)):
            label_name = label_names.get(label_id, f"Label_{label_id}")
            print(f"  {label_name}: {weight:.2f}x")
        
        
        # 6. Training Setup - Adjusted for Imbalance
      
        print("\n Configuring training for imbalanced dataset...")
        
        # Create output directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = os.path.join(PATHS["outputs"], f"incremental_imbalanced_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments optimized for imbalanced data
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-5,  # Lower LR for stable fine-tuning
            per_device_train_batch_size=MODEL_CONFIG["batch_size"],
            per_device_eval_batch_size=MODEL_CONFIG["batch_size"],
            num_train_epochs=4,  # Slightly more epochs for imbalance
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",  # Use macro F1 for imbalance
            greater_is_better=True,
            report_to="none",
            fp16=True,
            logging_steps=50,
            save_total_limit=2,
            push_to_hub=False,
            warmup_steps=100,
            dataloader_drop_last=True,  # Help with batch normalization
        )
        
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
     
        # 7. Initialize Trainer with Class Weights
       
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            class_weights=weights_tensor,  # Apply class weights for imbalance
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
   
        # 8. Train Model
        
        print("\n Starting training with class weights...")
        print(" Using class weights to handle imbalance")
        print(" Monitoring macro F1 score for better imbalance evaluation")
        train_result = trainer.train()
        
       
        # 9. Comprehensive Evaluation
       
        print("\n Comprehensive evaluation...")
        evaluation_results = trainer.evaluate()
        
        # Per-class evaluation
        predictions = trainer.predict(tokenized_test)
        preds = np.argmax(predictions.predictions, axis=1)
        labels = predictions.label_ids
        
        print("\n" + "="*60)
        print("INCREMENTAL TRAINING RESULTS - IMBALANCED DATASET")
        print("="*60)
        print(f"Accuracy:        {evaluation_results['eval_accuracy']:.4f}")
        print(f"F1 Weighted:     {evaluation_results['eval_f1_weighted']:.4f}")
        print(f"F1 Macro:        {evaluation_results['eval_f1_macro']:.4f}")
        print(f"Precision:       {evaluation_results['eval_precision']:.4f}")
        print(f"Recall:          {evaluation_results['eval_recall']:.4f}")
        
        # Per-class metrics
        print(f"\n Per-class F1 Scores:")
        per_class_f1 = f1_score(labels, preds, average=None)
        label_names = {0: "Hope", 1: "Happy", 2: "Neutral", 3: "Suprise", 
                      4: "Disgust", 5: "Sad", 6: "Anger", 7: "Fear"}
        for i, f1_cls in enumerate(per_class_f1):
            label_name = label_names.get(i, f"Label_{i}")
            print(f"  {label_name}: {f1_cls:.4f}")
        
      
        # 10. Save Model and Metadata
        
        incremental_model_path = PATHS["incremental_finetuned_model"]
        print(f"\n Saving model to {incremental_model_path}...")
        trainer.save_model(incremental_model_path)
        tokenizer.save_pretrained(incremental_model_path)
        
        # Save comprehensive training metadata
        metadata = {
            "training_type": "incremental_fine_tuning_imbalanced",
            "base_model": previous_model_path,
            "training_date": datetime.datetime.now().isoformat(),
            "dataset_size": len(df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "label_distribution": label_counts.to_dict(),
            "class_weights": {i: float(w) for i, w in enumerate(weights)},
            "final_metrics": evaluation_results,
            "per_class_f1": {label_names[i]: float(score) for i, score in enumerate(per_class_f1)},
            "training_args": {k: str(v) for k, v in training_args.to_dict().items()},
            "imbalance_notes": {
                "majority_class": "Happy",
                "minority_class": "Fear", 
                "imbalance_ratio": f"{max(label_counts.values)/min(label_counts.values):.1f}:1",
                "fear_samples": label_counts[7]
            }
        }
        
        with open(os.path.join(incremental_model_path, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(" Training completed successfully!")
        print(f" Model saved: {incremental_model_path}")
        print(f" Key metric - F1 Macro: {evaluation_results['eval_f1_macro']:.4f}")
        print(f"  Note: Severe class imbalance detected (Fear: {label_counts[7]} samples)")
        
    except Exception as e:
        print(f" Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
