"""
Save Model Predictions for Bootstrap Resampling

This script extracts model predictions and true labels from a trained model 
and saves them to CSV files for bootstrap resampling.

Usage in your notebook:
```python
# After training and evaluation
import save_predictions
save_predictions.save_model_predictions(
    trainer, 
    test_dataset, 
    target_names, 
    model_name="my_model",
    data_fraction="100%"
)
```
"""

import torch
import pandas as pd
import numpy as np
import os

def save_model_predictions(trainer, dataset, target_names, model_name="model", data_fraction="all"):
    """
    Extract and save model predictions and true labels from a trained model.
    
    Args:
        trainer: HuggingFace Trainer object with trained model
        dataset: Test dataset containing the data to generate predictions for
        target_names: List of target names (column names for the CSV files)
        model_name: Name of the model (used in output filenames)
        data_fraction: Data fraction identifier (used in output filenames)
    """
    print(f"Extracting predictions for {model_name} with {data_fraction} data...")
    
    # Create output directory
    os.makedirs("bootstrap_data", exist_ok=True)
    
    # Sanitize model name for filename
    safe_model_name = model_name.replace('/', '_')
    
    # Get predictions from the model
    trainer.args.per_device_eval_batch_size = 8  # Small batch size to avoid out of memory
    predictions = trainer.predict(dataset)
    
    # Extract logits and labels
    logits = predictions.predictions
    labels = predictions.label_ids
    
    # Apply sigmoid to get probabilities
    raw_probs = torch.sigmoid(torch.tensor(logits)).numpy()
    
    # Apply threshold to get binary predictions
    binary_preds = (raw_probs > 0.5).astype(int)
    
    # Convert to DataFrames
    true_df = pd.DataFrame(labels, columns=target_names)
    pred_df = pd.DataFrame(binary_preds, columns=target_names)
    prob_df = pd.DataFrame(raw_probs, columns=target_names)
    
    # Save to CSV files
    true_df.to_csv(f"bootstrap_data/{safe_model_name}_{data_fraction}_true_labels.csv", index=False)
    pred_df.to_csv(f"bootstrap_data/{safe_model_name}_{data_fraction}_predictions.csv", index=False)
    prob_df.to_csv(f"bootstrap_data/{safe_model_name}_{data_fraction}_probabilities.csv", index=False)
    
    print(f"Saved true labels, predictions, and probabilities to bootstrap_data/ directory")
    
    # Calculate some basic metrics for reference
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    micro_f1 = f1_score(labels, binary_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, binary_preds, average='macro', zero_division=0)
    
    print(f"Micro F1 score: {micro_f1:.4f}")
    print(f"Macro F1 score: {macro_f1:.4f}")
    
    return {
        'true_labels_file': f"bootstrap_data/{safe_model_name}_{data_fraction}_true_labels.csv",
        'predictions_file': f"bootstrap_data/{safe_model_name}_{data_fraction}_predictions.csv",
        'probabilities_file': f"bootstrap_data/{safe_model_name}_{data_fraction}_probabilities.csv",
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }

# Example usage in notebook:
"""
# After training and evaluation
import save_predictions
results_files = save_predictions.save_model_predictions(
    trainer, 
    test_dataset, 
    target_names, 
    model_name="distilbert-base-cased",
    data_fraction="100%"
)

# Then use bootstrap_f1_scores.py to get confidence intervals
# python bootstrap_f1_scores.py --true_labels results_files['true_labels_file'] --predictions results_files['predictions_file']
""" 