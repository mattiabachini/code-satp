# Bootstrap Resampling for F1 Scores

This directory contains utilities for saving model predictions and performing bootstrap resampling to get confidence intervals for F1 scores.

## Overview

The scripts in this directory help you:

1. Save model predictions and true labels to CSV files
2. Perform bootstrap resampling to get confidence intervals for F1 scores
3. Run permutation tests to compare the performance of different models

## Files

- `save_predictions.py`: A script to extract predictions from a trained model and save them to CSV files
- `bootstrap_f1_scores.py`: A script to perform bootstrap resampling and permutation tests
- `save_predictions_example.py`: Example of how to use the scripts in your workflow

## Usage

### Step 1: Save Model Predictions

After training your model with the HuggingFace Trainer, add the following code to save the predictions:

```python
import save_predictions

# After training your model
target_names = data.drop(columns=["incident_summary"]).columns.tolist()

# Save predictions for bootstrap resampling
results_files = save_predictions.save_model_predictions(
    trainer,             # Your trained Trainer object
    test_dataset,        # Your test dataset
    target_names,        # List of target names (column names)
    model_name="distilbert-base-cased",  # Name of model
    data_fraction="100%"                 # Data fraction identifier
)
```

This will save three files:
- `bootstrap_data/distilbert-base-cased_100%_true_labels.csv`: True labels
- `bootstrap_data/distilbert-base-cased_100%_predictions.csv`: Binary predictions (after threshold)
- `bootstrap_data/distilbert-base-cased_100%_probabilities.csv`: Raw probability scores

### Step 2: Run Bootstrap Resampling

You can run the bootstrap resampling script from the command line:

```bash
python bootstrap_f1_scores.py \
    --true_labels bootstrap_data/distilbert-base-cased_100%_true_labels.csv \
    --predictions bootstrap_data/distilbert-base-cased_100%_predictions.csv \
    --output bootstrap_results.csv \
    --n_bootstrap 1000
```

Or you can import the functions directly in your notebook:

```python
from bootstrap_f1_scores import bootstrap_f1_scores, calculate_bootstrap_statistics
import pandas as pd
import numpy as np

# Load saved data
true_labels_df = pd.read_csv("bootstrap_data/distilbert-base-cased_100%_true_labels.csv")
pred_df = pd.read_csv("bootstrap_data/distilbert-base-cased_100%_predictions.csv")

# Convert to numpy arrays
true_labels = true_labels_df.to_numpy()
predictions = pred_df.to_numpy()

# Get target names
target_names = true_labels_df.columns.tolist()

# Run bootstrap resampling
bootstrap_df = bootstrap_f1_scores(true_labels, predictions, target_names, n_bootstrap=1000)

# Calculate statistics
statistics = calculate_bootstrap_statistics(bootstrap_df)

# Print results
print("\nBootstrap Results:")
print(f"Micro F1: {statistics['micro_f1_mean']:.4f} (95% CI: {statistics['micro_f1_ci_lower']:.4f}-{statistics['micro_f1_ci_upper']:.4f})")
print(f"Macro F1: {statistics['macro_f1_mean']:.4f} (95% CI: {statistics['macro_f1_ci_lower']:.4f}-{statistics['macro_f1_ci_upper']:.4f})")
```

### Step 3: Compare Models with Permutation Test

To compare the performance of two models:

```python
from bootstrap_f1_scores import permutation_test

# Load predictions from two models
true_labels_df = pd.read_csv("bootstrap_data/true_labels.csv")
pred_a_df = pd.read_csv("bootstrap_data/distilbert-base-cased_100%_predictions.csv")
pred_b_df = pd.read_csv("bootstrap_data/bert-base-cased_100%_predictions.csv")

# Convert to numpy arrays
true_labels = true_labels_df.to_numpy()
predictions_a = pred_a_df.to_numpy()
predictions_b = pred_b_df.to_numpy()

# Run permutation test
perm_results = permutation_test(true_labels, predictions_a, predictions_b, n_permutations=1000)

# Print results
print("\nPermutation Test Results:")
print(f"Model A F1: {perm_results['f1_model_a']:.4f}")
print(f"Model B F1: {perm_results['f1_model_b']:.4f}")
print(f"Absolute difference: {perm_results['observed_diff']:.4f}")
print(f"P-value: {perm_results['p_value']:.4f}")
print(f"Significant difference: {'Yes' if perm_results['significant'] else 'No'}")
```

## Integrating with Your Existing Workflow

If you want to save predictions for all models automatically, you can modify your `run_all_experiments_and_save` function as shown in `save_predictions_example.py`.

## Reporting Results

In your paper, you can report the F1 scores with confidence intervals, for example:

"The DistilBERT model achieved a micro-F1 score of 0.845 (95% CI: 0.835-0.855) and a macro-F1 score of 0.792 (95% CI: 0.781-0.803)."

For model comparisons, you can report the p-value from the permutation test:

"The difference in F1 scores between DistilBERT and BERT was statistically significant (p < 0.05 by permutation test)." 