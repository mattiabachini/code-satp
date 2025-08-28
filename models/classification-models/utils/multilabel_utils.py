import pandas as pd
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import hamming_loss, accuracy_score, classification_report, f1_score
from skmultilearn.model_selection import IterativeStratification
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

## Fixed split function

def create_fixed_splits(
        df_full, 
        stratify_cols, 
        test_size=0.1, 
        val_size=0.111, 
        random_state=42
):
    """
    Splits a multi-label classification dataset into fixed training, validation, and test sets
    using iterative stratification to preserve label distributions across splits.

    This function is designed for experiments where:
      - The test set must remain fixed across all model runs.
      - The validation set is also fixed and used for model selection.
      - Only the training set is progressively sampled in size.

    Parameters:
        df_full (pd.DataFrame): Full labeled dataset, including the input text column and multi-label targets.
        stratify_cols (list of str): Column names representing the multi-label targets used for stratification.
        test_size (float): Fraction of the full dataset to reserve as the test set (default: 0.1).
        val_size (float): Fraction of the remaining training pool to use as the validation set (default: 0.111).
        random_state (int): Seed for reproducibility. Default is 42.

    Returns:
        df_train_pool (pd.DataFrame): Training pool to draw progressively larger subsets from.
        df_val (pd.DataFrame): Fixed validation set (real-only, for model tuning).
        df_test (pd.DataFrame): Fixed test set (real-only, for final evaluation).

    Notes:
        - All DataFrames returned preserve the original index from df_full.
        - Iterative stratification ensures that rare labels are proportionally distributed across splits.
    """
    # Restrict to only the text and specified label columns to avoid extra metadata columns
    keep_cols = ["incident_summary"] + stratify_cols
    df_full = df_full[keep_cols].copy()

    # Step 1: Split full data into train+val vs. test using iterative stratification

    # Pre-shuffle once to avoid chronological bias while keeping determinism
    df_full_shuf = df_full.sample(frac=1, random_state=random_state)

    # Extract text and multi-label target arrays
    X = df_full_shuf["incident_summary"].values
    y = df_full_shuf[stratify_cols].values

    # Create iterative stratification object
    stratifier = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[1 - test_size, test_size]
    )

    # Unpack: stratifier.split() returns (smaller_set, larger_set) 
    # We want test_idx (smaller) and trainval_idx (larger)
    for test_idx, trainval_idx in stratifier.split(X, y):
        break

    # Store the train+val and test DataFrames
    # Use .iloc to preserve original DataFrame indices
    df_trainval = df_full_shuf.iloc[trainval_idx]
    df_test = df_full_shuf.iloc[test_idx]

    # Step 2: Split trainval into training pool vs. validation set

    # Extract text and multi-label target arrays
    X_trainval = df_trainval["incident_summary"].values
    y_trainval = df_trainval[stratify_cols].values

    # Create iterative stratification object
    stratifier2 = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[1 - val_size, val_size]
    )

    # Unpack: stratifier2.split() returns (smaller_set, larger_set) 
    # We want val_idx (smaller) and train_idx (larger)
    for val_idx, train_idx in stratifier2.split(X_trainval, y_trainval):
        break

    # Store the test, val, and train pool DataFrames
    # Use .iloc to preserve original DataFrame indices
    df_test = df_full_shuf.iloc[test_idx]
    df_val = df_trainval.iloc[val_idx]
    df_train_pool = df_trainval.iloc[train_idx]

    # Return the DataFrames
    return df_train_pool, df_val, df_test

## Dataset class

class MultiLabelDataset(Dataset):
    """
    PyTorch Dataset for multi-label text classification.

    Args:
        texts (List[str]): List of input text samples.
        labels (np.ndarray or List[List[int]]): Multi-label targets for each sample.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to encode the texts.
        max_len (int): Maximum sequence length for tokenization.

    Returns:
        dict: Dictionary containing input_ids, attention_mask, and labels for each sample.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }

## Metrics function

def compute_metrics(eval_pred, target_names, exclusive_label=None, context_label=None):
    """
    Compute evaluation metrics for multi-label classification.
    Includes Hamming Loss, Subset Accuracy, and Classification Report for all labels.
    """
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy()  # Apply threshold
    # Optionally enforce exclusivity of a specific label (e.g., "no_target")
    if exclusive_label is not None and exclusive_label in target_names:
        exclusive_idx = target_names.index(exclusive_label)
        other_indices = [i for i, name in enumerate(target_names) if i != exclusive_idx]
        has_other = (predictions[:, other_indices].sum(axis=1) > 0)
        predictions[has_other, exclusive_idx] = 0
    labels = labels.astype(int)

    # Verify Labels
    print("Shape of labels:", labels.shape)  # Ensures correct dimensions
    print("First few rows of labels:\n", labels[:5])  # Shows the first few rows to check for issues
    print("Final target names:", target_names)

    # Hamming Loss
    hamming = hamming_loss(labels, predictions)

    # Subset Accuracy
    subset_acc = accuracy_score(labels, predictions)

    # Explicit micro-F1 for selection and reporting (maximize)
    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)

    # Classification Report
    report = classification_report(
        labels, predictions,
        target_names=target_names,
        zero_division=0, output_dict=True
    )

    # Print complete report for reference with context
    if context_label:
        print(f"\n=== Classification Report Context: {context_label} ===")
    else:
        print("\n=== Classification Report Context: (unspecified) ===")
    print("Full Classification Report:")
    print(classification_report(labels, predictions, target_names=target_names, zero_division=0))


    # Summary Metrics for Trainer
    metrics = {
        "hamming_loss": hamming,
        "subset_accuracy": subset_acc,
        "micro_f1": micro_f1,
    }
    metrics.update(report)
    return metrics

## Model training function (with fixed splits)

def train_transformer_model(
        model_name, 
        df_train, 
        df_val, 
        df_test, 
        max_len=512, 
        batch_size=16, 
        epochs=2,
        seed=42,
        exclusive_label=None
):
    """
    Trains a transformer model for multi-label classification using fixed train/val/test splits.

    This version assumes the data has already been stratified and split externally,
    and that original DataFrame indices have been preserved for traceability.

    Parameters:
        model_name (str): Name or path of the pre-trained HuggingFace model to fine-tune.
        df_train (pd.DataFrame): Training set with 'incident_summary' and label columns.
        df_val (pd.DataFrame): Validation set for model selection and early stopping.
        df_test (pd.DataFrame): Test set used for final evaluation.
        max_len (int): Maximum input token length for the tokenizer (default: 512).
        batch_size (int): Batch size for both training and evaluation (default: 16).
        epochs (int): Number of fine-tuning epochs (default: 2).

    Returns:
        trainer (transformers.Trainer): The HuggingFace Trainer object after training.
        test_results (dict): Evaluation metrics on the test set.
        pred_df (pd.DataFrame): DataFrame with true labels, predictions, probabilities,
                                incident summary text, and original indices.
    """
    # Set seeds for reproducibility (Python, NumPy, PyTorch, CUDA)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=df_train.shape[1] - 1,  # All columns except 'incident_summary'
        problem_type="multi_label_classification",
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Identify label columns
    target_names = [col for col in df_train.columns if col != "incident_summary"]

    # Prepare datasets
    train_dataset = MultiLabelDataset(df_train["incident_summary"].tolist(), df_train[target_names].values, tokenizer, max_len)
    val_dataset = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[target_names].values, tokenizer, max_len)
    test_dataset = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[target_names].values, tokenizer, max_len)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='eval_micro_f1',
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    # Set up Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda eval_pred: compute_metrics(
            eval_pred,
            target_names,
            exclusive_label=exclusive_label,
            context_label="Validation (tuning)"
        )
    )

    # Train the model
    trainer.train()

    # Final evaluation on df_test: retag compute_metrics with test context and predict() once
    trainer.compute_metrics = lambda eval_pred: compute_metrics(
        eval_pred,
        target_names,
        exclusive_label=exclusive_label,
        context_label="Final test evaluation"
    )
    predictions_output = trainer.predict(test_dataset)
    test_results = predictions_output.metrics
    logits = predictions_output.predictions
    labels = predictions_output.label_ids

    # Convert to probabilities and binary predictions
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    binary_preds = (probs > 0.5).astype(int)
    # Enforce exclusivity at test-time predictions as well if configured
    if exclusive_label is not None and exclusive_label in target_names:
        exclusive_idx = target_names.index(exclusive_label)
        other_indices = [i for i, name in enumerate(target_names) if i != exclusive_idx]
        has_other = (binary_preds[:, other_indices].sum(axis=1) > 0)
        binary_preds[has_other, exclusive_idx] = 0

    # Build predictions DataFrame
    pred_df = pd.DataFrame()
    for i, col in enumerate(target_names):
        pred_df[f"true_{col}"] = labels[:, i]
        pred_df[f"pred_{col}"] = binary_preds[:, i]
        pred_df[f"prob_{col}"] = probs[:, i]

    pred_df["incident_summary"] = df_test["incident_summary"].values
    pred_df["original_idx"] = df_test.index

    return trainer, test_results, pred_df

## Experiments function

def run_model_experiments(
    df_train_pool,
    df_val,
    df_test,
    model_names,
    stratify_cols,
    output_csv="test_summary.csv",
    predictions_csv="test_predictions.csv",
    model_labels=None,
    max_len=512,
    batch_size=16,
    epochs=2,
    fractions=[1/32, 1/16, 1/8, 1/4, 1/2, 1.0],
    exclusive_label=None
):
    """
    Runs post-tuning final experiments on the held-out test set by training models on
    progressively larger subsets of the training pool using stratified sampling.

    This function maintains label distribution consistency across different training subset
    sizes by using iterative stratification. This ensures that:
    - Rare labels appear proportionally in all training subsets
    - Model performance comparisons across data fractions are more meaningful
    - Training subsets have balanced label distributions similar to the full dataset

    Parameters:
        df_train_pool (pd.DataFrame): Fixed training pool (from `create_fixed_splits()`).
        df_val (pd.DataFrame): Fixed validation set (for model selection during training).
        df_test (pd.DataFrame): Final held-out test set used for true evaluation.
        model_names (list of str): HuggingFace model identifiers to evaluate.
        stratify_cols (list of str): Target column names for multi-label classification.
                                   Used for stratified sampling to preserve label distributions.
        output_csv (str): Output path for saving the summary of test results.
        predictions_csv (str): Output path for saving all test predictions.
        model_labels (dict, optional): Mapping from model name to human-readable label. If None, model names are used as labels.
        max_len (int): Max sequence length for tokenization.
        batch_size (int): Training/evaluation batch size.
        epochs (int): Number of training epochs.
        fractions (list of float): Fractions of training pool to use in experiments.
        random_state (int): Seed for reproducible stratified sampling.

    Returns:
        results_df (pd.DataFrame): Test metrics for all model/fraction combinations.
        full_pred_df (pd.DataFrame): Predictions with true labels and probabilities.

    Notes:
        - For 100% data fraction, the full training pool is used directly
        - For smaller fractions, iterative stratification preserves multi-label distributions
        - Original DataFrame indices are preserved for traceability across all subsets
    """
    if model_labels is None:
        model_labels = {name: name for name in model_names}

    results_list = []
    all_predictions = []

    for frac in fractions:
        subset_size = int(len(df_train_pool) * frac)
        
        # Use stratified sampling to preserve label distribution
        if frac == 1.0:
            # Use the full training pool when fraction is 100%
            df_train_subset = df_train_pool
        else:
            # Extract text and multi-label target arrays for stratified sampling
            X_pool = df_train_pool["incident_summary"].values
            y_pool = df_train_pool[stratify_cols].values
            
            # Create iterative stratification object for subset sampling
            stratifier = IterativeStratification(
                n_splits=2,
                order=1,
                sample_distribution_per_fold=[frac, 1-frac]
            )
            
            # Get stratified subset indices (split yields (train_idx, test_idx));
            # take the test fold of size `frac` as our subset
            _, subset_idx = next(stratifier.split(X_pool, y_pool))
            
            # Create stratified subset using iloc to preserve original indices
            df_train_subset = df_train_pool.iloc[subset_idx]
        
        frac_label = f"{frac*100:.1f}%"

        for model_name in model_names:
            model_label = model_labels.get(model_name, model_name)
            print(f"\n=== FINAL TEST EVAL | MODEL: {model_label} | FRACTION: {frac_label} ===")

            # `trainer` not used, but required for unpacking the results
            trainer, test_results, pred_df = train_transformer_model(
                model_name,
                df_train=df_train_subset,
                df_val=df_val,
                df_test=df_test, # evaluate on holdout test set
                max_len=max_len,
                batch_size=batch_size,
                epochs=epochs,
                exclusive_label=exclusive_label
            ) 

            run_result = {
                "fraction_raw": frac,
                "fraction_label": frac_label,
                "subset_size": subset_size,
                "model_raw": model_name,
                "model_label": model_label
            }
            for key, value in test_results.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        run_result[f"{key}_{subkey}"] = subvalue
                else:
                    run_result[key] = value

            results_list.append(run_result)

            pred_df["model"] = model_name
            pred_df["model_label"] = model_label
            pred_df["fraction"] = frac
            pred_df["fraction_label"] = frac_label
            all_predictions.append(pred_df)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_csv, index=False)
    full_pred_df = pd.concat(all_predictions, ignore_index=True)
    full_pred_df.to_csv(predictions_csv, index=False)

    print(f"Test results saved to {output_csv}")
    print(f"Test predictions saved to {predictions_csv}")
    return results_df, full_pred_df




