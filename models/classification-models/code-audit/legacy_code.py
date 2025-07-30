########################################################
# Target Type
########################################################

# #### Previous method for splitting data into train/val/test
# ####Create new split with each iteration

# import numpy as np
# import pandas as pd
# from skmultilearn.model_selection import IterativeStratification

# def iterative_train_test_split_custom(X, y, test_size, random_state=None):
#     """
#     Splits X and y using iterative stratification and returns data + indices.
#     """
#     stratifier = IterativeStratification(
#         n_splits=2,
#         order=1,
#         sample_distribution_per_fold=[1 - test_size, test_size]
#     )

#     indices = np.arange(len(X))
#     for train_idx, test_idx in stratifier.split(X, y):
#         return (
#             X[test_idx], y[test_idx], test_idx,
#             X[train_idx], y[train_idx], train_idx
#         )


# def stratified_split(df, x_col, stratify_cols, test_size, val_size=None, random_state=None, return_indices=False):
#     """
#     Splits a DataFrame into stratified sets using iterative stratification.

#     Parameters:
#       df           : pandas DataFrame.
#       x_col        : column name containing the features (e.g., "incident_summary").
#       stratify_cols: list of column names to use for stratification.
#       train_size   : fraction for training (used for information; priority is given to training and validation).
#       test_size    : fraction for test set.
#       val_size     : (optional) fraction for validation set. If None, a two-split (train/test) is performed.
#       random_state : seed for reproducibility.

#     Returns:
#       If val_size is None:
#          (X_train, y_train, X_test, y_test)
#       Else:
#          (X_train, y_train, X_val, y_val, X_test, y_test)

#     Note: When using a validation set, it is assumed that train_size + test_size + val_size == 1.
#     """
#     # Shuffle the DataFrame if a random state is provided.
#     if random_state is not None:
#         df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

#     # Extract features and stratification labels.
#     X = df[x_col].values
#     y = df[stratify_cols].values

#     if val_size is None:
#         X_test, y_test, test_idx, X_train, y_train, train_idx = iterative_train_test_split_custom(X, y, test_size, random_state)

#         if return_indices:
#             return X_train, y_train, X_test, y_test, train_idx, test_idx
#         else:
#             return X_train, y_train, X_test, y_test

#     else:
#         temp_size = test_size + val_size
#         X_temp, y_temp, temp_idx, X_train, y_train, train_idx = iterative_train_test_split_custom(X, y, temp_size, random_state)

#         ratio = test_size / temp_size
#         X_test, y_test, test_idx, X_val, y_val, val_idx = iterative_train_test_split_custom(X_temp, y_temp, ratio, random_state)

#         if return_indices:
#             return X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx
#         else:
#             return X_train, y_train, X_val, y_val, X_test, y_test

# #### Older training function


# # =======================
# # Reusable Training Function
# # =======================
# def train_transformer_model(model_name, data, max_len=512, test_size=0.1, val_size=0.1, batch_size=40, epochs=3):
#     """
#     Generalized function to train a transformer model for multi-label classification.
#     Args:
#         model_name: Name of the pre-trained model (e.g., "bert-base-uncased", "distilbert-base-uncased").
#         data: Pandas DataFrame with columns "incident_summary" and multi-label columns.
#         max_len: Maximum sequence length.
#         batch_size: Batch size for training and evaluation.
#         epochs: Number of training epochs.
#     """
#     # Load tokenizer and model
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_name,
#         num_labels=data.shape[1] - 1,  # Number of labels (all columns except "incident_summary")
#         problem_type="multi_label_classification",
#     )
#     model.to("cuda" if torch.cuda.is_available() else "cpu")

#     target_names = data.drop(columns=["incident_summary"]).columns.tolist()

#     # Split data into train, val, and test
#     X = data["incident_summary"]
#     y = data.drop('incident_summary', axis=1).values

#     # Keep original indices
#     original_indices = X.index

#     # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42) #stratify=y)
#     # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)




#     # X_train, y_train, X_test, y_test = stratified_split(
#     #     data,
#     #     x_col="incident_summary",
#     #     stratify_cols=[col for col in data.columns if col != "incident_summary"],
#     #     train_size=0.9,
#     #     test_size=0.1,
#     #     val_size=None,
#     #     random_state=42
#     # )

#     X_train, y_train, X_val, y_val, X_test, y_test, _, _, test_idx = stratified_split(
#         data,
#         x_col="incident_summary",
#         stratify_cols=[col for col in data.columns if col != "incident_summary"],
#         test_size=test_size,
#         val_size=val_size,
#         random_state=42,
#         return_indices=True
#     )


#     # Create datasets
#     train_dataset = MultiLabelDataset(X_train.tolist(), y_train, tokenizer, max_len)
#     val_dataset = MultiLabelDataset(X_val.tolist(), y_val, tokenizer, max_len)
#     test_dataset = MultiLabelDataset(X_test.tolist(), y_test, tokenizer, max_len)

#     # Define training arguments
#     training_args = TrainingArguments(
#         output_dir="./results",
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         num_train_epochs=epochs,
#         weight_decay=0.01,
#         logging_dir="./logs",
#         logging_steps=10,
#         load_best_model_at_end=True,
#         metric_for_best_model='eval_hamming_loss',
#         greater_is_better=True,
#         save_total_limit=2,
#         report_to="none",
#     )

#     # Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         processing_class=tokenizer,
#         compute_metrics=lambda eval_pred: compute_metrics(eval_pred, target_names)
#     )

#     # Train and Evaluate
#     trainer.train()

#     # Final Evaluation on Test Set
#     test_results = trainer.evaluate(test_dataset)
#     print("Test Set Results:", test_results)

#     # ---------------------------
#     # Generate Predictions on Test Set
#     # ---------------------------
#     predictions_output = trainer.predict(test_dataset)
#     logits = predictions_output.predictions
#     labels = predictions_output.label_ids

#     # Convert logits to probabilities
#     probs = torch.sigmoid(torch.tensor(logits)).numpy()

#     # Apply threshold to get binary predictions
#     binary_preds = (probs > 0.5).astype(int)

#     # Build predictions DataFrame
#     predictions_df = pd.DataFrame()
#     for i, col in enumerate(target_names):
#         predictions_df[f"true_{col}"] = labels[:, i]
#         predictions_df[f"pred_{col}"] = binary_preds[:, i]
#         predictions_df[f"prob_{col}"] = probs[:, i]

#     # Add original row index from the full dataset for traceability
#     predictions_df["original_idx"] = data.index[test_idx]

#     # Add original text summary for manual inspection of predictions
#     predictions_df["incident_summary"] = data.iloc[test_idx]["incident_summary"].values

#     return trainer, test_results, predictions_df

# #### Older experiments function

# def run_all_experiments_and_save(df_full, output_csv="results_summary.csv", predictions_csv="all_predictions.csv"):
#     """
#     1. Iterates over the defined fractions & model list
#     2. Samples df_full according to fraction
#     3. Trains & evaluates using train_multiclass_model_3way_split
#     4. Saves the collected results in a DataFrame
#     5. Exports to CSV

#     Args:
#         df_full (pd.DataFrame): Full dataset with columns [label_col, text_col].
#         output_csv (str): File path to save the experiment results.
#     Returns:
#         results_df (pd.DataFrame): Contains experiment results for analysis.
#     """
#     results_list = []

#     all_predictions = []

#     for frac in fractions:
#         # Sample a fraction of the data
#         subset_size = int(len(df_full) * frac)
#         df_subset = df_full.sample(n=subset_size, random_state=42)

#         # Friendly fraction label if you want
#         frac_label = fraction_labels.get(frac, f"{frac*100:.1f}%")
#         print(f"\n=== DATA FRACTION: {frac} ({subset_size} rows) ===")

#         for model_name in models_list:
#             # Model label
#             model_label = model_name_labels.get(model_name, model_name)
#             print(f"Training model: {model_label}")

#             # Train & evaluate
#             # write the model funtion here
#             trainer, test_results, test_predictions = train_transformer_model(model_name, df_subset, max_len=512, test_size=0.1, val_size=0.1, batch_size=16, epochs=2)


#             # Build a result dict
#             run_result = {
#                 "fraction_raw": frac,
#                 "fraction_label": frac_label,
#                 "subset_size": subset_size,
#                 "model_raw": model_name,
#                 "model_label": model_label
#             }

#             # Flatten the nested dictionary
#             for key, value in test_results.items():
#                 if isinstance(value, dict):
#                     for subkey, subvalue in value.items():
#                         # Create new key names like "armed_assault_precision"
#                         run_result[f"{key}_{subkey}"] = subvalue
#                 else:
#                     run_result[key] = value

#             # Append to results_list
#             results_list.append(run_result)

#             # Add metadata columns
#             test_predictions["model"] = model_name
#             test_predictions["model_label"] = model_label
#             test_predictions["fraction"] = frac
#             test_predictions["fraction_label"] = frac_label

#             # Store in full list
#             all_predictions.append(test_predictions)

#     # Convert to DataFrame
#     results_df = pd.DataFrame(results_list)
#     # Save to CSV
#     results_df.to_csv(output_csv, index=False)
#     print(f"\nResults saved to {output_csv}")

#     # Concatenate and save all predictions
#     full_pred_df = pd.concat(all_predictions, ignore_index=True)
#     full_pred_df.to_csv(predictions_csv, index=False)
#     print("All predictions saved to all_predictions.csv")

#     # also save to JSON
#     # results_df.to_json("experiment_results.json", orient="records")

#     return results_df, full_pred_df

# #### Notebook Utility Function for Colab

# import nbformat
# from google.colab import _message
# from nbformat import from_dict

# # Get current notebook content as dict
# nb_json = _message.blocking_request('get_ipynb')['ipynb']

# # Remove broken widgets metadata
# nb_json['metadata'].pop('widgets', None)

# # Convert dict to NotebookNode (needed for nbformat.write)
# nb_node = from_dict(nb_json)

# # Save cleaned notebook
# output_path = "/content/cleaned_notebook.ipynb"
# with open(output_path, "w") as f:
#     nbformat.write(nb_node, f)

# # Download it
# from google.colab import files
# files.download(output_path)

# #### Old Training Function for Perpetrator Model

# import pandas as pd
# import torch
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     Trainer,
#     TrainingArguments
# )
# from torch.utils.data import Dataset

# # -------------------------------------------------------
# # 1. Dataset for Single-Label Multi-Class Classification
# # -------------------------------------------------------
# class SingleLabelDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=128):
#         """
#         Args:
#             texts: List of text strings.
#             labels: List of integer class labels (e.g., [0, 1, 2, ...]).
#             tokenizer: A Hugging Face AutoTokenizer.
#             max_length: Max sequence length for tokenization.
#         """
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]

#         encoding = self.tokenizer(
#             text,
#             max_length=self.max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )

#         # Squeeze batch dimension
#         item = {k: v.squeeze() for k, v in encoding.items()}
#         # Integer labels for cross-entropy
#         item["labels"] = torch.tensor(label, dtype=torch.long)

#         return item

# # -------------------------------------------------------
# # 2. Custom Metric Function
# # -------------------------------------------------------
# def compute_metricss(eval_pred, label_names=None):
#     """
#     Computes accuracy and micro/macro precision/recall/F1 for multi-class classification.
#     """
#     logits, labels = eval_pred
#     preds = torch.argmax(torch.tensor(logits), dim=1).numpy()

#     # Accuracy
#     acc = accuracy_score(labels, preds)

#     # Classification Report (optionally naming classes)
#     report = classification_report(
#         labels,
#         preds,
#         target_names=label_names if label_names else None,
#         zero_division=0,
#         output_dict=True
#     )

#     if label_names:
#         print("\nFull Classification Report:\n",
#               classification_report(labels, preds, target_names=label_names, zero_division=0))
#     else:
#         print("\nFull Classification Report:\n",
#               classification_report(labels, preds, zero_division=0))
#     return {
#         "accuracy": acc,
#         "precision_macro": report["macro avg"]["precision"],
#         "recall_macro": report["macro avg"]["recall"],
#         "f1_macro": report["macro avg"]["f1-score"],
#         "precision_weighted": report["weighted avg"]["precision"],
#         "recall_weighted": report["weighted avg"]["recall"],
#         "f1_weighted": report["weighted avg"]["f1-score"],
#     }

# # -------------------------------------------------------
# # 3. Main Training Function with Train/Val/Test Splits
# # -------------------------------------------------------
# def train_multiclass_model_3way_split(
#     df,
#     text_col="incident_summary",
#     label_col="perpetrator",
#     model_name="bert-base-uncased",
#     test_size=0.1,
#     val_size=0.1,
#     epochs=2,
#     batch_size=8
# ):
#     """
#     Trains a multi-class classifier with separate train, val, and test sets.

#     Args:
#         data_path (str): CSV file path. Must contain `text_col` and `label_col`.
#         text_col (str): Name of the column containing the text.
#         label_col (str): Name of the column containing the class label (string form).
#         model_name (str): HF model identifier, e.g. 'bert-base-uncased', 'roberta-base', etc.
#         test_size (float): Fraction of entire dataset to hold out for final test set.
#         val_size (float): Fraction of entire dataset to hold out for validation set
#                           (relative to total, not just leftover).
#         epochs (int): Number of training epochs.
#         batch_size (int): Batch size for training and evaluation.

#     Returns:
#         model: The trained model (same as trainer.model).
#         tokenizer: The tokenizer used.
#         test_metrics: Evaluation metrics on the test set.
#     """

#     # 1. Load the data
#     # df = pd.read_csv(data_path)

#     # 2. Convert label from string to integer IDs
#     unique_labels = df[label_col].unique()
#     label2id = {label: i for i, label in enumerate(unique_labels)}
#     id2label = {v: k for k, v in label2id.items()}

#     df["label_id"] = df[label_col].map(label2id)

#     # 3. Create initial train+val vs test split
#     #    e.g. if test_size=0.1, then 10% of data is test, 90% is train+val
#     train_val_df, test_df = train_test_split(
#         df,
#         test_size=test_size,
#         random_state=42,
#         stratify=df["label_id"]  # optional, if you want stratified splits
#     )

#     # 4. From the train_val_df, create train vs val split
#     #    val_size is fraction of the entire dataset, so the fraction
#     #    within train_val_df is val_size / (1 - test_size).
#     val_fraction_of_trainval = val_size / (1 - test_size)
#     train_df, val_df = train_test_split(
#         train_val_df,
#         test_size=val_fraction_of_trainval,
#         random_state=42,
#         stratify=train_val_df["label_id"]
#     )

#     # 5. Prepare data lists
#     train_texts = train_df[text_col].tolist()
#     train_labels = train_df["label_id"].tolist()

#     val_texts = val_df[text_col].tolist()
#     val_labels = val_df["label_id"].tolist()

#     test_texts = test_df[text_col].tolist()
#     test_labels = test_df["label_id"].tolist()

#     # 6. Tokenizer & Model
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     num_labels = len(unique_labels)

#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_name,
#         num_labels=num_labels
#     )
#     model.to(device)


#     # 7. Create Dataset objects
#     train_dataset = SingleLabelDataset(train_texts, train_labels, tokenizer)
#     val_dataset = SingleLabelDataset(val_texts, val_labels, tokenizer)
#     test_dataset = SingleLabelDataset(test_texts, test_labels, tokenizer)

#     # 8. Training Arguments
#     training_args = TrainingArguments(
#         output_dir="./model_output",
#         num_train_epochs=epochs,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         logging_dir="./logs",
#         logging_steps=10,
#         report_to="none",
#         load_best_model_at_end=False  # set to True if you'd like to restore best checkpoint
#     )

#     # 9. Create Trainer
#     label_names = [id2label[i] for i in range(num_labels)]

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         compute_metrics=lambda eval_pred: compute_metricss(eval_pred, label_names=label_names)
#     )

#     # 10. Train
#     trainer.train()

#     # 11. Evaluate on the test set
#     #     By default, trainer.evaluate() returns a dictionary of metric values
#     test_metrics = trainer.evaluate(test_dataset)

#     print("\nTest Set Evaluation Results:", test_metrics)

#     # Return the final model, tokenizer, and test set metrics
#     return trainer.model, tokenizer, test_metrics

# ### Old Experiments Function for Perpetrator Model

# import pandas as pd

# def run_all_experiments_and_save(df_full, output_csv="experiment_results.csv"):
#     """
#     1. Iterates over the defined fractions & model list
#     2. Samples df_full according to fraction
#     3. Trains & evaluates using train_multiclass_model_3way_split
#     4. Saves the collected results in a DataFrame
#     5. Exports to CSV

#     Args:
#         df_full (pd.DataFrame): Full dataset with columns [label_col, text_col].
#         output_csv (str): File path to save the experiment results.
#     Returns:
#         results_df (pd.DataFrame): Contains experiment results for analysis.
#     """
#     results_list = []

#     for frac in fractions:
#         # Sample a fraction of the data
#         subset_size = int(len(df_full) * frac)
#         df_subset = df_full.sample(n=subset_size, random_state=42)

#         # Friendly fraction label if you want
#         frac_label = fraction_labels.get(frac, f"{frac*100:.1f}%")
#         print(f"\n=== DATA FRACTION: {frac} ({subset_size} rows) ===")

#         for model_name in models_list:
#             # Model label
#             model_label = model_name_labels.get(model_name, model_name)
#             print(f"Training model: {model_label}")

#             # Train & evaluate
#             model, tokenizer, test_metrics = train_multiclass_model_3way_split(
#                 df_subset,
#                 text_col="incident_summary",
#                 label_col="perpetrator",
#                 model_name=model_name,
#                 test_size=0.1,   # 10% for test
#                 val_size=0.1,   # 10% for val
#                 epochs=2,
#                 batch_size=32
#             )

#             # Build a result dict
#             run_result = {
#                 "fraction_raw": frac,
#                 "fraction_label": frac_label,
#                 "subset_size": subset_size,
#                 "model_raw": model_name,
#                 "model_label": model_label
#             }

#             # Merge test_metrics (e.g., accuracy, macro-F1, etc.)
#             for k, v in test_metrics.items():
#                 run_result[k] = v

#             # Append to results_list
#             results_list.append(run_result)

#     # Convert to DataFrame
#     results_df = pd.DataFrame(results_list)
#     # Save to CSV
#     results_df.to_csv(output_csv, index=False)
#     print(f"\nResults saved to {output_csv}")

#     # (Optionally, also save to JSON if desired)
#     # results_df.to_json("experiment_results.json", orient="records")

#     return results_df


