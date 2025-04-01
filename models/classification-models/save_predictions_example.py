"""
Example of how to add prediction saving to your existing notebook workflow.

Here's how to modify your workflow to save predictions for bootstrap resampling:

1. First, make sure to import the save_predictions module
2. After training and evaluating your model, call save_model_predictions
3. Then use bootstrap_f1_scores.py to get confidence intervals for F1 scores
"""

# =========================================================================
# EXAMPLE: Add this to your Jupyter notebook after training your model
# =========================================================================

# First, import the module (assuming it's in the same directory)
import save_predictions

# After training your model with train_transformer_model (when you have trainer and test_dataset)
target_names = data.drop(columns=["incident_summary"]).columns.tolist()

# Save predictions for bootstrap resampling
results_files = save_predictions.save_model_predictions(
    trainer,             # Your trained Trainer object
    test_dataset,        # Your test dataset
    target_names,        # List of target names (column names)
    model_name="distilbert-base-cased",  # Name of model
    data_fraction="100%"                 # Data fraction identifier
)

# This will save three files:
# - bootstrap_data/distilbert-base-cased_100%_true_labels.csv
# - bootstrap_data/distilbert-base-cased_100%_predictions.csv
# - bootstrap_data/distilbert-base-cased_100%_probabilities.csv

# =========================================================================
# OPTION 1: Run bootstrap resampling from command line
# =========================================================================

# Run this in a terminal (not in the notebook):
# python bootstrap_f1_scores.py \
#     --true_labels bootstrap_data/distilbert-base-cased_100%_true_labels.csv \
#     --predictions bootstrap_data/distilbert-base-cased_100%_predictions.csv \
#     --output bootstrap_results.csv

# =========================================================================
# OPTION 2: If you want to modify run_all_experiments_and_save to save
# predictions for all models automatically:
# =========================================================================

"""
def run_all_experiments_and_save(df_full, output_csv="experiment_results.csv"):
    # Import the save_predictions module
    import save_predictions
    
    results_list = []

    for frac in fractions:
        # Sample a fraction of the data
        subset_size = int(len(df_full) * frac)
        df_subset = df_full.sample(n=subset_size, random_state=42)

        # Friendly fraction label
        frac_label = fraction_labels.get(frac, f"{frac*100:.1f}%")
        print(f"\n=== DATA FRACTION: {frac} ({subset_size} rows) ===")

        for model_name in models_list:
            model_label = model_name_labels.get(model_name, model_name)
            print(f"Training model: {model_label}")

            # Train & evaluate
            trainer, test_results = train_transformer_model(
                model_name, 
                df_subset, 
                max_len=512, 
                test_size=0.1, 
                val_size=0.1, 
                batch_size=16, 
                epochs=2
            )
            
            # Save predictions for bootstrap resampling
            target_names = df_subset.drop(columns=["incident_summary"]).columns.tolist()
            save_predictions.save_model_predictions(
                trainer,
                test_dataset,  # You'll need to keep a reference to the test_dataset
                target_names,
                model_name=model_name,
                data_fraction=frac_label
            )

            # Build a result dict and continue as before...
            # ...
    
    return results_df
""" 