# Bootstrap Significance Testing for Location Extraction Models
# Compares all models to GLiNER baseline using bootstrap
# Generates GT tables with formatting for significant differences
# Uses tidymodels and tidyverse workflow

library(tidymodels)  # Includes rsample for bootstrapping
library(dplyr)
library(tidyr)
library(purrr)
library(readr)
library(gt)
library(reticulate)

# Configuration
# Paths are relative to papers/location-extraction directory
# Script should be run from papers/location-extraction directory
N_BOOT <- 5000
set.seed(42)
ALPHA <- 0.05  # 95% significance level

# Model groups
BERT_MODELS <- c(
  "deberta-v3",
  "muril",
  "spanbert",
  "xlm-roberta"
)

SEQ2SEQ_MODELS <- c(
  "flan-t5-base",
  "flan-t5-large",
  "flan-t5-xl-lora",
  "indicbart",
  "mt5-base"
)

LLM_MODELS <- c(
  "gpt4o_mini",
  "llama3_8b",
  "mistral_7b",
  "mixtral_8x7b"
)

BASELINE_NAME <- "gliner"

# Setup Python environment and import metrics function
cat("Setting up Python environment...\n")
tryCatch({
  use_virtualenv("../../venv", required = FALSE)
  py_run_string("
import sys
sys.path.append('../../models/location-models')
from utils.metrics_utils import compute_metrics_from_strings
")
  cat("✅ Python environment initialized successfully\n")
}, error = function(e) {
  cat("⚠️  Warning: Could not initialize Python environment\n")
  cat("Error:", conditionMessage(e), "\n")
  cat("Make sure you have:\n")
  cat("  1. Python virtualenv at ../../venv\n")
  cat("  2. rapidfuzz installed in the environment\n")
  stop("Cannot proceed without Python metrics function")
})

# Metric computation functions
compute_location_metrics <- function(predictions, ground_truths) {
  # Call Python function via reticulate
  metrics <- py$compute_metrics_from_strings(predictions, ground_truths)
  
  # Extract key metrics from nested structure
  tibble(
    micro_exact_f1 = metrics$overall$micro_exact_f1,
    micro_fuzzy_f1 = metrics$overall$micro_fuzzy_f1,
    exact_match = metrics$overall$exact_match,
    exact_core_match = metrics$overall$exact_core_match,
    fuzzy_match = metrics$overall$fuzzy_match,
    fuzzy_core_match = metrics$overall$fuzzy_core_match
  )
}

# Compute two-tailed p-value from paired bootstrap difference distribution
compute_pvalue_paired <- function(diff_dist) {
  # Remove NA values
  diff_dist <- diff_dist[!is.na(diff_dist)]
  
  if (length(diff_dist) == 0) {
    return(NA)
  }
  
  # Two-tailed test around 0: probability mass on or beyond zero
  # This tests H0: mean difference == 0 using the paired bootstrap difference distribution
  p_value <- 2 * min(mean(diff_dist <= 0), mean(diff_dist >= 0))
  pmin(pmax(p_value, 0), 1)
}

# Load baseline predictions (GLiNER)
load_baseline <- function(results_dir) {
  baseline_file <- file.path(results_dir, "location-extraction-bert", 
                              paste0(BASELINE_NAME, "_predictions.csv"))
  
  if (!file.exists(baseline_file)) {
    stop(paste("Baseline file not found:", baseline_file))
  }
  
  df <- read_csv(baseline_file, show_col_types = FALSE)
  
  # Normalize column names and incident_number
  df <- df |>
    rename(
      true_label = ground_truth,
      prediction = prediction
    ) |>
    mutate(
      # Remove .0 suffix from incident_number if present
      incident_number = as.character(as.integer(as.numeric(incident_number)))
    ) |>
    select(incident_number, true_label, prediction) |>
    filter(!is.na(true_label), !is.na(prediction))
  
  df
}

# Load model predictions for BERT models
load_bert_predictions <- function(model_name, results_dir) {
  csv_file <- file.path(results_dir, "location-extraction-bert", 
                        paste0(model_name, "_predictions.csv"))
  
  if (!file.exists(csv_file)) {
    warning(paste("File not found:", csv_file))
    return(NULL)
  }
  
  df <- read_csv(csv_file, show_col_types = FALSE)
  
  # Normalize column names and incident_number
  df <- df |>
    rename(
      true_label = ground_truth,
      prediction = prediction
    ) |>
    mutate(
      incident_number = as.character(as.integer(as.numeric(incident_number)))
    ) |>
    select(incident_number, true_label, prediction) |>
    filter(!is.na(true_label), !is.na(prediction))
  
  df
}

# Load model predictions for Seq2seq models
load_seq2seq_predictions <- function(model_name, results_dir) {
  csv_file <- file.path(results_dir, "location-extraction-seq2seq", 
                        paste0("location_", model_name, "_predictions.csv"))
  
  if (!file.exists(csv_file)) {
    warning(paste("File not found:", csv_file))
    return(NULL)
  }
  
  df <- read_csv(csv_file, show_col_types = FALSE)
  
  # Column names are already: incident_number, true_label, prediction
  df <- df |>
    mutate(
      incident_number = as.character(as.integer(as.numeric(incident_number)))
    ) |>
    select(incident_number, true_label, prediction) |>
    filter(!is.na(true_label), !is.na(prediction))
  
  df
}

# Load LLM predictions from merged file
load_llm_predictions <- function(model_name, results_dir) {
  csv_file <- file.path(results_dir, "location-extraction-llms", 
                        "merged_results.csv")
  
  if (!file.exists(csv_file)) {
    warning(paste("File not found:", csv_file))
    return(NULL)
  }
  
  df <- read_csv(csv_file, show_col_types = FALSE)
  
  # Extract the specific model's predictions
  pred_col <- paste0(model_name, "_pred")
  
  if (!pred_col %in% names(df)) {
    warning(paste("Prediction column not found:", pred_col, "in", csv_file))
    return(NULL)
  }
  
  df <- df |>
    rename(
      true_label = true_location,
      prediction = !!sym(pred_col)
    ) |>
    mutate(
      incident_number = as.character(as.integer(as.numeric(incident_number)))
    ) |>
    select(incident_number, true_label, prediction) |>
    filter(!is.na(true_label), !is.na(prediction))
  
  df
}

# Paired bootstrap: compute differences between baseline and model
bootstrap_paired_differences <- function(combined_df, n = N_BOOT) {
  # combined_df should have columns: incident_number, true_label, prediction_baseline, prediction_model
  
  cat(paste0("  Running ", n, " bootstrap iterations...\n"))
  
  bootstraps(combined_df, times = n) |>
    mutate(
      # Compute metrics for baseline and model on same resampled data
      baseline_metrics = map(splits, ~ {
        df_split <- analysis(.x)
        compute_location_metrics(
          df_split$prediction_baseline,
          df_split$true_label
        )
      }),
      model_metrics = map(splits, ~ {
        df_split <- analysis(.x)
        compute_location_metrics(
          df_split$prediction_model,
          df_split$true_label
        )
      })
    ) |>
    # Unnest metrics
    unnest_wider(baseline_metrics, names_sep = "_base_") |>
    unnest_wider(model_metrics, names_sep = "_model_") |>
    # Compute differences (model - baseline)
    mutate(
      micro_exact_f1_diff = micro_exact_f1_model_ - micro_exact_f1_base_,
      micro_fuzzy_f1_diff = micro_fuzzy_f1_model_ - micro_fuzzy_f1_base_,
      exact_match_diff = exact_match_model_ - exact_match_base_,
      exact_core_match_diff = exact_core_match_model_ - exact_core_match_base_,
      fuzzy_match_diff = fuzzy_match_model_ - fuzzy_match_base_,
      fuzzy_core_match_diff = fuzzy_core_match_model_ - fuzzy_core_match_base_
    )
}

# Process a model group and generate GT table
process_model_group <- function(model_names, results_dir, model_type, 
                                load_func, output_file) {
  cat("\n=== Processing", model_type, "models ===\n")
  
  # Load baseline
  cat("Loading baseline (GLiNER)...\n")
  baseline_df <- load_baseline(results_dir)
  
  # Compute baseline means directly (for display in table)
  baseline_metrics <- compute_location_metrics(
    baseline_df$prediction,
    baseline_df$true_label
  )
  
  cat(paste0("✓ Baseline: ", nrow(baseline_df), " incidents\n"))
  cat(paste0("  micro_exact_f1 = ", round(baseline_metrics$micro_exact_f1, 2), 
             ", micro_fuzzy_f1 = ", round(baseline_metrics$micro_fuzzy_f1, 2), "\n"))
  
  # Process each model with paired bootstrapping
  all_results_list <- map(model_names, function(model_name) {
    cat("\n", model_name, ":\n", sep = "")
    
    # Load predictions
    model_df <- load_func(model_name, results_dir)
    if (is.null(model_df)) {
      warning(paste("Skipping", model_name, "- could not load predictions"))
      return(NULL)
    }
    
    # Merge baseline and model predictions on incident_number
    combined_df <- baseline_df |>
      rename(prediction_baseline = prediction) |>
      inner_join(
        model_df |> rename(prediction_model = prediction),
        by = c("incident_number", "true_label")
      )
    
    if (nrow(combined_df) == 0) {
      warning(paste("No matching incidents between baseline and", model_name))
      return(NULL)
    }
    
    cat(paste0("  Paired bootstrap on ", nrow(combined_df), " incidents\n"))
    
    # Paired bootstrap: same resampling for both baseline and model
    paired_bootstrap <- bootstrap_paired_differences(combined_df, n = N_BOOT)
    
    # Compute model means from bootstrap
    model_means <- paired_bootstrap |>
      summarise(
        micro_exact_f1 = mean(micro_exact_f1_model_, na.rm = TRUE),
        micro_fuzzy_f1 = mean(micro_fuzzy_f1_model_, na.rm = TRUE),
        exact_match = mean(exact_match_model_, na.rm = TRUE),
        exact_core_match = mean(exact_core_match_model_, na.rm = TRUE),
        fuzzy_match = mean(fuzzy_match_model_, na.rm = TRUE),
        fuzzy_core_match = mean(fuzzy_core_match_model_, na.rm = TRUE)
      )
    
    # Compute p-values from paired bootstrap difference distribution
    p_values <- tibble(
      micro_exact_f1_p = compute_pvalue_paired(paired_bootstrap$micro_exact_f1_diff),
      micro_fuzzy_f1_p = compute_pvalue_paired(paired_bootstrap$micro_fuzzy_f1_diff),
      exact_match_p = compute_pvalue_paired(paired_bootstrap$exact_match_diff),
      exact_core_match_p = compute_pvalue_paired(paired_bootstrap$exact_core_match_diff),
      fuzzy_match_p = compute_pvalue_paired(paired_bootstrap$fuzzy_match_diff),
      fuzzy_core_match_p = compute_pvalue_paired(paired_bootstrap$fuzzy_core_match_diff)
    )
    
    # Print diagnostic info
    cat(paste0("  P-values: micro_exact_f1=", round(p_values$micro_exact_f1_p, 4),
               ", micro_fuzzy_f1=", round(p_values$micro_fuzzy_f1_p, 4), "\n"))
    
    # Combine results
    bind_cols(
      tibble(model = model_name),
      model_means,
      p_values
    )
  })
  
  # Filter out NULL results and bind
  all_results <- all_results_list |>
    compact() |>
    bind_rows()
  
  if (nrow(all_results) == 0) {
    warning("No models processed successfully. Skipping table generation.")
    return(NULL)
  }
  
  # Create table data
  table_data <- tibble(
    Metric = c("Micro Exact F1 ↑", "Micro Fuzzy F1 ↑", "Exact Match ↑", 
               "Exact Core Match ↑", "Fuzzy Match ↑", "Fuzzy Core Match ↑")
  )
  
  # Add baseline column
  baseline_col_name <- "GLiNER"
  table_data <- table_data |>
    mutate(
      !!baseline_col_name := c(
        baseline_metrics$micro_exact_f1,
        baseline_metrics$micro_fuzzy_f1,
        baseline_metrics$exact_match,
        baseline_metrics$exact_core_match,
        baseline_metrics$fuzzy_match,
        baseline_metrics$fuzzy_core_match
      )
    )
  
  # Add each model column
  for (i in seq_len(nrow(all_results))) {
    model_name <- all_results$model[i]
    model_vals <- all_results[i, c("micro_exact_f1", "micro_fuzzy_f1", "exact_match",
                                    "exact_core_match", "fuzzy_match", "fuzzy_core_match")]
    
    # Add column with values
    table_data <- table_data |>
      mutate(
        !!model_name := c(
          model_vals$micro_exact_f1,
          model_vals$micro_fuzzy_f1,
          model_vals$exact_match,
          model_vals$exact_core_match,
          model_vals$fuzzy_match,
          model_vals$fuzzy_core_match
        )
      )
  }
  
  # Create GT table
  gt_table <- table_data |>
    gt() |>
    tab_header(
      title = paste("Location Extraction Model Comparison:", tools::toTitleCase(model_type)),
      subtitle = paste("Bootstrap significance testing (n =", N_BOOT, "reps) vs GLiNER baseline")
    ) |>
    fmt_number(
      columns = -Metric,
      decimals = 2
    ) |>
    cols_label(
      Metric = "Metric"
    ) |>
    tab_style(
      style = cell_text(weight = "bold"),
      locations = cells_body(columns = Metric)
    )
  
  # Format significant values (bold and color)
  # Only bold values that are significantly better than baseline (higher is better for all metrics)
  for (i in seq_len(nrow(all_results))) {
    model_name <- all_results$model[i]
    model_vals <- all_results[i, c("micro_exact_f1", "micro_fuzzy_f1", "exact_match",
                                    "exact_core_match", "fuzzy_match", "fuzzy_core_match")]
    model_pvals <- all_results[i, c("micro_exact_f1_p", "micro_fuzzy_f1_p", "exact_match_p",
                                     "exact_core_match_p", "fuzzy_match_p", "fuzzy_core_match_p")]
    
    baseline_vals <- c(
      baseline_metrics$micro_exact_f1,
      baseline_metrics$micro_fuzzy_f1,
      baseline_metrics$exact_match,
      baseline_metrics$exact_core_match,
      baseline_metrics$fuzzy_match,
      baseline_metrics$fuzzy_core_match
    )
    
    # Determine which rows are significant AND better (higher is better for all location metrics)
    significant_and_better <- c(
      model_pvals$micro_exact_f1_p < ALPHA & model_vals$micro_exact_f1 > baseline_metrics$micro_exact_f1,
      model_pvals$micro_fuzzy_f1_p < ALPHA & model_vals$micro_fuzzy_f1 > baseline_metrics$micro_fuzzy_f1,
      model_pvals$exact_match_p < ALPHA & model_vals$exact_match > baseline_metrics$exact_match,
      model_pvals$exact_core_match_p < ALPHA & model_vals$exact_core_match > baseline_metrics$exact_core_match,
      model_pvals$fuzzy_match_p < ALPHA & model_vals$fuzzy_match > baseline_metrics$fuzzy_match,
      model_pvals$fuzzy_core_match_p < ALPHA & model_vals$fuzzy_core_match > baseline_metrics$fuzzy_core_match
    )
    
    if (any(significant_and_better)) {
      gt_table <- gt_table |>
        tab_style(
          style = cell_text(weight = "bold", color = "#0066CC"),
          locations = cells_body(
            columns = !!sym(model_name),
            rows = significant_and_better
          )
        )
    }
  }
  
  # Format baseline column header
  gt_table <- gt_table |>
    tab_style(
      style = cell_text(weight = "bold"),
      locations = cells_column_labels(columns = baseline_col_name)
    )
  
  # Save results to CSV
  results_csv <- file.path(dirname(output_file), 
                           paste0("bootstrap_significance_", model_type, "_results.csv"))
  cat("\nSaving results to:", results_csv, "\n")
  
  # Create comprehensive results table with baseline comparison
  results_summary <- all_results |>
    mutate(
      baseline_micro_exact_f1 = baseline_metrics$micro_exact_f1,
      baseline_micro_fuzzy_f1 = baseline_metrics$micro_fuzzy_f1,
      baseline_exact_match = baseline_metrics$exact_match,
      baseline_exact_core_match = baseline_metrics$exact_core_match,
      baseline_fuzzy_match = baseline_metrics$fuzzy_match,
      baseline_fuzzy_core_match = baseline_metrics$fuzzy_core_match,
      micro_exact_f1_diff = micro_exact_f1 - baseline_micro_exact_f1,
      micro_fuzzy_f1_diff = micro_fuzzy_f1 - baseline_micro_fuzzy_f1,
      exact_match_diff = exact_match - baseline_exact_match,
      exact_core_match_diff = exact_core_match - baseline_exact_core_match,
      fuzzy_match_diff = fuzzy_match - baseline_fuzzy_match,
      fuzzy_core_match_diff = fuzzy_core_match - baseline_fuzzy_core_match,
      micro_exact_f1_significant = micro_exact_f1_p < ALPHA,
      micro_fuzzy_f1_significant = micro_fuzzy_f1_p < ALPHA,
      exact_match_significant = exact_match_p < ALPHA,
      exact_core_match_significant = exact_core_match_p < ALPHA,
      fuzzy_match_significant = fuzzy_match_p < ALPHA,
      fuzzy_core_match_significant = fuzzy_core_match_p < ALPHA,
      micro_exact_f1_better = micro_exact_f1 > baseline_micro_exact_f1,
      micro_fuzzy_f1_better = micro_fuzzy_f1 > baseline_micro_fuzzy_f1,
      exact_match_better = exact_match > baseline_exact_match,
      exact_core_match_better = exact_core_match > baseline_exact_core_match,
      fuzzy_match_better = fuzzy_match > baseline_fuzzy_match,
      fuzzy_core_match_better = fuzzy_core_match > baseline_fuzzy_core_match
    )
  
  write_csv(results_summary, results_csv)
  cat("Results saved!\n")
  
  # Save as PNG
  cat("Saving GT table to:", output_file, "\n")
  gtsave(gt_table, output_file)
  
  cat("Done!\n")
  
  return(list(
    table_data = table_data,
    bootstrap_results = all_results,
    results_summary = results_summary,
    gt_table = gt_table
  ))
}

# Main execution
cat(paste0(rep("=", 70), collapse = ""), "\n")
cat("Bootstrap Significance Testing for Location Extraction Models\n")
cat(paste0(rep("=", 70), collapse = ""), "\n\n")

# Set results directory (relative to papers/location-extraction)
results_dir <- "results"

# Process BERT models
bert_results <- process_model_group(
  model_names = BERT_MODELS,
  results_dir = results_dir,
  model_type = "bert",
  load_func = load_bert_predictions,
  output_file = "data-viz/images/bootstrap_significance_bert.png"
)

# Process Seq2seq models
seq2seq_results <- process_model_group(
  model_names = SEQ2SEQ_MODELS,
  results_dir = results_dir,
  model_type = "seq2seq",
  load_func = load_seq2seq_predictions,
  output_file = "data-viz/images/bootstrap_significance_seq2seq.png"
)

# Process LLM models
llm_results <- process_model_group(
  model_names = LLM_MODELS,
  results_dir = results_dir,
  model_type = "llms",
  load_func = load_llm_predictions,
  output_file = "data-viz/images/bootstrap_significance_llms.png"
)

cat("\n", paste0(rep("=", 70), collapse = ""), "\n")
cat("All done!\n")
cat(paste0(rep("=", 70), collapse = ""), "\n")

