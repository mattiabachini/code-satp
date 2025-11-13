# Bootstrap Significance Testing for Death Count Models
# Compares all models to ConfliBERT-Poisson baseline using bootstrap
# Generates GT tables with formatting for significant differences
# Uses tidymodels and tidyverse workflow

library(tidymodels)  # Includes rsample for bootstrapping
library(dplyr)
library(tidyr)
library(purrr)
library(readr)
library(gt)
library(jsonlite)

# Configuration
# Paths are relative to papers/death-counts directory
# Script should be run from papers/death-counts directory
N_BOOT <- 5000
set.seed(42)
ALPHA <- 0.05  # 95% significance level

# Model groups
SEQ2SEQ_MODELS <- c(
  "flan-t5-base",
  "flan-t5-large",
  "flan-t5-xl-lora",
  "indicbart",
  "mt5-base",
  "nt5-small"
)

LLM_MODELS <- c(
  "gpt4o_mini",
  "llama3_8b",
  "mistral_7b",
  "mixtral_8x7b"
)

BASELINE_NAME <- "conflibert-poisson"

# Metric computation functions
compute_mae <- function(truth, pred) {
  mean(abs(truth - pred), na.rm = TRUE)
}

compute_rmse <- function(truth, pred) {
  sqrt(mean((truth - pred)^2, na.rm = TRUE))
}

compute_within_1 <- function(truth, pred) {
  mean(abs(truth - pred) <= 1, na.rm = TRUE)
}

compute_within_2 <- function(truth, pred) {
  mean(abs(truth - pred) <= 2, na.rm = TRUE)
}

compute_nonzero_mae <- function(truth, pred) {
  nonzero_idx <- truth > 0
  if (sum(nonzero_idx) == 0) return(NA)
  mean(abs(truth[nonzero_idx] - pred[nonzero_idx]), na.rm = TRUE)
}

# Bootstrap all metrics for a model (used for baseline only)
bootstrap_metrics <- function(df_model, n = N_BOOT) {
  bootstraps(df_model, times = n) |>
    mutate(
      mae = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_mae(df_split$true_label, df_split$prediction)
      }),
      rmse = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_rmse(df_split$true_label, df_split$prediction)
      }),
      within_1 = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_within_1(df_split$true_label, df_split$prediction)
      }),
      within_2 = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_within_2(df_split$true_label, df_split$prediction)
      }),
      nonzero_mae = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_nonzero_mae(df_split$true_label, df_split$prediction)
      })
    )
}

# Paired bootstrap: compute differences between baseline and model
# Uses the same bootstrap resampling for both, ensuring paired comparisons
bootstrap_paired_differences <- function(combined_df, n = N_BOOT) {
  # combined_df should have columns: true_label, prediction_baseline, prediction_model
  bootstraps(combined_df, times = n) |>
    mutate(
      # Compute metrics for baseline and model on same resampled data
      mae_baseline = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_mae(df_split$true_label, df_split$prediction_baseline)
      }),
      mae_model = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_mae(df_split$true_label, df_split$prediction_model)
      }),
      rmse_baseline = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_rmse(df_split$true_label, df_split$prediction_baseline)
      }),
      rmse_model = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_rmse(df_split$true_label, df_split$prediction_model)
      }),
      within_1_baseline = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_within_1(df_split$true_label, df_split$prediction_baseline)
      }),
      within_1_model = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_within_1(df_split$true_label, df_split$prediction_model)
      }),
      within_2_baseline = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_within_2(df_split$true_label, df_split$prediction_baseline)
      }),
      within_2_model = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_within_2(df_split$true_label, df_split$prediction_model)
      }),
      nonzero_mae_baseline = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_nonzero_mae(df_split$true_label, df_split$prediction_baseline)
      }),
      nonzero_mae_model = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_nonzero_mae(df_split$true_label, df_split$prediction_model)
      })
    ) |>
    # Compute differences (model - baseline)
    mutate(
      mae_diff = mae_model - mae_baseline,
      rmse_diff = rmse_model - rmse_baseline,
      within_1_diff = within_1_model - within_1_baseline,
      within_2_diff = within_2_model - within_2_baseline,
      nonzero_mae_diff = nonzero_mae_model - nonzero_mae_baseline
    ) |>
    # Also keep individual metrics for means
    mutate(
      mae = mae_model,
      rmse = rmse_model,
      within_1 = within_1_model,
      within_2 = within_2_model,
      nonzero_mae = nonzero_mae_model,
      mae_baseline_val = mae_baseline,
      rmse_baseline_val = rmse_baseline,
      within_1_baseline_val = within_1_baseline,
      within_2_baseline_val = within_2_baseline,
      nonzero_mae_baseline_val = nonzero_mae_baseline
    )
}

# Load baseline predictions
load_baseline <- function(results_dir) {
  baseline_file <- file.path(results_dir, paste0("death_counts_", BASELINE_NAME, "_predictions.csv"))
  
  if (!file.exists(baseline_file)) {
    stop(paste("Baseline file not found:", baseline_file))
  }
  
  df <- read_csv(baseline_file, show_col_types = FALSE)
  
  # Use incident_number if available, otherwise use true_label for matching
  if ("incident_number" %in% names(df)) {
    df |>
      select(incident_number, true_label, prediction) |>
      filter(!is.na(true_label), !is.na(prediction))
  } else {
    df |>
      select(true_label, prediction) |>
      filter(!is.na(true_label), !is.na(prediction)) |>
      mutate(incident_number = row_number())  # Create dummy ID if not available
  }
}

# Load model predictions (handles both seq2seq and LLM formats)
load_model_predictions <- function(model_name, results_dir, model_type = c("seq2seq", "llm")) {
  model_type <- match.arg(model_type)
  
  if (model_type == "llm") {
    csv_file <- file.path(results_dir, paste0(model_name, ".csv"))
    pred_col <- paste0(model_name, "_prediction")
    
    if (!file.exists(csv_file)) {
      warning(paste("File not found:", csv_file))
      return(NULL)
    }
    
    df <- read_csv(csv_file, show_col_types = FALSE)
    
    if (!pred_col %in% names(df)) {
      warning(paste("Prediction column not found:", pred_col, "in", csv_file))
      return(NULL)
    }
    
    # Select columns, including incident_number if available
    cols_to_select <- c("true_label", pred_col)
    if ("incident_number" %in% names(df)) {
      cols_to_select <- c("incident_number", cols_to_select)
    }
    
    result <- df |>
      select(all_of(cols_to_select)) |>
      rename(true_label = true_label, prediction = !!sym(pred_col)) |>
      filter(!is.na(true_label), !is.na(prediction))
    
    # Add incident_number if not available
    if (!"incident_number" %in% names(result)) {
      result <- result |>
        mutate(incident_number = row_number())
    }
    result
    
  } else {  # seq2seq
    # Special handling for mt5-base
    if (model_name == "mt5-base") {
      csv_file <- file.path(results_dir, "death_counts_predictions_combined.csv")
      if (!file.exists(csv_file)) {
        warning(paste("File not found:", csv_file))
        return(NULL)
      }
      df <- read_csv(csv_file, show_col_types = FALSE)
      pred_col <- "mt5-base_pred"
      
      if (!pred_col %in% names(df)) {
        warning(paste("Prediction column not found:", pred_col, "in", csv_file))
        return(NULL)
      }
      
      # Select columns, including incident_number if available
      cols_to_select <- c("true_label", pred_col)
      if ("incident_number" %in% names(df)) {
        cols_to_select <- c("incident_number", cols_to_select)
      }
      
      result <- df |>
        select(all_of(cols_to_select)) |>
        rename(true_label = true_label, prediction = !!sym(pred_col)) |>
        filter(!is.na(true_label), !is.na(prediction))
      
      # Add incident_number if not available
      if (!"incident_number" %in% names(result)) {
        result <- result |>
          mutate(incident_number = row_number())
      }
      result
    } else {
      csv_file <- file.path(results_dir, paste0("death_counts_", model_name, "_predictions.csv"))
      
      if (!file.exists(csv_file)) {
        warning(paste("File not found:", csv_file))
        return(NULL)
      }
      
      df <- read_csv(csv_file, show_col_types = FALSE)
      
      # Select columns, including incident_number if available
      cols_to_select <- c("true_label", "prediction")
      if ("incident_number" %in% names(df)) {
        cols_to_select <- c("incident_number", cols_to_select)
      }
      
      result <- df |>
        select(all_of(cols_to_select)) |>
        filter(!is.na(true_label), !is.na(prediction))
      
      # Add incident_number if not available
      if (!"incident_number" %in% names(result)) {
        result <- result |>
          mutate(incident_number = row_number())
      }
      result
    }
  }
}

# Compute two-tailed p-value from paired bootstrap difference distribution
# Uses the difference distribution directly from paired bootstrap
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

# Process a model group and generate GT table
process_model_group <- function(model_names, results_dir, model_type, output_file) {
  cat("\n=== Processing", model_type, "models ===\n")
  
  # Load baseline
  cat("Loading baseline...\n")
  baseline_df <- load_baseline(results_dir)
  
  # Compute baseline means directly (for display in table)
  # We'll compute baseline means from paired bootstrap later
  baseline_means <- baseline_df |>
    summarise(
      mae = compute_mae(true_label, prediction),
      rmse = compute_rmse(true_label, prediction),
      within_1 = compute_within_1(true_label, prediction),
      within_2 = compute_within_2(true_label, prediction),
      nonzero_mae = compute_nonzero_mae(true_label, prediction)
    )
  
  # Validate baseline against JSON metrics file
  baseline_json_file <- file.path(results_dir, paste0("death_counts_", BASELINE_NAME, "_metrics.json"))
  if (file.exists(baseline_json_file)) {
    baseline_json_metrics <- fromJSON(baseline_json_file)
    baseline_json_overall <- baseline_json_metrics$overall
    
    tolerance <- 0.01
    if (abs(baseline_means$mae - baseline_json_overall$mae) > tolerance ||
        abs(baseline_means$rmse - baseline_json_overall$rmse) > tolerance ||
        abs(baseline_means$within_1 - baseline_json_overall$within_1) > tolerance ||
        abs(baseline_means$within_2 - baseline_json_overall$within_2) > tolerance ||
        abs(baseline_means$nonzero_mae - baseline_json_overall$nonzero_mae) > tolerance) {
      warning("Baseline computed means differ from JSON metrics")
    } else {
      cat("✓ Baseline validation passed against JSON metrics\n")
    }
  }
  
  # Process each model with paired bootstrapping
  all_results_list <- map(model_names, function(model_name) {
    cat("Processing", model_name, "...\n")
    
    # Load predictions
    model_df <- load_model_predictions(model_name, results_dir, model_type)
    if (is.null(model_df)) {
      warning(paste("Skipping", model_name, "- could not load predictions"))
      return(NULL)
    }
    
    # Merge baseline and model predictions on incident_number (or true_label if no incident_number)
    # This ensures we're comparing the same incidents
    if ("incident_number" %in% names(baseline_df) && "incident_number" %in% names(model_df)) {
      combined_df <- baseline_df |>
        rename(prediction_baseline = prediction) |>
        inner_join(
          model_df |> rename(prediction_model = prediction),
          by = c("incident_number", "true_label")
        )
    } else {
      # Fallback to true_label matching if incident_number not available
      combined_df <- baseline_df |>
        rename(prediction_baseline = prediction) |>
        inner_join(
          model_df |> rename(prediction_model = prediction),
          by = "true_label"
        )
    }
    
    if (nrow(combined_df) == 0) {
      warning(paste("No matching incidents between baseline and", model_name))
      return(NULL)
    }
    
    cat("  Paired bootstrapping (", N_BOOT, " reps) on", nrow(combined_df), "incidents...\n")
    
    # Paired bootstrap: same resampling for both baseline and model
    paired_bootstrap <- bootstrap_paired_differences(combined_df, n = N_BOOT)
    
    # Compute model means from bootstrap
    model_means <- paired_bootstrap |>
      summarise(
        mae = mean(mae, na.rm = TRUE),
        rmse = mean(rmse, na.rm = TRUE),
        within_1 = mean(within_1, na.rm = TRUE),
        within_2 = mean(within_2, na.rm = TRUE),
        nonzero_mae = mean(nonzero_mae, na.rm = TRUE)
      )
    
    # Compute baseline means from paired bootstrap (for consistency)
    baseline_means_boot <- paired_bootstrap |>
      summarise(
        mae = mean(mae_baseline_val, na.rm = TRUE),
        rmse = mean(rmse_baseline_val, na.rm = TRUE),
        within_1 = mean(within_1_baseline_val, na.rm = TRUE),
        within_2 = mean(within_2_baseline_val, na.rm = TRUE),
        nonzero_mae = mean(nonzero_mae_baseline_val, na.rm = TRUE)
      )
    
    # Validate against JSON metrics file if it exists
    if (model_type == "llm") {
      json_file <- file.path(results_dir, paste0(model_name, "_metrics.json"))
    } else {
      json_file <- file.path(results_dir, paste0("death_counts_", model_name, "_metrics.json"))
    }
    
    if (file.exists(json_file)) {
      json_metrics <- fromJSON(json_file)
      json_overall <- json_metrics$overall
      
      # Compare computed means to JSON (allow small tolerance for rounding)
      tolerance <- 0.01
      if (abs(model_means$mae - json_overall$mae) > tolerance ||
          abs(model_means$rmse - json_overall$rmse) > tolerance ||
          abs(model_means$within_1 - json_overall$within_1) > tolerance ||
          abs(model_means$within_2 - json_overall$within_2) > tolerance ||
          abs(model_means$nonzero_mae - json_overall$nonzero_mae) > tolerance) {
        warning(paste("Validation warning for", model_name, 
                     ": Bootstrap means differ from JSON metrics"))
      } else {
        cat("  ✓ Validation passed against JSON metrics\n")
      }
    }
    
    # Compute p-values from paired bootstrap difference distribution
    p_values <- tibble(
      mae_p = compute_pvalue_paired(paired_bootstrap$mae_diff),
      rmse_p = compute_pvalue_paired(paired_bootstrap$rmse_diff),
      within_1_p = compute_pvalue_paired(paired_bootstrap$within_1_diff),
      within_2_p = compute_pvalue_paired(paired_bootstrap$within_2_diff),
      nonzero_mae_p = compute_pvalue_paired(paired_bootstrap$nonzero_mae_diff)
    )
    
    # Print diagnostic info
    cat("  P-values: MAE=", round(p_values$mae_p, 4), 
        ", RMSE=", round(p_values$rmse_p, 4),
        ", within_1=", round(p_values$within_1_p, 4),
        ", within_2=", round(p_values$within_2_p, 4),
        ", nonzero_mae=", round(p_values$nonzero_mae_p, 4), "\n")
    
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
    Metric = c("MAE ↓", "RMSE ↓", "within-1 ↑", "within-2 ↑", "nonzero-mae ↓")
  )
  
  # Add baseline column (use readable name)
  baseline_col_name <- "ConfliBERT-Poisson"
  table_data <- table_data |>
    mutate(
      !!baseline_col_name := c(
        baseline_means$mae,
        baseline_means$rmse,
        baseline_means$within_1,
        baseline_means$within_2,
        baseline_means$nonzero_mae
      )
    )
  
  # Add each model column
  for (i in seq_len(nrow(all_results))) {
    model_name <- all_results$model[i]
    model_means <- all_results[i, c("mae", "rmse", "within_1", "within_2", "nonzero_mae")]
    model_pvals <- all_results[i, c("mae_p", "rmse_p", "within_1_p", "within_2_p", "nonzero_mae_p")]
    
    # Determine which values are significant and better
    significant <- c(
      model_pvals$mae_p < ALPHA & model_means$mae < baseline_means$mae,  # Lower is better
      model_pvals$rmse_p < ALPHA & model_means$rmse < baseline_means$rmse,  # Lower is better
      model_pvals$within_1_p < ALPHA & model_means$within_1 > baseline_means$within_1,  # Higher is better
      model_pvals$within_2_p < ALPHA & model_means$within_2 > baseline_means$within_2,  # Higher is better
      model_pvals$nonzero_mae_p < ALPHA & model_means$nonzero_mae < baseline_means$nonzero_mae  # Lower is better
    )
    
    # Add column with values
    table_data <- table_data |>
      mutate(
        !!model_name := c(
          model_means$mae,
          model_means$rmse,
          model_means$within_1,
          model_means$within_2,
          model_means$nonzero_mae
        )
      )
  }
  
  # Create readable title label for model type
  title_label <- if (model_type == "llm") {
    "LLMs"
  } else if (model_type == "seq2seq") {
    "seq2seq"
  } else {
    tools::toTitleCase(model_type)
  }
  
  # Create GT table
  gt_table <- table_data |>
    gt() |>
    tab_header(
      title = paste("Death Count Model Comparison:", title_label),
      subtitle = paste("Bootstrap significance testing (n =", N_BOOT, "reps) vs ConfliBERT-Poisson baseline")
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
  # Only bold values that are significantly better than baseline
  for (i in seq_len(nrow(all_results))) {
    model_name <- all_results$model[i]
    model_means <- all_results[i, c("mae", "rmse", "within_1", "within_2", "nonzero_mae")]
    model_pvals <- all_results[i, c("mae_p", "rmse_p", "within_1_p", "within_2_p", "nonzero_mae_p")]
    
    # Determine which rows are significant AND better
    # Lower is better for MAE, RMSE, nonzero_mae
    # Higher is better for within_1, within_2
    significant_and_better <- c(
      model_pvals$mae_p < ALPHA & model_means$mae < baseline_means$mae,
      model_pvals$rmse_p < ALPHA & model_means$rmse < baseline_means$rmse,
      model_pvals$within_1_p < ALPHA & model_means$within_1 > baseline_means$within_1,
      model_pvals$within_2_p < ALPHA & model_means$within_2 > baseline_means$within_2,
      model_pvals$nonzero_mae_p < ALPHA & model_means$nonzero_mae < baseline_means$nonzero_mae
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
  results_csv <- file.path(dirname(output_file), paste0("bootstrap_significance_", model_type, "_results.csv"))
  cat("\nSaving results to:", results_csv, "\n")
  
  # Create comprehensive results table with baseline comparison
  results_summary <- all_results |>
    mutate(
      baseline_mae = baseline_means$mae,
      baseline_rmse = baseline_means$rmse,
      baseline_within_1 = baseline_means$within_1,
      baseline_within_2 = baseline_means$within_2,
      baseline_nonzero_mae = baseline_means$nonzero_mae,
      mae_diff = mae - baseline_mae,
      rmse_diff = rmse - baseline_rmse,
      within_1_diff = within_1 - baseline_within_1,
      within_2_diff = within_2 - baseline_within_2,
      nonzero_mae_diff = nonzero_mae - baseline_nonzero_mae,
      mae_significant = mae_p < ALPHA,
      rmse_significant = rmse_p < ALPHA,
      within_1_significant = within_1_p < ALPHA,
      within_2_significant = within_2_p < ALPHA,
      nonzero_mae_significant = nonzero_mae_p < ALPHA,
      mae_better = mae < baseline_mae,
      rmse_better = rmse < baseline_rmse,
      within_1_better = within_1 > baseline_within_1,
      within_2_better = within_2 > baseline_within_2,
      nonzero_mae_better = nonzero_mae < baseline_nonzero_mae
    ) |>
    select(
      model,
      mae, mae_p, mae_diff, mae_significant, mae_better,
      rmse, rmse_p, rmse_diff, rmse_significant, rmse_better,
      within_1, within_1_p, within_1_diff, within_1_significant, within_1_better,
      within_2, within_2_p, within_2_diff, within_2_significant, within_2_better,
      nonzero_mae, nonzero_mae_p, nonzero_mae_diff, nonzero_mae_significant, nonzero_mae_better
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
cat("Bootstrap Significance Testing for Death Count Models\n")
cat(paste0(rep("=", 70), collapse = ""), "\n\n")

# Process seq2seq models
seq2seq_results <- process_model_group(
  model_names = SEQ2SEQ_MODELS,
  results_dir = "results/death-counts-seq2seq",
  model_type = "seq2seq",
  output_file = "data-viz/images/bootstrap_significance_seq2seq.png"
)

# Process LLM models
llm_results <- process_model_group(
  model_names = LLM_MODELS,
  results_dir = "results/death-counts-llms",
  model_type = "llm",
  output_file = "data-viz/images/bootstrap_significance_llms.png"
)

cat("\n", paste0(rep("=", 70), collapse = ""), "\n")
cat("All done!\n")
cat(paste0(rep("=", 70), collapse = ""), "\n")

