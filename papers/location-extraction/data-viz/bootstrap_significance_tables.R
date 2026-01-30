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

cat("Starting bootstrap significance testing for location extraction models...\n")
cat("Using pure R implementation for fast, stable processing\n\n")

# Helper function to parse location string into a simple list
parse_location_simple <- function(loc_str) {
  if (is.na(loc_str) || loc_str == "") {
    return(list(state = NA, district = NA, village = NA, other_locations = NA))
  }
  
  result <- list(state = NA, district = NA, village = NA, other_locations = NA)
  
  # Split by comma
  parts <- strsplit(loc_str, ",")[[1]]
  for (part in parts) {
    part <- trimws(part)
    if (grepl("^state:", part, ignore.case = TRUE)) {
      result$state <- trimws(sub("^state:\\s*", "", part, ignore.case = TRUE))
    } else if (grepl("^district:", part, ignore.case = TRUE)) {
      result$district <- trimws(sub("^district:\\s*", "", part, ignore.case = TRUE))
    } else if (grepl("^village:", part, ignore.case = TRUE)) {
      result$village <- trimws(sub("^village:\\s*", "", part, ignore.case = TRUE))
    } else if (grepl("^other_locations:", part, ignore.case = TRUE)) {
      result$other_locations <- trimws(sub("^other_locations:\\s*", "", part, ignore.case = TRUE))
    }
  }
  
  result
}

# Compute per-example match indicators (pure R, no Python calls)
compute_example_matches <- function(predictions, ground_truths) {
  n <- length(predictions)
  
  # Initialize result vectors
  state_match <- logical(n)
  district_match <- logical(n)
  village_match <- logical(n)
  other_match <- logical(n)
  exact_match <- logical(n)
  exact_core_match <- logical(n)
  
  # Parse and compare each example
  for (i in 1:n) {
    pred <- parse_location_simple(predictions[i])
    truth <- parse_location_simple(ground_truths[i])
    
    # Case-insensitive comparison, treating NA as matching NA
    state_match[i] <- (is.na(pred$state) & is.na(truth$state)) || 
                      (!is.na(pred$state) && !is.na(truth$state) && 
                       tolower(pred$state) == tolower(truth$state))
    
    district_match[i] <- (is.na(pred$district) & is.na(truth$district)) || 
                         (!is.na(pred$district) && !is.na(truth$district) && 
                          tolower(pred$district) == tolower(truth$district))
    
    village_match[i] <- (is.na(pred$village) & is.na(truth$village)) || 
                        (!is.na(pred$village) && !is.na(truth$village) && 
                         tolower(pred$village) == tolower(truth$village))
    
    other_match[i] <- (is.na(pred$other_locations) & is.na(truth$other_locations)) || 
                      (!is.na(pred$other_locations) && !is.na(truth$other_locations) && 
                       tolower(pred$other_locations) == tolower(truth$other_locations))
    
    # Overall matches
    exact_match[i] <- state_match[i] & district_match[i] & village_match[i] & other_match[i]
    exact_core_match[i] <- state_match[i] & district_match[i] & village_match[i]
  }
  
  tibble(
    example_idx = 1:n,
    prediction = predictions,
    ground_truth = ground_truths,
    state_match = state_match,
    district_match = district_match,
    village_match = village_match,
    other_match = other_match,
    exact_match = exact_match,
    exact_core_match = exact_core_match
  )
}

# Aggregate per-example matches to metrics (percentage)
aggregate_matches_to_metrics <- function(match_df) {
  tibble(
    micro_f1 = mean(c(match_df$state_match, match_df$district_match, 
                      match_df$village_match, match_df$other_match), na.rm = TRUE) * 100,
    exact_match = mean(match_df$exact_match, na.rm = TRUE) * 100,
    exact_core_match = mean(match_df$exact_core_match, na.rm = TRUE) * 100
  )
}

# Wrapper function: compute metrics from predictions and ground truths
# This is called once per model on full dataset, not in bootstrap loop
compute_location_metrics <- function(predictions, ground_truths) {
  matches <- compute_example_matches(predictions, ground_truths)
  aggregate_matches_to_metrics(matches)
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
# OPTIMIZED: Precomputes match indicators once, then just aggregates in bootstrap loop
bootstrap_paired_differences <- function(combined_df, n = N_BOOT) {
  # combined_df should have columns: incident_number, true_label, prediction_baseline, prediction_model
  
  cat(paste0("  Precomputing match indicators...\n"))
  
  # Precompute all match indicators ONCE (not in loop)
  baseline_matches <- compute_example_matches(
    combined_df$prediction_baseline,
    combined_df$true_label
  )
  
  model_matches <- compute_example_matches(
    combined_df$prediction_model,
    combined_df$true_label
  )
  
  # Add match indicators to dataframe
  df_with_matches <- combined_df |>
    mutate(
      state_match_base = baseline_matches$state_match,
      district_match_base = baseline_matches$district_match,
      village_match_base = baseline_matches$village_match,
      other_match_base = baseline_matches$other_match,
      exact_match_base = baseline_matches$exact_match,
      exact_core_match_base = baseline_matches$exact_core_match,
      
      state_match_model = model_matches$state_match,
      district_match_model = model_matches$district_match,
      village_match_model = model_matches$village_match,
      other_match_model = model_matches$other_match,
      exact_match_model = model_matches$exact_match,
      exact_core_match_model = model_matches$exact_core_match
    )
  
  cat(paste0("  Running ", n, " bootstrap iterations (fast aggregation only)...\n"))
  
  # Now just resample and aggregate (pure R operations, very fast)
  bootstraps(df_with_matches, times = n) |>
    mutate(
      # Aggregate baseline matches (no parsing, just mean of boolean vectors)
      baseline_metrics = map(splits, ~ {
        df_split <- analysis(.x)
        tibble(
          micro_f1 = mean(c(df_split$state_match_base, df_split$district_match_base,
                           df_split$village_match_base, df_split$other_match_base), 
                          na.rm = TRUE) * 100,
          exact_match = mean(df_split$exact_match_base, na.rm = TRUE) * 100,
          exact_core_match = mean(df_split$exact_core_match_base, na.rm = TRUE) * 100
        )
      }),
      # Aggregate model matches
      model_metrics = map(splits, ~ {
        df_split <- analysis(.x)
        tibble(
          micro_f1 = mean(c(df_split$state_match_model, df_split$district_match_model,
                           df_split$village_match_model, df_split$other_match_model), 
                          na.rm = TRUE) * 100,
          exact_match = mean(df_split$exact_match_model, na.rm = TRUE) * 100,
          exact_core_match = mean(df_split$exact_core_match_model, na.rm = TRUE) * 100
        )
      })
    ) |>
    # Extract metrics from nested tibbles
    mutate(
      # Baseline metrics
      micro_f1_base = map_dbl(baseline_metrics, ~.x$micro_f1),
      exact_match_base = map_dbl(baseline_metrics, ~.x$exact_match),
      exact_core_match_base = map_dbl(baseline_metrics, ~.x$exact_core_match),
      # Model metrics
      micro_f1_model = map_dbl(model_metrics, ~.x$micro_f1),
      exact_match_model = map_dbl(model_metrics, ~.x$exact_match),
      exact_core_match_model = map_dbl(model_metrics, ~.x$exact_core_match),
      # Compute differences (model - baseline)
      micro_f1_diff = micro_f1_model - micro_f1_base,
      exact_match_diff = exact_match_model - exact_match_base,
      exact_core_match_diff = exact_core_match_model - exact_core_match_base
    ) |>
    # Drop the list columns
    select(-baseline_metrics, -model_metrics)
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
  cat(paste0("  micro_f1 = ", round(baseline_metrics$micro_f1, 2), "\n"))
  
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
        micro_f1 = mean(micro_f1_model, na.rm = TRUE),
        exact_match = mean(exact_match_model, na.rm = TRUE),
        exact_core_match = mean(exact_core_match_model, na.rm = TRUE)
      )
    
    # Compute p-values from paired bootstrap difference distribution
    p_values <- tibble(
      micro_f1_p = compute_pvalue_paired(paired_bootstrap$micro_f1_diff),
      exact_match_p = compute_pvalue_paired(paired_bootstrap$exact_match_diff),
      exact_core_match_p = compute_pvalue_paired(paired_bootstrap$exact_core_match_diff)
    )
    
    # Print diagnostic info
    cat(paste0("  P-values: micro_f1=", round(p_values$micro_f1_p, 4), "\n"))
    
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
  
  # Create table data with 3 metrics (no fuzzy duplicates)
  table_data <- tibble(
    Metric = c("Micro F1 ↑", "Exact Match ↑", "Exact Core Match ↑")
  )
  
  # Add baseline column
  baseline_col_name <- "GLiNER"
  table_data <- table_data |>
    mutate(
      !!baseline_col_name := c(
        baseline_metrics$micro_f1,
        baseline_metrics$exact_match,
        baseline_metrics$exact_core_match
      )
    )
  
  # Add each model column
  for (i in seq_len(nrow(all_results))) {
    model_name <- all_results$model[i]
    model_vals <- all_results[i, c("micro_f1", "exact_match", "exact_core_match")]
    
    # Add column with values
    table_data <- table_data |>
      mutate(
        !!model_name := c(
          model_vals$micro_f1,
          model_vals$exact_match,
          model_vals$exact_core_match
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
    model_vals <- all_results[i, c("micro_f1", "exact_match", "exact_core_match")]
    model_pvals <- all_results[i, c("micro_f1_p", "exact_match_p", "exact_core_match_p")]
    
    baseline_vals <- c(
      baseline_metrics$micro_f1,
      baseline_metrics$exact_match,
      baseline_metrics$exact_core_match
    )
    
    # Determine which rows are significant AND better (higher is better for all location metrics)
    significant_and_better <- c(
      model_pvals$micro_f1_p < ALPHA & model_vals$micro_f1 > baseline_metrics$micro_f1,
      model_pvals$exact_match_p < ALPHA & model_vals$exact_match > baseline_metrics$exact_match,
      model_pvals$exact_core_match_p < ALPHA & model_vals$exact_core_match > baseline_metrics$exact_core_match
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
      baseline_micro_f1 = baseline_metrics$micro_f1,
      baseline_exact_match = baseline_metrics$exact_match,
      baseline_exact_core_match = baseline_metrics$exact_core_match,
      micro_f1_diff = micro_f1 - baseline_micro_f1,
      exact_match_diff = exact_match - baseline_exact_match,
      exact_core_match_diff = exact_core_match - baseline_exact_core_match,
      micro_f1_significant = micro_f1_p < ALPHA,
      exact_match_significant = exact_match_p < ALPHA,
      exact_core_match_significant = exact_core_match_p < ALPHA,
      micro_f1_better = micro_f1 > baseline_micro_f1,
      exact_match_better = exact_match > baseline_exact_match,
      exact_core_match_better = exact_core_match > baseline_exact_core_match
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

