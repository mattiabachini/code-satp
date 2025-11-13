# Bootstrap MAE Visualization for Seq2Seq Death Count Models
# Generates horizontal plot with side-by-side confidence intervals for
# MAE across all events and non-zero events
# Baseline: ConfliBERT-Poisson (appears first on left)

library(dplyr)
library(tidyr)
library(ggplot2)
library(purrr)
library(rsample)
library(forcats)
library(readr)
library(viridis)

# Configuration
# Paths are relative to papers/death-counts directory
# Script should be run from papers/death-counts directory
RESULTS_DIR <- "results/death-counts-seq2seq"
OUTPUT_DIR <- "data-viz/images"
N_BOOT <- 5000
set.seed(42)

# Model files to process
models <- c(
  "flan-t5-base",
  "flan-t5-large",
  "flan-t5-xl-lora",
  "indicbart",
  "mt5-base",
  "nt5-small"
)

# Function to compute MAE for all events
compute_mae_all <- function(truth, pred) {
  mean(abs(truth - pred), na.rm = TRUE)
}

# Function to compute MAE for non-zero events
compute_mae_nonzero <- function(truth, pred) {
  nonzero_idx <- truth > 0
  if (sum(nonzero_idx) == 0) return(NA)
  mean(abs(truth[nonzero_idx] - pred[nonzero_idx]), na.rm = TRUE)
}

# Function to bootstrap MAE for a model
bootstrap_model_mae <- function(df_model, n = N_BOOT) {
  bootstraps(df_model, times = n) |>
    mutate(
      mae_all = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_mae_all(df_split$true_label, df_split$prediction)
      }),
      mae_nonzero = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        compute_mae_nonzero(df_split$true_label, df_split$prediction)
      })
    )
}

# Load and process each model
all_results <- map_dfr(models, function(model_name) {
  # Construct file path - mt5-base uses combined file
  if (model_name == "mt5-base") {
    # Extract from combined predictions file
    csv_file <- file.path(RESULTS_DIR, "death_counts_predictions_combined.csv")
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
    
    # Prepare data frame with standardized column names
    df_model <- df |>
      select(true_label = true_label, prediction = !!sym(pred_col)) |>
      filter(!is.na(true_label), !is.na(prediction))
  } else {
    csv_file <- file.path(RESULTS_DIR, paste0("death_counts_", model_name, "_predictions.csv"))
    
    if (!file.exists(csv_file)) {
      warning(paste("File not found:", csv_file))
      return(NULL)
    }
    
    # Load data
    df <- read_csv(csv_file, show_col_types = FALSE)
    
    # Prepare data frame with standardized column names
    df_model <- df |>
      select(true_label = true_label, prediction = prediction) |>
      filter(!is.na(true_label), !is.na(prediction))
  }
  
  # Run bootstrap
  cat("Bootstrapping", model_name, "...\n")
  bootstrap_results <- bootstrap_model_mae(df_model, n = N_BOOT)
  
  # Add model label
  bootstrap_results |>
    mutate(model_label = model_name)
})

# Summarize bootstrap results
mae_summary <- all_results |>
  group_by(model_label) |>
  summarise(
    mae_all_mean = mean(mae_all, na.rm = TRUE),
    mae_all_lower = quantile(mae_all, 0.025, na.rm = TRUE),
    mae_all_upper = quantile(mae_all, 0.975, na.rm = TRUE),
    mae_nonzero_mean = mean(mae_nonzero, na.rm = TRUE),
    mae_nonzero_lower = quantile(mae_nonzero, 0.025, na.rm = TRUE),
    mae_nonzero_upper = quantile(mae_nonzero, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

# Reorder models by overall MAE (all events)
mae_summary <- mae_summary |>
  mutate(
    model_label = fct_reorder(model_label, mae_all_mean)
  )

# Reshape for plotting (side-by-side CIs)
mae_plot_data <- mae_summary |>
  pivot_longer(
    cols = c(mae_all_mean, mae_nonzero_mean),
    names_to = "metric_type",
    values_to = "mae_mean"
  ) |>
  mutate(
    metric_type = case_when(
      metric_type == "mae_all_mean" ~ "All Events",
      metric_type == "mae_nonzero_mean" ~ "Non-zero Events"
    ),
    mae_lower = case_when(
      metric_type == "All Events" ~ mae_all_lower,
      metric_type == "Non-zero Events" ~ mae_nonzero_lower
    ),
    mae_upper = case_when(
      metric_type == "All Events" ~ mae_all_upper,
      metric_type == "Non-zero Events" ~ mae_nonzero_upper
    )
  ) |>
  select(model_label, metric_type, mae_mean, mae_lower, mae_upper)

# Create plot with side-by-side CIs
# Model names on x-axis, MAE on y-axis
p <- ggplot(mae_plot_data, aes(x = model_label, y = mae_mean, color = metric_type)) +
  geom_point(size = 3, position = position_dodge(width = 0.5)) +
  geom_errorbar(
    aes(ymin = mae_lower, ymax = mae_upper),
    width = 0.2,
    position = position_dodge(width = 0.5)
  ) +
  scale_color_viridis_d(option = "turbo", name = "Metric Type") +
  scale_y_continuous(limits = c(0, 0.85)) +
  labs(
    title = "Bootstrap MAE CIs: Seq2Seq Models",
    x = NULL,  # Common x-axis label will be in combined plot
    y = "Mean Absolute Error",
    caption = NULL  # Common caption will be in combined plot
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  )

# Save plot
output_file <- file.path(OUTPUT_DIR, "bootstrap_mae_seq2seq.png")
ggsave(
  output_file,
  plot = p,
  width = 10,
  height = 6,
  dpi = 300
)

cat("Plot saved to:", output_file, "\n")

# Return plot object for use in combined plots
p

