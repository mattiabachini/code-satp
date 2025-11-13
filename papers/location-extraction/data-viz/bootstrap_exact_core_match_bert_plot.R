# Bootstrap Exact Core Match Visualization for BERT Span NER Location Extraction Models
# Exact core match = state + district + village all correct
# Excludes GLiNER

library(dplyr)
library(tidyr)
library(ggplot2)
library(purrr)
library(rsample)
library(forcats)
library(readr)
library(viridis)
library(jsonlite)
library(stringr)

# Configuration
# Paths are relative to papers/location-extraction directory
# Script should be run from papers/location-extraction directory
RESULTS_DIR <- "results/location-extraction-bert"
OUTPUT_DIR <- "data-viz/images"
N_BOOT <- 5000
set.seed(42)  # Set seed for reproducibility

# Model files to process (excluding GLiNER)
models <- c(
  "confliBERT",
  "deberta-v3",
  "muril",
  "spanbert",
  "xlm-roberta"
)

# Function to parse structured location string
parse_location <- function(loc_str) {
  if (is.na(loc_str) || loc_str == "") {
    return(list(state = NA, district = NA, village = NA))
  }
  
  # Extract fields using regex
  state <- str_extract(loc_str, "(?<=state: )[^,]+") |> str_trim() |> tolower()
  district <- str_extract(loc_str, "(?<=district: )[^,]+") |> str_trim() |> tolower()
  village <- str_extract(loc_str, "(?<=village: )[^,]+") |> str_trim() |> tolower()
  
  # Handle empty strings as NA
  if (!is.na(state) && state == "") state <- NA
  if (!is.na(district) && district == "") district <- NA
  if (!is.na(village) && village == "") village <- NA
  
  return(list(state = state, district = district, village = village))
}

# Function to compute exact core match (state + district + village)
compute_exact_core_match <- function(true_str, pred_str) {
  true_loc <- parse_location(true_str)
  pred_loc <- parse_location(pred_str)
  
  # Check if all three fields match (case-insensitive, already lowercased)
  state_match <- identical(true_loc$state, pred_loc$state)
  district_match <- identical(true_loc$district, pred_loc$district)
  village_match <- identical(true_loc$village, pred_loc$village)
  
  # All three must match
  if (state_match && district_match && village_match) {
    return(1)
  } else {
    return(0)
  }
}

# Function to bootstrap exact core match for a model
bootstrap_model_core_match <- function(df_model, n = N_BOOT) {
  bootstraps(df_model, times = n) |>
    mutate(
      core_match_rate = map_dbl(splits, ~ {
        df_split <- analysis(.x)
        mean(df_split$exact_core_match, na.rm = TRUE)
      })
    )
}

# Load and process each model
all_results <- map_dfr(models, function(model_name) {
  # Construct file path - BERT models use different naming
  csv_file <- file.path(RESULTS_DIR, paste0(model_name, "_predictions.csv"))
  
  if (!file.exists(csv_file)) {
    warning(paste("File not found:", csv_file))
    return(NULL)
  }
  
  # Load data
  df <- read_csv(csv_file, show_col_types = FALSE)
  
  # Prepare data frame with standardized column names
  df_model <- df |>
    select(true_label = ground_truth, prediction = prediction) |>
    filter(!is.na(true_label), !is.na(prediction)) |>
    mutate(
      exact_core_match = map2_dbl(true_label, prediction, compute_exact_core_match)
    )
  
  # Compute exact core match rate for validation
  computed_rate <- mean(df_model$exact_core_match, na.rm = TRUE) * 100
  
  # Validate against metrics JSON
  metrics_file <- file.path(RESULTS_DIR, paste0(model_name, "_metrics_fused.json"))
  if (file.exists(metrics_file)) {
    metrics <- fromJSON(metrics_file)
    json_rate <- metrics$overall$exact_core_match
    
    diff <- abs(computed_rate - json_rate)
    if (diff > 1.0) {
      warning(paste("Validation warning for", model_name, 
                   ": Computed rate =", round(computed_rate, 2),
                   "%, JSON rate =", round(json_rate, 2), "%",
                   "(diff =", round(diff, 2), "%)"))
    } else {
      cat("✓", model_name, ": Computed =", round(computed_rate, 2), 
          "%, JSON =", round(json_rate, 2), "%\n")
    }
  }
  
  # Run bootstrap
  cat("Bootstrapping", model_name, "...\n")
  bootstrap_results <- bootstrap_model_core_match(df_model, n = N_BOOT)
  
  # Add model label
  bootstrap_results |>
    mutate(model_label = model_name)
})

# Summarize bootstrap results
core_match_summary <- all_results |>
  group_by(model_label) |>
  summarise(
    core_match_mean = mean(core_match_rate, na.rm = TRUE) * 100,  # Convert to percentage
    core_match_lower = quantile(core_match_rate, 0.025, na.rm = TRUE) * 100,
    core_match_upper = quantile(core_match_rate, 0.975, na.rm = TRUE) * 100,
    .groups = "drop"
  ) |>
  # Reorder models by core match rate
  mutate(
    model_label = fct_reorder(model_label, core_match_mean)
  )

# Create plot
# Get red color from end of viridis turbo palette
turbo_red <- viridis(100, option = "turbo")[100]

p <- ggplot(core_match_summary, aes(x = model_label, y = core_match_mean)) +
  geom_point(size = 2, color = turbo_red) +
  geom_errorbar(
    aes(ymin = core_match_lower, ymax = core_match_upper),
    width = 0.15,
    linewidth = 0.5,
    color = turbo_red
  ) +
  scale_y_continuous(limits = c(10, 85)) +
  labs(
    title = "Bootstrap Exact Core Match: BERT Span NER Models",
    x = NULL,  # Common x-axis label will be in combined plot
    y = "Exact Core Match (%)",
    caption = NULL  # Common caption will be in combined plot
  ) +
  theme_minimal() +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    axis.text.x = element_text(hjust = 0.5),  # No rotation
    legend.position = "none"
  )

# Save plot
output_file <- file.path(OUTPUT_DIR, "bootstrap_exact_core_match_bert.png")
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

