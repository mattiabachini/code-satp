#!/usr/bin/env Rscript
# Geocoding: Model → Baseline distances
# - Boxplot across strategies
# - Summary table (median, mean, within 10km, within 50km)
#
# Run from papers/location-extraction

suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)  # for consistency with repo conventions
  library(gt)
  library(viridis)
  library(scales)
})

# Configuration (relative to papers/location-extraction)
RESULTS_DIR <- "results/geocoding"
OUTPUT_DIR  <- "data-viz/images"

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Helper to read one strategy file and return a standardized tibble
read_strategy <- function(filename, strategy_label) {
  path <- file.path(RESULTS_DIR, filename)
  if (!file.exists(path)) {
    stop(paste("File not found:", path))
  }
  df <- readr::read_csv(path, show_col_types = FALSE) |>
    dplyr::select(dist_model_baseline_km) |>
    dplyr::rename(distance_km = dist_model_baseline_km) |>
    dplyr::mutate(
      strategy = factor(
        strategy_label,
        levels = c("Strategy 1: Full Components",
                   "Strategy 2: Village Only",
                   "Strategy 3: Comma-Delimited")
      )
    ) |>
    dplyr::filter(!is.na(distance_km))
  df
}

cat("Loading geocoding results...\n")
s1 <- read_strategy("geocoding_strategy1_full_components.csv", "Strategy 1: Full Components")
s2 <- read_strategy("geocoding_strategy2_village_only.csv", "Strategy 2: Village Only")
s3 <- read_strategy("geocoding_strategy3_comma_delimited.csv", "Strategy 3: Comma-Delimited")

plot_data <- dplyr::bind_rows(s1, s2, s3)
cat("Rows per strategy:\n")
print(plot_data |>
        dplyr::count(strategy) |>
        dplyr::rename(n_examples = n))

# Boxplot: Model → Baseline distances by strategy
cat("Creating boxplot...\n")
p <- ggplot(plot_data, aes(x = strategy, y = distance_km, fill = strategy)) +
  geom_boxplot(outlier.alpha = 0.25, width = 0.7) +
  scale_fill_viridis_d(option = "D", end = 0.85, guide = "none") +
  scale_y_continuous(labels = label_number(accuracy = 0.1)) +
  theme_minimal() +
  labs(
    title = "Model → Manual Baseline Distance by Geocoding Strategy",
    x = NULL,
    y = "Distance (km)"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.text.x = element_text(angle = 15, hjust = 1),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    panel.grid.major.x = element_blank()
  )

boxplot_file <- file.path(OUTPUT_DIR, "geocoding_boxplot.png")
ggsave(
  boxplot_file,
  plot = p,
  width = 10, height = 6, dpi = 300
)
cat("Boxplot saved to:", boxplot_file, "\n")

# Summary table: median, mean, within 10km, within 50km
cat("Computing summary table...\n")
summary_tbl <- plot_data |>
  group_by(strategy) |>
  summarise(
    n = n(),
    median_km = median(distance_km, na.rm = TRUE),
    mean_km = mean(distance_km, na.rm = TRUE),
    within_10 = mean(distance_km <= 10, na.rm = TRUE),
    within_50 = mean(distance_km <= 50, na.rm = TRUE),
    .groups = "drop"
  ) |>
  mutate(
    median_km = round(median_km, 2),
    mean_km   = round(mean_km, 2),
    within_10 = scales::percent(within_10, accuracy = 0.1),
    within_50 = scales::percent(within_50, accuracy = 0.1)
  ) |>
  # reorder columns for display
  select(
    Strategy = strategy,
    `Median (km)` = median_km,
    `Mean (km)` = mean_km,
    `Within 10 km` = within_10,
    `Within 50 km` = within_50,
    `N` = n
  )

# Write CSV alongside PNG
summary_csv <- file.path(OUTPUT_DIR, "geocoding_summary_table.csv")
readr::write_csv(summary_tbl, summary_csv)
cat("Summary CSV saved to:", summary_csv, "\n")

# Build gt table
gt_table <- summary_tbl |>
  gt::gt() |>
  gt::tab_header(
    title = gt::md("Model → Manual Baseline Summary by Strategy")
  ) |>
  gt::fmt_number(
    columns = c(`Median (km)`, `Mean (km)`),
    decimals = 2
  ) |>
  gt::tab_options(
    table.background.color = "white",
    heading.background.color = "white"
  )

table_png <- file.path(OUTPUT_DIR, "geocoding_summary_table.png")
gt::gtsave(gt_table, table_png)
cat("Summary table PNG saved to:", table_png, "\n")

cat("Done.\n")


