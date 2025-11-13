# Death Counts Seq2Seq Models: MAE by Bin
# Faceted column chart showing how MAE increases across count bins

library(tidyverse)
library(jsonlite)
library(viridis)

# Define models to include (exclude baseline conflibert-poisson)
models <- c(
  "flan-t5-base",
  "flan-t5-large", 
  "flan-t5-xl-lora",
  "indicbart",
  "mt5-base",
  "nt5-small"
)

# Configuration
# Paths are relative to papers/death-counts directory
# Script should be run from papers/death-counts directory
RESULTS_DIR <- "results/death-counts-seq2seq"
OUTPUT_DIR <- "data-viz/images"

# Define bin order
bin_order <- c("0", "1", "2", "3-5", "6+")

# Read and process metrics
data_list <- list()

for (model in models) {
  # Read JSON file
  json_file <- file.path(RESULTS_DIR, paste0("death_counts_", model, "_metrics.json"))
  metrics <- fromJSON(json_file)
  
  # Extract bin MAE values
  bins_data <- metrics$bins
  
  # Create data frame
  df <- data.frame(
    model = model,
    bin = names(bins_data),
    mae = sapply(bins_data, function(x) x$mae),
    stringsAsFactors = FALSE
  )
  
  data_list[[model]] <- df
}

# Combine all data
plot_data <- bind_rows(data_list)

# Order bins and create better labels
plot_data <- plot_data %>%
  mutate(
    bin = factor(bin, levels = bin_order),
    model_label = case_when(
      model == "flan-t5-base" ~ "FLAN-T5 Base",
      model == "flan-t5-large" ~ "FLAN-T5 Large",
      model == "flan-t5-xl-lora" ~ "FLAN-T5 XL (LoRA)",
      model == "indicbart" ~ "IndicBART",
      model == "mt5-base" ~ "mT5 Base",
      model == "nt5-small" ~ "nT5 Small"
    ),
    model_label = factor(model_label, levels = c(
      "FLAN-T5 Base", "FLAN-T5 Large", "FLAN-T5 XL (LoRA)",
      "IndicBART", "mT5 Base", "nT5 Small"
    ))
  )

# Create plot
p <- ggplot(plot_data, aes(x = bin, y = mae)) +
  geom_col(fill = viridis(10, option = "D")[4], alpha = 0.9) +
  facet_wrap(~ model_label, nrow = 2, ncol = 3) +
  theme_minimal() +
  labs(
    title = "Seq2Seq Models: MAE Across Death Count Bins",
    x = "Death Count Bin",
    y = "Mean Absolute Error (MAE)"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    strip.text = element_text(size = 11, face = "bold"),
    panel.spacing = unit(1, "lines")
  )

# Save plot
output_file <- file.path(OUTPUT_DIR, "death_counts_seq2seq_bins.png")
ggsave(
  output_file,
  plot = p,
  width = 12,
  height = 8,
  dpi = 300
)

cat("Plot saved to:", output_file, "\n")

