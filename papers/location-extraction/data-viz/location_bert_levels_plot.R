# Location Extraction BERT Models: Fuzzy F1 by Administrative Level
# Faceted column chart showing how Fuzzy F1 declines across administrative levels

library(tidyverse)
library(jsonlite)
library(viridis)

# Define models to include (exclude baseline GLiNER)
models <- c(
  "deberta-v3",
  "muril",
  "spanbert",
  "xlm-roberta"
)

# Configuration
# Paths are relative to papers/location-extraction directory
# Script should be run from papers/location-extraction directory
RESULTS_DIR <- "results/location-extraction-bert"
OUTPUT_DIR <- "data-viz/images"

# Define level order
level_order <- c("state", "district", "village", "other_locations")
level_labels <- c("State", "District", "Village", "Other Areas")

# Read and process metrics
data_list <- list()

for (model in models) {
  # Read JSON file
  json_file <- file.path(RESULTS_DIR, paste0(model, "_metrics_fused.json"))
  metrics <- fromJSON(json_file)
  
  # Extract level fuzzy_f1 values
  levels_data <- metrics$levels
  
  # Create data frame
  df <- data.frame(
    model = model,
    level = names(levels_data),
    fuzzy_f1 = sapply(levels_data, function(x) x$fuzzy_f1),
    stringsAsFactors = FALSE
  )
  
  data_list[[model]] <- df
}

# Combine all data
plot_data <- bind_rows(data_list)

# Order levels and create better labels
plot_data <- plot_data %>%
  mutate(
    level = factor(level, levels = level_order, labels = level_labels),
    model_label = case_when(
      model == "deberta-v3" ~ "DeBERTa-v3",
      model == "muril" ~ "MuRIL",
      model == "spanbert" ~ "SpanBERT",
      model == "xlm-roberta" ~ "XLM-RoBERTa"
    ),
    model_label = factor(model_label, levels = c(
      "DeBERTa-v3", "MuRIL", "SpanBERT",
      "XLM-RoBERTa"
    ))
  )

# Create plot
p <- ggplot(plot_data, aes(x = level, y = fuzzy_f1)) +
  geom_col(fill = viridis(10, option = "D")[7], alpha = 0.9) +
  facet_wrap(~ model_label, nrow = 2, ncol = 3) +
  theme_minimal() +
  labs(
    title = "BERT Models: Fuzzy F1 Across Administrative Levels",
    x = "Administrative Level",
    y = "Fuzzy F1 Score"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 11, face = "bold"),
    panel.spacing = unit(1, "lines"),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  )

# Save plot
output_file <- file.path(OUTPUT_DIR, "location_bert_levels.png")
ggsave(
  output_file,
  plot = p,
  width = 12,
  height = 8,
  dpi = 300
)

cat("Plot saved to:", output_file, "\n")

