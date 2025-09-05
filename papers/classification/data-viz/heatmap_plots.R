# Heatmap Function for Showing Per-Label Model Performance

per_label_f1_heatmap <- function(data, title = NULL) {
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(forcats)
  library(ggplot2)
  library(viridis)
  
  # Filter to only 100% training data
  df_filtered <- data |>
    filter(fraction_label == "100.0%")
  
  # Select label-level f1 score columns (exclude averages)
  label_cols <- df_filtered |>
    select(starts_with("test_")) |>
    select(contains("f1_score"), -contains("avg")) |>
    names()
  
  # Pivot to long format and clean label names
  df_long <- df_filtered |>
    select(model_label, all_of(label_cols)) |>
    pivot_longer(cols = -model_label, names_to = "label_raw", values_to = "f1_score") |>
    mutate(
      label = label_raw |>
        str_remove("^test_") |>
        str_remove("_f1_score$") |>
        str_replace_all("_", " ") |>
        str_to_title()
    )
  
  # Sort labels by mean F1 score
  label_order <- df_long |>
    group_by(label) |>
    summarise(mean_f1 = mean(f1_score, na.rm = TRUE)) |>
    arrange(desc(mean_f1)) |>
    pull(label)
  
  # Sort models by micro F1 score
  model_order <- df_filtered |>
    select(model_label, test_micro_avg_f1_score) |>
    arrange(test_micro_avg_f1_score) |>
    pull(model_label)
  
  # Apply factor ordering
  df_long <- df_long |>
    mutate(
      label = factor(label, levels = label_order),
      model_label = factor(model_label, levels = model_order)
    )
  
  # Plot
  ggplot(df_long, aes(x = label, y = model_label, fill = f1_score)) +
    geom_tile(color = "gray90", alpha = 0.9) +
    geom_text(aes(label = sprintf("%.2f", f1_score)), size = 3) +
    scale_fill_viridis_c(option = "cividis", name = "F1 Score", begin = .1) +
    labs(
      title = title,
      x = NULL, y = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )
}
