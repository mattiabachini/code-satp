# Heatmap Function for Showing Per-Label Model Performance

per_label_f1_heatmap <- function(data,
                                 title = NULL,
                                 extra_rows = NULL,
                                 extra_model_scores = NULL) {
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
  
  # Append extra rows (e.g., LLM baselines) if provided
  if (!is.null(extra_rows)) {
    df_long <- bind_rows(df_long, extra_rows)
  }

  # Sort labels by mean F1 score
  label_order <- df_long |>
    group_by(label) |>
    summarise(mean_f1 = mean(f1_score, na.rm = TRUE)) |>
    arrange(desc(mean_f1)) |>
    pull(label)
  
  # Sort models by micro F1 score (optionally include extra model scores)
  base_model_scores <- df_filtered |>
    select(model_label, test_micro_avg_f1_score) |>
    distinct() |>
    rename(score = test_micro_avg_f1_score)

  if (is.null(extra_model_scores) && !is.null(extra_rows)) {
    extra_model_scores <- extra_rows |>
      group_by(model_label) |>
      summarise(score = mean(f1_score, na.rm = TRUE), .groups = "drop")
  }

  model_scores <- base_model_scores
  if (!is.null(extra_model_scores)) {
    model_scores <- bind_rows(model_scores, extra_model_scores) |>
      distinct(model_label, .keep_all = TRUE)
  }

  model_order <- model_scores |>
    arrange(score) |>
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
    geom_text(
      aes(label = sprintf("%.2f", f1_score)),
      size = 3,
      show.legend = FALSE
    ) +
    scale_fill_viridis_c(option = "cividis", name = "F1 Score", begin = .2, end = .95) +
    guides(
      fill = guide_colorbar(
        title.position = "top",
        title.hjust = 0.5,
        barwidth = unit(3, "cm"),
        barheight = unit(0.25, "cm")
      )
    ) +
    labs(
      title = title,
      x = NULL, y = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank(),
      legend.position = "bottom",
      legend.key.width = unit(1.2, "cm"),
      legend.key.height = unit(0.2, "cm"),
      legend.text = element_text(size = 8),
      legend.title = element_text(size = 9, margin = margin(b = 2))
    )
}
