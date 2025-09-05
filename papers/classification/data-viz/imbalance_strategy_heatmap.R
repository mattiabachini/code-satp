# Imbalance Strategy Heatmap Function

imbalance_strategy_heatmap <- function(data, 
                                       strategy_col = "strategy",
                                       label_col = "label", 
                                       value_col = "f1",
                                       title = NULL,
                                       subtitle = NULL,
                                       palette = "cividis",
                                       strategy_labels = NULL) {
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(forcats)
  library(ggplot2)
  library(viridis)
  
  # Default strategy label mapping
  if (is.null(strategy_labels)) {
    strategy_labels <- c(
      "baseline" = "Baseline",
      "threshold_tuned" = "Threshold Tuning",
      "focal_tuned" = "Focal Loss",
      "class_weights_tuned" = "Class Weights",
      "weighted_sampler_tuned" = "Weighted Sampling",
      "augmentation_bt_tuned" = "Back Translation",
      "augmentation_t5_tuned" = "T5 Augmentation"
    )
  }
  
  # Create pivot table with strategies as rows and labels as columns
  pivot_data <- data |>
    select(all_of(c(strategy_col, label_col, value_col))) |>
    pivot_wider(names_from = all_of(label_col), values_from = all_of(value_col))
  
  # Convert back to long format for plotting but with formatted labels
  plot_data <- pivot_data |>
    pivot_longer(cols = -all_of(strategy_col), names_to = "label_raw", values_to = "f1_score") |>
    mutate(
      # Format label names to match Python formatting (replace _ with space, title case)
      label = .data$label_raw |>
        str_replace_all("_", " ") |>
        str_to_title(),
      strategy = !!sym(strategy_col),
      # Apply strategy label mapping
      strategy_display = ifelse(.data$strategy %in% names(strategy_labels),
                               strategy_labels[.data$strategy],
                               .data$strategy)
    )
  
  # Sort labels by their average F1 score across strategies (descending)
  label_order <- plot_data |>
    group_by(.data$label) |>
    summarise(mean_f1 = mean(.data$f1_score, na.rm = TRUE), .groups = "drop") |>
    arrange(desc(.data$mean_f1)) |>
    pull(.data$label)
  
  # Sort strategies by their average F1 score across labels (descending for display)
  # Note: We reverse the order because ggplot displays factor levels from bottom to top
  strategy_order <- plot_data |>
    group_by(.data$strategy_display) |>
    summarise(mean_f1 = mean(.data$f1_score, na.rm = TRUE), .groups = "drop") |>
    arrange(.data$mean_f1) |>  # ascending order so best appears at top
    pull(.data$strategy_display)
  
  # Apply factor ordering
  plot_data <- plot_data |>
    mutate(
      label = factor(.data$label, levels = label_order),
      strategy_display = factor(.data$strategy_display, levels = strategy_order)
    )
  
  # Create the heatmap
  p <- ggplot(plot_data, aes(x = .data$label, y = .data$strategy_display, fill = .data$f1_score)) +
    geom_tile(color = "gray90", linewidth = 0.5, alpha = 0.9) +
    geom_text(aes(label = sprintf("%.2f", .data$f1_score)), size = 3, color = "black") +
    scale_fill_viridis_c(option = palette, name = "F1 Score", begin = 0.1) +
    labs(
      title = title,
      subtitle = subtitle,
      x = "Label",
      y = "Strategy"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      axis.title.x = element_blank(),
      axis.title.y = element_blank()
    )
  
  return(p)
}
