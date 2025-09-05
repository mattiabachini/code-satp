# Precision-Recall Plots for Rare Labels
# Mirrors Python's create_rare_labels_comparison_plots function

# Helper function to extract precision, recall, and F1 from predictions data
# Updated to handle hybrid data structure where different strategies use different probability columns
extract_pr_metrics_from_predictions <- function(predictions_df, 
                                                label_cols, 
                                                strategy_col = "strategy") {
  library(dplyr)
  library(purrr)
  
  results <- list()
  
  strategies <- unique(predictions_df[[strategy_col]])
  
  for (strategy in strategies) {
    strategy_data <- predictions_df[predictions_df[[strategy_col]] == strategy, ]
    
    for (label in label_cols) {
      true_col <- paste0("true_", label)
      pred_col <- paste0("pred_", label)
      
      if (true_col %in% names(strategy_data) && pred_col %in% names(strategy_data)) {
        y_true <- strategy_data[[true_col]]
        y_pred <- strategy_data[[pred_col]]

        # Convert to binary numeric (matches Python implementation)
        to_binary_numeric <- function(x) {
          if (is.logical(x)) {
            return(as.integer(x))
          }
          if (is.factor(x) || is.character(x)) {
            lx <- tolower(as.character(x))
            return(as.integer(lx %in% c("1", "true", "t", "yes")))
          }
          return(as.numeric(x))  # Ensure numeric output
        }
        y_true <- to_binary_numeric(y_true)
        y_pred <- to_binary_numeric(y_pred)

        # Calculate metrics directly from binary predictions (like Python sklearn)
        tp <- sum(y_true == 1 & y_pred == 1, na.rm = TRUE)
        fp <- sum(y_true == 0 & y_pred == 1, na.rm = TRUE)
        fn <- sum(y_true == 1 & y_pred == 0, na.rm = TRUE)
        
        precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
        recall <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
        f1 <- ifelse(precision + recall == 0, 0, 2 * (precision * recall) / (precision + recall))
        
        results <- append(results, list(data.frame(
          strategy = strategy,
          label = label,
          precision = precision,
          recall = recall,
          f1 = f1
        )))
      }
    }
  }
  
  bind_rows(results)
}

# Function to create iso-F1 contour lines
create_iso_f1_contours <- function(f1_levels = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) {
  library(dplyr)
  library(purrr)
  
  # Create smoother contour lines using the mathematical formula for iso-F1 curves
  # For a given F1 score, precision = F1 * recall / (2 * recall - F1)
  contour_data <- map_dfr(f1_levels, function(f1_level) {
    # Create recall values, avoiding the asymptote at recall = f1_level/2
    recall_min <- max(f1_level/2 + 0.001, 0.001)  # Avoid asymptote
    recall_max <- 1.0
    recall_vals <- seq(recall_min, recall_max, length.out = 200)  # More points for smoother curves
    
    # Calculate corresponding precision values
    precision_vals <- f1_level * recall_vals / (2 * recall_vals - f1_level)
    
    # Filter valid precision values (between 0 and 1)
    valid_idx <- precision_vals > 0 & precision_vals <= 1 & !is.infinite(precision_vals)
    
    data.frame(
      recall = recall_vals[valid_idx],
      precision = precision_vals[valid_idx],
      f1_level = f1_level
    )
  })
  
  return(contour_data)
}

# Function to create a single precision-recall plot for one rare label
create_label_pr_plot <- function(metrics_df, 
                                 label_name,
                                 title = NULL,
                                 show_iso_f1 = TRUE,
                                 f1_levels = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                 strategy_labels = NULL) {
  library(ggplot2)
  library(dplyr)
  library(stringr)
  
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
  
  # Filter data for the specific label
  plot_data <- metrics_df |> 
    filter(.data$label == label_name) |>
    mutate(
      # Apply strategy label mapping
      strategy_display = ifelse(.data$strategy %in% names(strategy_labels),
                               strategy_labels[.data$strategy],
                               .data$strategy)
    )
  
  if (nrow(plot_data) == 0) {
    warning(paste("No data found for label:", label_name))
    return(NULL)
  }
  
  # Create base plot
  p <- ggplot(plot_data, aes(x = .data$recall, y = .data$precision, color = .data$strategy_display)) +
    geom_point(size = 3, alpha = 0.7, stroke = 0.5, position = position_jitter(width = 0.012, height = 0.012)) +
    scale_color_viridis_d(option = "turbo", name = "Strategy", end = 0.9) +
    xlim(-0.02, 1.02) + ylim(-0.02, 1.02) +
    labs(
      x = "Recall",
      y = "Precision",
      title = if (is.null(title)) {
        # Custom label formatting for specific cases
        if (label_name == "non_maoist_armed_group") {
          "Non-Maoist Armed Group"
        } else {
          str_to_title(str_replace_all(label_name, "_", " "))
        }
      } else title
    ) +
    theme_minimal(base_size = 10) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "plain", hjust = 0.5, size = 11, color = "gray40"),
      legend.position = "right"
    )
  
  # Add diagonal line (where precision = recall)
  p <- p + geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 0.5, color = "black")
  
  # Add iso-F1 contours if requested
  if (show_iso_f1) {
    contour_data <- create_iso_f1_contours(f1_levels)
    
    # Add contour lines
    for (f1_level in f1_levels) {
      level_data <- contour_data |> filter(f1_level == !!f1_level)
      if (nrow(level_data) > 0) {
        p <- p + geom_path(data = level_data, 
                          aes(x = .data$recall, y = .data$precision), 
                          color = "gray70", 
                          linetype = "dashed", 
                          alpha = 0.6,
                          linewidth = 0.3,  # Thinner lines
                          inherit.aes = FALSE)
      }
    }
    
    # Add F1 contour labels at strategic points
    label_positions <- data.frame(
      recall = c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
      f1_level = c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    ) |>
      mutate(precision = .data$f1_level * .data$recall / (2 * .data$recall - .data$f1_level)) |>
      filter(.data$precision > 0 & .data$precision <= 1)
    
    if (nrow(label_positions) > 0) {
      p <- p + geom_text(data = label_positions,
                        aes(x = .data$recall, y = .data$precision, label = paste0("F1=", .data$f1_level)),
                        color = "#666666", size = 2.5, inherit.aes = FALSE)
    }
  }
  
  
  return(p)
}

# Main function to create combined precision-recall plots for rare labels
# Mirrors Python's create_rare_labels_comparison_plots function
create_rare_labels_comparison_plots <- function(predictions_df,
                                               task_name,
                                               rare_labels,
                                               return_details = FALSE,
                                               strategy_labels = NULL) {
  library(ggplot2)
  library(dplyr)
  library(stringr)
  library(patchwork)
  
  # Extract label columns that exist in the predictions data
  available_label_cols <- rare_labels[sapply(rare_labels, function(label) {
    paste0("true_", label) %in% names(predictions_df)
  })]
  
  if (length(available_label_cols) == 0) {
    warning("No matching rare labels found in predictions data")
    return(NULL)
  }
  
  # Extract precision-recall metrics
  metrics_df <- extract_pr_metrics_from_predictions(predictions_df, available_label_cols)
  
  
  if (nrow(metrics_df) == 0) {
    warning("No metrics could be extracted from predictions data")
    return(NULL)
  }
  
  # Create individual plots as objects only (no printing)
  individual_plots <- list()
  plot_list <- list()
  
  for (i in seq_along(available_label_cols)) {
    label <- available_label_cols[i]
    
    # Create individual plot without legend (will be added to combined plot)
    p <- create_label_pr_plot(
      metrics_df, 
      label,
      show_iso_f1 = TRUE,
      strategy_labels = strategy_labels
    )
    
    if (!is.null(p)) {
      plot_list[[i]] <- p
      individual_plots[[label]] <- p
    }
  }
  
  # Create combined plot using patchwork
  if (length(plot_list) > 0) {
    # Determine layout
    n_plots <- length(plot_list)
    if (n_plots <= 2) {
      ncol <- n_plots
      nrow <- 1
    } else if (n_plots <= 4) {
      ncol <- 2
      nrow <- 2
    } else if (n_plots <= 6) {
      ncol <- 3
      nrow <- 2
    } else {
      ncol <- 3
      nrow <- ceiling(n_plots / 3)
    }
    
    # Combine plots with patchwork using built-in axis collection
    combined_plot <- wrap_plots(plot_list, ncol = ncol, nrow = nrow) +
      plot_layout(
        guides = "collect",
        axis_titles = "collect"
      ) &
      theme(legend.position = "right")
    if (isTRUE(return_details)) {
      return(list(
        combined_plot = combined_plot,
        individual_plots = individual_plots,
        metrics_df = metrics_df
      ))
    } else {
      return(combined_plot)
    }
  } else {
    warning("No plots could be created")
    return(NULL)
  }
}

# Note: This function expects predictions data with columns:
# - 'strategy': Strategy name (e.g., 'baseline', 'threshold_tuned')
# - 'true_{label}': True binary labels (0/1) for each label
# - 'pred_{label}': Predicted binary labels (0/1) for each label

# Utility function to create a simple precision-recall scatter plot
create_pr_scatter_plot <- function(metrics_df, 
                                  rare_labels = NULL,
                                  title = "Precision-Recall Scatter Plot",
                                  show_iso_f1 = TRUE,
                                  strategy_labels = NULL) {
  library(ggplot2)
  library(dplyr)
  
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
  
  # Filter for rare labels if specified
  if (!is.null(rare_labels)) {
    plot_data <- metrics_df |> filter(.data$label %in% rare_labels)
  } else {
    plot_data <- metrics_df
  }
  
  if (nrow(plot_data) == 0) {
    warning("No data to plot after filtering")
    return(NULL)
  }
  
  # Apply strategy label mapping
  plot_data <- plot_data |>
    mutate(
      strategy_display = ifelse(.data$strategy %in% names(strategy_labels),
                               strategy_labels[.data$strategy],
                               .data$strategy)
    )
  
  # Create base plot
  p <- ggplot(plot_data, aes(x = .data$recall, y = .data$precision, color = .data$strategy_display, shape = .data$label)) +
    geom_point(size = 3, alpha = 0.7, position = position_jitter(width = 0.012, height = 0.012)) +
    scale_color_viridis_d(option = "turbo", name = "Strategy", end = 0.9) +
    scale_shape_manual(values = c(16, 17, 15, 18, 19, 8, 11, 12), name = "Label") +
    xlim(-0.02, 1.02) + ylim(-0.02, 1.02) +
    labs(
      x = "Recall",
      y = "Precision", 
      title = title
    ) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0.5),
      legend.position = "right"
    )
  
  # Add diagonal line
  p <- p + geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 0.5, color = "black")
  
  # Add iso-F1 contours if requested
  if (show_iso_f1) {
    contour_data <- create_iso_f1_contours()
    
    # Add contour lines for major F1 levels
    major_f1_levels <- c(0.2, 0.4, 0.6, 0.8)
    for (f1_level in major_f1_levels) {
      level_data <- contour_data |> filter(f1_level == !!f1_level)
      if (nrow(level_data) > 0) {
        p <- p + geom_path(data = level_data,
                          aes(x = .data$recall, y = .data$precision),
                          color = "gray70",
                          linetype = "dashed",
                          alpha = 0.6,
                          linewidth = 0.3,  # Thinner lines
                          inherit.aes = FALSE)
      }
    }
  }
  
  return(p)
}
