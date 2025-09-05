# Classification Metrics Visualization Functions
# Purpose: Functions for visualizing confusion matrices, ROC curves, and other classification metrics

#' Plot Confusion Matrix
#' 
#' Creates a heatmap visualization of a confusion matrix with optional normalization
#'
#' @param conf_matrix A confusion matrix (table or matrix object)
#' @param normalize Whether to normalize the confusion matrix (default: TRUE)
#' @param title Plot title
#' @param palette Color palette to use (options: "blues" (default), "viridis", "magma", "inferno", "plasma", "cividis", "turbo", "reds", "YlGnBu")
#' @param begin Starting point in the color range [0,1] (default: 0)
#' @param end End point in the color range [0,1] (default: 1)
#' @param text_size Size of text annotations in cells
#' @return A ggplot2 object
#'
#' @examples
#' # Example usage:
#' # cm <- table(predicted_labels, true_labels)
#' # plot_confusion_matrix(cm, normalize = TRUE)
#'
plot_confusion_matrix <- function(conf_matrix, 
                                 normalize = TRUE,
                                 title = "Confusion Matrix",
                                 palette = "blues",
                                 begin = 0,
                                 end = 1,
                                 text_size = 4) {
  
  # Load required packages
  require(ggplot2)
  require(dplyr)
  require(tidyr)
  
  # Convert to matrix if it's a table
  if (class(conf_matrix)[1] == "table") {
    conf_matrix <- as.matrix(conf_matrix)
  }
  
  # Get class names
  class_names <- rownames(conf_matrix)
  if (is.null(class_names)) {
    class_names <- 1:nrow(conf_matrix)
  }
  
  # Normalize if requested
  if (normalize) {
    # Normalize by row (true class)
    conf_matrix_normalized <- sweep(conf_matrix, 1, rowSums(conf_matrix), "/")
    
    # Create data frame for plotting
    plot_data <- as.data.frame(conf_matrix_normalized) %>%
      tibble::rownames_to_column("TrueClass") %>%
      pivot_longer(-TrueClass, names_to = "PredictedClass", values_to = "Proportion")
    
    # Add raw counts
    raw_counts <- as.data.frame(conf_matrix) %>%
      tibble::rownames_to_column("TrueClass") %>%
      pivot_longer(-TrueClass, names_to = "PredictedClass", values_to = "Count")
    
    # Merge normalized and raw counts
    plot_data <- plot_data %>%
      left_join(raw_counts, by = c("TrueClass", "PredictedClass")) %>%
      mutate(Label = sprintf("%.1f%%\n(%d)", Proportion * 100, Count))
    
    fill_label <- "Proportion"
    
  } else {
    # Use raw counts
    plot_data <- as.data.frame(conf_matrix) %>%
      tibble::rownames_to_column("TrueClass") %>%
      pivot_longer(-TrueClass, names_to = "PredictedClass", values_to = "Count") %>%
      mutate(Label = as.character(Count))
    
    fill_label <- "Count"
  }
  
  # Ensure classes are factors with the right order
  plot_data$TrueClass <- factor(plot_data$TrueClass, levels = class_names)
  plot_data$PredictedClass <- factor(plot_data$PredictedClass, levels = class_names)
  
  # Create the plot
  p <- ggplot(plot_data, 
              aes(x = PredictedClass, y = TrueClass, 
                  fill = if(normalize) Proportion else Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Label), size = text_size) +
    
    # Labels
    labs(
      title = title,
      x = "Predicted Class",
      y = "True Class",
      fill = fill_label
    ) +
    
    # Theme
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.title = element_text(face = "bold")
    )
  
  # Apply color palette based on selection
  if (palette %in% c("viridis", "magma", "inferno", "plasma", "cividis", "turbo")) {
    # Use viridis palettes
    p <- p + scale_fill_viridis_c(option = palette, begin = begin, end = end)
  } else if (palette == "blues") {
    p <- p + scale_fill_distiller(palette = "Blues", direction = 1)
  } else if (palette == "reds") {
    p <- p + scale_fill_distiller(palette = "Reds", direction = 1)
  } else if (palette == "YlGnBu") {
    p <- p + scale_fill_distiller(palette = "YlGnBu", direction = 1)
  }
  
  return(p)
}

#' Plot ROC Curve
#' 
#' Creates a Receiver Operating Characteristic (ROC) curve for binary classification
#'
#' @param true_labels Vector of true class labels (factor or numeric 0/1)
#' @param pred_probs Vector of predicted probabilities for the positive class
#' @param title Plot title
#' @param add_auc Whether to add AUC to the plot title (default: TRUE)
#' @param line_color Color of the ROC curve
#' @param line_size Width of the line
#' @return A ggplot2 object
#'
#' @examples
#' # Example usage:
#' # true_labels <- factor(c(0, 1, 1, 0, 1))
#' # pred_probs <- c(0.1, 0.9, 0.8, 0.3, 0.7)
#' # plot_roc_curve(true_labels, pred_probs)
#'
plot_roc_curve <- function(true_labels, 
                          pred_probs,
                          title = "ROC Curve",
                          add_auc = TRUE,
                          line_color = "steelblue",
                          line_size = 1.5) {
  
  # Load required packages
  require(ggplot2)
  require(pROC)
  
  # Calculate ROC curve
  roc_obj <- pROC::roc(true_labels, pred_probs)
  auc_value <- round(as.numeric(pROC::auc(roc_obj)), 3)
  
  # Extract data for plotting
  roc_data <- data.frame(
    specificity = roc_obj$specificities,
    sensitivity = roc_obj$sensitivities
  )
  
  # Adjust title if AUC should be included
  if (add_auc) {
    title <- paste0(title, " (AUC = ", auc_value, ")")
  }
  
  # Create the plot
  p <- ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity)) +
    geom_line(color = line_color, size = line_size) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40", alpha = 0.8) +
    
    # Labels
    labs(
      title = title,
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)"
    ) +
    
    # Theme and coordinates
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12)
    ) +
    coord_equal() +  # Equal aspect ratio
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1))
  
  return(p)
}

#' Plot Precision-Recall Curve
#' 
#' Creates a precision-recall curve for binary classification
#'
#' @param true_labels Vector of true class labels (factor or numeric 0/1)
#' @param pred_probs Vector of predicted probabilities for the positive class
#' @param title Plot title
#' @param add_auc Whether to add AUC to the plot title (default: TRUE)
#' @param line_color Color of the PR curve
#' @param line_size Width of the line
#' @return A ggplot2 object
#'
plot_precision_recall_curve <- function(true_labels, 
                                       pred_probs,
                                       title = "Precision-Recall Curve",
                                       add_auc = TRUE,
                                       line_color = "darkred",
                                       line_size = 1.5) {
  
  # Load required packages
  require(ggplot2)
  require(PRROC)
  
  # Ensure true_labels are 0/1
  if (is.factor(true_labels)) {
    true_labels_binary <- as.numeric(true_labels) - 1
  } else {
    true_labels_binary <- true_labels
  }
  
  # Calculate precision-recall curve
  # PRROC expects separate vectors for positive and negative class scores
  pos_scores <- pred_probs[true_labels_binary == 1]
  neg_scores <- pred_probs[true_labels_binary == 0]
  
  pr_curve <- PRROC::pr.curve(
    scores.class0 = pos_scores,
    scores.class1 = neg_scores,
    curve = TRUE
  )
  
  # Extract PR curve data
  pr_data <- data.frame(
    recall = pr_curve$curve[, 1],
    precision = pr_curve$curve[, 2]
  )
  
  # Calculate AUC for PR curve
  pr_auc <- round(pr_curve$auc.integral, 3)
  
  # Adjust title if AUC should be included
  if (add_auc) {
    title <- paste0(title, " (AUC = ", pr_auc, ")")
  }
  
  # Create the plot
  p <- ggplot(pr_data, aes(x = recall, y = precision)) +
    geom_line(color = line_color, size = line_size) +
    
    # Labels
    labs(
      title = title,
      x = "Recall",
      y = "Precision"
    ) +
    
    # Theme and coordinates
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12)
    ) +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1))
  
  return(p)
}

#' Plot Multiple ROC Curves
#' 
#' Creates a plot with multiple ROC curves for comparing models
#'
#' @param true_labels Vector of true class labels
#' @param pred_probs_list List of predicted probability vectors (one per model)
#' @param model_names Vector of model names (for legend)
#' @param title Plot title
#' @param add_auc Whether to add AUC to the legend (default: TRUE)
#' @param palette Viridis color palette option: "viridis" (default), "magma", "inferno", "plasma", "cividis", "turbo" 
#' @param begin Starting point in the color range [0,1] (default: 0)
#' @param end End point in the color range [0,1] (default: 1)
#' @param line_types Vector of line types for the ROC curves
#' @param line_size Width of the lines
#' @return A ggplot2 object
#'
plot_multiple_roc_curves <- function(true_labels, 
                                    pred_probs_list,
                                    model_names = NULL,
                                    title = "ROC Curve Comparison",
                                    add_auc = TRUE,
                                    palette = "viridis",
                                    begin = 0,
                                    end = 1,
                                    line_types = NULL,
                                    line_size = 1.2) {
  
  # Load required packages
  require(ggplot2)
  require(pROC)
  require(dplyr)
  require(purrr)
  
  # Set default model names if not provided
  if (is.null(model_names)) {
    model_names <- paste("Model", 1:length(pred_probs_list))
  }
  
  # Calculate ROC curves for each model
  roc_list <- map(pred_probs_list, ~ pROC::roc(true_labels, .x))
  auc_values <- map_dbl(roc_list, ~ round(as.numeric(pROC::auc(.x)), 3))
  
  # Create combined data frame for plotting
  roc_data_list <- map2(roc_list, 1:length(roc_list), ~ {
    data.frame(
      specificity = .x$specificities,
      sensitivity = .x$sensitivities,
      model = if (add_auc) {
        paste0(model_names[.y], " (AUC = ", auc_values[.y], ")")
      } else {
        model_names[.y]
      }
    )
  })
  
  roc_data <- bind_rows(roc_data_list)
  
  # Set line types if not provided
  if (is.null(line_types)) {
    line_types <- rep("solid", length(pred_probs_list))
  }
  
  # Create the plot
  p <- ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity, color = model, linetype = model)) +
    geom_line(size = line_size) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40", alpha = 0.8) +
    
    # Labels
    labs(
      title = title,
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)",
      color = "Model",
      linetype = "Model"
    ) +
    
    # Theme and coordinates
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      legend.position = "bottom",
      legend.title = element_text(face = "bold")
    ) +
    coord_equal() +  # Equal aspect ratio
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    
    # Apply color palette from viridis
    scale_color_viridis_d(option = palette, begin = begin, end = end) +
    
    scale_linetype_manual(values = line_types)
  
  return(p)
} 