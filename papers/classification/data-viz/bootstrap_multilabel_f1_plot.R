bootstrap_multilabel_f1_plot <- function(data,
                                         fraction_value = "25.0%",
                                         title = NULL,
                                         caption = NULL,
                                         xtitle = NULL,
                                         ytitle = NULL,
                                         color = "#8337A0",
                                         n_boot = 1000) {
  
  library(rsample)
  library(purrr)
  library(dplyr)
  library(forcats)
  library(ggplot2)
  library(tidyr)
  
  #Filter
  df_25 <- data |> 
    filter(fraction_label == fraction_value)
  
  # reshape
  
  df_long <- df_25 |>
    mutate(row_id = row_number()) |>
    pivot_longer(
      cols = starts_with("true_"),
      names_to = "label",
      names_prefix = "true_",
      values_to = "truth"
    ) |>
    select(row_id, model_label, label, truth) |>   
    
    left_join(
      df_25 |>
        mutate(row_id = row_number()) |>
        pivot_longer(
          cols = starts_with("pred_"),
          names_to = "label",
          names_prefix = "pred_",
          values_to = "pred"
        ) |>
        select(row_id, label, pred),                
      by = c("row_id", "label")
    )

  #Define Micro F1
  compute_micro_f1 <- function(df) {
    tp <- sum(df$truth == 1 & df$pred == 1)
    fp <- sum(df$truth == 0 & df$pred == 1)
    fn <- sum(df$truth == 1 & df$pred == 0)
    
    if ((2 * tp + fp + fn) == 0) return(NA)
    
    (2 * tp) / (2 * tp + fp + fn)
  }
  
  # Bootstrap function to compute micro-F1 for a given data frame
  bootstrap_micro_f1 <- function(df, n = n_boot) {
    bootstraps(df, times = n) |>
      mutate(f1 = map_dbl(splits, ~ compute_micro_f1(assessment(.x))))
  }
  
  # Run bootstrap per model and retain model_label
  f1_ci_df <- df_long |>
    group_by(model_label) |>
    group_split() |>
    map_dfr(~ bootstrap_micro_f1(.x) |> mutate(model_label = unique(.x$model_label)))
  
  # Summarize to get mean and 95% confidence intervals
  f1_summary <- f1_ci_df |>
    group_by(model_label) |>
    summarise(
      f1_mean = mean(f1, na.rm = TRUE),
      f1_lower = quantile(f1, 0.025, na.rm = TRUE),
      f1_upper = quantile(f1, 0.975, na.rm = TRUE)
    ) |>
    mutate(model_label = fct_reorder(model_label, f1_mean))
  
  f1_ci_plot <- ggplot(f1_summary, aes(x = model_label, y = f1_mean)) +
    geom_point(size = 3, color = "#8337A0") +
    geom_errorbar(aes(ymin = f1_lower, ymax = f1_upper), width = 0.2, color = "#8337A0") +
    labs(
      title = title,
      y = ytitle,
      x = xtitle,
      caption = caption
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  f1_ci_plot
}