bootstrap_multiclass_macro_f1_plot <- function(data,
                                               fraction_value = "25%",
                                               title = NULL,
                                               caption = NULL,
                                               xtitle = NULL,
                                               ytitle = NULL,
                                               color = "#8337A0",
                                               n_boot = 1000) {
  library(dplyr)
  library(ggplot2)
  library(purrr)
  library(rsample)
  library(forcats)
  
  df <- data |> filter(fraction_label == fraction_value)
  
  compute_macro_f1 <- function(truth, pred) {
    labels <- union(unique(truth), unique(pred))
    f1_scores <- map_dbl(labels, function(label) {
      tp <- sum(truth == label & pred == label)
      fp <- sum(truth != label & pred == label)
      fn <- sum(truth == label & pred != label)
      
      precision <- if ((tp + fp) > 0) tp / (tp + fp) else 0
      recall <- if ((tp + fn) > 0) tp / (tp + fn) else 0
      if ((precision + recall) > 0) 2 * precision * recall / (precision + recall) else 0
    })
    
    mean(f1_scores)
  }
  
  bootstrap_model_f1 <- function(df_model, n = n_boot) {
    bootstraps(df_model, times = n) |>
      mutate(f1 = map_dbl(splits, ~ {
        df_split <- assessment(.x)
        compute_macro_f1(df_split$true_label, df_split$pred_label)
      }))
  }
  
  f1_ci_df <- df |> 
    group_by(model_label) |> 
    group_split() |> 
    map_dfr(~ bootstrap_model_f1(.x) |> mutate(model_label = unique(.x$model_label)))
  
  f1_summary <- f1_ci_df |>
    group_by(model_label) |>
    summarise(
      f1_mean = mean(f1, na.rm = TRUE),
      f1_lower = quantile(f1, 0.025, na.rm = TRUE),
      f1_upper = quantile(f1, 0.975, na.rm = TRUE)
    ) |>
    mutate(model_label = fct_reorder(model_label, f1_mean))
  
  ggplot(f1_summary, aes(x = model_label, y = f1_mean)) +
    geom_point(size = 3, color = color) +
    geom_errorbar(aes(ymin = f1_lower, ymax = f1_upper), width = 0.2, color = color) +
    labs(
      title = title,
      y = ytitle,
      x = xtitle,
      caption = caption
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}
