# Function to show per label support

per_label_support_barplot <- function(data, title = NULL, label_levels = NULL) {
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(forcats)
  library(ggplot2)
  
  df_filtered <- data |>
    filter(fraction_label == "100.0%")
  
  # Select and reshape support columns
  support_cols <- df_filtered |>
    select(starts_with("test_")) |>
    select(ends_with("_support"), -contains("avg")) |>
    names()
  
  df_support <- df_filtered |>
    select(model_label, all_of(support_cols)) |>
    slice(1) |>  # assuming support values are the same across models
    pivot_longer(-model_label, names_to = "label_raw", values_to = "support") |>
    mutate(
      label = label_raw |>
        str_remove("^test_") |>
        str_remove("_support$") |>
        str_replace_all("_", " ") |>
        str_to_title()
    )
  
  if (!is.null(label_levels)) {
    df_support <- df_support |>
      mutate(label = factor(label, levels = label_levels))
  }
  
  ggplot(df_support, aes(x = label, y = support)) +
    geom_col(fill = "#00204C") +
    labs(
      x = NULL,
      y = "Number of Instances",
      title = title
    ) +
    theme_minimal(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )
}
