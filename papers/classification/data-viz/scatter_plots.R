# Accuracy Versus Speed Scatter Plot

scatter_plot_speed_vs_accuracy <- function(data,
                                           xcol, ycol,
                                           color, size,
                                           title = NULL,
                                           caption = NULL,
                                           palette = "turbo",
                                           size_range = c(2, 12)) {
  library(ggplot2)
  library(stringr)

  ggplot(data, aes(x = {{ xcol }}, y = {{ ycol }}, color = {{ color }}, size = {{ size }})) +
    geom_point(alpha = 0.7) +
    scale_color_viridis_d(option = palette) +
    scale_size_continuous(
      range = size_range
      ) +
    labs(
      title = title,
      x = str_to_title(str_replace_all(str_remove(as_label(enquo(xcol)), "^test_"), "_", " ")),
      y = str_to_title(str_replace_all(str_remove(as_label(enquo(ycol)), "^test_"), "_", " ")),
      color = str_to_title(str_replace_all(as_label(enquo(color)), "_", " ")),
      size  = str_to_title(str_replace_all(as_label(enquo(size)), "_", " "))
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "right",
      plot.title = element_text(face = "plain", hjust = 0.5)
    )
}