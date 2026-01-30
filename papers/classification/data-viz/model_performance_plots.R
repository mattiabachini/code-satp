# Model Performance Plots Function

model_performance_plot <- function(data,
                                   xvar,
                                   yvar,
                                   color = model_label,
                                   title,
                                   xtitle = NULL,
                                   ytitle = NULL,
                                   palette = "turbo",
                                   end = 1,
                                   begin = 0,
                                   hlines = NULL){

  p <- ggplot(data, aes(
    x = {{xvar}},
    y = {{yvar}},
    color = {{color}}
  )) +
    geom_point() +
    geom_line() +
    labs(
      x = xtitle,
      y = ytitle,
      title = title,
      color = "Model"
    ) +
    scale_x_log10(
      breaks = c(0.03125, 0.0625, 0.125, .25, .5, 1),
      labels = c("1/32", "1/16", "1/8", "1/4", "1/2", "1")
    ) +
    scale_color_viridis_d(option = palette, end = end, begin = begin) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 9),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom",
      legend.key.size = unit(0.4, "cm"),
      legend.text = element_text(size = 8),
      legend.title = element_text(size = 9)
    )

  # Add optional horizontal reference lines (e.g., LLM baselines)
  if (!is.null(hlines)) {
    p <- p + add_reference_lines(hlines = hlines)
  }

  p
}

ref_line <- function(y,
                     label = "",
                     x = NULL,
                     y_offset = NULL,
                     hjust = NULL,
                     vjust = NULL,
                     color = NULL,
                     linetype = NULL,
                     size = NULL){
  list(
    y = y,
    label = label,
    x = x,
    y_offset = y_offset,
    hjust = hjust,
    vjust = vjust,
    color = color,
    linetype = linetype,
    size = size
  )
}

add_reference_lines <- function(...,
                                hlines = NULL,
                                default_x = 0.035,
                                default_y_offset = -0.015,
                                default_color = "gray40",
                                default_linetype = "dashed",
                                default_label_size = 2.5){
  if (length(list(...)) > 0) {
    hlines <- list(...)
  }

  if (is.null(hlines) || length(hlines) == 0) {
    return(list())
  }

  layers <- lapply(hlines, function(h){
    if (is.null(h$y)) {
      return(NULL)
    }

    line_layer <- geom_hline(
      yintercept = h$y,
      linetype = h$linetype %||% default_linetype,
      color = h$color %||% default_color,
      linewidth = 0.5
    )

    label <- h$label %||% ""
    if (label == "") {
      return(list(line_layer))
    }

    text_layer <- annotate(
      "text",
      x = h$x %||% default_x,
      y = h$y + (h$y_offset %||% default_y_offset),
      label = label,
      hjust = h$hjust %||% 0,
      vjust = h$vjust %||% 0,
      size = h$size %||% default_label_size,
      color = h$color %||% default_color
    )

    list(line_layer, text_layer)
  })

  Filter(Negate(is.null), unlist(layers, recursive = FALSE))
}
