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
                                   begin = 0){
  
  ggplot(data, aes(
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
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
}