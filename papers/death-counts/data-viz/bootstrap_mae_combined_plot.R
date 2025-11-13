# Combined Bootstrap MAE Visualization
# Combines LLM and Seq2Seq bootstrap MAE plots using patchwork
# Arranges vertically with seq2seq on top

library(patchwork)

# Configuration
# Script should be run from papers/death-counts directory
OUTPUT_DIR <- "data-viz/images"

# Source the individual plot scripts to get plot objects
# Create a local environment to capture plots
seq2seq_env <- new.env()
source("data-viz/bootstrap_mae_seq2seq_plot.R", local = seq2seq_env)
seq2seq_plot <- seq2seq_env$p

llm_env <- new.env()
source("data-viz/bootstrap_mae_llms_plot.R", local = llm_env)
llm_plot <- llm_env$p

# Combine plots vertically with seq2seq on top
# Add common x-axis label to bottom plot and caption
combined_plot <- seq2seq_plot / (llm_plot + labs(x = "Model", caption = "Based on 5000 bootstrap samples. Side-by-side CIs show MAE for all events (left) and non-zero events (right)")) +
  plot_layout(
    guides = "collect",  # Combine legends
    heights = c(1, 1)     # Equal heights for both panels
  ) &
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  )

# Save combined plot
output_file <- file.path(OUTPUT_DIR, "bootstrap_mae_combined.png")
ggsave(
  output_file,
  plot = combined_plot,
  width = 10,
  height = 8,  # Taller for vertical layout
  dpi = 300
)

cat("Combined plot saved to:", output_file, "\n")

