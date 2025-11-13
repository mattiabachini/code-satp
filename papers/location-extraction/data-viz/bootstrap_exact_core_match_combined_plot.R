# Combined Bootstrap Exact Core Match Visualization
# Combines Seq2Seq, BERT, and LLM bootstrap exact core match plots using patchwork
# Arranges vertically with seq2seq on top, BERT in middle, LLMs on bottom

library(patchwork)

# Configuration
# Script should be run from papers/location-extraction directory
OUTPUT_DIR <- "data-viz/images"

# Source the individual plot scripts to get plot objects
# Create local environments to capture plots
seq2seq_env <- new.env()
source("data-viz/bootstrap_exact_core_match_seq2seq_plot.R", local = seq2seq_env)
seq2seq_plot <- seq2seq_env$p

bert_env <- new.env()
source("data-viz/bootstrap_exact_core_match_bert_plot.R", local = bert_env)
bert_plot <- bert_env$p

llm_env <- new.env()
source("data-viz/bootstrap_exact_core_match_llms_plot.R", local = llm_env)
llm_plot <- llm_env$p

# Combine plots vertically with seq2seq on top
# Add common x-axis label to bottom plot and caption
combined_plot <- seq2seq_plot / bert_plot / (llm_plot + labs(x = "Model", caption = "Based on 5000 bootstrap samples. Exact core match = state + district + village all correct")) +
  plot_layout(
    guides = "collect",  # Combine legends (though we have none)
    heights = c(1, 1, 1)  # Equal heights for all three panels
  ) &
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA)
  )

# Save combined plot
output_file <- file.path(OUTPUT_DIR, "bootstrap_exact_core_match_combined.png")
ggsave(
  output_file,
  plot = combined_plot,
  width = 10,
  height = 8,  # Not too tall
  dpi = 300
)

cat("Combined plot saved to:", output_file, "\n")

