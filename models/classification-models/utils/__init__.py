"""
Utility functions for classification models.

This package provides reusable functions for multi-label classification
and visualization tasks.
"""

# Import main functions from multilabel_utils
from .multilabel_utils import (
    create_fixed_splits,
    MultiLabelDataset,
    compute_metrics,
    train_transformer_model,
    run_model_experiments
)

# Strategy experiments
from .strategy_experiments import (
    run_strategy_experiments,
    choose_thresholds_micro,
    apply_thresholds,
)

# Import main functions from visualization_utils
from .visualization_utils import (
    scatter_plot_speed_vs_accuracy,
    heatmap_label_f1_scores,
    plot_heatmap,
    heatmap_label_f1_by_strategy,
    # Precision-recall visualization utilities
    create_iso_f1_contours,
    extract_pr_metrics_from_predictions,
    create_pr_scatter_plot,
    create_rare_labels_pr_plot,
    create_strategy_comparison_plot,
    create_label_focused_plot,
    create_all_rare_labels_plots,
    create_sample_predictions_data,
    create_rare_labels_comparison_plots,
    TARGET_TYPE_RARE_LABELS,
    ACTION_TYPE_RARE_LABELS,
)

# Import plot combination utilities
from .plot_combination_utils import (
    create_multi_panel_figure,
    copy_plot_elements,
    create_comparison_grid,
    create_rare_labels_summary_plot,
    create_strategy_performance_summary,
    create_comprehensive_visualization,
)

# Define what gets imported with "from utils import *"
__all__ = [
    # Multi-label utilities
    'create_fixed_splits',
    'MultiLabelDataset', 
    'compute_metrics',
    'train_transformer_model',
    'run_model_experiments',
    'run_strategy_experiments',
    'choose_thresholds_micro',
    'apply_thresholds',
    
    # Visualization utilities
    'scatter_plot_speed_vs_accuracy',
    'heatmap_label_f1_scores',
    'plot_heatmap',
    'heatmap_label_f1_by_strategy',
    
    # Precision-recall visualization utilities
    'create_iso_f1_contours',
    'extract_pr_metrics_from_predictions',
    'create_pr_scatter_plot',
    'create_rare_labels_pr_plot',
    'create_strategy_comparison_plot',
    'create_label_focused_plot',
    'create_all_rare_labels_plots',
    'create_sample_predictions_data',
    'create_rare_labels_comparison_plots',
    'TARGET_TYPE_RARE_LABELS',
    'ACTION_TYPE_RARE_LABELS',
    
    # Plot combination utilities
    'create_multi_panel_figure',
    'copy_plot_elements',
    'create_comparison_grid',
    'create_rare_labels_summary_plot',
    'create_strategy_performance_summary',
    'create_comprehensive_visualization',
]