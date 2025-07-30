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

# Import main functions from visualization_utils
from .visualization_utils import (
    scatter_plot_speed_vs_accuracy,
    heatmap_label_f1_scores
)

# Define what gets imported with "from utils import *"
__all__ = [
    # Multi-label utilities
    'create_fixed_splits',
    'MultiLabelDataset', 
    'compute_metrics',
    'train_transformer_model',
    'run_model_experiments',
    
    # Visualization utilities
    'scatter_plot_speed_vs_accuracy',
    'heatmap_label_f1_scores'
] 