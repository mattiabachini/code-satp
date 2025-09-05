# Precision-Recall Visualization Utilities

This module provides comprehensive utilities for creating precision-recall scatter plots with iso-F1 contours to analyze the performance of different imbalance handling strategies on rare labels.

## Overview

The visualization utilities are designed to help you:
1. **Visualize rare label performance** across different imbalance handling strategies
2. **Compare strategies** using precision-recall scatter plots with iso-F1 contours
3. **Combine multiple plots** into single figures (similar to R's patchwork)
4. **Generate comprehensive reports** for both target-type and action-type tasks

## Key Features

- ✅ **Iso-F1 contours** for easy interpretation of F1 score levels
- ✅ **Rare label focus** with predefined rare labels for each task
- ✅ **Strategy comparison** across multiple imbalance handling approaches
- ✅ **Multi-panel figures** for comprehensive analysis
- ✅ **Automatic plot saving** with organized output
- ✅ **Customizable styling** with colors, markers, and layouts

## Rare Labels Identified

### Target Type (< 1% frequency)
- `ngos`: 0.07%
- `mining_company`: 0.69%
- `non_maoist_armed_group`: 0.60%
- `former_maoist`: 0.73%
- `high_caste_landowner`: 0.12%
- `businessman`: 0.57%
- `aspiring_politician`: 0.24%

### Action Type (< 5% frequency)
- `abduction`: 4.77%

## Quick Start

### 1. Basic Usage

```python
from utils import (
    create_rare_labels_pr_plot,
    create_strategy_comparison_plot,
    TARGET_TYPE_RARE_LABELS,
    ACTION_TYPE_RARE_LABELS
)

# Load your predictions data
predictions_df = load_dataframe_csv('target-type_strategy_predictions.csv', task_name='target-type')

# Create precision-recall plot for rare labels
fig, ax = create_rare_labels_pr_plot(
    predictions_df, 
    'target-type', 
    TARGET_TYPE_RARE_LABELS
)
plt.show()
```

### 2. Strategy Comparison

```python
# Compare strategies on rare labels
fig, ax = create_strategy_comparison_plot(
    predictions_df,
    'target-type',
    TARGET_TYPE_RARE_LABELS
)
plt.show()
```

### 3. Focus on Specific Label

```python
# Focus on a very rare label
fig, ax = create_label_focused_plot(
    predictions_df,
    'target-type',
    'ngos'  # Only 0.07% of data
)
plt.show()
```

### 4. Comprehensive Analysis

```python
# Create all visualizations and save them
output_dir = "./pr_visualizations"
os.makedirs(output_dir, exist_ok=True)

all_plots = create_all_rare_labels_plots(
    predictions_df,
    'target-type',
    save_dir=output_dir
)
```

## Advanced Usage

### Multi-Panel Figures (R Patchwork Equivalent)

```python
from utils import create_multi_panel_figure, create_comprehensive_visualization

# Combine multiple plots into a single figure
plots_data = [(fig1, ax1), (fig2, ax2), (fig3, ax3), (fig4, ax4)]
titles = ["Plot 1", "Plot 2", "Plot 3", "Plot 4"]

combined_fig, combined_axes = create_multi_panel_figure(
    plots_data,
    layout=(2, 2),  # 2 rows, 2 columns
    titles=titles,
    main_title="Comprehensive Analysis",
    figsize=(16, 12)
)
plt.show()
```

### Cross-Task Analysis

```python
# Analyze both target-type and action-type together
predictions_data = {
    'target-type': target_predictions_df,
    'action-type': action_predictions_df
}

figures = create_comprehensive_visualization(predictions_data, output_dir)
```

### Custom Styling

```python
# Custom colors for strategies
strategy_colors = {
    'baseline': '#1f77b4',
    'threshold_tuned': '#ff7f0e', 
    'focal_tuned': '#2ca02c',
    'class_weights_tuned': '#d62728',
    'conservative_class_weights_tuned': '#9467bd',
    'weighted_sampler_tuned': '#8c564b'
}

# Create custom plot
fig, ax = create_pr_scatter_plot(
    metrics_df,
    rare_labels=TARGET_TYPE_RARE_LABELS,
    title="Custom Styled Plot",
    strategy_colors=strategy_colors,
    alpha=0.8,
    s=120
)
```

## Integration with Notebooks

### For targettype.ipynb

Add this code after your strategy experiments:

```python
# Load strategy predictions
target_predictions_df = load_dataframe_csv('target-type_strategy_predictions.csv', task_name=TASK_NAME)

# Create visualizations
fig, ax = create_rare_labels_pr_plot(
    target_predictions_df, 
    TASK_NAME, 
    TARGET_TYPE_RARE_LABELS
)
plt.show()
```

### For actiontype.ipynb

Add this code after your strategy experiments:

```python
# Load strategy predictions
action_predictions_df = load_dataframe_csv('action-type_strategy_predictions.csv', task_name=TASK_NAME)

# Create visualizations
fig, ax = create_rare_labels_pr_plot(
    action_predictions_df, 
    TASK_NAME, 
    ACTION_TYPE_RARE_LABELS
)
plt.show()
```

## Available Functions

### Core Visualization Functions

- `create_rare_labels_pr_plot()` - Main function for rare label PR plots
- `create_strategy_comparison_plot()` - Compare strategies on rare labels
- `create_label_focused_plot()` - Focus on a specific rare label
- `create_all_rare_labels_plots()` - Generate all plots and save them

### Plot Combination Functions

- `create_multi_panel_figure()` - Combine multiple plots into one figure
- `create_comparison_grid()` - Create strategy comparison grid
- `create_rare_labels_summary_plot()` - Summary across tasks
- `create_strategy_performance_summary()` - Strategy performance summary
- `create_comprehensive_visualization()` - Generate all visualizations

### Utility Functions

- `extract_pr_metrics_from_predictions()` - Extract PR metrics from predictions
- `create_iso_f1_contours()` - Add iso-F1 contours to plots
- `create_pr_scatter_plot()` - Core scatter plot function

## Data Format Requirements

Your predictions DataFrame should have columns like:
- `strategy` - Strategy name (e.g., 'baseline', 'focal_tuned')
- `true_{label}` - True labels for each label
- `pred_{label}` - Predicted labels for each label
- `prob_{label}` - Predicted probabilities (optional)

Example:
```
strategy | true_ngos | pred_ngos | prob_ngos | true_mining_company | pred_mining_company | ...
baseline | 0         | 0         | 0.1       | 1                   | 0                   | ...
focal_tuned | 0     | 1         | 0.7       | 1                   | 1                   | ...
```

## Interpretation Guide

### Reading the Plots

1. **X-axis (Recall)**: Proportion of true positives correctly identified
2. **Y-axis (Precision)**: Proportion of predicted positives that are correct
3. **Iso-F1 contours**: Lines of constant F1 score (harmonic mean of precision and recall)
4. **Points**: Each point represents a strategy-label combination

### What to Look For

- **Top-right corner**: Best precision-recall balance
- **Higher F1 contours**: Better overall performance
- **Strategy clustering**: Similar strategies should cluster together
- **Rare label performance**: Points for rare labels should move toward higher F1 contours with better strategies

### Strategy Effectiveness

- **Baseline**: Usually in lower-left (low precision, low recall)
- **Threshold tuning**: Often improves precision-recall balance
- **Focal loss**: Should improve recall for rare labels
- **Class weights**: May improve recall but watch for precision drop
- **Conservative weights**: Balance between recall and precision
- **Augmentation**: Should improve overall performance

## Example Output

The utilities will generate:
1. **Individual plots** for each rare label
2. **Strategy comparison plots** showing all strategies
3. **Multi-panel figures** combining multiple analyses
4. **Saved plots** in organized directories

## Troubleshooting

### Common Issues

1. **No data to plot**: Check that your predictions DataFrame has the correct column names
2. **Empty plots**: Ensure rare labels exist in your data
3. **Import errors**: Make sure you're importing from the correct utils module
4. **Missing strategies**: Check that strategy names match between your data and the visualization

### Debug Tips

```python
# Check your data structure
print(predictions_df.columns.tolist())
print(predictions_df['strategy'].unique())
print(predictions_df.shape)

# Check rare label frequencies
for label in TARGET_TYPE_RARE_LABELS:
    true_col = f'true_{label}'
    if true_col in predictions_df.columns:
        count = predictions_df[true_col].sum()
        print(f"{label}: {count} positive examples")
```

## Performance Tips

1. **Use sample data** for testing during development
2. **Save plots** to avoid regenerating expensive visualizations
3. **Filter strategies** if you have too many to compare clearly
4. **Use custom colors** to distinguish strategies more clearly

## Future Enhancements

Potential improvements:
- Interactive plots with plotly
- Statistical significance testing
- Confidence intervals for metrics
- Automated report generation
- Integration with other evaluation metrics

## References

- Precision-Recall curves: [Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
- F1 score: [Wikipedia](https://en.wikipedia.org/wiki/F-score)
- Imbalance handling: [Imbalanced-learn documentation](https://imbalanced-learn.org/)
