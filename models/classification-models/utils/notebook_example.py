"""
Example code to add to your notebooks for creating precision-recall visualizations.

Copy and paste the relevant sections into your targettype.ipynb and actiontype.ipynb notebooks.
"""

# =============================================================================
# SECTION 1: Add this to your notebook imports
# =============================================================================

# Add to your existing imports
from utils import (
    create_rare_labels_pr_plot,
    create_strategy_comparison_plot,
    create_label_focused_plot,
    create_all_rare_labels_plots,
    create_comprehensive_visualization,
)

# =============================================================================
# SECTION 2: For targettype.ipynb - Add this after your strategy experiments
# =============================================================================

# Load the strategy predictions data
TASK_NAME = 'target-type'
TARGET_TYPE_RARE_LABELS = [
    'mining_company', 'non_maoist_armed_group', 'former_maoist',
    'high_caste_landowner', 'businessman', 'aspiring_politician'
]

target_predictions_df = load_dataframe_csv('target-type_strategy_predictions.csv', task_name=TASK_NAME)

# Create comprehensive visualizations for target-type
print("Creating precision-recall visualizations for target-type rare labels...")

# 1. Overall rare labels plot
fig1, ax1 = create_rare_labels_pr_plot(
    target_predictions_df, 
    TASK_NAME, 
    TARGET_TYPE_RARE_LABELS,
    title_suffix="(ConfliBERT, 100% train)"
)
plt.show()

# 2. Strategy comparison plot
fig2, ax2 = create_strategy_comparison_plot(
    target_predictions_df,
    TASK_NAME,
    TARGET_TYPE_RARE_LABELS
)
plt.show()

# 3. Focus on very rare labels (individual plots)
very_rare_labels = ['ngos', 'mining_company', 'high_caste_landowner', 'businessman']
for label in very_rare_labels:
    fig, ax = create_label_focused_plot(
        target_predictions_df,
        TASK_NAME,
        label
    )
    plt.show()

# 4. Create all plots and save them
output_dir = f"{get_task_results_dir(TASK_NAME)}/pr_visualizations"
os.makedirs(output_dir, exist_ok=True)

all_plots = create_all_rare_labels_plots(
    target_predictions_df,
    TASK_NAME,
    save_dir=output_dir
)

print(f"Created {len(all_plots)} plots and saved to: {output_dir}")

# =============================================================================
# SECTION 3: For actiontype.ipynb - Add this after your strategy experiments
# =============================================================================

# Load the strategy predictions data
action_predictions_df = load_dataframe_csv('action-type_strategy_predictions.csv', task_name=TASK_NAME)

# Create comprehensive visualizations for action-type
print("Creating precision-recall visualizations for action-type rare labels...")

# 1. Overall rare labels plot (abduction is the main rare label)
fig1, ax1 = create_rare_labels_pr_plot(
    action_predictions_df, 
    TASK_NAME, 
    ACTION_TYPE_RARE_LABELS,
    title_suffix="(ConfliBERT, 100% train)"
)
plt.show()

# 2. Strategy comparison plot
fig2, ax2 = create_strategy_comparison_plot(
    action_predictions_df,
    TASK_NAME,
    ACTION_TYPE_RARE_LABELS
)
plt.show()

# 3. Focus on abduction label
fig3, ax3 = create_label_focused_plot(
    action_predictions_df,
    TASK_NAME,
    'abduction'
)
plt.show()

# 4. Create all plots and save them
output_dir = f"{get_task_results_dir(TASK_NAME)}/pr_visualizations"
os.makedirs(output_dir, exist_ok=True)

all_plots = create_all_rare_labels_plots(
    action_predictions_df,
    TASK_NAME,
    save_dir=output_dir
)

print(f"Created {len(all_plots)} plots and saved to: {output_dir}")

# =============================================================================
# SECTION 4: For combining both tasks - Add this to a new cell
# =============================================================================

# Create comprehensive visualization combining both tasks
predictions_data = {
    'target-type': target_predictions_df,
    'action-type': action_predictions_df
}

# Create output directory for combined visualizations
combined_output_dir = "./combined_pr_visualizations"
os.makedirs(combined_output_dir, exist_ok=True)

# Generate comprehensive visualizations
figures = create_comprehensive_visualization(predictions_data, combined_output_dir)

print(f"Created comprehensive visualizations:")
for name, fig_data in figures.items():
    if isinstance(fig_data, tuple):
        print(f"  - {name}: 1 figure")
    else:
        print(f"  - {name}: {len(fig_data)} figures")

print(f"\nAll combined plots saved to: {combined_output_dir}")

# =============================================================================
# SECTION 5: Custom styling and advanced usage
# =============================================================================

# Custom colors for strategies
strategy_colors = {
    'baseline': '#1f77b4',
    'threshold_tuned': '#ff7f0e', 
    'focal_tuned': '#2ca02c',
    'class_weights_tuned': '#d62728',
    'conservative_class_weights_tuned': '#9467bd',
    'weighted_sampler_tuned': '#8c564b',
    'augmentation_bt_tuned': '#e377c2',
    'augmentation_t5_tuned': '#7f7f7f'
}

# Create a custom plot with specific styling
from utils import create_pr_scatter_plot, extract_pr_metrics_from_predictions

# Extract metrics for target-type
target_metrics = extract_pr_metrics_from_predictions(
    target_predictions_df, 
    TARGET_TYPE_RARE_LABELS
)

# Create custom plot
fig, ax = create_pr_scatter_plot(
    target_metrics,
    rare_labels=TARGET_TYPE_RARE_LABELS,
    title="Custom Styled Precision-Recall Plot",
    strategy_colors=strategy_colors,
    alpha=0.8,
    s=120,
    figsize=(12, 10)
)
plt.show()

# =============================================================================
# SECTION 6: Analysis and interpretation
# =============================================================================

# Print summary statistics for rare labels
print("\n" + "="*60)
print("RARE LABEL PERFORMANCE SUMMARY")
print("="*60)

for task_name, predictions_df in predictions_data.items():
    print(f"\n{task_name.upper()}:")
    
    if task_name == 'target-type':
        rare_labels = TARGET_TYPE_RARE_LABELS
    else:
        rare_labels = ACTION_TYPE_RARE_LABELS
    
    metrics_df = extract_pr_metrics_from_predictions(predictions_df, rare_labels)
    
    # Group by strategy and calculate average F1 for rare labels
    strategy_performance = metrics_df.groupby('strategy')['f1'].agg(['mean', 'std', 'count']).round(3)
    strategy_performance = strategy_performance.sort_values('mean', ascending=False)
    
    print("Strategy Performance (Average F1 on Rare Labels):")
    print(strategy_performance)
    
    # Find best performing strategy for each rare label
    print("\nBest Strategy per Rare Label:")
    best_per_label = metrics_df.loc[metrics_df.groupby('label')['f1'].idxmax()]
    for _, row in best_per_label.iterrows():
        print(f"  {row['label']}: {row['strategy']} (F1={row['f1']:.3f})")

print("\n" + "="*60)
print("INTERPRETATION GUIDE")
print("="*60)
print("""
1. Points closer to the top-right corner indicate better precision-recall balance
2. Iso-F1 contours show lines of constant F1 score
3. Strategies that push points toward higher F1 contours are better for rare labels
4. Look for strategies that improve recall without sacrificing too much precision
5. Very rare labels (< 1%) are most challenging and benefit most from imbalance strategies
""")
