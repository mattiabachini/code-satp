import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Scatter plot for throughput vs accuracy

def scatter_plot_speed_vs_accuracy(df, x_col, y_col, hue_col, size_col, title):
    """
    Creates a scatter (bubble) plot with customizable x and y axes.

    Parameters:
    - df: DataFrame containing the data
    - x_col: Column name for the x-axis (e.g., throughput, latency)
    - y_col: Column name for the y-axis (e.g., accuracy, latency)
    - hue_col: Column name for the hue (color grouping, e.g., model name)
    - size_col: Column name for the size (bubble size, e.g., data fraction)
    - title: Custom plot title
    """
    plt.figure(figsize=(7, 5))
    scatter = sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        size=size_col,
        sizes=(20, 400),
        alpha=0.7,
        palette="turbo"
    )
    x_label_cleaned = x_col.replace("eval_", "").replace("test_", "").replace("_", " ").title()
    y_label_cleaned = y_col.replace("eval_", "").replace("test_", "").replace("_", " ").title()
    scatter.set_title(title)
    scatter.set_xlabel(x_label_cleaned)
    scatter.set_ylabel(y_label_cleaned)
    handles, labels = scatter.get_legend_handles_labels()
    model_names = df[hue_col].astype(str).unique()
    fraction_values = sorted(df[size_col].astype(str).unique())
    hue_handles = [h for h, l in zip(handles, labels) if l in model_names]
    hue_labels = [l for l in labels if l in model_names]
    size_handles = [h for h, l in zip(handles, labels) if l in fraction_values]
    size_labels = [l for l in labels if l in fraction_values]
    hue_legend = plt.legend(hue_handles, hue_labels, title="Model Name", loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.gca().add_artist(hue_legend)
    size_legend = plt.legend(size_handles, size_labels, title="Fraction Raw", loc="lower left", bbox_to_anchor=(1.05, 0))
    plt.tight_layout()
    plt.show()

# Heatmap for F1 scores by label

def heatmap_label_f1_scores(
    df,
    fraction_col="fraction_label",
    model_label_col="model_label",
    f1_score_prefix="test_",
    f1_score_suffix="_f1-score",
    avg_identifier="avg",
    title="F1 Scores for Each Label Across Models",
    note=None,
    figsize=(10, 7),
    cmap="cividis"
):
    """
    Plots a heatmap of F1 scores for each label across models, for a given fraction of data.
    Parameters:
    - df: DataFrame containing the results
    - fraction_col: Column indicating data fraction (default: 'fraction_label')
    - model_label_col: Column for model names (default: 'model_label')
    - f1_score_prefix: Prefix for F1-score columns (default: 'test_')
    - f1_score_suffix: Suffix for F1-score columns (default: '_f1-score')
    - avg_identifier: Substring to exclude average columns (default: 'avg')
    - title: Plot title
    - note: Optional note to add below the plot
    - figsize: Figure size
    - cmap: Colormap for heatmap
    """
    # Filter for 100% data (or the max fraction if not present)
    if df[fraction_col].dtype == float or df[fraction_col].dtype == int:
        max_fraction = df[fraction_col].max()
        df_100 = df[df[fraction_col] == max_fraction]
    else:
        df_100 = df[df[fraction_col] == "100.0%"]
        if df_100.empty:
            # fallback: use max fraction
            max_fraction = df[fraction_col].max()
            df_100 = df[df[fraction_col] == max_fraction]
    # Identify label F1 columns (not averages)
    label_f1_columns = [
        col for col in df_100.columns
        if col.startswith(f1_score_prefix) and col.endswith(f1_score_suffix) and avg_identifier not in col
    ]
    
    # Debug: print what we found
    print(f"Found {len(label_f1_columns)} label F1 columns: {label_f1_columns}")
    print(f"DataFrame shape: {df_100.shape}")
    print(f"Available columns: {list(df_100.columns)}")
    
    # Check if we have any label columns
    if not label_f1_columns:
        print("Warning: No label F1 columns found! This might cause the error.")
        return
    
    # Prepare DataFrame for heatmap
    df_f1_100 = df_100[[model_label_col] + label_f1_columns]
    df_f1_melted_100 = df_f1_100.melt(id_vars=[model_label_col], var_name="Label", value_name="F1 Score")
    df_f1_melted_100["Label"] = (
        df_f1_melted_100["Label"]
        .str.replace(f1_score_prefix, "")
        .str.replace(f1_score_suffix, "")
        .str.replace("_", " ")
        .str.title()
    )
    df_f1_pivot_100 = df_f1_melted_100.pivot(index=model_label_col, columns="Label", values="F1 Score")
    # Remove 'Incident Summary' if present
    df_f1_pivot_100 = df_f1_pivot_100.drop(columns="Incident Summary", errors="ignore")
    # Sort event types (columns) by their average F1 score across models (descending order)
    event_order = df_f1_pivot_100.mean().sort_values(ascending=False).index
    df_f1_pivot_100 = df_f1_pivot_100[event_order]
    # Sort models (rows) by their Micro F1 score (descending order) if available
    # Try different possible column names for micro F1 score
    micro_f1_col = None
    for col in df_100.columns:
        if "micro" in col and "f1" in col and "avg" in col:
            micro_f1_col = col
            break
    
    if micro_f1_col is not None:
        df_model_avg_f1 = df_100.set_index(model_label_col)[micro_f1_col].sort_values(ascending=False)
        model_order = df_model_avg_f1.index
        df_f1_pivot_100 = df_f1_pivot_100.loc[model_order]
    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(df_f1_pivot_100, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5, linecolor="gray", cbar_kws={'label': 'F1 Score'})
    ax.set_title(title, pad=20)
    if note:
        plt.figtext(0.5, -0.08, note, ha="center", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show() 

# Heatmap for plotting metrics by model and fraction

def plot_heatmap(results_df, value, title_suffix=""):
    """
    Creates a heatmap where:
      - Rows = Models (model_label)
      - Columns = Fractions (fraction_label)  
      - Cell Values = specified metric (color-coded)
    """
    # Pivot the DataFrame: index=Model, columns=Fraction, values=metric
    heatmap_data = results_df.pivot(
        index="model_label",
        columns="fraction_label", 
        values=value
    )
    
    # Sort columns in ascending order of fraction
    heatmap_data = heatmap_data[["3%", "6%", "12%", "25%", "50%", "100%"]]
    
    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,         # writes the values in each cell
        fmt=".3f",          # format for floating point
        cmap="YlGnBu",      # color palette
        cbar_kws={"label": value.replace("eval_", "").replace("test_", "").replace("_", " ").title()}
    )
    
    # Labeling and layout
    metric_name = value.replace("eval_", "").replace("test_", "").replace("_", " ").title()
    plt.title(f"{metric_name} by Model and Data Fraction{title_suffix}")
    plt.xlabel("Data Fraction")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()    

# Heatmap for strategy vs label (rows=labels, cols=strategies)

def heatmap_label_f1_by_strategy(
    df,
    strategy_col="strategy",
    label_col="label",
    value_col="f1",
    title="F1 Scores per Label across Imbalance Strategies",
    note=None,
    figsize=(10, 7),
    cmap="cividis"
):
    """
    Plot a heatmap with strategies on y-axis and labels on x-axis using per-label F1,
    matching the orientation and sorting style of the individual-label heatmap.

    Parameters:
    - df: long-form DataFrame with columns [strategy, label, f1]
    - strategy_col: column name for strategies
    - label_col: column name for labels
    - value_col: column name for F1 values
    - title: plot title
    - note: optional subtitle/note
    """
    # Pivot so rows=strategies, columns=labels
    pivot = df.pivot(index=strategy_col, columns=label_col, values=value_col)

    # Sort columns (labels) by their average F1 across strategies (descending)
    col_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot = pivot[col_order]

    # Sort rows (strategies) by their average F1 across labels (descending)
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5, linecolor="gray", cbar_kws={"label": "F1 Score"})
    ax.set_xlabel("Label")
    ax.set_ylabel("Strategy")
    ax.set_title(title, pad=20)
    if note:
        plt.figtext(0.5, -0.08, note, ha="center", fontsize=10)
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# =============================================================================
# PRECISION-RECALL VISUALIZATION UTILITIES
# =============================================================================

# Define rare labels for each task
TARGET_TYPE_RARE_LABELS = [
    'ngos', 'mining_company', 'non_maoist_armed_group', 'former_maoist',
    'high_caste_landowner', 'businessman', 'aspiring_politician'
]

ACTION_TYPE_RARE_LABELS = [
    'abduction'
]

def create_iso_f1_contours(ax, f1_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                          alpha=0.3, colors='gray', linestyles='--'):
    """
    Create iso-F1 contours on a precision-recall plot.
    
    Args:
        ax: matplotlib axes object
        f1_levels: List of F1 scores to draw contours for
        alpha: Transparency of contour lines
        colors: Color of contour lines
        linestyles: Line style for contours
    """
    # Create precision and recall grids
    precision = np.linspace(0.01, 1.0, 1000)
    recall = np.linspace(0.01, 1.0, 1000)
    P, R = np.meshgrid(precision, recall)
    
    # Calculate F1 for each point
    F1 = 2 * (P * R) / (P + R + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Draw contours
    contours = ax.contour(P, R, F1, levels=f1_levels, colors=colors, 
                         linestyles=linestyles, alpha=alpha, linewidths=1)
    
    # Add contour labels
    ax.clabel(contours, inline=True, fontsize=8, fmt='F1=%.1f')
    
    return contours

def extract_pr_metrics_from_predictions(predictions_df, label_cols, strategy_col='strategy'):
    """
    Extract precision and recall metrics for each label and strategy from predictions DataFrame.
    
    Args:
        predictions_df: DataFrame with columns like 'true_{label}', 'pred_{label}', 'strategy'
        label_cols: List of label column names
        strategy_col: Name of the strategy column
        
    Returns:
        DataFrame with columns: strategy, label, precision, recall, f1
    """
    results = []
    
    for strategy in predictions_df[strategy_col].unique():
        strategy_data = predictions_df[predictions_df[strategy_col] == strategy]
        
        for label in label_cols:
            true_col = f'true_{label}'
            pred_col = f'pred_{label}'
            
            if true_col in strategy_data.columns and pred_col in strategy_data.columns:
                y_true = strategy_data[true_col].values
                y_pred = strategy_data[pred_col].values
                
                # Calculate metrics
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                results.append({
                    'strategy': strategy,
                    'label': label,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
    
    return pd.DataFrame(results)

def create_pr_scatter_plot(metrics_df, rare_labels=None, title="Precision-Recall Scatter Plot", 
                          figsize=(10, 8), show_iso_f1=True, f1_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                          strategy_colors=None, label_markers=None, alpha=0.7, s=100):
    """
    Create a precision-recall scatter plot with iso-F1 contours.
    
    Args:
        metrics_df: DataFrame with columns: strategy, label, precision, recall, f1
        rare_labels: List of rare labels to highlight (if None, shows all labels)
        title: Plot title
        figsize: Figure size
        show_iso_f1: Whether to show iso-F1 contours
        f1_levels: F1 levels for contours
        strategy_colors: Dict mapping strategy names to colors
        label_markers: Dict mapping label names to markers
        alpha: Transparency of points
        s: Size of points
        
    Returns:
        matplotlib figure and axes objects
    """
    # Filter for rare labels if specified
    if rare_labels is not None:
        plot_data = metrics_df[metrics_df['label'].isin(rare_labels)].copy()
    else:
        plot_data = metrics_df.copy()
    
    if plot_data.empty:
        print("Warning: No data to plot after filtering for rare labels")
        return None, None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add iso-F1 contours if requested
    if show_iso_f1:
        create_iso_f1_contours(ax, f1_levels=f1_levels)
    
    # Set up colors and markers
    strategies = plot_data['strategy'].unique()
    if strategy_colors is None:
        strategy_colors = {strategy: f'C{i}' for i, strategy in enumerate(strategies)}
    
    labels = plot_data['label'].unique()
    if label_markers is None:
        markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'H', '+', 'x', 'X']
        label_markers = {label: markers[i % len(markers)] for i, label in enumerate(labels)}
    
    # Plot points
    for strategy in strategies:
        strategy_data = plot_data[plot_data['strategy'] == strategy]
        
        for label in strategy_data['label'].unique():
            label_data = strategy_data[strategy_data['label'] == label]
            
            ax.scatter(label_data['recall'], label_data['precision'], 
                      c=strategy_colors[strategy], marker=label_markers[label],
                      alpha=alpha, s=s, label=f'{strategy} - {label}',
                      edgecolors='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add diagonal line (F1 = 0.5 when precision = recall)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    
    # Legend
    handles, labels_legend = ax.get_legend_handles_labels()
    if len(handles) > 20:  # If too many legend entries, create a more compact legend
        # Group by strategy
        strategy_handles = {}
        for handle, label in zip(handles, labels_legend):
            strategy = label.split(' - ')[0]
            if strategy not in strategy_handles:
                strategy_handles[strategy] = handle
        
        ax.legend(strategy_handles.values(), strategy_handles.keys(), 
                 loc='center left', bbox_to_anchor=(1, 0.5), title='Strategy')
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig, ax

def create_rare_labels_pr_plot(predictions_df, task_name, rare_labels, 
                              title_suffix="", figsize=(12, 8), save_path=None):
    """
    Create a precision-recall plot specifically for rare labels.
    
    Args:
        predictions_df: DataFrame with prediction results
        task_name: Name of the task (e.g., 'target-type', 'action-type')
        rare_labels: List of rare label names
        title_suffix: Additional text for the title
        figsize: Figure size
        save_path: Path to save the plot (optional)
        
    Returns:
        matplotlib figure and axes objects
    """
    # Extract metrics
    label_cols = [col.replace('true_', '') for col in predictions_df.columns 
                  if col.startswith('true_') and col.replace('true_', '') in rare_labels]
    
    if not label_cols:
        print(f"Warning: No matching rare labels found in predictions data")
        return None, None
    
    metrics_df = extract_pr_metrics_from_predictions(predictions_df, label_cols)
    
    # Create plot
    title = f"Precision-Recall Plot for Rare Labels - {task_name.replace('-', ' ').title()}"
    if title_suffix:
        title += f" {title_suffix}"
    
    fig, ax = create_pr_scatter_plot(
        metrics_df, 
        rare_labels=rare_labels,
        title=title,
        figsize=figsize,
        show_iso_f1=True
    )
    
    # Save if path provided
    if save_path and fig is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig, ax

def create_strategy_comparison_plot(predictions_df, task_name, rare_labels, 
                                  strategies_to_compare=None, figsize=(12, 8), save_path=None):
    """
    Create a focused comparison plot for specific strategies on rare labels.
    
    Args:
        predictions_df: DataFrame with prediction results
        task_name: Name of the task
        rare_labels: List of rare label names
        strategies_to_compare: List of strategies to compare (if None, uses all)
        figsize: Figure size
        save_path: Path to save the plot (optional)
        
    Returns:
        matplotlib figure and axes objects
    """
    # Filter strategies if specified
    if strategies_to_compare is not None:
        plot_data = predictions_df[predictions_df['strategy'].isin(strategies_to_compare)].copy()
    else:
        plot_data = predictions_df.copy()
    
    # Extract metrics
    label_cols = [col.replace('true_', '') for col in plot_data.columns 
                  if col.startswith('true_') and col.replace('true_', '') in rare_labels]
    
    if not label_cols:
        print(f"Warning: No matching rare labels found in predictions data")
        return None, None
    
    metrics_df = extract_pr_metrics_from_predictions(plot_data, label_cols)
    
    # Create focused plot
    title = f"Strategy Comparison on Rare Labels - {task_name.replace('-', ' ').title()}"
    
    # Use distinct colors for strategies
    strategies = metrics_df['strategy'].unique()
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
    
    fig, ax = create_pr_scatter_plot(
        metrics_df,
        rare_labels=rare_labels,
        title=title,
        figsize=figsize,
        show_iso_f1=True,
        strategy_colors=strategy_colors,
        alpha=0.8,
        s=120
    )
    
    # Save if path provided
    if save_path and fig is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig, ax

def create_label_focused_plot(predictions_df, task_name, specific_label, 
                             figsize=(10, 8), save_path=None):
    """
    Create a precision-recall plot focused on a single rare label across all strategies.
    
    Args:
        predictions_df: DataFrame with prediction results
        task_name: Name of the task
        specific_label: Name of the specific label to focus on
        figsize: Figure size
        save_path: Path to save the plot (optional)
        
    Returns:
        matplotlib figure and axes objects
    """
    # Extract metrics for the specific label
    metrics_df = extract_pr_metrics_from_predictions(predictions_df, [specific_label])
    
    if metrics_df.empty:
        print(f"Warning: No data found for label '{specific_label}'")
        return None, None
    
    # Create focused plot
    title = f"Precision-Recall for '{specific_label}' - {task_name.replace('-', ' ').title()}"
    
    fig, ax = create_pr_scatter_plot(
        metrics_df,
        rare_labels=[specific_label],
        title=title,
        figsize=figsize,
        show_iso_f1=True,
        alpha=0.8,
        s=150
    )
    
    # Save if path provided
    if save_path and fig is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig, ax

def create_sample_predictions_data(task_name, n_samples=1000):
    """
    Create sample predictions data for demonstration purposes.
    
    Args:
        task_name: Name of the task
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with sample predictions data
    """
    np.random.seed(42)
    
    # Define strategies
    strategies = [
        'baseline', 'threshold_tuned', 'focal_tuned', 'class_weights_tuned',
        'conservative_class_weights_tuned', 'weighted_sampler_tuned'
    ]
    
    # Define rare labels based on task
    if task_name == 'target-type':
        rare_labels = TARGET_TYPE_RARE_LABELS
    elif task_name == 'action-type':
        rare_labels = ACTION_TYPE_RARE_LABELS
    else:
        rare_labels = ['rare_label_1', 'rare_label_2']
    
    # Create sample data
    data = []
    for strategy in strategies:
        for i in range(n_samples):
            row = {'strategy': strategy, 'incident_summary': f'Sample incident {i}'}
            
            # Generate realistic predictions for each rare label
            for label in rare_labels:
                # True labels (mostly 0, some 1s for rare labels)
                true_prob = 0.01 if label in ['ngos', 'mining_company', 'high_caste_landowner'] else 0.05
                true_label = np.random.binomial(1, true_prob)
                
                # Predicted probabilities (vary by strategy)
                if strategy == 'baseline':
                    pred_prob = np.random.beta(1, 10)  # Low probabilities
                elif strategy == 'threshold_tuned':
                    pred_prob = np.random.beta(2, 8)   # Slightly higher
                elif strategy == 'focal_tuned':
                    pred_prob = np.random.beta(3, 7)   # Higher for rare labels
                elif strategy == 'class_weights_tuned':
                    pred_prob = np.random.beta(4, 6)   # Even higher
                elif strategy == 'conservative_class_weights_tuned':
                    pred_prob = np.random.beta(3, 6)   # Moderate
                else:  # weighted_sampler_tuned
                    pred_prob = np.random.beta(2, 7)   # Moderate-low
                
                # Predicted labels (threshold at 0.5)
                pred_label = 1 if pred_prob > 0.5 else 0
                
                row[f'true_{label}'] = true_label
                row[f'pred_{label}'] = pred_label
                row[f'prob_{label}'] = pred_prob
            
            data.append(row)
    
    return pd.DataFrame(data)

def create_all_rare_labels_plots(predictions_df, task_name, save_dir=None):
    """
    Create all rare label plots for a given task.
    
    Args:
        predictions_df: DataFrame with prediction results
        task_name: Name of the task ('target-type' or 'action-type')
        save_dir: Directory to save plots (optional)
        
    Returns:
        List of (figure, axes) tuples
    """
    # Determine rare labels based on task
    if task_name == 'target-type':
        rare_labels = TARGET_TYPE_RARE_LABELS
    elif task_name == 'action-type':
        rare_labels = ACTION_TYPE_RARE_LABELS
    else:
        print(f"Unknown task: {task_name}")
        return []
    
    plots = []
    
    # 1. Overall rare labels plot
    save_path = f"{save_dir}/{task_name}_rare_labels_pr_plot.png" if save_dir else None
    fig1, ax1 = create_rare_labels_pr_plot(predictions_df, task_name, rare_labels, save_path=save_path)
    if fig1 is not None:
        plots.append((fig1, ax1))
    
    # 2. Strategy comparison plot
    save_path = f"{save_dir}/{task_name}_strategy_comparison_pr_plot.png" if save_dir else None
    fig2, ax2 = create_strategy_comparison_plot(predictions_df, task_name, rare_labels, save_path=save_path)
    if fig2 is not None:
        plots.append((fig2, ax2))
    
    # 3. Individual label plots for very rare labels (< 1%)
    very_rare_labels = [label for label in rare_labels if label in ['ngos', 'mining_company', 'non_maoist_armed_group', 'former_maoist', 'high_caste_landowner', 'businessman', 'aspiring_politician']]
    
    for label in very_rare_labels:
        save_path = f"{save_dir}/{task_name}_{label}_pr_plot.png" if save_dir else None
        fig, ax = create_label_focused_plot(predictions_df, task_name, label, save_path=save_path)
        if fig is not None:
            plots.append((fig, ax))
    
    return plots

def create_rare_labels_comparison_plots(predictions_df, task_name, rare_labels, 
                                       save_path=None, show_individual=False, 
                                       figsize_individual=(8, 6), figsize_combined=(16, 12)):
    """
    Create one plot per rare label comparing strategies, then combine them into a single figure.
    This is the streamlined approach that creates individual plots first, then combines them.
    
    Args:
        predictions_df: DataFrame with prediction results
        task_name: Name of the task (e.g., 'target-type', 'action-type')
        rare_labels: List of rare label names
        save_path: Path to save the combined plot (optional)
        show_individual: Whether to show individual plots
        figsize_individual: Figure size for individual plots
        figsize_combined: Figure size for combined plot
        
    Returns:
        tuple: (combined_figure, combined_axes, individual_plots)
    """
    individual_plots = []

    # Optionally create individual plots - one per rare label
    if show_individual:
        for label in rare_labels:
            fig, ax = create_label_focused_plot(
                predictions_df,
                task_name,
                label,
                figsize=figsize_individual
            )
            if fig is not None:
                individual_plots.append((fig, ax))
                plt.show()
        if not individual_plots:
            print("Warning: No individual plots were created")
    
    # Create combined figure directly (avoid matplotlib artist copying issues)
    n_labels = len(rare_labels)
    
    # Determine optimal subplot layout
    if n_labels <= 2:
        rows, cols = 1, n_labels
    elif n_labels <= 4:
        rows, cols = 2, 2
    elif n_labels <= 6:
        rows, cols = 2, 3
    elif n_labels <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 3  # For more than 9 labels
    
    # Create combined figure
    combined_fig, combined_axes = plt.subplots(rows, cols, figsize=figsize_combined)
    if n_labels == 1:
        combined_axes = [combined_axes]
    elif rows == 1 or cols == 1:
        combined_axes = combined_axes.flatten()
    else:
        combined_axes = combined_axes.flatten()
    
    # Set main title
    combined_fig.suptitle(f"Strategy Comparison on Rare Labels - {task_name.replace('-', ' ').title()}", 
                         fontsize=16, fontweight='bold')
    
    # Create plots directly in the combined figure
    for i, label in enumerate(rare_labels):
        if i >= len(combined_axes):
            break
            
        ax = combined_axes[i]
        
        # Extract metrics for the specific label
        metrics_df = extract_pr_metrics_from_predictions(predictions_df, [label])
        
        if metrics_df.empty:
            ax.text(0.5, 0.5, f'No data for {label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Precision-Recall for '{label}'")
            continue
        
        # Add iso-F1 contours
        create_iso_f1_contours(ax, f1_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        # Plot points for each strategy
        strategies = metrics_df['strategy'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
        
        for j, strategy in enumerate(strategies):
            strategy_data = metrics_df[metrics_df['strategy'] == strategy]
            ax.scatter(strategy_data['recall'], strategy_data['precision'], 
                      c=[colors[j]], marker='o', s=100, alpha=0.8,
                      label=strategy, edgecolors='black', linewidth=0.5)
        
        # Customize subplot
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title(f"Precision-Recall for '{label}'", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
        
        # Add legend for first subplot only (to avoid clutter)
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Hide unused subplots
    for i in range(len(rare_labels), len(combined_axes)):
        combined_axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path and combined_fig is not None:
        combined_fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path}")
    
    return combined_fig, combined_axes, individual_plots

def create_rare_labels_comparison_plot(predictions_df, task_name, rare_labels,
                                       save_path=None, show_individual=False,
                                       figsize_individual=(8, 6), figsize_combined=(16, 12)):
    """
    Backwards-compatible singular alias for create_rare_labels_comparison_plots.
    Returns the same 3-tuple (combined_fig, combined_axes, individual_plots).
    """
    return create_rare_labels_comparison_plots(
        predictions_df=predictions_df,
        task_name=task_name,
        rare_labels=rare_labels,
        save_path=save_path,
        show_individual=show_individual,
        figsize_individual=figsize_individual,
        figsize_combined=figsize_combined
    )