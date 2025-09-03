"""
Plot Combination Utilities for Creating Multi-Panel Figures

This module provides utilities to combine multiple plots into a single figure,
similar to R's patchwork functionality.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np

def create_multi_panel_figure(plots_data, layout='auto', figsize=None, 
                            titles=None, main_title=None, 
                            save_path=None, dpi=300, **kwargs):
    """
    Create a multi-panel figure from multiple plot data.
    
    Args:
        plots_data: List of (fig, ax) tuples or list of axes objects
        layout: Layout specification - 'auto', (rows, cols), or 'custom'
        figsize: Figure size tuple (width, height)
        titles: List of subplot titles
        main_title: Main figure title
        save_path: Path to save the figure
        dpi: Resolution for saved figure
        **kwargs: Additional arguments passed to plt.subplots
        
    Returns:
        matplotlib figure and axes objects
    """
    n_plots = len(plots_data)
    
    # Determine layout
    if layout == 'auto':
        # Auto-determine best layout
        if n_plots <= 2:
            rows, cols = 1, n_plots
        elif n_plots <= 4:
            rows, cols = 2, 2
        elif n_plots <= 6:
            rows, cols = 2, 3
        elif n_plots <= 9:
            rows, cols = 3, 3
        else:
            # For more plots, use a more rectangular layout
            cols = int(np.ceil(np.sqrt(n_plots)))
            rows = int(np.ceil(n_plots / cols))
    elif isinstance(layout, (tuple, list)) and len(layout) == 2:
        rows, cols = layout
    else:
        raise ValueError("layout must be 'auto' or a tuple (rows, cols)")
    
    # Set default figure size
    if figsize is None:
        figsize = (cols * 6, rows * 5)
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize, **kwargs)
    
    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]
    elif rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Copy plots to subplots
    for i, (plot_fig, plot_ax) in enumerate(plots_data):
        if i >= len(axes):
            break
            
        # Get the current subplot
        current_ax = axes[i]
        
        # Copy plot elements from source to target
        copy_plot_elements(plot_ax, current_ax)
        
        # Set title if provided
        if titles and i < len(titles):
            current_ax.set_title(titles[i], fontsize=12, fontweight='bold')
        
        # Remove the original plot
        plt.close(plot_fig)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Add main title
    if main_title:
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    if main_title:
        plt.subplots_adjust(top=0.92)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Multi-panel figure saved to: {save_path}")
    
    return fig, axes

def copy_plot_elements(source_ax, target_ax):
    """
    Copy plot elements from source axes to target axes.
    
    Args:
        source_ax: Source matplotlib axes
        target_ax: Target matplotlib axes
    """
    # Clear target axes
    target_ax.clear()
    
    # Copy collections (scatter plots, etc.)
    for collection in source_ax.collections:
        target_ax.add_collection(collection)
    
    # Copy lines (contours, etc.)
    for line in source_ax.lines:
        target_ax.add_line(line)
    
    # Copy patches (rectangles, etc.)
    for patch in source_ax.patches:
        target_ax.add_patch(patch)
    
    # Copy text elements
    for text in source_ax.texts:
        target_ax.add_artist(text)
    
    # Copy images
    for image in source_ax.images:
        target_ax.add_image(image)
    
    # Copy legends
    if source_ax.get_legend():
        legend = source_ax.get_legend()
        target_ax.legend(legend.get_lines(), [t.get_text() for t in legend.get_texts()],
                        loc=legend._loc, bbox_to_anchor=legend.bbox_to_anchor)
    
    # Copy axis properties
    target_ax.set_xlim(source_ax.get_xlim())
    target_ax.set_ylim(source_ax.get_ylim())
    target_ax.set_xlabel(source_ax.get_xlabel())
    target_ax.set_ylabel(source_ax.get_ylabel())
    target_ax.set_title(source_ax.get_title())
    
    # Copy grid
    target_ax.grid(source_ax.grid, alpha=0.3)
    
    # Copy axis properties
    target_ax.set_aspect(source_ax.get_aspect())

def create_comparison_grid(plots_data, task_names, strategy_names=None, 
                          figsize=None, save_path=None, dpi=300):
    """
    Create a comparison grid for multiple tasks and strategies.
    
    Args:
        plots_data: Dictionary with structure {task_name: {strategy_name: (fig, ax)}}
        task_names: List of task names
        strategy_names: List of strategy names (if None, uses all available)
        figsize: Figure size
        save_path: Path to save the figure
        dpi: Resolution for saved figure
        
    Returns:
        matplotlib figure and axes objects
    """
    if strategy_names is None:
        # Get all strategy names from the first task
        first_task = task_names[0]
        strategy_names = list(plots_data[first_task].keys())
    
    n_tasks = len(task_names)
    n_strategies = len(strategy_names)
    
    # Set default figure size
    if figsize is None:
        figsize = (n_strategies * 5, n_tasks * 4)
    
    # Create figure
    fig, axes = plt.subplots(n_tasks, n_strategies, figsize=figsize)
    
    # Handle single row/column case
    if n_tasks == 1:
        axes = axes.reshape(1, -1)
    if n_strategies == 1:
        axes = axes.reshape(-1, 1)
    
    # Fill the grid
    for i, task in enumerate(task_names):
        for j, strategy in enumerate(strategy_names):
            if task in plots_data and strategy in plots_data[task]:
                plot_fig, plot_ax = plots_data[task][strategy]
                copy_plot_elements(plot_ax, axes[i, j])
                plt.close(plot_fig)
            else:
                axes[i, j].text(0.5, 0.5, f'No data\n{task}\n{strategy}', 
                              ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
    
    # Add row and column labels
    for i, task in enumerate(task_names):
        axes[i, 0].set_ylabel(task.replace('-', ' ').title(), fontsize=12, fontweight='bold')
    
    for j, strategy in enumerate(strategy_names):
        axes[0, j].set_title(strategy.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    
    # Add main title
    fig.suptitle('Strategy Comparison Across Tasks', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Comparison grid saved to: {save_path}")
    
    return fig, axes

def create_rare_labels_summary_plot(predictions_data, task_names, save_path=None, dpi=300):
    """
    Create a summary plot showing rare label performance across tasks and strategies.
    
    Args:
        predictions_data: Dictionary with structure {task_name: predictions_df}
        task_names: List of task names
        save_path: Path to save the figure
        dpi: Resolution for saved figure
        
    Returns:
        matplotlib figure and axes objects
    """
    from .pr_visualization_utils import (
        create_rare_labels_pr_plot, 
        TARGET_TYPE_RARE_LABELS, 
        ACTION_TYPE_RARE_LABELS
    )
    
    # Create individual plots for each task
    plots_data = {}
    
    for task_name in task_names:
        if task_name not in predictions_data:
            continue
            
        predictions_df = predictions_data[task_name]
        
        # Determine rare labels
        if task_name == 'target-type':
            rare_labels = TARGET_TYPE_RARE_LABELS
        elif task_name == 'action-type':
            rare_labels = ACTION_TYPE_RARE_LABELS
        else:
            continue
        
        # Create plot
        fig, ax = create_rare_labels_pr_plot(
            predictions_df, 
            task_name, 
            rare_labels,
            title_suffix="(All Strategies)"
        )
        
        if fig is not None:
            plots_data[task_name] = {'all_strategies': (fig, ax)}
    
    # Create multi-panel figure
    if plots_data:
        plot_list = [plots_data[task]['all_strategies'] for task in task_names if task in plots_data]
        titles = [f"{task.replace('-', ' ').title()} - Rare Labels" for task in task_names if task in plots_data]
        
        fig, axes = create_multi_panel_figure(
            plot_list,
            layout='auto',
            titles=titles,
            main_title="Precision-Recall Performance on Rare Labels Across Tasks",
            save_path=save_path,
            dpi=dpi
        )
        
        return fig, axes
    else:
        print("No plots created - check predictions data")
        return None, None

def create_strategy_performance_summary(predictions_data, task_names, save_path=None, dpi=300):
    """
    Create a summary plot showing strategy performance across tasks.
    
    Args:
        predictions_data: Dictionary with structure {task_name: predictions_df}
        task_names: List of task names
        save_path: Path to save the figure
        dpi: Resolution for saved figure
        
    Returns:
        matplotlib figure and axes objects
    """
    from .pr_visualization_utils import (
        create_strategy_comparison_plot,
        TARGET_TYPE_RARE_LABELS, 
        ACTION_TYPE_RARE_LABELS
    )
    
    # Create individual plots for each task
    plots_data = {}
    
    for task_name in task_names:
        if task_name not in predictions_data:
            continue
            
        predictions_df = predictions_data[task_name]
        
        # Determine rare labels
        if task_name == 'target-type':
            rare_labels = TARGET_TYPE_RARE_LABELS
        elif task_name == 'action-type':
            rare_labels = ACTION_TYPE_RARE_LABELS
        else:
            continue
        
        # Create plot
        fig, ax = create_strategy_comparison_plot(
            predictions_df, 
            task_name, 
            rare_labels
        )
        
        if fig is not None:
            plots_data[task_name] = {'strategy_comparison': (fig, ax)}
    
    # Create multi-panel figure
    if plots_data:
        plot_list = [plots_data[task]['strategy_comparison'] for task in task_names if task in plots_data]
        titles = [f"{task.replace('-', ' ').title()} - Strategy Comparison" for task in task_names if task in plots_data]
        
        fig, axes = create_multi_panel_figure(
            plot_list,
            layout='auto',
            titles=titles,
            main_title="Strategy Comparison on Rare Labels Across Tasks",
            save_path=save_path,
            dpi=dpi
        )
        
        return fig, axes
    else:
        print("No plots created - check predictions data")
        return None, None

# Example usage function
def create_comprehensive_visualization(predictions_data, output_dir=None):
    """
    Create a comprehensive set of visualizations for the imbalance handling strategies.
    
    Args:
        predictions_data: Dictionary with structure {task_name: predictions_df}
        output_dir: Directory to save all plots
        
    Returns:
        Dictionary of created figures
    """
    import os
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    figures = {}
    
    # 1. Rare labels summary across tasks
    save_path = f"{output_dir}/rare_labels_summary.png" if output_dir else None
    fig1, axes1 = create_rare_labels_summary_plot(
        predictions_data, 
        ['target-type', 'action-type'],
        save_path=save_path
    )
    if fig1 is not None:
        figures['rare_labels_summary'] = (fig1, axes1)
    
    # 2. Strategy performance summary
    save_path = f"{output_dir}/strategy_performance_summary.png" if output_dir else None
    fig2, axes2 = create_strategy_performance_summary(
        predictions_data,
        ['target-type', 'action-type'],
        save_path=save_path
    )
    if fig2 is not None:
        figures['strategy_performance_summary'] = (fig2, axes2)
    
    # 3. Individual task plots
    for task_name in ['target-type', 'action-type']:
        if task_name in predictions_data:
            from .pr_visualization_utils import create_all_rare_labels_plots
            
            task_plots = create_all_rare_labels_plots(
                predictions_data[task_name],
                task_name,
                save_dir=output_dir
            )
            figures[f'{task_name}_individual_plots'] = task_plots
    
    return figures
