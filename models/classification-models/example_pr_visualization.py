"""
Example script demonstrating precision-recall visualization utilities for imbalance handling strategies.

This script shows how to:
1. Create precision-recall scatter plots with iso-F1 contours for rare labels
2. Compare different imbalance handling strategies
3. Combine multiple plots into a single figure (similar to R's patchwork)
4. Generate comprehensive visualizations for both target-type and action-type tasks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add utils to path
sys.path.append('./utils')

# Import our visualization utilities
from utils import (
    # Precision-recall visualization
    create_rare_labels_pr_plot,
    create_strategy_comparison_plot,
    create_label_focused_plot,
    create_all_rare_labels_plots,
    TARGET_TYPE_RARE_LABELS,
    ACTION_TYPE_RARE_LABELS,
    
    # Plot combination utilities
    create_multi_panel_figure,
    create_comparison_grid,
    create_rare_labels_summary_plot,
    create_strategy_performance_summary,
    create_comprehensive_visualization,
    
    # File I/O utilities
    load_dataframe_csv,
    get_task_results_dir
)

def load_predictions_data(task_name):
    """
    Load predictions data for a given task.
    
    Args:
        task_name: Name of the task ('target-type' or 'action-type')
        
    Returns:
        DataFrame with predictions data
    """
    try:
        # Try to load from the strategy predictions file
        predictions_file = f"{task_name}_strategy_predictions.csv"
        predictions_df = load_dataframe_csv(predictions_file, task_name=task_name)
        print(f"✅ Loaded predictions data for {task_name}: {len(predictions_df)} rows")
        return predictions_df
    except Exception as e:
        print(f"❌ Could not load predictions data for {task_name}: {e}")
        return None

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

def demonstrate_individual_plots():
    """Demonstrate individual precision-recall plots."""
    print("\n" + "="*60)
    print("DEMONSTRATING INDIVIDUAL PRECISION-RECALL PLOTS")
    print("="*60)
    
    # Create sample data
    target_predictions = create_sample_predictions_data('target-type')
    action_predictions = create_sample_predictions_data('action-type')
    
    # 1. Target Type - All rare labels
    print("\n1. Creating target-type rare labels plot...")
    fig1, ax1 = create_rare_labels_pr_plot(
        target_predictions, 
        'target-type', 
        TARGET_TYPE_RARE_LABELS,
        title_suffix="(Sample Data)"
    )
    if fig1:
        plt.show()
        plt.close(fig1)
    
    # 2. Action Type - All rare labels
    print("\n2. Creating action-type rare labels plot...")
    fig2, ax2 = create_rare_labels_pr_plot(
        action_predictions, 
        'action-type', 
        ACTION_TYPE_RARE_LABELS,
        title_suffix="(Sample Data)"
    )
    if fig2:
        plt.show()
        plt.close(fig2)
    
    # 3. Strategy comparison for target-type
    print("\n3. Creating strategy comparison plot for target-type...")
    fig3, ax3 = create_strategy_comparison_plot(
        target_predictions,
        'target-type',
        TARGET_TYPE_RARE_LABELS
    )
    if fig3:
        plt.show()
        plt.close(fig3)
    
    # 4. Focus on a specific rare label
    print("\n4. Creating focused plot for 'ngos' label...")
    fig4, ax4 = create_label_focused_plot(
        target_predictions,
        'target-type',
        'ngos'
    )
    if fig4:
        plt.show()
        plt.close(fig4)

def demonstrate_plot_combination():
    """Demonstrate combining multiple plots into a single figure."""
    print("\n" + "="*60)
    print("DEMONSTRATING PLOT COMBINATION (R PATCHWORK EQUIVALENT)")
    print("="*60)
    
    # Create sample data
    target_predictions = create_sample_predictions_data('target-type')
    action_predictions = create_sample_predictions_data('action-type')
    
    # Create individual plots
    print("\n1. Creating individual plots...")
    
    # Target type plots
    fig1, ax1 = create_rare_labels_pr_plot(
        target_predictions, 'target-type', TARGET_TYPE_RARE_LABELS
    )
    fig2, ax2 = create_strategy_comparison_plot(
        target_predictions, 'target-type', TARGET_TYPE_RARE_LABELS
    )
    
    # Action type plots
    fig3, ax3 = create_rare_labels_pr_plot(
        action_predictions, 'action-type', ACTION_TYPE_RARE_LABELS
    )
    fig4, ax4 = create_strategy_comparison_plot(
        action_predictions, 'action-type', ACTION_TYPE_RARE_LABELS
    )
    
    # Combine into multi-panel figure
    print("\n2. Combining plots into multi-panel figure...")
    plots_data = [(fig1, ax1), (fig2, ax2), (fig3, ax3), (fig4, ax4)]
    titles = [
        "Target Type - Rare Labels",
        "Target Type - Strategy Comparison", 
        "Action Type - Rare Labels",
        "Action Type - Strategy Comparison"
    ]
    
    combined_fig, combined_axes = create_multi_panel_figure(
        plots_data,
        layout=(2, 2),  # 2 rows, 2 columns
        titles=titles,
        main_title="Comprehensive Analysis: Rare Label Performance Across Tasks and Strategies",
        figsize=(16, 12)
    )
    
    if combined_fig:
        plt.show()
        plt.close(combined_fig)
    
    # Alternative: Use the summary functions
    print("\n3. Using summary functions...")
    predictions_data = {
        'target-type': target_predictions,
        'action-type': action_predictions
    }
    
    # Rare labels summary
    fig_summary, axes_summary = create_rare_labels_summary_plot(
        predictions_data, 
        ['target-type', 'action-type']
    )
    if fig_summary:
        plt.show()
        plt.close(fig_summary)
    
    # Strategy performance summary
    fig_strategy, axes_strategy = create_strategy_performance_summary(
        predictions_data,
        ['target-type', 'action-type']
    )
    if fig_strategy:
        plt.show()
        plt.close(fig_strategy)

def demonstrate_comprehensive_visualization():
    """Demonstrate the comprehensive visualization function."""
    print("\n" + "="*60)
    print("DEMONSTRATING COMPREHENSIVE VISUALIZATION")
    print("="*60)
    
    # Create sample data
    predictions_data = {
        'target-type': create_sample_predictions_data('target-type'),
        'action-type': create_sample_predictions_data('action-type')
    }
    
    # Create output directory
    output_dir = "./pr_visualization_examples"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating comprehensive visualizations in: {output_dir}")
    
    # Generate all visualizations
    figures = create_comprehensive_visualization(predictions_data, output_dir)
    
    print(f"\nCreated {len(figures)} figure groups:")
    for name, fig_data in figures.items():
        if isinstance(fig_data, tuple):
            print(f"  - {name}: 1 figure")
        else:
            print(f"  - {name}: {len(fig_data)} figures")
    
    print(f"\nAll plots saved to: {output_dir}")

def demonstrate_with_real_data():
    """Demonstrate with real data if available."""
    print("\n" + "="*60)
    print("DEMONSTRATING WITH REAL DATA (IF AVAILABLE)")
    print("="*60)
    
    # Try to load real data
    target_predictions = load_predictions_data('target-type')
    action_predictions = load_predictions_data('action-type')
    
    if target_predictions is not None and action_predictions is not None:
        print("\n✅ Real data loaded successfully!")
        
        predictions_data = {
            'target-type': target_predictions,
            'action-type': action_predictions
        }
        
        # Create comprehensive visualization with real data
        output_dir = "./real_data_pr_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        figures = create_comprehensive_visualization(predictions_data, output_dir)
        
        print(f"\nCreated {len(figures)} figure groups with real data")
        print(f"Saved to: {output_dir}")
        
    else:
        print("\n⚠️ Real data not available, using sample data instead")
        demonstrate_comprehensive_visualization()

def main():
    """Main demonstration function."""
    print("PRECISION-RECALL VISUALIZATION UTILITIES DEMONSTRATION")
    print("="*60)
    print("This script demonstrates how to create precision-recall scatter plots")
    print("with iso-F1 contours for rare labels, comparing different imbalance")
    print("handling strategies, and combining plots into single figures.")
    print("="*60)
    
    # Set up matplotlib for better display
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    try:
        # Demonstrate individual plots
        demonstrate_individual_plots()
        
        # Demonstrate plot combination
        demonstrate_plot_combination()
        
        # Demonstrate comprehensive visualization
        demonstrate_comprehensive_visualization()
        
        # Try with real data if available
        demonstrate_with_real_data()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE!")
        print("="*60)
        print("\nKey features demonstrated:")
        print("1. ✅ Precision-recall scatter plots with iso-F1 contours")
        print("2. ✅ Focus on rare labels across imbalance strategies")
        print("3. ✅ Strategy comparison visualizations")
        print("4. ✅ Multi-panel figure creation (R patchwork equivalent)")
        print("5. ✅ Comprehensive visualization generation")
        print("6. ✅ Automatic plot saving and organization")
        
        print("\nNext steps:")
        print("- Use these utilities in your notebooks")
        print("- Customize colors, markers, and layouts as needed")
        print("- Integrate with your existing analysis pipeline")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
