#!/usr/bin/env python3
"""
Example script demonstrating the streamlined precision-recall visualization approach.

This script shows how to use the new create_rare_labels_comparison_plots function
to create one plot per rare label comparing strategies, then combine them into a single figure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import create_sample_predictions_data, create_rare_labels_comparison_plots

def main():
    print("=== Streamlined Precision-Recall Visualization Example ===\n")
    
    # Create sample data (replace with your actual predictions data)
    print("1. Creating sample predictions data...")
    target_predictions = create_sample_predictions_data('target-type', n_samples=500)
    print(f"   Created sample data: {target_predictions.shape}")
    
    # Define rare labels directly (as you suggested)
    TARGET_TYPE_RARE_LABELS = [
        'ngos', 'mining_company', 'non_maoist_armed_group', 'former_maoist',
        'high_caste_landowner', 'businessman', 'aspiring_politician'
    ]
    print(f"   Rare labels: {TARGET_TYPE_RARE_LABELS}")
    
    # Use the streamlined function
    print("\n2. Creating streamlined rare labels comparison plots...")
    combined_fig, combined_axes, individual_plots = create_rare_labels_comparison_plots(
        target_predictions,
        'target-type',
        TARGET_TYPE_RARE_LABELS,
        show_individual=True,  # Show individual plots
        figsize_individual=(8, 6),
        figsize_combined=(16, 12),
        save_path='rare_labels_strategy_comparison.png'  # Save the combined plot
    )
    
    if combined_fig is not None:
        print("✅ Successfully created streamlined plots!")
        print(f"   - Individual plots: {len(individual_plots)}")
        print(f"   - Combined figure subplots: {len(combined_axes)}")
        print(f"   - Combined plot saved to: rare_labels_strategy_comparison.png")
        
        # Show the combined plot
        plt.show()
        
        # Clean up
        plt.close(combined_fig)
        for fig, ax in individual_plots:
            plt.close(fig)
    else:
        print("❌ Failed to create plots")

if __name__ == "__main__":
    main()
