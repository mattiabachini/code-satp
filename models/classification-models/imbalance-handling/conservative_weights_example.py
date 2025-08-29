"""
Example: Using Conservative Class Weights for Imbalanced Classification

This script demonstrates how to use the new conservative class weights strategy
that caps and sqrt-scales raw pos_weight (N/P) to prevent precision collapse
while maintaining recall gains.

Key benefits:
1. Prevents extreme class weights that cause precision collapse
2. Maintains relative importance ordering between classes
3. Provides configurable capping and scaling parameters
4. Supports label-specific configurations
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys

# Add the parent directory to the path to import our functions
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from strategy_experiments import train_with_conservative_class_weights

# Import the conservative weight functions directly
from imbalance_handling_strategies import (
    compute_conservative_class_weights,
    compute_adaptive_conservative_weights,
    compute_label_specific_conservative_weights
)


def demonstrate_conservative_weights():
    """Demonstrate the conservative class weights approach with synthetic data."""
    
    # Create synthetic imbalanced dataset
    np.random.seed(42)
    n_samples = 1000
    n_labels = 4
    
    # Create very imbalanced labels (some extremely rare)
    labels = np.zeros((n_samples, n_labels))
    
    # Label 0: 50% positive (balanced)
    labels[:, 0] = np.random.binomial(1, 0.5, n_samples)
    
    # Label 1: 10% positive (imbalanced)
    labels[:, 1] = np.random.binomial(1, 0.1, n_samples)
    
    # Label 2: 1% positive (very imbalanced)
    labels[:, 2] = np.random.binomial(1, 0.01, n_samples)
    
    # Label 3: 0.1% positive (extremely imbalanced)
    labels[:, 3] = np.random.binomial(1, 0.001, n_samples)
    
    # Create synthetic text data
    texts = [f"Sample {i} with some text content" for i in range(n_samples)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'incident_summary': texts,
        'label_0': labels[:, 0],
        'label_1': labels[:, 1],
        'label_2': labels[:, 2],
        'label_3': labels[:, 3]
    })
    
    label_cols = ['label_0', 'label_1', 'label_2', 'label_3']
    
    print("=== Synthetic Dataset Statistics ===")
    for col in label_cols:
        pos_count = df[col].sum()
        neg_count = len(df) - pos_count
        ratio = neg_count / pos_count if pos_count > 0 else float('inf')
        print(f"{col}: {pos_count} positive, {neg_count} negative, N/P ratio: {ratio:.2f}")
    
    print("\n=== Conservative Class Weights Comparison ===")
    
    # 1. Raw pos_weight (N/P)
    P = df[label_cols].sum(axis=0).values.astype(np.float32)
    N = len(df) - P
    P = np.where(P == 0, 1.0, P)
    raw_weights = N / P
    print(f"Raw weights (N/P): {raw_weights}")
    
    # 2. Conservative weights with default settings
    conservative_weights = compute_conservative_class_weights(df, label_cols)
    print(f"Conservative weights (cap=3.0, sqrt=True): {conservative_weights.numpy()}")
    
    # 3. Conservative weights with custom settings
    custom_weights = compute_conservative_class_weights(
        df, label_cols, 
        cap_ratio=5.0, 
        sqrt_scaling=False, 
        min_weight=1.0, 
        max_weight=15.0
    )
    print(f"Custom weights (cap=5.0, sqrt=False, max=15.0): {custom_weights.numpy()}")
    
    # 4. Label-specific conservative weights
    label_configs = {
        'label_0': {'cap_ratio': 2.0, 'sqrt_scaling': True, 'max_weight': 5.0},      # Balanced class
        'label_1': {'cap_ratio': 4.0, 'sqrt_scaling': True, 'max_weight': 8.0},      # Imbalanced class
        'label_2': {'cap_ratio': 6.0, 'sqrt_scaling': False, 'max_weight': 12.0},    # Very imbalanced
        'label_3': {'cap_ratio': 8.0, 'sqrt_scaling': False, 'max_weight': 20.0}     # Extremely imbalanced
    }
    
    label_specific_weights = compute_label_specific_conservative_weights(
        df, label_cols, label_configs
    )
    print(f"Label-specific weights: {label_specific_weights.numpy()}")
    
    print("\n=== Weight Reduction Analysis ===")
    reduction_factors = raw_weights / conservative_weights.numpy()
    for i, col in enumerate(label_cols):
        print(f"{col}: Raw={raw_weights[i]:.1f} → Conservative={conservative_weights[i]:.1f} "
              f"(Reduction: {reduction_factors[i]:.1f}x)")
    
    return df, label_cols


def demonstrate_training_integration():
    """Demonstrate how to integrate conservative weights into training."""
    
    print("\n=== Training Integration Example ===")
    print("To use conservative class weights in training, you can:")
    print("1. Use the dedicated function:")
    print("   train_with_conservative_class_weights(...)")
    print("2. Or import and use the weight functions directly:")
    print("   from imbalance_handling_strategies import compute_conservative_class_weights")
    print("3. Or add to strategy experiments:")
    print("   strategies=['conservative_class_weights']")
    
    print("\n=== Example Usage ===")
    print("""
# Basic usage with default parameters
trainer, metrics, pred_df = train_with_conservative_class_weights(
    model_name="bert-base-uncased",
    df_train=df_train,
    df_val=df_val, 
    df_test=df_test,
    cap_ratio=3.0,        # Cap weights at 3x
    sqrt_scaling=True,    # Apply sqrt scaling
    min_weight=1.0,       # Minimum weight
    max_weight=10.0       # Maximum weight
)

# Custom parameters for specific use cases
trainer, metrics, pred_df = train_with_conservative_class_weights(
    model_name="bert-base-uncased",
    df_train=df_train,
    df_val=df_val,
    df_test=df_test,
    cap_ratio=5.0,        # More aggressive capping
    sqrt_scaling=False,   # No sqrt scaling
    min_weight=0.5,       # Allow weights below 1.0
    max_weight=20.0       # Higher maximum weight
)

# Label-specific configurations
label_configs = {
    'rare_label': {'cap_ratio': 8.0, 'sqrt_scaling': False, 'max_weight': 25.0},
    'common_label': {'cap_ratio': 2.0, 'sqrt_scaling': True, 'max_weight': 5.0}
}

trainer, metrics, pred_df = train_with_conservative_class_weights(
    model_name="bert-base-uncased",
    df_train=df_train,
    df_val=df_val,
    df_test=df_test,
    label_specific_configs=label_configs
)
    """)


def demonstrate_parameter_effects():
    """Demonstrate how different parameters affect the weights."""
    
    print("\n=== Parameter Effects Analysis ===")
    
    # Create a simple example with one very imbalanced class
    df_simple = pd.DataFrame({
        'incident_summary': [f"Sample {i}" for i in range(1000)],
        'rare_label': [1 if i < 10 else 0 for i in range(1000)]  # 1% positive
    })
    
    label_cols = ['rare_label']
    
    print("Dataset: 1000 samples, 10 positive (1%), 990 negative (99%)")
    print(f"Raw N/P ratio: {990/10:.1f}")
    
    # Test different cap_ratio values
    print("\nEffect of cap_ratio:")
    for cap in [1.0, 2.0, 3.0, 5.0, 10.0]:
        weights = compute_conservative_class_weights(
            df_simple, label_cols, 
            cap_ratio=cap, 
            sqrt_scaling=False
        )
        print(f"  cap_ratio={cap}: weight={weights[0]:.1f}")
    
    # Test sqrt scaling effect
    print("\nEffect of sqrt_scaling:")
    weights_no_sqrt = compute_conservative_class_weights(
        df_simple, label_cols, 
        cap_ratio=3.0, 
        sqrt_scaling=False
    )
    weights_with_sqrt = compute_conservative_class_weights(
        df_simple, label_cols, 
        cap_ratio=3.0, 
        sqrt_scaling=True
    )
    print(f"  No sqrt scaling: weight={weights_no_sqrt[0]:.1f}")
    print(f"  With sqrt scaling: weight={weights_with_sqrt[0]:.1f}")
    
    # Test min/max bounds
    print("\nEffect of min/max bounds:")
    weights_unbounded = compute_conservative_class_weights(
        df_simple, label_cols, 
        cap_ratio=3.0, 
        sqrt_scaling=True,
        min_weight=0.1,
        max_weight=100.0
    )
    weights_bounded = compute_conservative_class_weights(
        df_simple, label_cols, 
        cap_ratio=3.0, 
        sqrt_scaling=True,
        min_weight=1.0,
        max_weight=10.0
    )
    print(f"  Unbounded: weight={weights_unbounded[0]:.1f}")
    print(f"  Bounded (1.0-10.0): weight={weights_bounded[0]:.1f}")


if __name__ == "__main__":
    print("Conservative Class Weights Demonstration")
    print("=" * 50)
    
    # Demonstrate the core functionality
    df, label_cols = demonstrate_conservative_weights()
    
    # Show training integration
    demonstrate_training_integration()
    
    # Show parameter effects
    demonstrate_parameter_effects()
    
    print("\n" + "=" * 50)
    print("Conservative class weights help balance recall gains with precision preservation!")
    print("Use cap_ratio to prevent extreme weights and sqrt_scaling to reduce their impact.")
