# Conservative Class Weights Strategy

## Overview

The Conservative Class Weights strategy addresses a critical issue in imbalanced classification: **precision collapse** caused by extremely high class weights. Traditional class weighting uses raw `pos_weight = N/P` (negative samples / positive samples), which can lead to weights of 100+ for extremely rare classes, causing the model to over-predict positive classes and lose precision.

## The Problem

When using raw `pos_weight = N/P`:
- **Balanced classes** (50% positive): weight = 1.0 ✅
- **Imbalanced classes** (10% positive): weight = 9.0 ⚠️
- **Very imbalanced** (1% positive): weight = 99.0 ❌
- **Extremely imbalanced** (0.1% positive): weight = 999.0 ❌❌

Extreme weights cause:
- **Precision collapse**: Model over-predicts positive classes
- **Threshold sensitivity**: Small changes in threshold cause large precision swings
- **Poor calibration**: Model probabilities become unreliable
- **Validation instability**: Hard to tune hyperparameters

## The Solution

Conservative Class Weights apply two key transformations:

### 1. Capping (`cap_ratio`)
```python
capped_weights = min(raw_weights, cap_ratio)
```
- Prevents weights from exceeding a reasonable maximum
- Default: `cap_ratio = 3.0` (weights capped at 3x)
- Maintains relative ordering while preventing extremes

### 2. Square Root Scaling (`sqrt_scaling`)
```python
scaled_weights = sqrt(capped_weights)
```
- Reduces the impact of large weights while preserving ordering
- Helps balance recall gains with precision preservation
- Default: `sqrt_scaling = True`

### 3. Bounds (`min_weight`, `max_weight`)
```python
bounded_weights = clip(scaled_weights, min_weight, max_weight)
```
- Ensures numerical stability
- Default: `min_weight = 1.0`, `max_weight = 10.0`

## Example Transformation

```python
# Raw weights for a 4-class problem
raw_weights = [1.0, 9.0, 99.0, 999.0]

# Conservative weights (cap=3.0, sqrt=True, bounds=1.0-10.0)
conservative_weights = [1.0, 3.0, 1.73, 1.73]
#                    ↑     ↑     ↑      ↑
#                  Balanced |   Capped  |  Sqrt scaled & bounded
#                        Capped    Sqrt scaled
```

## Usage

### Basic Usage

```python
from imbalance_handling_strategies import compute_conservative_class_weights

# Default conservative weights
weights = compute_conservative_class_weights(df, label_cols)

# Custom parameters
weights = compute_conservative_class_weights(
    df, label_cols,
    cap_ratio=5.0,        # More aggressive capping
    sqrt_scaling=False,   # No sqrt scaling
    min_weight=0.5,       # Allow weights below 1.0
    max_weight=20.0       # Higher maximum weight
)
```

### Training Integration

```python
from strategy_experiments import train_with_conservative_class_weights

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
```

### Strategy Experiments

```python
# Add to your strategy experiments
strategies = [
    "baseline",
    "conservative_class_weights",  # New strategy
    "class_weights",               # Traditional approach
    "focal"
]

# Run experiments
results = run_strategy_experiments(
    df_train, df_val, df_test, label_cols,
    strategies=strategies
)
```

### Label-Specific Configurations

```python
from imbalance_handling_strategies import compute_label_specific_conservative_weights

# Different strategies for different labels
label_configs = {
    'common_label': {
        'cap_ratio': 2.0,      # Conservative for balanced classes
        'sqrt_scaling': True,   # Apply sqrt scaling
        'max_weight': 5.0      # Lower maximum
    },
    'rare_label': {
        'cap_ratio': 8.0,      # More aggressive for rare classes
        'sqrt_scaling': False,  # No sqrt scaling
        'max_weight': 25.0     # Higher maximum
    }
}

weights = compute_label_specific_conservative_weights(
    df, label_cols, label_configs
)
```

## Parameters

### `cap_ratio` (float, default: 3.0)
- **Purpose**: Maximum allowed weight ratio
- **Effect**: Prevents extreme weights that cause precision collapse
- **Recommendations**:
  - `1.0-2.0`: Very conservative, minimal precision impact
  - `3.0-5.0`: Balanced approach (recommended)
  - `6.0-10.0`: More aggressive, allows higher weights
  - `None`: No capping (not recommended)

### `sqrt_scaling` (bool, default: True)
- **Purpose**: Reduce impact of large weights while preserving ordering
- **Effect**: Helps balance recall gains with precision preservation
- **Recommendations**:
  - `True`: Recommended for most cases
  - `False`: Use when you want to preserve exact weight ratios

### `min_weight` (float, default: 1.0)
- **Purpose**: Minimum allowed weight value
- **Effect**: Ensures numerical stability
- **Recommendations**:
  - `1.0`: Standard approach
  - `0.5-0.9`: Allow under-weighting of common classes
  - `>1.0`: Ensure all classes get some minimum weighting

### `max_weight` (float, default: 10.0)
- **Purpose**: Maximum allowed weight value
- **Effect**: Prevents numerical instability
- **Recommendations**:
  - `5.0-10.0`: Conservative approach
  - `10.0-20.0`: Moderate approach
  - `>20.0`: Aggressive approach (use with caution)

## When to Use

### ✅ Use Conservative Weights When:
- You have extremely imbalanced classes (prevalence < 1%)
- Traditional class weights cause precision collapse
- You want to balance recall gains with precision preservation
- You need stable, tunable hyperparameters
- You're doing production deployment

### ❌ Don't Use When:
- Your classes are relatively balanced (prevalence > 10%)
- You specifically need the exact N/P ratio
- You're doing research on raw class weighting effects

## Performance Impact

### Training Time
- **Negligible**: Weight computation adds <1ms
- **Memory**: Same as traditional class weights
- **Convergence**: Often faster due to more stable gradients

### Model Performance
- **Recall**: Maintains most of the gains from class weighting
- **Precision**: Significantly better than raw class weights
- **F1 Score**: Usually improved due to better precision-recall balance
- **Calibration**: Better probability estimates

## Comparison with Alternatives

| Strategy | Recall | Precision | Stability | Tuning Ease |
|----------|--------|-----------|-----------|-------------|
| **No weights** | ❌ Low | ✅ High | ✅ High | ✅ Easy |
| **Raw pos_weight** | ✅ High | ❌ Low | ❌ Low | ❌ Hard |
| **Conservative weights** | ✅ High | ✅ Medium | ✅ High | ✅ Easy |
| **Focal Loss** | ✅ High | ✅ Medium | ✅ Medium | ⚠️ Medium |
| **Data augmentation** | ✅ High | ✅ High | ✅ High | ❌ Hard |

## Best Practices

### 1. Start Conservative
```python
# Start with default settings
weights = compute_conservative_class_weights(df, label_cols)

# Gradually increase if needed
weights = compute_conservative_class_weights(df, label_cols, cap_ratio=5.0)
```

### 2. Monitor Precision-Recall Trade-off
```python
# Check if you're getting the right balance
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Adjust cap_ratio based on results
if precision < 0.3:
    cap_ratio = max(2.0, cap_ratio - 1.0)  # More conservative
elif recall < 0.7:
    cap_ratio = min(10.0, cap_ratio + 1.0)  # More aggressive
```

### 3. Use Label-Specific Configs for Mixed Imbalance
```python
# Different strategies for different imbalance levels
label_configs = {}
for col in label_cols:
    prevalence = df[col].mean()
    if prevalence > 0.1:
        label_configs[col] = {'cap_ratio': 2.0, 'sqrt_scaling': True}
    elif prevalence > 0.01:
        label_configs[col] = {'cap_ratio': 5.0, 'sqrt_scaling': True}
    else:
        label_configs[col] = {'cap_ratio': 8.0, 'sqrt_scaling': False}
```

### 4. Combine with Other Strategies
```python
# Conservative weights + Focal Loss
alpha_pos = compute_conservative_class_weights(df, label_cols)
focal_loss = FocalLoss(alpha_pos=alpha_pos, gamma=2.0)

# Conservative weights + Data augmentation
augmented_df = augment_rare_classes(df, label_cols)
weights = compute_conservative_class_weights(augmented_df, label_cols)
```

## Troubleshooting

### Problem: Still getting low precision
**Solution**: Reduce `cap_ratio` or enable `sqrt_scaling`
```python
weights = compute_conservative_class_weights(
    df, label_cols, 
    cap_ratio=2.0,        # More conservative
    sqrt_scaling=True     # Enable sqrt scaling
)
```

### Problem: Not enough recall improvement
**Solution**: Increase `cap_ratio` or disable `sqrt_scaling`
```python
weights = compute_conservative_class_weights(
    df, label_cols, 
    cap_ratio=6.0,        # More aggressive
    sqrt_scaling=False    # Disable sqrt scaling
)
```

### Problem: Weights seem too extreme
**Solution**: Check your data and adjust bounds
```python
weights = compute_conservative_class_weights(
    df, label_cols, 
    min_weight=0.5,       # Allow lower weights
    max_weight=5.0        # Reduce maximum weight
)
```

## Research Context

This approach is inspired by research on:
- **Class imbalance handling** in deep learning
- **Precision-recall trade-offs** in imbalanced classification
- **Numerical stability** in training with extreme weights
- **Hyperparameter tuning** for production systems

The strategy balances theoretical correctness (maintaining class importance) with practical concerns (numerical stability and precision preservation).

## Future Enhancements

- **Adaptive capping**: Dynamically adjust `cap_ratio` based on validation performance
- **Multi-objective optimization**: Optimize weights for specific precision-recall targets
- **Cross-validation integration**: Automatically tune parameters across folds
- **Performance monitoring**: Track weight effectiveness over time
