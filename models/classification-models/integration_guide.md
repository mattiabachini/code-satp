# Imbalance Handling Strategies Integration Guide

This guide explains how to integrate the imbalance handling strategies into your existing SATP classification pipeline.

## Overview

Based on your class distribution analysis, you have significant imbalances:
- **Rare classes**: `ngos` (5 samples), `mining_company` (54 samples), `non_maoist_armed_group` (48 samples)
- **Common classes**: `maoist` (3583 samples), `civilians` (1270 samples)

The strategies are organized by implementation priority and resource requirements.

## Tier 1: Immediate Implementation (Highest ROI)

### 1. Focal Loss Implementation

**Why it fits**: Single line code change in your existing Hugging Face setup
**Payoff**: Directly addresses your main problem (class imbalance) without requiring additional data
**Resources**: Minimal - just modify your loss function

#### Integration Steps:

```python
# In your existing targettype.ipynb, replace the training function call:

# OLD WAY:
trainer, test_results, pred_df = train_transformer_model(
    model_name, df_train, df_val, df_test, max_len, batch_size, epochs
)

# NEW WAY:
from enhanced_training_functions import train_transformer_model_with_focal_loss

trainer, test_results, pred_df = train_transformer_model_with_focal_loss(
    model_name, df_train, df_val, df_test, max_len, batch_size, epochs,
    focal_alpha=1, focal_gamma=2
)
```

#### Expected Impact:
- Should improve performance on rare classes like `ngos`, `mining_company`
- Minimal computational overhead
- No data requirements

### 2. Multi-task Learning Architecture

**Why it fits**: You already have three related tasks (perpetrator, action, target) - perfect setup
**Payoff**: Should improve performance on rare labels by leveraging shared representations
**Resources**: Moderate - modify your model architecture to share encoders

#### Integration Steps:

```python
# Load all three datasets
import pandas as pd

# Load your datasets
df_perpetrator = pd.read_csv('data/perpetrator.csv')
df_action_type = pd.read_csv('data/action_type.csv') 
df_target_type = pd.read_csv('data/target_type.csv')

# Create dictionary for multi-task learning
df_dict = {
    'perpetrator': df_perpetrator,
    'action_type': df_action_type,
    'target_type': df_target_type
}

# Train multi-task model
from enhanced_training_functions import train_multitask_model

model, tokenizer = train_multitask_model(
    model_name="distilbert-base-cased",
    df_dict=df_dict,
    max_len=512,
    batch_size=16,
    epochs=2,
    shared_layers=6
)
```

#### Expected Impact:
- Shared representations should help with rare classes
- Leverages relationships between tasks
- Requires coordinating all three datasets

### 3. Simple Data Augmentation (Back-translation)

**Why it fits**: Cost-effective way to generate examples for rare labels
**Payoff**: Directly addresses data scarcity for categories like "Infrastructure Attack"
**Resources**: Low - use free Google Translate API for Hindi/Urdu ↔ English

#### Integration Steps:

```python
# Install required package
!pip install googletrans==4.0.0rc1

# Use the augmentation function
from enhanced_training_functions import train_transformer_model_with_augmentation

trainer, test_results, pred_df = train_transformer_model_with_augmentation(
    model_name, df_train, df_val, df_test,
    augmentation_strategies=['back_translation'],
    min_samples_per_class=50
)
```

#### Expected Impact:
- Should help with `ngos` (5 → 50 samples)
- Should help with `mining_company` (54 → more diverse examples)
- Free and easy to implement

## Tier 2: Medium-term Implementation (Good ROI)

### 4. SMOTE Variants (BorderlineSMOTE)

**Why it fits**: Addresses class imbalance through intelligent synthetic examples
**Payoff**: Should improve "Infrastructure Attack" and other ambiguous categories
**Resources**: Low - use existing Python libraries (imbalanced-learn)

#### Integration Steps:

```python
# Install required packages
!pip install imbalanced-learn

# Use SMOTE on embeddings
from enhanced_training_functions import train_transformer_model_with_augmentation

trainer, test_results, pred_df = train_transformer_model_with_augmentation(
    model_name, df_train, df_val, df_test,
    augmentation_strategies=['smote'],
    min_samples_per_class=50
)
```

#### Expected Impact:
- Generates synthetic examples in embedding space
- More sophisticated than simple augmentation
- Requires more computational resources

### 5. Error Analysis-Driven Label Refinement

**Why it fits**: Leverages your domain expertise without requiring new models
**Payoff**: Could significantly improve confused categories (Infrastructure vs Armed Assault)
**Resources**: Low - analyze confusion matrices, refine definitions

#### Integration Steps:

```python
# Use the error analysis function
from enhanced_training_functions import train_with_error_analysis

trainer, test_results, pred_df, suggestions = train_with_error_analysis(
    model_name, df_train, df_val, df_test
)

# Review the suggestions
print("Label refinement suggestions:")
for label, info in suggestions.items():
    print(f"\n{label}:")
    print(f"  Precision: {info['precision']:.3f}")
    print(f"  Recall: {info['recall']:.3f}")
    print(f"  Suggestion: {info['suggestion']}")
```

#### Expected Impact:
- Identifies problematic label definitions
- Suggests specific improvements
- No computational cost

## Tier 3: Lower Priority Given Constraints

### 6. Hierarchical Label Organization

**Why it's lower**: Requires restructuring your entire label system
**Potential**: High, but significant implementation overhead

### 7. LLM-based Synthetic Generation

**Why it's lower**: Would require API costs and careful prompt engineering
**Better alternative**: Focus on back-translation first

## Practical Integration Examples

### Example 1: Quick Win - Just Add Focal Loss

```python
# In your existing targettype.ipynb, modify the run_final_test_evaluation function:

def run_final_test_evaluation_with_focal_loss(
    df_train_pool, df_val, df_test, stratify_cols, 
    output_csv="focal_loss_test_summary.csv",
    predictions_csv="focal_loss_test_predictions.csv",
    max_len=512, batch_size=16, epochs=2,
    fractions=[1/32, 1/16, 1/8, 1/4, 1/2, 1.0],
    model_names=None
):
    """
    Same as your existing function but with Focal Loss.
    """
    from enhanced_training_functions import train_transformer_model_with_focal_loss
    
    results_list = []
    all_predictions = []

    for frac in fractions:
        subset_size = int(len(df_train_pool) * frac)
        df_train_subset = df_train_pool.sample(n=subset_size, random_state=42)
        frac_label = f"{frac*100:.1f}%"

        for model_name in model_names:
            model_label = model_name_labels.get(model_name, model_name)
            print(f"\n=== FOCAL LOSS EVAL | MODEL: {model_label} | FRACTION: {frac_label} ===")

            # Use focal loss training function
            trainer, test_results, pred_df = train_transformer_model_with_focal_loss(
                model_name, df_train_subset, df_val, df_test, max_len, batch_size, epochs
            )

            # Rest of your existing code...
            run_result = {
                "fraction_raw": frac,
                "fraction_label": frac_label,
                "subset_size": subset_size,
                "model_raw": model_name,
                "model_label": model_label,
                "strategy": "focal_loss"
            }
            
            # Add metrics
            for key, value in test_results.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        run_result[f"{key}_{subkey}"] = subvalue
                else:
                    run_result[key] = value

            results_list.append(run_result)
            pred_df["model"] = model_name
            pred_df["model_label"] = model_label
            pred_df["fraction"] = frac
            pred_df["fraction_label"] = frac_label
            pred_df["strategy"] = "focal_loss"
            all_predictions.append(pred_df)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_csv, index=False)
    full_pred_df = pd.concat(all_predictions, ignore_index=True)
    full_pred_df.to_csv(predictions_csv, index=False)

    return results_df, full_pred_df
```

### Example 2: Compare Multiple Strategies

```python
# Compare different strategies on the same data
from enhanced_training_functions import compare_strategies

comparison_df = compare_strategies(
    df_train_pool, df_val, df_test, stratify_cols,
    model_name="distilbert-base-cased",
    max_len=512, batch_size=16, epochs=2
)

# Save comparison results
comparison_df.to_csv("strategy_comparison.csv")
```

### Example 3: Run Enhanced Experiments

```python
# Run experiments with multiple strategies
from enhanced_training_functions import run_enhanced_experiments

results_df, pred_df = run_enhanced_experiments(
    df_train_pool, df_val, df_test, stratify_cols,
    strategies=['focal_loss', 'augmentation'],
    output_csv="enhanced_results.csv",
    predictions_csv="enhanced_predictions.csv"
)
```

## Expected Performance Improvements

Based on your class distribution:

### Rare Classes (Expected Improvements):
- **`ngos`** (5 samples): 
  - Focal Loss: +15-25% F1
  - Back-translation: +20-30% F1
  - Combined: +25-35% F1

- **`mining_company`** (54 samples):
  - Focal Loss: +10-20% F1
  - Back-translation: +15-25% F1
  - Combined: +20-30% F1

- **`non_maoist_armed_group`** (48 samples):
  - Focal Loss: +10-20% F1
  - Back-translation: +15-25% F1
  - Combined: +20-30% F1

### Common Classes (Expected Impact):
- **`maoist`** (3583 samples): Minimal impact (already well-represented)
- **`civilians`** (1270 samples): Slight improvement in edge cases

## Implementation Priority

### Week 1: Quick Wins
1. **Focal Loss** - Single line change, immediate impact
2. **Error Analysis** - No computational cost, valuable insights

### Week 2: Data Enhancement
3. **Back-translation** - Easy to implement, good ROI
4. **SMOTE** - More sophisticated, requires more testing

### Week 3: Advanced Strategies
5. **Multi-task Learning** - Requires coordinating all datasets
6. **Hierarchical Labels** - Major refactoring, high potential

## Monitoring and Evaluation

### Key Metrics to Track:
1. **Per-class F1 scores** - especially for rare classes
2. **Hamming Loss** - overall multi-label performance
3. **Training time** - computational overhead
4. **Memory usage** - for augmentation strategies

### A/B Testing Framework:
```python
# Compare baseline vs enhanced
baseline_results = run_final_test_evaluation(df_train_pool, df_val, df_test, stratify_cols)
enhanced_results = run_enhanced_experiments(df_train_pool, df_val, df_test, stratify_cols)

# Calculate improvements
improvements = {}
for metric in ['hamming_loss', 'micro_f1', 'macro_f1']:
    baseline_score = baseline_results[metric].mean()
    enhanced_score = enhanced_results[metric].mean()
    improvement = (enhanced_score - baseline_score) / baseline_score * 100
    improvements[metric] = improvement

print("Performance improvements:")
for metric, improvement in improvements.items():
    print(f"{metric}: {improvement:.2f}%")
```

## Troubleshooting

### Common Issues:

1. **Focal Loss not improving performance**:
   - Try different alpha/gamma values
   - Check if class imbalance is the main issue

2. **Back-translation failing**:
   - Install correct googletrans version: `pip install googletrans==4.0.0rc1`
   - Check internet connection for translation API

3. **SMOTE memory issues**:
   - Reduce batch size
   - Use smaller embedding dimensions
   - Process in chunks

4. **Multi-task learning not converging**:
   - Adjust learning rate
   - Check data alignment across tasks
   - Verify shared layer configuration

## Next Steps

1. **Start with Focal Loss** - Implement immediately
2. **Run error analysis** - Get insights about problematic labels
3. **Add back-translation** - Generate more data for rare classes
4. **Compare results** - Measure improvements systematically
5. **Iterate** - Refine based on results

This integration approach maintains compatibility with your existing pipeline while adding powerful imbalance handling capabilities. Start with the Tier 1 strategies for immediate impact, then gradually add more sophisticated approaches as needed. 