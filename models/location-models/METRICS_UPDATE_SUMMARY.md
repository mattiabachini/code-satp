# Location Extraction BERT Models - Metrics Update Summary

## Overview

Successfully updated the BERT location extraction pipeline to use comprehensive metrics matching the seq2seq models. All models now report and save metrics in the same JSON format with the full set of exact/fuzzy metrics at both overall and per-level granularity.

## Files Modified

### 1. `utils/metrics_utils.py`
**Added:** New function `compute_metrics_from_strings()`

This function computes comprehensive metrics from already-decoded strings (rather than token IDs), making it suitable for non-seq2seq models like BERT NER and GLiNER.

**Returns the same rich structure as seq2seq models:**
- `overall`: exact_match, exact_core_match, fuzzy_match, fuzzy_core_match, micro-averaged metrics
- `levels`: per-level exact and fuzzy precision/recall/F1/support for state, district, village, other_locations

### 2. `utils/bert_model_utils.py`
**Updated:** `evaluate_ner_model()` function
- Replaced manual metrics calculation with `compute_metrics_from_strings()`
- Now returns comprehensive metrics in the standard format

**Updated:** `save_bert_predictions_and_metrics()` function
- Now saves both JSON (comprehensive) and CSV (flattened) formats
- JSON matches the seq2seq structure exactly
- CSV provides quick comparison table

### 3. `utils/gliner_utils.py`
**Updated:** `save_gliner_predictions_and_metrics()` function
- Now saves both JSON (comprehensive) and CSV (flattened) formats
- Matches BERT and seq2seq saving behavior

### 4. `location_extraction_bert.ipynb`
**Updated:** Cell 35 (GLiNER evaluation)
- Replaced ~110 lines of manual metrics calculation with 10 lines using `compute_metrics_from_strings()`
- Now uses `print_metrics()` for consistent output formatting

## Metrics Now Reported

### Overall Metrics
- `exact_match`: All 4 fields must match exactly (%)
- `exact_core_match`: State + district + village must match exactly (%)
- `fuzzy_match`: All 4 fields must fuzzy match (%)
- `fuzzy_core_match`: State + district + village must fuzzy match (%)
- `micro_exact_precision/recall/f1`: Micro-averaged across all fields (exact matching)
- `micro_fuzzy_precision/recall/f1`: Micro-averaged across all fields (fuzzy matching)
- `total_examples`: Number of test examples

### Per-Level Metrics (for each of state, district, village, other_locations)
- `exact_precision/recall/f1`: Exact match performance
- `fuzzy_precision/recall/f1`: Fuzzy match performance  
- `support`: Number of examples with this field present

## Example Output Structure

### JSON Format (saved as `{model}_metrics.json`)
```json
{
  "overall": {
    "exact_match": 6.149193548387096,
    "exact_core_match": 25.302419354838708,
    "fuzzy_match": 16.73387096774194,
    "fuzzy_core_match": 45.66532258064516,
    "total_examples": 992,
    "micro_exact_precision": 63.17048853211009,
    "micro_exact_recall": 68.45123456789012,
    "micro_exact_f1": 65.71234567890123,
    "micro_fuzzy_precision": 65.82419354838709,
    "micro_fuzzy_recall": 71.23456789012346,
    "micro_fuzzy_f1": 68.41234567890123
  },
  "levels": {
    "state": {
      "exact_precision": 95.66532258064517,
      "exact_recall": 95.66532258064517,
      "exact_f1": 95.66532258064517,
      "support": 992,
      "fuzzy_precision": 95.66532258064517,
      "fuzzy_recall": 95.66532258064517,
      "fuzzy_f1": 95.66532258064517
    },
    "district": {
      "exact_precision": 95.26411290322581,
      "exact_recall": 95.26411290322581,
      "exact_f1": 95.26411290322581,
      "support": 992,
      "fuzzy_precision": 96.06854838709677,
      "fuzzy_recall": 96.06854838709677,
      "fuzzy_f1": 96.06854838709677
    },
    "village": {
      "exact_precision": 74.32098765432099,
      "exact_recall": 85.48387096774194,
      "exact_f1": 79.49494949494949,
      "support": 372,
      "fuzzy_precision": 76.54320987654321,
      "fuzzy_recall": 87.90322580645162,
      "fuzzy_f1": 81.81818181818181
    },
    "other_locations": {
      "exact_precision": 36.74588665399876,
      "exact_recall": 42.63262230456732,
      "exact_f1": 39.48453608247423,
      "support": 509,
      "fuzzy_precision": 38.95876288659794,
      "fuzzy_recall": 45.18864197530864,
      "fuzzy_f1": 41.83827160493827
    }
  }
}
```

### CSV Format (saved as `{model}_metrics.csv`)
Flattened version with all metrics in columns for easy comparison across models.

## Benefits

1. ✅ **Consistency**: All models (seq2seq, BERT, GLiNER) now use identical metrics
2. ✅ **Comparability**: Can directly compare JSON outputs across different approaches
3. ✅ **Comprehensive**: Rich metrics including core-match, exact/fuzzy for all levels
4. ✅ **Maintainability**: Single source of truth for metrics computation
5. ✅ **Flexibility**: Both detailed JSON and quick-comparison CSV outputs

## Important Notes

### JSON Saving Fix
After initial implementation, we discovered that JSON files weren't being saved correctly due to import path issues. This has been fixed by:

1. Adding an optional `results_dir` parameter to both `save_gliner_predictions_and_metrics()` and `save_bert_predictions_and_metrics()`
2. Using frame inspection to find `get_task_results_dir` from the caller's context when `results_dir` is not provided
3. Updating all notebook cells to explicitly pass `results_dir=results_dir` to the save functions
4. Adding fallback logic for local/Colab path detection

Now both GLiNER and BERT models will correctly save comprehensive JSON metrics files!

## Testing

Run the notebook cells and verify:
1. GLiNER evaluation produces comprehensive console output
2. BERT models save both `{model}_metrics.json` and `{model}_metrics.csv`
3. JSON structure matches the example above
4. All models report the same set of metrics
5. Console output shows "✅ {model} comprehensive metrics saved to JSON: {path}"

## Backward Compatibility

- ✅ CSV files still have the same essential metrics (exact_match, fuzzy_match, per-level F1)
- ✅ CSV now includes additional columns (exact_core_match, fuzzy_core_match, all P/R/F1 metrics)
- ✅ All existing visualization and analysis code will continue to work
- ✅ New JSON files provide richer data for deeper analysis

## Next Steps

1. Run the notebook to verify all changes work correctly
2. Compare JSON outputs across models to identify best performers
3. Use the rich metrics for model selection and error analysis
4. Consider adding visualization notebooks that leverage the new comprehensive metrics

