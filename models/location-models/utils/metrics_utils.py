"""Utilities for computing and displaying location extraction evaluation metrics."""

import numpy as np


def parse_structured_location(text):
    """Parse structured location string into a dictionary."""
    location_dict = {
        'state': None,
        'district': None,
        'village': None,
        'other_locations': None
    }
    
    if not text or text.strip() == '':
        return location_dict
    
    # Split by comma and parse each part
    parts = [part.strip() for part in text.split(',')]
    for part in parts:
        if ':' in part:
            label, value = part.split(':', 1)
            label = label.strip().lower()
            value = value.strip()
            if label in location_dict:
                location_dict[label] = value
    
    return location_dict


def fuzzy_match(str1, str2, threshold=85):
    """
    Check if two strings match with fuzzy matching (handles spelling variations).
    
    Returns True if:
    - Both strings are None (both fields are empty)
    - Both strings exist and have >= threshold similarity
    
    Returns False if:
    - Only one string is None (one field exists, the other doesn't)
    - Both strings exist but similarity < threshold
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        raise ImportError("rapidfuzz is required for fuzzy matching. Install with: pip install rapidfuzz")
    
    # Both None means both fields are empty - this is a match
    if str1 is None and str2 is None:
        return True
    # One is None but not the other - this is NOT a match
    if str1 is None or str2 is None:
        return False
    # Both exist - check fuzzy similarity
    return fuzz.ratio(str1.lower(), str2.lower()) >= threshold


def compute_metrics(predictions, labels, tokenizer, fuzzy_threshold=85):
    """
    Compute comprehensive location extraction metrics with both exact and fuzzy matching.
    
    Args:
        predictions: array of prediction token IDs from model
        labels: array of label token IDs (with -100 for padding)
        tokenizer: tokenizer used to decode predictions and labels
        fuzzy_threshold: similarity threshold for fuzzy matching (default: 85)
    
    Returns:
        dict with 'overall' and 'levels' keys:
        - overall: overall metrics (exact_match, exact_core_match, fuzzy_match, 
                  fuzzy_core_match, micro-averaged metrics, total_examples)
        - levels: per-level metrics (state, district, village, other_locations)
                  each with exact and fuzzy precision/recall/F1/support
    """
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = [
        tokenizer.decode([l for l in label if l != -100], skip_special_tokens=True)
        for label in labels
    ]
    
    # Initialize counters for EXACT matching
    exact_matches = 0
    exact_core_matches = 0  # state + district + village only
    exact_level_metrics = {
        'state': {'correct': 0, 'predicted': 0, 'total': 0},
        'district': {'correct': 0, 'predicted': 0, 'total': 0},
        'village': {'correct': 0, 'predicted': 0, 'total': 0},
        'other_locations': {'correct': 0, 'predicted': 0, 'total': 0}
    }
    
    # Initialize counters for FUZZY matching
    fuzzy_matches = 0
    fuzzy_core_matches = 0  # state + district + village only
    fuzzy_level_metrics = {
        'state': {'correct': 0, 'predicted': 0, 'total': 0},
        'district': {'correct': 0, 'predicted': 0, 'total': 0},
        'village': {'correct': 0, 'predicted': 0, 'total': 0},
        'other_locations': {'correct': 0, 'predicted': 0, 'total': 0}
    }
    
    total_examples = len(decoded_preds)
    
    for pred, label in zip(decoded_preds, decoded_labels):
        pred_dict = parse_structured_location(pred)
        label_dict = parse_structured_location(label)
        
        # Check exact match for entire prediction (all 4 fields)
        if pred_dict == label_dict:
            exact_matches += 1
        
        # Check exact match for core geographic hierarchy (state + district + village only)
        core_exact_match = True
        for level in ['state', 'district', 'village']:
            if pred_dict[level] != label_dict[level]:
                core_exact_match = False
                break
        if core_exact_match:
            exact_core_matches += 1
        
        # Check fuzzy match for entire prediction (all 4 fields must fuzzy match)
        all_levels_fuzzy_match = True
        for level in ['state', 'district', 'village', 'other_locations']:
            if not fuzzy_match(pred_dict[level], label_dict[level], threshold=fuzzy_threshold):
                all_levels_fuzzy_match = False
                break
        if all_levels_fuzzy_match:
            fuzzy_matches += 1
        
        # Check fuzzy match for core geographic hierarchy (state + district + village only)
        core_fuzzy_match = True
        for level in ['state', 'district', 'village']:
            if not fuzzy_match(pred_dict[level], label_dict[level], threshold=fuzzy_threshold):
                core_fuzzy_match = False
                break
        if core_fuzzy_match:
            fuzzy_core_matches += 1
        
        # Compute per-level metrics (EXACT)
        for level in ['state', 'district', 'village']:
            if label_dict[level] is not None:
                exact_level_metrics[level]['total'] += 1
            if pred_dict[level] is not None:
                exact_level_metrics[level]['predicted'] += 1
            if pred_dict[level] is not None and label_dict[level] is not None:
                if pred_dict[level].lower() == label_dict[level].lower():
                    exact_level_metrics[level]['correct'] += 1
        
        # Compute per-level metrics (FUZZY)
        for level in ['state', 'district', 'village']:
            if label_dict[level] is not None:
                fuzzy_level_metrics[level]['total'] += 1
            if pred_dict[level] is not None:
                fuzzy_level_metrics[level]['predicted'] += 1
            if pred_dict[level] is not None and label_dict[level] is not None:
                if fuzzy_match(pred_dict[level], label_dict[level], threshold=fuzzy_threshold):
                    fuzzy_level_metrics[level]['correct'] += 1
        
        # Special handling for other_locations (token-level F1) - EXACT
        if label_dict['other_locations'] is not None:
            label_tokens = set([t.strip().lower() for t in label_dict['other_locations'].split(',') if t.strip()])
            exact_level_metrics['other_locations']['total'] += len(label_tokens)
            
            if pred_dict['other_locations'] is not None:
                pred_tokens = set([t.strip().lower() for t in pred_dict['other_locations'].split(',') if t.strip()])
                exact_level_metrics['other_locations']['predicted'] += len(pred_tokens)
                exact_level_metrics['other_locations']['correct'] += len(pred_tokens & label_tokens)
        elif pred_dict['other_locations'] is not None:
            # Prediction exists but label is None (false positives)
            pred_tokens = set([t.strip().lower() for t in pred_dict['other_locations'].split(',') if t.strip()])
            exact_level_metrics['other_locations']['predicted'] += len(pred_tokens)
        
        # Special handling for other_locations (token-level F1) - FUZZY
        if label_dict['other_locations'] is not None:
            label_tokens = [t.strip() for t in label_dict['other_locations'].split(',') if t.strip()]
            fuzzy_level_metrics['other_locations']['total'] += len(label_tokens)
            
            if pred_dict['other_locations'] is not None:
                pred_tokens = [t.strip() for t in pred_dict['other_locations'].split(',') if t.strip()]
                fuzzy_level_metrics['other_locations']['predicted'] += len(pred_tokens)
                
                # For fuzzy matching, count matches using fuzzy string similarity
                matched_pred = set()
                matched_label = set()
                for p_idx, p_token in enumerate(pred_tokens):
                    for l_idx, l_token in enumerate(label_tokens):
                        if l_idx not in matched_label and fuzzy_match(p_token, l_token, threshold=fuzzy_threshold):
                            fuzzy_level_metrics['other_locations']['correct'] += 1
                            matched_pred.add(p_idx)
                            matched_label.add(l_idx)
                            break
        elif pred_dict['other_locations'] is not None:
            # Prediction exists but label is None (false positives)
            pred_tokens = [t.strip() for t in pred_dict['other_locations'].split(',') if t.strip()]
            fuzzy_level_metrics['other_locations']['predicted'] += len(pred_tokens)
    
    # Compute precision, recall, F1 for each level (EXACT)
    overall_metrics = {
        'exact_match': exact_matches / total_examples * 100,
        'exact_core_match': exact_core_matches / total_examples * 100,
        'fuzzy_match': fuzzy_matches / total_examples * 100,
        'fuzzy_core_match': fuzzy_core_matches / total_examples * 100,
        'total_examples': total_examples
    }
    
    # Compute micro-averaged metrics (EXACT)
    total_correct = sum(exact_level_metrics[level]['correct'] for level in ['state', 'district', 'village', 'other_locations'])
    total_predicted = sum(exact_level_metrics[level]['predicted'] for level in ['state', 'district', 'village', 'other_locations'])
    total_actual = sum(exact_level_metrics[level]['total'] for level in ['state', 'district', 'village', 'other_locations'])
    
    micro_exact_precision = (total_correct / total_predicted * 100) if total_predicted > 0 else 0
    micro_exact_recall = (total_correct / total_actual * 100) if total_actual > 0 else 0
    micro_exact_f1 = (2 * micro_exact_precision * micro_exact_recall / (micro_exact_precision + micro_exact_recall)) if (micro_exact_precision + micro_exact_recall) > 0 else 0
    
    overall_metrics['micro_exact_precision'] = micro_exact_precision
    overall_metrics['micro_exact_recall'] = micro_exact_recall
    overall_metrics['micro_exact_f1'] = micro_exact_f1
    
    # Compute micro-averaged metrics (FUZZY)
    total_correct = sum(fuzzy_level_metrics[level]['correct'] for level in ['state', 'district', 'village', 'other_locations'])
    total_predicted = sum(fuzzy_level_metrics[level]['predicted'] for level in ['state', 'district', 'village', 'other_locations'])
    total_actual = sum(fuzzy_level_metrics[level]['total'] for level in ['state', 'district', 'village', 'other_locations'])
    
    micro_fuzzy_precision = (total_correct / total_predicted * 100) if total_predicted > 0 else 0
    micro_fuzzy_recall = (total_correct / total_actual * 100) if total_actual > 0 else 0
    micro_fuzzy_f1 = (2 * micro_fuzzy_precision * micro_fuzzy_recall / (micro_fuzzy_precision + micro_fuzzy_recall)) if (micro_fuzzy_precision + micro_fuzzy_recall) > 0 else 0
    
    overall_metrics['micro_fuzzy_precision'] = micro_fuzzy_precision
    overall_metrics['micro_fuzzy_recall'] = micro_fuzzy_recall
    overall_metrics['micro_fuzzy_f1'] = micro_fuzzy_f1
    
    # Compute per-level metrics
    level_metrics = {}
    for level in ['state', 'district', 'village', 'other_locations']:
        level_metrics[level] = {}
        
        # Exact metrics
        exact_metrics = exact_level_metrics[level]
        exact_precision = (exact_metrics['correct'] / exact_metrics['predicted'] * 100) if exact_metrics['predicted'] > 0 else 0
        exact_recall = (exact_metrics['correct'] / exact_metrics['total'] * 100) if exact_metrics['total'] > 0 else 0
        exact_f1 = (2 * exact_precision * exact_recall / (exact_precision + exact_recall)) if (exact_precision + exact_recall) > 0 else 0
        
        level_metrics[level]['exact_precision'] = exact_precision
        level_metrics[level]['exact_recall'] = exact_recall
        level_metrics[level]['exact_f1'] = exact_f1
        level_metrics[level]['support'] = exact_metrics['total']
        
        # Fuzzy metrics
        fuzzy_metrics = fuzzy_level_metrics[level]
        fuzzy_precision = (fuzzy_metrics['correct'] / fuzzy_metrics['predicted'] * 100) if fuzzy_metrics['predicted'] > 0 else 0
        fuzzy_recall = (fuzzy_metrics['correct'] / fuzzy_metrics['total'] * 100) if fuzzy_metrics['total'] > 0 else 0
        fuzzy_f1 = (2 * fuzzy_precision * fuzzy_recall / (fuzzy_precision + fuzzy_recall)) if (fuzzy_precision + fuzzy_recall) > 0 else 0
        
        level_metrics[level]['fuzzy_precision'] = fuzzy_precision
        level_metrics[level]['fuzzy_recall'] = fuzzy_recall
        level_metrics[level]['fuzzy_f1'] = fuzzy_f1
    
    return {
        'overall': overall_metrics,
        'levels': level_metrics
    }


def print_metrics(metrics, model_name="Model"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary from compute_metrics() with 'overall' and 'levels' keys
        model_name: Name of the model for display
    """
    overall = metrics.get('overall', {})
    levels = metrics.get('levels', {})
    
    print("="*80)
    print(f"{model_name.upper()} EVALUATION RESULTS")
    print("="*80)
    print(f"Total Examples: {overall.get('total_examples', 0)}")
    print()
    
    # Overall accuracy (full record matches)
    print(f"{'FULL RECORD ACCURACY':<40} {'Exact Match':<15} {'Fuzzy Match':<15}")
    print("-" * 70)
    print(f"{'All fields correct (incl. other_locations)':<40} {overall.get('exact_match', 0):>6.2f}%        {overall.get('fuzzy_match', 0):>6.2f}%")
    print(f"{'Core geography correct (state+district+village)':<40} {overall.get('exact_core_match', 0):>6.2f}%        {overall.get('fuzzy_core_match', 0):>6.2f}%")
    print()
    print("Note: 'Core geography' excludes other_locations field to focus on the main")
    print("      administrative hierarchy (state → district → village).")
    print()
    
    # Micro-averaged metrics (aggregated across all fields)
    print(f"{'MICRO-AVERAGED METRICS':<40} {'Exact Match':<15} {'Fuzzy Match':<15}")
    print("(Aggregated across all fields: state, district, village, other_locations)")
    print("-" * 70)
    print(f"{'Precision':<40} {overall.get('micro_exact_precision', 0):>6.2f}%        {overall.get('micro_fuzzy_precision', 0):>6.2f}%")
    print(f"{'Recall':<40} {overall.get('micro_exact_recall', 0):>6.2f}%        {overall.get('micro_fuzzy_recall', 0):>6.2f}%")
    print(f"{'F1 Score':<40} {overall.get('micro_exact_f1', 0):>6.2f}%        {overall.get('micro_fuzzy_f1', 0):>6.2f}%")
    print()
    
    # Field-level metrics table
    print(f"{'FIELD-LEVEL METRICS':<20} {'Match Type':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Support':<10}")
    print("-" * 78)
    
    for level in ['state', 'district', 'village', 'other_locations']:
        if level in levels:
            level_display = level.replace('_', ' ').title()
            
            # Exact match row
            print(f"{level_display:<20} {'Exact':<12} "
                  f"{levels[level].get('exact_precision', 0):>6.2f}%     "
                  f"{levels[level].get('exact_recall', 0):>6.2f}%     "
                  f"{levels[level].get('exact_f1', 0):>6.2f}%     "
                  f"{levels[level].get('support', 0):>6.0f}")
            
            # Fuzzy match row
            print(f"{'':<20} {'Fuzzy':<12} "
                  f"{levels[level].get('fuzzy_precision', 0):>6.2f}%     "
                  f"{levels[level].get('fuzzy_recall', 0):>6.2f}%     "
                  f"{levels[level].get('fuzzy_f1', 0):>6.2f}%     "
                  f"{'':<10}")
            print()
    
    print("="*80)
    print("\nKEY INSIGHTS:")
    print("- Micro-averaged metrics: Aggregate performance across all field extractions")
    print("- Field-level metrics: Individual performance for each location component")
    print("- Support: Number of examples in test set with this field present")
    print("- Fuzzy matching: Allows for minor spelling variations (85% similarity threshold)")
    print("="*80)


def flatten_metrics_for_csv(metrics):
    """
    Flatten metrics dictionary for CSV export.
    
    Converts the nested structure (overall + levels) to a flat dictionary
    suitable for pandas DataFrame.
    
    Args:
        metrics: Dictionary from compute_metrics() with 'overall' and 'levels' keys
    
    Returns:
        dict: Flattened metrics with all values at top level
    """
    overall = metrics.get('overall', {})
    levels = metrics.get('levels', {})
    
    flat = {}
    
    # Add overall metrics
    flat.update(overall)
    
    # Add per-level metrics with prefixes
    for level in ['state', 'district', 'village', 'other_locations']:
        if level in levels:
            level_data = levels[level]
            flat[f'{level}_exact_precision'] = level_data.get('exact_precision', 0)
            flat[f'{level}_exact_recall'] = level_data.get('exact_recall', 0)
            flat[f'{level}_exact_f1'] = level_data.get('exact_f1', 0)
            flat[f'{level}_fuzzy_precision'] = level_data.get('fuzzy_precision', 0)
            flat[f'{level}_fuzzy_recall'] = level_data.get('fuzzy_recall', 0)
            flat[f'{level}_fuzzy_f1'] = level_data.get('fuzzy_f1', 0)
            flat[f'{level}_support'] = level_data.get('support', 0)
    
    return flat
