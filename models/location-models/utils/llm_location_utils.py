"""Utilities for location extraction using LLMs (non-tokenized string outputs)."""

import re
from .metrics_utils import parse_structured_location, fuzzy_match


def parse_location_from_llm(text: str) -> dict:
    """
    Parse location from LLM output (handles LLM-specific formatting artifacts).
    
    This function wraps parse_structured_location() and adds cleaning for common
    LLM output artifacts like code fences, markdown formatting, and conversational text.
    
    Args:
        text: Raw LLM output string
        
    Returns:
        dict with keys: state, district, village, other_locations (None if not present)
    """
    location_dict = {
        'state': None,
        'district': None,
        'village': None,
        'other_locations': None
    }
    
    if not text or not isinstance(text, str):
        return location_dict
    
    # Clean up common LLM artifacts
    text = str(text).strip()
    
    # Remove code fences (```text``` or ```json```)
    text = re.sub(r'```(?:text|json)?\s*([\s\S]*?)\s*```', r'\1', text, flags=re.IGNORECASE)
    text = text.replace('`', '')
    
    # Try to extract just the location part if there's conversational text
    # Look for the structured format: state: ..., district: ..., etc.
    match = re.search(
        r'(state:\s*[^,]+(?:,\s*district:\s*[^,]+)?(?:,\s*village:\s*[^,]+)?(?:,\s*other_locations:\s*[^,]+)?)',
        text,
        re.IGNORECASE
    )
    if match:
        text = match.group(1)
    
    # Parse using standard structured location parser
    try:
        return parse_structured_location(text)
    except:
        # Fallback: manual parsing
        parts = [part.strip() for part in text.split(',')]
        for part in parts:
            if ':' in part:
                label, value = part.split(':', 1)
                label = label.strip().lower()
                value = value.strip()
                if label in location_dict and value:
                    location_dict[label] = value
        return location_dict


def dict_to_structured_string(location_dict: dict) -> str:
    """
    Convert location dictionary back to structured string format.
    
    Args:
        location_dict: Dictionary with state, district, village, other_locations keys
        
    Returns:
        Structured string in format: "state: X, district: Y, village: Z, other_locations: W"
    """
    parts = []
    for key in ['state', 'district', 'village', 'other_locations']:
        if location_dict.get(key):
            parts.append(f"{key}: {location_dict[key]}")
    return ', '.join(parts) if parts else ''


def compute_location_metrics_from_strings(predictions: list, labels: list, fuzzy_threshold: int = 85) -> dict:
    """
    Compute comprehensive location extraction metrics from string predictions.
    
    This is the string-based version of metrics_utils.compute_metrics(), designed for
    LLM outputs that are already decoded strings (not token IDs).
    
    Args:
        predictions: List of predicted location strings
        labels: List of true location strings
        fuzzy_threshold: Similarity threshold for fuzzy matching (default: 85)
    
    Returns:
        dict with 'overall' and 'levels' keys:
        - overall: overall metrics (exact_match, exact_core_match, fuzzy_match, 
                  fuzzy_core_match, micro-averaged metrics, total_examples)
        - levels: per-level metrics (state, district, village, other_locations)
                  each with exact and fuzzy precision/recall/F1/support
    """
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
    fuzzy_core_matches = 0
    fuzzy_level_metrics = {
        'state': {'correct': 0, 'predicted': 0, 'total': 0},
        'district': {'correct': 0, 'predicted': 0, 'total': 0},
        'village': {'correct': 0, 'predicted': 0, 'total': 0},
        'other_locations': {'correct': 0, 'predicted': 0, 'total': 0}
    }
    
    total_examples = len(predictions)
    
    for pred_str, label_str in zip(predictions, labels):
        # Parse both strings
        pred_dict = parse_location_from_llm(pred_str)
        label_dict = parse_location_from_llm(label_str)
        
        # Check exact match for entire prediction
        if pred_dict == label_dict:
            exact_matches += 1
        
        # Check exact match for core geography
        core_exact_match = True
        for level in ['state', 'district', 'village']:
            if pred_dict[level] != label_dict[level]:
                core_exact_match = False
                break
        if core_exact_match:
            exact_core_matches += 1
        
        # Check fuzzy match for entire prediction
        all_levels_fuzzy_match = True
        for level in ['state', 'district', 'village', 'other_locations']:
            if not fuzzy_match(pred_dict[level], label_dict[level], threshold=fuzzy_threshold):
                all_levels_fuzzy_match = False
                break
        if all_levels_fuzzy_match:
            fuzzy_matches += 1
        
        # Check fuzzy match for core geography
        core_fuzzy_match = True
        for level in ['state', 'district', 'village']:
            if not fuzzy_match(pred_dict[level], label_dict[level], threshold=fuzzy_threshold):
                core_fuzzy_match = False
                break
        if core_fuzzy_match:
            fuzzy_core_matches += 1
        
        # Compute per-level metrics (EXACT)
        for level in ['state', 'district', 'village', 'other_locations']:
            if label_dict[level] is not None:
                exact_level_metrics[level]['total'] += 1
            if pred_dict[level] is not None:
                exact_level_metrics[level]['predicted'] += 1
            if pred_dict[level] is not None and label_dict[level] is not None:
                if pred_dict[level].lower() == label_dict[level].lower():
                    exact_level_metrics[level]['correct'] += 1
        
        # Compute per-level metrics (FUZZY)
        for level in ['state', 'district', 'village', 'other_locations']:
            if label_dict[level] is not None:
                fuzzy_level_metrics[level]['total'] += 1
            if pred_dict[level] is not None:
                fuzzy_level_metrics[level]['predicted'] += 1
            if pred_dict[level] is not None and label_dict[level] is not None:
                if fuzzy_match(pred_dict[level], label_dict[level], threshold=fuzzy_threshold):
                    fuzzy_level_metrics[level]['correct'] += 1
    
    # Compute overall metrics
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


def print_location_metrics(metrics: dict, model_name: str = "Model"):
    """
    Pretty print location extraction metrics (alias for metrics_utils.print_metrics).
    
    Args:
        metrics: Dictionary from compute_location_metrics_from_strings() with 'overall' and 'levels' keys
        model_name: Name of the model for display
    """
    # Import here to avoid circular dependency
    from .metrics_utils import print_metrics
    print_metrics(metrics, model_name)


def run_and_save_llm_location_results(
    model_name: str, 
    outputs: list, 
    df_input, 
    id_col: str, 
    output_dir,
    timing: dict = None
):
    """
    Parse LLM outputs, compute metrics, save results, and return dataframe with metrics.
    
    This is the standard workflow for evaluating LLM location extraction:
    1. Parse raw LLM outputs into structured format
    2. Compute comprehensive metrics (exact + fuzzy, per-level, micro-averaged)
    3. Save results CSV and metrics JSON
    4. Print metrics summary
    5. Return results dataframe and metrics dict
    
    Args:
        model_name: Model identifier (used for file naming and display)
        outputs: List of raw LLM output strings
        df_input: Input dataframe with incident data and true labels
        id_col: Name of ID column in dataframe
        output_dir: Directory to save results (Path or str)
        timing: Optional timing dictionary from time_inference_call (added to metrics)
    
    Returns:
        tuple: (results_df, metrics_dict)
            - results_df: DataFrame with predictions and true labels
            - metrics_dict: Dictionary with 'overall' and 'levels' metrics
    """
    import json
    import pandas as pd
    from pathlib import Path
    
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    
    # Parse predictions and convert to structured strings
    parsed_dicts = [parse_location_from_llm(s) for s in outputs]
    parsed_strings = [dict_to_structured_string(d) for d in parsed_dicts]
    
    # Get true labels
    true_labels = df_input['human_annotated_location'].values
    
    # Compute metrics
    metrics = compute_location_metrics_from_strings(parsed_strings, true_labels)
    
    # Add timing if provided
    if timing:
        metrics['timing'] = timing
        print(f"\n⏱️  Timing: {timing['total_time_seconds']:.2f}s total, "
              f"{timing['time_per_item_seconds']:.3f}s/incident, "
              f"{timing['throughput_items_per_second']:.2f} incidents/s")
    
    # Print metrics
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} Results")
    print(f"{'='*80}")
    print_location_metrics(metrics, model_name)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        id_col: df_input[id_col].values,
        'incident_summary': df_input['incident_summary'].values,
        'true_location': true_labels,
        f'{model_name}_raw': outputs,
        f'{model_name}_prediction': parsed_strings
    })
    
    # Save results
    results_path = output_dir / f"{model_name}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Saved results to: {results_path}")
    
    # Save metrics
    metrics_path = output_dir / f"{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to: {metrics_path}")
    
    return results_df, metrics

