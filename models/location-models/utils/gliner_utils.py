"""Utilities for zero-shot location extraction using GLiNER with slot-filling.

GLiNER is a zero-shot NER model that can extract entities given natural language descriptions.
This module provides inference and slot-filling logic to convert GLiNER predictions to
the structured format used by other location extraction models.
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import re


# Label descriptions for GLiNER zero-shot inference
LOCATION_LABELS = {
    "STATE": "the Indian state or union territory where the event occurred",
    "DISTRICT": "the district (administrative area level 2) where the event occurred",
    "VILLAGE": "the village, town, or small settlement where the event occurred",
    "OTHER_LOCATION": "any other geographic place mentioned that is not state, district, or village"
}


def predict_entities_gliner(model, text: str, labels: Dict[str, str] = None, threshold: float = 0.4) -> List[Dict]:
    """
    Run GLiNER zero-shot entity prediction on text.
    
    Args:
        model: GLiNER model instance
        text: Input text to extract entities from
        labels: Dict mapping label names to descriptions (default: LOCATION_LABELS)
        threshold: Minimum confidence threshold (default: 0.4)
        
    Returns:
        List of entity dicts: [{"text": str, "label": str, "start": int, "end": int, "score": float}, ...]
    """
    if labels is None:
        labels = LOCATION_LABELS
    
    # GLiNER expects just the label descriptions as a list
    label_list = list(labels.keys())
    
    # Run prediction
    predictions = model.predict_entities(text, label_list, threshold=threshold)
    
    # Standardize output format
    entities = []
    for pred in predictions:
        entities.append({
            'text': pred.get('text', ''),
            'label': pred.get('label', ''),
            'start': pred.get('start', 0),
            'end': pred.get('end', 0),
            'score': pred.get('score', 0.0)
        })
    
    return entities


def apply_label_specific_thresholds(entities: List[Dict], thresholds: Dict[str, float] = None) -> List[Dict]:
    """
    Filter entities by label-specific confidence thresholds.
    
    Args:
        entities: List of entity predictions
        thresholds: Dict mapping labels to threshold values
                   (default: STATE=0.6, DISTRICT=0.55, VILLAGE=0.5, OTHER_LOCATION=0.4)
        
    Returns:
        Filtered list of entities
    """
    if thresholds is None:
        thresholds = {
            'STATE': 0.6,
            'DISTRICT': 0.55,
            'VILLAGE': 0.5,
            'OTHER_LOCATION': 0.4
        }
    
    filtered = []
    for entity in entities:
        label = entity['label']
        score = entity['score']
        threshold = thresholds.get(label, 0.4)
        
        if score >= threshold:
            filtered.append(entity)
    
    return filtered


def apply_context_boosting(entities: List[Dict], text: str, boost_amount: float = 0.1) -> List[Dict]:
    """
    Boost scores for entities that have supporting context nearby.
    
    Args:
        entities: List of entity predictions
        text: Original text
        boost_amount: Amount to boost score (default: 0.1)
        
    Returns:
        Entities with adjusted scores
    """
    text_lower = text.lower()
    
    boosted = []
    for entity in entities:
        score = entity['score']
        start = entity['start']
        end = entity['end']
        label = entity['label']
        
        # Get context window (±50 characters)
        context_start = max(0, start - 50)
        context_end = min(len(text), end + 50)
        context = text_lower[context_start:context_end]
        
        # Context indicators for each label type
        indicators = {
            'STATE': ['state', 'territory'],
            'DISTRICT': ['district', 'dt.', 'dist.'],
            'VILLAGE': ['village', 'town', 'settlement'],
            'OTHER_LOCATION': ['area', 'forest', 'police station', 'ps', 'region']
        }
        
        # Check for context indicators
        if label in indicators:
            for indicator in indicators[label]:
                if indicator in context:
                    score = min(1.0, score + boost_amount)
                    break
        
        entity_copy = entity.copy()
        entity_copy['score'] = score
        boosted.append(entity_copy)
    
    return boosted


def prefer_multitoken_entities(entities: List[Dict], boost_amount: float = 0.05) -> List[Dict]:
    """
    Boost multi-token entities when scores are close (tie-breaker).
    
    Args:
        entities: List of entity predictions
        boost_amount: Amount to boost multi-token entities (default: 0.05)
        
    Returns:
        Entities with adjusted scores
    """
    adjusted = []
    for entity in entities:
        text = entity['text']
        score = entity['score']
        
        # Count tokens (simple split on spaces)
        num_tokens = len(text.strip().split())
        
        # Boost if multi-token
        if num_tokens > 1:
            score = min(1.0, score + boost_amount)
        
        entity_copy = entity.copy()
        entity_copy['score'] = score
        adjusted.append(entity_copy)
    
    return adjusted


def normalize_location_text(text: str) -> str:
    """
    Normalize location text by removing common geographic descriptors.
    
    Strips common suffixes and prefixes like "District", "village", "State", etc.
    to match the ground truth format which uses simplified location names.
    
    Args:
        text: Raw location text extracted by GLiNER
        
    Returns:
        Normalized location name
    """
    if not text:
        return text
    
    # Common geographic descriptors to remove (case-insensitive)
    # Ordered from most specific to least specific
    descriptors = [
        # Full words with spaces - multi-word phrases first
        r'\s+Police\s+Station\b',
        r'\s+Police\s+camp\b',
        r'\s+Police\s+outpost\b',
        # Single word descriptors
        r'\s+District\b',
        r'\s+Dt\.?\b',
        r'\s+Dist\.?\b',
        r'\s+village\b',
        r'\s+Village\b',
        r'\s+town\b',
        r'\s+Town\b',
        r'\s+State\b',
        r'\s+mandal\b',
        r'\s+Mandal\b',
        r'\s+block\b',
        r'\s+Block\b',
        r'\s+area\b',
        r'\s+Area\b',
        r'\s+PS\b',
        r'\s+P\.S\.?\b',
        r'\s+panchayat\b',
        r'\s+Panchayat\b',
        r'\s+Tehsil\b',
        r'\s+tehsil\b',
        r'\s+Taluk\b',
        r'\s+taluk\b',
        r'\s+Division\b',
        r'\s+division\b',
        r'\s+camp\b',
        r'\s+outpost\b',
        r'\s+Outpost\b',
        r'\s+road\b',
        r'\s+Road\b',
        r'\s+forest\b',
        r'\s+Forest\b',
        r'\s+region\b',
        r'\s+Region\b',
        r'\s+locality\b',
        r'\s+Locality\b',
    ]
    
    normalized = text.strip()
    
    # Apply each pattern
    for pattern in descriptors:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
    
    # Clean up any extra whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized


def slot_fill_locations(entities: List[Dict]) -> Dict[str, Optional[str]]:
    """
    Convert entity predictions to slot-filled location structure.
    
    Applies slot-filling logic:
    - STATE, DISTRICT, VILLAGE: take top-1 prediction by score
    - OTHER_LOCATION: collect all remaining + explicit OTHER_LOCATION predictions
    - Normalizes location names by removing common geographic descriptors
    
    Args:
        entities: List of entity predictions with label, text, and score
        
    Returns:
        Dict with slots: {"STATE": str, "DISTRICT": str, "VILLAGE": str, "OTHER_LOCATION": [str]}
    """
    # Group entities by label
    by_label = defaultdict(list)
    for entity in entities:
        by_label[entity['label']].append(entity)
    
    # Initialize slots
    slots = {
        'STATE': None,
        'DISTRICT': None,
        'VILLAGE': None,
        'OTHER_LOCATION': []
    }
    
    # Fill single-value slots (take highest scoring)
    for label in ['STATE', 'DISTRICT', 'VILLAGE']:
        if by_label[label]:
            # Sort by score descending
            sorted_entities = sorted(by_label[label], key=lambda x: x['score'], reverse=True)
            best = sorted_entities[0]
            # Normalize the location text before storing
            slots[label] = normalize_location_text(best['text'])
    
    # Collect other locations
    # 1. Explicit OTHER_LOCATION predictions
    for entity in by_label['OTHER_LOCATION']:
        normalized_text = normalize_location_text(entity['text'])
        if normalized_text and normalized_text not in slots['OTHER_LOCATION']:
            slots['OTHER_LOCATION'].append(normalized_text)
    
    # 2. STATE/DISTRICT/VILLAGE predictions that weren't chosen (lower scoring alternatives)
    chosen_texts = {slots[k] for k in ['STATE', 'DISTRICT', 'VILLAGE'] if slots[k]}
    
    for label in ['STATE', 'DISTRICT', 'VILLAGE']:
        for entity in by_label[label]:
            normalized_text = normalize_location_text(entity['text'])
            # If this entity wasn't chosen and has decent score, add to other_locations
            if normalized_text and normalized_text not in chosen_texts and entity['score'] >= 0.4:
                if normalized_text not in slots['OTHER_LOCATION']:
                    slots['OTHER_LOCATION'].append(normalized_text)
    
    return slots


def slots_to_structured_string(slots: Dict[str, Optional[str]]) -> str:
    """
    Convert slot-filled dict to structured string format.
    
    Format: "state: X, district: Y, village: Z, other_locations: A, B"
    
    Args:
        slots: Dict with STATE, DISTRICT, VILLAGE, OTHER_LOCATION keys
        
    Returns:
        Structured location string
    """
    parts = []
    
    if slots.get('STATE'):
        parts.append(f"state: {slots['STATE']}")
    
    if slots.get('DISTRICT'):
        parts.append(f"district: {slots['DISTRICT']}")
    
    if slots.get('VILLAGE'):
        parts.append(f"village: {slots['VILLAGE']}")
    
    if slots.get('OTHER_LOCATION') and len(slots['OTHER_LOCATION']) > 0:
        other_locs = ', '.join(slots['OTHER_LOCATION'])
        parts.append(f"other_locations: {other_locs}")
    
    return ', '.join(parts) if parts else ''


def run_gliner_extraction(model, text: str, 
                          apply_boosting: bool = True,
                          apply_thresholds: bool = True) -> Tuple[str, Dict]:
    """
    Run complete GLiNER extraction pipeline on a single text.
    
    Args:
        model: GLiNER model instance
        text: Input text
        apply_boosting: Whether to apply context boosting and multi-token preference
        apply_thresholds: Whether to apply label-specific thresholds
        
    Returns:
        Tuple of (structured_string, slots_dict)
    """
    # Get raw predictions
    entities = predict_entities_gliner(model, text)
    
    # Apply boosting if requested
    if apply_boosting:
        entities = apply_context_boosting(entities, text)
        entities = prefer_multitoken_entities(entities)
    
    # Apply label-specific thresholds if requested
    if apply_thresholds:
        entities = apply_label_specific_thresholds(entities)
    
    # Slot-fill
    slots = slot_fill_locations(entities)
    
    # Convert to structured string
    structured = slots_to_structured_string(slots)
    
    return structured, slots


def batch_extract_locations(model, texts: List[str], 
                            show_progress: bool = True) -> List[Dict]:
    """
    Run GLiNER extraction on a batch of texts.
    
    Args:
        model: GLiNER model instance
        texts: List of input texts
        show_progress: Whether to show progress bar
        
    Returns:
        List of dicts with structured_location and slots
    """
    results = []
    
    if show_progress:
        try:
            from tqdm import tqdm
            texts = tqdm(texts, desc="GLiNER extraction")
        except ImportError:
            pass
    
    for text in texts:
        structured, slots = run_gliner_extraction(model, text)
        results.append({
            'structured_location': structured,
            'slots': slots,
            'text': text
        })
    
    return results


def save_gliner_predictions_and_metrics(
    model_name: str,
    predictions: List[str],
    ground_truth: List[str],
    metrics: Dict,
    test_data: List[Dict],
    task_name: str,
    save_dataframe_csv_func,
    results_dir = None,
) -> None:
    """
    Save GLiNER predictions and metrics to CSV and JSON files.
    
    This function provides a consistent interface for saving predictions,
    matching the behavior of save_bert_predictions_and_metrics.
    
    Args:
        model_name: Model identifier (e.g., 'gliner')
        predictions: List of predicted location strings
        ground_truth: List of ground truth location strings
        metrics: Dict of evaluation metrics
        test_data: Test dataset with metadata (incident_number, text, etc.)
        task_name: Task name for organizing results
        save_dataframe_csv_func: Function to save dataframes (from file_io)
        results_dir: Optional Path to results directory (will be inferred if not provided)
    """
    import pandas as pd
    import json
    from pathlib import Path
    
    # Save predictions with incident_number and incident_summary
    predictions_df = pd.DataFrame({
        'incident_number': [ex['metadata']['incident_number'] for ex in test_data],
        'incident_summary': [ex['text'] for ex in test_data],
        'ground_truth': ground_truth,
        'prediction': predictions,
    })
    save_dataframe_csv_func(predictions_df, f"{model_name}_predictions.csv", task_name)
    
    # Get results directory if not provided
    if results_dir is None:
        try:
            # Try to get get_task_results_dir from caller's frame
            import inspect
            frame = inspect.currentframe()
            caller_globals = frame.f_back.f_globals if frame and frame.f_back else {}
            
            if 'get_task_results_dir' in caller_globals:
                get_task_results_dir = caller_globals['get_task_results_dir']
                results_dir = get_task_results_dir(task_name)
            else:
                # Fallback: construct path based on typical Colab structure
                results_dir = Path("/content/drive/MyDrive/colab/satp-results") / task_name
                if not results_dir.exists():
                    # Try local path if Colab path doesn't exist
                    results_dir = Path(f"./results/{task_name}")
                results_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # Final fallback for local mode
            results_dir = Path(f"./results/{task_name}")
            results_dir.mkdir(parents=True, exist_ok=True)
    else:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comprehensive metrics to JSON (matching seq2seq format)
    metrics_path = results_dir / f"{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ {model_name} comprehensive metrics saved to JSON: {metrics_path}")
    
    print(f"✅ {model_name} predictions saved")

