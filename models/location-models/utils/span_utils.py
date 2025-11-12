"""Utilities for generating silver span annotations from structured location data.

This module provides functions to automatically align location names (state, district, village, 
other_locations) from structured columns back to the incident text, generating character-offset 
span annotations for NER training.

The alignment process follows a multi-step strategy:
1. Normalization: standardize both text and location names
2. Exact matching: find direct string matches
3. Relaxed matching: use fuzzy/token-based matching with context
4. Overlap resolution: handle overlapping spans with precedence rules
5. QC validation: check alignment quality
"""

import re
from typing import List, Dict, Tuple, Optional
from rapidfuzz import fuzz
import pandas as pd


def normalize_text(text: str) -> str:
    """
    Normalize text for matching: lowercase, collapse spaces, expand abbreviations.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Expand common abbreviations
    abbreviations = {
        r'\bdist\.?\b': 'district',
        r'\bdt\.?\b': 'district',
        r'\btaluk\b': 'taluka',
        r'\bmandal\b': 'mandalam',
        r'\bps\b': 'police station',
        r'\bst\.?\b': 'state',
    }
    for abbr, expansion in abbreviations.items():
        text = re.sub(abbr, expansion, text, flags=re.IGNORECASE)
    
    # Remove punctuation except spaces and hyphens
    text = re.sub(r'[^\w\s\-]', ' ', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def exact_match_span(text: str, location: str) -> Optional[Tuple[int, int]]:
    """
    Find exact match of location in text, return character offsets.
    
    Args:
        text: Text to search in
        location: Location string to find
        
    Returns:
        Tuple of (start, end) character offsets, or None if not found
    """
    if not location or not text:
        return None
    
    # Try exact match first (case-insensitive)
    pattern = re.escape(location)
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return (match.start(), match.end())
    
    return None


def normalized_match_span(text: str, location: str, original_text: str) -> Optional[Tuple[int, int]]:
    """
    Find match using normalized text, return offsets in original text.
    
    Args:
        text: Normalized text to search in
        location: Location string to find (normalized)
        original_text: Original unnormalized text
        
    Returns:
        Tuple of (start, end) character offsets in original text, or None
    """
    if not location or not text:
        return None
    
    norm_text = normalize_text(text)
    norm_location = normalize_text(location)
    
    if not norm_location:
        return None
    
    # Find in normalized text
    pattern = re.escape(norm_location)
    match = re.search(pattern, norm_text)
    
    if match:
        # Map back to original text offsets (approximate)
        # This is a simplified mapping - assumes whitespace compression is main change
        start_norm = match.start()
        end_norm = match.end()
        
        # Try to find the matched substring in original text near the normalized position
        matched_substr = text[start_norm:end_norm]
        window_start = max(0, start_norm - 10)
        window_end = min(len(original_text), end_norm + 10)
        window = original_text[window_start:window_end]
        
        # Search for similar substring in window
        for i in range(len(window) - len(matched_substr) + 1):
            candidate = window[i:i + len(matched_substr)]
            if normalize_text(candidate) == norm_location:
                return (window_start + i, window_start + i + len(candidate))
    
    return None


def fuzzy_match_span(text: str, location: str, threshold: int = 80, context_words: int = 3) -> Optional[Tuple[int, int, float]]:
    """
    Find fuzzy match using token-based similarity with context validation.
    
    Args:
        text: Text to search in
        location: Location string to find
        threshold: Minimum fuzzy match score (0-100)
        context_words: Number of words before/after to check for context cues
        
    Returns:
        Tuple of (start, end, score) character offsets and match score, or None
    """
    if not location or not text:
        return None
    
    norm_text = normalize_text(text)
    norm_location = normalize_text(location)
    
    if not norm_location:
        return None
    
    # Tokenize
    location_tokens = norm_location.split()
    text_words = norm_text.split()
    
    if not location_tokens or not text_words:
        return None
    
    best_match = None
    best_score = 0
    
    # Sliding window over text
    for i in range(len(text_words)):
        for j in range(i + 1, min(i + len(location_tokens) + 3, len(text_words) + 1)):
            window = ' '.join(text_words[i:j])
            score = fuzz.ratio(window, norm_location)
            
            if score >= threshold and score > best_score:
                # Check context for risky terms
                needs_context = any(term in norm_location for term in ['district', 'state', 'village'])
                
                if needs_context:
                    # Look for context cues nearby
                    context_start = max(0, i - context_words)
                    context_end = min(len(text_words), j + context_words)
                    context = ' '.join(text_words[context_start:context_end])
                    
                    # Check for geographic context indicators
                    has_context = any(indicator in context for indicator in [
                        'district', 'state', 'village', 'mandal', 'taluka', 
                        'area', 'region', 'territory', 'police station'
                    ])
                    
                    if not has_context:
                        continue
                
                # Convert token positions to character positions
                char_start = len(' '.join(text_words[:i]))
                if i > 0:
                    char_start += 1  # Add space
                char_end = char_start + len(window)
                
                best_match = (char_start, char_end, score)
                best_score = score
    
    # Prefer longer matches when scores are close
    if best_match:
        # Try to find actual positions in original text
        matched_text = norm_text[best_match[0]:best_match[1]]
        actual_match = re.search(re.escape(matched_text), text, flags=re.IGNORECASE)
        if actual_match:
            return (actual_match.start(), actual_match.end(), best_match[2])
        return best_match
    
    return None


def resolve_overlaps(entities: List[Dict]) -> List[Dict]:
    """
    Resolve overlapping spans using precedence rules and length preference.
    
    Precedence: STATE > DISTRICT > VILLAGE > OTHER_LOCATION
    If same label, prefer longer span.
    
    Args:
        entities: List of entity dicts with 'start', 'end', 'label', 'text'
        
    Returns:
        List of non-overlapping entities
    """
    if not entities:
        return []
    
    # Sort by start position, then by length (descending), then by label precedence
    label_precedence = {'STATE': 0, 'DISTRICT': 1, 'VILLAGE': 2, 'OTHER_LOCATION': 3}
    
    def sort_key(e):
        return (e['start'], -(e['end'] - e['start']), label_precedence.get(e['label'], 999))
    
    sorted_entities = sorted(entities, key=sort_key)
    
    # Remove overlaps using greedy selection
    selected = []
    for entity in sorted_entities:
        overlaps = False
        for sel in selected:
            # Check for overlap
            if not (entity['end'] <= sel['start'] or entity['start'] >= sel['end']):
                # There's an overlap
                # Keep the one with higher precedence
                if label_precedence.get(entity['label'], 999) < label_precedence.get(sel['label'], 999):
                    # Remove the selected one, add this one
                    selected.remove(sel)
                    overlaps = False
                    break
                elif label_precedence.get(entity['label'], 999) == label_precedence.get(sel['label'], 999):
                    # Same precedence, keep longer one
                    if (entity['end'] - entity['start']) > (sel['end'] - sel['start']):
                        selected.remove(sel)
                        overlaps = False
                        break
                    else:
                        overlaps = True
                        break
                else:
                    overlaps = True
                    break
        
        if not overlaps:
            selected.append(entity)
    
    # Sort by start position for output
    return sorted(selected, key=lambda e: e['start'])


def align_location_to_spans(row: pd.Series, original_text_col: str = 'incident_summary') -> List[Dict]:
    """
    Align all location fields from a row to character spans in the text.
    
    Args:
        row: Pandas Series with location columns and incident text
        original_text_col: Name of column containing the text
        
    Returns:
        List of entity dicts: [{'start': int, 'end': int, 'label': str, 'text': str}, ...]
    """
    text = str(row.get(original_text_col, ''))
    if not text or text == 'nan':
        return []
    
    entities = []
    
    # Process each location type
    location_fields = {
        'state': 'STATE',
        'district': 'DISTRICT', 
        'village_name': 'VILLAGE',
        'other_locations': 'OTHER_LOCATION'
    }
    
    for field, label in location_fields.items():
        value = row.get(field)
        if pd.isna(value) or not str(value).strip():
            continue
        
        value_str = str(value).strip()
        
        # Handle comma-separated values (for other_locations)
        if field == 'other_locations' and ',' in value_str:
            locations = [loc.strip() for loc in value_str.split(',') if loc.strip()]
        else:
            locations = [value_str]
        
        for location in locations:
            if not location:
                continue
            
            # Try exact match first
            span = exact_match_span(text, location)
            
            # Try normalized match
            if not span:
                span = normalized_match_span(text, location, text)
            
            # Try fuzzy match as fallback
            if not span:
                fuzzy_result = fuzzy_match_span(text, location, threshold=85)
                if fuzzy_result:
                    span = (fuzzy_result[0], fuzzy_result[1])
            
            if span:
                entities.append({
                    'start': span[0],
                    'end': span[1],
                    'label': label,
                    'text': text[span[0]:span[1]]
                })
    
    # Resolve overlaps
    entities = resolve_overlaps(entities)
    
    return entities


def create_ner_dataset(df: pd.DataFrame, text_col: str = 'incident_summary') -> List[Dict]:
    """
    Convert DataFrame with location columns to NER format dataset.
    
    Args:
        df: DataFrame with location columns (state, district, village_name, other_locations)
        text_col: Name of text column
        
    Returns:
        List of NER examples: [{'text': str, 'entities': [...]}, ...]
    """
    ner_data = []
    
    for idx, row in df.iterrows():
        text = str(row.get(text_col, ''))
        if not text or text == 'nan':
            continue
        
        entities = align_location_to_spans(row, text_col)
        
        # Build clean metadata with ground-truth fields to support evaluation later
        def _clean_val(v):
            return '' if pd.isna(v) or not str(v).strip() else str(v).strip()
        
        ner_data.append({
            'text': text,
            'entities': entities,
            'metadata': {
                'incident_number': str(row.get('incident_number', '')),
                'date': str(row.get('date', '')),
                'state': _clean_val(row.get('state', '')),
                'district': _clean_val(row.get('district', '')),
                'village_name': _clean_val(row.get('village_name', '')),
                'other_locations': _clean_val(row.get('other_locations', '')),
            }
        })
    
    return ner_data


def validate_ner_data(ner_data: List[Dict]) -> Dict:
    """
    Validate NER data quality and return statistics.
    
    Args:
        ner_data: List of NER examples
        
    Returns:
        Dict with validation statistics
    """
    stats = {
        'total_examples': len(ner_data),
        'examples_with_entities': 0,
        'examples_without_entities': 0,
        'total_entities': 0,
        'entities_by_label': {},
        'avg_entities_per_example': 0,
        'examples_with_all_types': 0,
    }
    
    for example in ner_data:
        entities = example.get('entities', [])
        
        if entities:
            stats['examples_with_entities'] += 1
            stats['total_entities'] += len(entities)
            
            # Count by label
            labels = set()
            for ent in entities:
                label = ent['label']
                stats['entities_by_label'][label] = stats['entities_by_label'].get(label, 0) + 1
                labels.add(label)
            
            # Check if has state, district, village
            if {'STATE', 'DISTRICT', 'VILLAGE'}.issubset(labels):
                stats['examples_with_all_types'] += 1
        else:
            stats['examples_without_entities'] += 1
    
    if stats['examples_with_entities'] > 0:
        stats['avg_entities_per_example'] = stats['total_entities'] / stats['examples_with_entities']
    
    return stats

