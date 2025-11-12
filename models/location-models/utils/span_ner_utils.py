"""Utilities for span-based NER data processing and inference.

This module provides functions for:
- Converting span annotations to BIO/BIOES tags
- Tokenizing data with span alignment
- Non-Maximum Suppression (NMS) for overlapping predictions
- Extracting structured locations from NER predictions
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict


def spans_to_bio_tags(text: str, entities: List[Dict], tokenizer, label_list: List[str]) -> Tuple[List[str], List[int]]:
    """
    Convert character-offset spans to BIO tags aligned with tokenizer output.
    
    Args:
        text: Input text
        entities: List of entity dicts with 'start', 'end', 'label'
        tokenizer: HuggingFace tokenizer
        label_list: List of entity labels (e.g., ['STATE', 'DISTRICT', 'VILLAGE', 'OTHER_LOCATION'])
        
    Returns:
        Tuple of (tokens_list, tags_list) where tags use BIO encoding
    """
    # Create label to index mapping
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    
    # Tokenize
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
    tokens = encoding.tokens()
    offsets = encoding['offset_mapping']
    
    # Initialize all tags as 'O' (outside)
    tags = ['O'] * len(tokens)
    
    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda e: e['start'])
    
    # Assign BIO tags based on character offsets
    for entity in sorted_entities:
        entity_start = entity['start']
        entity_end = entity['end']
        label = entity['label']
        
        if label not in label_to_id:
            continue
        
        # Find tokens that overlap with this entity
        is_first_token = True
        for idx, (token_start, token_end) in enumerate(offsets):
            # Skip special tokens
            if token_start == token_end == 0:
                continue
            
            # Check if token overlaps with entity span
            if token_start < entity_end and token_end > entity_start:
                if is_first_token:
                    tags[idx] = f'B-{label}'
                    is_first_token = False
                else:
                    tags[idx] = f'I-{label}'
    
    return tokens, tags


def bio_tags_to_label_ids(tags: List[str], label_list: List[str]) -> List[int]:
    """
    Convert BIO tags to label IDs for model training.
    
    Args:
        tags: List of BIO tags (e.g., ['O', 'B-STATE', 'I-STATE', ...])
        label_list: List of entity labels
        
    Returns:
        List of label IDs
    """
    # Create comprehensive tag vocabulary
    tag_to_id = {'O': 0}
    current_id = 1
    
    for label in label_list:
        tag_to_id[f'B-{label}'] = current_id
        current_id += 1
        tag_to_id[f'I-{label}'] = current_id
        current_id += 1
    
    # Convert tags to IDs
    label_ids = []
    for tag in tags:
        if tag in tag_to_id:
            label_ids.append(tag_to_id[tag])
        else:
            # Unknown tag, treat as 'O'
            label_ids.append(0)
    
    return label_ids


def tokenize_and_align_labels(examples: Dict, tokenizer, label_list: List[str], 
                               max_length: int = 512) -> Dict:
    """
    Tokenize texts and align entity labels for NER training.
    
    Args:
        examples: Dict with 'text' and 'entities' keys (batch format from datasets)
        tokenizer: HuggingFace tokenizer
        label_list: List of entity labels
        max_length: Maximum sequence length
        
    Returns:
        Dict with tokenized inputs and aligned labels
    """
    texts = examples['text']
    entities_list = examples['entities']
    
    # Tokenize all texts
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # Will pad in data collator
        return_offsets_mapping=True
    )
    
    # Align labels for each example
    all_labels = []
    for i, (text, entities) in enumerate(zip(texts, entities_list)):
        offsets = tokenized['offset_mapping'][i]
        
        # Initialize labels as 'O' (-100 for special tokens)
        labels = []
        
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda e: e['start'])
        
        # Create label map for each character position
        char_to_label = {}
        for entity in sorted_entities:
            for pos in range(entity['start'], entity['end']):
                char_to_label[pos] = entity['label']
        
        # Assign labels to tokens based on character positions
        entity_start_positions = {e['start']: e['label'] for e in sorted_entities}
        
        for idx, (token_start, token_end) in enumerate(offsets):
            # Special tokens
            if token_start == token_end == 0:
                labels.append(-100)
                continue
            
            # Check if this token is in an entity
            token_labels = [char_to_label.get(pos) for pos in range(token_start, token_end) 
                          if pos in char_to_label]
            
            if not token_labels:
                labels.append(0)  # 'O' tag
            else:
                # Get most common label in token span
                label = max(set(token_labels), key=token_labels.count)
                label_idx = label_list.index(label) + 1 if label in label_list else 0
                
                # Determine if B- or I- tag
                is_start = token_start in entity_start_positions
                if is_start:
                    # B- tag: label_idx * 2 - 1
                    labels.append(label_idx * 2 - 1)
                else:
                    # I- tag: label_idx * 2
                    labels.append(label_idx * 2)
        
        all_labels.append(labels)
    
    tokenized['labels'] = all_labels
    
    # Remove offset mapping (not needed for training)
    tokenized.pop('offset_mapping')
    
    return tokenized


def get_label_list() -> List[str]:
    """Get the standard label list for location NER."""
    return ['STATE', 'DISTRICT', 'VILLAGE', 'OTHER_LOCATION']


def get_id_to_label_mapping(label_list: List[str] = None) -> Dict[int, str]:
    """
    Create mapping from label IDs to BIO tags.
    
    Args:
        label_list: List of entity labels (default: standard location labels)
        
    Returns:
        Dict mapping ID to tag (e.g., {0: 'O', 1: 'B-STATE', 2: 'I-STATE', ...})
    """
    if label_list is None:
        label_list = get_label_list()
    
    id_to_label = {0: 'O'}
    current_id = 1
    
    for label in label_list:
        id_to_label[current_id] = f'B-{label}'
        current_id += 1
        id_to_label[current_id] = f'I-{label}'
        current_id += 1
    
    return id_to_label


def predictions_to_entities(tokens: List[str], predictions: List[int], 
                            id_to_label: Dict[int, str]) -> List[Dict]:
    """
    Convert model predictions (BIO tags) to entity spans.
    
    Args:
        tokens: List of tokens
        predictions: List of predicted label IDs
        id_to_label: Mapping from ID to BIO tag
        
    Returns:
        List of entity dicts with 'text', 'label', 'start_token', 'end_token'
    """
    entities = []
    current_entity = None
    
    special_tokens = {'[CLS]', '[SEP]', '<s>', '</s>', '<pad>', '[PAD]', '[UNK]', '[BOS]', '[EOS]'}
    
    for idx, (token, pred_id) in enumerate(zip(tokens, predictions)):
        # Skip special tokens (close any open entity)
        if token in special_tokens:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        
        tag = id_to_label.get(pred_id, 'O')
        
        if tag == 'O':
            # Outside entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        
        elif tag.startswith('B-'):
            # Beginning of new entity
            if current_entity:
                entities.append(current_entity)
            
            label = tag[2:]  # Remove 'B-' prefix
            current_entity = {
                'text': token,
                'label': label,
                'start_token': idx,
                'end_token': idx + 1,
                'tokens': [token]
            }
        
        elif tag.startswith('I-'):
            # Inside entity
            label = tag[2:]  # Remove 'I-' prefix
            
            if current_entity and current_entity['label'] == label:
                # Continue current entity
                current_entity['text'] += ' ' + token
                current_entity['end_token'] = idx + 1
                current_entity['tokens'].append(token)
            else:
                # Start of new entity (malformed, treat as B-)
                if current_entity:
                    entities.append(current_entity)
                
                current_entity = {
                    'text': token,
                    'label': label,
                    'start_token': idx,
                    'end_token': idx + 1,
                    'tokens': [token]
                }
    
    # Add final entity if exists
    if current_entity:
        entities.append(current_entity)
    
    return entities


def non_maximum_suppression(entities: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping entity predictions.
    
    Args:
        entities: List of entities with 'start_token', 'end_token', and optional 'score'
        iou_threshold: IoU threshold for suppression (default: 0.5)
        
    Returns:
        Filtered list of entities
    """
    if not entities:
        return []
    
    # Sort by score (if available), otherwise by span length
    if 'score' in entities[0]:
        sorted_entities = sorted(entities, key=lambda e: e.get('score', 0), reverse=True)
    else:
        sorted_entities = sorted(entities, key=lambda e: e['end_token'] - e['start_token'], reverse=True)
    
    selected = []
    
    for entity in sorted_entities:
        # Check overlap with already selected entities
        should_add = True
        
        for sel in selected:
            # Calculate IoU
            start_max = max(entity['start_token'], sel['start_token'])
            end_min = min(entity['end_token'], sel['end_token'])
            
            if start_max < end_min:
                # There's overlap
                intersection = end_min - start_max
                union = (entity['end_token'] - entity['start_token']) + \
                       (sel['end_token'] - sel['start_token']) - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > iou_threshold:
                    should_add = False
                    break
        
        if should_add:
            selected.append(entity)
    
    return selected


def slot_fill_from_entities(entities: List[Dict]) -> Dict[str, Optional[str]]:
    """
    Convert NER entities to slot-filled location structure.
    
    Args:
        entities: List of entities with 'text', 'label', and optional 'score'
        
    Returns:
        Dict with slots: {"STATE": str, "DISTRICT": str, "VILLAGE": str, "OTHER_LOCATION": [str]}
    """
    # Group by label
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
    
    # Fill single-value slots (take first or highest scoring)
    for label in ['STATE', 'DISTRICT', 'VILLAGE']:
        if by_label[label]:
            if 'score' in by_label[label][0]:
                # Sort by score
                sorted_ents = sorted(by_label[label], key=lambda e: e.get('score', 0), reverse=True)
            else:
                # Just take first
                sorted_ents = by_label[label]
            
            slots[label] = sorted_ents[0]['text']
    
    # Collect other locations
    for entity in by_label['OTHER_LOCATION']:
        if entity['text'] not in slots['OTHER_LOCATION']:
            slots['OTHER_LOCATION'].append(entity['text'])
    
    # Add non-selected STATE/DISTRICT/VILLAGE to other_locations
    chosen_texts = {slots[k] for k in ['STATE', 'DISTRICT', 'VILLAGE'] if slots[k]}
    
    for label in ['STATE', 'DISTRICT', 'VILLAGE']:
        for entity in by_label[label]:
            if entity['text'] not in chosen_texts and entity['text'] not in slots['OTHER_LOCATION']:
                slots['OTHER_LOCATION'].append(entity['text'])
    
    return slots


def slots_to_structured_string(slots: Dict) -> str:
    """
    Convert slot-filled dict to structured string format.
    
    Args:
        slots: Dict with STATE, DISTRICT, VILLAGE, OTHER_LOCATION keys
        
    Returns:
        Structured location string (e.g., "state: X, district: Y, village: Z")
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


def compute_class_weights(ner_data: List[Dict], label_list: List[str] = None) -> Dict[int, float]:
    """
    Compute class weights for handling label imbalance.
    
    Args:
        ner_data: List of NER examples with 'entities'
        label_list: List of entity labels
        
    Returns:
        Dict mapping label ID to weight
    """
    if label_list is None:
        label_list = get_label_list()
    
    # Count occurrences of each label
    label_counts = defaultdict(int)
    label_counts['O'] = 0
    
    for label in label_list:
        label_counts[f'B-{label}'] = 0
        label_counts[f'I-{label}'] = 0
    
    # Count from data
    total_tokens = 0
    for example in ner_data:
        text = example['text']
        entities = example.get('entities', [])
        
        # Estimate token count (rough approximation)
        num_tokens = len(text.split())
        total_tokens += num_tokens
        
        # Count O tags (rough estimate)
        entity_tokens = sum(len(e['text'].split()) for e in entities)
        label_counts['O'] += (num_tokens - entity_tokens)
        
        # Count entity tags
        for entity in entities:
            label = entity['label']
            entity_len = len(entity['text'].split())
            label_counts[f'B-{label}'] += 1
            label_counts[f'I-{label}'] += (entity_len - 1)
    
    # Compute weights (inverse frequency)
    weights = {}
    for tag, count in label_counts.items():
        if count > 0:
            weights[tag] = total_tokens / (len(label_counts) * count)
        else:
            weights[tag] = 1.0
    
    # Convert to ID-based weights
    id_to_label = get_id_to_label_mapping(label_list)
    label_to_id = {v: k for k, v in id_to_label.items()}
    
    id_weights = {}
    for tag, weight in weights.items():
        if tag in label_to_id:
            id_weights[label_to_id[tag]] = weight
    
    return id_weights

