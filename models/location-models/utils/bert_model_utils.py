"""Utilities for training and evaluating BERT-based span-NER models.

This module provides functions for:
- Training BERT models for span-based NER
- Running inference with NER models
- Converting NER predictions to structured location format
- Evaluating NER models using existing metrics
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import inspect
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
import pandas as pd

from .span_ner_utils import (
    tokenize_and_align_labels,
    get_label_list,
    get_id_to_label_mapping,
    predictions_to_entities,
    non_maximum_suppression,
    slot_fill_from_entities,
    slots_to_structured_string,
    compute_class_weights,
)
from .metrics_utils import compute_metrics, parse_structured_location


# Model configurations
BERT_MODEL_CONFIGS = {
    'confliBERT': {
        'model_name': 'snowood1/ConfliBERT-scr-cased',
        'description': 'BERT model pretrained on conflict-related text',
    },
    'deberta-v3': {
        'model_name': 'microsoft/deberta-v3-base',
        'description': 'DeBERTa v3 base model with improved efficiency',
    },
    'xlm-roberta': {
        'model_name': 'xlm-roberta-base',
        'description': 'Multilingual RoBERTa model',
    },
    'spanbert': {
        'model_name': 'SpanBERT/spanbert-base-cased',
        'description': 'SpanBERT optimized for span-based tasks',
    },
    'muril': {
        'model_name': 'google/muril-base-cased',
        'description': 'Multilingual Representations for Indian Languages',
    },
}


def prepare_ner_dataset(
    ner_data: List[Dict],
    tokenizer,
    label_list: List[str] = None,
    max_length: int = 512,
) -> Dataset:
    """
    Prepare NER data for training/evaluation.
    
    Args:
        ner_data: List of examples with 'text' and 'entities'
        tokenizer: HuggingFace tokenizer
        label_list: List of entity labels
        
    Returns:
        HuggingFace Dataset with tokenized inputs and labels
    """
    if label_list is None:
        label_list = get_label_list()
    
    # Convert to Dataset format
    texts = [ex['text'] for ex in ner_data]
    entities = [ex['entities'] for ex in ner_data]
    
    dataset = Dataset.from_dict({
        'text': texts,
        'entities': entities,
    })
    
    # Tokenize and align labels
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples, tokenizer, label_list, max_length=max_length
        ),
        batched=True,
        remove_columns=['text', 'entities'],
    )
    
    return tokenized_dataset


def train_span_ner_model(
    model_name: str,
    train_data: List[Dict],
    val_data: List[Dict],
    output_dir: str,
    label_list: List[str] = None,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    early_stopping_patience: int = 3,
    save_model: bool = True,
) -> Tuple[Any, Any, Dict]:
    """
    Train a BERT-based span-NER model.
    
    Args:
        model_name: HuggingFace model name or path
        train_data: List of training examples with 'text' and 'entities'
        val_data: List of validation examples
        output_dir: Directory to save model and checkpoints
        label_list: List of entity labels
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
        early_stopping_patience: Patience for early stopping
        save_model: Whether to save the trained model
        
    Returns:
        Tuple of (model, tokenizer, training_metrics)
    """
    if label_list is None:
        label_list = get_label_list()
    
    # Create label mappings
    # Label IDs: 0=O, 1=B-STATE, 2=I-STATE, 3=B-DISTRICT, 4=I-DISTRICT, ...
    id_to_label = get_id_to_label_mapping(label_list)
    label_to_id = {v: k for k, v in id_to_label.items()}
    num_labels = len(id_to_label)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
    )
    
    # Prepare datasets
    train_dataset = prepare_ner_dataset(
        train_data, tokenizer, label_list, max_length=max_length
    )
    val_dataset = prepare_ner_dataset(
        val_data, tokenizer, label_list, max_length=max_length
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Training arguments
    # Build kwargs compatibly across transformers versions
    candidate_kwargs: Dict[str, Any] = {
        "output_dir": output_dir,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": num_epochs,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "save_total_limit": 2,
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 50,
        "report_to": ["tensorboard"],
    }
    # Filter unsupported args for older/newer transformers versions
    accepted_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted_params}
    training_args = TrainingArguments(**filtered_kwargs)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    train_result = trainer.train()
    
    # Save model if requested
    if save_model:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    # Get training metrics
    training_metrics = {
        'train_loss': train_result.training_loss,
        'train_runtime': train_result.metrics.get('train_runtime', 0),
        'train_samples_per_second': train_result.metrics.get('train_samples_per_second', 0),
    }
    
    return model, tokenizer, training_metrics


def predict_ner_batch(
    model,
    tokenizer,
    texts: List[str],
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    max_length: int = 512,
    apply_nms: bool = True,
    nms_threshold: float = 0.5,
) -> List[List[Dict]]:
    """
    Run NER predictions on a batch of texts.
    
    Args:
        model: Trained NER model
        tokenizer: Corresponding tokenizer
        texts: List of input texts
        device: Device to run on
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        apply_nms: Whether to apply Non-Maximum Suppression
        nms_threshold: IoU threshold for NMS
        
    Returns:
        List of entity lists (one per text)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    all_entities = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors='pt',
            return_offsets_mapping=True,
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        offsets = encodings['offset_mapping']
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            # Force 'O' on padding positions to avoid entities over [PAD]
            predictions = predictions.masked_fill(attention_mask == 0, 0)
        
        # Convert to entities
        for j, (pred, text) in enumerate(zip(predictions, batch_texts)):
            pred_ids = pred.cpu().numpy().tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[j])
            
            # Get id to label mapping
            id_to_label = model.config.id2label
            
            # Convert predictions to entities
            entities = predictions_to_entities(tokens, pred_ids, id_to_label)
            
            # Apply NMS if requested
            if apply_nms and entities:
                entities = non_maximum_suppression(entities, iou_threshold=nms_threshold)
            # Post-process entity text using character offsets to avoid subword artifacts
            if entities:
                # Offsets are on CPU; ensure list of tuples
                example_offsets = offsets[j].tolist()
                original_text = batch_texts[j]
                for ent in entities:
                    start_idx = ent.get('start_token', 0)
                    end_idx = max(ent.get('end_token', start_idx + 1) - 1, 0)
                    
                    # Skip special tokens (offset (0, 0))
                    while start_idx < len(example_offsets) and tuple(example_offsets[start_idx]) == (0, 0):
                        start_idx += 1
                    while end_idx >= 0 and tuple(example_offsets[end_idx]) == (0, 0):
                        end_idx -= 1
                    
                    if 0 <= start_idx < len(example_offsets) and 0 <= end_idx < len(example_offsets) and start_idx <= end_idx:
                        char_start = int(example_offsets[start_idx][0])
                        char_end = int(example_offsets[end_idx][1])
                        if 0 <= char_start < len(original_text) and 0 < char_end <= len(original_text) and char_end > char_start:
                            ent['text'] = original_text[char_start:char_end].strip()
                        else:
                            # Fallback to decoding tokens if offsets look odd
                            try:
                                token_span_ids = input_ids[j][ent['start_token']:ent['end_token']]
                                ent['text'] = tokenizer.decode(token_span_ids, skip_special_tokens=True).strip()
                            except Exception:
                                # Keep existing token-joined text
                                pass
            all_entities.append(entities)
    
    return all_entities


def predict_structured_locations(
    model,
    tokenizer,
    texts: List[str],
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    max_length: int = 512,
    apply_nms: bool = True,
) -> List[str]:
    """
    Predict structured location strings from texts using NER model.
    
    Args:
        model: Trained NER model
        tokenizer: Corresponding tokenizer
        texts: List of input texts
        device: Device to run on
        batch_size: Batch size for inference
        apply_nms: Whether to apply NMS
        
    Returns:
        List of structured location strings
    """
    # Get entities
    all_entities = predict_ner_batch(
        model,
        tokenizer,
        texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        apply_nms=apply_nms,
    )
    
    # Convert to structured strings
    structured_locations = []
    for entities in all_entities:
        # Slot-fill
        slots = slot_fill_from_entities(entities)
        
        # Convert to structured string
        structured = slots_to_structured_string(slots)
        structured_locations.append(structured)
    
    return structured_locations


def evaluate_ner_model(
    model,
    tokenizer,
    test_data: List[Dict],
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    max_length: int = 512,
    fuzzy_threshold: int = 85,
) -> Dict[str, Any]:
    """
    Evaluate NER model and compute metrics compatible with seq2seq models.
    
    Args:
        model: Trained NER model
        tokenizer: Corresponding tokenizer
        test_data: List of test examples with 'text' and ground truth location fields
        device: Device to run on
        batch_size: Batch size
        fuzzy_threshold: Threshold for fuzzy matching
        
    Returns:
        Dict with metrics and predictions
    """
    # Extract texts
    texts = [ex['text'] for ex in test_data]
    
    # Get predictions
    predicted_structured = predict_structured_locations(
        model,
        tokenizer,
        texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    
    # Build ground truth structured strings
    # Import data_utils to use build_structured_location
    from .data_utils import build_structured_location
    
    ground_truth = []
    for ex in test_data:
        # Create a row-like dict for build_structured_location
        row_dict = {
            'state': ex.get('metadata', {}).get('state', ''),
            'district': ex.get('metadata', {}).get('district', ''),
            'village_name': ex.get('metadata', {}).get('village_name', ''),
            'other_locations': ex.get('metadata', {}).get('other_locations', ''),
        }
        # Convert to Series for compatibility
        import pandas as pd
        row = pd.Series(row_dict)
        gt_structured = build_structured_location(row)
        ground_truth.append(gt_structured)
    
    # Compute metrics using existing function
    # Note: compute_metrics expects token IDs, but we'll adapt it
    # For now, we'll compute metrics directly from strings
    from .metrics_utils import (
        parse_structured_location,
        fuzzy_match,
    )
    
    # Compute metrics manually
    exact_matches = 0
    fuzzy_matches = 0
    
    # Per-level metrics
    level_metrics = {
        'state': {'tp': 0, 'fp': 0, 'fn': 0},
        'district': {'tp': 0, 'fp': 0, 'fn': 0},
        'village': {'tp': 0, 'fp': 0, 'fn': 0},
        'other_locations': {'tp': 0, 'fp': 0, 'fn': 0},
    }
    
    for pred_str, gt_str in zip(predicted_structured, ground_truth):
        # Exact match
        if pred_str.strip() == gt_str.strip():
            exact_matches += 1
            fuzzy_matches += 1
        else:
            # Fuzzy match
            if fuzzy_match(pred_str, gt_str, threshold=fuzzy_threshold):
                fuzzy_matches += 1
        
        # Parse predictions and ground truth
        pred_dict = parse_structured_location(pred_str)
        gt_dict = parse_structured_location(gt_str)
        
        # Compute per-level metrics
        for level in ['state', 'district', 'village', 'other_locations']:
            pred_val = pred_dict.get(level, '')
            gt_val = gt_dict.get(level, '')
            
            if level == 'other_locations':
                # Handle list comparison
                pred_list = [x.strip() for x in pred_val.split(',') if x.strip()] if pred_val else []
                gt_list = [x.strip() for x in gt_val.split(',') if x.strip()] if gt_val else []
                
                pred_set = set(pred_list)
                gt_set = set(gt_list)
                
                tp = len(pred_set & gt_set)
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)
            else:
                # Single value comparison
                pred_clean = pred_val.strip().lower() if pred_val else ''
                gt_clean = gt_val.strip().lower() if gt_val else ''
                
                if pred_clean and gt_clean:
                    if pred_clean == gt_clean:
                        tp = 1
                        fp = 0
                        fn = 0
                    else:
                        tp = 0
                        fp = 1
                        fn = 1
                elif pred_clean and not gt_clean:
                    tp = 0
                    fp = 1
                    fn = 0
                elif not pred_clean and gt_clean:
                    tp = 0
                    fp = 0
                    fn = 1
                else:
                    tp = 0
                    fp = 0
                    fn = 0
            
            level_metrics[level]['tp'] += tp
            level_metrics[level]['fp'] += fp
            level_metrics[level]['fn'] += fn
    
    # Calculate aggregate metrics
    total_examples = len(predicted_structured)
    exact_match_acc = exact_matches / total_examples if total_examples > 0 else 0
    fuzzy_match_acc = fuzzy_matches / total_examples if total_examples > 0 else 0
    
    # Calculate per-level P/R/F1
    per_level_metrics = {}
    for level, counts in level_metrics.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_level_metrics[level] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    # Micro-averaged F1
    total_tp = sum(m['tp'] for m in level_metrics.values())
    total_fp = sum(m['fp'] for m in level_metrics.values())
    total_fn = sum(m['fn'] for m in level_metrics.values())
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Compile results
    metrics = {
        'exact_match': exact_match_acc,
        'fuzzy_match': fuzzy_match_acc,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'per_level': per_level_metrics,
    }
    
    results = {
        'metrics': metrics,
        'predictions': predicted_structured,
        'ground_truth': ground_truth,
    }
    
    return results


def run_span_ner_model(
    model_key: str,
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Complete pipeline: train, evaluate, and save a span-NER model.
    
    Args:
        model_key: Key from BERT_MODEL_CONFIGS (e.g., 'deberta-v3')
        train_data: Training data with 'text' and 'entities'
        val_data: Validation data
        test_data: Test data with ground truth location metadata
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to run on
        
    Returns:
        Dict with model, tokenizer, metrics, and predictions
    """
    if model_key not in BERT_MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Choose from {list(BERT_MODEL_CONFIGS.keys())}")
    
    model_name = BERT_MODEL_CONFIGS[model_key]['model_name']
    
    print(f"\n{'='*60}")
    print(f"Training {model_key}: {BERT_MODEL_CONFIGS[model_key]['description']}")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    # Train model
    print("Training...")
    model, tokenizer, training_metrics = train_span_ner_model(
        model_name=model_name,
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
    )
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_ner_model(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    
    # Print metrics
    print("\nTest Metrics:")
    print(f"Exact Match: {test_results['metrics']['exact_match']:.4f}")
    print(f"Fuzzy Match: {test_results['metrics']['fuzzy_match']:.4f}")
    print(f"Micro F1: {test_results['metrics']['micro_f1']:.4f}")
    print("\nPer-Level F1:")
    for level, metrics in test_results['metrics']['per_level'].items():
        print(f"  {level}: {metrics['f1']:.4f}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'training_metrics': training_metrics,
        'test_metrics': test_results['metrics'],
        'predictions': test_results['predictions'],
        'ground_truth': test_results['ground_truth'],
    }


def save_bert_predictions_and_metrics(
    model_key: str,
    results: Dict[str, Any],
    test_data: List[Dict],
    task_name: str,
    save_dataframe_csv_func,
) -> None:
    """
    Save BERT model predictions and metrics to CSV files.
    
    This function provides a consistent interface for saving predictions,
    similar to run_and_save_llm_location_results for LLM models.
    
    Args:
        model_key: Model identifier (e.g., 'confliBERT', 'deberta-v3')
        results: Results dict from run_span_ner_model containing:
            - predictions: List of predicted location strings
            - ground_truth: List of ground truth location strings
            - test_metrics: Dict of evaluation metrics
        test_data: Test dataset with metadata (incident_number, text, etc.)
        task_name: Task name for organizing results
        save_dataframe_csv_func: Function to save dataframes (from file_io)
    """
    # Save predictions with incident_number and incident_summary
    predictions_df = pd.DataFrame({
        'incident_number': [ex['metadata']['incident_number'] for ex in test_data],
        'incident_summary': [ex['text'] for ex in test_data],
        'ground_truth': results['ground_truth'],
        'prediction': results['predictions'],
    })
    save_dataframe_csv_func(predictions_df, f"{model_key}_predictions.csv", task_name)
    
    # Save metrics summary
    metrics_df = pd.DataFrame([{
        'model': model_key,
        'exact_match': results['test_metrics']['exact_match'],
        'fuzzy_match': results['test_metrics']['fuzzy_match'],
        'micro_f1': results['test_metrics']['micro_f1'],
        'state_f1': results['test_metrics']['per_level']['state']['f1'],
        'district_f1': results['test_metrics']['per_level']['district']['f1'],
        'village_f1': results['test_metrics']['per_level']['village']['f1'],
        'other_locations_f1': results['test_metrics']['per_level']['other_locations']['f1'],
    }])
    save_dataframe_csv_func(metrics_df, f"{model_key}_metrics.csv", task_name)
    
    print(f"✅ {model_key} predictions and metrics saved")

