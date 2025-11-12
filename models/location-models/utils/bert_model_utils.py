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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import inspect
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
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


# =========================
# Multi-task model (NER + State classifier)
# =========================

class MultiTaskLocationModel(nn.Module):
    """
    Shared-backbone model with:
      - Token classification head for span NER (BIO tags)
      - Sequence classifier head for state (softmax over K states)
    Computes combined loss when 'labels' (token labels) and/or 'state_labels' are provided.
    """
    def __init__(
        self,
        base_model_name: str,
        num_ner_labels: int,
        num_state_labels: int,
        lambda_state: float = 1.0,
        mu_kl: float = 0.0,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = getattr(self.backbone.config, "hidden_size", None) or getattr(self.backbone.config, "hidden_sizes", [])[0]
        if isinstance(hidden_size, list):
            hidden_size = hidden_size[0]
        self.dropout = nn.Dropout(getattr(self.backbone.config, "hidden_dropout_prob", 0.1))
        # Token classification head
        self.token_classifier = nn.Linear(hidden_size, num_ner_labels)
        # State classification head
        self.state_classifier = nn.Linear(hidden_size, num_state_labels)
        self.num_state_labels = num_state_labels
        self.lambda_state = lambda_state
        self.mu_kl = mu_kl
        # Keep mapping to identify STATE tag ids for optional KL usage
        self.id2label = id2label or {}
        # Precompute ids corresponding to STATE BIO tags
        self.state_tag_ids = set()
        for i, tag in self.id2label.items():
            if isinstance(tag, str) and (tag == "B-STATE" or tag == "I-STATE"):
                self.state_tag_ids.add(i)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,           # token labels for NER (with -100 on pads)
        state_labels: Optional[torch.Tensor] = None,     # int64 class ids for state
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Filter out training-specific kwargs that the backbone doesn't expect
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in {"labels", "state_labels", "num_items_in_batch"}}
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **filtered_kwargs)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        pooled_output = sequence_output[:, 0, :]     # use first token ([CLS] or <s>) as pooled representation
        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)

        # Heads
        logits = self.token_classifier(sequence_output)         # (batch, seq_len, num_ner_labels)
        state_logits = self.state_classifier(pooled_output)     # (batch, num_state_labels)

        loss = None
        # Token-level loss
        if labels is not None:
            # Flatten to compute CE over active positions
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss_ner = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss_ner
        else:
            loss_ner = None

        # State classification loss
        loss_state = None
        if state_labels is not None:
            loss_fct_state = nn.CrossEntropyLoss()
            loss_state = loss_fct_state(state_logits, state_labels)
            loss = loss + self.lambda_state * loss_state if loss is not None else self.lambda_state * loss_state

        # Optional KL alignment when STATE spans are present in token labels
        if self.mu_kl and self.mu_kl > 0 and labels is not None and state_labels is not None and len(self.state_tag_ids) > 0:
            with torch.no_grad():
                # Determine for each example whether a STATE tag appears in gold token labels
                batch_size = labels.size(0)
                mask = torch.zeros_like(labels, dtype=torch.bool)
                for tid in self.state_tag_ids:
                    mask |= (labels == tid)
                has_state_span = mask.view(batch_size, -1).any(dim=1)
            if has_state_span.any():
                # Teacher distribution as one-hot of state_labels for those examples
                p_teacher = F.one_hot(state_labels, num_classes=self.num_state_labels).float()
                p_teacher = torch.where(has_state_span.unsqueeze(1), p_teacher, p_teacher.new_zeros(p_teacher.shape))
                # Student distribution from classifier
                log_p_student = F.log_softmax(state_logits, dim=-1)
                # KLDiv expects log-probs for input, probs for target
                kl = F.kl_div(log_p_student, p_teacher, reduction='batchmean')
                loss = loss + self.mu_kl * kl if loss is not None else self.mu_kl * kl

        return {
            "loss": loss,
            "logits": logits,
            "state_logits": state_logits,
        }


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
    state2id: Optional[Dict[str, int]] = None,
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
    # Optional state labels from metadata
    state_labels: Optional[List[int]] = None
    if state2id is not None:
        def _canon(s: Any) -> str:
            return str(s).strip()
        state_labels = [state2id.get(_canon(ex.get('metadata', {}).get('state', '')), -1) for ex in ner_data]
        # Replace unknowns with a valid id if necessary by mapping to a special bucket; here we clip to 0 if any -1
        if any(sl is None or sl < 0 for sl in state_labels):
            # Fallback: map unknowns to a small bucket (e.g., first class) to keep CE defined
            state_labels = [sl if (sl is not None and sl >= 0) else 0 for sl in state_labels]
    
    dataset = Dataset.from_dict({
        'text': texts,
        'entities': entities,
        **({'state_labels': state_labels} if state_labels is not None else {}),
    })
    
    # Tokenize and align labels
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples, tokenizer, label_list, max_length=max_length
        ),
        batched=True,
        remove_columns=['text', 'entities'],
    )
    # Note: state_labels is automatically preserved since it's not in remove_columns
    
    return tokenized_dataset


def _build_state_label_mapping(ner_data: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build a simple mapping {canonical_state_name -> id} from ner_data metadata.
    """
    states: List[str] = []
    for ex in ner_data:
        s = ex.get('metadata', {}).get('state', '')
        if s and str(s).strip():
            states.append(str(s).strip())
    # Stable sorted unique list to keep ids consistent across runs
    uniq_states = sorted(set(states))
    state2id = {s: i for i, s in enumerate(uniq_states)}
    id2state = {i: s for s, i in state2id.items()}
    return state2id, id2state


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
    use_multitask: bool = True,
    lambda_state: float = 1.0,
    mu_kl: float = 0.0,
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
        use_multitask: If True, use MultiTaskLocationModel with state classifier head
        lambda_state: Weight for state classification loss
        mu_kl: Weight for KL alignment term (0 to disable)
        
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
    if use_multitask:
        # Build state label mapping from train+val metadata
        state2id, id2state = _build_state_label_mapping(train_data + val_data)
        num_state_labels = len(state2id)
        model = MultiTaskLocationModel(
            base_model_name=model_name,
            num_ner_labels=num_labels,
            num_state_labels=num_state_labels,
            lambda_state=lambda_state,
            mu_kl=mu_kl,
            id2label=id_to_label,
        )
        # Persist mapping
        try:
            import json, os
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/state_id_mapping.json", "w") as f:
                json.dump({"state2id": state2id, "id2state": id2state}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id_to_label,
            label2id=label_to_id,
        )
    
    # Prepare datasets
    if use_multitask:
        # Rebuild mapping to ensure the same ids in dataset
        state2id, _ = _build_state_label_mapping(train_data + val_data)
        train_dataset = prepare_ner_dataset(
            train_data, tokenizer, label_list, max_length=max_length, state2id=state2id
        )
        val_dataset = prepare_ner_dataset(
            val_data, tokenizer, label_list, max_length=max_length, state2id=state2id
        )
    else:
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
    
    # Trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
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
            # Support both HF ModelOutput and dict
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            # Force 'O' on padding positions to avoid entities over [PAD]
            predictions = predictions.masked_fill(attention_mask == 0, 0)
        
        # Convert to entities
        for j, (pred, text) in enumerate(zip(predictions, batch_texts)):
            pred_ids = pred.cpu().numpy().tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[j])
            
            # Get id to label mapping (from custom model's id2label attribute)
            id_to_label = model.id2label
            
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


def predict_state_logits(
    model,
    tokenizer,
    texts: List[str],
    device: Optional[torch.device] = None,
    batch_size: int = 16,
    max_length: int = 512,
) -> Optional[List[np.ndarray]]:
    """
    Predict per-text state classifier logits if the model provides them (multi-task).
    Returns a list of numpy arrays of shape (num_states,) or None if unavailable.
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    all_logits: List[np.ndarray] = []
    
    # Check capability (models without state head won't have 'state_logits')
    supports_state = True
    try:
        # Quick probe on empty to check attr later
        supports_state = hasattr(model, "state_classifier") or hasattr(model, "module") and hasattr(model.module, "state_classifier")
    except Exception:
        supports_state = False
    
    if not supports_state:
        return None
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors='pt',
        )
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            state_logits = outputs.get("state_logits") if isinstance(outputs, dict) else getattr(outputs, "state_logits", None)
            if state_logits is None:
                return None
            for j in range(state_logits.size(0)):
                all_logits.append(state_logits[j].detach().cpu().numpy())
    return all_logits


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
    
    # Compute comprehensive metrics using shared function
    from .metrics_utils import compute_metrics_from_strings
    
    metrics = compute_metrics_from_strings(
        predicted_structured,
        ground_truth,
        fuzzy_threshold=fuzzy_threshold
    )
    
    # metrics has structure: {'overall': {...}, 'levels': {...}}
    # This matches the seq2seq format and will be saved to JSON as-is
    
    # Add top-level convenience keys for backward compatibility with print statements
    # Extract from nested structure
    overall = metrics.get('overall', {})
    levels = metrics.get('levels', {})
    
    # Add flat keys at top level for easy access in print statements
    metrics['exact_match'] = overall.get('exact_match', 0)
    metrics['fuzzy_match'] = overall.get('fuzzy_match', 0)
    metrics['micro_precision'] = overall.get('micro_exact_precision', 0)
    metrics['micro_recall'] = overall.get('micro_exact_recall', 0)
    metrics['micro_f1'] = overall.get('micro_exact_f1', 0)
    
    # Add per_level dict for backward compatibility (maps to 'levels' structure)
    metrics['per_level'] = {}
    for level in ['state', 'district', 'village', 'other_locations']:
        if level in levels:
            metrics['per_level'][level] = {
                'precision': levels[level].get('exact_precision', 0),
                'recall': levels[level].get('exact_recall', 0),
                'f1': levels[level].get('exact_f1', 0),
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
    early_stopping_patience: int = 3,
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
        max_length: Max sequence length
        early_stopping_patience: Patience for early stopping
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
        early_stopping_patience=early_stopping_patience,
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
    results_dir = None,
) -> None:
    """
    Save BERT model predictions and metrics to CSV and JSON files.
    
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
        results_dir: Optional Path to results directory (will be inferred if not provided)
    """
    import json
    from pathlib import Path
    
    # Save predictions with incident_number and incident_summary
    predictions_df = pd.DataFrame({
        'incident_number': [ex['metadata']['incident_number'] for ex in test_data],
        'incident_summary': [ex['text'] for ex in test_data],
        'ground_truth': results['ground_truth'],
        'prediction': results['predictions'],
    })
    save_dataframe_csv_func(predictions_df, f"{model_key}_predictions.csv", task_name)
    
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
    metrics_path = results_dir / f"{model_key}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(results['test_metrics'], f, indent=2)
    print(f"✅ {model_key} comprehensive metrics saved to JSON: {metrics_path}")
    
    print(f"✅ {model_key} predictions saved")

