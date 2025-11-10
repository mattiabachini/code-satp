"""Custom model architectures and training utilities for extraction tasks."""

import torch
import numpy as np
import json
import pandas as pd
from torch.nn import Linear, PoissonNLLLoss
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, EarlyStoppingCallback
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset


class PoissonRegressionModel(torch.nn.Module):
    """
    DistilBERT-based Poisson regression model for count prediction.
    
    Uses Poisson NLL loss which is appropriate for modeling count data.
    """
    
    def __init__(self, pretrained_model_name, num_labels=1):
        """
        Initialize the model.
        
        Args:
            pretrained_model_name: HuggingFace model identifier (e.g., 'distilbert-base-cased')
            num_labels: Number of output labels (default: 1 for single count prediction)
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.regressor = Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for padded sequences
            labels: True count values (for computing loss during training)
            
        Returns:
            SequenceClassifierOutput with loss and logits
        """
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0][:, 0, :]  # [CLS] token
        # Predict log-rate (log of Poisson mean) for numerical stability
        log_rate = self.regressor(sequence_output).squeeze(-1)
        # Convert to mean count (mu) for outputs
        mu = torch.exp(log_rate)
        
        loss = None
        if labels is not None:
            # Use Poisson loss with log_input=True (inputs are log of mean)
            loss_fct = PoissonNLLLoss(log_input=True)
            loss = loss_fct(log_rate, labels.float())
            
        return SequenceClassifierOutput(
            loss=loss,
            logits=mu,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None
        )


def extract_qa_answer(start_logits, end_logits, input_ids, tokenizer, n_best=1):
    """
    Extract answer text from QA model predictions.
    
    Given start and end logits from a QA model, this function finds the most likely
    answer span and decodes it to text.
    
    Args:
        start_logits: Start position logits from QA model (shape: [batch_size, seq_len])
        end_logits: End position logits from QA model (shape: [batch_size, seq_len])
        input_ids: Input token IDs (shape: [batch_size, seq_len])
        tokenizer: HuggingFace tokenizer used for encoding
        n_best: Number of best answers to consider
        
    Returns:
        List of answer texts (one per example in batch)
    """
    # Convert to numpy for easier manipulation
    start_logits = start_logits.cpu().numpy() if torch.is_tensor(start_logits) else start_logits
    end_logits = end_logits.cpu().numpy() if torch.is_tensor(end_logits) else end_logits
    input_ids = input_ids.cpu().numpy() if torch.is_tensor(input_ids) else input_ids
    
    answers = []
    batch_size = start_logits.shape[0]
    
    for i in range(batch_size):
        # Get top n_best start and end positions
        start_indexes = np.argsort(start_logits[i])[-n_best:][::-1]
        end_indexes = np.argsort(end_logits[i])[-n_best:][::-1]
        
        # Find best valid span (end >= start)
        best_score = float('-inf')
        best_start = 0
        best_end = 0
        
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Check if span is valid
                if end_index >= start_index and (end_index - start_index) <= 30:  # Max answer length
                    score = start_logits[i][start_index] + end_logits[i][end_index]
                    if score > best_score:
                        best_score = score
                        best_start = start_index
                        best_end = end_index
        
        # Handle impossible answers (best_start == 0 might indicate impossible, but we check validity)
        # If the span is invalid or very short and at position 0, it might be impossible
        if best_start == 0 and best_end == 0 and best_score < 0:
            # Likely an impossible answer - return empty string
            answers.append("")
        else:
            # Extract answer tokens
            answer_tokens = input_ids[i][best_start:best_end + 1]
            answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
            answers.append(answer_text)
    
    return answers


def run_seq2seq_location_model(
    model_id: str,
    name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    test_df: pd.DataFrame,
    task_name: str,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    num_epochs: int = 10,
    generation_max_length: int = 128,
    seed: int = 42,
    generation_num_beams: int = None,
) -> dict:
    """
    Train and evaluate a seq2seq model for location extraction.
    
    This function handles model loading, training, evaluation, and saving results.
    Expects pre-tokenized HuggingFace datasets.
    
    Args:
        model_id: HuggingFace model identifier (e.g., 'google/flan-t5-base')
        name: Short name for saving files (e.g., 'flan-t5-base')
        train_dataset: Pre-tokenized training dataset (HuggingFace Dataset)
        val_dataset: Pre-tokenized validation dataset (HuggingFace Dataset)
        test_dataset: Pre-tokenized test dataset (HuggingFace Dataset)
        test_df: Original test DataFrame for saving predictions with metadata
        task_name: Task name for organizing results (e.g., 'location-extraction')
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Maximum training epochs
        generation_max_length: Max tokens to generate
        seed: Random seed
        generation_num_beams: Number of beams for beam search (default: None, uses greedy decoding)
        
    Returns:
        Dictionary of metrics
    """
    # Import utilities here to avoid circular imports
    from .training_utils import create_seq2seq_training_args, cleanup_model
    from .metrics_utils import compute_metrics, print_metrics
    from .file_io import save_dataframe_csv, get_task_results_dir
    
    print("\n" + "="*80)
    print(f"{name.upper()}: Training location extraction model")
    print("="*80)
    
    # Load tokenizer and model
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    print(f"Model parameters: {model.num_parameters():,}")
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Setup data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Training arguments
    training_args = create_seq2seq_training_args(
        output_dir=f"results/model_checkpoints/{name}",
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        seed=seed,
        generation_max_length=generation_max_length,
        generation_num_beams=generation_num_beams
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("Trainer configured")
    print(f"  Output dir: {training_args.output_dir}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    
    # Train the model
    print(f"Training {name}...")
    trainer.train()
    print("Training complete!")
    
    # Generate predictions on test set
    print("Generating predictions on test set...")
    predictions = trainer.predict(test_dataset)
    predicted_ids = predictions.predictions
    label_ids = predictions.label_ids

    # Handle tuple outputs (e.g., when past key values are returned)
    if isinstance(predicted_ids, tuple):
        predicted_ids = predicted_ids[0]
    if isinstance(label_ids, tuple):
        label_ids = label_ids[0]

    # Ensure predictions/labels are integer arrays before decoding
    predicted_ids = np.asarray(predicted_ids)
    label_ids = np.asarray(label_ids)

    if np.issubdtype(predicted_ids.dtype, np.floating):
        predicted_ids = np.rint(predicted_ids).astype(np.int32)
    else:
        predicted_ids = predicted_ids.astype(np.int32)

    if np.issubdtype(label_ids.dtype, np.floating):
        label_ids = np.rint(label_ids).astype(np.int32)
    else:
        label_ids = label_ids.astype(np.int32)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    decoded_labels = [
        tokenizer.decode([l for l in label if l != -100], skip_special_tokens=True)
        for label in label_ids
    ]
    
    # Compute metrics using location-specific metrics
    metrics = compute_metrics(predicted_ids, label_ids, tokenizer)
    print_metrics(metrics, name)
    
    # Save metrics to JSON
    results_dir = get_task_results_dir(task_name)
    metrics_path = results_dir / f"location_{name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to {metrics_path}")
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'incident_number': test_df['incident_number'].values,
        'incident_summary': test_df['incident_summary'].values,
        'true_label': decoded_labels,
        'prediction': decoded_preds,
    })
    save_dataframe_csv(predictions_df, f'location_{name}_predictions.csv', task_name=task_name)
    print(f"✅ Saved predictions to {results_dir}/location_{name}_predictions.csv")
    
    # Save the model and tokenizer
    try:
        model_dir = results_dir / f"{name}_finetuned_model"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"✅ Model and tokenizer saved to: {model_dir}")
    except Exception as e:
        print(f"⚠️ Could not save model: {e}")
    
    # Clear GPU memory
    cleanup_model(model, trainer, tokenizer)
    print(f"\n{name}: GPU memory cleared")
    
    return metrics


def run_flan_t5_xl_lora_location_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    test_df: pd.DataFrame,
    task_name: str,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    generation_max_length: int = 128,
    seed: int = 42,
    generation_num_beams: int = None,
) -> dict:
    """
    Train and evaluate Flan-T5-XL with QLoRA for location extraction.
    
    This function handles 4-bit quantization and LoRA fine-tuning for the large Flan-T5-XL model.
    Requires PEFT, bitsandbytes, and accelerate packages.
    Expects pre-tokenized HuggingFace datasets.
    
    Args:
        train_dataset: Pre-tokenized training dataset (HuggingFace Dataset)
        val_dataset: Pre-tokenized validation dataset (HuggingFace Dataset)
        test_dataset: Pre-tokenized test dataset (HuggingFace Dataset)
        test_df: Original test DataFrame for saving predictions with metadata
        task_name: Task name for organizing results (e.g., 'location-extraction')
        batch_size: Training batch size (default 2 for memory efficiency)
        learning_rate: Learning rate (default 5e-5)
        num_epochs: Maximum training epochs (default 3)
        generation_max_length: Max tokens to generate
        seed: Random seed
        generation_num_beams: Number of beams for beam search (default: None, uses greedy decoding)
        
    Returns:
        Dictionary of metrics
    """
    # Import utilities here to avoid circular imports
    from .training_utils import cleanup_model
    from .metrics_utils import compute_metrics, print_metrics
    from .file_io import save_dataframe_csv, get_task_results_dir
    
    # Import PEFT/LoRA dependencies
    try:
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Please install required packages:")
        print("  pip install peft bitsandbytes accelerate")
        return {}
    
    print("\n" + "="*80)
    print("FLAN-T5-XL-LORA: Training location extraction with QLoRA")
    print("="*80)
    
    # Load tokenizer and model with 4-bit quantization
    model_id = "google/flan-t5-xl"
    print(f"Loading {model_id} with 4-bit quantization...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        device_map="auto",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Prepare model for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.enable_input_require_grads()
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print("Model prepared with LoRA adapters")
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Setup data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Training arguments
    from .training_utils import create_seq2seq_training_args
    training_args = create_seq2seq_training_args(
        output_dir=f"results/model_checkpoints/flan-t5-xl-lora",
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        seed=seed,
        generation_max_length=generation_max_length,
        generation_num_beams=generation_num_beams
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("Trainer configured")
    print(f"  Output dir: {training_args.output_dir}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    
    # Train the model
    print("Training Flan-T5-XL (QLoRA)...")
    trainer.train()
    print("Training complete!")
    
    # Generate predictions on test set
    print("Generating predictions on test set...")
    predictions = trainer.predict(test_dataset)
    predicted_ids = predictions.predictions
    label_ids = predictions.label_ids
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    decoded_labels = [
        tokenizer.decode([l for l in label if l != -100], skip_special_tokens=True)
        for label in label_ids
    ]
    
    # Compute metrics using location-specific metrics
    metrics = compute_metrics(predicted_ids, label_ids, tokenizer)
    print_metrics(metrics, 'flan-t5-xl-lora')
    
    # Save metrics to JSON
    results_dir = get_task_results_dir(task_name)
    metrics_path = results_dir / "location_flan-t5-xl-lora_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to {metrics_path}")
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'incident_number': test_df['incident_number'].values,
        'incident_summary': test_df['incident_summary'].values,
        'true_label': decoded_labels,
        'prediction': decoded_preds,
    })
    save_dataframe_csv(predictions_df, 'location_flan-t5-xl-lora_predictions.csv', task_name=task_name)
    print(f"✅ Saved predictions to {results_dir}/location_flan-t5-xl-lora_predictions.csv")
    
    # Save LoRA adapter
    try:
        adapter_dir = results_dir / "flan-t5-xl-lora_finetuned_model"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"✅ LoRA adapter and tokenizer saved to: {adapter_dir}")
    except Exception as e:
        print(f"⚠️ Could not save LoRA adapter: {e}")
    
    # Clear GPU memory
    cleanup_model(model, trainer, tokenizer)
    print("\nFlan-T5-XL (QLoRA): GPU memory cleared")
    
    return metrics

