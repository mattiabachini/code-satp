"""Utilities for model training configuration and cleanup."""

import gc
import torch
from transformers import Seq2SeqTrainingArguments, TrainingArguments


def create_seq2seq_training_args(
    output_dir,
    batch_size=8,
    learning_rate=3e-5,
    num_epochs=3,
    seed=42,
    generation_max_length=32
):
    """
    Create standard Seq2SeqTrainingArguments for seq2seq models.
    
    Args:
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training and evaluation
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        seed: Random seed for reproducibility
        generation_max_length: Maximum length for generated outputs during evaluation (default: 32, suitable for most structured extraction tasks)
        
    Returns:
        Seq2SeqTrainingArguments object
    """
    # Use bf16 on Ampere+ (e.g., A100) for speed+stability, otherwise fp16 on GPU
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    supports_tf32 = False
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        supports_tf32 = major >= 8

    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        logging_steps=50,
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=supports_tf32,
        optim="adafactor",
        report_to="none",
        seed=seed
    )


def create_regression_training_args(
    output_dir,
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=3,
    seed=42
):
    """
    Create standard TrainingArguments for regression models.
    
    Args:
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training and evaluation
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        seed: Random seed for reproducibility
        
    Returns:
        TrainingArguments object
    """
    supports_tf32 = False
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        supports_tf32 = major >= 8

    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        fp16=False,
        bf16=False,
        tf32=supports_tf32,
        report_to="none",
        seed=seed
    )


def create_qa_training_args(
    output_dir,
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=3,
    seed=42
):
    """
    Create standard TrainingArguments for QA models.
    
    Args:
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training and evaluation
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        seed: Random seed for reproducibility
        
    Returns:
        TrainingArguments object
    """
    # Use bf16 on Ampere+ (e.g., A100) for speed+stability, otherwise fp16 on GPU
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    supports_tf32 = False
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        supports_tf32 = major >= 8

    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=supports_tf32,
        report_to="none",
        seed=seed
    )


def cleanup_model(*objects):
    """
    Clean up model objects and free GPU memory.
    
    Args:
        *objects: Variable number of objects to delete (model, trainer, tokenizer, etc.)
    """
    # Delete all passed objects
    for obj in objects:
        del obj
    
    # Run garbage collection
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

