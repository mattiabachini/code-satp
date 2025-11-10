"""Evaluation helpers for location extraction models."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics_utils import (
    compute_metrics,
    decode_location_labels,
    decode_location_predictions,
)


def evaluate_location_model(
    model,
    dataloader: DataLoader,
    tokenizer,
    device: Optional[torch.device] = None,
    generation_max_length: int = 64,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    fuzzy_threshold: int = 85,
    decode_outputs: bool = True,
) -> Dict[str, Any]:
    """
    Generate predictions for a seq2seq location extraction model and compute metrics.

    Args:
        model: HuggingFace seq2seq model.
        dataloader: PyTorch DataLoader yielding batches with ``input_ids``, ``attention_mask``,
            and ``labels`` tensors.
        tokenizer: HuggingFace tokenizer aligned with the model.
        device: Torch device to run inference on. If ``None`` the model's device is used.
        generation_max_length: Maximum length for generated sequences.
        generation_kwargs: Optional additional kwargs forwarded to ``model.generate``.
        fuzzy_threshold: Similarity threshold for fuzzy matching metrics.
        decode_outputs: If ``True``, include decoded text strings in the returned dictionary.

    Returns:
        Dictionary with:
            - ``metrics``: Output from :func:`compute_metrics`.
            - ``predicted_ids``: Numpy array of generated token IDs.
            - ``label_ids``: Numpy array of label token IDs.
            - ``predicted_text`` / ``label_text``: Decoded strings (if ``decode_outputs``).
    """
    if device is None:
        device = next(model.parameters()).device  # type: ignore[arg-type]
    generation_kwargs = generation_kwargs or {}

    model.eval()

    predicted_batches = []
    label_batches = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=generation_max_length,
                **generation_kwargs,
            )
            predicted_batches.append(outputs.cpu().numpy())

            labels_tensor = batch["labels"]
            if torch.is_tensor(labels_tensor):
                label_batches.append(labels_tensor.cpu().numpy())
            else:
                label_batches.append(np.array(labels_tensor))

    if predicted_batches:
        predicted_ids = np.concatenate(predicted_batches, axis=0)
    else:
        predicted_ids = np.empty((0, 0), dtype=np.int64)

    if label_batches:
        label_ids = np.concatenate(label_batches, axis=0)
    else:
        label_ids = np.empty((0, 0), dtype=np.int64)

    metrics = compute_metrics(
        predicted_ids,
        label_ids,
        tokenizer,
        fuzzy_threshold=fuzzy_threshold,
    )

    result: Dict[str, Any] = {
        "metrics": metrics,
        "predicted_ids": predicted_ids,
        "label_ids": label_ids,
    }

    if decode_outputs:
        result["predicted_text"] = decode_location_predictions(predicted_ids, tokenizer)
        result["label_text"] = decode_location_labels(label_ids, tokenizer)

    return result

