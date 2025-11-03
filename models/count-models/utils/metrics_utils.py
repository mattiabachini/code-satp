"""Utilities for computing and displaying evaluation metrics."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(predictions, labels, extraction_success=None):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: array of predicted counts
        labels: array of true counts
        extraction_success: optional boolean array indicating successful number extraction
    
    Returns:
        dict of metrics
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Basic metrics
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    exact_match = np.mean(predictions == labels)
    within_1 = np.mean(np.abs(predictions - labels) <= 1)
    within_2 = np.mean(np.abs(predictions - labels) <= 2)
    
    # Zero-class metrics
    zero_mask = labels == 0
    nonzero_mask = labels > 0
    
    zero_accuracy = np.mean((predictions == 0) == zero_mask) if zero_mask.any() else 0
    zero_precision = np.mean(labels[predictions == 0] == 0) if (predictions == 0).any() else 0
    zero_recall = np.mean(predictions[zero_mask] == 0) if zero_mask.any() else 0
    zero_f1 = 2 * (zero_precision * zero_recall) / (zero_precision + zero_recall) if (zero_precision + zero_recall) > 0 else 0
    
    # Non-zero MAE
    nonzero_mae = mean_absolute_error(labels[nonzero_mask], predictions[nonzero_mask]) if nonzero_mask.any() else 0
    
    # Median absolute error (robust to outliers)
    mdae = np.median(np.abs(predictions - labels))
    
    # Extraction success rate
    if extraction_success is not None:
        extraction_rate = np.mean(extraction_success)
    else:
        extraction_rate = 1.0  # Assume all successful if not provided
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mdae': mdae,
        'exact_match': exact_match,
        'within_1': within_1,
        'within_2': within_2,
        'zero_accuracy': zero_accuracy,
        'zero_precision': zero_precision,
        'zero_recall': zero_recall,
        'zero_f1': zero_f1,
        'nonzero_mae': nonzero_mae,
        'extraction_rate': extraction_rate
    }
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics from compute_metrics()
        model_name: Name of the model for display
    """
    print(f"\n{model_name} Performance:")
    print(f"  MAE: {metrics['mae']:.3f}")
    print(f"  RMSE: {metrics['rmse']:.3f}")
    print(f"  MdAE: {metrics['mdae']:.3f}")
    print(f"  Exact Match: {metrics['exact_match']:.3f}")
    print(f"  Within-1: {metrics['within_1']:.3f}")
    print(f"  Within-2: {metrics['within_2']:.3f}")
    print(f"  Zero F1: {metrics['zero_f1']:.3f}")
    print(f"  Non-zero MAE: {metrics['nonzero_mae']:.3f}")
    print(f"  Extraction Rate: {metrics['extraction_rate']:.3f}")

