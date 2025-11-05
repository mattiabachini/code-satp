"""Utilities for computing and displaying evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(predictions, labels, extraction_success=None):
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: array of predicted counts
        labels: array of true counts
        extraction_success: optional boolean array indicating successful number extraction
    
    Returns:
        dict with 'overall' key containing overall metrics
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Basic metrics
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    exact_match = np.mean(predictions == labels)
    within_1 = np.mean(np.abs(predictions - labels) <= 1)
    within_2 = np.mean(np.abs(predictions - labels) <= 2)
    
    # Non-zero MAE
    nonzero_mask = labels > 0
    nonzero_mae = mean_absolute_error(labels[nonzero_mask], predictions[nonzero_mask]) if nonzero_mask.any() else 0
    
    # Median absolute error (robust to outliers)
    mdae = np.median(np.abs(predictions - labels))
    
    # Extraction success rate
    if extraction_success is not None:
        extraction_rate = np.mean(extraction_success)
    else:
        extraction_rate = 1.0  # Assume all successful if not provided
    
    overall_metrics = {
        'mae': mae,
        'rmse': rmse,
        'mdae': mdae,
        'exact_match': exact_match,
        'within_1': within_1,
        'within_2': within_2,
        'nonzero_mae': nonzero_mae,
        'extraction_rate': extraction_rate
    }
    
    # Compute per-bin metrics
    bin_metrics = compute_bin_metrics(predictions, labels)
    
    return {
        'overall': overall_metrics,
        'bins': bin_metrics
    }


def compute_bin_metrics(predictions, labels, bins=None, bin_labels=None):
    """
    Compute metrics for each bin of death counts.
    
    Args:
        predictions: array of predicted counts
        labels: array of true counts
        bins: bin edges (default: [0, 1, 2, 3, 6, np.inf])
        bin_labels: bin labels (default: ['0', '1', '2', '3-5', '6+'])
    
    Returns:
        dict with bin labels as keys, each containing metrics dict
    """
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Default bins matching notebook analysis
    if bins is None:
        bins = [0, 1, 2, 3, 6, np.inf]
    if bin_labels is None:
        bin_labels = ['0', '1', '2', '3-5', '6+']
    
    # Bin the labels
    label_bins = pd.cut(labels, bins=bins, labels=bin_labels, right=False)
    
    bin_results = {}
    
    for bin_label in bin_labels:
        mask = label_bins == bin_label
        n = mask.sum()
        
        if n == 0:
            bin_results[bin_label] = {
                'mae': 0.0,
                'rmse': 0.0,
                'exact_match': 0.0,
                'within_1': 0.0,
                'within_2': 0.0,
                'n': 0
            }
            continue
        
        bin_preds = predictions[mask]
        bin_labels_actual = labels[mask]
        
        bin_mae = mean_absolute_error(bin_labels_actual, bin_preds)
        bin_rmse = np.sqrt(mean_squared_error(bin_labels_actual, bin_preds))
        bin_exact = np.mean(bin_preds == bin_labels_actual)
        bin_within_1 = np.mean(np.abs(bin_preds - bin_labels_actual) <= 1)
        bin_within_2 = np.mean(np.abs(bin_preds - bin_labels_actual) <= 2)
        
        bin_results[bin_label] = {
            'mae': bin_mae,
            'rmse': bin_rmse,
            'exact_match': bin_exact,
            'within_1': bin_within_1,
            'within_2': bin_within_2,
            'n': int(n)
        }
    
    return bin_results


def print_metrics(metrics, model_name="Model"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary from compute_metrics() with 'overall' and 'bins' keys
        model_name: Name of the model for display
    """
    overall = metrics.get('overall', metrics)  # Support both old and new format
    
    print(f"\n{model_name} Performance:")
    print(f"  MAE: {overall['mae']:.3f}")
    print(f"  RMSE: {overall['rmse']:.3f}")
    print(f"  MdAE: {overall['mdae']:.3f}")
    print(f"  Exact Match: {overall['exact_match']:.3f}")
    print(f"  Within-1: {overall['within_1']:.3f}")
    print(f"  Within-2: {overall['within_2']:.3f}")
    print(f"  Non-zero MAE: {overall['nonzero_mae']:.3f}")
    print(f"  Extraction Rate: {overall['extraction_rate']:.3f}")
    
    # Print per-bin metrics if available
    if 'bins' in metrics:
        print(f"\n  Per-Bin Metrics:")
        for bin_label in ['0', '1', '2', '3-5', '6+']:
            if bin_label in metrics['bins']:
                bin_metrics = metrics['bins'][bin_label]
                print(f"    {bin_label}: MAE={bin_metrics['mae']:.3f}, "
                      f"RMSE={bin_metrics['rmse']:.3f}, "
                      f"Exact={bin_metrics['exact_match']:.3f}, "
                      f"W1={bin_metrics['within_1']:.3f}, "
                      f"W2={bin_metrics['within_2']:.3f}, "
                      f"N={bin_metrics['n']}")

