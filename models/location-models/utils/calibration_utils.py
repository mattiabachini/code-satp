"""Temperature scaling for calibrating classifier probabilities."""

from typing import Tuple
import numpy as np
from scipy.optimize import minimize


def _nll_with_temperature(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood after scaling logits by 1/T."""
    # Avoid degenerate temperatures
    T = max(T, 1e-3)
    scaled = logits / T
    # log softmax
    m = scaled.max(axis=1, keepdims=True)
    log_probs = scaled - m - np.log(np.exp(scaled - m).sum(axis=1, keepdims=True))
    # NLL
    idx = np.arange(labels.shape[0])
    return float(-log_probs[idx, labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray, T_init: float = 1.0) -> float:
    """
    Fit temperature T on validation set by minimizing NLL.
    Args:
        logits: (N, C) array of uncalibrated logits
        labels: (N,) array of true class indices
        T_init: initial temperature
    Returns:
        scalar temperature T > 0
    """
    res = minimize(lambda t: _nll_with_temperature(float(t[0]), logits, labels),
                   x0=np.array([T_init]), method="L-BFGS-B", bounds=[(1e-3, 1e3)])
    T = float(res.x[0])
    return max(T, 1e-3)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Scale logits by 1/T."""
    T = max(float(T), 1e-3)
    return logits / T


