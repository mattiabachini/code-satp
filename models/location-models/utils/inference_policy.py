"""Inference policy for selecting state using NER spans, classifier logits, and P(state|district)."""

from typing import Dict, List, Optional
import math
import numpy as np

from .state_utils import canonicalize_state_name, get_state_alias_map


def select_state_with_fusion(
    slots: Dict[str, Optional[str]],
    classifier_logits: Optional[np.ndarray],
    id2state: Optional[Dict[int, str]] = None,
    p_state_given_district: Optional[Dict[str, Dict[str, float]]] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    tau: float = 0.5,
) -> Optional[str]:
    """
    Policy:
      1) If a validated state span exists in slots, return it.
      2) Else, fuse calibrated classifier distribution with P(state|district) learned from data:
            P*(s|x) ∝ P_cls(s|x)^alpha × Π_{d ∈ districts} P(s|d)^beta
         Pick argmax if its probability ≥ tau; otherwise None.
    """
    # 1) Trust NER state if present
    state_text = slots.get('STATE')
    if state_text and isinstance(state_text, str) and state_text.strip():
        return state_text.strip()
    
    # 2) Fallback to fusion
    if classifier_logits is None or id2state is None:
        return None
    # Softmax
    logits = classifier_logits.astype(np.float64)
    logits = logits - logits.max()
    p_cls = np.exp(logits) / np.maximum(np.exp(logits).sum(), 1e-12)
    
    # Build prior from districts in slots (if any)
    district = slots.get('DISTRICT')
    p_prior = None
    if p_state_given_district is not None and district:
        dist_key = str(district).strip()
        if dist_key in p_state_given_district:
            # align prior to id2state ordering
            prior_list: List[float] = []
            psd = p_state_given_district[dist_key]
            for idx in range(len(id2state)):
                s_name = id2state[idx]
                prior_list.append(psd.get(s_name, 1e-12))
            p_prior = np.array(prior_list, dtype=np.float64)
            p_prior = p_prior / max(p_prior.sum(), 1e-12)
    
    # Combine
    if p_prior is not None:
        # exponentiate mixing
        p_comb = (np.power(p_cls, alpha) * np.power(p_prior, beta))
        z = max(p_comb.sum(), 1e-12)
        p_star = p_comb / z
    else:
        p_star = p_cls
    
    top_idx = int(np.argmax(p_star))
    top_p = float(p_star[top_idx])
    if top_p >= tau:
        return id2state[top_idx]
    return None


