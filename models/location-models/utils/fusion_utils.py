"""Utilities for applying state fusion after NER model runs."""

from typing import Optional, Dict, Any, List
import json
import numpy as np
import pandas as pd

from .bert_model_utils import predict_state_logits
from .inference_policy import select_state_with_fusion
from .metrics_utils import parse_structured_location
from .llm_location_utils import compute_location_metrics_from_strings
from .calibration_utils import fit_temperature, apply_temperature
from .data_utils import load_p_state_given_district


def apply_state_fusion(
    model_key: str,
    all_results: Dict[str, Any],
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    results_dir,
    device,
    TRAINING_CONFIG: Dict[str, Any],
    FUSION_CONFIG: Dict[str, Any],
    save_dataframe_csv_func=None,
    task_name: Optional[str] = None,
) -> Optional[dict]:
    """
    Apply classifier + P(state|district) fusion to fill/validate state after NER.
    Saves fused predictions/metrics and updates all_results[model_key].
    """
    output_dir = results_dir / model_key
    mapping_path = output_dir / "state_id_mapping.json"
    if not mapping_path.exists():
        print(f"{model_key}: no state_id_mapping.json; skipping fusion.")
        return None

    with open(mapping_path, "r") as fh:
        mapping = json.load(fh)
    id2state = {int(k): v for k, v in mapping["id2state"].items()}
    state2id = {k: int(v) for k, v in mapping["state2id"].items()}

    psgd = load_p_state_given_district(str(results_dir / "p_state_given_district.json"))

    # Calibrate on dev
    T = 1.0
    if FUSION_CONFIG.get("calibrate", True):
        dev_texts = [ex["text"] for ex in val_data]
        dev_logits = predict_state_logits(
            all_results[model_key]["model"],
            all_results[model_key]["tokenizer"],
            dev_texts,
            device=device,
            batch_size=16,
            max_length=TRAINING_CONFIG["max_length"],
        )
        if dev_logits is not None and len(dev_logits) == len(dev_texts):
            y_dev = np.array(
                [state2id.get(str(ex.get("metadata", {}).get("state", "")).strip(), 0) for ex in val_data],
                dtype=int,
            )
            dev_logits_arr = np.stack(dev_logits)
            T = fit_temperature(dev_logits_arr, y_dev)
            with open(output_dir / "state_temperature.txt", "w") as fh:
                fh.write(str(T))
            print(f"{model_key}: fitted temperature T={T:.3f}")
        else:
            print(f"{model_key}: no state logits available for calibration.")

    # Test logits
    test_texts = [ex["text"] for ex in test_data]
    test_logits = predict_state_logits(
        all_results[model_key]["model"],
        all_results[model_key]["tokenizer"],
        test_texts,
        device=device,
        batch_size=16,
        max_length=TRAINING_CONFIG["max_length"],
    )
    if test_logits is None or len(test_logits) != len(test_texts):
        print(f"{model_key}: no state logits; skipping fusion.")
        return None

    test_logits_arr = np.stack(test_logits)
    if T != 1.0:
        test_logits_arr = apply_temperature(test_logits_arr, T)

    # Build fused predictions
    alpha, beta, tau = FUSION_CONFIG["alpha"], FUSION_CONFIG["beta"], FUSION_CONFIG["tau"]
    original_preds = all_results[model_key]["predictions"]
    fused_preds = []
    for pred_str, logit in zip(original_preds, test_logits_arr):
        d = parse_structured_location(pred_str)
        slots = {
            "STATE": d.get("state"),
            "DISTRICT": d.get("district"),
            "VILLAGE": d.get("village"),
            "OTHER_LOCATION": d.get("other_locations"),
        }
        fused_state = select_state_with_fusion(
            slots=slots,
            classifier_logits=logit,
            id2state=id2state,
            p_state_given_district=psgd,
            alpha=alpha,
            beta=beta,
            tau=tau,
        )
        if fused_state and (not d.get("state") or not str(d.get("state")).strip()):
            d["state"] = fused_state
        parts = []
        if d.get("state"): parts.append(f"state: {d['state']}")
        if d.get("district"): parts.append(f"district: {d['district']}")
        if d.get("village"): parts.append(f"village: {d['village']}")
        if d.get("other_locations"): parts.append(f"other_locations: {d['other_locations']}")
        fused_preds.append(", ".join(parts))

    # Metrics (percent)
    gts = all_results[model_key]["ground_truth"]
    fused_metrics = compute_location_metrics_from_strings(fused_preds, gts, fuzzy_threshold=85)
    print(
        f"\n{model_key} (fused) — Exact: {fused_metrics['overall']['exact_match']:.2f}%, "
        f"Fuzzy: {fused_metrics['overall']['fuzzy_match']:.2f}%, "
        f"State exact F1: {fused_metrics['levels']['state']['exact_f1']:.2f}%"
    )

    # Save fused predictions and metrics
    fused_df = pd.DataFrame({
        "incident_number": [ex["metadata"]["incident_number"] for ex in test_data],
        "incident_summary": [ex["text"] for ex in test_data],
        "ground_truth": gts,
        "prediction": fused_preds,
    })
    if save_dataframe_csv_func and task_name:
        save_dataframe_csv_func(fused_df, f"{model_key}_predictions_fused.csv", task_name)
    else:
        fused_df.to_csv(output_dir / f"{model_key}_predictions_fused.csv", index=False)

    fm = fused_metrics
    fused_metrics_row = pd.DataFrame([{
        "model": model_key,
        "exact_match": fm["overall"]["exact_match"],
        "fuzzy_match": fm["overall"]["fuzzy_match"],
        "micro_precision": fm["overall"]["micro_exact_precision"],
        "micro_recall": fm["overall"]["micro_exact_recall"],
        "micro_f1": fm["overall"]["micro_exact_f1"],
        "state_f1": fm["levels"]["state"]["exact_f1"],
        "district_f1": fm["levels"]["district"]["exact_f1"],
        "village_f1": fm["levels"]["village"]["exact_f1"],
        "other_locations_f1": fm["levels"]["other_locations"]["exact_f1"],
    }])
    if save_dataframe_csv_func and task_name:
        save_dataframe_csv_func(fused_metrics_row, f"{model_key}_metrics_fused.csv", task_name)
    else:
        fused_metrics_row.to_csv(output_dir / f"{model_key}_metrics_fused.csv", index=False)

    # Expose fused results back to all_results
    all_results[model_key]["predictions_fused"] = fused_preds
    all_results[model_key]["metrics_fused"] = {
        "exact_match": fm["overall"]["exact_match"],
        "fuzzy_match": fm["overall"]["fuzzy_match"],
        "micro_f1": fm["overall"]["micro_exact_f1"],
        "micro_precision": fm["overall"]["micro_exact_precision"],
        "micro_recall": fm["overall"]["micro_exact_recall"],
        "per_level": {
            "state": {
                "f1": fm["levels"]["state"]["exact_f1"],
                "precision": fm["levels"]["state"]["exact_precision"],
                "recall": fm["levels"]["state"]["exact_recall"],
            },
            "district": {
                "f1": fm["levels"]["district"]["exact_f1"],
                "precision": fm["levels"]["district"]["exact_precision"],
                "recall": fm["levels"]["district"]["exact_recall"],
            },
            "village": {
                "f1": fm["levels"]["village"]["exact_f1"],
                "precision": fm["levels"]["village"]["exact_precision"],
                "recall": fm["levels"]["village"]["exact_recall"],
            },
            "other_locations": {
                "f1": fm["levels"]["other_locations"]["exact_f1"],
                "precision": fm["levels"]["other_locations"]["exact_precision"],
                "recall": fm["levels"]["other_locations"]["exact_recall"],
            },
        },
    }

    return fused_metrics

