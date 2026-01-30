"""Utilities for running LLM classification inference via OpenAI API."""

import os
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------------
# Prompt templates (from prompt-drafts.md)
# ---------------------------------------------------------------------------

PERPETRATOR_SYSTEM = (
    "You are classifying conflict events from India's Maoist insurgency. "
    "Given an event description, identify the perpetrator/action-taker.\n\n"
    "Categories:\n"
    "- Maoist: Maoist insurgents, Naxalites, or affiliated groups\n"
    "- Security: Government security forces, police, CRPF, military\n"
    "- Unknown: Cannot be determined from the text"
)

ACTION_TYPE_SYSTEM = (
    "You are classifying conflict events. An event may involve multiple action types.\n\n"
    "Categories:\n"
    "- Armed Assault: Armed attacks, shootings, ambushes, firefights\n"
    "- Arrest: Detention, capture, apprehension by authorities\n"
    "- Bombing: Explosions, IEDs, landmines, blasts\n"
    "- Infrastructure: Attacks on facilities, sabotage, arson of buildings\n"
    "- Surrender: Voluntary surrender, laying down arms\n"
    "- Seizure: Raids, confiscation of weapons/materials\n"
    "- Abduction: Kidnapping, hostage-taking"
)

TARGET_TYPE_SYSTEM = (
    "You are classifying the targets of conflict events. An event may have multiple targets.\n\n"
    "Categories:\n"
    "- Maoist: Maoist insurgents or their camps/facilities\n"
    "- Security: Police, military, paramilitary forces\n"
    "- Civilians: Non-combatant individuals\n"
    "- Government Officials: Politicians, bureaucrats, elected officials\n"
    "- Government Infrastructure: Government buildings, roads, bridges, railways\n"
    "- Private Property: Civilian homes, businesses, vehicles\n"
    "- Mining Company: Mining operations, personnel, or equipment\n"
    "- Non-Maoist Armed Group: Other armed groups (not Maoist or government)\n"
    "- No Target: No specific target (e.g., surrenders)"
)

PERPETRATOR_USER_TEMPLATE = (
    "Event: {text}\n\n"
    "Respond with only one word: Maoist, Security, or Unknown"
)

ACTION_TYPE_USER_TEMPLATE = (
    "Event: {text}\n\n"
    'List all applicable categories, separated by commas. If none apply clearly, respond "None".'
)

TARGET_TYPE_USER_TEMPLATE = (
    "Event: {text}\n\n"
    "List all applicable categories, separated by commas."
)


# Label name mappings (API output -> CSV column name)
ACTION_LABELS = [
    "armed_assault", "arrest", "bombing", "infrastructure",
    "surrender", "seizure", "abduction",
]
ACTION_LABEL_MAP = {
    "armed assault": "armed_assault",
    "arrest": "arrest",
    "bombing": "bombing",
    "infrastructure": "infrastructure",
    "surrender": "surrender",
    "seizure": "seizure",
    "abduction": "abduction",
}

TARGET_LABELS = [
    "civilians", "maoist", "government_officials", "security",
    "private_property", "mining_company", "government_infrastructure",
    "non_maoist_armed_group", "no_target",
]
TARGET_LABEL_MAP = {
    "maoist": "maoist",
    "security": "security",
    "civilians": "civilians",
    "government officials": "government_officials",
    "government infrastructure": "government_infrastructure",
    "private property": "private_property",
    "mining company": "mining_company",
    "non-maoist armed group": "non_maoist_armed_group",
    "non maoist armed group": "non_maoist_armed_group",
    "no target": "no_target",
}

PERPETRATOR_LABELS = ["Maoist", "Security", "Unknown"]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_messages(
    task: str,
    text: str,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build OpenAI chat messages for a classification task.

    Args:
        task: One of "perpetrator", "action_type", "target_type".
        text: The incident summary to classify.
        few_shot_examples: Optional list of {"text": ..., "label": ...} dicts.

    Returns:
        List of message dicts for the OpenAI API.
    """
    system_map = {
        "perpetrator": PERPETRATOR_SYSTEM,
        "action_type": ACTION_TYPE_SYSTEM,
        "target_type": TARGET_TYPE_SYSTEM,
    }
    user_template_map = {
        "perpetrator": PERPETRATOR_USER_TEMPLATE,
        "action_type": ACTION_TYPE_USER_TEMPLATE,
        "target_type": TARGET_TYPE_USER_TEMPLATE,
    }

    messages = [{"role": "system", "content": system_map[task]}]

    if few_shot_examples:
        for ex in few_shot_examples:
            messages.append({
                "role": "user",
                "content": user_template_map[task].format(text=ex["text"]),
            })
            messages.append({"role": "assistant", "content": ex["label"]})

    messages.append({
        "role": "user",
        "content": user_template_map[task].format(text=text),
    })
    return messages


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _best_fuzzy_match(candidate: str, valid: List[str], threshold: float = 0.5) -> Optional[str]:
    """Return the best fuzzy match from valid labels, or None if below threshold."""
    candidate = candidate.strip().lower()
    best, best_score = None, 0.0
    for v in valid:
        score = SequenceMatcher(None, candidate, v.lower()).ratio()
        if score > best_score:
            best, best_score = v, score
    return best if best_score >= threshold else None


def parse_singlelabel(output: str, valid_labels: List[str] = PERPETRATOR_LABELS) -> str:
    """Parse single-label output, fuzzy-matching to closest valid label."""
    output = output.strip()
    # Exact match (case-insensitive)
    for v in valid_labels:
        if output.lower() == v.lower():
            return v
    # Fuzzy match
    match = _best_fuzzy_match(output, valid_labels)
    return match if match else valid_labels[-1]  # default to last (Unknown)


def parse_multilabel(output: str, label_map: Dict[str, str]) -> Dict[str, int]:
    """Parse comma-separated multi-label output into a binary dict.

    Args:
        output: Model output string (comma-separated labels).
        label_map: Mapping from lowercase display name to column name.

    Returns:
        Dict mapping column names to 0/1.
    """
    all_cols = list(set(label_map.values()))
    result = {col: 0 for col in all_cols}

    if not output or output.strip().lower() == "none":
        return result

    candidates = [c.strip() for c in output.split(",")]
    for cand in candidates:
        cand_lower = cand.strip().lower()
        # Direct match
        if cand_lower in label_map:
            result[label_map[cand_lower]] = 1
            continue
        # Fuzzy match
        match = _best_fuzzy_match(cand_lower, list(label_map.keys()))
        if match:
            result[label_map[match]] = 1

    return result


# ---------------------------------------------------------------------------
# OpenAI batch runner (adapted from count-models pattern)
# ---------------------------------------------------------------------------

def run_openai_classification_batch(
    texts: List[str],
    task: str,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    max_tokens: int = 30,
    rate_limit_delay: float = 0.05,
    max_concurrency: int = 16,
    max_retries: int = 5,
    show_progress: bool = True,
) -> List[str]:
    """Run OpenAI classification on a list of texts. Returns raw string outputs."""
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    all_messages = [build_messages(task, t, few_shot_examples) for t in texts]
    outs: List[str] = [""] * len(texts)
    total = len(texts)
    errors = 0

    progress_bar = (
        tqdm(total=total, desc=f"Classifying ({task})", leave=False)
        if (show_progress and tqdm and total > 0)
        else None
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Reasoning models (o1, o3, gpt-5.x) don't support temperature or max_tokens
    is_reasoning = any(tag in model_name for tag in ["o1", "o3", "gpt-5"])
    if is_reasoning and "chat" in model_name:
        is_reasoning = False  # gpt-5.2-chat-latest is non-reasoning

    def _one(idx: int, msgs: List[Dict]) -> Tuple[int, str, bool]:
        for attempt in range(max_retries):
            try:
                api_kwargs = {
                    "model": model_name,
                    "messages": msgs,
                }
                if is_reasoning:
                    api_kwargs["max_completion_tokens"] = max_tokens
                else:
                    api_kwargs["max_tokens"] = max_tokens
                    api_kwargs["temperature"] = 0.0
                response = client.chat.completions.create(**api_kwargs)
                out = response.choices[0].message.content.strip()
                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay)
                return idx, out, True
            except Exception:
                backoff = rate_limit_delay * (2 ** attempt)
                time.sleep(backoff)
        return idx, "", False

    if total > 0:
        with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as executor:
            futures = [executor.submit(_one, i, m) for i, m in enumerate(all_messages)]
            for fut in as_completed(futures):
                idx, out, ok = fut.result()
                outs[idx] = out
                if not ok:
                    errors += 1
                if progress_bar is not None:
                    progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()
    if errors > 0:
        print(f"Warning: {errors} errors occurred after {max_retries} retries each")

    return outs


# ---------------------------------------------------------------------------
# Timing wrapper (reused from count-models)
# ---------------------------------------------------------------------------

def time_inference_call(
    inference_func: Callable, *args, **kwargs
) -> Tuple[Any, Dict[str, float]]:
    """Time an inference function call and return (result, timing_dict)."""
    start = time.time()
    result = inference_func(*args, **kwargs)
    elapsed = time.time() - start
    n = len(result) if isinstance(result, list) else 1
    return result, {
        "total_time_seconds": elapsed,
        "time_per_item_seconds": elapsed / n if n else 0,
        "throughput_items_per_second": n / elapsed if elapsed > 0 else 0,
        "num_items": n,
    }


# ---------------------------------------------------------------------------
# Metrics bridge: convert LLM outputs to encoder-compatible format
# ---------------------------------------------------------------------------

def singlelabel_predictions_to_df(
    texts: List[str],
    raw_outputs: List[str],
    true_labels: List[str],
    valid_labels: List[str] = PERPETRATOR_LABELS,
):
    """Convert single-label predictions to a DataFrame matching encoder output format."""
    import pandas as pd
    from sklearn.metrics import classification_report, f1_score

    preds = [parse_singlelabel(o, valid_labels) for o in raw_outputs]
    df = pd.DataFrame({
        "incident_summary": texts,
        "true_label": true_labels,
        "pred_label": preds,
        "raw_output": raw_outputs,
    })

    report = classification_report(
        true_labels, preds, labels=valid_labels,
        output_dict=True, zero_division=0,
    )
    micro_f1 = f1_score(true_labels, preds, labels=valid_labels, average="micro", zero_division=0)
    macro_f1 = f1_score(true_labels, preds, labels=valid_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, preds, labels=valid_labels, average="weighted", zero_division=0)

    metrics = {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_label": {lbl: report.get(lbl, {}) for lbl in valid_labels},
    }
    return df, metrics


def multilabel_predictions_to_df(
    texts: List[str],
    raw_outputs: List[str],
    true_labels_df,  # DataFrame with binary columns
    label_map: Dict[str, str],
    label_cols: List[str],
):
    """Convert multi-label predictions to a DataFrame matching encoder output format."""
    import pandas as pd
    import numpy as np
    from sklearn.metrics import classification_report, f1_score

    parsed = [parse_multilabel(o, label_map) for o in raw_outputs]
    pred_df = pd.DataFrame(parsed)[label_cols]

    result_df = pd.DataFrame({"incident_summary": texts, "raw_output": raw_outputs})
    for col in label_cols:
        result_df[f"true_{col}"] = true_labels_df[col].values
        result_df[f"pred_{col}"] = pred_df[col].values

    y_true = true_labels_df[label_cols].values
    y_pred = pred_df[label_cols].values

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_true, y_pred, target_names=label_cols,
        output_dict=True, zero_division=0,
    )

    metrics = {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_label": {col: report.get(col, {}) for col in label_cols},
    }
    return result_df, metrics


def save_run_results(
    pred_df,
    metrics: Dict,
    timing: Dict,
    output_dir: Path,
    filename_prefix: str,
):
    """Save prediction CSV and metrics JSON for one run."""
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{filename_prefix}.csv"
    pred_df.to_csv(csv_path, index=False)

    metrics_path = output_dir / f"{filename_prefix}_metrics.json"
    combined = {**metrics, "timing": timing}
    with open(metrics_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"Saved: {csv_path}")
    print(f"Saved: {metrics_path}")
    return csv_path, metrics_path
