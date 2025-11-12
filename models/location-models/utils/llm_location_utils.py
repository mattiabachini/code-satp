"""Utilities for location extraction using LLMs (non-tokenized string outputs)."""

import re
import time
from typing import List, Optional, Dict, Any
from .metrics_utils import parse_structured_location, fuzzy_match

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

import torch


# ============================================================================
# Prompt Creation for Location Extraction
# ============================================================================

# Instruction template for LLM location extraction (verbose for zero-shot)
LOCATION_EXTRACTION_INSTRUCTION = (
    "Extract the location hierarchy from this incident. "
    "Return exactly in format: state: <name>, district: <name>, village: <name>, other_locations: <name>. "
    "Use exact format with labels. Omit any missing administrative levels. "
    "Do not repeat the incident text; output only the structured fields. "
    "If no locations are mentioned, return an empty string."
)


def make_location_prompt(text: str) -> str:
    """
    Create a verbose prompt for LLM-based location extraction.
    
    This prompt is optimized for zero-shot inference with instruction-following LLMs.
    It differs from seq2seq prompts (which use shorter formats for fine-tuning)
    because LLMs need explicit, verbose instructions to understand the task.
    
    Args:
        text: The incident summary text
        
    Returns:
        Formatted prompt string ready for LLM inference
        
    Example:
        >>> prompt = make_location_prompt("Maoists attacked in Sukma district...")
        >>> print(prompt)
        Extract the location hierarchy from this incident. Return exactly in format: ...
        
        Incident: Maoists attacked in Sukma district...
        
        Answer:
    """
    return f"{LOCATION_EXTRACTION_INSTRUCTION}\n\nIncident: {text}\n\nAnswer:"


def parse_location_from_llm(text: str) -> dict:
    """
    Parse location from LLM output (handles LLM-specific formatting artifacts).
    
    This function wraps parse_structured_location() and adds cleaning for common
    LLM output artifacts like code fences, markdown formatting, and conversational text.
    
    Args:
        text: Raw LLM output string
        
    Returns:
        dict with keys: state, district, village, other_locations (None if not present)
    """
    location_dict = {
        'state': None,
        'district': None,
        'village': None,
        'other_locations': None
    }
    
    if not text or not isinstance(text, str):
        return location_dict
    
    # Clean up common LLM artifacts
    text = str(text).strip()
    
    # Remove code fences (```text``` or ```json```)
    text = re.sub(r'```(?:text|json)?\s*([\s\S]*?)\s*```', r'\1', text, flags=re.IGNORECASE)
    text = text.replace('`', '')
    
    # Try to extract just the location part if there's conversational text
    # Look for the structured format: state: ..., district: ..., etc.
    match = re.search(
        r'(state:\s*[^,]+(?:,\s*district:\s*[^,]+)?(?:,\s*village:\s*[^,]+)?(?:,\s*other_locations:\s*[^,]+)?)',
        text,
        re.IGNORECASE
    )
    if match:
        text = match.group(1)
    
    # Parse using standard structured location parser
    try:
        return parse_structured_location(text)
    except:
        # Fallback: manual parsing
        parts = [part.strip() for part in text.split(',')]
        for part in parts:
            if ':' in part:
                label, value = part.split(':', 1)
                label = label.strip().lower()
                value = value.strip()
                if label in location_dict and value:
                    location_dict[label] = value
        return location_dict


def dict_to_structured_string(location_dict: dict) -> str:
    """
    Convert location dictionary back to structured string format.
    
    Args:
        location_dict: Dictionary with state, district, village, other_locations keys
        
    Returns:
        Structured string in format: "state: X, district: Y, village: Z, other_locations: W"
    """
    parts = []
    for key in ['state', 'district', 'village', 'other_locations']:
        if location_dict.get(key):
            parts.append(f"{key}: {location_dict[key]}")
    return ', '.join(parts) if parts else ''


def compute_location_metrics_from_strings(predictions: list, labels: list, fuzzy_threshold: int = 85) -> dict:
    """
    Compute comprehensive location extraction metrics from string predictions.
    
    This is the string-based version of metrics_utils.compute_metrics(), designed for
    LLM outputs that are already decoded strings (not token IDs).
    
    Args:
        predictions: List of predicted location strings
        labels: List of true location strings
        fuzzy_threshold: Similarity threshold for fuzzy matching (default: 85)
    
    Returns:
        dict with 'overall' and 'levels' keys:
        - overall: overall metrics (exact_match, exact_core_match, fuzzy_match, 
                  fuzzy_core_match, micro-averaged metrics, total_examples)
        - levels: per-level metrics (state, district, village, other_locations)
                  each with exact and fuzzy precision/recall/F1/support
    """
    # Initialize counters for EXACT matching
    exact_matches = 0
    exact_core_matches = 0  # state + district + village only
    exact_level_metrics = {
        'state': {'correct': 0, 'predicted': 0, 'total': 0},
        'district': {'correct': 0, 'predicted': 0, 'total': 0},
        'village': {'correct': 0, 'predicted': 0, 'total': 0},
        'other_locations': {'correct': 0, 'predicted': 0, 'total': 0}
    }
    
    # Initialize counters for FUZZY matching
    fuzzy_matches = 0
    fuzzy_core_matches = 0
    fuzzy_level_metrics = {
        'state': {'correct': 0, 'predicted': 0, 'total': 0},
        'district': {'correct': 0, 'predicted': 0, 'total': 0},
        'village': {'correct': 0, 'predicted': 0, 'total': 0},
        'other_locations': {'correct': 0, 'predicted': 0, 'total': 0}
    }
    
    total_examples = len(predictions)
    
    for pred_str, label_str in zip(predictions, labels):
        # Parse both strings
        pred_dict = parse_location_from_llm(pred_str)
        label_dict = parse_location_from_llm(label_str)
        
        # Check exact match for entire prediction
        if pred_dict == label_dict:
            exact_matches += 1
        
        # Check exact match for core geography
        core_exact_match = True
        for level in ['state', 'district', 'village']:
            if pred_dict[level] != label_dict[level]:
                core_exact_match = False
                break
        if core_exact_match:
            exact_core_matches += 1
        
        # Check fuzzy match for entire prediction
        all_levels_fuzzy_match = True
        for level in ['state', 'district', 'village', 'other_locations']:
            if not fuzzy_match(pred_dict[level], label_dict[level], threshold=fuzzy_threshold):
                all_levels_fuzzy_match = False
                break
        if all_levels_fuzzy_match:
            fuzzy_matches += 1
        
        # Check fuzzy match for core geography
        core_fuzzy_match = True
        for level in ['state', 'district', 'village']:
            if not fuzzy_match(pred_dict[level], label_dict[level], threshold=fuzzy_threshold):
                core_fuzzy_match = False
                break
        if core_fuzzy_match:
            fuzzy_core_matches += 1
        
        # Compute per-level metrics (EXACT)
        for level in ['state', 'district', 'village', 'other_locations']:
            if label_dict[level] is not None:
                exact_level_metrics[level]['total'] += 1
            if pred_dict[level] is not None:
                exact_level_metrics[level]['predicted'] += 1
            if pred_dict[level] is not None and label_dict[level] is not None:
                if pred_dict[level].lower() == label_dict[level].lower():
                    exact_level_metrics[level]['correct'] += 1
        
        # Compute per-level metrics (FUZZY)
        for level in ['state', 'district', 'village', 'other_locations']:
            if label_dict[level] is not None:
                fuzzy_level_metrics[level]['total'] += 1
            if pred_dict[level] is not None:
                fuzzy_level_metrics[level]['predicted'] += 1
            if pred_dict[level] is not None and label_dict[level] is not None:
                if fuzzy_match(pred_dict[level], label_dict[level], threshold=fuzzy_threshold):
                    fuzzy_level_metrics[level]['correct'] += 1
    
    # Compute overall metrics
    overall_metrics = {
        'exact_match': exact_matches / total_examples * 100,
        'exact_core_match': exact_core_matches / total_examples * 100,
        'fuzzy_match': fuzzy_matches / total_examples * 100,
        'fuzzy_core_match': fuzzy_core_matches / total_examples * 100,
        'total_examples': total_examples
    }
    
    # Compute micro-averaged metrics (EXACT)
    total_correct = sum(exact_level_metrics[level]['correct'] for level in ['state', 'district', 'village', 'other_locations'])
    total_predicted = sum(exact_level_metrics[level]['predicted'] for level in ['state', 'district', 'village', 'other_locations'])
    total_actual = sum(exact_level_metrics[level]['total'] for level in ['state', 'district', 'village', 'other_locations'])
    
    micro_exact_precision = (total_correct / total_predicted * 100) if total_predicted > 0 else 0
    micro_exact_recall = (total_correct / total_actual * 100) if total_actual > 0 else 0
    micro_exact_f1 = (2 * micro_exact_precision * micro_exact_recall / (micro_exact_precision + micro_exact_recall)) if (micro_exact_precision + micro_exact_recall) > 0 else 0
    
    overall_metrics['micro_exact_precision'] = micro_exact_precision
    overall_metrics['micro_exact_recall'] = micro_exact_recall
    overall_metrics['micro_exact_f1'] = micro_exact_f1
    
    # Compute micro-averaged metrics (FUZZY)
    total_correct = sum(fuzzy_level_metrics[level]['correct'] for level in ['state', 'district', 'village', 'other_locations'])
    total_predicted = sum(fuzzy_level_metrics[level]['predicted'] for level in ['state', 'district', 'village', 'other_locations'])
    total_actual = sum(fuzzy_level_metrics[level]['total'] for level in ['state', 'district', 'village', 'other_locations'])
    
    micro_fuzzy_precision = (total_correct / total_predicted * 100) if total_predicted > 0 else 0
    micro_fuzzy_recall = (total_correct / total_actual * 100) if total_actual > 0 else 0
    micro_fuzzy_f1 = (2 * micro_fuzzy_precision * micro_fuzzy_recall / (micro_fuzzy_precision + micro_fuzzy_recall)) if (micro_fuzzy_precision + micro_fuzzy_recall) > 0 else 0
    
    overall_metrics['micro_fuzzy_precision'] = micro_fuzzy_precision
    overall_metrics['micro_fuzzy_recall'] = micro_fuzzy_recall
    overall_metrics['micro_fuzzy_f1'] = micro_fuzzy_f1
    
    # Compute per-level metrics
    level_metrics = {}
    for level in ['state', 'district', 'village', 'other_locations']:
        level_metrics[level] = {}
        
        # Exact metrics
        exact_metrics = exact_level_metrics[level]
        exact_precision = (exact_metrics['correct'] / exact_metrics['predicted'] * 100) if exact_metrics['predicted'] > 0 else 0
        exact_recall = (exact_metrics['correct'] / exact_metrics['total'] * 100) if exact_metrics['total'] > 0 else 0
        exact_f1 = (2 * exact_precision * exact_recall / (exact_precision + exact_recall)) if (exact_precision + exact_recall) > 0 else 0
        
        level_metrics[level]['exact_precision'] = exact_precision
        level_metrics[level]['exact_recall'] = exact_recall
        level_metrics[level]['exact_f1'] = exact_f1
        level_metrics[level]['support'] = exact_metrics['total']
        
        # Fuzzy metrics
        fuzzy_metrics = fuzzy_level_metrics[level]
        fuzzy_precision = (fuzzy_metrics['correct'] / fuzzy_metrics['predicted'] * 100) if fuzzy_metrics['predicted'] > 0 else 0
        fuzzy_recall = (fuzzy_metrics['correct'] / fuzzy_metrics['total'] * 100) if fuzzy_metrics['total'] > 0 else 0
        fuzzy_f1 = (2 * fuzzy_precision * fuzzy_recall / (fuzzy_precision + fuzzy_recall)) if (fuzzy_precision + fuzzy_recall) > 0 else 0
        
        level_metrics[level]['fuzzy_precision'] = fuzzy_precision
        level_metrics[level]['fuzzy_recall'] = fuzzy_recall
        level_metrics[level]['fuzzy_f1'] = fuzzy_f1
    
    return {
        'overall': overall_metrics,
        'levels': level_metrics
    }


def print_location_metrics(metrics: dict, model_name: str = "Model"):
    """
    Pretty print location extraction metrics (alias for metrics_utils.print_metrics).
    
    Args:
        metrics: Dictionary from compute_location_metrics_from_strings() with 'overall' and 'levels' keys
        model_name: Name of the model for display
    """
    # Import here to avoid circular dependency
    from .metrics_utils import print_metrics
    print_metrics(metrics, model_name)


def run_and_save_llm_location_results(
    model_name: str, 
    outputs: list, 
    df_input, 
    id_col: str, 
    output_dir,
    timing: dict = None
):
    """
    Parse LLM outputs, compute metrics, save results, and return dataframe with metrics.
    
    This is the standard workflow for evaluating LLM location extraction:
    1. Parse raw LLM outputs into structured format
    2. Compute comprehensive metrics (exact + fuzzy, per-level, micro-averaged)
    3. Save results CSV and metrics JSON
    4. Print metrics summary
    5. Return results dataframe and metrics dict
    
    Args:
        model_name: Model identifier (used for file naming and display)
        outputs: List of raw LLM output strings
        df_input: Input dataframe with incident data and true labels
        id_col: Name of ID column in dataframe
        output_dir: Directory to save results (Path or str)
        timing: Optional timing dictionary from time_inference_call (added to metrics)
    
    Returns:
        tuple: (results_df, metrics_dict)
            - results_df: DataFrame with predictions and true labels
            - metrics_dict: Dictionary with 'overall' and 'levels' metrics
    """
    import json
    import pandas as pd
    from pathlib import Path
    
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    
    # Parse predictions and convert to structured strings
    parsed_dicts = [parse_location_from_llm(s) for s in outputs]
    parsed_strings = [dict_to_structured_string(d) for d in parsed_dicts]
    
    # Get true labels
    true_labels = df_input['human_annotated_location'].values
    
    # Compute metrics
    metrics = compute_location_metrics_from_strings(parsed_strings, true_labels)
    
    # Add timing if provided
    if timing:
        metrics['timing'] = timing
        print(f"\n⏱️  Timing: {timing['total_time_seconds']:.2f}s total, "
              f"{timing['time_per_item_seconds']:.3f}s/incident, "
              f"{timing['throughput_items_per_second']:.2f} incidents/s")
    
    # Print metrics
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} Results")
    print(f"{'='*80}")
    print_location_metrics(metrics, model_name)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        id_col: df_input[id_col].values,
        'incident_summary': df_input['incident_summary'].values,
        'true_location': true_labels,
        f'{model_name}_raw': outputs,
        f'{model_name}_prediction': parsed_strings
    })
    
    # Save results
    results_path = output_dir / f"{model_name}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Saved results to: {results_path}")
    
    # Save metrics
    metrics_path = output_dir / f"{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to: {metrics_path}")
    
    return results_df, metrics


# ============================================================================
# Location-Specific LLM Inference Runners
# ============================================================================
# These functions take pre-formatted prompts (not raw texts) and run inference
# WITHOUT applying any additional prompt wrapping. This differs from the count-models
# utilities which internally apply death-count prompts.


@torch.inference_mode()
def run_location_causal_batch(
    tok,
    model,
    prompts: List[str],
    max_new_tokens: int = 64,
    max_input_tokens: int = 512,
    batch_size: int = 16,
    show_progress: bool = True
):
    """
    Run inference on location extraction prompts using a causal LM with proper batching.
    
    NOTE: Takes PRE-FORMATTED prompts (already include location extraction instruction).
    Does NOT apply any additional prompt wrapping (unlike count-models utilities).
    
    Performance optimizations:
    - Batch processing (default 16-32 prompts at once) for GPU efficiency
    - Truncated inputs (max_input_tokens=512) for speed
    - Shorter outputs (max_new_tokens=64) - locations are typically short
    
    Args:
        tok: Tokenizer
        model: Causal language model
        prompts: List of PRE-FORMATTED location extraction prompts
        max_new_tokens: Maximum tokens to generate (default 64, locations are short)
        max_input_tokens: Maximum input tokens (default 512, truncate long inputs)
        batch_size: Number of prompts to process in parallel (default 16)
        show_progress: Whether to show progress
        
    Returns:
        list: List of model output strings
    """
    # Configure generation settings for deterministic output
    if hasattr(model, "generation_config"):
        try:
            model.generation_config.do_sample = False
        except AttributeError:
            pass
        try:
            if getattr(model.generation_config, "temperature", None) not in (None, 1.0):
                model.generation_config.temperature = 1.0
        except AttributeError:
            pass

    # Ensure padding token is set
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    outs = []
    total = len(prompts)
    num_batches = (total + batch_size - 1) // batch_size

    progress_bar = None
    use_simple_progress = False
    if show_progress and total > 0:
        if tqdm is not None:
            progress_bar = tqdm(total=total, desc="Generating", leave=False)
        else:
            use_simple_progress = True
            print("  Processing 0/{}...".format(total), end='\r', flush=True)
    
    # Process in batches for GPU efficiency
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_prompts = prompts[start_idx:end_idx]
        
        if use_simple_progress:
            print(f"  Processing batch {batch_idx + 1}/{num_batches} ({end_idx}/{total})...", end='\r', flush=True)
        
        # Tokenize batch with padding and truncation
        inputs = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens
        ).to(model.device)
        
        # Generate for entire batch
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id
        )
        
        # Decode each output in the batch
        for i in range(len(batch_prompts)):
            # Extract only the generated tokens (skip input prompt)
            input_len = inputs["input_ids"][i].shape[0]
            out = tok.decode(
                gen[i][input_len:], 
                skip_special_tokens=True
            ).strip()
            outs.append(out)
            
            if progress_bar is not None:
                progress_bar.update(1)
    
    if progress_bar is not None:
        progress_bar.close()
    elif show_progress and total > 0:
        print(f"  Completed {total}/{total}      ")
    
    return outs


@torch.inference_mode()
def run_location_t5_batch(
    tok,
    model,
    prompts: List[str],
    max_new_tokens: int = 64,
    max_input_tokens: int = 512,
    batch_size: int = 16,
    show_progress: bool = True
):
    """
    Run inference on location extraction prompts using a T5 model with proper batching.
    
    NOTE: Takes PRE-FORMATTED prompts (already include location extraction instruction).
    Does NOT apply any additional prompt wrapping (unlike count-models utilities).
    
    Performance optimizations:
    - Batch processing (default 16-32 prompts at once) for GPU efficiency
    - Truncated inputs (max_input_tokens=512) for speed
    - Shorter outputs (max_new_tokens=64) - locations are typically short
    
    Args:
        tok: Tokenizer
        model: T5 seq2seq model
        prompts: List of PRE-FORMATTED location extraction prompts
        max_new_tokens: Maximum tokens to generate (default 64, locations are short)
        max_input_tokens: Maximum tokens for encoder input (default 512)
        batch_size: Number of prompts to process in parallel (default 16)
        show_progress: Whether to show progress
        
    Returns:
        list: List of model output strings
    """
    outs = []
    total = len(prompts)
    num_batches = (total + batch_size - 1) // batch_size

    progress_bar = None
    use_simple_progress = False
    if show_progress and total > 0:
        if tqdm is not None:
            progress_bar = tqdm(total=total, desc="Generating", leave=False)
        else:
            use_simple_progress = True
            print("  Processing 0/{}...".format(total), end='\r', flush=True)
    
    # Process in batches for GPU efficiency
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_prompts = prompts[start_idx:end_idx]
        
        if use_simple_progress:
            print(f"  Processing batch {batch_idx + 1}/{num_batches} ({end_idx}/{total})...", end='\r', flush=True)
        
        # Tokenize batch with padding and truncation
        encoded = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        tensor_inputs = {
            "input_ids": encoded["input_ids"].to(model.device),
            "attention_mask": encoded["attention_mask"].to(model.device),
        }
        
        # Generate for entire batch
        gen = model.generate(
            **tensor_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        
        # Decode each output in the batch
        for i in range(len(batch_prompts)):
            out = tok.decode(gen[i], skip_special_tokens=True).strip()
            outs.append(out)
            
            if progress_bar is not None:
                progress_bar.update(1)
    
    if progress_bar is not None:
        progress_bar.close()
    elif show_progress and total > 0:
        print(f"  Completed {total}/{total}      ")
    
    return outs


def run_location_openai_batch(
    prompts: List[str],
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    max_tokens: int = 256,
    rate_limit_delay: float = 0.1,
    show_progress: bool = True
):
    """
    Run inference on location extraction prompts using OpenAI API.
    
    NOTE: Takes PRE-FORMATTED prompts (already include location extraction instruction).
    Does NOT apply any additional prompt wrapping (unlike count-models utilities).
    
    Args:
        prompts: List of PRE-FORMATTED location extraction prompts
        api_key: OpenAI API key
        model_name: OpenAI model name
        max_tokens: Maximum tokens to generate
        rate_limit_delay: Delay between requests (seconds)
        show_progress: Whether to show progress
        
    Returns:
        list: List of model output strings
    """
    if api_key is None:
        try:
            from google.colab import userdata
            api_key = userdata.get('openai_api_key')
        except ImportError:
            import os
            api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. "
            "Set OPENAI_API_KEY environment variable or add 'openai_api_key' to Colab secrets."
        )
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError(
            "openai package not installed. Install with: pip install openai>=1.0.0"
        )
    
    outs = []
    total = len(prompts)
    errors = 0

    progress_bar = None
    use_simple_progress = False
    if show_progress and total > 0:
        if tqdm is not None:
            progress_bar = tqdm(total=total, desc="Generating (OpenAI)", leave=False)
        else:
            use_simple_progress = True
            print(f"  Processing 0/{total}...", flush=True)
    
    for i, prompt in enumerate(prompts):
        try:
            # Use prompt as-is (already formatted by caller)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0
            )
            out = response.choices[0].message.content.strip()
            outs.append(out)
        except Exception as e:
            outs.append("")
            errors += 1
        
        if progress_bar is not None:
            progress_bar.update(1)
        elif use_simple_progress:
            print(f"  Processing {i + 1}/{total}...", flush=True)
        
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
    
    if progress_bar is not None:
        progress_bar.close()
    
    if show_progress:
        print(f"  Completed {total}/{total} (errors: {errors})      ")
    
    if errors > 0:
        print(f"⚠️  Warning: {errors} errors occurred during API calls")
    
    return outs


def run_location_gemini_batch(
    prompts: List[str],
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    max_output_tokens: int = 256,
    rate_limit_delay: float = 0.1,
    show_progress: bool = True,
    max_retries: int = 4,
    max_concurrency: int = 8,
):
    """
    Run inference on location extraction prompts using Google Gemini API.
    
    NOTE: Takes PRE-FORMATTED prompts (already include location extraction instruction).
    Does NOT apply any additional prompt wrapping (unlike count-models utilities).
    
    Args:
        prompts: List of PRE-FORMATTED location extraction prompts
        api_key: Gemini API key
        model_name: Gemini model name
        max_output_tokens: Maximum tokens to generate
        rate_limit_delay: Delay between requests (seconds)
        show_progress: Whether to show progress
        max_retries: Maximum retries per item
        max_concurrency: Maximum number of concurrent API calls (default 8)
        
    Returns:
        list: List of model output strings
    """
    if api_key is None:
        try:
            from google.colab import userdata
            api_key = userdata.get('gemini_api_key')
        except ImportError:
            import os
            api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "Gemini API key not found. "
            "Set GEMINI_API_KEY environment variable or add 'gemini_api_key' to Colab secrets."
        )
    
    try:
        import google.generativeai as genai
        import importlib
        google_api_exceptions = None
        try:
            google_api_exceptions = importlib.import_module("google.api_core.exceptions")
        except ImportError:
            pass
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except ImportError:
        raise ImportError(
            "google-generativeai package not installed. Install with: pip install google-generativeai"
        )
    except Exception as exc:
        if google_api_exceptions and isinstance(exc, google_api_exceptions.GoogleAPIError):
            raise RuntimeError(
                f"Failed to initialize Gemini model '{model_name}'. "
                "Ensure your account has access and that you are on google-generativeai>=0.7.0."
            ) from exc
        raise
    
    outs = [""] * len(prompts)
    total = len(prompts)
    errors = 0

    progress_bar = None
    use_simple_progress = False
    if show_progress and total > 0:
        if tqdm is not None:
            progress_bar = tqdm(total=total, desc="Generating (Gemini)", leave=False)
        else:
            use_simple_progress = True
            print(f"  Processing 0/{total}...", flush=True)
    
    # Configure generation (no JSON mode for location extraction)
    gen_config: Dict[str, Any] = {
        "max_output_tokens": max_output_tokens,
        "temperature": 0.0,
        # Hint the model to avoid narrative formatting
        "response_mime_type": "text/plain",
    }

    # Configure permissive safety; prefer typed enums if available, else fall back to SDK-introspected strings
    safety_settings = None
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore
        # If typed enums exist, at least disable blocking on violence
        safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_VIOLENCE,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            }
        ]
    except Exception:
        safety_settings = None
    if safety_settings is None:
        # Fallback: attempt to set BLOCK_NONE across all categories supported by this SDK version
        try:
            import google.generativeai.types.safety_types as _st  # type: ignore
            _cats = list(getattr(_st, "_HARM_CATEGORIES", {}).keys())
            _ths = getattr(_st, "_HARM_BLOCK_THRESHOLDS", {})
            _t = "block_none" if "block_none" in _ths else (list(_ths.keys())[0] if _ths else None)
            if _cats and _t:
                safety_settings = [{"category": c, "threshold": _t} for c in _cats]
        except Exception:
            safety_settings = None

    # Worker function for a single prompt with retries and rate-limiting
    def _process_one(idx: int, prompt: str) -> tuple[int, str, Optional[str]]:
        last_error_local: Optional[str] = None
        out_local = ""
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=gen_config,
                    safety_settings=safety_settings
                )
                if hasattr(response, 'text') and response.text:
                    out_local = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        out_local = candidate.content.parts[0].text.strip()
                break
            except Exception as exc:
                last_error_local = str(exc)
                base = 0.5
                sleep_s = base * (2 ** attempt)
                try:
                    import random
                    sleep_s += random.random() * 0.2
                except Exception:
                    pass
                time.sleep(sleep_s)
        # Gentle pacing to avoid bursting the API
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
        return idx, out_local.strip(), last_error_local if not out_local else None

    # Execute with bounded concurrency while preserving output order
    if total > 0:
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
        except Exception:
            # Fallback to sequential if futures not available
            for i, p in enumerate(prompts):
                idx, out, err = _process_one(i, p)
                outs[idx] = out
                if err:
                    errors += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                elif use_simple_progress:
                    print(f"  Processing {i + 1}/{total}...", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as executor:
                futures = [executor.submit(_process_one, i, p) for i, p in enumerate(prompts)]
                for fut in as_completed(futures):
                    try:
                        idx, out, err = fut.result()
                        outs[idx] = out
                        if err:
                            errors += 1
                    except Exception as exc:
                        # Count as error and leave out as empty
                        errors += 1
                    finally:
                        if progress_bar is not None:
                            progress_bar.update(1)
                        elif use_simple_progress:
                            # Try to approximate completed count
                            done = sum(1 for f in futures if f.done())
                            print(f"  Processing {done}/{total}...", flush=True)
    
    if progress_bar is not None:
        progress_bar.close()
    
    if show_progress:
        print(f"  Completed {total}/{total} (errors: {errors})      ")
    
    if errors > 0:
        print(f"⚠️  Warning: {errors} errors occurred during API calls")
    
    return outs


def run_location_gemini_json_batch(
    texts: List[str],
    api_key: Optional[str] = None,
    model_name: str = "models/gemini-2.5-flash",
    max_output_tokens: int = 512,
    rate_limit_delay: float = 0.1,
    show_progress: bool = True,
    max_retries: int = 4,
    max_concurrency: int = 8,
    max_chars: int = 350,
):
    """
    Run inference for location extraction using Gemini with JSON-only responses.
    
    This variant:
      - Builds a strict JSON prompt per text
      - Truncates inputs to reduce safety blocks
      - Requests application/json MIME type
      - Converts JSON to the standard structured string
        "state: X, district: Y, village: Z, other_locations: W"
    
    Args:
        texts: List of incident summaries (raw text)
        api_key: Gemini API key
        model_name: Gemini model (prefer full resource name: "models/gemini-2.5-flash")
        max_output_tokens: Max tokens for JSON output (default 512)
        rate_limit_delay: Delay between requests (seconds)
        show_progress: Show progress bar if tqdm available
        max_retries: Retries per item with backoff
        max_concurrency: Max parallel requests
        max_chars: Truncate input to this many characters
    
    Returns:
        list[str]: Structured location strings for each input text
    """
    # Deps imported lazily to avoid hard requirements outside this path
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai package not installed. Install with: pip install google-generativeai"
        )
    import json as _json
    import math as _math
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import random as _random
    
    # Resolve API key
    if api_key is None:
        try:
            from google.colab import userdata  # type: ignore
            api_key = userdata.get('gemini_api_key')
        except ImportError:
            import os as _os
            api_key = _os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError(
            "Gemini API key not found. "
            "Set GEMINI_API_KEY environment variable or add 'gemini_api_key' to Colab secrets."
        )
    
    # Configure SDK and model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Build prompts
    def _truncate(s: str, n: int) -> str:
        return s if len(s) <= n else s[:n] + "..."
    
    def _make_json_prompt(text: str) -> str:
        return (
            "Extract the location as JSON with exactly these string keys:\n"
            '  {\"state\": \"\", \"district\": \"\", \"village\": \"\", \"other_locations\": \"\"}\n'
            "- Use empty string if unknown. Output ONLY one-line JSON, nothing else.\n\n"
            f"Incident: {text}\n\nJSON:"
        )
    
    prompts = [_make_json_prompt(_truncate(s or "", max_chars)) for s in texts]
    
    # Generation config
    gen_config: Dict[str, Any] = {
        "max_output_tokens": max_output_tokens,
        "temperature": 0.0,
        "response_mime_type": "application/json",
    }
    
    # Helper to extract first text part
    def _first_text_part(resp) -> str:
        try:
            if getattr(resp, "candidates", None):
                c0 = resp.candidates[0]
                if getattr(c0, "content", None) and getattr(c0.content, "parts", None):
                    for part in c0.content.parts:
                        if hasattr(part, "text") and part.text:
                            return part.text.strip()
        except Exception:
            pass
        return ""
    
    # Convert JSON string to standardized structured string
    def _json_to_structured(s: str) -> str:
        if not s:
            return ""
        try:
            d = _json.loads(s)
            norm = {
                "state": (d.get("state") or "").strip() or None,
                "district": (d.get("district") or "").strip() or None,
                "village": (d.get("village") or "").strip() or None,
                "other_locations": (d.get("other_locations") or "").strip() or None,
            }
            return dict_to_structured_string(norm)
        except Exception:
            return ""
    
    outs_structured = [""] * len(prompts)
    total = len(prompts)
    errors = 0
    
    progress_bar = None
    use_simple_progress = False
    if show_progress and total > 0:
        if tqdm is not None:
            progress_bar = tqdm(total=total, desc="Generating (Gemini JSON)", leave=False)
        else:
            use_simple_progress = True
            print(f"  Processing 0/{total}...", flush=True)
    
    # Worker with retries and pacing
    def _process_one(idx: int, prompt: str) -> tuple[int, str, Optional[str]]:
        last_error_local: Optional[str] = None
        out_local = ""
        for attempt in range(max_retries):
            try:
                resp = model.generate_content(
                    prompt,
                    generation_config=gen_config,
                    safety_settings=None
                )
                txt = _first_text_part(resp)
                out_local = _json_to_structured(txt)
                break
            except Exception as exc:
                last_error_local = str(exc)
                base = 0.5
                sleep_s = base * (2 ** attempt)
                try:
                    sleep_s += _random.random() * 0.2
                except Exception:
                    pass
                time.sleep(sleep_s)
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
        return idx, out_local.strip(), last_error_local if not out_local else None
    
    # Execute with concurrency, preserving order
    if total > 0:
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
        except Exception:
            # Fallback to sequential
            for i, p in enumerate(prompts):
                idx, out, err = _process_one(i, p)
                outs_structured[idx] = out
                if err:
                    errors += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                elif use_simple_progress:
                    print(f"  Processing {i + 1}/{total}...", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as executor:
                futures = [executor.submit(_process_one, i, p) for i, p in enumerate(prompts)]
                for fut in as_completed(futures):
                    try:
                        idx, out, err = fut.result()
                        outs_structured[idx] = out
                        if err:
                            errors += 1
                    except Exception:
                        errors += 1
                    finally:
                        if progress_bar is not None:
                            progress_bar.update(1)
                        elif use_simple_progress:
                            done = sum(1 for f in futures if f.done())
                            print(f"  Processing {done}/{total}...", flush=True)
    
    if progress_bar is not None:
        progress_bar.close()
    
    if show_progress:
        print(f"  Completed {total}/{total} (errors: {errors})      ")
    
    if errors > 0:
        print(f"⚠️  Warning: {errors} errors occurred during API calls")
    
    return outs_structured
