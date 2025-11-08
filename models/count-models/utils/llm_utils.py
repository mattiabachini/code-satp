"""Utilities for running LLM inference for death count extraction."""

import os
import re
import time
from typing import List, Optional
from pathlib import Path
import importlib

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)
from huggingface_hub.utils import GatedRepoError


# Configuration constants
DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16
USE_4BIT = True

# Instruction template for prompts
INSTR = (
    "How many people were killed? Answer with only a number. "
    "Return JSON exactly as: {\"fatalities\": <integer>}. If no fatalities are mentioned, use 0."
)


def make_input(text: str) -> str:
    """Create input prompt for the model."""
    return f"{INSTR}\n\nText: {text}\nAnswer:"


def parse_fatalities(s: str) -> int:
    """
    Parse fatalities count from model output.
    
    Args:
        s: Model output string (may contain JSON or plain number)
        
    Returns:
        int: Extracted fatalities count (0 if not found)
    """
    if not s:
        return 0
    
    # Try to extract from JSON format first
    m = re.search(r'"fatalities"\s*:\s*(-?\d+)', s or "")
    if m:
        return max(0, int(m.group(1)))
    
    # Try to find any number in the output
    m = re.search(r'-?\d+', s or "")
    return max(0, int(m.group(0))) if m else 0


def _resolve_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """
    Resolve a Hugging Face token from explicit argument or environment.

    Args:
        explicit_token: Token passed to the function

    Returns:
        Optional[str]: Token string if available
    """
    if explicit_token:
        return explicit_token

    for env_var in ("HUGGINGFACE_TOKEN", "HF_TOKEN"):
        token = os.environ.get(env_var)
        if token:
            return token

    return None


def load_causal(model_id: str, token: Optional[str] = None):
    """
    Load a causal language model (for instruction-tuned models like Llama, Mistral).
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        tuple: (tokenizer, model)
    """
    hf_token = _resolve_hf_token(token)

    quantization_config = None
    if USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE,
        )

    try:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            token=hf_token
        )
        model_kwargs = {
            "device_map": "auto",
            "token": hf_token,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["dtype"] = DTYPE

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
    except GatedRepoError as exc:
        hint = (
            f"Access to the gated model '{model_id}' requires an approved Hugging Face token. "
            "Visit the model card, request access if needed, then provide your token via "
            "`HUGGINGFACE_TOKEN` or `HF_TOKEN` environment variables, or call "
            "`huggingface_hub.login()` before loading the model."
        )
        if hf_token is None:
            hint += " No token was detected in the current environment."
        raise RuntimeError(hint) from exc

    return tok, model


def load_t5(model_id: str):
    """
    Load a T5 seq2seq model.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        tuple: (tokenizer, model)
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=DTYPE
    )
    return tok, model


@torch.inference_mode()
def run_causal_batch(
    tok, 
    model, 
    texts: List[str], 
    max_new_tokens: int = 48,
    show_progress: bool = True
):
    """
    Run inference on a batch of texts using a causal LM.
    
    Args:
        tok: Tokenizer
        model: Causal language model
        texts: List of input texts
        max_new_tokens: Maximum tokens to generate
        show_progress: Whether to show progress
        
    Returns:
        list: List of model output strings
    """
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

    outs = []
    total = len(texts)

    progress_bar = None
    use_simple_progress = False
    if show_progress and total > 0:
        if tqdm is not None:
            progress_bar = tqdm(total=total, desc="Generating", leave=False)
        else:
            use_simple_progress = True
            print("  Processing 0/{}...".format(total), end='\r', flush=True)
    
    for i, t in enumerate(texts):
        if use_simple_progress:
            print(f"  Processing {i + 1}/{total}...", end='\r', flush=True)
        
        # Use standardized basic prompt for all models (fair comparison)
        prompt = make_input(t)
        
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )
        out = tok.decode(
            gen[0][inputs["input_ids"].shape[1]:], 
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
def run_t5_batch(
    tok, 
    model, 
    texts: List[str], 
    max_new_tokens: int = 32,
    max_input_tokens: int = 512,
    show_progress: bool = True
):
    """
    Run inference on a batch of texts using a T5 model.
    
    Args:
        tok: Tokenizer
        model: T5 seq2seq model
        texts: List of input texts
        max_new_tokens: Maximum tokens to generate
        max_input_tokens: Maximum tokens for encoder input (truncates beyond limit)
        show_progress: Whether to show progress
        
    Returns:
        list: List of model output strings
    """
    outs = []
    total = len(texts)
    truncated = 0

    progress_bar = None
    use_simple_progress = False
    if show_progress and total > 0:
        if tqdm is not None:
            progress_bar = tqdm(total=total, desc="Generating", leave=False)
        else:
            use_simple_progress = True
            print("  Processing 0/{}...".format(total), end='\r', flush=True)
    
    for i, t in enumerate(texts):
        if use_simple_progress:
            print(f"  Processing {i + 1}/{total}...", end='\r', flush=True)
        
        # Use standardized basic prompt for all models (fair comparison)
        prompt = make_input(t)
        encoded = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        if encoded.get("overflowing_tokens"):
            truncated += 1
        tensor_inputs = {
            k: v.to(model.device)
            for k, v in encoded.items()
            if isinstance(v, torch.Tensor) and k != "overflowing_tokens"
        }
        gen = model.generate(
            **tensor_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        out = tok.decode(gen[0], skip_special_tokens=True).strip()
        outs.append(out)

        if progress_bar is not None:
            progress_bar.update(1)
    
    if progress_bar is not None:
        progress_bar.close()
    elif show_progress and total > 0:
        print(f"  Completed {total}/{total}      ")

    if truncated > 0:
        print(f"⚠️  Warning: {truncated} input(s) truncated to {max_input_tokens} tokens.")
    
    return outs


def run_openai_batch(
    texts: List[str],
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    max_tokens: int = 50,
    rate_limit_delay: float = 0.1,
    show_progress: bool = True
):
    """
    Run inference on a batch of texts using OpenAI API.
    
    Args:
        texts: List of input texts
        api_key: OpenAI API key (if None, tries to get from environment/secrets)
        model_name: OpenAI model name
        max_tokens: Maximum tokens to generate
        rate_limit_delay: Delay between requests (seconds)
        show_progress: Whether to show progress
        
    Returns:
        list: List of model output strings
        
    Raises:
        ValueError: If API key is not found
        Exception: If API call fails
    """
    # Try to get API key from various sources
    if api_key is None:
        try:
            # Try Colab secrets first
            from google.colab import userdata
            api_key = userdata.get('openai_api_key')
        except ImportError:
            # Try environment variable
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
    total = len(texts)
    errors = 0
    
    for i, text in enumerate(texts):
        if show_progress and (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{total}...", end='\r')
        
        try:
            # Use standardized basic prompt for all models (fair comparison)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": make_input(text)}
                ],
                max_tokens=max_tokens,
                temperature=0.0
            )
            out = response.choices[0].message.content.strip()
            outs.append(out)
        except Exception as e:
            # Don't print error here to avoid disrupting progress bar
            outs.append("")  # Empty string on error
            errors += 1
        
        # Rate limiting
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
    
    if show_progress:
        print(f"  Completed {total}/{total} (errors: {errors})      ")
    
    if errors > 0:
        print(f"⚠️  Warning: {errors} errors occurred during API calls")
    
    return outs


def run_gemini_batch(
    texts: List[str],
    api_key: Optional[str] = None,
    model_name: str = "gemini-1.5-flash-latest",
    max_output_tokens: int = 50,
    rate_limit_delay: float = 0.1,
    show_progress: bool = True
):
    """
    Run inference on a batch of texts using Google Gemini API.
    
    Args:
        texts: List of input texts
        api_key: Gemini API key (if None, tries to get from environment/secrets)
        model_name: Gemini model name
        max_output_tokens: Maximum tokens to generate
        rate_limit_delay: Delay between requests (seconds)
        show_progress: Whether to show progress
        
    Returns:
        list: List of model output strings
        
    Raises:
        ValueError: If API key is not found
        Exception: If API call fails
    """
    # Try to get API key from various sources
    if api_key is None:
        try:
            # Try Colab secrets first
            from google.colab import userdata
            api_key = userdata.get('gemini_api_key')
        except ImportError:
            # Try environment variable
            import os
            api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "Gemini API key not found. "
            "Set GEMINI_API_KEY environment variable or add 'gemini_api_key' to Colab secrets."
        )
    
    google_api_exceptions = None

    try:
        import google.generativeai as genai
        try:
            google_api_exceptions = importlib.import_module("google.api_core.exceptions")
        except ImportError:  # pragma: no cover - optional dependency
            google_api_exceptions = None
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
                "If you are using Gemini 1.5 models, ensure your account has access and "
                "that you are on google-generativeai>=0.7.0."
            ) from exc
        raise RuntimeError(
            f"Failed to initialize Gemini client: {exc}"
        ) from exc
    
    outs = []
    total = len(texts)
    errors = 0
    
    for i, text in enumerate(texts):
        if show_progress and (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{total}...", end='\r')
        
        try:
            prompt = make_input(text)
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_output_tokens,
                    "temperature": 0.0
                }
            )
            # Handle response based on finish_reason
            # finish_reason: 1=STOP (normal), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:
                    # MAX_TOKENS - try to get partial content or fail gracefully
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        out = candidate.content.parts[0].text.strip()
                    else:
                        # Don't print warning here to avoid disrupting progress bar
                        errors += 1
                        out = ""
                elif hasattr(response, 'text') and response.text:
                    out = response.text.strip()
                else:
                    out = ""
            else:
                out = response.text.strip() if hasattr(response, 'text') and response.text else ""
            outs.append(out)
        except Exception as exc:
            if google_api_exceptions and isinstance(exc, google_api_exceptions.NotFound):
                raise RuntimeError(
                    f"Gemini model '{model_name}' is not available for generate_content in this project/API version. "
                    "Upgrade to the latest google-generativeai package (pip install --upgrade google-generativeai) "
                    "or switch to a supported model such as 'gemini-1.0-pro-latest'."
                ) from exc
            if google_api_exceptions and isinstance(exc, google_api_exceptions.GoogleAPIError):
                print(f"\n  Gemini API error on item {i + 1}: {exc}")
            else:
                print(f"\n  Error on item {i + 1}: {exc}")
            outs.append("")  # Empty string on error
            errors += 1
        
        # Rate limiting
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
    
    if show_progress:
        print(f"  Completed {total}/{total} (errors: {errors})      ")
    
    if errors > 0:
        print(f"⚠️  Warning: {errors} errors occurred during API calls")
    
    return outs


def already_done(model_name: str, output_dir: Path) -> bool:
    """
    Check if a model's results already exist.
    
    Args:
        model_name: Model identifier
        output_dir: Output directory to check
        
    Returns:
        bool: True if results file exists
    """
    return (output_dir / f"{model_name}.csv").exists()

