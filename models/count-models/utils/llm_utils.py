"""Utilities for running LLM inference for death count extraction."""

import os
import re
import time
from typing import List, Optional
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM
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

    try:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            token=hf_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=DTYPE,
            load_in_4bit=USE_4BIT,
            bnb_4bit_compute_dtype=DTYPE if USE_4BIT else None,
            token=hf_token
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
        torch_dtype=DTYPE
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
    outs = []
    total = len(texts)
    
    for i, t in enumerate(texts):
        if show_progress and (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{total}...", end='\r')
        
        # Use chat template if available (for instruction-tuned models)
        if hasattr(tok, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are a precise information extractor."},
                {"role": "user", "content": make_input(t)}
            ]
            prompt = tok.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            prompt = make_input(t)
        
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            pad_token_id=tok.eos_token_id
        )
        out = tok.decode(
            gen[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        outs.append(out)
    
    if show_progress:
        print(f"  Completed {total}/{total}      ")
    
    return outs


@torch.inference_mode()
def run_t5_batch(
    tok, 
    model, 
    texts: List[str], 
    max_new_tokens: int = 32,
    show_progress: bool = True
):
    """
    Run inference on a batch of texts using a T5 model.
    
    Args:
        tok: Tokenizer
        model: T5 seq2seq model
        texts: List of input texts
        max_new_tokens: Maximum tokens to generate
        show_progress: Whether to show progress
        
    Returns:
        list: List of model output strings
    """
    outs = []
    total = len(texts)
    
    for i, t in enumerate(texts):
        if show_progress and (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{total}...", end='\r')
        
        prompt = f"Extract deaths as JSON.\n\n{make_input(t)}"
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0
        )
        out = tok.decode(gen[0], skip_special_tokens=True).strip()
        outs.append(out)
    
    if show_progress:
        print(f"  Completed {total}/{total}      ")
    
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
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a precise information extractor."},
                    {"role": "user", "content": make_input(text)}
                ],
                max_tokens=max_tokens,
                temperature=0.0
            )
            out = response.choices[0].message.content.strip()
            outs.append(out)
        except Exception as e:
            print(f"\n  Error on item {i + 1}: {e}")
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
    model_name: str = "gemini-1.5-flash",
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
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except ImportError:
        raise ImportError(
            "google-generativeai package not installed. Install with: pip install google-generativeai"
        )
    
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
            out = response.text.strip() if response.text else ""
            outs.append(out)
        except Exception as e:
            print(f"\n  Error on item {i + 1}: {e}")
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

