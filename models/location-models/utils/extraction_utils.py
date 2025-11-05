"""Utilities for extracting and parsing numbers from model outputs."""

import re
import pandas as pd


def extract_number(text):
    """
    Extract numeric count from model output.
    Returns None if extraction fails.
    
    Args:
        text: String output from model
        
    Returns:
        int: Extracted number or None if extraction fails
    """
    if pd.isna(text) or text == "":
        return None
    
    # Convert to string and strip
    text = str(text).strip().rstrip('.')
    
    # Try direct integer conversion first
    try:
        return max(0, int(text))
    except ValueError:
        pass
    
    # Try regex to find first number
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    
    # Try to convert words to numbers
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    text_lower = text.lower()
    for word, num in word_to_num.items():
        if word in text_lower:
            return num
    
    # Extraction failed
    return None


def parse_prediction(raw_output, model_type='seq2seq'):
    """
    Parse model output to numeric count.
    
    Args:
        raw_output: Raw output from model
        model_type: Type of model ('seq2seq', 'qa', or 'regression')
        
    Returns:
        int: Parsed numeric count (defaults to 0 if parsing fails)
    """
    if model_type == 'regression':
        # DistilBERT-Poisson outputs float
        try:
            return max(0, round(float(raw_output)))
        except:
            return 0
    else:
        # Seq2seq and QA models output text
        num = extract_number(raw_output)
        return num if num is not None else 0  # Default to 0 if extraction fails

