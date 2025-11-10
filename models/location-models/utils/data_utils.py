"""Utilities for preparing and tokenizing data for different model types."""

import re

import pandas as pd


def build_structured_location(row):
    """
    Build structured location string from component columns.

    Args:
        row: Pandas Series with location fields (state, district, village_name, other_locations).

    Returns:
        Structured location string with labeled components, or empty string if no components present.
    """
    parts = []

    if pd.notna(row.get('state')) and str(row.get('state')).strip():
        parts.append(f"state: {str(row['state']).strip()}")

    if pd.notna(row.get('district')) and str(row.get('district')).strip():
        parts.append(f"district: {str(row['district']).strip()}")

    if pd.notna(row.get('village_name')) and str(row.get('village_name')).strip():
        parts.append(f"village: {str(row['village_name']).strip()}")

    if pd.notna(row.get('other_locations')) and str(row.get('other_locations')).strip():
        parts.append(f"other_locations: {str(row['other_locations']).strip()}")

    return ', '.join(parts) if parts else ''


def prepare_seq2seq_data(df, model_type='nt5'):
    """
    Prepare data in format for seq2seq models (NT5, Flan-T5, IndicBART, mT5).
    
    Args:
        df: DataFrame with 'incident_summary' and 'total_fatalities' columns
        model_type: Type of model ('nt5', 'flan-t5', 'indicbart', 'mt5')
        
    Returns:
        dict with 'input' and 'target' lists
    """
    inputs = []
    targets = []
    
    for _, row in df.iterrows():
        # Format prompt based on model type
        if model_type == 'nt5':
            input_text = f"answer_me: How many people were killed? Answer with only a number. context: {row['incident_summary']}"
        elif model_type == 'mt5':
            input_text = f"<2en> How many people were killed? Answer with only a number.\n\n{row['incident_summary']}"
        else:  # flan-t5, indicbart
            input_text = f"How many people were killed? Answer with only a number.\n\n{row['incident_summary']}"
        
        target_text = str(row['total_fatalities'])
        inputs.append(input_text)
        targets.append(target_text)
    
    return {'input': inputs, 'target': targets}


def prepare_location_seq2seq_data(df):
    """
    Prepare data for structured location extraction seq2seq tasks.

    Args:
        df: DataFrame with location fields and incident metadata.

    Returns:
        dict with 'input' and 'target' text plus associated metadata columns.
    """
    inputs = [
        (
            "Extract location hierarchy from incident: "
            f"{summary}\nFormat: state: <name>, district: <name>, "
            "village: <name>, other_locations: <name>. "
            "Use exact format with labels. Omit missing levels."
        )
        for summary in df['incident_summary']
    ]
    targets = df['human_annotated_location'].tolist()

    dates = []
    for d in df['date']:
        if pd.isna(d):
            dates.append(None)
        else:
            try:
                dates.append(pd.to_datetime(d).strftime('%Y-%m-%d'))
            except Exception:
                dates.append(None)

    incident_numbers = [str(x) if pd.notna(x) else None for x in df['incident_number']]
    incident_summaries = [str(x) if pd.notna(x) else None for x in df['incident_summary']]

    return {
        'input': inputs,
        'target': targets,
        'date': dates,
        'incident_number': incident_numbers,
        'incident_summary': incident_summaries,
    }


def preview_location_examples(df, title, num_examples=3, text_limit=200):
    """
    Print sample incidents with location annotations for quick inspection.

    Args:
        df: DataFrame containing incident data.
        title: Heading to display before the examples.
        num_examples: Number of rows to show.
        text_limit: Max characters to display from incident_summary.
    """
    print(f"\n{title}")
    print("=" * 80)

    for _, row in df.head(num_examples).iterrows():
        summary = row.get('incident_summary', '') or ''
        if isinstance(summary, str) and len(summary) > text_limit:
            summary = summary[:text_limit] + "..."

        print(f"Date: {row.get('date')}")
        print(f"State: {row.get('state')}")
        print(f"District: {row.get('district')}")
        print(f"Village: {row.get('village_name')}")
        print(f"Other Locations: {row.get('other_locations')}")
        print(f"Structured Labels: {row.get('human_annotated_location')}")
        print(f"Incident Summary: {summary}")
        print("-" * 80)


def prepare_regression_data(df):
    """
    Prepare data for regression models (DistilBERT-Poisson).
    
    Args:
        df: DataFrame with 'incident_summary' and 'total_fatalities' columns
        
    Returns:
        dict with 'text' and 'labels' lists
    """
    return {
        'text': df['incident_summary'].tolist(),
        'labels': df['total_fatalities'].tolist()
    }


def tokenize_seq2seq(examples, tokenizer, max_input_length=512, max_target_length=128):
    """
    Tokenize inputs and targets for seq2seq models.
    
    Args:
        examples: Dict with 'input' and 'target' keys
        tokenizer: HuggingFace tokenizer
        max_input_length: Maximum length for input sequences
        max_target_length: Maximum length for target sequences
        
    Returns:
        Tokenized examples with input_ids, attention_mask, and labels
    """
    model_inputs = tokenizer(
        examples['input'], 
        max_length=max_input_length, 
        truncation=True, 
        padding='max_length'
    )
    
    # Setup targets with proper target encoding pathway
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
        examples['target'], 
        max_length=max_target_length, 
        truncation=True, 
        padding='max_length'
    )
    
    # Replace padding token id's in the labels with -100 so they are ignored by the loss
    labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in target_encodings['input_ids']]
    model_inputs['labels'] = labels
    
    return model_inputs


def tokenize_for_regression(examples, tokenizer, max_length=512):
    """
    Tokenize inputs for regression models.
    
    Args:
        examples: Dict with 'text' key
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples with input_ids and attention_mask
    """
    return tokenizer(
        examples['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=max_length
    )


def make_tokenized_seq2seq_datasets(
    model_id: str,
    train_dict: dict,
    val_dict: dict,
    test_dict: dict,
    max_input_length: int = 512,
    max_target_length: int = 128,
):
    """
    Create tokenized HuggingFace Datasets for a specific seq2seq model.
    
    Tokenization is performed with the tokenizer that corresponds to the given model_id.
    This prevents tokenizer/model mismatches when training multiple models in one notebook.
    
    Args:
        model_id: HuggingFace model identifier (e.g., 'google/flan-t5-base', 'ai4bharat/IndicBARTSS')
        train_dict: Dict with 'input' and 'target' for the training split
        val_dict: Dict with 'input' and 'target' for the validation split
        test_dict: Dict with 'input' and 'target' for the test split
        max_input_length: Max number of tokens for inputs
        max_target_length: Max number of tokens for targets
    
    Returns:
        (train_dataset, val_dataset, test_dataset) tuple of tokenized Datasets
    """
    # Lazy imports to avoid hard dependency at import time
    from transformers import AutoTokenizer
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def _map_fn(batch):
        return tokenize_seq2seq(
            batch,
            tokenizer=tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
        )

    train_dataset = Dataset.from_dict(train_dict).map(
        _map_fn, batched=True, remove_columns=['input', 'target']
    )
    val_dataset = Dataset.from_dict(val_dict).map(
        _map_fn, batched=True, remove_columns=['input', 'target']
    )
    test_dataset = Dataset.from_dict(test_dict).map(
        _map_fn, batched=True, remove_columns=['input', 'target']
    )

    return train_dataset, val_dataset, test_dataset

def prepare_qa_data(df, question="How many people were killed?"):
    """
    Prepare data for question answering models.
    
    This function searches for the answer (death count number) in the text
    and creates proper span annotations for QA training.
    
    Args:
        df: DataFrame with 'incident_summary' and 'total_fatalities' columns
        question: Question to ask (default: "How many people were killed?")
        
    Returns:
        dict with 'question', 'context', and 'answers' keys where answers contains
        'text' and 'answer_start' lists for each example
    """
    questions = []
    contexts = []
    answers = []
    
    for _, row in df.iterrows():
        context = row['incident_summary']
        count = row['total_fatalities']
        answer_text = str(count)
        
        # Try to find the number in the context
        # We'll look for the exact number or number with common surrounding words
        answer_start = -1
        
        # Pattern 1: Exact match of the number
        pattern_exact = r'\b' + re.escape(answer_text) + r'\b'
        match = re.search(pattern_exact, context)
        if match:
            answer_start = match.start()
        else:
            # Pattern 2: Look for number with death/killed/fatality context
            # This helps find numbers like "5 people were killed"
            patterns = [
                r'\b(\d+)\s+(?:people\s+)?(?:were\s+)?(?:killed|dead|died)',
                r'(?:killed|killing)\s+(\d+)',
                r'(?:death|deaths)\s+(?:of\s+)?(\d+)',
                r'(\d+)\s+(?:fatalities|casualties)',
            ]
            for pattern in patterns:
                matches = list(re.finditer(pattern, context, re.IGNORECASE))
                if matches:
                    # Use first match
                    answer_start = matches[0].start(1)
                    answer_text = matches[0].group(1)
                    break
        
        # If still not found, try to find any number in the text
        if answer_start == -1:
            numbers = list(re.finditer(r'\b\d+\b', context))
            if numbers:
                # Use first number found
                answer_start = numbers[0].start()
                answer_text = numbers[0].group()
        
        # If no number found in text, keep answer_start as -1 to mark as impossible answer
        # This will be handled properly in tokenize_qa() with impossible answer flag
        if answer_start == -1:
            answer_text = str(count)  # Keep the true count for reference, but mark as impossible
        
        questions.append(question)
        contexts.append(context)
        answers.append({
            'text': [answer_text],
            'answer_start': [answer_start]
        })
    
    return {
        'question': questions,
        'context': contexts,
        'answers': answers
    }


def tokenize_qa(examples, tokenizer, max_length=512, stride=128):
    """
    Tokenize inputs for QA models.
    
    Converts character-level answer positions to token-level positions.
    Handles long contexts with sliding window approach.
    
    Args:
        examples: Dict with 'question', 'context', and 'answers' keys
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        stride: Stride for sliding window on long contexts
        
    Returns:
        Tokenized examples with input_ids, attention_mask, start_positions, end_positions
    """
    # Tokenize questions and contexts
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation="only_second",  # Only truncate context, not question
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length'
    )
    
    # Map back to original examples since we may have overflow
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    
    # Initialize lists for start and end positions
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        # Get the corresponding example index
        sample_index = sample_mapping[i]
        answers = examples['answers'][sample_index]
        
        # If no answer or answer not in text, use -1 to mark as impossible answer
        # HuggingFace QA models handle -1 as "impossible answer" (no answer in context)
        if len(answers['answer_start']) == 0 or answers['answer_start'][0] == -1:
            start_positions.append(-1)
            end_positions.append(-1)
            continue
        
        # Get answer span
        start_char = answers['answer_start'][0]
        end_char = start_char + len(answers['text'][0])
        
        # Find token start position
        token_start_index = 0
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        token_start_index = token_start_index - 1
        
        # Find token end position
        token_end_index = len(offsets) - 1
        while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        token_end_index = token_end_index + 1
        
        # Check if answer is within the current chunk
        if token_start_index < 0 or token_end_index >= len(offsets):
            # Answer not in this chunk, mark as impossible answer
            start_positions.append(-1)
            end_positions.append(-1)
        else:
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)
    
    tokenized_examples['start_positions'] = start_positions
    tokenized_examples['end_positions'] = end_positions
    
    return tokenized_examples

