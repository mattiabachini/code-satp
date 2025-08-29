"""
Imbalance Handling Strategies for SATP Classification Tasks

This module provides various strategies to handle class imbalance in multi-label classification
tasks. It's designed to integrate seamlessly with the existing Hugging Face training pipeline.

Strategies included:
1. Focal Loss Implementation
2. Multi-task Learning Architecture  
3. Data Augmentation (Back-translation)
4. Error Analysis-Driven Label Refinement
5. Hierarchical Label Organization
6. LLM-based Synthetic Generation
"""

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONSERVATIVE CLASS WEIGHT REWEIGHTING
# =============================================================================

def compute_conservative_class_weights(df, label_cols, cap_ratio=3.0, sqrt_scaling=True, min_weight=1.0, max_weight=10.0):
    """
    Compute conservative class weights that prevent precision collapse while maintaining recall gains.
    
    This function replaces raw pos_weight (N/P) with capped and sqrt-scaled weights to balance
    the trade-off between recall improvement and precision preservation.
    
    Args:
        df: DataFrame containing the training data
        label_cols: List of label column names
        cap_ratio: Maximum ratio for capping weights (default: 3.0)
        sqrt_scaling: Whether to apply sqrt scaling to reduce extreme weights (default: True)
        min_weight: Minimum weight value (default: 1.0)
        max_weight: Maximum weight value (default: 10.0)
    
    Returns:
        torch.Tensor: Conservative class weights of shape (num_labels,)
    
    Example:
        # Raw weights might be [50.0, 25.0, 100.0] for very imbalanced classes
        # Conservative weights become [3.0, 3.0, 3.0] with cap_ratio=3.0
        # Or [7.07, 5.0, 10.0] with sqrt scaling and max_weight=10.0
    """
    # Calculate raw pos_weight (N/P)
    P = df[label_cols].sum(axis=0).values.astype(np.float32)
    N = len(df) - P
    P = np.where(P == 0, 1.0, P)  # Guard against division by zero
    raw_weights = N / P
    
    # Apply conservative capping
    if cap_ratio is not None:
        # Cap weights to prevent extreme values that cause precision collapse
        capped_weights = np.minimum(raw_weights, cap_ratio)
    else:
        capped_weights = raw_weights
    
    # Apply sqrt scaling to reduce extreme weights while preserving relative ordering
    if sqrt_scaling:
        # sqrt scaling reduces the impact of very large weights
        # This helps maintain recall gains while preventing precision collapse
        scaled_weights = np.sqrt(capped_weights)
    else:
        scaled_weights = capped_weights
    
    # Apply min/max bounds for numerical stability
    bounded_weights = np.clip(scaled_weights, min_weight, max_weight)
    
    # Convert to torch tensor
    conservative_weights = torch.tensor(bounded_weights, dtype=torch.float)
    
    return conservative_weights


def compute_adaptive_conservative_weights(df, label_cols, base_cap_ratio=3.0, precision_threshold=0.3, recall_threshold=0.7):
    """
    Compute adaptive conservative weights based on validation performance.
    
    This function dynamically adjusts the capping ratio based on whether
    the model is achieving the desired precision-recall balance.
    
    Args:
        df: DataFrame containing the training data
        label_cols: List of label column names
        base_cap_ratio: Base capping ratio (default: 3.0)
        precision_threshold: Minimum precision threshold (default: 0.3)
        recall_threshold: Minimum recall threshold (default: 0.7)
    
    Returns:
        torch.Tensor: Adaptive conservative class weights
    """
    # Start with base conservative weights
    base_weights = compute_conservative_class_weights(
        df, label_cols, 
        cap_ratio=base_cap_ratio, 
        sqrt_scaling=True
    )
    
    # For now, return base weights
    # In practice, this could be extended to use validation metrics
    # to dynamically adjust the capping ratio
    return base_weights


def compute_label_specific_conservative_weights(df, label_cols, label_configs=None):
    """
    Compute label-specific conservative weights with different strategies per label.
    
    Args:
        df: DataFrame containing the training data
        label_cols: List of label column names
        label_configs: Dict mapping label names to specific configs
                       Format: {label_name: {'cap_ratio': float, 'sqrt_scaling': bool, 'min_weight': float, 'max_weight': float}}
    
    Returns:
        torch.Tensor: Label-specific conservative class weights
    """
    if label_configs is None:
        # Default configuration for all labels
        return compute_conservative_class_weights(df, label_cols)
    
    # Calculate raw weights first
    P = df[label_cols].sum(axis=0).values.astype(np.float32)
    N = len(df) - P
    P = np.where(P == 0, 1.0, P)
    raw_weights = N / P
    
    conservative_weights = np.zeros_like(raw_weights)
    
    for i, label_name in enumerate(label_cols):
        if label_name in label_configs:
            config = label_configs[label_name]
            cap_ratio = config.get('cap_ratio', 3.0)
            sqrt_scaling = config.get('sqrt_scaling', True)
            min_weight = config.get('min_weight', 1.0)
            max_weight = config.get('max_weight', 10.0)
            
            # Apply label-specific conservative strategy
            weight = raw_weights[i]
            if cap_ratio is not None:
                weight = min(weight, cap_ratio)
            if sqrt_scaling:
                weight = np.sqrt(weight)
            weight = np.clip(weight, min_weight, max_weight)
            conservative_weights[i] = weight
        else:
            # Use default conservative strategy
            weight = raw_weights[i]
            weight = min(weight, 3.0)  # Default cap
            weight = np.sqrt(weight)   # Default sqrt scaling
            weight = np.clip(weight, 1.0, 10.0)  # Default bounds
            conservative_weights[i] = weight
    
    return torch.tensor(conservative_weights, dtype=torch.float)


# =============================================================================
# TIER 1: IMMEDIATE IMPLEMENTATION (HIGHEST ROI)
# =============================================================================

class FocalLoss(nn.Module):
    """
    Numerically stable multi-label Focal Loss with per-label alpha.

    Uses log-sigmoid form to avoid explicit sigmoid and log instabilities.
    alpha_pos is a vector of shape [num_labels] with values typically in [0.25, 0.75].
    """

    def __init__(self, alpha_pos, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        # Register buffer so it follows model/device without being a parameter
        alpha_tensor = torch.as_tensor(alpha_pos, dtype=torch.float)
        self.register_buffer("alpha_pos", alpha_tensor)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs of shape (batch_size, num_labels)
            targets: Ground truth binary labels of shape (batch_size, num_labels)
        """
        # Stable log-sigmoid terms
        logp = F.logsigmoid(logits)
        log1mp = F.logsigmoid(-logits)
        # Select per-element log-prob of the true class
        logpt = torch.where(targets == 1, logp, log1mp)
        pt = torch.exp(logpt)
        # Per-element alpha_t using per-label alpha_pos
        alpha_t = torch.where(targets == 1, self.alpha_pos, 1 - self.alpha_pos)
        loss = -(alpha_t * (1 - pt).pow(self.gamma) * logpt)

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class MultiTaskModel(nn.Module):
    """
    Multi-task learning model that shares encoder across related tasks.
    
    This model uses a shared BERT encoder with separate classification heads
    for each task (perpetrator, action_type, target_type).
    """
    
    def __init__(self, model_name, num_labels_dict, shared_layers=6):
        """
        Args:
            model_name: Hugging Face model name
            num_labels_dict: Dict with task names as keys and number of labels as values
            shared_layers: Number of shared BERT layers (default: 6)
        """
        super(MultiTaskModel, self).__init__()
        
        from transformers import AutoModel, AutoTokenizer
        
        # Load base model
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze some layers for shared representation
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze first N layers
        for layer in self.bert.encoder.layer[:shared_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Task-specific classification heads
        self.classifiers = nn.ModuleDict()
        hidden_size = self.bert.config.hidden_size
        
        for task_name, num_labels in num_labels_dict.items():
            self.classifiers[task_name] = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_labels)
            )
    
    def forward(self, input_ids, attention_mask, task_name):
        """
        Forward pass for a specific task.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task_name: Name of the task to predict
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifiers[task_name](pooled_output)
        return logits

class BackTranslationAugmentation:
    """
    Data augmentation using back-translation for rare classes.
    
    Uses Google Translate API to translate text to another language and back
    to create synthetic examples for underrepresented classes.
    """
    
    def __init__(self, target_languages=['hi', 'ur', 'bn']):  # Hindi, Urdu, Bengali
        self.target_languages = target_languages
        try:
            from deep_translator import GoogleTranslator
            self.translator_class = GoogleTranslator
            self.available = True
            print("✅ Using deep-translator for back-translation")
        except ImportError:
            try:
                from googletrans import Translator
                self.translator = Translator()
                self.available = True
                self.use_googletrans = True
                print("✅ Using googletrans for back-translation")
            except ImportError:
                print("⚠️ Warning: No translation library available for back-translation.")
                print("🔧 Install with: !pip install deep-translator")
                self.available = False
    
    def augment_text(self, text, num_augmentations=1, seed=None):
        """
        Augment a single text using back-translation.
        
        Args:
            text: Original text
            num_augmentations: Number of augmented versions to create
        """
        if not self.available:
            return []
        
        augmented_texts = []
        
        if seed is not None:
            # Local seeding for reproducible language selection/order
            np.random.seed(seed)
        for _ in range(num_augmentations):
            try:
                # Choose random target language
                target_lang = np.random.choice(self.target_languages)
                
                if hasattr(self, 'translator_class'):  # Using deep-translator
                    # Translate to target language
                    translator_to = self.translator_class(source='en', target=target_lang)
                    translated = translator_to.translate(text)
                    
                    # Translate back to English
                    translator_back = self.translator_class(source=target_lang, target='en')
                    back_translated = translator_back.translate(translated)
                    
                else:  # Using googletrans (fallback)
                    # Translate to target language
                    translated = self.translator.translate(text, dest=target_lang)
                    
                    # Translate back to English
                    back_translated = self.translator.translate(translated.text, dest='en')
                    back_translated = back_translated.text
                
                augmented_texts.append(back_translated)
                
            except Exception as e:
                print(f"Translation error: {e}")
                continue
        
        return augmented_texts
    
    def augment_rare_classes(self, df, label_columns, min_samples=50, max_new_per_label=500, max_synth_to_real_ratio=1.0, seed=42, prefer_single_label=True):
        """
        Augment rare classes to have at least min_samples examples.
        
        Args:
            df: DataFrame with 'incident_summary' and label columns
            label_columns: List of label column names
            min_samples: Minimum number of samples per class
            seed: Random seed for reproducible selection of examples
            prefer_single_label: Prefer positives with only this label to reduce drift
        """
        if not self.available:
            return df
        
        augmented_data = []
        
        # Reproducibility seeds for selection and any torch ops used hereafter
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for col in label_columns:
            class_counts = df[col].value_counts()
            rare_classes = class_counts[class_counts < min_samples].index
            
            for rare_class in rare_classes:
                if rare_class == 0:  # Skip negative class
                    continue
                
                # Get positives for this label
                pos_df = df[df[col] == 1]
                # Prefer single-label positives to minimize label drift
                if prefer_single_label:
                    single_label_pos = pos_df[pos_df[label_columns].sum(axis=1) == 1]
                    pool = single_label_pos if len(single_label_pos) > 0 else pos_df
                else:
                    pool = pos_df
                # Reproducible randomization of example order
                rare_examples = pool.sample(frac=1.0, random_state=seed)
                
                # Calculate how many augmentations needed (capped)
                needed = max(min_samples - len(pos_df), 0)
                cap_by_ratio = int(len(pos_df) * max_synth_to_real_ratio)
                to_add = min(needed, max_new_per_label, cap_by_ratio)
                
                if to_add <= 0:
                    continue
                
                print(f"Augmenting label '{col}': {len(rare_examples)} -> target {min_samples} (adding up to {to_add})")
                
                # Augment each example
                added_for_label = 0
                for _, row in rare_examples.iterrows():
                    if added_for_label >= to_add:
                        break
                    
                    augmented_texts = self.augment_text(row['incident_summary'], num_augmentations=1, seed=seed)
                    
                    for aug_text in augmented_texts:
                        if added_for_label >= to_add:
                            break
                        
                        # Create new row with augmented text
                        new_row = row.copy()
                        new_row['incident_summary'] = aug_text
                        augmented_data.append(new_row)
                        added_for_label += 1
                if added_for_label > 0:
                    print(f"Added {added_for_label} synthetic rows for label '{col}'")
        
        # Combine original and augmented data
        if augmented_data:
            augmented_df = pd.DataFrame(augmented_data)
            return pd.concat([df, augmented_df], ignore_index=True)
        
        return df

# =============================================================================
# T5 PARAPHRASE AUGMENTATION
# =============================================================================

class T5ParaphraseAugmentation:
    """
    Paraphrase-based augmentation using a T5/FLAN-T5 model.
    
    Generates label-preserving paraphrases for positives of rare classes.
    """
    
    def __init__(self, model_name="google/flan-t5-large", device=None):
        self.model_name = model_name
        self.device = device  # e.g., "cuda" or "cpu"; if None, auto-select by HF pipeline
        self._pipeline = None
        self._tokenizer = None
        self.available = True
        # Semantic similarity embedder (sentence-transformers)
        self.embedder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._embedder = None
        self.embedder_available = True
    
    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Create pipeline with better GPU efficiency settings
            pipeline_kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "batch_size": 4,  # Process multiple texts at once for GPU efficiency
                "device_map": "auto" if self.device is None else self.device
            }
            
            # With transformers >= 4.46, batch_size should work properly for T5 models
            self._pipeline = pipeline("text2text-generation", **pipeline_kwargs)
            self._tokenizer = tokenizer
            
            # Check and report model capabilities
            self._check_model_capabilities()
        except Exception as e:
            print(f"⚠️ T5 paraphraser unavailable: {e}")
            self.available = False
    
    def _check_model_capabilities(self):
        """Check what generation features this model supports."""
        if not hasattr(self, '_pipeline') or self._pipeline is None:
            return
        
        try:
            model = self._pipeline.model
            generation_config = getattr(model, 'generation_config', None)
            
            print(f"🔍 Model: {self.model_name}")
            print(f"   - Generation config available: {generation_config is not None}")
            
            if generation_config:
                # Check for seed parameter support
                seed_supported = hasattr(generation_config, 'seed') or 'seed' in generation_config.__dict__
                print(f"   - Seed parameter supported: {seed_supported}")
                
                # Check for other useful parameters
                if hasattr(generation_config, 'do_sample'):
                    print(f"   - do_sample supported: ✅")
                if hasattr(generation_config, 'temperature'):
                    print(f"   - temperature supported: ✅")
                if hasattr(generation_config, 'top_p'):
                    print(f"   - top_p supported: ✅")
            
            print(f"   - Using transformers version: {self._get_transformers_version()}")
            
        except Exception as e:
            print(f"⚠️ Could not check model capabilities: {e}")
    
    def _get_transformers_version(self):
        """Get the installed transformers version."""
        try:
            import transformers
            return transformers.__version__
        except:
            return "Unknown"
    
    def _truncate_text(self, text, max_input_tokens=480):
        """
        Truncate input text to avoid exceeding max length once the prompt prefix is added.
        """
        if self._tokenizer is None:
            return text
        try:
            ids = self._tokenizer.encode(text, truncation=True, max_length=max_input_tokens, add_special_tokens=False)
            return self._tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            return text[:2000]
    
    def paraphrase(self, text, num_return_sequences=2, temperature=0.7, top_p=0.9, max_new_tokens=64, seed=None):
        """
        Generate paraphrases for a single text.
        """
        if not self.available:
            return []
        self._ensure_pipeline()
        if self._pipeline is None:
            return []
        safe_text = self._truncate_text(text, max_input_tokens=480)
        prompt = (
            "Paraphrase the following incident summary without changing meaning, entities, actors, or event type. "
            "Use a neutral, report-like tone and do not add or remove facts.\n\n"
            f"Text: {safe_text}\n\nParaphrase:"
        )
        try:
            # Handle seeding for reproducible generation
            gen_kwargs = {}
            if seed is not None:
                # Set torch seed for reproducibility (this always works)
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
                
                # Check if this specific model supports the seed parameter
                # FLAN-T5 models often don't support it even in newer transformers versions
                if hasattr(self._pipeline.model, 'generation_config'):
                    generation_config = self._pipeline.model.generation_config
                    # Only add seed if the model's generation config supports it
                    if hasattr(generation_config, 'seed') or 'seed' in generation_config.__dict__:
                        gen_kwargs["seed"] = int(seed)
                        # Don't print this every time - it's already shown during model initialization
                    else:
                        # Don't print this every time - it's already shown during model initialization
                        pass
                else:
                    # Don't print this every time - it's already shown during model initialization
                    pass
            
            # Generate with the model - will use seed parameter if available
            outputs = self._pipeline(
                prompt,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )
            return [o.get("generated_text", "").strip() for o in outputs if o.get("generated_text")]
        except Exception as e:
            print(f"T5 generation error: {e}")
            return []
    
    def _ensure_embedder(self):
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedder_model_name)
        except Exception as e:
            print(f"⚠️ Similarity embedder unavailable: {e}")
            self.embedder_available = False

    @staticmethod
    def _cosine_sim_matrix(a, b):
        # a: (1, d) or (n, d); b: (m, d) -> sims shape (n, m)
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return np.matmul(a_norm, b_norm.T)

    def augment_rare_classes(
        self,
        df,
        label_columns,
        min_samples=50,
        max_new_per_label=500,
        max_synth_to_real_ratio=1.0,
        seed=42,
        prefer_single_label=True,
        per_seed=1,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=64,
        dedup=True,
        min_similarity=0.85,
        max_similarity=0.98,
        embedder_model_name=None
    ):
        """
        Augment rare classes by adding paraphrases for positive examples.
        """
        if not self.available:
            return df
        self._ensure_pipeline()
        if self._pipeline is None:
            return df
        # Prepare embedder for similarity filtering
        if embedder_model_name:
            self.embedder_model_name = embedder_model_name
        self._ensure_embedder()
        
        augmented_rows = []
        existing_texts = set(df["incident_summary"].astype(str).tolist()) if dedup else set()
        
        # Seeding for deterministic sampling and generation
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for col in label_columns:
            pos_df = df[df[col] == 1]
            cur = len(pos_df)
            needed = max(min_samples - cur, 0)
            cap_by_ratio = int(cur * max_synth_to_real_ratio)
            to_add = min(needed, max_new_per_label, cap_by_ratio)
            if to_add <= 0:
                continue
            
            # Prefer single-label positives
            if prefer_single_label:
                single_label_pos = pos_df[pos_df[label_columns].sum(axis=1) == 1]
                pool = single_label_pos if len(single_label_pos) > 0 else pos_df
            else:
                pool = pos_df
            
            # Reproducible randomization
            seeds_df = pool.sample(frac=1.0, random_state=seed)
            
            print(f"T5 paraphrase augment '{col}': {cur} -> target {min_samples} (adding up to {to_add})")
            added_for_label = 0
            
            for _, row in seeds_df.iterrows():
                if added_for_label >= to_add:
                    break
                text = str(row["incident_summary"]) if pd.notna(row["incident_summary"]) else ""
                if not text:
                    continue
                candidates = self.paraphrase(
                    text,
                    num_return_sequences=max(1, per_seed),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                )
                # Similarity filter
                filtered_candidates = candidates
                if self.embedder_available and self._embedder is not None and len(candidates) > 0:
                    try:
                        seed_vec = self._embedder.encode([text], convert_to_numpy=True)
                        cand_vecs = self._embedder.encode(candidates, convert_to_numpy=True)
                        sims = self._cosine_sim_matrix(seed_vec, cand_vecs)[0]
                        filtered_candidates = [c for c, s in zip(candidates, sims) if (min_similarity <= float(s) <= max_similarity)]
                    except Exception as e:
                        print(f"Similarity filtering failed: {e}")
                        # fall back to unfiltered candidates
                for cand in candidates:
                    if added_for_label >= to_add:
                        break
                    # if similarity filter applied, skip those rejected
                    if self.embedder_available and self._embedder is not None and len(filtered_candidates) > 0 and cand not in filtered_candidates:
                        continue
                    if dedup and cand in existing_texts:
                        continue
                    new_row = row.copy()
                    new_row["incident_summary"] = cand
                    new_row["is_synthetic"] = 1
                    new_row["synthetic_source"] = "t5"
                    augmented_rows.append(new_row)
                    added_for_label += 1
                    if dedup:
                        existing_texts.add(cand)
            if added_for_label > 0:
                print(f"Added {added_for_label} T5-paraphrased rows for label '{col}'")
        
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            return pd.concat([df, augmented_df], ignore_index=True)
        return df

# (Removed SMOTE implementation)

class ErrorAnalysisRefinement:
    """
    Error analysis and label refinement for confused categories.
    """
    
    def __init__(self):
        self.confusion_patterns = {}
        self.refinement_suggestions = {}
    
    def analyze_confusion_matrix(self, y_true, y_pred, label_names):
        """
        Analyze confusion patterns to identify problematic label pairs.
        """
        from sklearn.metrics import confusion_matrix
        
        for i, label_name in enumerate(label_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            
            if cm.shape == (2, 2):
                # Calculate metrics
                tn, fp, fn, tp = cm.ravel()
                
                # Identify problematic patterns
                if fp > 0 and fn > 0:
                    self.confusion_patterns[label_name] = {
                        'false_positives': fp,
                        'false_negatives': fn,
                        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
                    }
    
    def suggest_refinements(self, df, label_columns, threshold=0.3):
        """
        Suggest label refinements based on error analysis.
        """
        suggestions = {}
        
        for label in self.confusion_patterns:
            pattern = self.confusion_patterns[label]
            
            if pattern['precision'] < threshold or pattern['recall'] < threshold:
                # Find examples that might need relabeling
                label_examples = df[df[label] == 1]['incident_summary'].head(10)
                
                suggestions[label] = {
                    'precision': pattern['precision'],
                    'recall': pattern['recall'],
                    'examples': label_examples.tolist(),
                    'suggestion': f"Consider refining definition or adding examples for {label}"
                }
        
        return suggestions
    
    def generate_refinement_report(self, suggestions):
        """
        Generate a detailed report for label refinement.
        """
        report = "=== LABEL REFINEMENT SUGGESTIONS ===\n\n"
        
        for label, info in suggestions.items():
            report += f"Label: {label}\n"
            report += f"Precision: {info['precision']:.3f}\n"
            report += f"Recall: {info['recall']:.3f}\n"
            report += f"Suggestion: {info['suggestion']}\n"
            report += f"Example texts:\n"
            
            for i, example in enumerate(info['examples'][:3]):
                report += f"  {i+1}. {example[:100]}...\n"
            
            report += "\n"
        
        return report

# =============================================================================
# TIER 3: LOWER PRIORITY (BUT HIGH POTENTIAL)
# =============================================================================

class HierarchicalLabels:
    """
    Hierarchical label organization for better rare class handling.
    """
    
    def __init__(self):
        self.hierarchy = {}
        self.parent_child_mapping = {}
    
    def create_hierarchy(self, label_columns):
        """
        Create a hierarchical structure for labels.
        """
        # Example hierarchy for target types
        self.hierarchy = {
            'human_targets': ['civilians', 'government_officials', 'security'],
            'property_targets': ['private_property', 'government_infrastructure'],
            'organizational_targets': ['mining_company', 'ngos'],
            'armed_groups': ['maoist', 'non_maoist_armed_group'],
            'other': ['no_target']
        }
        
        # Create parent-child mapping
        for parent, children in self.hierarchy.items():
            for child in children:
                if child in label_columns:
                    self.parent_child_mapping[child] = parent
    
    def get_parent_labels(self, df, label_columns):
        """
        Create parent-level labels based on hierarchy.
        """
        parent_labels = {}
        
        for parent, children in self.hierarchy.items():
            # Check if any child labels are present
            present_children = [child for child in children if child in label_columns]
            
            if present_children:
                # Create parent label (1 if any child is 1)
                parent_labels[parent] = df[present_children].max(axis=1)
        
        return parent_labels

class LLMSyntheticGeneration:
    """
    LLM-based synthetic generation for rare classes.
    """
    
    def __init__(
        self,
        provider="openai",
        model=None,
        api_key=None,
        temperature=0.5,
        max_tokens=400,
        rate_limit_per_min=30
    ):
        self.provider = provider
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rate_limit_s = max(1.0 / max(rate_limit_per_min, 1), 0.5)
        self.available = api_key is not None

        self.model = model or (
            "gpt-4o-mini" if provider == "openai" else "claude-3-haiku-20240307"
        )

        self._client = None
        if not self.available:
            print("⚠️ LLM generation unavailable: missing API key.")
            return

        try:
            if provider == "openai":
                try:
                    from openai import OpenAI
                    self._client = OpenAI(api_key=api_key)
                    self._is_new_openai = True
                except Exception:
                    import openai
                    openai.api_key = api_key
                    self._client = openai
                    self._is_new_openai = False
            elif provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            else:
                print(f"⚠️ Unknown provider '{provider}'.")
                self.available = False
        except Exception as e:
            print(f"LLM client init failed: {e}")
            self.available = False

    def _build_prompt(self, class_name, class_definition="", few_shots=None, num_examples=5):
        """
        Build a concise, constrained prompt with few-shot in-domain examples.
        """
        few_shots = few_shots or []
        header = (
            f"You are generating incident summaries for the SATP dataset.\n"
            f"Target label: {class_name}\n"
            f"Definition/guidance: {class_definition.strip()}\n\n"
            "Constraints:\n"
            "- 1–3 sentences each; concise and factual; no dates/identifiers from real cases.\n"
            "- Use neutral, report-like tone consistent with SATP style.\n"
            "- Do not mention the label by name. Avoid policy claims or speculation.\n"
            "- Avoid personally identifiable info; no exact copying of examples.\n"
            f"Generate {num_examples} distinct examples.\n"
            "Return each on a new line starting with '- '."
        )
        shots = ""
        if few_shots:
            ex_txt = "\n".join([f"- {t}" for t in few_shots[:5]])
            shots = f"\n\nExamples (style and content archetypes):\n{ex_txt}\n"

        return header + shots

    def _call_openai(self, prompt):
        import time
        if self._is_new_openai:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            out = resp.choices[0].message.content
        else:
            out = self._client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ).choices[0].message["content"]
        time.sleep(self.rate_limit_s)
        return out

    def _call_anthropic(self, prompt):
        import time
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = []
        for block in getattr(msg, "content", []) or []:
            txt = getattr(block, "text", None)
            if txt:
                parts.append(txt)
        time.sleep(self.rate_limit_s)
        return "\n".join(parts) if parts else ""

    def _generate(self, prompt):
        if not self.available:
            return ""
        try:
            if self.provider == "openai":
                return self._call_openai(prompt)
            if self.provider == "anthropic":
                return self._call_anthropic(prompt)
        except Exception as e:
            print(f"LLM generation failed: {e}")
        return ""

    @staticmethod
    def _parse_bullets(text):
        lines = [l.strip() for l in text.splitlines()]
        out = []
        for l in lines:
            if not l:
                continue
            if l.startswith("- "):
                out.append(l[2:].strip())
            else:
                out.append(l.strip("• ").strip())
        out = [t for t in out if 20 <= len(t) <= 600]
        return out

    def generate_synthetic_examples(self, class_name, num_examples=10, class_definition="", few_shots=None):
        """
        Generate class-conditional examples with few-shot guidance.
        """
        prompt = self._build_prompt(
            class_name, class_definition=class_definition, few_shots=few_shots, num_examples=num_examples
        )
        raw = self._generate(prompt)
        return self._parse_bullets(raw)

    def augment_rare_classes_with_llm(
        self,
        df,
        label_columns,
        label_to_definition=None,
        min_samples=50,
        max_new_per_label=500,
        max_synth_to_real_ratio=1.0,
        few_shots_per_label=3,
        dedup=True
    ):
        """
        For each label column (binary 0/1), top up rare positives with LLM-generated texts.

        Adds a column 'is_synthetic'=1 for new rows.
        """
        if not self.available:
            return df

        label_to_definition = label_to_definition or {}
        augmented_rows = []
        for col in label_columns:
            pos_df = df[df[col] == 1]
            cur = len(pos_df)
            needed = max(min_samples - cur, 0)
            cap_by_ratio = int(cur * max_synth_to_real_ratio)
            to_add = min(needed, max_new_per_label, cap_by_ratio)
            if to_add <= 0:
                continue

            seeds = pos_df["incident_summary"].dropna().astype(str).tolist()
            seeds = seeds[:few_shots_per_label] if len(seeds) >= few_shots_per_label else seeds

            definition = label_to_definition.get(col, "")

            print(f"LLM augment '{col}': {cur} -> target {min_samples} (adding up to {to_add})")
            batch = 10
            generated = []
            remaining = to_add
            while remaining > 0:
                n = min(batch, remaining)
                texts = self.generate_synthetic_examples(
                    class_name=col,
                    num_examples=n,
                    class_definition=definition,
                    few_shots=seeds
                )
                generated.extend(texts)
                remaining -= len(texts)
                if len(texts) == 0:
                    break

            if dedup:
                existing = set(df["incident_summary"].astype(str).tolist())
                generated = [t for t in generated if t not in existing]
                generated = list(dict.fromkeys(generated))

            for t in generated[:to_add]:
                new_row = pos_df.iloc[0].copy() if cur > 0 else pd.Series({c: 0 for c in df.columns})
                new_row["incident_summary"] = t
                for l in label_columns:
                    new_row[l] = 1 if l == col else new_row.get(l, 0)
                new_row["is_synthetic"] = 1
                augmented_rows.append(new_row)

            if len(generated[:to_add]) > 0:
                print(f"Added {len(generated[:to_add])} LLM-generated rows for '{col}'")

        if augmented_rows:
            aug_df = pd.DataFrame(augmented_rows)
            return pd.concat([df, aug_df], ignore_index=True)

        return df

# =============================================================================
# CALIBRATION (TEMPERATURE SCALING)
# =============================================================================

class TemperatureScaling(nn.Module):
    """
    Temperature scaling calibration for neural network logits.
    
    This module learns a single temperature parameter to calibrate
    the confidence of predictions, improving reliability of probability
    estimates before threshold tuning.
    
    Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
    """
    
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        # Single temperature parameter for all classes
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model outputs (batch_size, num_classes)
            
        Returns:
            Calibrated logits (batch_size, num_classes)
        """
        return logits / self.temperature
    
    def fit(self, logits, labels, max_iter=50, lr=0.01, verbose=False):
        """
        Fit temperature parameter on validation set to minimize NLL.
        
        Args:
            logits: Validation logits (n_samples, n_classes)
            labels: Validation labels (n_samples, n_classes) - binary for multi-label
            max_iter: Maximum optimization iterations
            lr: Learning rate for optimization
            verbose: Print optimization progress
        """
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, dtype=torch.float32)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.float32)
            
        # Move to same device as logits
        device = logits.device
        self.to(device)
        labels = labels.to(device)
        
        # Optimizer for temperature parameter
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            # Apply temperature scaling
            scaled_logits = self.forward(logits)
            # Multi-label binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
            loss.backward()
            return loss
        
        if verbose:
            print(f"Initial temperature: {self.temperature.item():.4f}")
        
        # Optimize temperature
        optimizer.step(eval_loss)
        
        if verbose:
            print(f"Optimized temperature: {self.temperature.item():.4f}")
        
        return self.temperature.item()
    
    def calibrate_probs(self, logits):
        """
        Get calibrated probabilities from logits.
        
        Args:
            logits: Raw model outputs
            
        Returns:
            Calibrated probabilities
        """
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits, dtype=torch.float32)
        
        # Apply temperature scaling and sigmoid
        with torch.no_grad():
            calibrated_logits = self.forward(logits)
            calibrated_probs = torch.sigmoid(calibrated_logits)
        
        return calibrated_probs.numpy() if isinstance(calibrated_probs, torch.Tensor) else calibrated_probs

def calibrate_and_tune_thresholds(
    trainer, 
    val_dataset, 
    test_dataset, 
    label_cols, 
    objective="micro", 
    max_temp_iter=50, 
    verbose=True
):
    """
    Perform temperature scaling calibration followed by threshold tuning.
    
    This implements the "Calibrate, then threshold" strategy for better
    precision/recall balance on rare labels.
    
    Args:
        trainer: Trained Hugging Face trainer
        val_dataset: Validation dataset for calibration and threshold tuning
        test_dataset: Test dataset for final evaluation
        label_cols: List of label column names
        objective: "micro" or "macro" F1 for threshold tuning
        max_temp_iter: Maximum iterations for temperature optimization
        verbose: Print calibration and tuning progress
        
    Returns:
        Dictionary with calibrated test metrics and predictions
    """
    from sklearn.metrics import classification_report, f1_score, hamming_loss, accuracy_score
    
    # Get validation predictions for calibration
    if verbose:
        print("Getting validation predictions for calibration...")
    val_out = trainer.predict(val_dataset)
    val_logits = val_out.predictions
    val_labels = val_out.label_ids
    
    # Fit temperature scaling on validation set
    if verbose:
        print("Fitting temperature scaling...")
    temp_scaler = TemperatureScaling()
    final_temp = temp_scaler.fit(val_logits, val_labels, max_iter=max_temp_iter, verbose=verbose)
    
    # Get calibrated validation probabilities
    val_probs_calibrated = temp_scaler.calibrate_probs(val_logits)
    
    # Tune thresholds on calibrated validation probabilities
    if verbose:
        print("Tuning thresholds on calibrated probabilities...")
    
    if objective == "micro":
        # Global threshold for micro-F1
        from sklearn.metrics import f1_score
        grid = np.linspace(0.05, 0.95, 19)
        best_thresh = 0.5
        best_f1 = 0.0
        
        for thresh in grid:
            preds = (val_probs_calibrated >= thresh).astype(int)
            f1 = f1_score(val_labels, preds, average="micro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        thresholds = np.full(len(label_cols), best_thresh)
        if verbose:
            print(f"Optimal global threshold: {best_thresh:.3f} (micro-F1: {best_f1:.4f})")
    
    else:  # macro or per-label
        # Per-label thresholds for macro-F1
        # Define the function inline to avoid circular imports
        def choose_thresholds_per_label_local(val_probs, val_true, grid=np.linspace(0.0, 1.0, 101), beta=1.0):
            """Select a threshold per label independently by maximizing F-beta."""
            from sklearn.metrics import fbeta_score
            n_labels = val_true.shape[1]
            th = np.full(n_labels, 0.5, dtype=float)
            for j in range(n_labels):
                y = val_true[:, j]
                p = val_probs[:, j]
                best_t, best_s = 0.5, -1.0
                for t in grid:
                    s = fbeta_score(y, (p >= t).astype(int), beta=beta, zero_division=0)
                    if s > best_s:
                        best_s, best_t = s, t
                th[j] = best_t
            return th
        thresholds = choose_thresholds_per_label_local(val_probs_calibrated, val_labels.astype(int))
        if verbose:
            print(f"Per-label thresholds: {thresholds}")
    
    # Get test predictions and apply calibration + thresholds
    if verbose:
        print("Applying calibration and thresholds to test set...")
    test_out = trainer.predict(test_dataset)
    test_logits = test_out.predictions
    test_labels = test_out.label_ids
    
    # Calibrate test probabilities
    test_probs_calibrated = temp_scaler.calibrate_probs(test_logits)
    
    # Apply tuned thresholds
    test_preds = (test_probs_calibrated >= thresholds[None, :]).astype(int)
    
    # Compute final metrics
    test_true = test_labels.astype(int)
    
    # Print comparison: uncalibrated vs calibrated+tuned
    if verbose:
        print("\n=== COMPARISON: Uncalibrated vs Calibrated+Tuned ===")
        
        # Uncalibrated with 0.5 threshold
        test_probs_uncal = torch.sigmoid(torch.tensor(test_logits)).numpy()
        test_preds_uncal = (test_probs_uncal > 0.5).astype(int)
        
        uncal_micro_f1 = f1_score(test_true, test_preds_uncal, average="micro", zero_division=0)
        uncal_macro_f1 = f1_score(test_true, test_preds_uncal, average="macro", zero_division=0)
        
        cal_micro_f1 = f1_score(test_true, test_preds, average="micro", zero_division=0)
        cal_macro_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
        
        print(f"Uncalibrated (thresh=0.5): Micro-F1={uncal_micro_f1:.4f}, Macro-F1={uncal_macro_f1:.4f}")
        print(f"Calibrated+Tuned: Micro-F1={cal_micro_f1:.4f}, Macro-F1={cal_macro_f1:.4f}")
        print(f"Improvement: Micro-F1={cal_micro_f1-uncal_micro_f1:+.4f}, Macro-F1={cal_macro_f1-uncal_macro_f1:+.4f}")
    
    # Final classification report
    if verbose:
        print("\n=== Classification Report: Calibrated + Threshold Tuned ===")
        print(classification_report(test_true, test_preds, target_names=label_cols, zero_division=0))
    
    # Compute all metrics
    report = classification_report(test_true, test_preds, target_names=label_cols, zero_division=0, output_dict=True)
    micro_f1 = f1_score(test_true, test_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    
    metrics = {
        "hamming_loss": hamming_loss(test_true, test_preds),
        "subset_accuracy": accuracy_score(test_true, test_preds),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "temperature": final_temp,
        "thresholds": thresholds.tolist(),
        "objective": objective
    }
    metrics.update(report)
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame()
    for i, col in enumerate(label_cols):
        pred_df[f"true_{col}"] = test_true[:, i]
        pred_df[f"pred_{col}"] = test_preds[:, i]
        pred_df[f"prob_calibrated_{col}"] = test_probs_calibrated[:, i]
        pred_df[f"prob_uncalibrated_{col}"] = test_probs_uncal[:, i]
    
    return {
        "metrics": metrics,
        "predictions": pred_df,
        "temperature": final_temp,
        "thresholds": thresholds,
        "calibrated_probs": test_probs_calibrated
    }

# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def integrate_focal_loss(trainer, alpha=1, gamma=2):
    """
    Backward-compatible wrapper retained for legacy calls.
    Defaults to scalar alpha converted to a vector for compatibility.
    """
    # Create a trivial vector alpha_pos later inside the closure after first batch to infer K
    scalar_alpha = float(alpha)

    # Override the compute_loss method
    def compute_loss_with_focal(model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        # Build a per-label alpha_pos vector on first use
        num_labels = logits.shape[-1]
        alpha_pos_vec = torch.full((num_labels,), scalar_alpha, dtype=torch.float, device=logits.device)
        loss_fn = FocalLoss(alpha_pos=alpha_pos_vec, gamma=gamma)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    trainer.compute_loss = compute_loss_with_focal
    return trainer

def compute_alpha_pos(y_train, clamp_min=0.25, clamp_max=0.75):
    """
    Compute per-label alpha_pos from training labels y_train (tensor or array) of shape [N, K].
    alpha_pos_k is proportional to sqrt(median(pi) / pi_k), then clamped to [clamp_min, clamp_max].
    """
    if not torch.is_tensor(y_train):
        y_train = torch.as_tensor(y_train)
    y_train = y_train.float()
    pi = y_train.mean(dim=0).clamp_(min=1e-6, max=1 - 1e-6)
    median_pi = pi.median()
    # Weight up rare labels relative to the median prevalence
    w = torch.sqrt(median_pi / pi)
    alpha_pos = torch.clamp(w, min=float(clamp_min), max=float(clamp_max))
    return alpha_pos

def integrate_focal_loss_advanced(
    trainer,
    y_train=None,
    alpha_pos=None,
    gamma=2.0,
    use_ultra_rare_gamma=False,
    ultra_rare_threshold=0.01,
    ultra_rare_gamma=2.5,
    clamp_min=0.25,
    clamp_max=0.75,
):
    """
    Integrate numerically-stable focal loss with per-label alpha into an existing Trainer.

    If y_train is provided, alpha_pos is computed once from training labels; otherwise alpha_pos must be provided.
    Optionally bump gamma to ultra_rare_gamma if any label prevalence in y_train is < ultra_rare_threshold.
    """
    if alpha_pos is None:
        if y_train is None:
            raise ValueError("Either y_train or alpha_pos must be provided")
        alpha_pos = compute_alpha_pos(y_train, clamp_min=clamp_min, clamp_max=clamp_max)

    if use_ultra_rare_gamma and y_train is not None:
        yt = y_train if torch.is_tensor(y_train) else torch.as_tensor(y_train)
        pi = yt.float().mean(dim=0)
        if torch.any(pi < float(ultra_rare_threshold)):
            gamma = float(ultra_rare_gamma)

    def compute_loss_with_focal(model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        # Ensure alpha on correct device
        ap = alpha_pos.to(logits.device) if torch.is_tensor(alpha_pos) else torch.as_tensor(alpha_pos, device=logits.device)
        loss_fn = FocalLoss(alpha_pos=ap, gamma=gamma)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    trainer.compute_loss = compute_loss_with_focal
    return trainer

def create_balanced_sampler(df, label_columns):
    """
    Create a balanced sampler for training.
    """
    from torch.utils.data import WeightedRandomSampler
    
    # Calculate sample weights based on label frequency
    label_counts = df[label_columns].sum(axis=1)
    weights = 1.0 / (label_counts + 1)  # Add 1 to avoid division by zero
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(df),
        replacement=True
    )
    
    return sampler

def apply_imbalance_strategies(
    df,
    label_columns,
    strategies=None,
    min_samples_per_class=50,
    max_new_per_label=500,
    max_synth_to_real_ratio=1.0,
    embedding_model_name=None,
    embedding_max_len=512,
    embedding_batch_size=32,
    embedding_device=None,
):
    """
    Apply multiple imbalance handling strategies to the dataset.
    
    Args:
        df: Input DataFrame
        label_columns: List of label column names
        strategies: List of strategy names to apply
    
    Returns:
        Modified DataFrame with applied strategies
    """
    if strategies is None:
        strategies = ['focal_loss', 'back_translation']
    
    modified_df = df.copy()
    
    if 'back_translation' in strategies:
        print("Applying back-translation augmentation...")
        augmenter = BackTranslationAugmentation()
        modified_df = augmenter.augment_rare_classes(
            modified_df,
            label_columns,
            min_samples=min_samples_per_class,
            max_new_per_label=max_new_per_label,
            max_synth_to_real_ratio=max_synth_to_real_ratio
        )
    
    if 'llm_generation' in strategies:
        print("Applying LLM-based augmentation...")
        # Read keys from env to avoid hardcoding
        import os
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        provider = "openai" if os.getenv("OPENAI_API_KEY") else ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else None)
        if provider is None:
            print("⚠️ No LLM API key found in env; skipping llm_generation.")
        else:
            llm = LLMSyntheticGeneration(provider=provider, api_key=api_key)
            modified_df = llm.augment_rare_classes_with_llm(
                modified_df,
                label_columns,
                min_samples=min_samples_per_class,
                max_new_per_label=max_new_per_label,
                max_synth_to_real_ratio=max_synth_to_real_ratio
            )

    if 't5_paraphrase' in strategies:
        print("Applying T5 paraphrase augmentation...")
        t5 = T5ParaphraseAugmentation()
        modified_df = t5.augment_rare_classes(
            modified_df,
            label_columns,
            min_samples=min_samples_per_class,
            max_new_per_label=max_new_per_label,
            max_synth_to_real_ratio=max_synth_to_real_ratio
        )

    # SMOTE path removed
    
    return modified_df

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_integration():
    """
    Example of how to integrate these strategies into your existing pipeline.
    """
    
    # Example 1: Add Focal Loss to existing trainer
    def train_with_focal_loss(model_name, df_train, df_val, df_test, max_len=512, batch_size=16, epochs=2):
        # Dynamically import to avoid package path issues
        import importlib.util
        from pathlib import Path
        import sys
        utils_path = (Path(__file__).resolve().parent.parent / "utils" / "multilabel_utils.py")
        if not utils_path.exists():
            raise FileNotFoundError(f"Could not locate {utils_path}")
        sys.path.insert(0, str(utils_path.parent))
        spec = importlib.util.spec_from_file_location("multilabel_utils_local", str(utils_path))
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        train_transformer_model = getattr(mod, "train_transformer_model")

        trainer, test_results, pred_df = train_transformer_model(
            model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs
        )
        
        # Integrate focal loss
        trainer = integrate_focal_loss(trainer, alpha=1, gamma=2)
        
        return trainer, test_results, pred_df
    
    # Example 2: Multi-task learning
    def train_multitask_model(model_name, df_dict, max_len=512, batch_size=16, epochs=2):
        """
        df_dict should contain:
        {
            'perpetrator': df_perp,
            'action_type': df_action, 
            'target_type': df_target
        }
        """
        num_labels_dict = {
            'perpetrator': len([col for col in df_dict['perpetrator'].columns if col != 'incident_summary']),
            'action_type': len([col for col in df_dict['action_type'].columns if col != 'incident_summary']),
            'target_type': len([col for col in df_dict['target_type'].columns if col != 'incident_summary'])
        }
        
        model = MultiTaskModel(model_name, num_labels_dict)
        # ... rest of training logic
    
    # Example 3: Error analysis
    def analyze_and_refine(df, predictions, label_columns):
        analyzer = ErrorAnalysisRefinement()
        analyzer.analyze_confusion_matrix(predictions['true'], predictions['pred'], label_columns)
        suggestions = analyzer.suggest_refinements(df, label_columns)
        report = analyzer.generate_refinement_report(suggestions)
        print(report)
        return suggestions

if __name__ == "__main__":
    print("Imbalance handling strategies module loaded successfully!")
    print("Available strategies:")
    print("1. FocalLoss - Direct loss function replacement")
    print("2. MultiTaskModel - Shared encoder architecture")
    print("3. BackTranslationAugmentation - Data augmentation")
    print("4. T5ParaphraseAugmentation - Paraphrase augmentation")
    print("5. ErrorAnalysisRefinement - Label refinement")
    print("6. HierarchicalLabels - Label hierarchy")
    print("7. LLMSyntheticGeneration - LLM-based generation")