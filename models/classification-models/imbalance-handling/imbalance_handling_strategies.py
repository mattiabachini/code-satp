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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# TIER 1: IMMEDIATE IMPLEMENTATION (HIGHEST ROI)
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in multi-label classification.
    
    Focal Loss reduces the relative loss for well-classified examples and
    puts more focus on hard, misclassified examples.
    
    Reference: Lin, T. Y., et al. "Focal loss for dense object detection." ICCV 2017.
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits) of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size, num_classes)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal loss
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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
    
    def augment_text(self, text, num_augmentations=1):
        """
        Augment a single text using back-translation.
        
        Args:
            text: Original text
            num_augmentations: Number of augmented versions to create
        """
        if not self.available:
            return []
        
        augmented_texts = []
        
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
                    
                    augmented_texts = self.augment_text(row['incident_summary'], num_augmentations=1)
                    
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
            if self.device is not None:
                self._pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=self.device)
            else:
                self._pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        except Exception as e:
            print(f"⚠️ T5 paraphraser unavailable: {e}")
            self.available = False
    
    def paraphrase(self, text, num_return_sequences=2, temperature=0.7, top_p=0.9, max_new_tokens=64):
        """
        Generate paraphrases for a single text.
        """
        if not self.available:
            return []
        self._ensure_pipeline()
        if self._pipeline is None:
            return []
        prompt = (
            "Paraphrase the following incident summary without changing meaning, entities, actors, or event type. "
            "Use a neutral, report-like tone and do not add or remove facts.\n\n"
            f"Text: {text}\n\nParaphrase:"
        )
        try:
            outputs = self._pipeline(
                prompt,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
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
# INTEGRATION HELPERS
# =============================================================================

def integrate_focal_loss(trainer, alpha=1, gamma=2):
    """
    Integrate Focal Loss into existing Hugging Face Trainer.
    """
    focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
    
    # Override the compute_loss method
    original_compute_loss = trainer.compute_loss
    
    def compute_loss_with_focal(model, inputs, return_outputs=False, **kwargs):
        # Hugging Face Trainer may pass extra kwargs (e.g., num_items_in_batch); accept and ignore them
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        
        # Apply focal loss
        loss = focal_loss(logits, labels)
        
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