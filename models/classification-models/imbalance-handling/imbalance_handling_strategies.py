"""
Imbalance Handling Strategies for SATP Classification Tasks

This module provides various strategies to handle class imbalance in multi-label classification
tasks. It's designed to integrate seamlessly with the existing Hugging Face training pipeline.

Strategies included:
1. Focal Loss Implementation
2. Multi-task Learning Architecture  
3. Data Augmentation (Back-translation)
4. SMOTE Variants (BorderlineSMOTE)
5. Error Analysis-Driven Label Refinement
6. Hierarchical Label Organization
7. LLM-based Synthetic Generation

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
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
            from googletrans import Translator
            self.translator = Translator()
            self.available = True
        except ImportError:
            print("⚠️ Warning: googletrans not available for back-translation.")
            print("🔧 Install with:")
            print("   !pip install googletrans==3.1.0a0 --no-deps")
            print("   !pip install httpx==0.13.3 chardet==3.0.4 hstspreload")
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
                
                # Translate to target language
                translated = self.translator.translate(text, dest=target_lang)
                
                # Translate back to English
                back_translated = self.translator.translate(translated.text, dest='en')
                
                augmented_texts.append(back_translated.text)
                
            except Exception as e:
                print(f"Translation error: {e}")
                continue
        
        return augmented_texts
    
    def augment_rare_classes(self, df, label_columns, min_samples=50):
        """
        Augment rare classes to have at least min_samples examples.
        
        Args:
            df: DataFrame with 'incident_summary' and label columns
            label_columns: List of label column names
            min_samples: Minimum number of samples per class
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
                
                # Get examples of this rare class
                rare_examples = df[df[col] == 1]
                
                # Calculate how many augmentations needed
                needed = min_samples - len(rare_examples)
                
                if needed <= 0:
                    continue
                
                print(f"Augmenting class '{col}' (rare class {rare_class}): {len(rare_examples)} -> {min_samples}")
                
                # Augment each example
                for _, row in rare_examples.iterrows():
                    if len(augmented_data) >= needed:
                        break
                    
                    augmented_texts = self.augment_text(row['incident_summary'], num_augmentations=1)
                    
                    for aug_text in augmented_texts:
                        if len(augmented_data) >= needed:
                            break
                        
                        # Create new row with augmented text
                        new_row = row.copy()
                        new_row['incident_summary'] = aug_text
                        augmented_data.append(new_row)
        
        # Combine original and augmented data
        if augmented_data:
            augmented_df = pd.DataFrame(augmented_data)
            return pd.concat([df, augmented_df], ignore_index=True)
        
        return df

# =============================================================================
# TIER 2: MEDIUM-TERM IMPLEMENTATION (GOOD ROI)
# =============================================================================

class EmbeddingSMOTE:
    """
    SMOTE for text embeddings to handle class imbalance.
    
    This class applies SMOTE variants to BERT embeddings to generate
    synthetic examples for rare classes.
    """
    
    def __init__(self, model_name="bert-base-cased", max_len=512):
        self.model_name = model_name
        self.max_len = max_len
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def get_embeddings(self, texts):
        """
        Extract BERT embeddings for a list of texts.
        """
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text, 
                    max_length=self.max_len, 
                    padding='max_length', 
                    truncation=True, 
                    return_tensors='pt'
                )
                
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def apply_smote(self, df, label_columns, sampling_strategy='auto'):
        """
        Apply SMOTE to embeddings for each label.
        
        Args:
            df: DataFrame with 'incident_summary' and label columns
            label_columns: List of label column names
            sampling_strategy: SMOTE sampling strategy
        """
        print("Extracting BERT embeddings...")
        embeddings = self.get_embeddings(df['incident_summary'].tolist())
        
        augmented_data = []
        
        for col in label_columns:
            print(f"Applying SMOTE for label: {col}")
            
            # Get labels for this column
            labels = df[col].values
            
            # Apply BorderlineSMOTE
            smote = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42)
            
            try:
                # Reshape embeddings for SMOTE
                X_reshaped = embeddings.reshape(embeddings.shape[0], -1)
                
                # Apply SMOTE
                X_resampled, y_resampled = smote.fit_resample(X_reshaped, labels)
                
                # Find new synthetic samples
                original_count = len(embeddings)
                synthetic_count = len(X_resampled) - original_count
                
                if synthetic_count > 0:
                    print(f"Generated {synthetic_count} synthetic samples for {col}")
                    
                    # Create synthetic text examples (simplified - in practice you'd need to decode embeddings)
                    for i in range(synthetic_count):
                        # For now, we'll use the original text but mark as synthetic
                        # In a full implementation, you'd decode the embedding back to text
                        synthetic_row = df.iloc[i % len(df)].copy()
                        synthetic_row['incident_summary'] = f"[SYNTHETIC] {synthetic_row['incident_summary']}"
                        synthetic_row[col] = 1  # Set the target label
                        augmented_data.append(synthetic_row)
                
            except Exception as e:
                print(f"SMOTE failed for {col}: {e}")
                continue
        
        # Combine original and augmented data
        if augmented_data:
            augmented_df = pd.DataFrame(augmented_data)
            return pd.concat([df, augmented_df], ignore_index=True)
        
        return df

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
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.available = api_key is not None
    
    def generate_synthetic_examples(self, class_name, num_examples=10, context=""):
        """
        Generate synthetic examples using LLM.
        """
        if not self.available:
            return []
        
        try:
            import openai
            
            prompt = f"""
            Generate {num_examples} realistic incident summaries that would be classified as "{class_name}" 
            in the context of {context}. Each summary should be 1-3 sentences and describe a real-world 
            incident that would fit this category.
            
            Format each example on a new line starting with "- ".
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            # Parse response
            examples = response.choices[0].message.content.split('\n')
            examples = [ex.strip('- ') for ex in examples if ex.strip().startswith('- ')]
            
            return examples
            
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return []

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
    
    def compute_loss_with_focal(model, inputs, return_outputs=False):
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

def apply_imbalance_strategies(df, label_columns, strategies=None):
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
        strategies = ['focal_loss', 'back_translation', 'smote']
    
    modified_df = df.copy()
    
    if 'back_translation' in strategies:
        print("Applying back-translation augmentation...")
        augmenter = BackTranslationAugmentation()
        modified_df = augmenter.augment_rare_classes(modified_df, label_columns)
    
    if 'smote' in strategies:
        print("Applying SMOTE to embeddings...")
        smote_processor = EmbeddingSMOTE()
        modified_df = smote_processor.apply_smote(modified_df, label_columns)
    
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
        # Your existing training function
        trainer, test_results, pred_df = train_transformer_model(
            model_name, df_train, df_val, df_test, max_len, batch_size, epochs
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
    print("4. EmbeddingSMOTE - SMOTE on embeddings")
    print("5. ErrorAnalysisRefinement - Label refinement")
    print("6. HierarchicalLabels - Label hierarchy")
    print("7. LLMSyntheticGeneration - LLM-based generation") 