import numpy as np
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, WeightedRandomSampler

# Reuse core utilities
from .multilabel_utils import MultiLabelDataset, compute_metrics, train_transformer_model

# Stable focal loss with per-label alpha (kept local to avoid import path issues)
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha_pos, gamma=2.0, reduction='mean'):
        super().__init__()
        self.register_buffer("alpha_pos", torch.as_tensor(alpha_pos, dtype=torch.float))
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        # Ensure all tensors are on the same device
        device = logits.device
        alpha_pos = self.alpha_pos.to(device)
        
        logp = torch.nn.functional.logsigmoid(logits)
        log1mp = torch.nn.functional.logsigmoid(-logits)
        logpt = torch.where(targets == 1, logp, log1mp)
        pt = torch.exp(logpt)
        alpha_t = torch.where(targets == 1, alpha_pos, 1 - alpha_pos)
        loss = -(alpha_t * (1 - pt).pow(self.gamma) * logpt)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

def _compute_alpha_pos_from_df(df, label_cols, clamp_min=0.25, clamp_max=0.75):
    y = torch.as_tensor(df[label_cols].values, dtype=torch.float)
    pi = y.mean(dim=0).clamp_(min=1e-6, max=1-1e-6)
    median_pi = pi.median()
    w = torch.sqrt(median_pi / pi)
    # Ensure the tensor is on CPU initially (will be moved to device when needed)
    return torch.clamp(w, min=float(clamp_min), max=float(clamp_max)).cpu()

# Dynamic import helper to access augmentation function in sibling folder with hyphen in its name
def _load_augmented_trainer_fn():
    try:
        import importlib.util
        from pathlib import Path
        import sys
        enhanced_path = (Path(__file__).resolve().parent.parent / "imbalance-handling" / "enhanced_training_functions.py")
        if not enhanced_path.exists():
            return None
        # ensure sibling directory is importable for any relative imports inside the module
        sys.path.insert(0, str(enhanced_path.parent))
        spec = importlib.util.spec_from_file_location("imbalance_enhanced_training_functions", str(enhanced_path))
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return getattr(mod, "train_transformer_model_with_augmentation", None)
    except Exception:
        return None


def _predict_silent(trainer, dataset):
    """
    Run Trainer.predict without triggering compute_metrics prints.
    Restores the original compute_metrics afterwards.
    """
    original_cm = getattr(trainer, "compute_metrics", None)
    try:
        trainer.compute_metrics = None
        return trainer.predict(dataset)
    finally:
        trainer.compute_metrics = original_cm


def _print_pre_tuning_test_report(test_out, label_cols, header="Pre-tuning test (threshold=0.5)"):
    """
    Print a classification report on the test set using the default 0.5 threshold
    before any threshold tuning is applied.
    """
    from sklearn.metrics import classification_report
    test_true = test_out.label_ids.astype(int)
    test_probs = 1 / (1 + np.exp(-test_out.predictions))
    test_pred_05 = (test_probs >= 0.5).astype(int)
    print(f"\n=== Classification Report Context: {header} ===")
    print(classification_report(test_true, test_pred_05, target_names=label_cols, zero_division=0))

def choose_thresholds_micro(val_probs, val_true, grid=np.linspace(0.05, 0.95, 19), max_iters=3):
    n_labels = val_true.shape[1]
    thresholds = np.full(n_labels, 0.5, dtype=float)

    from sklearn.metrics import f1_score

    def micro_f1_for(th):
        preds = (val_probs >= th[None, :]).astype(int)
        return f1_score(val_true, preds, average="micro", zero_division=0)

    best = micro_f1_for(thresholds)
    for _ in range(max_iters):
        improved = False
        for j in range(n_labels):
            best_t, best_score = thresholds[j], best
            for t in grid:
                trial = thresholds.copy()
                trial[j] = t
                score = micro_f1_for(trial)
                if score > best_score:
                    best_t, best_score = t, score
            if best_t != thresholds[j]:
                thresholds[j] = best_t
                best = best_score
                improved = True
        if not improved:
            break
    return thresholds


def apply_thresholds(probs, thresholds):
    return (probs >= thresholds[None, :]).astype(int)


# =============================================================================
# Per-label threshold tuning helpers (macro- or micro-F1 objective)
# =============================================================================

def _apply_thresholds(probs, th):
    return (probs >= th[None, :]).astype(int)


def choose_thresholds(val_probs, val_true, objective="macro", grid=np.linspace(0.05, 0.95, 19), max_iters=3):
    """
    Coordinate-ascent per-label threshold tuning on validation set.

    objective: "macro" or "micro" F1.
    Returns an array of thresholds, one per label.
    """
    from sklearn.metrics import f1_score

    n_labels = val_true.shape[1]
    th = np.full(n_labels, 0.5, dtype=float)
    avg = "micro" if objective == "micro" else "macro"

    def score_for(cur):
        preds = _apply_thresholds(val_probs, cur)
        return f1_score(val_true, preds, average=avg, zero_division=0)

    best = score_for(th)
    for _ in range(max_iters):
        improved = False
        for j in range(n_labels):
            best_t, best_s = th[j], best
            for t in grid:
                trial = th.copy()
                trial[j] = t
                s = score_for(trial)
                if s > best_s:
                    best_t, best_s = t, s
            if best_t != th[j]:
                th[j] = best_t
                best = best_s
                improved = True
        if not improved:
            break
    return th


def choose_thresholds_per_label(val_probs, val_true, grid=np.linspace(0.0, 1.0, 101), beta=1.0):
    """
    Select a threshold per label independently by maximizing F-beta on the
    validation set. Defaults to F1 (beta=1.0).
    Returns an array of thresholds, one per label.
    """
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

def tuned_metrics_from_trainer(trainer, tokenizer, df_val, df_test, label_cols, max_len=512, objective="macro"):
    """
    Compute test metrics after tuning per-label thresholds on validation set.
    Uses the provided trained Trainer to generate probabilities.
    """
    from sklearn.metrics import classification_report, f1_score, hamming_loss, accuracy_score

    val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
    test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)

    val_out = _predict_silent(trainer, val_ds)
    test_out = _predict_silent(trainer, test_ds)

    # Print pre-tuning test report @0.5 before selecting thresholds
    _print_pre_tuning_test_report(test_out, label_cols, header="Pre-tuning test (threshold=0.5)")
    val_probs = 1 / (1 + np.exp(-val_out.predictions))
    test_probs = 1 / (1 + np.exp(-test_out.predictions))

    th = choose_thresholds_per_label(val_probs, val_out.label_ids.astype(int))
    test_true = test_out.label_ids.astype(int)
    test_pred = _apply_thresholds(test_probs, th)

    # Print a single labeled report for tuned final test
    print("\n=== Classification Report Context: Threshold tuned final test ===")
    print(classification_report(test_true, test_pred, target_names=label_cols, zero_division=0))
    report = classification_report(test_true, test_pred, target_names=label_cols, zero_division=0, output_dict=True)
    micro_f1 = f1_score(test_true, test_pred, average="micro", zero_division=0)
    metrics = {
        "hamming_loss": hamming_loss(test_true, test_pred),
        "subset_accuracy": accuracy_score(test_true, test_pred),
        "micro_f1": micro_f1,
    }
    metrics.update(report)
    return metrics


# -----------------------------------------------------------------------------
# Weighted sampler helpers (class-aware sampling)
# -----------------------------------------------------------------------------

def _label_inverse_frequency_sample_weights(df, label_cols):
    """
    Compute per-sample weights that emphasize rare labels using inverse label frequency.

    For each label j: w_label[j] = 1 / max(P_j, 1)
    For each sample i:  w_i = sum_j y_ij * w_label[j]

    We then normalize weights to have mean 1.0 and ensure strictly positive values.
    """
    labels_matrix = df[label_cols].values.astype(float)
    # Count positives per label; guard against zeros
    positives_per_label = labels_matrix.sum(axis=0)
    inv_freq_per_label = 1.0 / np.maximum(positives_per_label, 1.0)
    # Sample weights as sum of inverse frequencies for positive labels in the row
    sample_weights = labels_matrix @ inv_freq_per_label
    sample_weights = sample_weights.astype(float)
    # Ensure strictly positive weights
    if np.any(sample_weights <= 0):
        min_positive = np.min(sample_weights[sample_weights > 0]) if np.any(sample_weights > 0) else 1.0
        sample_weights = np.where(sample_weights <= 0, min_positive, sample_weights)
    # Normalize to mean 1 for stability
    mean_w = sample_weights.mean() if sample_weights.mean() > 0 else 1.0
    sample_weights = sample_weights / mean_w
    return sample_weights


class WeightedSamplerTrainer(Trainer):
    """
    Trainer that injects a WeightedRandomSampler for the training dataloader.
    """

    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if sample_weights is None:
            raise ValueError("sample_weights must be provided for WeightedSamplerTrainer")
        # Store as double for WeightedRandomSampler numerical stability
        self._sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            return super().get_train_dataloader()
        sampler = WeightedRandomSampler(
            weights=self._sample_weights,
            num_samples=len(self._sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
        )


def train_with_weighted_sampler(model_name, df_train, df_val, df_test, max_len=512, batch_size=16, epochs=2, seed=42):
    # Reproducibility seeds for this training run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    label_cols = [c for c in df_train.columns if c != "incident_summary"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_cols), problem_type="multi_label_classification"
    )

    # Datasets
    train_ds = MultiLabelDataset(df_train["incident_summary"].tolist(), df_train[label_cols].values, tokenizer, max_len)
    val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
    test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)

    # Arguments
    args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='eval_pr_auc_macro',
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    # Compute class-aware per-sample weights from df_train
    sample_weights = _label_inverse_frequency_sample_weights(df_train, label_cols)

    trainer = WeightedSamplerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda x: compute_metrics(
            x, label_cols,
            context_label="Validation (tuning)",
            mask_zero_labels=True,
            mask_ultra_rare_threshold=0.01,
        ),
        sample_weights=sample_weights,
    )
    trainer.train()

    # Evaluate on test
    predictions_output = _predict_silent(trainer, test_ds)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    bin_preds = (probs > 0.5).astype(int)

    pred_df = pd.DataFrame()
    for i, col in enumerate(label_cols):
        pred_df[f"true_{col}"] = labels[:, i]
        pred_df[f"pred_{col}"] = bin_preds[:, i]
        pred_df[f"prob_{col}"] = probs[:, i]
    pred_df["incident_summary"] = df_test["incident_summary"].values

    test_results = predictions_output.metrics
    return trainer, test_results, pred_df

def _pos_weight_from_df(df, label_cols):
    P = df[label_cols].sum(axis=0).values.astype(np.float32)
    N = len(df) - P
    P = np.where(P == 0, 1.0, P)
    return torch.tensor((N / P), dtype=torch.float)


def train_with_class_weights(model_name, df_train, df_val, df_test, max_len=512, batch_size=16, epochs=2, seed=42):
    # Reproducibility seeds for this training run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    label_cols = [c for c in df_train.columns if c != "incident_summary"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_cols), problem_type="multi_label_classification"
    )
    pos_weight = _pos_weight_from_df(df_train, label_cols).to(model.device)

    train_ds = MultiLabelDataset(df_train["incident_summary"].tolist(), df_train[label_cols].values, tokenizer, max_len)
    val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
    test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)

    args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='eval_pr_auc_macro',
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    class WeightedBCETrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # Ensure tensors are on the same device
            device = logits.device
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels.to(device),
                pos_weight=pos_weight.to(device)
            )
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedBCETrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda x: compute_metrics(
            x, label_cols,
            context_label="Validation (tuning)",
            mask_zero_labels=True,
            mask_ultra_rare_threshold=0.01,
        )
    )
    trainer.train()

    # Evaluate on test: use predict() once to gather both metrics and logits
    predictions_output = _predict_silent(trainer, test_ds)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    bin_preds = (probs > 0.5).astype(int)

    # Conform to pred_df format used elsewhere
    pred_df = pd.DataFrame()
    for i, col in enumerate(label_cols):
        pred_df[f"true_{col}"] = labels[:, i]
        pred_df[f"pred_{col}"] = bin_preds[:, i]
        pred_df[f"prob_{col}"] = probs[:, i]
    pred_df["incident_summary"] = df_test["incident_summary"].values

    # Also compute standard metrics dict from predict() output to avoid duplicate prints
    test_results = predictions_output.metrics
    return trainer, test_results, pred_df


def train_with_conservative_class_weights(
    model_name, df_train, df_val, df_test, max_len=512, batch_size=16, epochs=2, seed=42,
    cap_ratio=3.0, sqrt_scaling=True, min_weight=1.0, max_weight=10.0,
    label_specific_configs=None
):
    """
    Train model with conservative class weights to prevent precision collapse.
    
    This function uses capped and sqrt-scaled class weights instead of raw pos_weight (N/P)
    to maintain recall gains while preventing precision collapse.
    
    Args:
        model_name: Hugging Face model name
        df_train, df_val, df_test: Training, validation, and test DataFrames
        max_len: Maximum sequence length for tokenization
        batch_size: Training batch size
        epochs: Number of training epochs
        seed: Random seed for reproducibility
        cap_ratio: Maximum ratio for capping weights (default: 3.0)
        sqrt_scaling: Whether to apply sqrt scaling (default: True)
        min_weight: Minimum weight value (default: 1.0)
        max_weight: Maximum weight value (default: 10.0)
        label_specific_configs: Dict for label-specific configurations
    
    Returns:
        tuple: (trainer, test_results, pred_df)
    """
    # Reproducibility seeds for this training run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    label_cols = [c for c in df_train.columns if c != "incident_summary"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_cols), problem_type="multi_label_classification"
    )
    
    # Import and use the conservative weight function
    from imbalance_handling_strategies import compute_label_specific_conservative_weights
    
    if label_specific_configs is not None:
        pos_weight = compute_label_specific_conservative_weights(
            df_train, label_cols, label_specific_configs
        ).to(model.device)
    else:
        pos_weight = compute_label_specific_conservative_weights(
            df_train, label_cols
        ).to(model.device)
    
    # Print weight comparison for transparency
    raw_weights = _pos_weight_from_df(df_train, label_cols)
    print(f"Conservative class weights applied:")
    print(f"Raw weights (N/P): {raw_weights.numpy()}")
    print(f"Conservative weights: {pos_weight.cpu().numpy()}")
    print(f"Reduction factor: {(raw_weights / pos_weight.cpu()).numpy()}")
    
    train_ds = MultiLabelDataset(df_train["incident_summary"].tolist(), df_train[label_cols].values, tokenizer, max_len)
    val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
    test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)

    args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='eval_pr_auc_macro',
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    class ConservativeWeightedBCETrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # Ensure tensors are on the same device
            device = logits.device
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels.to(device),
                pos_weight=pos_weight.to(device)
            )
            return (loss, outputs) if return_outputs else loss

    trainer = ConservativeWeightedBCETrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda x: compute_metrics(
            x, label_cols,
            context_label="Validation (conservative weights)",
            mask_zero_labels=True,
            mask_ultra_rare_threshold=0.01,
        )
    )
    trainer.train()

    # Evaluate on test: use predict() once to gather both metrics and logits
    predictions_output = _predict_silent(trainer, test_ds)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    bin_preds = (probs > 0.5).astype(int)

    # Conform to pred_df format used elsewhere
    pred_df = pd.DataFrame()
    for i, col in enumerate(label_cols):
        pred_df[f"true_{col}"] = labels[:, i]
        pred_df[f"pred_{col}"] = bin_preds[:, i]
        pred_df[f"prob_{col}"] = probs[:, i]
    pred_df["incident_summary"] = df_test["incident_summary"].values

    test_results = predictions_output.metrics
    return trainer, test_results, pred_df


# -----------------------------------------------------------------------------

def run_strategy_experiments(
    df_train,
    df_val,
    df_test,
    label_cols,
    model_name="snowood1/ConfliBERT-scr-cased",
    strategies=None,
    max_len=512,
    batch_size=16,
    epochs=2,
    results_csv=None,
    predictions_csv=None,
    seed=42
):
    # Global seeds for a consistent experiment run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strategies is None:
        strategies = [
            "baseline",
            "focal",
            "class_weights",
            "conservative_class_weights",  # New conservative reweighting strategy
            "threshold_tuned",
            "weighted_sampler",  # placeholder uses class_weights trainer for now
            "augmentation_bt",
            # "augmentation_t5",  # enable explicitly when needed
        ]

    per_strategy_reports = {}
    all_predictions = []
    
    # Fallback function for basic threshold tuning
    def _basic_threshold_tuning(trainer, test_ds, label_cols):
        """Basic threshold tuning when calibration is not available."""
        try:
            # Get predictions
            predictions_output = _predict_silent(trainer, test_ds)
            logits = predictions_output.predictions
            labels = predictions_output.label_ids
            
            # Ensure logits are on CPU for numpy conversion
            if hasattr(logits, 'cpu'):
                logits = logits.cpu()
            logits_tensor = torch.tensor(logits)
            probs = torch.sigmoid(logits_tensor).numpy()
            
            # Use basic threshold tuning
            thresholds = choose_thresholds_micro(probs, labels)
            bin_preds = (probs >= thresholds[None, :]).astype(int)
            
            # Create results DataFrame
            pred_df = pd.DataFrame()
            for i, col in enumerate(label_cols):
                pred_df[f"true_{col}"] = labels[:, i]
                pred_df[f"pred_{col}"] = bin_preds[:, i]
                pred_df[f"prob_{col}"] = probs[:, i]
            
            # Compute metrics
            from sklearn.metrics import f1_score
            micro_f1 = f1_score(labels, bin_preds, average="micro", zero_division=0)
            macro_f1 = f1_score(labels, bin_preds, average="macro", zero_division=0)
            
            metrics = {
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "objective": "micro_f1"
            }
            
            return {
                "metrics": metrics,
                "predictions": pred_df,
                "temperature": 1.0,  # No temperature scaling
                "thresholds": thresholds
            }
            
        except Exception as e:
            print(f"Basic threshold tuning failed: {e}")
            raise e
    
    # Helper function to apply calibration + threshold tuning to any strategy
    def apply_calibration_and_thresholds(trainer, strategy_name, verbose=True):
        """Apply temperature scaling calibration + threshold tuning to any trained model."""
        try:
            # Build datasets for prediction
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
            test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)
            
            # Import calibration functions
            try:
                import sys
                from pathlib import Path
                # Try multiple import paths for flexibility
                possible_paths = [
                    str(Path(__file__).resolve().parent.parent / "imbalance-handling"),
                    str(Path(__file__).resolve().parent.parent / "imbalance-handling" / "imbalance_handling_strategies.py"),
                    "./imbalance-handling",
                    "../imbalance-handling"
                ]
                
                imported = False
                for path in possible_paths:
                    try:
                        if path.endswith('.py'):
                            # Direct import from file
                            import importlib.util
                            spec = importlib.util.spec_from_file_location("imbalance_handling_strategies", path)
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            calibrate_and_tune_thresholds = getattr(mod, "calibrate_and_tune_thresholds")
                            imported = True
                            break
                        else:
                            # Add to path and import
                            sys.path.insert(0, path)
                            from imbalance_handling_strategies import calibrate_and_tune_thresholds
                            imported = True
                            break
                    except Exception:
                        continue
                
                if not imported:
                    raise ImportError("Could not import calibration functions from any path")
                    
            except ImportError as e:
                print(f"Warning: Could not import calibration functions: {e}")
                print("Falling back to basic threshold tuning...")
                # Fallback to basic threshold tuning
                return _basic_threshold_tuning(trainer, test_ds, label_cols)
            
            if verbose:
                print(f"Applying temperature scaling calibration + threshold tuning to {strategy_name}...")
            
            # Apply calibration + threshold tuning
            calibration_results = calibrate_and_tune_thresholds(
                trainer, 
                val_ds, 
                test_ds, 
                label_cols, 
                objective="micro",  # Use micro-F1 for threshold tuning
                max_temp_iter=50, 
                verbose=verbose
            )
            
            return calibration_results
            
        except Exception as e:
            print(f"Calibration failed for {strategy_name}: {e}")
            raise e

    # Baseline
    if "baseline" in strategies:
        try:
            print("\n[Strategy] baseline: starting...")
            trainer, base_metrics, base_pred_df = train_transformer_model(
                model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs, seed=seed
            )
            per_strategy_reports["baseline"] = base_metrics
            # attach strategy label and retain probabilities for downstream PR/AUPRC
            base_pred_df = base_pred_df.copy()
            base_pred_df["strategy"] = "baseline"
            all_predictions.append(base_pred_df)
            print("[Strategy] baseline: ✅ completed")
        except Exception as e:
            per_strategy_reports["baseline"] = None
            print(f"[Strategy] baseline: ❌ failed — {e}")

    # Focal (local override of compute_loss)
    if "focal" in strategies:
        try:
            print("\n[Strategy] focal: starting...")
            # Use the label_cols parameter instead of redefining it
            focal_label_cols = label_cols
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(focal_label_cols), problem_type="multi_label_classification"
            )
            train_ds = MultiLabelDataset(df_train["incident_summary"].tolist(), df_train[focal_label_cols].values, tokenizer, max_len)
            val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[focal_label_cols].values, tokenizer, max_len)
            test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[focal_label_cols].values, tokenizer, max_len)

            args = TrainingArguments(
                output_dir="./results", eval_strategy="epoch", save_strategy="epoch",
                learning_rate=2e-5, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs, weight_decay=0.01, logging_dir="./logs", logging_steps=10,
                load_best_model_at_end=True, metric_for_best_model='eval_micro_f1', greater_is_better=True,
                save_total_limit=2, report_to="none", seed=seed, data_seed=seed,
            )

            alpha_pos = _compute_alpha_pos_from_df(df_train, focal_label_cols)

            # Ultra-rare gamma bump option based on train prevalence < 1%
            pi = df_train[focal_label_cols].values.mean(axis=0)
            gamma_use = 2.5 if (pi < 0.01).any() else 2.0

            focal = FocalLoss(alpha_pos=alpha_pos, gamma=gamma_use)
            
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # Debug: Print device information
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Alpha pos device: {alpha_pos.device}")
            print(f"Target device: {device}")

            class FocalTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # Ensure labels are on the same device as logits
                    if hasattr(labels, 'to'):
                        labels = labels.to(logits.device)
                    
                    loss = focal(logits, labels)
                    return (loss, outputs) if return_outputs else loss

            trainer = FocalTrainer(
                model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                compute_metrics=lambda x: compute_metrics(
                    x, focal_label_cols,
                    context_label="Validation (tuning)",
                    mask_zero_labels=True,
                    mask_ultra_rare_threshold=0.01,
                )
            )
            trainer.train()

            # Apply calibration + threshold tuning to focal loss model
            calibration_results = apply_calibration_and_thresholds(trainer, "focal", verbose=True)
            
            metrics = calibration_results["metrics"]
            per_strategy_reports["focal_tuned"] = metrics

            # Use calibration results for predictions DataFrame
            focal_tuned_df = calibration_results["predictions"]
            focal_tuned_df["incident_summary"] = df_test["incident_summary"].values
            focal_tuned_df["strategy"] = "focal_tuned"
            focal_tuned_df["temperature"] = calibration_results["temperature"]
            focal_tuned_df["calibration_objective"] = calibration_results["metrics"]["objective"]
            all_predictions.append(focal_tuned_df)
            print("[Strategy] focal: ✅ completed")
        except Exception as e:
            per_strategy_reports["focal"] = None
            print(f"[Strategy] focal: ❌ failed — {e}")

    # Class-weighted (report tuned only)
    if "class_weights" in strategies:
        try:
            print("\n[Strategy] class_weights: starting...")
            trainer, cw_metrics, cw_pred_df = train_with_class_weights(
                model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs, seed=seed
            )
            # Apply calibration + threshold tuning to class weights model
            calibration_results = apply_calibration_and_thresholds(trainer, "class_weights", verbose=True)
            
            metrics = calibration_results["metrics"]
            per_strategy_reports["class_weights_tuned"] = metrics

            # Use calibration results for predictions DataFrame
            cw_tuned_df = calibration_results["predictions"]
            cw_tuned_df["incident_summary"] = df_test["incident_summary"].values
            cw_tuned_df["strategy"] = "class_weights_tuned"
            cw_tuned_df["temperature"] = calibration_results["temperature"]
            cw_tuned_df["calibration_objective"] = calibration_results["metrics"]["objective"]
            all_predictions.append(cw_tuned_df)
            print("[Strategy] class_weights: ✅ completed")
        except Exception as e:
            per_strategy_reports["class_weights"] = None
            print(f"[Strategy] class_weights: ❌ failed — {e}")

    # Conservative class weights (capped and sqrt-scaled to prevent precision collapse)
    if "conservative_class_weights" in strategies:
        try:
            print("\n[Strategy] conservative_class_weights: starting...")
            trainer, cw_metrics, cw_pred_df = train_with_conservative_class_weights(
                model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs, seed=seed,
                cap_ratio=3.0, sqrt_scaling=True, min_weight=1.0, max_weight=10.0
            )
            per_strategy_reports["conservative_class_weights"] = cw_metrics
            
            # Apply calibration + threshold tuning to conservative class weights model
            calibration_results = apply_calibration_and_thresholds(trainer, "conservative_class_weights", verbose=True)
            
            metrics = calibration_results["metrics"]
            per_strategy_reports["conservative_class_weights_tuned"] = metrics

            # Use calibration results for predictions DataFrame
            cw_tuned_df = calibration_results["predictions"]
            cw_tuned_df["incident_summary"] = df_test["incident_summary"].values
            cw_tuned_df["strategy"] = "conservative_class_weights_tuned"
            cw_tuned_df["temperature"] = calibration_results["temperature"]
            cw_tuned_df["calibration_objective"] = calibration_results["metrics"]["objective"]
            all_predictions.append(cw_tuned_df)
            print("[Strategy] conservative_class_weights: ✅ completed")
        except Exception as e:
            per_strategy_reports["conservative_class_weights"] = None
            print(f"[Strategy] conservative_class_weights: ❌ failed — {e}")

    # Weighted sampler (report tuned only)
    if "weighted_sampler" in strategies:
        try:
            print("\n[Strategy] weighted_sampler: starting...")
            trainer, ws_metrics, ws_pred_df = train_with_weighted_sampler(
                model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs, seed=seed
            )
            # Apply calibration + threshold tuning to weighted sampler model
            calibration_results = apply_calibration_and_thresholds(trainer, "weighted_sampler", verbose=True)
            
            metrics = calibration_results["metrics"]
            per_strategy_reports["weighted_sampler_tuned"] = metrics

            # Use calibration results for predictions DataFrame
            ws_tuned_df = calibration_results["predictions"]
            ws_tuned_df["incident_summary"] = df_test["incident_summary"].values
            ws_tuned_df["strategy"] = "weighted_sampler_tuned"
            ws_tuned_df["temperature"] = calibration_results["temperature"]
            ws_tuned_df["calibration_objective"] = calibration_results["metrics"]["objective"]
            all_predictions.append(ws_tuned_df)
            print("[Strategy] weighted_sampler: ✅ completed")
        except Exception as e:
            per_strategy_reports["weighted_sampler_tuned"] = None
            print(f"[Strategy] weighted_sampler: ❌ failed — {e}")



    # Threshold-tuned (baseline model with calibration + threshold tuning)
    if "threshold_tuned" in strategies:
        try:
            print("\n[Strategy] threshold_tuned: starting...")
            # Train a clean baseline model for threshold selection
            trained_trainer, _, _ = train_transformer_model(
                model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs, seed=seed
            )

            # Apply calibration + threshold tuning
            calibration_results = apply_calibration_and_thresholds(trained_trainer, "threshold_tuned", verbose=True)
            
            metrics = calibration_results["metrics"]
            per_strategy_reports["threshold_tuned"] = metrics

            # Use calibration results for predictions DataFrame
            th_pred_df = calibration_results["predictions"]
            th_pred_df["incident_summary"] = df_test["incident_summary"].values
            th_pred_df["original_idx"] = df_test.index.tolist()
            th_pred_df["strategy"] = "threshold_tuned"
            th_pred_df["temperature"] = calibration_results["temperature"]
            th_pred_df["calibration_objective"] = calibration_results["metrics"]["objective"]
            all_predictions.append(th_pred_df)
            print("[Strategy] threshold_tuned: ✅ completed")
        except Exception as e:
            per_strategy_reports["threshold_tuned"] = None
            print(f"[Strategy] threshold_tuned: ❌ failed — {e}")

    # (weighted_sampler handled above)

    # Augmentation
    # Augmentation via back-translation (report tuned only)
    if "augmentation_bt" in strategies:
        train_with_aug = _load_augmented_trainer_fn()
        if train_with_aug is None:
            print("\n[Strategy] augmentation_bt: ⚠️ augmentation trainer not available; skipping")
        else:
            try:
                print("\n[Strategy] augmentation_bt: starting...")
                trainer, aug_metrics, aug_pred_df = train_with_aug(
                    model_name, df_train, df_val, df_test,
                    max_len=max_len, batch_size=batch_size, epochs=epochs,
                    augmentation_strategies=['back_translation']
                )
                # Apply calibration + threshold tuning to back-translation augmentation model
                calibration_results = apply_calibration_and_thresholds(trainer, "augmentation_bt", verbose=True)
                
                metrics = calibration_results["metrics"]
                per_strategy_reports["augmentation_bt_tuned"] = metrics

                # Use calibration results for predictions DataFrame
                aug_bt_tuned_df = calibration_results["predictions"]
                aug_bt_tuned_df["incident_summary"] = df_test["incident_summary"].values
                aug_bt_tuned_df["strategy"] = "augmentation_bt_tuned"
                aug_bt_tuned_df["temperature"] = calibration_results["temperature"]
                aug_bt_tuned_df["calibration_objective"] = calibration_results["metrics"]["objective"]
                all_predictions.append(aug_bt_tuned_df)
                print("[Strategy] augmentation_bt: ✅ completed")
            except Exception as e:
                print(f"[Strategy] augmentation_bt: ❌ failed — {e}")
                per_strategy_reports["augmentation_bt_tuned"] = None

    # Augmentation via T5 paraphrase (report tuned only)
    if "augmentation_t5" in strategies:
        train_with_aug = _load_augmented_trainer_fn()
        if train_with_aug is None:
            print("\n[Strategy] augmentation_t5: ⚠️ augmentation trainer not available; skipping")
        else:
            try:
                print("\n[Strategy] augmentation_t5: starting...")
                trainer, aug_metrics, _ = train_with_aug(
                    model_name, df_train, df_val, df_test,
                    max_len=max_len, batch_size=batch_size, epochs=epochs,
                    augmentation_strategies=['t5_paraphrase']
                )
                # Apply calibration + threshold tuning to T5 paraphrase augmentation model
                calibration_results = apply_calibration_and_thresholds(trainer, "augmentation_t5", verbose=True)
                
                metrics = calibration_results["metrics"]
                per_strategy_reports["augmentation_t5_tuned"] = metrics

                # Use calibration results for predictions DataFrame
                aug_t5_tuned_df = calibration_results["predictions"]
                aug_t5_tuned_df["incident_summary"] = df_test["incident_summary"].values
                aug_t5_tuned_df["strategy"] = "augmentation_t5_tuned"
                aug_t5_tuned_df["temperature"] = calibration_results["temperature"]
                aug_t5_tuned_df["calibration_objective"] = calibration_results["metrics"]["objective"]
                all_predictions.append(aug_t5_tuned_df)
                print("[Strategy] augmentation_t5: ✅ completed")
            except Exception as e:
                print(f"[Strategy] augmentation_t5: ❌ failed — {e}")
                per_strategy_reports["augmentation_t5_tuned"] = None

    # Build long-form results for plotting strategy heatmap
    rows = []
    for strat_name, metrics in per_strategy_reports.items():
        if metrics is None:
            continue
        for lbl in label_cols:
            # metrics may store per-label reports under raw, eval_, or test_ keys
            for key in (lbl, f"eval_{lbl}", f"test_{lbl}"):
                if key in metrics and isinstance(metrics[key], dict):
                    f1 = metrics[key].get("f1-score", 0.0)
                    prec = metrics[key].get("precision", 0.0)
                    rec = metrics[key].get("recall", 0.0)
                    break
            else:
                f1, prec, rec = 0.0, 0.0, 0.0
            rows.append({"strategy": strat_name, "label": lbl, "f1": f1, "precision": prec, "recall": rec})
    results_df = pd.DataFrame(rows)

    # Pivot is often handy to save directly; guard for empty results
    if results_df.empty:
        # Optionally save empty CSVs to keep downstream steps simple
        if results_csv is not None:
            results_df.to_csv(results_csv, index=False)
        if predictions_csv is not None and len(all_predictions) > 0:
            pd.concat(all_predictions, ignore_index=True).to_csv(predictions_csv, index=False)
        return pd.DataFrame(), results_df
    pivot = results_df.pivot(index="label", columns="strategy", values="f1")
    # Optional persistence of results and predictions
    if results_csv is not None:
        results_df.to_csv(results_csv, index=False)
    if predictions_csv is not None and len(all_predictions) > 0:
        combined_pred_df = pd.concat(all_predictions, ignore_index=True)
        combined_pred_df.to_csv(predictions_csv, index=False)
    return pivot, results_df


