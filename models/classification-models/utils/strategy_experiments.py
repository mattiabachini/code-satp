import numpy as np
import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, WeightedRandomSampler

# Reuse core utilities
from .multilabel_utils import MultiLabelDataset, compute_metrics, train_transformer_model

# Local focal loss implementation (to avoid importing from a hyphenated folder)
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

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
        metric_for_best_model='eval_micro_f1',
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
        compute_metrics=lambda x: compute_metrics(x, label_cols, context_label="Validation (tuning)"),
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
        metric_for_best_model='eval_micro_f1',
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
        compute_metrics=lambda x: compute_metrics(x, label_cols, context_label="Validation (tuning)")
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
            "threshold_tuned",
            "weighted_sampler",  # placeholder uses class_weights trainer for now
            "augmentation_bt",
            # "augmentation_t5",  # enable explicitly when needed
        ]

    per_strategy_reports = {}
    all_predictions = []

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
            label_cols = [c for c in df_train.columns if c != "incident_summary"]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(label_cols), problem_type="multi_label_classification"
            )
            train_ds = MultiLabelDataset(df_train["incident_summary"].tolist(), df_train[label_cols].values, tokenizer, max_len)
            val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
            test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)

            args = TrainingArguments(
                output_dir="./results", eval_strategy="epoch", save_strategy="epoch",
                learning_rate=2e-5, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs, weight_decay=0.01, logging_dir="./logs", logging_steps=10,
                load_best_model_at_end=True, metric_for_best_model='eval_micro_f1', greater_is_better=True,
                save_total_limit=2, report_to="none", seed=seed, data_seed=seed,
            )

            focal = FocalLoss(alpha=1.0, gamma=2.0)

            class FocalTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    labels = inputs.pop("labels")
                    outputs = model(**inputs)
                    logits = outputs.logits
                    loss = focal(logits, labels)
                    return (loss, outputs) if return_outputs else loss

            trainer = FocalTrainer(
                model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                compute_metrics=lambda x: compute_metrics(x, label_cols, context_label="Validation (tuning)")
            )
            trainer.train()

            # Per-label macro-F1 tuned variant only
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            val_out = _predict_silent(trainer, val_ds)
            test_out = _predict_silent(trainer, test_ds)
            # Print pre-tuning test report @0.5
            _print_pre_tuning_test_report(test_out, label_cols, header="Pre-tuning test (threshold=0.5)")
            val_probs = 1/(1+np.exp(-val_out.predictions))
            test_probs = 1/(1+np.exp(-test_out.predictions))
            th = choose_thresholds_per_label(val_probs, val_out.label_ids.astype(int))
            test_true = test_out.label_ids.astype(int)
            test_pred = apply_thresholds(test_probs, th)

            from sklearn.metrics import classification_report, f1_score, hamming_loss, accuracy_score
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
            per_strategy_reports["focal_tuned"] = metrics

            # Save tuned predictions for PR curves
            focal_tuned_df = pd.DataFrame()
            for i, col in enumerate(label_cols):
                focal_tuned_df[f"true_{col}"] = test_true[:, i]
                focal_tuned_df[f"pred_{col}"] = test_pred[:, i]
                focal_tuned_df[f"prob_{col}"] = test_probs[:, i]
            focal_tuned_df["incident_summary"] = df_test["incident_summary"].values
            focal_tuned_df["strategy"] = "focal_tuned"
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
            # Tuned variant — reuse tokenizer; compute predictions through trainer.predict
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Silence prints during tuning and evaluation by using _predict_silent inside tuned_metrics
            cw_tuned = tuned_metrics_from_trainer(
                trainer, tokenizer, df_val, df_test, label_cols, max_len=max_len, objective="macro"
            )
            per_strategy_reports["class_weights_tuned"] = cw_tuned
            # Save tuned predictions for PR curves
            # Recompute test probs via trainer.predict for consistency
            val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
            test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)
            val_out = _predict_silent(trainer, val_ds)
            test_out = _predict_silent(trainer, test_ds)
            val_probs = 1/(1+np.exp(-val_out.predictions))
            test_probs = 1/(1+np.exp(-test_out.predictions))
            th = choose_thresholds_per_label(val_probs, val_out.label_ids.astype(int))
            test_true = test_out.label_ids.astype(int)
            test_pred = apply_thresholds(test_probs, th)
            cw_tuned_df = pd.DataFrame()
            for i, col in enumerate(label_cols):
                cw_tuned_df[f"true_{col}"] = test_true[:, i]
                cw_tuned_df[f"pred_{col}"] = test_pred[:, i]
                cw_tuned_df[f"prob_{col}"] = test_probs[:, i]
            cw_tuned_df["incident_summary"] = df_test["incident_summary"].values
            cw_tuned_df["strategy"] = "class_weights_tuned"
            all_predictions.append(cw_tuned_df)
            print("[Strategy] class_weights: ✅ completed")
        except Exception as e:
            per_strategy_reports["class_weights"] = None
            print(f"[Strategy] class_weights: ❌ failed — {e}")

    # Weighted sampler (report tuned only)
    if "weighted_sampler" in strategies:
        try:
            print("\n[Strategy] weighted_sampler: starting...")
            trainer, ws_metrics, ws_pred_df = train_with_weighted_sampler(
                model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs, seed=seed
            )
            # Tuned variant — reuse tokenizer and compute predictions via trainer.predict
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            ws_tuned = tuned_metrics_from_trainer(
                trainer, tokenizer, df_val, df_test, label_cols, max_len=max_len, objective="macro"
            )
            per_strategy_reports["weighted_sampler_tuned"] = ws_tuned
            # Save tuned predictions
            val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
            test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)
            val_out = _predict_silent(trainer, val_ds)
            test_out = _predict_silent(trainer, test_ds)
            val_probs = 1/(1+np.exp(-val_out.predictions))
            test_probs = 1/(1+np.exp(-test_out.predictions))
            th = choose_thresholds_per_label(val_probs, val_out.label_ids.astype(int))
            test_true = test_out.label_ids.astype(int)
            test_pred = apply_thresholds(test_probs, th)
            ws_tuned_df = pd.DataFrame()
            for i, col in enumerate(label_cols):
                ws_tuned_df[f"true_{col}"] = test_true[:, i]
                ws_tuned_df[f"pred_{col}"] = test_pred[:, i]
                ws_tuned_df[f"prob_{col}"] = test_probs[:, i]
            ws_tuned_df["incident_summary"] = df_test["incident_summary"].values
            ws_tuned_df["strategy"] = "weighted_sampler_tuned"
            all_predictions.append(ws_tuned_df)
            print("[Strategy] weighted_sampler: ✅ completed")
        except Exception as e:
            per_strategy_reports["weighted_sampler_tuned"] = None
            print(f"[Strategy] weighted_sampler: ❌ failed — {e}")

    # Threshold-tuned (optimize micro-F1 on val, apply to test, then recompute per-label report)
    if "threshold_tuned" in strategies:
        try:
            print("\n[Strategy] threshold_tuned: starting...")
            # Always obtain a trained model to generate calibrated probabilities
            # Train a clean baseline model here and reuse it for threshold selection
            trained_trainer, _, _ = train_transformer_model(
                model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs, seed=seed
            )

            # Build datasets for prediction (aligned with max_len/tokenization)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
            test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)

            # Get validation/test predictions from the trained model (silence prints)
            val_out = _predict_silent(trained_trainer, val_ds)
            test_out = _predict_silent(trained_trainer, test_ds)

            # Print pre-tuning test report @0.5
            _print_pre_tuning_test_report(test_out, label_cols, header="Pre-tuning test (threshold=0.5)")

            val_probs = 1/(1+np.exp(-val_out.predictions))
            val_true = val_out.label_ids.astype(int)
            # Use independent per-label F1 tuning
            th = choose_thresholds_per_label(val_probs, val_true)
            # Apply thresholds to test predictions from the trained model (silence prints)
            test_probs = 1/(1+np.exp(-test_out.predictions))
            test_true = test_out.label_ids.astype(int)
            test_pred = apply_thresholds(test_probs, th)

            from sklearn.metrics import classification_report
            report = classification_report(test_true, test_pred, target_names=label_cols, zero_division=0, output_dict=True)
            # micro_f1 for consistency
            from sklearn.metrics import f1_score, hamming_loss, accuracy_score
            micro_f1 = f1_score(test_true, test_pred, average="micro", zero_division=0)
            metrics = {
                "hamming_loss": hamming_loss(test_true, test_pred),
                "subset_accuracy": accuracy_score(test_true, test_pred),
                "micro_f1": micro_f1,
            }
            metrics.update(report)
            per_strategy_reports["threshold_tuned"] = metrics

            # Also persist probabilities for PR curves
            th_pred_df = pd.DataFrame()
            for i, col in enumerate(label_cols):
                th_pred_df[f"true_{col}"] = test_true[:, i]
                th_pred_df[f"pred_{col}"] = test_pred[:, i]
                th_pred_df[f"prob_{col}"] = test_probs[:, i]
            th_pred_df["incident_summary"] = df_test["incident_summary"].values
            th_pred_df["strategy"] = "threshold_tuned"
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
                # Tuned variant
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                aug_bt_tuned = tuned_metrics_from_trainer(
                    trainer, tokenizer, df_val, df_test, label_cols, max_len=max_len, objective="macro"
                )
                per_strategy_reports["augmentation_bt_tuned"] = aug_bt_tuned
                # Save tuned predictions
                val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
                test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)
                val_out = _predict_silent(trainer, val_ds)
                test_out = _predict_silent(trainer, test_ds)
                val_probs = 1/(1+np.exp(-val_out.predictions))
                test_probs = 1/(1+np.exp(-test_out.predictions))
                th = choose_thresholds_per_label(val_probs, val_out.label_ids.astype(int))
                test_true = test_out.label_ids.astype(int)
                test_pred = apply_thresholds(test_probs, th)
                aug_bt_tuned_df = pd.DataFrame()
                for i, col in enumerate(label_cols):
                    aug_bt_tuned_df[f"true_{col}"] = test_true[:, i]
                    aug_bt_tuned_df[f"pred_{col}"] = test_pred[:, i]
                    aug_bt_tuned_df[f"prob_{col}"] = test_probs[:, i]
                aug_bt_tuned_df["incident_summary"] = df_test["incident_summary"].values
                aug_bt_tuned_df["strategy"] = "augmentation_bt_tuned"
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
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                aug_t5_tuned = tuned_metrics_from_trainer(
                    trainer, tokenizer, df_val, df_test, label_cols, max_len=max_len, objective="macro"
                )
                per_strategy_reports["augmentation_t5_tuned"] = aug_t5_tuned
                # Save tuned predictions
                val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
                test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)
                val_out = _predict_silent(trainer, val_ds)
                test_out = _predict_silent(trainer, test_ds)
                val_probs = 1/(1+np.exp(-val_out.predictions))
                test_probs = 1/(1+np.exp(-test_out.predictions))
                th = choose_thresholds_per_label(val_probs, val_out.label_ids.astype(int))
                test_true = test_out.label_ids.astype(int)
                test_pred = apply_thresholds(test_probs, th)
                aug_t5_tuned_df = pd.DataFrame()
                for i, col in enumerate(label_cols):
                    aug_t5_tuned_df[f"true_{col}"] = test_true[:, i]
                    aug_t5_tuned_df[f"pred_{col}"] = test_pred[:, i]
                    aug_t5_tuned_df[f"prob_{col}"] = test_probs[:, i]
                aug_t5_tuned_df["incident_summary"] = df_test["incident_summary"].values
                aug_t5_tuned_df["strategy"] = "augmentation_t5_tuned"
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


