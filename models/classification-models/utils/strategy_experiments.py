import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

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


def tuned_metrics_from_trainer(trainer, tokenizer, df_val, df_test, label_cols, max_len=512, objective="macro"):
    """
    Compute test metrics after tuning per-label thresholds on validation set.
    Uses the provided trained Trainer to generate probabilities.
    """
    from sklearn.metrics import classification_report, f1_score, hamming_loss, accuracy_score

    val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
    test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)

    val_out = trainer.predict(val_ds)
    test_out = trainer.predict(test_ds)
    val_probs = 1 / (1 + np.exp(-val_out.predictions))
    test_probs = 1 / (1 + np.exp(-test_out.predictions))

    th = choose_thresholds(val_probs, val_out.label_ids.astype(int), objective=objective)
    test_true = test_out.label_ids.astype(int)
    test_pred = _apply_thresholds(test_probs, th)

    report = classification_report(test_true, test_pred, target_names=label_cols, zero_division=0, output_dict=True)
    micro_f1 = f1_score(test_true, test_pred, average="micro", zero_division=0)
    metrics = {
        "hamming_loss": hamming_loss(test_true, test_pred),
        "subset_accuracy": accuracy_score(test_true, test_pred),
        "micro_f1": micro_f1,
    }
    metrics.update(report)
    return metrics


def _pos_weight_from_df(df, label_cols):
    P = df[label_cols].sum(axis=0).values.astype(np.float32)
    N = len(df) - P
    P = np.where(P == 0, 1.0, P)
    return torch.tensor((N / P), dtype=torch.float)


def train_with_class_weights(model_name, df_train, df_val, df_test, max_len=512, batch_size=16, epochs=2):
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
        compute_metrics=lambda x: compute_metrics(x, label_cols)
    )
    trainer.train()

    # Evaluate on test: use predict() once to gather both metrics and logits
    predictions_output = trainer.predict(test_ds)
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
    predictions_csv=None
):
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
                model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs
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
                save_total_limit=2, report_to="none",
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
                compute_metrics=lambda x: compute_metrics(x, label_cols)
            )
            trainer.train()

            # Per-label macro-F1 tuned variant only
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            val_out = trainer.predict(val_ds)
            test_out = trainer.predict(test_ds)
            val_probs = 1/(1+np.exp(-val_out.predictions))
            test_probs = 1/(1+np.exp(-test_out.predictions))
            th = choose_thresholds(val_probs, val_out.label_ids.astype(int), objective="macro")
            test_true = test_out.label_ids.astype(int)
            test_pred = apply_thresholds(test_probs, th)

            from sklearn.metrics import classification_report, f1_score, hamming_loss, accuracy_score
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
                model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs
            )
            # Tuned variant — reuse tokenizer; compute predictions through trainer.predict
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            cw_tuned = tuned_metrics_from_trainer(
                trainer, tokenizer, df_val, df_test, label_cols, max_len=max_len, objective="macro"
            )
            per_strategy_reports["class_weights_tuned"] = cw_tuned
            # Save tuned predictions for PR curves
            # Recompute test probs via trainer.predict for consistency
            val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
            test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)
            val_out = trainer.predict(val_ds)
            test_out = trainer.predict(test_ds)
            val_probs = 1/(1+np.exp(-val_out.predictions))
            test_probs = 1/(1+np.exp(-test_out.predictions))
            th = choose_thresholds(val_probs, val_out.label_ids.astype(int), objective="macro")
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

    # Threshold-tuned (optimize micro-F1 on val, apply to test, then recompute per-label report)
    if "threshold_tuned" in strategies:
        try:
            print("\n[Strategy] threshold_tuned: starting...")
            # Always obtain a trained model to generate calibrated probabilities
            # Train a clean baseline model here and reuse it for threshold selection
            trained_trainer, _, _ = train_transformer_model(
                model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs
            )

            # Build datasets for prediction (aligned with max_len/tokenization)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
            test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)

            # Get validation probabilities from the trained model
            val_out = trained_trainer.predict(val_ds)
            val_probs = 1/(1+np.exp(-val_out.predictions))
            val_true = val_out.label_ids.astype(int)
            # Use per-label macro-F1 tuning as requested
            th = choose_thresholds(val_probs, val_true, objective="macro")
            # Apply thresholds to test predictions from the trained model
            test_out = trained_trainer.predict(test_ds)
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

    # Weighted sampler placeholder: mirror class_weights_tuned
    if "weighted_sampler" in strategies and "weighted_sampler" not in per_strategy_reports:
        per_strategy_reports["weighted_sampler_tuned"] = per_strategy_reports.get("class_weights_tuned")
        print("\n[Strategy] weighted_sampler_tuned: ✅ completed (alias of class_weights_tuned)")

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
                val_out = trainer.predict(val_ds)
                test_out = trainer.predict(test_ds)
                val_probs = 1/(1+np.exp(-val_out.predictions))
                test_probs = 1/(1+np.exp(-test_out.predictions))
                th = choose_thresholds(val_probs, val_out.label_ids.astype(int), objective="macro")
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
                val_out = trainer.predict(val_ds)
                test_out = trainer.predict(test_ds)
                val_probs = 1/(1+np.exp(-val_out.predictions))
                test_probs = 1/(1+np.exp(-test_out.predictions))
                th = choose_thresholds(val_probs, val_out.label_ids.astype(int), objective="macro")
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


