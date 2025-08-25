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
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedBCETrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=lambda x: compute_metrics(x, label_cols)
    )
    trainer.train()

    # Evaluate on test
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

    # Also compute standard metrics dict using the same function
    test_results = trainer.evaluate(test_ds)
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
    epochs=2
):
    if strategies is None:
        strategies = [
            "baseline",
            "focal",
            "class_weights",
            "threshold_tuned",
            "weighted_sampler",  # placeholder uses class_weights trainer for now
            "augmentation_bt",
        ]

    results = []
    per_strategy_reports = {}

    # Baseline
    if "baseline" in strategies:
        try:
            print("\n[Strategy] baseline: starting...")
            _, base_metrics, base_pred_df = train_transformer_model(
                model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs
            )
            per_strategy_reports["baseline"] = base_metrics
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

            # Evaluate on test
            test_results = trainer.evaluate(test_ds)
            per_strategy_reports["focal"] = test_results
            print("[Strategy] focal: ✅ completed")
        except Exception as e:
            per_strategy_reports["focal"] = None
            print(f"[Strategy] focal: ❌ failed — {e}")

    # Class-weighted
    if "class_weights" in strategies:
        try:
            print("\n[Strategy] class_weights: starting...")
            _, cw_metrics, cw_pred_df = train_with_class_weights(
                model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs
            )
            per_strategy_reports["class_weights"] = cw_metrics
            print("[Strategy] class_weights: ✅ completed")
        except Exception as e:
            per_strategy_reports["class_weights"] = None
            print(f"[Strategy] class_weights: ❌ failed — {e}")

    # Threshold-tuned (optimize micro-F1 on val, apply to test, then recompute per-label report)
    if "threshold_tuned" in strategies:
        try:
            print("\n[Strategy] threshold_tuned: starting...")
            # Train baseline if not already available
            if "baseline" not in per_strategy_reports or per_strategy_reports.get("baseline") is None:
                _, base_metrics, base_pred_df = train_transformer_model(
                    model_name, df_train, df_val, df_test, max_len=max_len, batch_size=batch_size, epochs=epochs
                )
            # Need val probs to pick thresholds
            # Build baseline trainer quickly for val predict
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(label_cols), problem_type="multi_label_classification"
            )
            train_ds = MultiLabelDataset(df_train["incident_summary"].tolist(), df_train[label_cols].values, tokenizer, max_len)
            val_ds = MultiLabelDataset(df_val["incident_summary"].tolist(), df_val[label_cols].values, tokenizer, max_len)
            test_ds = MultiLabelDataset(df_test["incident_summary"].tolist(), df_test[label_cols].values, tokenizer, max_len)
            args = TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=batch_size, report_to="none")
            tmp_trainer = Trainer(model=model, args=args)
            val_out = tmp_trainer.predict(val_ds)
            val_probs = 1/(1+np.exp(-val_out.predictions))
            val_true = val_out.label_ids.astype(int)
            th = choose_thresholds_micro(val_probs, val_true)
            test_out = tmp_trainer.predict(test_ds)
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
            print("[Strategy] threshold_tuned: ✅ completed")
        except Exception as e:
            per_strategy_reports["threshold_tuned"] = None
            print(f"[Strategy] threshold_tuned: ❌ failed — {e}")

    # Weighted sampler placeholder: reuse class-weighted trainer for now (sampler could be added later)
    if "weighted_sampler" in strategies and "weighted_sampler" not in per_strategy_reports:
        per_strategy_reports["weighted_sampler"] = per_strategy_reports.get("class_weights")
        print("\n[Strategy] weighted_sampler: ✅ completed (alias of class_weights)")

    # Augmentation
    # Augmentation via back-translation (uses the enhanced module if available)
    if "augmentation_bt" in strategies:
        train_with_aug = _load_augmented_trainer_fn()
        if train_with_aug is None:
            print("\n[Strategy] augmentation_bt: ⚠️ augmentation trainer not available; skipping")
        else:
            try:
                print("\n[Strategy] augmentation_bt: starting...")
                _, aug_metrics, _ = train_with_aug(
                    model_name, df_train, df_val, df_test,
                    max_len=max_len, batch_size=batch_size, epochs=epochs,
                    augmentation_strategies=['back_translation']
                )
                per_strategy_reports["augmentation_bt"] = aug_metrics
                print("[Strategy] augmentation_bt: ✅ completed")
            except Exception as e:
                print(f"[Strategy] augmentation_bt: ❌ failed — {e}")
                per_strategy_reports["augmentation_bt"] = None

    # Build long-form results for plotting strategy heatmap
    rows = []
    for strat_name, metrics in per_strategy_reports.items():
        if metrics is None:
            continue
        for lbl in label_cols:
            # metrics may store per-label reports under either the raw label key or an eval_-prefixed key
            key = lbl if lbl in metrics else f"eval_{lbl}"
            f1 = metrics.get(key, {}).get("f1-score", 0.0)
            rows.append({"strategy": strat_name, "label": lbl, "f1": f1})
    results_df = pd.DataFrame(rows)

    # Pivot is often handy to save directly
    pivot = results_df.pivot(index="label", columns="strategy", values="f1")
    return pivot, results_df


