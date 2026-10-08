"""
SATP: compare the original DistilBERT action-type model with DeepSeek's action flags.

For every event that DeepSeek coded as a real event, the original multi-label DistilBERT
model (hugging-face-hosting-inference/action_type/distilbert_model) predicts the same seven
action flags. The two sets of flags are compared overall and separately for the region and
years the DistilBERT model was trained on (Maoist series, 2005 to 2016) and for everything else.
DeepSeek is the reference, so the numbers measure agreement, not accuracy against the truth.

Input:   scraped_data/satp_events_coded.csv   (code/code_events_deepseek.R)
Outputs: scraped_data/action_model_comparison.csv     (agreement per action and group)
         scraped_data/action_model_disagreements.csv  (events where the two differ, to read)

Run from the repo's virtual environment: .venv/bin/python code/compare_action_models.py
Author: Mattia Bachini
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp")
CODED_FILE = REPO / "scraped_data" / "satp_events_coded.csv"
MODEL_DIR = REPO / "hugging-face-hosting-inference" / "action_type" / "distilbert_model"
OUT_TABLE = REPO / "scraped_data" / "action_model_comparison.csv"
OUT_DISAGREE = REPO / "scraped_data" / "action_model_disagreements.csv"

# the order of the model's seven outputs, as in hugging-face-hosting-inference/app.py
ACTIONS = ["armed_assault", "arrest", "bombing", "infrastructure", "surrender", "seizure", "abduction"]
ANALYSIS_ACTIONS = ["armed_assault", "bombing", "infrastructure", "abduction"]

THRESHOLD = 0.5
BATCH_SIZE = 64
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def predict_actions(texts):
    """Boolean array (events x 7): the model's action flags."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    probabilities = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = tokenizer(texts[start:start + BATCH_SIZE], padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            probabilities.append(torch.sigmoid(model(**batch).logits).cpu().numpy())
        print(f"  {min(start + BATCH_SIZE, len(texts))} of {len(texts)} predicted")
    return np.vstack(probabilities) > THRESHOLD


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else np.nan


def compare(group, deepseek, model):
    """One row per action: how often the model's flag matches DeepSeek's flag."""
    rows = []
    flags = {a: (deepseek[:, j], model[:, j]) for j, a in enumerate(ACTIONS)}
    flags["any_analysis_action"] = (deepseek[:, [ACTIONS.index(a) for a in ANALYSIS_ACTIONS]].any(axis=1),
                                    model[:, [ACTIONS.index(a) for a in ANALYSIS_ACTIONS]].any(axis=1))
    for action, (d, m) in flags.items():
        true_pos, false_pos, false_neg = (d & m).sum(), (~d & m).sum(), (d & ~m).sum()
        precision, recall = ratio(true_pos, true_pos + false_pos), ratio(true_pos, true_pos + false_neg)
        rows.append({"group": group, "action": action, "events": len(d),
                     "deepseek_positive": d.sum(), "model_positive": m.sum(),
                     "agreement": (d == m).mean(),
                     "model_precision": precision, "model_recall": recall,
                     "f1": ratio(2 * precision * recall, precision + recall)})
    exact = (deepseek == model).all(axis=1).mean()
    rows.append({"group": group, "action": "all_seven_match", "events": len(deepseek), "agreement": exact})
    return rows


events = pd.read_csv(CODED_FILE)
events = events[events["is_event"] == True].reset_index(drop=True)
print(f"{len(events)} events coded by DeepSeek")

deepseek = events[ACTIONS].to_numpy().astype(bool)
model = predict_actions(events["event_text"].tolist())

in_scope = (events["series"].str.contains("india-maoistinsurgency", na=False)
            & pd.to_datetime(events["event_date"]).dt.year.between(2005, 2016))
groups = {"all events": np.ones(len(events), dtype=bool),
          "Maoist series 2005-2016 (training scope)": in_scope.to_numpy(),
          "everything else": ~in_scope.to_numpy()}

table = pd.DataFrame([row for name, mask in groups.items()
                      for row in compare(name, deepseek[mask], model[mask])])
table.to_csv(OUT_TABLE, index=False)
print(table.round(3).to_string(index=False))

disagree = events[(deepseek != model).any(axis=1)].copy()
for j, action in enumerate(ACTIONS):
    disagree[f"model_{action}"] = model[(deepseek != model).any(axis=1), j].astype(int)
disagree[["event_uid", "series", "event_text", *ACTIONS, *[f"model_{a}" for a in ACTIONS]]] \
    .to_csv(OUT_DISAGREE, index=False)
print(f"\n{len(disagree)} events with at least one different flag, written to {OUT_DISAGREE.name}")
