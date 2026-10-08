"""
SATP: classify every event's action types with the original DistilBERT model, locally.

Runs the multi-label action-type model from hugging-face-hosting-inference/action_type
over all event texts and saves its seven probabilities per event. It needs no API and no
credit. A flag is a probability above THRESHOLD (0.5 in the original app).

The model was trained on events only, so it has no "not an event" answer: policy news and
statements can still receive a flag, or no flag at all.

Input:   scraped_data/satp_event_texts.csv   (code/split_multi_incident_entries.R)
Output:  scraped_data/satp_events_actions_local.csv   (event_uid and p_<action> columns)

Run from the repo's virtual environment:
  .venv/bin/python code/classify_actions_local.py
Author: Mattia Bachini
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp")
EVENTS_FILE = REPO / "scraped_data" / "satp_event_texts.csv"
MODEL_DIR = REPO / "hugging-face-hosting-inference" / "action_type" / "distilbert_model"
OUT_FILE = REPO / "scraped_data" / "satp_events_actions_local.csv"

# the order of the model's seven outputs, as in hugging-face-hosting-inference/app.py
ACTIONS = ["armed_assault", "arrest", "bombing", "infrastructure", "surrender", "seizure", "abduction"]

THRESHOLD = 0.5
BATCH_SIZE = 64
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def predict_probabilities(texts):
    """Array (events x 7): the probability the model gives each action."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    probabilities = []
    for number, start in enumerate(range(0, len(texts), BATCH_SIZE)):
        batch = tokenizer(texts[start:start + BATCH_SIZE], padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            probabilities.append(torch.sigmoid(model(**batch).logits).cpu().numpy())
        if number % 50 == 0:
            print(f"  {min(start + BATCH_SIZE, len(texts))} of {len(texts)} predicted")
    return np.vstack(probabilities)


events = pd.read_csv(EVENTS_FILE, usecols=["event_uid", "event_text"])
events["event_text"] = events["event_text"].fillna("")
print(f"{len(events)} events, running on {DEVICE}")

# texts of similar length share a batch, which saves padding and time
order = events["event_text"].str.len().argsort().to_numpy()
probabilities = predict_probabilities(events["event_text"].iloc[order].tolist())

result = pd.DataFrame(probabilities, columns=[f"p_{a}" for a in ACTIONS])
result.insert(0, "event_uid", events["event_uid"].iloc[order].to_numpy())
result = result.sort_values("event_uid").round(4)
result.to_csv(OUT_FILE, index=False)

flags = result[[f"p_{a}" for a in ACTIONS]] > THRESHOLD
print(f"\nSaved {len(result)} events to {OUT_FILE.name}")
print(f"Events with at least one flag: {flags.any(axis=1).mean():.3f}")
print("Share of events with each flag:")
print(flags.mean().rename(lambda name: name.removeprefix("p_")).round(3).to_string())
