Date: 2024-09-01

# code-satp

SATP Conflict Data Pipeline: scrape, auto-code and visualise conflict events from the
South Asia Terrorism Portal.

## Overview

This repository contains data and code for a data pipeline that scrapes, auto-codes and visualizes data from the South Asia Terrorism Portal (SATP). The auto-coding is done by fine-tuning LLMs with a hand-coded set of approximately 10,000 conflict event descriptions.

The objective is to fine-tune a series of models that can used to code new events. The events break down into a few key categories:

1) The "perpetrator" or assailant (who took the action);
2) Action type (what was done);
3) Target (who was affected);
4) Casualties (how many people were killed or injured);
5) Location (where did the event take place);
6) Miscellaneous other information (e.g. property damage, kidnappings, surrenders, etc.)

## Folder structure

    .
    ├── code
    │   ├── build        source ingestion: raw scrape + hand-coded workbook
    │   ├── process      cross-source processing: split, match, code
    │   ├── analyze      validation, EDA, model comparison
    │   └── sandbox      scratch/experimental; not part of the pipeline
    ├── data
    │   ├── input        RAW inputs, read-only
    │   │   ├── satp_scrape      scraped_data-1/2/3.csv
    │   │   ├── satp_handcoded   satp-dataset.xlsx
    │   │   └── external         geonames, osm, lgd, shrug
    │   ├── intermediate derived and regenerable
    │   │   ├── scraped     merged scrape and event texts
    │   │   ├── coded       LLM coding outputs and caches
    │   │   ├── datasets    per-task CSVs (perpetrator, action_type, ...)
    │   │   ├── places      place extraction and district matching
    │   │   ├── gazetteers  geonames / OSM / SHRID reference tables
    │   │   ├── validation  hand checks and model comparisons
    │   │   └── models      model outputs
    │   └── final        analysis-ready datasets
    ├── documentation    coding manuals and pipeline diagram
    ├── models           classification, location and count notebooks
    ├── papers           Quarto papers
    ├── presentations    Quarto presentations
    ├── streamlit-app    SATP scraper (Streamlit Cloud)
    ├── hugging-face-hosting-inference
    ├── village-matching fuzzy place-name matching
    ├── notes            working notes and codebook
    └── old              retired notebooks, scripts, backups and lockfiles

### Data rules

- **`data/input/` is raw and read-only.** Never overwrite anything in it. It holds the
  hand-coded workbook, the three raw scrape files, and the external geodata downloads.
- **`data/intermediate/` is derived.** Everything here can be regenerated from
  `data/input/` and `code/`.
- **`data/final/` is analysis-ready.** `satp_clean.csv` (cleaned hand-coded events) and
  `analysis_set_events.rds` (the analysis set) live here.
- The data tree is mostly git-ignored (large inputs and derived files); the small model
  CSVs that were already tracked stay tracked.

## Pipeline order

1. `code/build/scrape_data_merge.R` — merge `data/input/satp_scrape/*.csv`, dedupe on
   `(date, summary)` → `data/intermediate/scraped/scraped_data.rds` and `scraped_incidents.*`.
2. `code/build/wrangle_satp.qmd` — clean the hand-coded workbook
   (`data/input/satp_handcoded/satp-dataset.xlsx`) → `data/final/satp_clean.csv`.
3. `code/process/select_vars_for_models.qmd` — split `satp_clean.csv` into per-task
   CSVs in `data/intermediate/datasets/`.
4. `code/process/split_multi_incident_entries.R` — split multi-incident entries into
   events → `data/intermediate/scraped/satp_event_texts.*`.
5. `code/process/match_event_districts.R` and `district_gazetteer_baseline.R` — attach
   2001 district units.
6. `code/process/code_events_deepseek.R` and `classify_actions_local.py` — LLM and
   DistilBERT coding → `data/intermediate/coded/`.
7. `code/analyze/*` — validation samples, EDA and model comparisons.

## Data

The original hand-coded data is in `data/input/satp_handcoded/satp-dataset.xlsx`. It has
been partially wrangled in R (see `code/build/wrangle_satp.qmd`). The cleaned data is in
`data/final/satp_clean.csv`, which is then used to generate the CSV files for each model
we want to fine-tune, e.g. `data/intermediate/datasets/perpetrator.csv`,
`data/intermediate/datasets/action_type.csv`, etc.

The raw scrape (2000–2025, all five SATP series) is in
`data/input/satp_scrape/scraped_data-1/2/3.csv`. The merged, deduplicated incident set is
`data/intermediate/scraped/scraped_incidents.csv` (127,466 unique incidents).

The data for location and the miscellaneous other categories like kidnappings, surrenders, etc. have not yet been hived off into separate files.

## Models

So far, we have been able to fine-tune a series of models using the Hugging Face `transformers` library. Specifically, we have used BERT-based models to code for the "perpetrator" and "action type" categories.

We also tried fine-tuning a DistilBERT model with a Poisson head for the death count, but this was not very successful. So one next step is to try to fine-tune a Named Entity Recognition (NER) model like GLiNER to get the casualty figures.

The team has discussed using an NER model to also get the location, but there is also a thought that some of this work could be done through the scraping process or third party API services.

The model files could probably be cleaned up and combined in some cases (e.g. multiple perpetrator models), but let's discuss this in person before taking any action.

There was some thought given to using RAG models to do some of the coding. I have not seen a RAG used in this way before, but I do like the idea of including a RAG model in the pipeline. One possibility is to use it to answer questions about the data in the web interface.
