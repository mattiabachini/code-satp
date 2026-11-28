# Repository Guidelines

## Project Structure & Module Organization
- `code/`: pipeline code, split by role.
  - `code/build/`: source ingestion — `scrape_data_merge.R` (raw scrape → merged incidents) and `wrangle_satp.qmd` (hand-coded workbook → cleaned data).
  - `code/process/`: cross-source processing — `split_multi_incident_entries.R`, `select_vars_for_models.qmd`, `match_event_districts.R`, `district_gazetteer_baseline.R`, `code_events_deepseek.R`, `classify_actions_local.py`.
  - `code/analyze/`: validation, EDA and model comparison (`compare_*`, `draw_validation_sample.R`, `export_acled_style.R`, `make_analysis_set_input.R`, `satp_EDA.R`, ...).
  - `code/sandbox/`: scratch and experimental scripts; not part of the pipeline.
- `data/`: all data; mostly git-ignored, with the small model CSVs kept tracked.
  - `data/input/`: **raw and read-only** — `satp_scrape/` (the three raw scrape CSVs), `satp_handcoded/` (`satp-dataset.xlsx`), `external/` (geonames, osm, lgd, shrug).
  - `data/intermediate/`: derived, regenerable — `scraped/`, `coded/`, `datasets/`, `places/`, `gazetteers/`, `validation/`, `models/`.
  - `data/final/`: analysis-ready — `satp_clean.csv`, `analysis_set_events.rds`.
- `models/`: model notebooks and utilities.
  - `models/classification-models/`: BERT-family classification notebooks + `utils/` and imbalance handling.
  - `models/location-models/`: T5 seq2seq location extraction notebooks + utilities.
  - `models/count-models/`: Count extraction notebooks + `utils/`.
- `streamlit-app/`: SATP scraping Streamlit app (`app.py`).
- `hugging-face-hosting-inference/`: Streamlit inference app orchestrating models + geocoding.
- `papers/`, `presentations/`: Quarto artifacts and generated assets.
- `village-matching/`: Fuzzy matching for place-name resolution.
- `documentation/`: coding manuals and the pipeline diagram.
- `old/`: retired notebooks, scripts, backups, Excel lock files and planning notes. Nothing here is part of the pipeline.

## Build, Test, and Development Commands
- Create/activate venv and install key deps:
  - `python -m venv venv` then `source venv/bin/activate`
  - `pip install -r models/classification-models/requirements.txt`
  - `pip install -r streamlit-app/requirements.txt`
  - `pip install -r hugging-face-hosting-inference/requirements.txt`
- Run apps locally:
  - `cd streamlit-app && streamlit run app.py`
  - `cd hugging-face-hosting-inference && streamlit run app.py`
- Run the pipeline (see `README.md` for the full order):
  - `Rscript code/build/scrape_data_merge.R`
  - `quarto render code/build/wrangle_satp.qmd`
  - `quarto render code/process/select_vars_for_models.qmd`
  - `Rscript code/process/split_multi_incident_entries.R`

## Coding Style & Naming Conventions
- Python in `utils/` uses 4-space indentation and snake_case; follow the surrounding file style.
- Keep notebook cells focused and reproducible; avoid hard-coded local paths when possible.
- Quarto/R scripts live in `.qmd` files; keep outputs in their existing `papers/` or `presentations/` trees.
- Never write into `data/input/`; derived outputs go to `data/intermediate/` or `data/final/`.

## Testing Guidelines
- There is no automated test suite or CI in this repo; validation is notebook-based.
- When changing model utilities, re-run the relevant notebook and note any metric changes.
- For Streamlit apps, smoke-test by running locally and confirming key pages load.
- After moving or renaming data, check that every script's paths still resolve before committing.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, and lowercase (e.g., "update geocoding script...").
- Include concise context on model/data changes and affected directories.
- PRs should include: summary, how to reproduce (commands/notebooks), and UI screenshots for Streamlit changes.

## Security & Configuration Tips
- External services (Google Sheets, Google Maps API) are configured via Streamlit secrets; do not commit credentials.
- Large model artifacts live under `hugging-face-hosting-inference/`; avoid duplicating binaries unless required.
