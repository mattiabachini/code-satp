# Repository Guidelines

## Project Structure & Module Organization
- `data/`: Source dataset (`satp-dataset.xlsx`), Quarto wrangling scripts (`.qmd`), and per-task CSVs (e.g., `perpetrator.csv`, `action_type.csv`).
- `models/`: Model notebooks and utilities.
  - `models/classification-models/`: BERT-family classification notebooks + `utils/` and imbalance handling.
  - `models/location-models/`: T5 seq2seq location extraction notebooks + utilities.
  - `models/count-models/`: Count extraction notebooks + `utils/`.
- `streamlit-app/`: SATP scraping Streamlit app (`app.py`).
- `hugging-face-hosting-inference/`: Streamlit inference app orchestrating models + geocoding.
- `papers/`, `presentations/`: Quarto artifacts and generated assets.
- `village-matching/`: Fuzzy matching for place-name resolution.

## Build, Test, and Development Commands
- Create/activate venv and install key deps:
  - `python -m venv venv` then `source venv/bin/activate`
  - `pip install -r models/classification-models/requirements.txt`
  - `pip install -r streamlit-app/requirements.txt`
  - `pip install -r hugging-face-hosting-inference/requirements.txt`
- Run apps locally:
  - `cd streamlit-app && streamlit run app.py`
  - `cd hugging-face-hosting-inference && streamlit run app.py`
- Render data wrangling or figures (Quarto):
  - `quarto render data/wrangle_satp.qmd`
  - `quarto render data/select_vars_for_models.qmd`

## Coding Style & Naming Conventions
- Python in `utils/` uses 4-space indentation and snake_case; follow the surrounding file style.
- Keep notebook cells focused and reproducible; avoid hard-coded local paths when possible.
- Quarto/R scripts live in `.qmd` files; keep outputs in their existing `papers/` or `presentations/` trees.

## Testing Guidelines
- There is no automated test suite or CI in this repo; validation is notebook-based.
- When changing model utilities, re-run the relevant notebook and note any metric changes.
- For Streamlit apps, smoke-test by running locally and confirming key pages load.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, and lowercase (e.g., "update geocoding script...").
- Include concise context on model/data changes and affected directories.
- PRs should include: summary, how to reproduce (commands/notebooks), and UI screenshots for Streamlit changes.

## Security & Configuration Tips
- External services (Google Sheets, Google Maps API) are configured via Streamlit secrets; do not commit credentials.
- Large model artifacts live under `hugging-face-hosting-inference/`; avoid duplicating binaries unless required.
