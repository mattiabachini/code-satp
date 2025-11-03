# Count Extraction Models

This directory contains notebooks and utilities for extracting count information (deaths, injuries, arrests, etc.) from conflict event descriptions using transformer models.

## Structure

```
count-models/
├── utils/                          # Shared utility functions
│   ├── __init__.py                # Module exports
│   ├── extraction_utils.py        # Number extraction from text
│   ├── metrics_utils.py           # Evaluation metrics
│   ├── data_utils.py              # Data preparation & tokenization
│   ├── model_utils.py             # Custom model architectures
│   └── file_io.py                 # File I/O utilities
├── death-count-extraction.ipynb   # Main notebook comparing 5 models
├── deaths-poisson.ipynb           # Poisson regression experiments
├── satp_gemma_counts.ipynb        # Gemma-based count extraction
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Notebooks

### death-count-extraction.ipynb

**Purpose**: Compare 5 different approaches for extracting death counts from Armed Assault and Bombing incidents.

**Models Evaluated**:
1. **NT5-small** - Numerical reasoning specialist
2. **Flan-T5-base** - Instruction-following generalist
3. **IndicBART** - India-specific multilingual model
4. **ConfliBERT-QA** - Conflict domain expert (placeholder)
5. **DistilBERT-Poisson** - Regression baseline with Poisson loss

**Dataset**: ~4,300 Armed Assault + Bombing incidents
- 37% with zero fatalities
- 63% with fatalities (mean: ~2.5, max: varies)

**Key Features**:
- Stratified train/val/test split (60%/20%/20%)
- Comprehensive metrics (MAE, RMSE, Exact Match, Within-K, Zero-class F1)
- Model comparison visualizations
- Error analysis by count bins

### deaths-poisson.ipynb

Experiments with Poisson regression models for count prediction.

### satp_gemma_counts.ipynb

Uses Gemma 2B model with prompt engineering to extract multiple count types (deaths, injuries, surrenders, arrests, abductions).

## Setup

### Google Colab (Recommended)

The notebooks are designed to work out-of-the-box on Colab:

1. Open the notebook in Colab
2. Run the first "Colab Setup" cell - it will:
   - Mount Google Drive
   - Clone the GitHub repository
   - Install all dependencies
   - Set up result directories
   - Add utils to Python path

Results automatically save to: `/content/drive/MyDrive/colab/satp-results/death-counts/`

### Local Development

1. Navigate to the count-models directory:
```bash
cd models/count-models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. In the notebook, comment out the "Colab Setup" cell and uncomment the "Local Setup" cell

4. Run the notebook - results save to `./results/`

## Utils Module

### extraction_utils.py

**Functions**:
- `extract_number(text)` - Extract numeric count from text using multiple strategies (direct parsing, regex, word-to-number conversion)
- `parse_prediction(raw_output, model_type)` - Parse model output to numeric count based on model type

### metrics_utils.py

**Functions**:
- `compute_metrics(predictions, labels, extraction_success=None)` - Calculate comprehensive evaluation metrics:
  - MAE, RMSE, MdAE
  - Exact match, Within-1, Within-2
  - Zero-class precision, recall, F1
  - Non-zero MAE
  - Extraction success rate
- `print_metrics(metrics, model_name)` - Pretty print metrics

### data_utils.py

**Functions**:
- `prepare_seq2seq_data(df, model_type)` - Format data for seq2seq models with appropriate prompts
- `prepare_regression_data(df)` - Format data for regression models
- `tokenize_seq2seq(examples, tokenizer, ...)` - Tokenize seq2seq inputs and targets
- `tokenize_for_regression(examples, tokenizer, ...)` - Tokenize regression inputs

### model_utils.py

**Classes**:
- `PoissonRegressionModel` - DistilBERT encoder + regression head with Poisson NLL loss

### file_io.py

**Functions**:
- `get_base_results_dir()` - Detect and return base results directory
- `get_task_results_dir(task_name)` - Get task-specific results directory
- `save_dataframe_csv(df, filename, task_name)` - Save DataFrame to results
- `load_dataframe_csv(filename, task_name)` - Load DataFrame from results

**Directory Priority**:
1. `SATP_RESULTS_DIR` environment variable
2. Google Drive on Colab: `/content/drive/MyDrive/colab/satp-results`
3. Colab local: `/content/satp-results`
4. Local workspace: `./results`

## Usage Examples

### Using Extraction Utils

```python
from utils import extract_number, parse_prediction

# Extract numbers from text
output = "5 people were killed"
count = extract_number(output)  # Returns: 5

# Parse model predictions
seq2seq_output = "The count is 12"
count = parse_prediction(seq2seq_output, model_type='seq2seq')  # Returns: 12

regression_output = 8.7
count = parse_prediction(regression_output, model_type='regression')  # Returns: 9
```

### Using Metrics Utils

```python
from utils import compute_metrics, print_metrics
import numpy as np

predictions = np.array([0, 1, 2, 5, 10])
labels = np.array([0, 1, 3, 5, 8])

metrics = compute_metrics(predictions, labels)
print_metrics(metrics, "My Model")
```

### Using Data Utils

```python
from utils import prepare_seq2seq_data, tokenize_seq2seq
from transformers import AutoTokenizer

# Prepare data for NT5
train_data = prepare_seq2seq_data(train_df, model_type='nt5')

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("nielsr/nt5-small-rc1")
tokenized = tokenize_seq2seq(train_data, tokenizer)
```

### Using File I/O

```python
from utils.file_io import save_dataframe_csv, get_task_results_dir

# Save results
save_dataframe_csv(results_df, 'model_comparison.csv', task_name='death-counts')

# Get results directory
results_dir = get_task_results_dir('death-counts')
print(f"Results saved to: {results_dir}")
```

## Model Training

The main notebook follows this workflow:

1. **Data Management** - Load, filter, split data with stratification
2. **Model Training** - Train each model sequentially:
   - Prepare data in model-specific format
   - Tokenize inputs and targets
   - Configure training arguments
   - Train with early stopping
   - Generate predictions on test set
   - Compute and store metrics
   - Clear GPU memory
3. **Results Analysis** - Compare models, visualize performance, analyze errors

## Results

Results are saved in the task-specific results directory:

- `train.csv`, `val.csv`, `test.csv` - Data splits
- `armed_assault_bombing.csv` - Full filtered dataset
- `death_counts_metrics.csv` - Model comparison metrics
- `death_counts_predictions.csv` - All model predictions
- `figures/` - Visualization outputs
- `model_checkpoints/` - Trained model checkpoints

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `transformers>=4.35.0` - HuggingFace Transformers
- `datasets>=2.14.0` - HuggingFace Datasets
- `torch>=2.0.0` - PyTorch
- `pandas`, `numpy`, `scikit-learn` - Data processing
- `matplotlib`, `seaborn` - Visualization

## Extending to Other Count Types

To create notebooks for other count types (injuries, arrests, etc.):

1. Copy `death-count-extraction.ipynb`
2. Update the task name: `TASK_NAME = "injury-counts"`
3. Update the target column: `'total_injuries'` instead of `'total_fatalities'`
4. Update filtering logic if needed (different action types, etc.)
5. The utils will work the same way!

## Notes

- The ConfliBERT-QA model is a placeholder - full QA implementation requires span-based annotations
- GPU memory is cleared after each model to prevent OOM errors
- Results use consistent file_io conventions across all SATP notebooks
- Follows the same structure as classification-models and location-models

## Related Directories

- `../classification-models/` - Action type, perpetrator, target classification
- `../location-models/` - Location extraction models
- `../../data/` - Source data files

## Contact

For questions or issues, refer to the main project README or open an issue on GitHub.

