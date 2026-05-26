# LLM-EGT Forecaster

PyTorch implementation of a news-driven time series forecasting framework with multiple LLM agents and competition-driven evolutionary training.

The framework combines:

- multi-agent LLM forecasting,
- news-aware signal selection,
- LoRA/PEFT fine-tuning,
- evolutionary loss terms for strategy, diversity, pruning, and prediction quality.

## Installation

Clone the repository and enter the project root:

```bash
git clone https://github.com/your-username/llm-egt-forecaster.git
cd llm-egt-forecaster
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

If you use `GPU_LLAMA`, make sure you have access to the configured Llama model. If the model is hosted on Hugging Face and requires access approval, run:

```bash
huggingface-cli login
```

## Configuration

Edit the active configuration file:

```text
llm_egt_forecaster/configs/base_config.py
```

Important options:

```python
MODE = "GPU_LLAMA"          # or "CPU_DEBUG" for quick local checks
NUM_AGENTS = 3              # number of LLM agents
BATCH_SIZE = 4
NUM_EPOCHS = 3
NEWS_SELECTOR_METHOD = "cosine"  # "cosine" or "api"
```

Model and tokenizer paths can be overridden without editing source code:

```bash
export BASE_LLM_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
export TOKENIZER_PATH="$BASE_LLM_MODEL"
```

For API-based news selection, set an OpenAI-compatible key via environment variable. Do not hard-code secrets in the config file.

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_API_BASE="https://api.openai.com/v1"   # optional
export OPENAI_API_MODEL="gpt-4o-mini"                # optional
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_API_BASE="https://api.openai.com/v1"
$env:OPENAI_API_MODEL="gpt-4o-mini"
```

The default `cosine` mode does not require an API key.

## Data Preparation

Training expects a JSON dataset in the package data directory:

```text
llm_egt_forecaster/data/real_dataset_enhanced.json
```

If this file is not present, the training script falls back to:

```text
llm_egt_forecaster/data/real_dataset.json
```

Each sample should contain:

```json
{
  "time_series": [1.0, 1.2, 1.4],
  "candidate_news": [
    {
      "summary": "Short news summary.",
      "publication_time": "2024-01-01 10:00:00",
      "category": "News"
    }
  ],
  "ground_truth": [2.1, 2.3, 2.5],
  "metadata": {
    "date": "2024-01-01",
    "region": "NSW"
  }
}
```

The loader uses `time_series`, `candidate_news`, and `ground_truth`. `metadata` is optional but useful for logging and evaluation.

## Training

Run training from the repository root:

```bash
python llm_egt_forecaster/train.py
```

The training script:

- reads configuration from `llm_egt_forecaster/configs/base_config.py`,
- loads `llm_egt_forecaster/data/real_dataset_enhanced.json` or falls back to `real_dataset.json`,
- splits data into 80% train, 10% validation, and 10% test,
- saves checkpoints under `checkpoints/`,
- prints final test metrics after loading the best checkpoint.

To run in the background on a Linux server:

```bash
mkdir -p llm_egt_forecaster/log
nohup python -u llm_egt_forecaster/train.py > llm_egt_forecaster/log/train.out 2>&1 &
tail -f llm_egt_forecaster/log/train.out
```

## Checkpoint Evaluation

Evaluate a saved checkpoint on the held-out 2026 test set:

```bash
python llm_egt_forecaster/evaluate_checkpoint.py \
  --checkpoint checkpoints/best_model.pt \
  --batch-size 4 \
  --split all \
  --news-selector-method cosine \
  --output-json eval_outputs/eval_2026_100_metrics.json \
  --save-predictions eval_outputs/eval_2026_100_predictions.json
```

By default, `evaluate_checkpoint.py` uses:

```text
llm_egt_forecaster/data/real_dataset_100_2026_test.json
```

This 100-sample 2026 dataset is intended as a held-out evaluation set for checking generalization and reducing the risk of evaluating on training-period data. You can still pass `--data-path` to evaluate a different JSON file. Relative paths are resolved from the repository root.

The evaluator reports MSE, RMSE, MAE, and MAPE in the original data scale, plus normalized-space metrics.

It can also save per-sample errors for inspection:

```bash
python llm_egt_forecaster/evaluate_checkpoint.py \
  --checkpoint checkpoints/best_model.pt \
  --split all \
  --sample-errors-json eval_outputs/eval_2026_100_sample_errors.json \
  --sample-errors-log eval_outputs/eval_2026_100_sample_errors.log
```

## Project Structure

```text
llm-egt-forecaster/
|-- llm_egt_forecaster/
|   |-- configs/base_config.py        # Active runtime configuration
|   |-- data/                         # Dataset utilities and local data files
|   |-- src/
|   |   |-- dataset.py                # Dataset and dataloader
|   |   |-- engine/                   # Loss, trainer, and logging
|   |   `-- models/                   # Framework, logic generator, news selector
|   |-- train.py                      # Main training entry point
|   `-- evaluate_checkpoint.py        # Checkpoint evaluation utility
|-- requirements.txt
|-- setup.py
`-- README.md
```

## Notes Before Publishing

Do not commit local artifacts such as:

- `checkpoints/`
- `llm_egt_forecaster/log/`
- `eval_outputs/`
- large dataset files under `llm_egt_forecaster/data/`
- local model directories
- API keys or machine-specific absolute paths

Use environment variables for all secrets.
