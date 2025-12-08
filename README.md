# Improving LLM Agent Performance via Competition-Driven Evolution in News-Driven Time Series Forecasting: Official PyTorch Implementation

This repository contains the official PyTorch implementation for the paper: *"Improving LLM Agent Performance via Competition-Driven Evolution in News-Driven Time Series Forecasting"*.

Our work introduces a novel multi-agent framework grounded in Evolutionary Game Theory (EGT) to enhance the reasoning and robustness of Large Language Model (LLM) agents for time series forecasting. By simulating a competition-driven evolutionary process, our framework enables agents to self-adaptively develop diverse and effective strategies for filtering news and predicting future trends.

[![Paper Abstract](https://img.shields.io/badge/Paper-Abstract-blue)](https://#) <!-- Replace with actual paper link -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

 <!-- It's highly recommended to add the framework diagram from the paper here -->

---

## 🚀 Core Contributions

-   **Competition-Driven Evolutionary Framework**: We translate core EGT principles—selection, mutation, and adaptation—into a unified, end-to-end differentiable loss function, allowing LLM agents to evolve through standard gradient-based optimization.
-   **Enhanced Agent Capabilities**: The evolutionary dynamics foster strategic diversity and robustness, significantly improving agents' abilities in **multi-source heterogeneity comprehension** and **robust signal discernment**.
-   **Theoretical Soundness**: We provide theoretical arguments for the stability of our system, suggesting the existence of a Bayesian Nash Equilibrium (BNE) and the potential for achieving a sublinear regret bound.

## 🛠️ Installation

Our framework is built using PyTorch and the Hugging Face ecosystem.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/llm-egt-forecaster.git
    cd llm-egt-forecaster
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Our implementation leverages 4-bit quantization via `bitsandbytes` to make large models like Llama-2-7B accessible on consumer-grade GPUs.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Hugging Face Authentication (Required for Llama 2):**
    You will need to have access to Llama 2 models. Please request access on the [Meta Llama 2 model page](https://huggingface.co/meta-llama/Llama-2-7b-hf) and then log in via the command line.
    ```bash
    huggingface-cli login
    ```

## ▶️ Getting Started: Running a Demo

1.  **Prepare your data:** Ensure you have real dataset at `data/real_dataset.json` (see Data Preparation section below).

2.  **Select your mode:** Open `configs/base_config.py` and set the `MODE` variable to either `'GPU_LLAMA'` or `'CPU_DEBUG'` based on your hardware.

3.  **Run the script:**
    ```bash
    python train.py
    ```

The script will load your real dataset and proceed with the training and evolution pipeline.

## ⚙️ Configuration and Environment Modes

Our framework is designed to be adaptable to different hardware environments. You can easily switch between modes by editing the `MODE` variable in `configs/base_config.py`.

### Modes
-   **`GPU_LLAMA` (Default)**: Recommended for users with a CUDA-enabled NVIDIA GPU (>=16GB VRAM recommended). This mode runs the powerful `Llama-2-7b-hf` model with 4-bit quantization for efficient memory usage. It delivers the best performance and is required to fully replicate the paper's results.

-   **`CPU_DEBUG`**: For users without a suitable GPU or for quick debugging. This mode runs the smaller `distilgpt2` model on the CPU. It is functionally identical and perfect for verifying the logic of the evolutionary framework, but the forecasting quality will be lower.

### Core Parameters
In addition to the mode, you can customize:

-   `BASE_LLM_MODEL`: The base language model to use.
-   `NUM_AGENTS`: The number of agents in the population.
-   `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`: Standard training hyperparameters.
-   `LAMBDA_DIVERSITY`, `LAMBDA_PRUNING`: Weights for the evolutionary loss components, controlling the balance between exploration and exploitation.

### News Filtering Configuration
Our framework supports dual-mode news filtering to help agents select relevant news:

-   `NEWS_SELECTOR_METHOD`: Choose between `'cosine'` (local embedding-based) or `'api'` (OpenAI-based) filtering.
-   **Cosine Mode** (Default): Uses semantic similarity with SentenceTransformer models. No API costs.
    -   `NEWS_COSINE_MODEL`: Embedding model (default: `'all-MiniLM-L6-v2'`)
    -   `NEWS_COSINE_THRESHOLD`: Similarity threshold (default: `0.3`)
    -   `NEWS_COSINE_TOP_K`: Number of news to select (default: `3`)
-   **API Mode**: Uses OpenAI GPT for intelligent news selection based on agent logic.
    -   `NEWS_API_KEY`: Set via environment variable `OPENAI_API_KEY`
    -   `NEWS_API_MODEL`: OpenAI model to use (default: `'gpt-3.5-turbo'`)
    -   `NEWS_API_BASE`: Optional custom API base URL

## 📂 Project Structure

The repository is organized as follows for clarity and modularity:

```
llm-egt-forecaster/
├── configs/             # All hyperparameters and configurations
├── data/                # Data generation scripts
│   └── data_generator.py          # Real data loader and converter
├── docs/                # Documentation
│   ├── ARCHITECTURE_CN.md         # Chinese architecture docs
│   └── ARCHITECTURE_EN.md         # English architecture docs
├── src/                 # Main source code
│   ├── agent.py           # Definition of the Agent class
│   ├── dataset.py         # PyTorch Dataset and DataLoader
│   ├── engine/            # Training and loss computation logic
│   │   ├── loss.py        # The core EvolutionaryLoss function
│   │   └── trainer.py     # The main Trainer class
│   └── models/            # Model architecture definitions
│       ├── evolutionary_framework.py # The main multi-agent framework
│       ├── logic_generator.py    # The logic evolution module
│       └── news_selector.py      # The semantic news filtering module
└── train.py             # Main entry point to start training
```

## 📈 Data Preparation

The framework requires real-world data for training.

1.  **Prepare your data:** Create a JSON file with instruction/input/output format:
    ```json
    {
      "instruction": "The historical load data is: 1.2, 1.5, 1.8, ... The region...",
      "input": "Your news text here...",
      "output": "2.1, 2.3, 2.5, 2.7, 2.9"
    }
    ```

2.  **Convert to standard format:** Use the built-in data converter:
    ```bash
    python -m llm_egt_forecaster.data.data_generator
    ```

3.  **Place your data:** Put the converted file as `data/real_dataset.json`

4.  **Launch training:** Run `python train.py`

### Data Format
The framework expects data in this standard format:
```json
{
  "time_series": [1.2, 1.5, 1.8, ...],
  "candidate_news": [
    {
      "summary": "Market sentiment is optimistic...",
      "publication_time": "2024-01-01 09:00:00",
      "category": "Business"
    },
    {
      "summary": "Heavy storms damaged power lines...",
      "publication_time": "2024-01-01 10:30:00",
      "category": "Weather"
    }
  ],
  "ground_truth": [2.1, 2.3, 2.5, 2.7, 2.9]
}
```

**Note**: `candidate_news` should be a list of dictionaries with `summary`, `publication_time`, and `category` fields for optimal news filtering performance.

## 📜 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{your_lastname_2025_improving,
  title={Improving LLM Agent Performance via Competition-Driven Evolution in News-Driven Time Series Forecasting},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## acknowledgements

We would like to thank the developers of PyTorch, Hugging Face, and the PEFT library for their invaluable contributions to the open-source community, which made this work possible.
