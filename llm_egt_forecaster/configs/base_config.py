# File: configs/base_config.py

import os

import torch


# --- Multi-GPU Configuration ---
NUM_GPUS = 1

# Device map strategy for distributing model across GPUs:
# - 'auto': Let accelerate/transformers decide layer distribution automatically.
# - 'balanced': Distribute layers evenly across available GPUs.
# - 'sequential': Fill GPUs one by one.
# - None or single dict {"": "cuda:0"}: Place all layers on one GPU.
DEVICE_MAP_STRATEGY = 'balanced'

# Visible GPU IDs as a comma-separated string, e.g. "0,1,2,3".
# Set to None to use all available GPUs up to NUM_GPUS.
VISIBLE_GPUS = None


# --- Mode Selection ---
# 'GPU_LLAMA': CUDA GPU with Llama-3.1-8B and 4-bit quantization.
# 'MPS_LLAMA': Apple Silicon MPS acceleration without bitsandbytes quantization.
# 'GPU_QWEN': CUDA GPU with Qwen2.5-7B-Instruct and 4-bit quantization.
# 'CPU_DEBUG': Small CPU-only debug model.
# MODE = 'MPS_LLAMA' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'CPU_DEBUG'
MODE = 'GPU_LLAMA'


# --- General Project Settings ---
PROJECT_NAME = "LLM_EGT_Forecaster"
SEED = 42


# --- Environment-Dependent Configurations ---
if MODE == 'GPU_LLAMA':
    print("--- RUNNING IN 'GPU_LLAMA' MODE ---")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    if DEVICE == 'cpu':
        print("WARNING: 'GPU_LLAMA' mode selected, but no CUDA device found. Falling back to CPU.")

    # --- Model Configurations ---
    BASE_LLM_MODEL = os.getenv("BASE_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", BASE_LLM_MODEL)
    USE_QUANTIZATION = True
    MAX_LENGTH = 4096

    # --- PEFT (LoRA) Configurations ---
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # --- Training Configurations ---
    BATCH_SIZE = 4
    LEARNING_RATE = 1.5e-4
    NUM_EPOCHS = 3

    # --- Loss Function Configuration ---
    # Options: 'mse', 'mae', 'mape', 'rmse', 'huber'
    PREDICTION_LOSS_TYPE = 'mae'
    HUBER_DELTA = 1.0
    MAX_GRAD_NORM = 1.0

elif MODE == 'MPS_LLAMA':
    print("--- RUNNING IN 'MPS_LLAMA' MODE (Apple Silicon GPU) ---")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        print("WARNING: MPS not available, falling back to CPU.")
        DEVICE = "cpu"

    # --- Model Configurations ---
    BASE_LLM_MODEL = os.getenv("BASE_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B")
    TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", BASE_LLM_MODEL)
    USE_QUANTIZATION = False
    MAX_LENGTH = 2048

    # --- PEFT (LoRA) Configurations ---
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # --- Training Configurations ---
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 3

elif MODE == 'GPU_QWEN':
    print("--- RUNNING IN 'GPU_QWEN' MODE (Qwen2.5 Model) ---")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    if DEVICE == 'cpu':
        print("WARNING: 'GPU_QWEN' mode selected, but no CUDA device found. Falling back to CPU.")

    # --- Model Configurations ---
    BASE_LLM_MODEL = os.getenv("BASE_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", BASE_LLM_MODEL)
    USE_QUANTIZATION = True
    MAX_LENGTH = 4096

    # --- PEFT (LoRA) Configurations ---
    LORA_RANK = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # --- Training Configurations ---
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 1

elif MODE == 'CPU_DEBUG':
    print("--- RUNNING IN 'CPU_DEBUG' MODE ---")
    DEVICE = "cpu"

    # --- Model Configurations ---
    BASE_LLM_MODEL = os.getenv("BASE_LLM_MODEL", "distilgpt2")
    TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", BASE_LLM_MODEL)
    USE_QUANTIZATION = False
    MAX_LENGTH = 1024

    # --- PEFT (LoRA) Configurations ---
    LORA_RANK = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["c_attn"]

    # --- Training Configurations ---
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 1

else:
    raise ValueError(f"Invalid MODE selected: {MODE}. Choose from 'GPU_LLAMA', 'MPS_LLAMA', 'GPU_QWEN', or 'CPU_DEBUG'.")


# --- Common Configurations ---
NUM_AGENTS = 3
WEIGHT_DECAY = 0.01


# --- Agent and Evolution Configurations ---
TEMPERATURE = 0.5
FITNESS_EMA_BETA = 0.9


# --- Loss Function Weights ---
LAMBDA_STRATEGY = 1e-5
LAMBDA_DIVERSITY = 0.1
LAMBDA_PRUNING = 0.01


# --- Logic LoRA Configuration ---
LOGIC_LORA_RANK = 2
LOGIC_LORA_ALPHA = 4
LOGIC_LORA_DROPOUT = 0.05


# --- KL Divergence Constraint ---
LAMBDA_KL = 0.01


# --- Logic Generation Configuration ---
LOGIC_MAX_NEW_TOKENS = 100
LOGIC_TEMPERATURE = 0.8
LOGIC_TOP_K = 50


# --- Data Generation Settings ---
# Electricity/Traffic: 48 steps with 30-minute or 1-hour granularity.
# Exchange/Bitcoin: 7 steps with 1-day granularity.
TS_LENGTH = 48
FUTURE_STEPS = 48
NUM_VIRTUAL_SAMPLES = 100


# --- News Filtering Configurations ---
# NEWS_SELECTOR_METHOD options:
# - 'cosine': Local embedding-based selection, no API key required.
# - 'api': OpenAI-compatible API-based selection.
NEWS_SELECTOR_METHOD = 'cosine'

# Cosine method configuration.
NEWS_COSINE_MODEL = os.getenv(
    "NEWS_COSINE_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
NEWS_COSINE_THRESHOLD = 0.3
NEWS_COSINE_TOP_K = 6

# API method configuration. Do not hard-code secrets in this file.
NEWS_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("NEWS_API_KEY")
NEWS_API_BASE = os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
NEWS_API_MODEL = os.getenv("OPENAI_API_MODEL") or "gpt-4o-mini"


# --- Prompt Configuration ---
MAX_NEWS_TOKENS = 2048
PROMPT_TEMPLATE = """###{input}

###{instruction}

### Response:
"""


# --- Dataset-Specific Overrides ---
# For electricity datasets:
# TS_LENGTH = 48
# FUTURE_STEPS = 48

# For exchange or bitcoin datasets:
# TS_LENGTH = 7
# FUTURE_STEPS = 7

# For traffic datasets:
# TS_LENGTH = 24
# FUTURE_STEPS = 24
