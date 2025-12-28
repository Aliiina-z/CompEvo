# File: configs/base_config.py
# Aligned with paper: "Improving LLM Agent Performance via Competition-Driven Evolution"

import torch
import os

# --- Multi-GPU Configuration ---
# Number of GPUs to use for model parallelism
# Set to 1 for single GPU, or higher for multi-GPU setups (e.g., 70B models)
NUM_GPUS = 1  # Default: single GPU

# Device map strategy for distributing model across GPUs:
# - 'auto': Let accelerate/transformers decide layer distribution automatically
# - 'balanced': Distribute layers evenly across available GPUs
# - 'sequential': Fill GPUs one by one (first GPU gets filled before second)
# - None: All layers on single GPU
DEVICE_MAP_STRATEGY = 'balanced'

# Visible GPU IDs (comma-separated string, e.g., "0,1,2,3")
# Set to None to use all available GPUs up to NUM_GPUS
VISIBLE_GPUS = None

# --- MODE SELECTION ---
# 'GPU_LLAMA' -> For users with a powerful CUDA GPU, runs Llama-3.1-8B with 4-bit quantization.
# 'CPU_DEBUG' -> For users without a powerful GPU, runs a smaller model (distilgpt2) on CPU.
MODE = 'GPU_LLAMA' # <-- CHANGE THIS TO 'CPU_DEBUG' FOR CPU-ONLY OR LOW-VRAM USAGE

# --- General Project Settings ---
PROJECT_NAME = "LLM_EGT_Forecaster"
SEED = 42

# --- Environment-Dependent Configurations ---
if MODE == 'GPU_LLAMA':
    print("--- RUNNING IN 'GPU_LLAMA' MODE ---")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # Primary device for multi-GPU
    if DEVICE == 'cpu':
        print("WARNING: 'GPU_LLAMA' mode selected, but no CUDA device found. Falling back to CPU.")
    
    # --- Model Configurations (Paper: Appendix "Implementation Details") ---
    BASE_LLM_MODEL = "meta-llama/Llama-3.1-8B"  # Paper uses Llama-3.1-8B
    TOKENIZER_PATH = BASE_LLM_MODEL
    USE_QUANTIZATION = True
    MAX_LENGTH = 8196  # Paper: "input context window is truncated to 4096 tokens"

    # --- PEFT (LoRA) Configurations (Paper: Appendix "Hyperparameters") ---
    LORA_RANK = 16  # Paper: r = 16
    LORA_ALPHA = 32  # Paper: α = 32
    LORA_DROPOUT = 0.05  # Paper: p = 0.05
    # Paper: "apply LoRA adapters to all linear projection layers"
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"       # FFN layers
    ]

    # --- Training Configurations (Paper: Appendix "Hyperparameters") ---
    BATCH_SIZE = 4
    LEARNING_RATE = 1.5e-4  # Paper: "initial learning rate of 1e-4"
    NUM_EPOCHS = 1  # Paper shows convergence within 2-3 epochs

elif MODE == 'CPU_DEBUG':
    print("--- RUNNING IN 'CPU_DEBUG' MODE ---")
    DEVICE = "cuda"
    
    # --- Model Configurations ---
    BASE_LLM_MODEL = "distilgpt2"
    TOKENIZER_PATH = BASE_LLM_MODEL
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
    NUM_EPOCHS = 3

else:
    raise ValueError(f"Invalid MODE selected: {MODE}. Choose from 'GPU_LLAMA' or 'CPU_DEBUG'.")

# --- Common Configurations (Paper-aligned) ---
NUM_AGENTS = 2  # Paper: "We set up 10 agents" (Experiments section)
WEIGHT_DECAY = 0.01

# --- Agent and Evolution Configurations (Paper: Appendix) ---
TEMPERATURE = 0.5  # Paper: "We set the temperature to 0.5" (Appendix "Impact of Temperature")
FITNESS_EMA_BETA = 0.9

# --- Loss Function Weights (Paper: Appendix "Balancing Innovation") ---
LAMBDA_STRATEGY = 1.0
LAMBDA_DIVERSITY = 0.4  # Paper: "optimal 'sweet spot' (λ ≈ 0.4)"
LAMBDA_PRUNING = 0.1  # Paper: Figure 4 shows optimal at 0.1

# --- Data Generation Settings ---
# Note: These vary by dataset (Paper: Table 4)
# Electricity/Traffic: 48 steps (30min/1hour granularity)
# Exchange/Bitcoin: 7 steps (1 day granularity)
TS_LENGTH = 48  # Input length (historical data)
FUTURE_STEPS = 48  # Prediction length (default for Electricity/Traffic)
NUM_VIRTUAL_SAMPLES = 100

# --- News Filtering Configurations ---
# 新闻筛选方法: 'cosine' (本地embedding) 或 'api' (OpenAI)
NEWS_SELECTOR_METHOD = 'cosine'  # 默认使用cosine,避免API调用成本

# Cosine方法配置
NEWS_COSINE_MODEL = 'all-MiniLM-L6-v2'  # SentenceTransformer模型
NEWS_COSINE_THRESHOLD = 0.3  # 相似度阈值
NEWS_COSINE_TOP_K = 3  # 选择top-k条新闻

# API方法配置 (仅当NEWS_SELECTOR_METHOD='api'时使用)
NEWS_API_KEY = os.getenv("OPENAI_API_KEY")  # 从环境变量读取
NEWS_API_BASE = os.getenv("OPENAI_API_BASE")  # 可选,自定义API base URL 
NEWS_API_MODEL = "gpt-4"  # 使用的OpenAI模型

# Token限制 (确保完整prompt不超过MAX_LENGTH=4096)
# 分配: ~2048 tokens for news, ~2048 tokens for time series + template
MAX_NEWS_TOKENS = 2048  # 新闻部分最大token数

# --- Prompt Engineering ---
PROMPT_TEMPLATE = """Instruction: You are an expert time series forecaster. Your task is to predict the next {future_steps} values of a sequence based on its history and relevant news events. Analyze the combined information to make an accurate forecast.

### Historical Data:
{time_series}

### Selected News Insights:
- {selected_news}

### Prediction:
"""

# --- Dataset-Specific Configurations ---
# These should be set based on the dataset being used
# Uncomment the appropriate configuration:

# For Electricity Dataset (30-minute granularity)
# TS_LENGTH = 48  # 1 day of history
# FUTURE_STEPS = 48  # Predict next 1 day

# For Exchange/Bitcoin Datasets (daily granularity)
# TS_LENGTH = 7  # 1 week of history
# FUTURE_STEPS = 7  # Predict next 1 week

# For Traffic Dataset (hourly granularity)
# TS_LENGTH = 24  # 1 day of history
# FUTURE_STEPS = 24  # Predict next 1 day
