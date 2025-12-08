# File: configs/base_config.py (Modified for Environment Friendliness)

import torch
import os

# --- MODE SELECTION ---
# 'GPU_LLAMA' -> For users with a powerful CUDA GPU, runs Llama-2-7B with 4-bit quantization.
# 'CPU_DEBUG' -> For users without a powerful GPU, runs a smaller model (distilgpt2) on CPU.
MODE = 'GPU_LLAMA' # <-- CHANGE THIS TO 'CPU_DEBUG' FOR CPU-ONLY OR LOW-VRAM USAGE

# --- General Project Settings ---
PROJECT_NAME = "LLM_EGT_Forecaster"
SEED = 42

# --- Environment-Dependent Configurations ---
if MODE == 'GPU_LLAMA':
    print("--- RUNNING IN 'GPU_LLAMA' MODE ---")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == 'cpu':
        print("WARNING: 'GPU_LLAMA' mode selected, but no CUDA device found. Falling back to CPU.")
    
    # --- Model Configurations ---
    BASE_LLM_MODEL = "meta-llama/Llama-2-7b-hf"
    TOKENIZER_PATH = BASE_LLM_MODEL
    USE_QUANTIZATION = True

    # --- PEFT (LoRA) Configurations ---
    LORA_RANK = 16
    LORA_ALPHA = 32
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]

    # --- Training Configurations ---
    BATCH_SIZE = 2
    LEARNING_RATE = 2e-5

elif MODE == 'CPU_DEBUG':
    print("--- RUNNING IN 'CPU_DEBUG' MODE ---")
    DEVICE = "cpu"
    
    # --- Model Configurations ---
    BASE_LLM_MODEL = "distilgpt2"
    TOKENIZER_PATH = BASE_LLM_MODEL
    USE_QUANTIZATION = False

    # --- PEFT (LoRA) Configurations ---
    LORA_RANK = 8
    LORA_ALPHA = 16
    LORA_TARGET_MODULES = ["c_attn"]

    # --- Training Configurations ---
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4

else:
    raise ValueError(f"Invalid MODE selected: {MODE}. Choose from 'GPU_LLAMA' or 'CPU_DEBUG'.")

# --- Common Configurations (independent of mode) ---
LORA_DROPOUT = 0.05
NUM_AGENTS = 5 # Reduced for faster debugging in both modes, can be increased
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01

# --- Agent and Evolution Configurations ---
TEMPERATURE = 0.7
FITNESS_EMA_BETA = 0.9

# --- Loss Function Weights ---
LAMBDA_STRATEGY = 1.0
LAMBDA_DIVERSITY = 0.5
LAMBDA_PRUNING = 0.1

# --- Data Generation Settings ---
TS_LENGTH = 50
FUTURE_STEPS = 5
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

# --- Prompt Engineering ---
PROMPT_TEMPLATE = """Instruction: You are an expert time series forecaster. Your task is to predict the next {future_steps} values of a sequence based on its history and relevant news events. Analyze the combined information to make an accurate forecast.

### Historical Data:
{time_series}

### Selected News Insights:
- {selected_news}

### Prediction:
"""
