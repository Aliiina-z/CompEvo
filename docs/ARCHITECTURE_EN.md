# LLM EGT Forecaster - Project Architecture Documentation

## Project Overview

This project implements a multi-agent Large Language Model framework based on Evolutionary Game Theory (EGT) for news-driven time series forecasting. Through a competition-driven evolutionary process, agents can self-adaptively develop diverse and effective strategies for filtering news and predicting future trends.

## Core Architecture

```
llm_egt_forecaster/
├── configs/                    # Configuration Module
│   ├── __init__.py
│   └── base_config.py         # Base configuration with GPU/CPU modes
├── data/                      # Data Module
│   ├── __init__.py
│   └── virtual_data_generator.py  # Virtual data generator
├── src/                       # Core Source Code
│   ├── __init__.py
│   ├── agent.py              # Agent definition
│   ├── dataset.py            # Dataset and loader
│   ├── engine/               # Training Engine
│   │   ├── __init__.py
│   │   ├── loss.py           # Evolutionary loss function
│   │   └── trainer.py        # Trainer
│   ├── models/               # Model Definitions
│   │   ├── __init__.py
│   │   ├── evolutionary_framework.py  # Main framework
│   │   ├── logic_generator.py        # Logic generator
│   │   └── news_selector.py          # News selector
│   └── utils/                # Utility functions
│       ├── __init__.py
│       └── helpers.py
├── docs/                     # Documentation
│   ├── ARCHITECTURE_CN.md    # Chinese architecture docs
│   └── ARCHITECTURE_EN.md    # English architecture docs
├── train.py                  # Training entry point
├── setup.py                  # Package configuration
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

## Module Details

### 1. Configuration Module (configs/)

**Function**: Manages project settings and supports different hardware environments

**Key Files**:
- `base_config.py`: Base configuration with GPU_LLAMA and CPU_DEBUG modes

**Main Configurations**:
```python
MODE = 'GPU_LLAMA'  # or 'CPU_DEBUG'
BASE_LLM_MODEL = "meta-llama/Llama-2-7b-hf"  # or "distilgpt2"
USE_QUANTIZATION = True  # Whether to use quantization
NUM_AGENTS = 5  # Number of agents
NUM_EPOCHS = 3  # Number of training epochs
```

### 2. Data Module (data/)

**Function**: Data generation and loading

**Key Files**:
- `virtual_data_generator.py`: Generates virtual test data

**Data Format**:
```json
{
  "time_series": [list of numbers],
  "candidate_news": [list of news strings],
  "ground_truth": [list of future values]
}
```

### 3. Agent Module (src/agent.py)

**Function**: Defines individual agents

**Key Features**:
- Inherits from `nn.Module` for automatic parameter collection
- Each agent has independent LoRA weights and gate parameters
- Contains fitness and logic attributes

**Core Methods**:
```python
class Agent(nn.Module):
    def __init__(self, agent_id, base_llm_model, lora_config, device)
    def update_fitness(self, reward, beta)  # Update fitness
    def update_logic(self, new_logic)       # Update strategy logic
    def forward(self, *args, **kwargs)      # Forward pass
```

### 4. Evolutionary Framework (src/models/evolutionary_framework.py)

**Function**: Main framework coordinating multi-agent evolution

**Key Features**:
- Supports conditional quantization model loading
- Robust batch data preparation
- Agent prediction aggregation

**Core Methods**:
```python
class EvolutionaryFramework(nn.Module):
    def __init__(self, config)              # Initialize framework
    def _prepare_batch(self, batch)         # Prepare batch data
    def forward(self, batch)                # Forward pass
```

### 5. News Selector (src/models/news_selector.py)

**Function**: Selects relevant news based on semantic similarity

**Key Features**:
- Uses sentence-transformers for semantic matching
- Supports threshold filtering and top-k selection
- Includes fallback mechanisms

### 6. Logic Generator (src/models/logic_generator.py)

**Function**: Generates agent strategy logic

**Key Features**:
- Supports policy gradient training
- Generates new logic based on peer information
- Returns log probabilities for PG loss

### 7. Loss Function (src/engine/loss.py)

**Function**: Computes evolutionary loss

**Loss Components**:
- Prediction loss (L_predict)
- Policy gradient loss (L_PG)
- Diversity loss (L_diversity)
- Pruning loss (L_pruning)

### 8. Trainer (src/engine/trainer.py)

**Function**: Manages training process

**Key Features**:
- Simplified optimizer initialization
- Policy gradient training loop
- Agent fitness updates

## Data Flow

```
1. Data Loading
   virtual_data_generator.py → dataset.py → DataLoader

2. Forward Pass
   EvolutionaryFramework → NewsSelector → Agent → LogicGenerator

3. Loss Calculation
   EvolutionaryLoss → Policy Gradient + Diversity + Pruning

4. Backward Pass
   Trainer → Optimizer → Parameter Updates

5. Evolutionary Update
   Agent.fitness ← Reward
   Agent.logic ← New Strategy
```

## Extension Guide

### Adding New Agent Strategies

1. Modify the `update_logic` method in `src/agent.py`
2. Add new generation logic in `src/models/logic_generator.py`
3. Update diversity calculation in `src/engine/loss.py`

### Adding New Loss Functions

1. Add new loss calculation methods in `src/engine/loss.py`
2. Integrate new loss in `EvolutionaryLoss.forward`
3. Update weight parameters in configuration

### Supporting New Data Sources

1. Create new data loader classes
2. Implement conversion to existing data format
3. Update `src/dataset.py` to support new format

### Adding New Model Architectures

1. Create new model classes in `src/models/`
2. Inherit from `nn.Module` and implement required methods
3. Integrate new models in `EvolutionaryFramework`

## Configuration Guide

### Environment Modes

**GPU_LLAMA Mode** (Recommended for production):
- Uses Llama-2-7B model
- 4-bit quantization to save memory
- Requires >=16GB VRAM

**CPU_DEBUG Mode** (For debugging):
- Uses distilgpt2 model
- CPU inference
- Suitable for quick logic verification

### Key Parameters

```python
# Model Configuration
BASE_LLM_MODEL = "meta-llama/Llama-2-7b-hf"
USE_QUANTIZATION = True
LORA_RANK = 16
LORA_ALPHA = 32

# Training Configuration
NUM_AGENTS = 5
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 2e-5

# Loss Weights
LAMBDA_STRATEGY = 1.0
LAMBDA_DIVERSITY = 0.5
LAMBDA_PRUNING = 0.1
```

## Contributing Guide

### Code Standards

1. **Import Standards**: Use `llm_egt_forecaster.*` prefix
2. **Type Hints**: Add type hints for function parameters and return values
3. **Docstrings**: Add detailed docstrings for all public methods
4. **Error Handling**: Add appropriate exception handling

### Testing Requirements

1. Add unit tests for each new feature
2. Ensure compatibility with both modes
3. Verify data format compatibility

### Commit Standards

1. Use clear commit messages
2. One commit per feature or fix
3. Update related documentation

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Switch to CPU_DEBUG mode or reduce BATCH_SIZE
2. **Model Loading Failed**: Check Hugging Face authentication and network connection
3. **Data Format Error**: Verify JSON file format and field completeness

### Debugging Tips

1. Use `python -m llm_egt_forecaster.data.virtual_data_generator` to test data generation
2. Set `MODE = 'CPU_DEBUG'` for quick debugging
3. Check `src/dataset.py` `__main__` block for data loading tests

## Running Examples

### Basic Training

```bash
# Train with virtual data
python train.py

# Train with real data
python train.py --data_path data/real_dataset.json
```

### Data Generation

```bash
# Generate virtual data
python -m llm_egt_forecaster.data.virtual_data_generator

# Test data loading
python -m llm_egt_forecaster.src.dataset
```

### Mode Switching

Modify the `MODE` variable in `configs/base_config.py`:
- `'GPU_LLAMA'`: Use Llama-2-7B + 4bit quantization
- `'CPU_DEBUG'`: Use distilgpt2 + CPU inference

## Key Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Extensibility**: Easy to add new agents, loss functions, or data sources
3. **Flexibility**: Supports both GPU and CPU environments
4. **Maintainability**: Clear code structure and comprehensive documentation

## Extension Points

- **Agent Strategies**: Modify `src/agent.py` and `src/models/logic_generator.py`
- **Loss Functions**: Add new components in `src/engine/loss.py`
- **Data Sources**: Create new loaders and update `src/dataset.py`
- **Model Architectures**: Implement new models in `src/models/`

## Getting Started for Contributors

1. **Fork the repository**
2. **Set up development environment**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run tests**:
   ```bash
   python -m llm_egt_forecaster.data.virtual_data_generator
   python train.py
   ```
4. **Make changes** following the coding standards
5. **Submit pull request** with clear description

## Support

For questions or issues, please:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed description
4. Contact the maintainers

---

*This architecture documentation is maintained alongside the codebase. Please update it when making significant changes to the project structure.*
