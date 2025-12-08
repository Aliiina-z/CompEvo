# File: train.py

import torch
import random
import numpy as np
import os

from llm_egt_forecaster.configs import base_config
from llm_egt_forecaster.data.data_generator import DataGenerator
from llm_egt_forecaster.src.dataset import get_dataloader
from llm_egt_forecaster.src.models.evolutionary_framework import EvolutionaryFramework
from llm_egt_forecaster.src.engine.loss import EvolutionaryLoss
from llm_egt_forecaster.src.models.logic_generator import LogicGenerator
from llm_egt_forecaster.src.engine.trainer import Trainer
from transformers import AutoTokenizer


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """
    Main function to run the entire training and evolution pipeline.
    """
    # --- 1. Load Configuration and Set Seed ---
    config = base_config
    set_seed(config.SEED)

    print("--- Configuration ---")
    print(f"Device: {config.DEVICE}")
    print(f"Base LLM: {config.BASE_LLM_MODEL}")
    print(f"Number of Agents: {config.NUM_AGENTS}")
    print(f"Number of Epochs: {config.NUM_EPOCHS}")
    print(f"News Selector Method: {config.NEWS_SELECTOR_METHOD}")
    print("-" * 21)

    # --- 2. Prepare Data ---
    print("Preparing real data...")
    data_filepath = "data/real_dataset.json"

    if not os.path.exists(data_filepath):
        raise FileNotFoundError(
            f"Real dataset not found at {data_filepath}. "
            "Please prepare your real dataset first."
        )

    # Load and convert real data to standard format
    print(f"Loading real dataset from {data_filepath}...")
    real_generator = DataGenerator(config, filepath=data_filepath)
    real_generator.load_dataset()

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataloader = get_dataloader(
        data_path=data_filepath,
        tokenizer=tokenizer,
        config=config,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    print(f"Data loaded with {len(train_dataloader.dataset)} samples.")

    # --- 3. Initialize Model Components ---
    print("Initializing model components...")

    # Initialize the main framework
    framework = EvolutionaryFramework(config).to(config.DEVICE)

    # Initialize the loss function
    loss_fn = EvolutionaryLoss(config)

    # Initialize the logic generator (shares the base model with the framework)
    logic_generator = LogicGenerator(framework.base_model, tokenizer, config)

    # --- 4. Initialize and Start the Trainer ---
    print("Initializing trainer...")
    trainer = Trainer(
        framework=framework,
        loss_fn=loss_fn,
        logic_generator=logic_generator,
        dataloader=train_dataloader,
        config=config
    )

    # --- 5. Run Training ---
    trainer.train()

    print("\n--- Evolution Process Complete ---")
    print("Final Agent Logics and Fitness:")
    for agent in trainer.agents:
        print(f"- Agent {agent.id}: Fitness={agent.fitness:.4f}, Logic='{agent.logic}'")


if __name__ == '__main__':
    # To run the training, simply execute this file from the terminal:
    # python train.py
    main()
