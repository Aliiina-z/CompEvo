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
    # Set HuggingFace environment variables for mirror and timeout
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    if 'HF_HUB_DOWNLOAD_TIMEOUT' not in os.environ:
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # Increase download timeout
    if 'HF_HUB_ENABLE_HF_TRANSFER' not in os.environ:
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # Disable HF_TRANSFER to avoid CDN issues
    
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
    # Use an absolute path so this script works no matter where you run it from.
    _PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    data_filepath = os.path.join(_PROJECT_ROOT, "llm_egt_forecaster", "data", "real_dataset.json")

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

    # Create train, validation, and test dataloaders (8:1:1 split)
    # For ~3000 samples: 2400 train, 300 val, 300 test
    train_dataloader = get_dataloader(
        data_path=data_filepath,
        tokenizer=tokenizer,
        config=config,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        split='train',
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    val_dataloader = get_dataloader(
        data_path=data_filepath,
        tokenizer=tokenizer,
        config=config,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        split='val',
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    test_dataloader = get_dataloader(
        data_path=data_filepath,
        tokenizer=tokenizer,
        config=config,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        split='test',
        train_ratio=0.8,
        val_ratio=0.1
    )
    
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    print(f"Test samples: {len(test_dataloader.dataset)}")

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
        val_dataloader=val_dataloader,
        config=config,
        checkpoint_dir=os.path.join(_PROJECT_ROOT, "checkpoints")
    )

    # --- 5. Run Training ---
    trainer.train()

    print("\n--- Evolution Process Complete ---")
    print("Final Agent Logics and Fitness:")
    for agent in trainer.agents:
        print(f"- Agent {agent.id}: Fitness={agent.fitness:.4f}, Logic='{agent.logic}'")
    
    # --- 6. Test Set Evaluation (Final Performance) ---
    print("\n" + "="*60)
    print("--- Test Set Evaluation (Final Performance) ---")
    print("="*60)
    
    # Load best model checkpoint
    checkpoint_path = getattr(trainer, "best_checkpoint_path", None) or os.path.join(_PROJECT_ROOT, "checkpoints", "best_model.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        trainer.framework.load_state_dict(checkpoint['framework_state_dict'])
        if 'agent_logics' in checkpoint:
            for i, agent in enumerate(trainer.agents):
                agent.logic = checkpoint['agent_logics'][i]
                agent.fitness = checkpoint['agent_fitness'][i]
        print("✓ Best model loaded for test evaluation")
    else:
        print("⚠️  No checkpoint found, using current model state")
    
    # Evaluate on test set (same as validation but on test set)
    trainer.framework.eval()
    total_test_loss = 0
    total_test_mse = 0
    total_test_mae = 0
    total_test_mape = 0
    num_test_batches = 0
    
    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            framework_output = trainer.framework(batch)
            ground_truth = batch["ground_truth"].to(config.DEVICE)
            
            # Calculate test loss (prediction only, no PG components)
            loss_dict = trainer.loss_fn(framework_output, ground_truth, trainer.agents, pg_components=None)
            total_test_loss += loss_dict["total_loss"].item()
            
            # Calculate metrics
            predictions = framework_output["aggregated_prediction"]
            mse = torch.nn.functional.mse_loss(predictions, ground_truth)
            mae = torch.nn.functional.l1_loss(predictions, ground_truth)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            # MAPE = mean(|pred - true| / |true|) * 100
            # Avoid division by zero by adding small epsilon
            epsilon = 1e-8
            abs_error = torch.abs(predictions - ground_truth)
            abs_true = torch.abs(ground_truth) + epsilon
            percentage_error = (abs_error / abs_true) * 100
            mape = percentage_error.mean()
            
            total_test_mse += mse.item()
            total_test_mae += mae.item()
            total_test_mape += mape.item()
            num_test_batches += 1
    
    avg_test_loss = total_test_loss / num_test_batches
    avg_test_mse = total_test_mse / num_test_batches
    avg_test_mae = total_test_mae / num_test_batches
    avg_test_rmse = avg_test_mse ** 0.5
    avg_test_mape = total_test_mape / num_test_batches
    
    print(f"\n--- Test Set Results (Final Performance) ---")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test MSE:  {avg_test_mse:.4f}")
    print(f"Test MAE:  {avg_test_mae:.4f}")
    print(f"Test RMSE: {avg_test_rmse:.4f}")
    print(f"Test MAPE: {avg_test_mape:.4f}%")
    print("="*60)


if __name__ == '__main__':
    # To run the training, simply execute this file from the terminal:
    # python train.py
    main()
