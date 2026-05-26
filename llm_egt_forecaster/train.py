# File: train.py

# IMPORTANT:
# This file lives inside the package directory (`llm_egt_forecaster/`).
# If you run it directly via `python llm_egt_forecaster/train.py`, Python may
# accidentally import an *installed* `llm_egt_forecaster` package from site-packages
# instead of this repo checkout, causing API mismatches (e.g., get_dataloader val_ratio).
#
# To avoid that, we ensure the repo root is on sys.path.
import os
import sys
import argparse
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import random



import numpy as np

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train or resume LLM-EGT forecaster.")
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from before training.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs to run in this invocation. With --resume-checkpoint, this is the number of additional epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config.BATCH_SIZE from the command line.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override config.LEARNING_RATE from the command line.",
    )
    return parser.parse_args()


def main():
    """
    Main function to run the entire training and evolution pipeline.
    """
    args = parse_args()
    
    # --- 1. Load Configuration and Set Seed ---
    config = base_config
    if args.num_epochs is not None:
        config.NUM_EPOCHS = args.num_epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    
    set_seed(config.SEED)

    print("--- Configuration ---")
    print(f"Device: {config.DEVICE}")
    print(f"Base LLM: {config.BASE_LLM_MODEL}")
    print(f"Number of Agents: {config.NUM_AGENTS}")
    print(f"Number of Epochs: {config.NUM_EPOCHS}")
    if args.resume_checkpoint:
        print(f"Resume Checkpoint: {args.resume_checkpoint}")
    print(f"News Selector Method: {config.NEWS_SELECTOR_METHOD}")
    print("-" * 21)

    # --- 2. Prepare Data ---
    print("Preparing real data...")
    # Use an absolute path so this script works no matter where you run it from.
    # Dataset file: Use enhanced dataset with 'output' field for SFT training
    # Default to real_dataset_enhanced.json, fallback to real_dataset.json if not found
    enhanced_data_filepath = os.path.join(_REPO_ROOT, "llm_egt_forecaster", "data", "real_dataset_enhanced.json")
    default_data_filepath = os.path.join(_REPO_ROOT, "llm_egt_forecaster", "data", "real_dataset.json")
    
    if os.path.exists(enhanced_data_filepath):
        data_filepath = enhanced_data_filepath
        print(f"Using enhanced dataset: {data_filepath}")
    elif os.path.exists(default_data_filepath):
        data_filepath = default_data_filepath
        print(f"Using default dataset: {data_filepath} (Note: enhanced dataset with 'output' field is recommended for SFT)")
    else:
        data_filepath = enhanced_data_filepath  # Will raise error below

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
        checkpoint_dir=os.path.join(_REPO_ROOT, "checkpoints")
    )

    # --- 5. Run Training ---
    start_epoch = 0
    initial_best_val_loss = None
    if args.resume_checkpoint:
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        loaded_epoch, loaded_val_loss = trainer.load_checkpoint(args.resume_checkpoint)
        start_epoch = loaded_epoch + 1
        initial_best_val_loss = loaded_val_loss
    
    trainer.train(
        start_epoch=start_epoch,
        num_epochs=config.NUM_EPOCHS,
        initial_best_val_loss=initial_best_val_loss,
    )

    print("\n--- Evolution Process Complete ---")
    print("Final Agent Logics and Fitness:")
    for agent in trainer.agents:
        print(f"- Agent {agent.id}: Fitness={agent.fitness:.4f}, Logic='{agent.logic}'")
    
    # --- 6. Test Set Evaluation (Final Performance) ---
    print("\n" + "="*60)
    print("--- Test Set Evaluation (Final Performance) ---")
    print("="*60)
    
    # Load best model checkpoint
    # Always save/load checkpoints under repo root
    checkpoint_path = getattr(trainer, "best_checkpoint_path", None) or os.path.join(_REPO_ROOT, "checkpoints", "best_model.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading best model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        
        # Filter out quantization-related keys that may not match current model state
        framework_state = checkpoint['framework_state_dict']
        if isinstance(framework_state, dict):
            # Remove quantization state keys (bitsandbytes quantization parameters)
            filtered_state = {}
            for key, value in framework_state.items():
                # Skip quantization-related keys
                if any(quant_key in key for quant_key in [
                    '.absmax', '.quant_map', '.nested_absmax', '.nested_quant_map', 
                    '.quant_state', 'bitsandbytes'
                ]):
                    continue
                filtered_state[key] = value
            framework_state = filtered_state
        
        # Load with strict=False to handle any remaining mismatches
        trainer.framework.load_state_dict(framework_state, strict=False)
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
    # Denormalized metrics (for reporting real-world metrics)
    total_test_mse_denorm = 0
    total_test_mae_denorm = 0
    total_test_mape_denorm = 0
    num_test_batches = 0
    total_forward_latency_s = 0.0
    total_test_samples = 0
    
    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_start = time.perf_counter()
            framework_output = trainer.framework(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_elapsed_s = time.perf_counter() - forward_start
            batch_size_actual = len(batch["time_series_str"])
            total_forward_latency_s += forward_elapsed_s
            total_test_samples += batch_size_actual
            ground_truth = batch["ground_truth"].to(config.DEVICE)
            
            # Get normalization parameters for denormalization
            norm_mean = batch["norm_mean"].to(config.DEVICE)
            norm_std = batch["norm_std"].to(config.DEVICE)
            
            # Calculate test loss (prediction only, no PG components)
            # Loss is calculated in NORMALIZED space for proper gradient computation
            loss_dict = trainer.loss_fn(framework_output, ground_truth, trainer.agents, pg_components=None)
            total_test_loss += loss_dict["total_loss"].item()
            
            # Denormalize for reporting real-world metrics (same as validation)
            predictions = framework_output["aggregated_prediction"]
            predictions_denorm = trainer._denormalize(predictions, norm_mean, norm_std)
            ground_truth_denorm = trainer._denormalize(ground_truth, norm_mean, norm_std)
            
            # Calculate metrics in ORIGINAL (denormalized) space
            mse_denorm = torch.nn.functional.mse_loss(predictions_denorm, ground_truth_denorm)
            mae_denorm = torch.nn.functional.l1_loss(predictions_denorm, ground_truth_denorm)
            
            # Calculate MAPE in original space (more meaningful)
            epsilon = 1e-8
            abs_error = torch.abs(predictions_denorm - ground_truth_denorm)
            abs_true = torch.abs(ground_truth_denorm) + epsilon
            mape_denorm = ((abs_error / abs_true) * 100).mean()
            
            total_test_mse_denorm += mse_denorm.item()
            total_test_mae_denorm += mae_denorm.item()
            total_test_mape_denorm += mape_denorm.item()
            num_test_batches += 1
    
    avg_test_loss = total_test_loss / num_test_batches
    
    # Denormalized metrics (for reporting)
    avg_test_mse_denorm = total_test_mse_denorm / num_test_batches
    avg_test_mae_denorm = total_test_mae_denorm / num_test_batches
    avg_test_rmse_denorm = avg_test_mse_denorm ** 0.5
    avg_test_mape_denorm = total_test_mape_denorm / num_test_batches
    avg_latency_ms_per_step = (total_forward_latency_s / num_test_batches) * 1000 if num_test_batches else 0.0
    avg_latency_ms_per_sample = (total_forward_latency_s / total_test_samples) * 1000 if total_test_samples else 0.0
    
    print(f"\n--- Test Set Results (Final Performance) ---")
    print(f"Test Loss (normalized): {avg_test_loss:.4f}")
    print(f"\n[Original Scale Metrics - for human understanding]")
    print(f"  MSE: {avg_test_mse_denorm:.4f}")
    print(f"  RMSE: {avg_test_rmse_denorm:.4f}")
    print(f"  MAE: {avg_test_mae_denorm:.4f}")
    print(f"  MAPE: {avg_test_mape_denorm:.4f}%")
    print(f"\n[Efficiency Metrics]")
    print(f"  Latency: {avg_latency_ms_per_step:.4f} ms/step")
    print(f"  Latency: {avg_latency_ms_per_sample:.4f} ms/sample")
    if hasattr(trainer, "efficiency_metrics"):
        train_time_h = trainer.efficiency_metrics.get("avg_train_time_h_per_epoch")
        peak_allocated_gb = trainer.efficiency_metrics.get("peak_memory_allocated_gb")
        peak_reserved_gb = trainer.efficiency_metrics.get("peak_memory_reserved_gb")
        if train_time_h is not None:
            print(f"  Train time: {train_time_h:.6f} h/ep")
        if peak_reserved_gb is not None:
            print(f"  Peak memory reserved: {peak_reserved_gb:.4f} GB")
        if peak_allocated_gb is not None:
            print(f"  Peak memory allocated: {peak_allocated_gb:.4f} GB")

    print("\n========== Efficiency Summary ==========")
    if hasattr(trainer, "efficiency_metrics") and trainer.efficiency_metrics.get("avg_train_time_h_per_epoch") is not None:
        print(f"Train time (h/ep): {trainer.efficiency_metrics['avg_train_time_h_per_epoch']:.6f}")
    print(f"Latency (ms/step): {avg_latency_ms_per_step:.4f}")
    if hasattr(trainer, "efficiency_metrics") and trainer.efficiency_metrics.get("peak_memory_reserved_gb") is not None:
        print(f"Peak memory (GB, reserved): {trainer.efficiency_metrics['peak_memory_reserved_gb']:.4f}")
    print(f"RMSE: {avg_test_rmse_denorm:.4f}")
    print("========================================")
    print("="*60)


if __name__ == '__main__':
    # To run the training, simply execute this file from the terminal:
    # python train.py
    main()
