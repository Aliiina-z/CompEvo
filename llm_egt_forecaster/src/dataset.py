# File: src/dataset.py

import json
import os

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from llm_egt_forecaster.configs import base_config
from llm_egt_forecaster.data.data_generator import DataGenerator

class NewsTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing the news-driven time series data.
    It tokenizes the text inputs and formats the numerical data for the model.
    """
    def __init__(self, data_path, tokenizer, config):
        self.config = config
        self.tokenizer = tokenizer
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found at {data_path}. "
                "Please prepare your real dataset first."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # We will prepare the raw data here.
        # The complex prompt formatting will be handled by the model's forward pass,
        # as each agent might select different news.
        
        # --- Z-Score Normalization ---
        # Compute normalization parameters from the historical time series
        time_series_raw = sample["time_series"]
        ts_tensor = torch.tensor(time_series_raw, dtype=torch.float32)
        norm_mean = ts_tensor.mean()
        norm_std = ts_tensor.std()
        eps = 1e-8  # Prevent division by zero for constant series
        
        # Normalize the time series
        normalized_ts = (ts_tensor - norm_mean) / (norm_std + eps)
        time_series_str = ", ".join(map(lambda x: f"{x:.4f}", normalized_ts.tolist()))
        
        # Normalize the ground truth with the SAME mean/std from history
        ground_truth_raw = sample["ground_truth"]
        ground_truth = torch.tensor(ground_truth_raw, dtype=torch.float32)
        normalized_gt = (ground_truth - norm_mean) / (norm_std + eps)
        
        candidate_news = sample["candidate_news"]
        
        # Extract metadata if available (not used in Linear head approach, but kept for compatibility)
        metadata = sample.get("metadata", {})
        
        # We return the raw text and let the main model forward pass handle tokenization
        # because each agent will dynamically construct its own prompt.
        return {
            "time_series_str": time_series_str,
            "candidate_news": candidate_news,
            "ground_truth": normalized_gt,  # Normalized ground truth
            "norm_mean": norm_mean,  # For denormalization during inference/logging
            "norm_std": norm_std,    # For denormalization during inference/logging
            "metadata": metadata,  # Keep for compatibility, but not used
        }

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length candidate_news lists and metadata.
    Also batches normalization parameters for denormalization.
    """
    # Extract all items from batch
    time_series_strs = [item["time_series_str"] for item in batch]
    candidate_news_list = [item["candidate_news"] for item in batch]  # Keep as list of lists
    ground_truths = torch.stack([item["ground_truth"] for item in batch])  # Stack tensors (normalized)
    metadata_list = [item.get("metadata", {}) for item in batch]  # List of metadata dicts (not used in Linear head approach)
    
    # Stack normalization parameters for denormalization
    norm_means = torch.stack([item["norm_mean"] for item in batch])  # Shape: (batch_size,)
    norm_stds = torch.stack([item["norm_std"] for item in batch])    # Shape: (batch_size,)
    
    return {
        "time_series_str": time_series_strs,  # List of strings
        "candidate_news": candidate_news_list,  # List of lists (variable length)
        "ground_truth": ground_truths,  # Tensor (normalized)
        "norm_mean": norm_means,  # Tensor for denormalization
        "norm_std": norm_stds,    # Tensor for denormalization
        "metadata": metadata_list,  # List of metadata dicts (kept for compatibility)
    }

def get_dataloader(data_path, tokenizer, config, batch_size, shuffle=True, split='train', train_ratio=0.8, val_ratio=0.1):
    """A helper function to create a DataLoader with train/val/test split support.
    
    Args:
        data_path: Path to the dataset JSON file
        tokenizer: Tokenizer instance
        config: Configuration object
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
        split: 'train', 'val', 'test', or 'all' (default: 'train')
        train_ratio: Ratio of training data (default: 0.8 for 8:1:1 split)
        val_ratio: Ratio of validation data (default: 0.1 for 8:1:1 split)
        test_ratio: Automatically calculated as 1 - train_ratio - val_ratio
    """
    dataset = NewsTimeSeriesDataset(data_path, tokenizer, config)
    
    if split == 'all':
        # Return full dataset
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=custom_collate_fn
        )
    
    # Split dataset into train, validation, and test
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size  # Remaining for test set
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    if split == 'train':
        return torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=custom_collate_fn
        )
    elif split == 'val':
        return torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=custom_collate_fn
        )
    elif split == 'test':
        return torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=custom_collate_fn
        )
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', 'test', or 'all'")


if __name__ == '__main__':
    # --- This block is for testing the Dataset and DataLoader ---
    
    # 1. Ensure real data exists
    data_filepath = "data/real_dataset.json"
    if not os.path.exists(data_filepath):
        raise FileNotFoundError(
            f"Real dataset not found at {data_filepath}. "
            "Please prepare your real dataset using DataGenerator first."
        )

    # 2. Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_config.TOKENIZER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Create DataLoader
    dataloader = get_dataloader(
        data_path=data_filepath,
        tokenizer=tokenizer,
        config=base_config,
        batch_size=base_config.BATCH_SIZE,
        shuffle=True
    )

    # 4. Fetch and print one batch to verify
    print(f"\nTesting DataLoader with batch size {base_config.BATCH_SIZE}...")
    batch = next(iter(dataloader))

    print("\n--- Batch Content ---")
    print(f"Keys: {batch.keys()}")
    print("\nTime Series (first sample in batch, NORMALIZED):")
    print(batch["time_series_str"][0])
    
    print("\nCandidate News (first sample in batch):")
    for i, news in enumerate(batch["candidate_news"]):
        # Note: candidate_news is a list of lists of dicts.
        if i == 0:
            for news_item in news:
                 print(f"  - {news_item.get('summary', str(news_item))[:100]}...")
    
    print("\nGround Truth Tensor (first sample in batch, NORMALIZED):")
    print(batch["ground_truth"][0])
    print(f"Shape of ground_truth batch: {batch['ground_truth'].shape}")
    
    # 5. Verify normalization parameters
    print("\n--- Normalization Verification ---")
    print(f"norm_mean (first sample): {batch['norm_mean'][0].item():.4f}")
    print(f"norm_std (first sample): {batch['norm_std'][0].item():.4f}")
    
    # Denormalization check
    normalized_gt = batch['ground_truth'][0]
    denormalized_gt = normalized_gt * (batch['norm_std'][0] + 1e-8) + batch['norm_mean'][0]
    print(f"\nDenormalized ground truth (first sample): {denormalized_gt.item():.4f}")
    print("(Compare with original data to verify correctness)")

    print("\nDataset and DataLoader test completed successfully!")

