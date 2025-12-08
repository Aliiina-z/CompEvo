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
        
        time_series_str = ", ".join(map(str, sample["time_series"]))
        candidate_news = sample["candidate_news"]
        ground_truth = torch.tensor(sample["ground_truth"], dtype=torch.float32)

        # We return the raw text and let the main model forward pass handle tokenization
        # because each agent will dynamically construct its own prompt.
        return {
            "time_series_str": time_series_str,
            "candidate_news": candidate_news,
            "ground_truth": ground_truth,
        }

def get_dataloader(data_path, tokenizer, config, batch_size, shuffle=True):
    """A helper function to create a DataLoader."""
    dataset = NewsTimeSeriesDataset(data_path, tokenizer, config)
    # A custom collate_fn is not strictly necessary here since we're not padding yet.
    # The main model logic will handle batching of tokenized inputs.
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
    print("\nTime Series (first sample in batch):")
    print(batch["time_series_str"][0])
    
    print("\nCandidate News (first sample in batch):")
    for i, news in enumerate(batch["candidate_news"]):
        # Note: candidate_news is a list of lists of dicts.
        if i == 0:
            for news_item in news:
                 print(f"  - {news_item.get('summary', str(news_item))[:100]}...")
    
    print("\nGround Truth Tensor (first sample in batch):")
    print(batch["ground_truth"][0])
    print(f"Shape of ground_truth batch: {batch['ground_truth'].shape}")

    print("\nDataset and DataLoader test completed successfully!")
