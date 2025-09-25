# File: data/virtual_data_generator.py

import os
import json
import numpy as np
from tqdm import tqdm

from llm_egt_forecaster.configs import base_config

class VirtualDataGenerator:
    """
    Generates a virtual dataset for news-driven time series forecasting.
    Each sample consists of a time series, a pool of candidate news,
    and the ground truth future values, where one of the news items
    causally influences the future trend.
    """
    def __init__(self, config):
        self.config = config
        np.random.seed(self.config.SEED)
        self.news_pool = [
            "Market sentiment is optimistic, tech stocks are soaring.",
            "Bad News: A key local factory has halted production due to supply chain issues.",
            "Weather Forecast: Continued sunny skies expected for the next week.",
            "Good News: The government has announced a new economic stimulus package.",
            "Sports Update: The home team won the championship.",
            "Consumer spending shows a slight decline in the recent quarter.",
            "A major tech company announced a breakthrough in AI, boosting investor confidence."
        ]

    def generate_single_sample(self):
        """Generates one data sample."""
        # 1. Generate base time series (e.g., a sine wave with noise)
        ts_length = self.config.TS_LENGTH
        future_steps = self.config.FUTURE_STEPS
        
        time_points = np.linspace(0, 10, ts_length)
        base_series = np.sin(time_points) + np.random.normal(0, 0.1, ts_length)
        
        # 2. Randomly select a key news item and determine its impact
        key_news_indices = [1, 3, 5, 6] # Indices of news that can have an impact
        key_news_index = np.random.choice(key_news_indices)
        
        impact_strength = np.random.uniform(0.3, 0.7)
        if key_news_index in [3, 6]: # Positive news
            impact = impact_strength * np.arange(1, future_steps + 1) / future_steps
        else: # Negative news
            impact = -impact_strength * np.arange(1, future_steps + 1) / future_steps
            
        # 3. Generate the ground truth future series
        last_value = base_series[-1]
        future_series = last_value + impact + np.random.normal(0, 0.1, future_steps)
        
        sample = {
            "time_series": list(np.round(base_series, 4)),
            "candidate_news": self.news_pool,
            "ground_truth": list(np.round(future_series, 4))
        }
        return sample

    def generate_dataset(self, num_samples, filepath="data/virtual_dataset.json"):
        """Generates a full dataset and saves it to a JSON file."""
        if os.path.exists(filepath):
            print(f"Dataset already exists at {filepath}. Loading...")
            with open(filepath, 'r') as f:
                dataset = json.load(f)
            return dataset

        print(f"Generating {num_samples} virtual samples...")
        dataset = [self.generate_single_sample() for _ in tqdm(range(num_samples))]
        
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        print(f"Saving dataset to {filepath}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
            
        return dataset

if __name__ == '__main__':
    # This block allows us to run this file directly to generate the data
    generator = VirtualDataGenerator(base_config)
    
    # Generate the dataset based on the number specified in the config
    dataset = generator.generate_dataset(base_config.NUM_VIRTUAL_SAMPLES)
    
    print("\n--- Sample 0 ---")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))
    
    print(f"\nSuccessfully generated/loaded {len(dataset)} samples.")