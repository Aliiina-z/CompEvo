#!/usr/bin/env python3
# Standalone script to run dataset enhancement

import os
import sys
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import holidays
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from llm_egt_forecaster.data.enhanced_data_generator import (
    EnhancedDataGenerator,
    check_holiday_or_not,
    check_weekday_or_weekend,
    categorize_state
)

# Simple config object
class SimpleConfig:
    pass

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_file = os.path.join(base_dir, "llm_egt_forecaster", "data", "real_dataset.json")
    weather_file = os.path.join(base_dir, "llm_egt_forecaster", "data", "raw_time_series_data", "weather_load_2019-2022.csv")
    output_file = os.path.join(base_dir, "llm_egt_forecaster", "data", "real_dataset_enhanced.json")
    
    print("=" * 60)
    print("Enhanced Dataset Generator")
    print("=" * 60)
    print(f"Input dataset: {data_file}")
    print(f"Weather data: {weather_file if os.path.exists(weather_file) else 'Not found (optional)'}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    
    # Create config
    config = SimpleConfig()
    
    # Create generator
    generator = EnhancedDataGenerator(config, filepath=data_file)
    
    # Load weather data if available
    if os.path.exists(weather_file):
        print("\n📊 Loading weather data...")
        generator.load_weather_data(weather_file)
    else:
        print("\n⚠️  Weather data file not found, continuing without weather information")
    
    # Load and enhance dataset
    print("\n🔄 Enhancing dataset...")
    enhanced_dataset = generator.load_and_enhance_dataset(
        weather_file_path=weather_file if os.path.exists(weather_file) else None
    )
    
    # Save enhanced dataset
    print("\n💾 Saving enhanced dataset...")
    output_path = generator.save_enhanced_dataset(enhanced_dataset, output_file)
    
    # Print summary
    print("\n" + "=" * 60)
    print("✅ Enhancement Complete!")
    print("=" * 60)
    print(f"Total samples: {len(enhanced_dataset)}")
    print(f"Output file: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Show sample
    if len(enhanced_dataset) > 0:
        print("\n📋 Sample 0 structure:")
        sample = enhanced_dataset[0]
        print(f"  Keys: {list(sample.keys())}")
        
        if 'metadata' in sample:
            print(f"  Metadata: {list(sample['metadata'].keys())}")
            if 'weather' in sample['metadata']:
                print(f"  Weather info: ✅")
            if 'is_weekend' in sample['metadata']:
                print(f"  Weekend info: ✅")
            if 'holiday_info' in sample['metadata']:
                print(f"  Holiday info: ✅")
        
        if 'formatted_news' in sample:
            print(f"  Formatted news count: {len(sample['formatted_news'])}")
            if len(sample['formatted_news']) > 0:
                print(f"  First formatted news (preview): {sample['formatted_news'][0][:100]}...")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()

