# File: data/data_generator.py

import os
import json
from tqdm import tqdm

from llm_egt_forecaster.configs import base_config


class DataGenerator:
    """
    Loads and converts real dataset (instruction/input/output style)
    into the standard format required by NewsTimeSeriesDataset.
    """

    def __init__(self, config, filepath="data/AULF_train_data_2019-2020.json"):
        self.config = config
        self.filepath = filepath

    def load_dataset(self):
        """Load dataset from a JSON file, convert to standard schema, then rewrite cleaned JSON."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Real dataset not found at {self.filepath}. "
                "Please provide a valid dataset file."
            )

        print(f"Loading real dataset from {self.filepath}...")
        with open(self.filepath, "r", encoding="utf-8") as f:
            raw_dataset = json.load(f)

        converted_samples = []
        for i, sample in enumerate(tqdm(raw_dataset, desc="Converting samples")):
            try:
                # 1. 从 instruction 里提取历史负荷序列
                ts_str = sample["instruction"].split("The historical load data is:")[-1].split("The region")[0].strip()
                time_series = [float(x) for x in ts_str.split(",")]

                # 2. 从 output 里提取未来负荷序列
                ground_truth = [float(x) for x in sample["output"].split(",")]

                # 3. candidate_news：这里用 input 整段文字代替
                candidate_news = [sample["input"]]

                # 4. 构造标准样本
                converted_samples.append({
                    "time_series": time_series,
                    "candidate_news": candidate_news,
                    "ground_truth": ground_truth
                })
            except Exception as e:
                print(f"[Warning] Failed to convert sample {i}: {e}")
                continue

        print(f"Successfully converted {len(converted_samples)} samples "
              f"out of {len(raw_dataset)} total.")

        # 覆盖写回文件
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(converted_samples, f, indent=2, ensure_ascii=False)

        return converted_samples


if __name__ == "__main__":
    generator = DataGenerator(base_config, filepath="data/real_dataset.json")
    dataset = generator.load_dataset()

    print("\n--- Sample 0 ---")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

    print(f"\nSuccessfully loaded {len(dataset)} samples.")
