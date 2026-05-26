# File: data/data_generator.py

import json
import os
import re

from tqdm import tqdm

from llm_egt_forecaster.configs import base_config


class DataGenerator:
    """
    Load and optionally convert datasets into the schema consumed by
    NewsTimeSeriesDataset.
    """

    def __init__(self, config, filepath="data/AULF_train_data_2019-2020.json"):
        self.config = config
        self.filepath = filepath

    def load_dataset(self):
        """Load a JSON dataset and convert instruction/input/output samples if needed."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Real dataset not found at {self.filepath}. "
                "Please provide a valid dataset file."
            )

        print(f"Loading real dataset from {self.filepath}...")
        with open(self.filepath, "r", encoding="utf-8") as f:
            raw_dataset = json.load(f)

        if raw_dataset:
            first_sample = raw_dataset[0]
            has_standard_fields = all(
                key in first_sample
                for key in ("time_series", "candidate_news", "ground_truth")
            )
            if has_standard_fields:
                print("Data is already in converted format. Skipping conversion.")
                return raw_dataset

        converted_samples = []
        for i, sample in enumerate(tqdm(raw_dataset, desc="Converting samples")):
            try:
                ts_str = (
                    sample["instruction"]
                    .split("The historical load data is:")[-1]
                    .split("The region")[0]
                    .strip()
                )
                time_series = [float(x) for x in ts_str.split(",")]
                ground_truth = [float(x) for x in sample["output"].split(",")]
                candidate_news = self._parse_news_from_input(sample["input"])

                converted_samples.append(
                    {
                        "time_series": time_series,
                        "candidate_news": candidate_news,
                        "ground_truth": ground_truth,
                    }
                )
            except Exception as exc:
                print(f"[Warning] Failed to convert sample {i}: {exc}")
                continue

        print(
            f"Successfully converted {len(converted_samples)} samples "
            f"out of {len(raw_dataset)} total."
        )

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(converted_samples, f, indent=2, ensure_ascii=False)

        return converted_samples

    def _parse_news_from_input(self, input_text):
        """
        Extract news items from the prompt input text.

        Expected pattern:
        "In YYYY-MM-DD HH:MM:SS, the news ... is that CONTENT"
        """
        news_list = []
        pattern = r"In (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), the news.*?is that ([^.]+\.)"
        matches = re.findall(pattern, input_text)

        for time_str, summary in matches:
            news_list.append(
                {
                    "summary": summary.strip(),
                    "publication_time": time_str,
                    "category": "News",
                }
            )

        if not news_list:
            news_list = [
                {
                    "summary": input_text[:500],
                    "publication_time": "Unknown",
                    "category": "General",
                }
            ]

        return news_list


if __name__ == "__main__":
    generator = DataGenerator(base_config, filepath="data/real_dataset.json")
    dataset = generator.load_dataset()

    print("\n--- Sample 0 ---")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

    print(f"\nSuccessfully loaded {len(dataset)} samples.")
