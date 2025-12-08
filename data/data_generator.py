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

                # 3. 解析 input 中的新闻为 dict 格式
                candidate_news = self._parse_news_from_input(sample["input"])

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

    def _parse_news_from_input(self, input_text):
        """
        解析input文本,提取新闻为dict列表格式。
        
        示例input格式:
        "...In 2019-01-03 16:40:00, the news that had the Long-Term Effect..."
        
        返回格式:
        [{"summary": "...", "publication_time": "2019-01-03 16:40:00", "category": "Long-Term"}]
        """
        import re
        
        news_list = []
        
        # 匹配模式: "In YYYY-MM-DD HH:MM:SS, the news ... is that CONTENT"
        pattern = r'In (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), the news.*?is that ([^.]+\.)'
        matches = re.findall(pattern, input_text)
        
        for time_str, summary in matches:
            news_list.append({
                "summary": summary.strip(),
                "publication_time": time_str,
                "category": "News"  # 可以进一步解析Long-Term/Short-Term等
            })
        
        # 如果没有匹配到新闻,返回整段input作为单条新闻
        if not news_list:
            news_list = [{
                "summary": input_text[:500],  # 截取前500字符
                "publication_time": "Unknown",
                "category": "General"
            }]
        
        return news_list


if __name__ == "__main__":
    generator = DataGenerator(base_config, filepath="data/real_dataset.json")
    dataset = generator.load_dataset()

    print("\n--- Sample 0 ---")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

    print(f"\nSuccessfully loaded {len(dataset)} samples.")
