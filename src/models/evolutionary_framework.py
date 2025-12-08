# File: src/models/evolutionary_framework.py (Modified for NewsSelector Integration)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from typing import List, Tuple
from llm_egt_forecaster.configs import base_config
from llm_egt_forecaster.src.agent import Agent
# --- NEW: Import the NewsSelector ---
from llm_egt_forecaster.src.models.news_selector import NewsSelector

class EvolutionaryFramework(nn.Module):
    """
    The main framework that orchestrates the competition-driven evolution of LLM agents
    for time series forecasting, as described in the paper (Figure 1).
    (Unchanged docstring)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.DEVICE

        # 1. --- Load Shared Components with Quantization ---
        # (This part is unchanged from the previous version)
        # --- Conditional model loading (quantized vs non-quantized) ---
        model_kwargs = {"trust_remote_code": True}
        if config.USE_QUANTIZATION:
            print(f"Loading base LLM: {config.BASE_LLM_MODEL} with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            print(f"Loading base LLM: {config.BASE_LLM_MODEL} without quantization...")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_LLM_MODEL, **model_kwargs
        )

        if not config.USE_QUANTIZATION:
            self.base_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_PATH)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        if config.USE_QUANTIZATION:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        self.base_model.eval()

        # A simple linear head (unchanged)
        llm_hidden_size = self.base_model.config.hidden_size
        self.prediction_head = nn.Linear(llm_hidden_size, config.FUTURE_STEPS)
        self.prediction_head.to(self.device)

        # 2. --- Initialize the Agent Population --- (unchanged)
        lora_config = LoraConfig(
            r=config.LORA_RANK, lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES, lora_dropout=config.LORA_DROPOUT,
            bias="none", task_type="CAUSAL_LM"
        )
        print(f"Initializing {config.NUM_AGENTS} agents...")
        self.agents = nn.ModuleList([
            Agent(agent_id=i, base_llm_model=self.base_model, lora_config=lora_config, device=self.device)
            for i in range(config.NUM_AGENTS)
        ])\r
        \r
        # --- NEW: Initialize the NewsSelector with config parameters ---\r
        print(f"Initializing NewsSelector with method: {config.NEWS_SELECTOR_METHOD}")\r
        self.news_selector = NewsSelector(\r
            method=config.NEWS_SELECTOR_METHOD,\r
            model_name=config.NEWS_COSINE_MODEL,\r
            device=self.device,\r
            threshold=config.NEWS_COSINE_THRESHOLD,\r
            top_k=config.NEWS_COSINE_TOP_K,\r
            api_key=config.NEWS_API_KEY,\r
            api_base=config.NEWS_API_BASE,\r
            api_model=config.NEWS_API_MODEL\r
        )

    # --- REPLACED: The old _select_news_for_agent method is now completely replaced ---
    # The new logic is now handled by the self.news_selector instance.
    
    def forward(self, batch):
        """
        Performs a full forward pass of the evolutionary cycle for a batch of data.
        Simplified to call each Agent module directly.
        """
        time_series_strs, candidate_news_batch, ground_truths = self._prepare_batch(batch)

        batch_size = len(time_series_strs)
        agent_predictions = []
        agent_rewards = []

        for agent in self.agents:
            prompts = [
                self.config.PROMPT_TEMPLATE.format(
                    future_steps=self.config.FUTURE_STEPS,
                    time_series=time_series_strs[i],
                    selected_news=self.news_selector.select(agent.logic, candidate_news_batch[i])
                ) for i in range(batch_size)
            ]
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
            ).to(self.device)

            agent.train()
            outputs = agent(**inputs, output_hidden_states=True)

            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            prediction = self.prediction_head(last_hidden_state)
            agent_predictions.append(prediction)

            reward = -F.mse_loss(prediction, ground_truths, reduction='none').mean(dim=1)
            agent_rewards.append(reward)

        agent_predictions_tensor = torch.stack(agent_predictions)
        agent_rewards_tensor = torch.stack(agent_rewards)

        agent_fitnesses = torch.tensor([agent.fitness for agent in self.agents], device=self.device)
        agent_gates = torch.stack([agent.gate for agent in self.agents])

        weights_logits = agent_gates * agent_fitnesses.unsqueeze(1).expand_as(agent_rewards_tensor)
        weights = F.softmax(weights_logits / self.config.TEMPERATURE, dim=0)

        agg_prediction = torch.sum(weights.unsqueeze(2) * agent_predictions_tensor, dim=0)

        return {
            "aggregated_prediction": agg_prediction,
            "agent_predictions": agent_predictions_tensor,
            "agent_rewards": agent_rewards_tensor,
            "agent_gates": agent_gates
        }

    def _prepare_batch(self, batch) -> Tuple[List[str], List[List[str]], torch.Tensor]:
        """
        Safely prepares batch data, especially the list-of-lists `candidate_news`.
        """
        ground_truths = batch["ground_truth"].to(self.device)
        time_series_strs = batch["time_series_str"]

        candidate_news_transposed = batch["candidate_news"]

        if not candidate_news_transposed:
            return time_series_strs, [[] for _ in time_series_strs], ground_truths

        batch_size = len(time_series_strs)

        # Transpose back from collated shape [(n1..), (n2..), ...] -> [[...sample1...], [...sample2...], ...]
        try:
            candidate_news_batch = [list(item) for item in zip(*candidate_news_transposed)]
        except TypeError:
            # If elements are not iterable (shouldn't happen), fallback to per-sample wrapping
            candidate_news_batch = [candidate_news_transposed]

        # Safety for batch_size == 1 or odd collations
        if batch_size == 1 and len(candidate_news_batch) != 1:
            if isinstance(candidate_news_transposed[0], str):
                candidate_news_batch = [candidate_news_transposed]

        return time_series_strs, candidate_news_batch, ground_truths