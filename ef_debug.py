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
    
    Supports multi-GPU model parallelism for large models (e.g., 70B parameters).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.primary_device = config.DEVICE  # Primary device for inputs/outputs

        # 1. --- Load Shared Components with Quantization + Multi-GPU Support ---
        # Build device map for multi-GPU support
        device_map = self._build_device_map(config)
        self.device_map = device_map
        self.is_multi_gpu = device_map not in [None, {"":"cuda:0"}, {"":self.primary_device}]
        
        if self.is_multi_gpu:
            print(f"Multi-GPU mode enabled with device_map: {device_map}")
        
        model_kwargs = {"trust_remote_code": True}
        if config.USE_QUANTIZATION:
            print(f"Loading base LLM: {config.BASE_LLM_MODEL} with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = device_map
        else:
            print(f"Loading base LLM: {config.BASE_LLM_MODEL} without quantization...")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_LLM_MODEL, **model_kwargs
        )

        if not config.USE_QUANTIZATION:
            self.base_model.to(self.primary_device)
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
        self.prediction_head.to(self.primary_device)

        # 2. --- Initialize the Agent Population --- (unchanged)
        lora_config = LoraConfig(
            r=config.LORA_RANK, lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES, lora_dropout=config.LORA_DROPOUT,
            bias="none", task_type="CAUSAL_LM"
        )
        print(f"Initializing {config.NUM_AGENTS} agents...")
        self.agents = nn.ModuleList([
            Agent(agent_id=i, base_llm_model=self.base_model, lora_config=lora_config, device=self.primary_device)
            for i in range(config.NUM_AGENTS)
        ])
        
        # --- NEW: Initialize the NewsSelector with config parameters ---
        print(f"Initializing NewsSelector with method: {config.NEWS_SELECTOR_METHOD}")
        self.news_selector = NewsSelector(
            method=config.NEWS_SELECTOR_METHOD,
            model_name=config.NEWS_COSINE_MODEL,
            device=self.primary_device,
            threshold=config.NEWS_COSINE_THRESHOLD,
            top_k=config.NEWS_COSINE_TOP_K,
            api_key=config.NEWS_API_KEY,
            api_base=config.NEWS_API_BASE,
            api_model=config.NEWS_API_MODEL
        )

    # --- REPLACED: The old _select_news_for_agent method is now completely replaced ---
    # The new logic is now handled by the self.news_selector instance.
    
    def _build_device_map(self, config):
        """
        Build device map based on multi-GPU configuration.
        
        Returns:
            device_map: Can be 'auto', 'balanced', 'sequential', or a dict for single GPU
        """
        import os
        
        num_gpus = getattr(config, 'NUM_GPUS', 1)
        strategy = getattr(config, 'DEVICE_MAP_STRATEGY', 'balanced')
        visible_gpus = getattr(config, 'VISIBLE_GPUS', None)
        
        # Set CUDA_VISIBLE_DEVICES if specified
        if visible_gpus is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus
            print(f"Setting CUDA_VISIBLE_DEVICES={visible_gpus}")
        
        if num_gpus <= 1:
            # Single GPU: put everything on the primary device
            print(f"Single GPU mode: all layers on {config.DEVICE}")
            return {"":config.DEVICE}
        
        # Multi-GPU: use specified strategy
        print(f"Multi-GPU mode: {num_gpus} GPUs with '{strategy}' strategy")
        return strategy  # 'balanced', 'auto', or 'sequential'
    
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
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.MAX_LENGTH
            ).to(self.primary_device)

            agent.train()
            outputs = agent(**inputs, output_hidden_states=True)

            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            prediction = self.prediction_head(last_hidden_state)
            agent_predictions.append(prediction)

            reward = -F.mse_loss(prediction, ground_truths, reduction='none').mean(dim=1)
            agent_rewards.append(reward)

        agent_predictions_tensor = torch.stack(agent_predictions)
        agent_rewards_tensor = torch.stack(agent_rewards)

        agent_fitnesses = torch.tensor([agent.fitness for agent in self.agents], device=self.primary_device)
        agent_gates = torch.stack([agent.gate for agent in self.agents])

        weights_logits = agent_gates.unsqueeze(1) * agent_fitnesses.unsqueeze(1).expand_as(agent_rewards_tensor)
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
        ground_truths = batch["ground_truth"].to(self.primary_device)
        time_series_strs = batch["time_series_str"]
        batch_size = len(time_series_strs)

        candidate_news_transposed = batch.get("candidate_news", None)

        # Handle empty or missing candidate_news
        if not candidate_news_transposed:
            return time_series_strs, [[] for _ in range(batch_size)], ground_truths

        # Transpose back from collated shape [(n1..), (n2..), ...] -> [[...sample1...], [...sample2...], ...]
        try:
            # Check if it's already in the correct format (list of lists)
            if isinstance(candidate_news_transposed, list) and len(candidate_news_transposed) > 0:
                if isinstance(candidate_news_transposed[0], list):
                    # Already in correct format: [[news1, news2], [news3, news4], ...]
                    candidate_news_batch = candidate_news_transposed
                elif isinstance(candidate_news_transposed[0], str):
                    # It's a flat list of news strings, treat as single sample's news
                    if batch_size == 1:
                        candidate_news_batch = [candidate_news_transposed]
                    else:
                        # Transposed format: [(news1_sample1, news1_sample2), (news2_sample1, news2_sample2), ...]
                        candidate_news_batch = [list(item) for item in zip(*candidate_news_transposed)]
                else:
                    # Try to transpose
                    candidate_news_batch = [list(item) for item in zip(*candidate_news_transposed)]
            else:
                candidate_news_batch = [[] for _ in range(batch_size)]
        except (TypeError, ValueError) as e:
            # If transpose fails, create empty lists
            import logging
            logging.getLogger(__name__).warning(f"Failed to process candidate_news: {e}")
            candidate_news_batch = [[] for _ in range(batch_size)]

        # CRITICAL: Ensure candidate_news_batch has exactly batch_size elements
        if len(candidate_news_batch) != batch_size:
            import logging
            logging.getLogger(__name__).warning(
                f"candidate_news_batch length ({len(candidate_news_batch)}) != batch_size ({batch_size}). Padding/truncating."
            )
            # Pad with empty lists if too short
            while len(candidate_news_batch) < batch_size:
                candidate_news_batch.append([])
            # Truncate if too long
            candidate_news_batch = candidate_news_batch[:batch_size]

        return time_series_strs, candidate_news_batch, ground_truths