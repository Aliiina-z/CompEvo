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
    for time series forecasting.
    
    Supports multi-GPU model parallelism for large models (e.g., 70B parameters).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.primary_device = config.DEVICE  # Primary device for inputs/outputs
        self.memory_breakdown = {}

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
        self._log_memory_breakdown("after_base_model", "Base model loaded")

        # --- Linear head for direct numerical prediction ---
        # Get hidden size from model config
        hidden_size = self.base_model.config.hidden_size if hasattr(self.base_model.config, 'hidden_size') else self.base_model.config.d_model
        self.prediction_head = nn.Linear(hidden_size, config.FUTURE_STEPS).to(self.primary_device)
        
        # Initialize prediction head to output reasonable values
        # Note: We will set bias dynamically based on historical data mean for each sample
        # So we initialize bias to 0, and add historical mean during forward pass
        with torch.no_grad():
            # Initialize weights with small values to prevent large initial outputs
            nn.init.xavier_uniform_(self.prediction_head.weight, gain=0.01)
            # Initialize bias to 0, since we'll add historical mean dynamically
            self.prediction_head.bias.fill_(0.0)
        self._log_memory_breakdown("after_prediction_head", "Prediction head initialized")

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
        self._log_memory_breakdown("after_agents", "Agent population initialized")
        self._log_agent_memory_estimate()
        
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
        self._log_memory_breakdown("after_news_selector", "NewsSelector initialized")

    # --- REPLACED: The old _select_news_for_agent method is now completely replaced ---
    # The new logic is now handled by the self.news_selector instance.

    def _cuda_memory_snapshot_gb(self):
        if not torch.cuda.is_available():
            return None

        torch.cuda.synchronize()
        allocated = 0.0
        reserved = 0.0
        per_device = []
        for device_idx in range(torch.cuda.device_count()):
            allocated_i = torch.cuda.memory_allocated(device_idx) / (1024 ** 3)
            reserved_i = torch.cuda.memory_reserved(device_idx) / (1024 ** 3)
            allocated += allocated_i
            reserved += reserved_i
            per_device.append({
                "device": device_idx,
                "allocated_gb": allocated_i,
                "reserved_gb": reserved_i,
            })

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "per_device": per_device,
        }

    def _log_memory_breakdown(self, key, label):
        snapshot = self._cuda_memory_snapshot_gb()
        if snapshot is None:
            return

        self.memory_breakdown[key] = snapshot
        print(
            f"[MEMORY_BREAKDOWN] {label}: "
            f"allocated={snapshot['allocated_gb']:.4f} GB, "
            f"reserved={snapshot['reserved_gb']:.4f} GB"
        )
        if len(snapshot["per_device"]) > 1:
            for item in snapshot["per_device"]:
                print(
                    f"[MEMORY_BREAKDOWN]   cuda:{item['device']}: "
                    f"allocated={item['allocated_gb']:.4f} GB, "
                    f"reserved={item['reserved_gb']:.4f} GB"
                )

    def _log_agent_memory_estimate(self):
        base = self.memory_breakdown.get("after_base_model")
        before_agents = self.memory_breakdown.get("after_prediction_head")
        after_agents = self.memory_breakdown.get("after_agents")
        num_agents = max(1, int(getattr(self.config, "NUM_AGENTS", 1)))

        if not (base and before_agents and after_agents):
            return

        extra_allocated = after_agents["allocated_gb"] - before_agents["allocated_gb"]
        extra_reserved = after_agents["reserved_gb"] - before_agents["reserved_gb"]
        per_agent_allocated = extra_allocated / num_agents
        per_agent_reserved = extra_reserved / num_agents

        print("========== Agent Memory Estimate ==========")
        print(f"Base model memory allocated: {base['allocated_gb']:.4f} GB")
        print(f"Base model memory reserved: {base['reserved_gb']:.4f} GB")
        print(f"Agent population extra allocated: {extra_allocated:.4f} GB")
        print(f"Agent population extra reserved: {extra_reserved:.4f} GB")
        print(f"Estimated per-agent extra allocated: {per_agent_allocated:.4f} GB")
        print(f"Estimated per-agent extra reserved: {per_agent_reserved:.4f} GB")
        print(
            f"Estimated 10-agent allocated from current run: "
            f"{before_agents['allocated_gb'] + per_agent_allocated * 10:.4f} GB"
        )
        print(
            f"Estimated 10-agent reserved from current run: "
            f"{before_agents['reserved_gb'] + per_agent_reserved * 10:.4f} GB"
        )
        print("===========================================")
    
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
    
    def _build_prompt(self, time_series_str, metadata, selected_news):
        """
        Build comprehensive prompt with all metadata and magnitude guidance.
        Format follows user's specified structure.
        
        Args:
            time_series_str: Comma-separated string of historical load data values
            metadata: Dict with keys like 'date', 'region', 'is_weekend', 'is_holiday', 
                     'holiday_info', 'weather', etc.
            selected_news: String of selected news
        
        Returns:
            Complete formatted prompt string
        """
        prompt_parts = []
        
        # 1. Start with historical load data
        prompt_parts.append(f"The historical load data is: {time_series_str}")
        
        # 2. Context information with all metadata
        context_parts = ["Based on the historical load data, please predict the load consumption in the next day."]
        
        # Region information
        if metadata and 'region' in metadata and metadata['region'] and metadata['region'] != 'Unknown':
            region = metadata['region']
            context_parts.append(f"The region for prediction is {region}.")
        else:
            context_parts.append("The region for prediction is not specified.")
        
        # Date information for historical data
        if metadata and 'date' in metadata:
            date_str = metadata['date']
            # Try to parse date to get weekday/weekend
            try:
                from datetime import datetime
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                day_type = "Weekend" if date_obj.weekday() >= 5 else "Weekday"
            except:
                day_type = metadata.get('is_weekend', False)
                day_type = "Weekend" if day_type else "Weekday"
            
            context_parts.append(f"The start date of historical data was on {date_str} that is {day_type}.")
            
            # Holiday information for historical data
            if metadata.get('is_holiday') or (metadata.get('holiday_info') and 'holiday' in str(metadata.get('holiday_info', '')).lower()):
                holiday_info = metadata.get('holiday_info', 'a public holiday')
                context_parts.append(f"and it is {holiday_info}.")
            else:
                context_parts.append("and it is not a public holiday.")
        else:
            context_parts.append("The start date of historical data is not specified.")
        
        # Data frequency and coverage
        context_parts.append("The data frequency is 30 minutes per point.")
        time_series_length = len(time_series_str.split(','))
        days_covered = time_series_length / 48.0
        context_parts.append(f"Historical data covers {days_covered:.1f} day{'s' if days_covered != 1 else ''}.")
        
        # Prediction date (assume next day after historical data)
        if metadata and 'date' in metadata:
            try:
                from datetime import datetime, timedelta
                date_obj = datetime.strptime(metadata['date'], '%Y-%m-%d')
                prediction_date = date_obj + timedelta(days=1)
                prediction_date_str = prediction_date.strftime('%Y-%m-%d')
                prediction_day_type = "Weekend" if prediction_date.weekday() >= 5 else "Weekday"
                context_parts.append(f"The date of prediction is on {prediction_date_str} that is {prediction_day_type}.")
                context_parts.append("and it is not a public holiday.")
            except:
                pass
        
        # Weather information
        if metadata and 'weather' in metadata and isinstance(metadata['weather'], dict):
            w = metadata['weather']
            weather_parts = []
            if 'min_temp' in w:
                weather_parts.append(f"the minimum temperature is {w['min_temp']}")
            if 'max_temp' in w:
                weather_parts.append(f"the maximum temperature is {w['max_temp']}")
            if 'humidity' in w:
                weather_parts.append(f"the humidity is {w['humidity']}")
            if 'pressure' in w:
                weather_parts.append(f"the pressure is {w['pressure']}")
            
            if weather_parts:
                context_parts.append(f"Weather of the start date: {'; '.join(weather_parts)}.")
                context_parts.append(f"Weather of the prediction date: {'; '.join(weather_parts)}.")
        
        # Selected news
        if selected_news:
            context_parts.append(selected_news)
        else:
            context_parts.append("No relevant news available.")
        
        prompt_parts.append("\n\n" + " ".join(context_parts))
        
        # 3. Task description
        prompt_parts.append(f"\n\nTask: Predict the electricity load consumption for the next {self.config.FUTURE_STEPS} time points (each point represents 30 minutes).")
        
        # 4. Magnitude guidance
        # Calculate statistics from historical data
        try:
            values = [float(x.strip()) for x in time_series_str.split(',') if x.strip()]
            if values:
                min_val = min(values)
                max_val = max(values)
                mean_val = sum(values) / len(values)
                typical_min = min_val * 0.8
                typical_max = max_val * 1.2
            else:
                min_val = max_val = mean_val = None
                typical_min = 3000
                typical_max = 10000
        except:
            min_val = max_val = mean_val = None
            typical_min = 3000
            typical_max = 10000
        
        prompt_parts.append("\n\nMagnitude Guidance:")
        if values and min_val is not None:
            prompt_parts.append(f"- Historical data range: {min_val:.1f} to {max_val:.1f} MW")
            prompt_parts.append(f"- Historical average: {mean_val:.1f} MW")
            prompt_parts.append(f"- Expected prediction range: approximately {typical_min:.1f} to {typical_max:.1f} MW")
            prompt_parts.append(f"- Predictions should be in the same magnitude order as historical values.")
        
        # Add region-specific typical ranges if available
        if metadata and 'region' in metadata:
            region = metadata['region']
            region_ranges = {
                'NSW': (4000, 12000),
                'VIC': (3000, 10000),
                'QLD': (4000, 9000),
                'SA': (600, 2000),
                'WA': (2000, 4000),
                'TAS': (800, 1500),
                'ACT': (200, 500),
                'NT': (100, 400)
            }
            if region in region_ranges:
                reg_min, reg_max = region_ranges[region]
                prompt_parts.append(f"- Typical range for {region}: {reg_min} to {reg_max} MW")
        
        # 5. Important notes
        prompt_parts.append("\n\nImportant: Your predictions must be numerical values in the same magnitude order as the historical data.")
        prompt_parts.append("Each prediction should be a real number representing electricity load in MW (megawatts).")
        
        return "".join(prompt_parts)
    
    def forward(self, batch):
        """
        Performs a full forward pass of the evolutionary cycle for a batch of data.
        Uses Linear head for direct numerical prediction.
        """
        # Initialize debug flag for first batch
        if not hasattr(self, '_debug_first_batch'):
            self._debug_first_batch = False
        
        time_series_strs, candidate_news_batch, ground_truths, metadata_list, _ = self._prepare_batch(batch)

        batch_size = len(time_series_strs)
        agent_predictions = []
        agent_rewards = []
        all_prompts = []  # Store prompts for each agent
        
        # Calculate historical data statistics for each sample to adjust predictions
        historical_means = []
        for time_series_str in time_series_strs:
            try:
                # Parse historical values from string
                values = [float(x.strip()) for x in time_series_str.split(',') if x.strip()]
                if values:
                    hist_mean = sum(values) / len(values)
                else:
                    hist_mean = 7500.0  # Fallback to default
            except:
                hist_mean = 7500.0  # Fallback to default
            historical_means.append(hist_mean)
        
        historical_means_tensor = torch.tensor(historical_means, device=self.primary_device, dtype=torch.float32)
        
        for agent_idx, agent in enumerate(self.agents):
            prompts = []
            for i in range(batch_size):
                metadata = metadata_list[i] if i < len(metadata_list) else {}
                selected_news = self.news_selector.select(agent.logic, candidate_news_batch[i])
                time_series_str = time_series_strs[i]
                
                # Build comprehensive prompt with all metadata and magnitude guidance
                prompt = self._build_prompt(
                    time_series_str=time_series_str,
                    metadata=metadata,
                    selected_news=selected_news
                )
                prompts.append(prompt)
            
            all_prompts.append(prompts)
            
            # Tokenize prompts
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.MAX_LENGTH
            ).to(self.primary_device)

            # Debug: Check prompt and tokenization (only for first batch and first agent)
            if agent_idx == 0 and hasattr(self, '_debug_first_batch'):
                if not self._debug_first_batch:
                    print(f"\n[DEBUG] Agent {agent_idx} - Prompt and Tokenization Check:")
                    print(f"  Number of prompts: {len(prompts)}")
                    print(f"  First prompt length: {len(prompts[0])} characters")
                    print(f"  First prompt (first 200 chars): {prompts[0][:200]}...")
                    print(f"  Input IDs shape: {inputs['input_ids'].shape}")
                    print(f"  Input IDs (first sample, first 20 tokens): {inputs['input_ids'][0, :20].tolist()}")
                    print(f"  Input IDs (first sample, last 20 tokens): {inputs['input_ids'][0, -20:].tolist()}")
                    # Decode to verify
                    decoded = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
                    print(f"  Decoded input (first 300 chars): {decoded[:300]}...")
                    print(f"  Attention mask (first sample): {inputs.get('attention_mask', None)[0, :20].tolist() if 'attention_mask' in inputs else 'N/A'}")
                    print(f"  Non-padding tokens count: {(inputs['input_ids'][0] != self.tokenizer.pad_token_id).sum().item()}")
                    self._debug_first_batch = True

            agent.train()
            
            # Forward pass to get hidden states
            outputs = agent.lora_model(**inputs, output_hidden_states=True)

            # IMPORTANT: Get the last NON-PADDING token's hidden state for each sample
            # If we use the last token directly, it might be a padding token, resulting in zero hidden state
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                # Find the last non-padding token index for each sample
                seq_lengths = attention_mask.sum(dim=1) - 1  # -1 because index is 0-based
                batch_indices = torch.arange(batch_size, device=self.primary_device)
                last_hidden_state = outputs.hidden_states[-1][batch_indices, seq_lengths, :]  # (batch_size, hidden_size)
            else:
                # Fallback: use last token (might be padding)
                last_hidden_state = outputs.hidden_states[-1][:, -1, :]  # (batch_size, hidden_size)
            
            # Debug: Check if we're using padding tokens
            if agent_idx == 0 and hasattr(self, '_debug_first_batch'):
                if self._debug_first_batch:
                    if attention_mask is not None:
                        last_token_ids = inputs['input_ids'][batch_indices, seq_lengths]
                        print(f"  Last non-padding token IDs (first 5 samples): {last_token_ids[:5].tolist()}")
                        print(f"  Last token is padding? (first sample): {last_token_ids[0].item() == self.tokenizer.pad_token_id}")
                    else:
                        print(f"  WARNING: No attention mask! Using last token which might be padding.")
                    self._debug_first_batch = False  # Set to False after first print
            
            # Debug: Check hidden state range (only for first batch and first agent)
            if agent_idx == 0 and hasattr(self, '_debug_first_batch'):
                if self._debug_first_batch:
                    print(f"\n[DEBUG] Agent {agent_idx} - Hidden state stats:")
                    print(f"  Hidden state shape: {last_hidden_state.shape}")
                    print(f"  Range: {last_hidden_state.min().item():.4f} to {last_hidden_state.max().item():.4f}")
                    print(f"  Mean: {last_hidden_state.mean().item():.4f}, Std: {last_hidden_state.std().item():.4f}")
                    print(f"  Linear Head weight range: {self.prediction_head.weight.min().item():.4f} to {self.prediction_head.weight.max().item():.4f}")
                    print(f"  Linear Head bias: {self.prediction_head.bias[0].item():.4f}")
                    self._debug_first_batch = False  # Set to False after first print
            
            # Linear head prediction: W @ h + bias
            # Since bias is initialized to 0, we add historical mean directly
            prediction_base = self.prediction_head(last_hidden_state)  # (batch_size, FUTURE_STEPS)
            
            # Add historical mean as dynamic bias for each sample
            # This centers predictions around historical data mean instead of 0
            prediction = prediction_base + historical_means_tensor.unsqueeze(1)  # (batch_size, FUTURE_STEPS)
            
            # Debug: Check prediction range (only for first batch and first agent)
            if agent_idx == 0 and hasattr(self, '_debug_first_batch'):
                if not self._debug_first_batch:  # This will be False after first hidden state check
                    print(f"  Base prediction (W@h) range: {prediction_base.min().item():.4f} to {prediction_base.max().item():.4f}")
                    print(f"  Base prediction mean: {prediction_base.mean().item():.4f}")
                    print(f"  Historical means (first 3 samples): {historical_means_tensor[:3].tolist()}")
                    print(f"  Final prediction range: {prediction.min().item():.4f} to {prediction.max().item():.4f}")
                    print(f"  Final prediction mean: {prediction.mean().item():.4f}")
                    print(f"  Final prediction (first sample, first 10 values): {prediction[0, :10].tolist()}")
                    print(f"  Ground truth range: {ground_truths.min().item():.4f} to {ground_truths.max().item():.4f}")
                    print(f"  Ground truth (first sample, first 10 values): {ground_truths[0, :10].tolist()}\n")
            
            # Use prediction with historical mean for agent predictions and reward calculation
            agent_predictions.append(prediction)

            # Calculate reward (negative loss, matching the loss type used in training)
            # Get loss type from config (default: 'mse')
            loss_type = getattr(self.config, 'PREDICTION_LOSS_TYPE', 'mse').lower()
            
            if loss_type == 'mae':
                # Negative MAE as reward
                reward = -F.l1_loss(prediction, ground_truths, reduction='none').mean(dim=1)
            elif loss_type == 'mape':
                # Negative MAPE as reward
                epsilon = 1e-8
                abs_error = torch.abs(prediction - ground_truths)
                abs_true = torch.abs(ground_truths) + epsilon
                percentage_error = (abs_error / abs_true).mean(dim=1)
                reward = -percentage_error
            elif loss_type == 'rmse':
                # Negative RMSE as reward
                mse = F.mse_loss(prediction, ground_truths, reduction='none')
                rmse = torch.sqrt(mse.mean(dim=1))
                reward = -rmse
            elif loss_type == 'huber':
                # Negative Huber loss as reward
                delta = getattr(self.config, 'HUBER_DELTA', 1.0)
                huber = F.smooth_l1_loss(prediction, ground_truths, reduction='none', beta=delta).mean(dim=1)
                reward = -huber
            else:
                # Default: Negative MSE as reward
                reward = -F.mse_loss(prediction, ground_truths, reduction='none').mean(dim=1)
            
            agent_rewards.append(reward)

        # Stack agent predictions and rewards
        agent_predictions_tensor = torch.stack(agent_predictions)  # (num_agents, batch_size, FUTURE_STEPS)
        agent_rewards_tensor = torch.stack(agent_rewards)  # (num_agents, batch_size)

        agent_fitnesses = torch.tensor([agent.fitness for agent in self.agents], device=self.primary_device, dtype=torch.float32)
        agent_gates = torch.stack([agent.gate for agent in self.agents])

        weights_logits = agent_gates.unsqueeze(1) * agent_fitnesses.unsqueeze(1).expand_as(agent_rewards_tensor)
        weights = F.softmax(weights_logits / self.config.TEMPERATURE, dim=0)

        agg_prediction = torch.sum(weights.unsqueeze(2) * agent_predictions_tensor, dim=0)

        return {
            "aggregated_prediction": agg_prediction,
            "agent_predictions": agent_predictions_tensor,
            "agent_rewards": agent_rewards_tensor,
            "agent_gates": agent_gates,
            "agent_prompts": all_prompts,
        }

    def _prepare_batch(self, batch) -> Tuple[List[str], List[List[str]], torch.Tensor, List[dict], None]:
        """
        Safely prepares batch data, especially the list-of-lists `candidate_news`.
        Returns: (time_series_strs, candidate_news_batch, ground_truths, metadata_list, None)
        """
        ground_truths = batch["ground_truth"].to(self.primary_device)
        time_series_strs = batch["time_series_str"]
        batch_size = len(time_series_strs)
        
        # Extract metadata if available (not used in Linear head approach, but kept for compatibility)
        metadata_list = batch.get("metadata", [{} for _ in range(batch_size)])
        if len(metadata_list) < batch_size:
            metadata_list.extend([{} for _ in range(batch_size - len(metadata_list))])
        elif len(metadata_list) > batch_size:
            metadata_list = metadata_list[:batch_size]

        candidate_news_transposed = batch.get("candidate_news", None)

        # Handle empty or missing candidate_news
        if not candidate_news_transposed:
            return time_series_strs, [[] for _ in range(batch_size)], ground_truths, metadata_list, None

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

        return time_series_strs, candidate_news_batch, ground_truths, metadata_list, None
