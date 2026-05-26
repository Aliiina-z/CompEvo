# File: src/models/logic_generator.py (Corrected and Completed)

import torch
import torch.nn as nn
import torch.nn.functional as F  # --- FIXED: Added missing import ---
from transformers import PreTrainedModel, AutoTokenizer
from llm_egt_forecaster.configs import base_config

class LogicGenerator(nn.Module):
    """
    Implements the Three-Stage Gated Logic Generation.
    (Docstring unchanged)
    """
    def __init__(self, base_model: PreTrainedModel, tokenizer: AutoTokenizer, config):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config
        
        # --- FIXED: Completed the prompt template ---
        self.prompt_template = """You are a strategy analyst for a time series forecasting agent. Your goal is to refine the agent's news selection logic to improve its prediction accuracy.

Here is the current context:
- Agent ID: {agent_id}
- Agent's Current Logic: "{current_logic}"
- Agent's Recent Performance (Reward): {last_reward:.4f} (Higher is better, negative values mean poor performance)

Here is information about the performance of other agents (peers):
{peer_info}

Reference: Factors that impact regional electricity load consumption include:

Positive Issues Leading to Increase in Load Consumption:

Short-Term:
1. Economic Growth: A surge in economic activity increases energy consumption.
2. Technological Advancements: New power-requiring technologies can spike demand.
3. Seasonal Factors: Extreme weather increases the use of air conditioning.
4. Social Events: Large-scale events temporarily boost energy use.

Long-Term:
1. Population Growth: Leads to higher residential energy consumption.
2. Industrial Development: Correlates with increased energy demands.
3. Urbanization: Expansion of cities contributes to higher energy usage.
4. Energy Transition: Shift towards electrically powered technologies.

Negative Issues Leading to Decrease in Load Consumption:

Short-Term:
1. Economic Downturns: Lead to decreased industrial activity and lower energy consumption.
2. Efficiency Improvements: Adoption of energy-efficient technologies reduces consumption.
3. Weather Patterns: Mild weather can reduce heating and cooling needs.
4. Public Health Crises: Can lead to reduced industrial and commercial activity.

Long-Term:
1. Energy Efficiency: Trends like better insulation and efficient appliances reduce consumption.
2. Demographic Changes: Aging populations or declining birth rates can lead to decreased energy use.
3. Policy and Regulation: Promote energy conservation and sustainability.
4. Technological Innovations: Development of more efficient technologies.

Other Factors:
- Political Stability: Impacts energy policies and investments.
- Global Market Dynamics: Affect local energy prices and consumption patterns.
- Environmental Consciousness: Leads to changes in consumption behavior and renewable energy adoption.


Based on all this information, generate a new, refined logic for the agent. The new logic should:
1. Be a single, concise sentence (20-80 words) - DO NOT copy the entire reference list above
2. Specify CONCRETE news categories or topics (e.g., "extreme weather", "economic indicators", "infrastructure disruptions", "energy policies")
3. Focus on WHAT types of news to select, not abstract methods (avoid terms like "meta-learning", "probabilistic forecasting", "scenario-based planning")
4. Be actionable for news filtering (e.g., "Focus on extreme weather events and power grid disruptions" is good)
5. Be an evolution of the current logic based on performance feedback
6. Synthesize a new logic rather than copying a peer's logic
New Refined Logic: \""""

    def _format_peer_info(self, all_agents, current_agent_id):
        """Formats the peer information for the prompt."""
        # --- FIXED: Completed the peer info formatting logic ---
        peers = [agent for agent in all_agents if agent.id != current_agent_id]
        if not peers:
            return "No other agents to compare with."

        # Find the best performing peer for inspiration
        best_peer = max(peers, key=lambda a: a.last_reward)
        
        info_str = "The best performing peer's strategy was:\n"
        info_str += f"- Peer Agent {best_peer.id}: Logic='{best_peer.logic}', Recent Reward={best_peer.last_reward:.4f}"
        
        return info_str

    @staticmethod
    def _looks_degenerate(text: str) -> bool:
        """
        Heuristic guard: GPT-2 style small models (and unstable LoRA updates) can
        collapse into repetitive token loops ("AboutAboutAbout..."). If so, we
        reject the new logic and keep the previous one to avoid breaking news selection.
        """
        t = (text or "").strip()
        if len(t) < 8:
            return True

        # Token-level repetition heuristic (whitespace split)
        words = [w for w in t.replace("\n", " ").split(" ") if w]
        if len(words) >= 8:
            from collections import Counter
            c = Counter(words)
            if (max(c.values()) / len(words)) > 0.45:
                return True

        # No whitespace at all and very long => often "token soup"
        if (" " not in t) and len(t) > 60:
            return True

        return False

    @staticmethod
    def _top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Keep only top_k tokens with highest logits; set others to -inf."""
        if top_k <= 0:
            return logits
        vocab = logits.size(-1)
        k = min(top_k, vocab)
        values, indices = torch.topk(logits, k, dim=-1)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(1, indices, values)
        return filtered

    def generate(self, agent_to_update, all_agents, with_log_prob=False):
        peer_info = self._format_peer_info(all_agents, agent_to_update.id)
        
        # --- FIXED: Completed prompt formatting call ---
        prompt = self.prompt_template.format(
            agent_id=agent_to_update.id,
            current_logic=agent_to_update.logic,
            last_reward=agent_to_update.last_reward,
            peer_info=peer_info
        )
        
        # Get the correct device for inputs
        # If model uses device_map="auto", need to find the primary device
        agent_lora_model = agent_to_update.lora_model
        
        # Determine the correct device for inputs
        # For multi-GPU models with device_map="auto", we need to find where the input embeddings are
        primary_device = None
        
        # Try to get device map from base model (PEFT wraps the base model)
        base_model = getattr(agent_lora_model, 'base_model', None)
        if base_model is None:
            base_model = getattr(agent_lora_model, 'model', agent_lora_model)
        
        # Check for device map
        device_map = None
        if hasattr(base_model, 'hf_device_map'):
            device_map = base_model.hf_device_map
        elif hasattr(agent_lora_model, 'hf_device_map'):
            device_map = agent_lora_model.hf_device_map
        
        if device_map:
            # Model is distributed across devices
            # For device_map="auto", we need to find where the input embeddings are
            # CRITICAL: Find the device where the first decoder layer's rotary_emb is located
            # This is where position_ids will be used, so inputs must be on this device
            try:
                # Try to access the first decoder layer and find rotary_emb device
                if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
                    # Try to get the first layer's rotary_emb device
                    first_layer = base_model.model.layers[0]
                    if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'rotary_emb'):
                        rotary_emb = first_layer.self_attn.rotary_emb
                        if hasattr(rotary_emb, 'inv_freq'):
                            primary_device = rotary_emb.inv_freq.device
                        else:
                            # Try to get device from any parameter in rotary_emb
                            for param in rotary_emb.parameters():
                                primary_device = param.device
                                break
                    else:
                        # Fallback: get device from first layer parameters
                        primary_device = next(first_layer.parameters()).device
                elif hasattr(base_model, 'get_input_embeddings'):
                    # Fallback: use input embeddings device
                    embed_layer = base_model.get_input_embeddings()
                    if hasattr(embed_layer, 'weight'):
                        primary_device = embed_layer.weight.device
                    else:
                        # Fallback: use first device in device_map
                        for module_path in sorted(device_map.keys()):
                            device = device_map[module_path]
                            if device not in ['cpu', 'disk']:
                                primary_device = device
                                break
                else:
                    # Fallback: use first device in device_map
                    for module_path in sorted(device_map.keys()):
                        device = device_map[module_path]
                        if device not in ['cpu', 'disk']:
                            primary_device = device
                            break
            except Exception as e:
                # If anything fails, use first non-CPU device from device_map
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not determine rotary_emb device, using device_map fallback: {e}")
                for module_path in sorted(device_map.keys()):
                    device = device_map[module_path]
                    if device not in ['cpu', 'disk']:
                        primary_device = device
                        break
            
            # Final fallback: use the first non-CPU device
            if primary_device is None:
                for device in device_map.values():
                    if device not in ['cpu', 'disk']:
                        primary_device = device
                        break
        else:
            # Model is on a single device, get device from parameters
            try:
                primary_device = next(agent_lora_model.parameters()).device
            except StopIteration:
                # If no parameters, fall back to config device
                primary_device = self.config.DEVICE
        
        # Final fallback
        if primary_device is None:
            primary_device = self.config.DEVICE
        
        # Log device information for debugging multi-GPU setups
        if device_map and len(set(device_map.values())) > 1:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"⚠️ Multi-GPU model detected: {set(device_map.values())}")
            logger.info(f"📍 Using primary device for inputs: {primary_device}")
            logger.info(f"📍 Device map: {dict(list(device_map.items())[:5])}...")  # Show first 5 entries
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(primary_device)
        
        if not with_log_prob:
            agent_lora_model.eval()
            # Clear CUDA cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Get generation parameters from config
            max_new_tokens = int(getattr(self.config, "LOGIC_MAX_NEW_TOKENS", 40))
            temperature = float(getattr(self.config, "LOGIC_TEMPERATURE", 0.8))
            top_k = int(getattr(self.config, "LOGIC_TOP_K", 50))
            
            with torch.no_grad():
                # Switch to logic adapter for generation (trainable via GRPO)
                agent_lora_model.set_adapter("logic")
                try:
                    outputs = agent_lora_model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else 1.0,
                        top_k=top_k if top_k > 0 else None,
                        do_sample=True,
                        num_beams=3,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                finally:
                    # Always switch back to prediction adapter
                    agent_lora_model.set_adapter("default")
            
            # Extract only the generated part (after the prompt)
            generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
            new_logic = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().split('\n')[0]
            
            return new_logic if new_logic else agent_to_update.logic
        
        # NOTE: For multi-GPU models with device_map="auto", manual loop generation
        # causes device mismatch issues with position_ids and rotary_emb.
        # Use model.generate() with output_scores=True instead to avoid device issues.
        agent_lora_model.eval()
        
        # Clear CUDA cache before generation to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        max_new_tokens = int(getattr(self.config, "LOGIC_MAX_NEW_TOKENS", 40))
        temperature = float(getattr(self.config, "LOGIC_TEMPERATURE", 0.8))
        top_k = int(getattr(self.config, "LOGIC_TOP_K", 50))

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        # Use model.generate() with output_scores=True to get logits and compute log_prob
        # This avoids device mismatch issues with device_map="auto"
        # The generate() method handles device transfers automatically for distributed models
        try:
            # Use generate with output_scores to get logits for each step
            # For multi-GPU models, generate() will automatically handle device placement
            # Switch to logic adapter for generation (trainable via GRPO)
            agent_lora_model.set_adapter("logic")
            try:
                generation_output = agent_lora_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_k=top_k if top_k > 0 else None,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2, # Prevent loop/underscore repetition
                    use_cache=False # Disable cache for compatibility
                )
            finally:
                # Always switch back to prediction adapter
                agent_lora_model.set_adapter("default")
            
            # Extract generated sequence and scores
            generated_sequence = generation_output.sequences[0]
            scores = generation_output.scores  # List of logits tensors
            
            # Compute log_prob from scores
            log_prob_sum = torch.tensor(0.0, device=input_ids.device)
            generated_ids = generated_sequence[input_ids.shape[1]:]  # Only newly generated tokens
            
            for i, (token_id, score_logits) in enumerate(zip(generated_ids, scores)):
                if token_id.item() == self.tokenizer.eos_token_id:
                    break
                
                # Apply temperature and top_k filtering to logits
                logits = score_logits[0]  # Remove batch dimension
                if temperature > 0:
                    logits = logits / temperature
                logits = self._top_k_filtering(logits.unsqueeze(0), top_k=top_k).squeeze(0)
                
                # Compute log probability of the sampled token
                probs = F.softmax(logits, dim=-1)
                log_prob = torch.log(probs[token_id.item()].clamp_min(1e-12))
                log_prob_sum = log_prob_sum + log_prob
            
            # Decode generated logic
            new_logic = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip().split("\n")[0]
            
            # Degeneracy guard: if the generated logic is garbage, keep old logic and don't update PG.
            if self._looks_degenerate(new_logic):
                return (agent_to_update.logic, torch.tensor(0.0, device=input_ids.device))
            
            return (new_logic if new_logic else agent_to_update.logic, log_prob_sum)
            
        except Exception as e:
            # Fallback to manual generation if generate() fails
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"generate() failed, falling back to manual loop: {e}")
            
            # Fallback: manual generation (may have device issues)
            log_prob_sum = torch.tensor(0.0, device=input_ids.device)
            generated_tokens = []
            
            for step in range(max_new_tokens):
                # Clear cache periodically to prevent memory buildup
                if torch.cuda.is_available() and step > 0 and step % 10 == 0:
                    torch.cuda.empty_cache()
                
                # For multi-GPU models with device_map="auto", ensure inputs are on the correct device
                # Critical: Always ensure inputs are on the primary device before model call
                # This prevents device mismatch errors with distributed models
                input_ids = input_ids.to(primary_device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(primary_device)
                
                # Call model forward
                # For device_map="auto" models, accelerate should handle device transfers automatically
                # Use prepare_inputs_for_generation if available to ensure proper device alignment
                try:
                    # Try to use prepare_inputs_for_generation if available (for better device handling)
                    if hasattr(agent_lora_model, 'prepare_inputs_for_generation'):
                        model_inputs = agent_lora_model.prepare_inputs_for_generation(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False
                        )
                        outputs = agent_lora_model(**model_inputs)
                    else:
                        # Fallback to direct call
                        outputs = agent_lora_model(input_ids=input_ids, attention_mask=attention_mask)
                except RuntimeError as e_inner:
                    if "Expected all tensors to be on the same device" in str(e_inner):
                        # Device mismatch error - this is a known issue with device_map="auto"
                        # Try using cuda:0 explicitly as fallback
                        logger.error(f"Device mismatch error: {e_inner}")
                        logger.error(f"Input device: {input_ids.device}, Expected: {primary_device}")
                        logger.error("This may be a bug in accelerate library with device_map='auto'")
                        logger.error("Trying fallback: moving inputs to cuda:0")
                        
                        # Force move to cuda:0 and try again
                        input_ids = input_ids.to('cuda:0')
                        if attention_mask is not None:
                            attention_mask = attention_mask.to('cuda:0')
                        
                        try:
                            outputs = agent_lora_model(input_ids=input_ids, attention_mask=attention_mask)
                        except RuntimeError as e2:
                            logger.error(f"Fallback also failed: {e2}")
                            raise RuntimeError(
                                f"Device mismatch error persists. This may indicate a bug in accelerate/transformers "
                                f"with device_map='auto'. Try using device_map='cuda:0' instead (single GPU mode)."
                            ) from e2
                    else:
                        raise
                
                logits = outputs.logits[:, -1, :]
                
                # Delete outputs immediately to free memory
                del outputs

                if temperature > 0:
                    logits = logits / temperature
                logits = self._top_k_filtering(logits, top_k=top_k)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens.append(next_token.item())

                # log_prob of sampled token (keep graph)
                log_prob = torch.log(probs.gather(1, next_token).clamp_min(1e-12))
                log_prob_sum += log_prob.squeeze()
                
                # Delete intermediate tensors
                del logits, probs
                
                # Ensure new token is on the primary device (important for multi-GPU models)
                next_token = next_token.to(primary_device)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                # Ensure input_ids stays on primary device after concatenation
                input_ids = input_ids.to(primary_device)
                
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
                    attention_mask = attention_mask.to(primary_device)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Truncate input_ids if getting too long to save memory
                if input_ids.shape[1] > 512:
                    # Keep only the last 256 tokens
                    input_ids = input_ids[:, -256:]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, -256:]
            
            # Decode generated logic from fallback path
            generated_ids = input_ids[0, inputs['input_ids'].shape[1]:]
            new_logic = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip().split("\n")[0]

            # Degeneracy guard: if the generated logic is garbage, keep old logic and don't update PG.
            if self._looks_degenerate(new_logic):
                return (agent_to_update.logic, torch.tensor(0.0, device=input_ids.device))
            
            return (new_logic if new_logic else agent_to_update.logic, log_prob_sum)