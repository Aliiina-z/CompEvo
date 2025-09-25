# File: src/models/logic_generator.py (Corrected and Completed)

import torch
import torch.nn as nn
import torch.nn.functional as F  # --- FIXED: Added missing import ---
from transformers import PreTrainedModel, AutoTokenizer
from llm_egt_forecaster.configs import base_config

class LogicGenerator(nn.Module):
    """
    Implements the Three-Stage Gated Logic Generation from the paper.
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

Based on all this information, generate a new, refined logic for the agent. The new logic should be a single, concise sentence. It should be an evolution of the current logic, either by specializing, generalizing, or correcting it based on the performance feedback. Do not just copy a peer's logic; synthesize a new one.

New Refined Logic:"""

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

    def generate(self, agent_to_update, all_agents, with_log_prob=False):
        peer_info = self._format_peer_info(all_agents, agent_to_update.id)
        
        # --- FIXED: Completed prompt formatting call ---
        prompt = self.prompt_template.format(
            agent_id=agent_to_update.id,
            current_logic=agent_to_update.logic,
            last_reward=agent_to_update.last_reward,
            peer_info=peer_info
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.DEVICE)
        agent_lora_model = agent_to_update.lora_model
        
        if not with_log_prob:
            agent_lora_model.eval()
            with torch.no_grad():
                # --- FIXED: Completed generation call with specific parameters ---
                outputs = agent_lora_model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=50,
                    num_beams=3,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Extract only the generated part (after the prompt)
            generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
            new_logic = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().split('\n')[0]
            
            return new_logic if new_logic else agent_to_update.logic
        
        agent_lora_model.train()
        max_new_tokens = 50
        input_ids = inputs['input_ids']
        log_prob_sum = 0
        
        for _ in range(max_new_tokens):
            outputs = agent_lora_model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            log_prob = torch.log(probs.gather(1, next_token))
            log_prob_sum += log_prob.squeeze()
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        generated_ids = input_ids[0, inputs['input_ids'].shape[1]:]
        new_logic = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().split('\n')[0]
        
        return (new_logic if new_logic else agent_to_update.logic, log_prob_sum)