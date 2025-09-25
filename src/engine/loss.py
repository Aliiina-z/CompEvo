# File: src/engine/loss.py (Modified to include LPG)

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from llm_egt_forecaster.configs import base_config

class EvolutionaryLoss(torch.nn.Module):
    # (Docstring and __init__ remain the same)
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=config.DEVICE)
        self.embedding_model.eval()

    # (_calculate_prediction_loss, _calculate_diversity_loss, _calculate_pruning_loss remain the same)

    def _calculate_prediction_loss(self, aggregated_prediction, ground_truth):
        return F.mse_loss(aggregated_prediction, ground_truth)

    def _calculate_diversity_loss(self, agents, new_logics):
        # MODIFIED: Calculate diversity on the *newly generated* logics
        if len(agents) < 2:
            return torch.tensor(0.0, device=self.config.DEVICE)
        logic_strings = new_logics
        with torch.no_grad():
            embeddings = self.embedding_model.encode(logic_strings, convert_to_tensor=True)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        cosine_similarity_matrix = torch.matmul(embeddings, embeddings.T)
        num_agents = len(agents)
        diversity_loss = (torch.sum(cosine_similarity_matrix) - num_agents) / 2.0
        num_pairs = num_agents * (num_agents - 1) / 2.0
        return diversity_loss / num_pairs

    def _calculate_pruning_loss(self, agent_gates):
        return torch.mean(torch.abs(agent_gates))
    
    # --- NEW: Policy Gradient Loss (L_PG) using GRPO ---
    def _calculate_pg_loss(self, action_log_probs, advantages):
        """
        Calculates the Policy Gradient loss.
        'action' here is the generated logic.
        
        Args:
            action_log_probs (torch.Tensor): Log probabilities of the generated logics. Shape: (num_agents,).
            advantages (torch.Tensor): Advantage of the new logic over the old one. Shape: (num_agents,).
        """
        # We want to maximize the log_prob of actions with high advantages.
        # So we minimize the negative of it.
        policy_loss = - (action_log_probs * advantages).mean()
        return policy_loss

    def forward(self, framework_output, ground_truth, agents, pg_components=None):
        # (Docstring updated)
        l_predict = self._calculate_prediction_loss(
            framework_output["aggregated_prediction"],
            ground_truth
        )
        
        # --- L_PG is now part of the main loss calculation ---
        if pg_components:
            new_logics = pg_components["new_logics"]
            action_log_probs = pg_components["log_probs"]
            advantages = pg_components["advantages"]

            l_pg = self._calculate_pg_loss(action_log_probs, advantages)
            l_diversity = self._calculate_diversity_loss(agents, new_logics)
        else:
            # Fallback for cases where we don't do a PG step (e.g., validation)
            l_pg = torch.tensor(0.0, device=self.config.DEVICE)
            # Use current agent logics for diversity if no new ones are generated
            l_diversity = self._calculate_diversity_loss(agents, [a.logic for a in agents])

        l_pruning = self._calculate_pruning_loss(framework_output["agent_gates"])
        
        l_strategy = (l_pg + 
                      self.config.LAMBDA_DIVERSITY * l_diversity +
                      self.config.LAMBDA_PRUNING * l_pruning)
        
        total_loss = l_predict + self.config.LAMBDA_STRATEGY * l_strategy
        
        return {
            "total_loss": total_loss,
            "l_predict": l_predict.item(),
            "l_strategy": l_strategy.item(),
            "l_pg": l_pg.item(),
            "l_diversity": l_diversity.item(),
            "l_pruning": l_pruning.item()
        }