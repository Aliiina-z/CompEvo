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
        model_name = getattr(config, 'NEWS_COSINE_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_model = SentenceTransformer(model_name, device=config.DEVICE)
        self.embedding_model.eval()

    # (_calculate_prediction_loss, _calculate_diversity_loss, _calculate_pruning_loss remain the same)

    def _calculate_prediction_loss(self, framework_output, ground_truth):
        """
        Calculate prediction loss. Supports MSE, MAE, or MAPE based on config.
        
        Args:
            framework_output: Dict containing 'aggregated_prediction'
            ground_truth: Ground truth tensor, shape (batch_size, FUTURE_STEPS)
        
        Returns:
            Prediction loss (MSE, MAE, or MAPE)
        """
        aggregated_prediction = framework_output.get("aggregated_prediction")
        
        if aggregated_prediction is None:
            return torch.tensor(0.0, device=self.config.DEVICE, requires_grad=True)
        
        # Get loss type from config (default: 'mse')
        loss_type = getattr(self.config, 'PREDICTION_LOSS_TYPE', 'mse').lower()
        
        if loss_type == 'mae':
            # Mean Absolute Error - more stable, smaller values
            mae_loss = F.l1_loss(aggregated_prediction, ground_truth)
            return mae_loss
        elif loss_type == 'mape':
            # Mean Absolute Percentage Error - normalized, scale-invariant
            epsilon = 1e-8
            abs_error = torch.abs(aggregated_prediction - ground_truth)
            abs_true = torch.abs(ground_truth) + epsilon
            percentage_error = abs_error / abs_true
            mape_loss = percentage_error.mean()
            return mape_loss
        elif loss_type == 'rmse':
            # Root Mean Squared Error - same scale as MAE but still sensitive to outliers
            mse = F.mse_loss(aggregated_prediction, ground_truth)
            rmse_loss = torch.sqrt(mse)
            return rmse_loss
        elif loss_type == 'huber':
            # Huber loss - combines benefits of MSE and MAE
            delta = getattr(self.config, 'HUBER_DELTA', 1.0)
            huber_loss = F.smooth_l1_loss(aggregated_prediction, ground_truth, beta=delta)
            return huber_loss
        else:
            # Default: MSE (Mean Squared Error)
            mse_loss = F.mse_loss(aggregated_prediction, ground_truth)
            return mse_loss

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
    
    # --- NEW: KL Divergence Loss (prevents logic LoRA from drifting too far from base) ---
    def _calculate_kl_loss(self, action_log_probs, ref_log_probs):
        """
        Calculates KL divergence between logic LoRA distribution and reference (base model) distribution.
        
        KL(π_logic || π_ref) ≈ mean(log_prob_logic - log_prob_ref)
        
        Args:
            action_log_probs (torch.Tensor): Log probs from logic LoRA. Shape: (num_agents,).
            ref_log_probs (torch.Tensor): Log probs from reference/base model. Shape: (num_agents,).
        """
        if ref_log_probs is None:
            return torch.tensor(0.0, device=self.config.DEVICE)
        
        # Approximate KL divergence: E[log π_logic - log π_ref]
        kl_loss = (action_log_probs - ref_log_probs).mean()
        return kl_loss

    def forward(self, framework_output, ground_truth, agents, pg_components=None):
        # Calculate MSE loss for prediction
        l_predict = self._calculate_prediction_loss(framework_output, ground_truth)
        
        # --- L_PG is now part of the main loss calculation ---
        l_kl = torch.tensor(0.0, device=self.config.DEVICE)
        if pg_components:
            new_logics = pg_components["new_logics"]
            action_log_probs = pg_components["log_probs"]
            advantages = pg_components["advantages"]
            ref_log_probs = pg_components.get("ref_log_probs", None)

            l_pg = self._calculate_pg_loss(action_log_probs, advantages)
            l_diversity = self._calculate_diversity_loss(agents, new_logics)
            
            # KL divergence constraint to prevent logic LoRA from drifting
            l_kl = self._calculate_kl_loss(action_log_probs, ref_log_probs)
        else:
            # Fallback for cases where we don't do a PG step (e.g., validation)
            l_pg = torch.tensor(0.0, device=self.config.DEVICE)
            # Use current agent logics for diversity if no new ones are generated
            l_diversity = self._calculate_diversity_loss(agents, [a.logic for a in agents])

        l_pruning = self._calculate_pruning_loss(framework_output["agent_gates"])
        
        # Include KL loss in strategy loss with configurable weight
        lambda_kl = getattr(self.config, 'LAMBDA_KL', 0.1)
        l_strategy = (l_pg + 
                      self.config.LAMBDA_DIVERSITY * l_diversity +
                      self.config.LAMBDA_PRUNING * l_pruning +
                      lambda_kl * l_kl)
        
        # total_loss = l_predict + self.config.LAMBDA_STRATEGY * l_strategy
        total_loss = l_predict + 1e-2 * l_strategy

        return {
            "total_loss": total_loss,
            "l_predict": l_predict.item(),
            "l_strategy": l_strategy.item(),
            "l_pg": l_pg.item(),
            "l_kl": l_kl.item() if hasattr(l_kl, 'item') else l_kl,
            "l_diversity": l_diversity.item(),
            "l_pruning": l_pruning.item()
        }