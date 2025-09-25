# File: src/engine/trainer.py (Modified for Policy Gradient)

import torch
from torch.optim import AdamW
from tqdm import tqdm

from llm_egt_forecaster.configs import base_config
from llm_egt_forecaster.src.models.evolutionary_framework import EvolutionaryFramework
from llm_egt_forecaster.src.engine.loss import EvolutionaryLoss
from llm_egt_forecaster.src.models.logic_generator import LogicGenerator

class Trainer:
    # Simplified optimizer by leveraging nn.Module parameter collection
    def __init__(self, framework: EvolutionaryFramework, loss_fn: EvolutionaryLoss, 
                 logic_generator: LogicGenerator, dataloader, config):
        self.framework = framework
        self.loss_fn = loss_fn
        self.logic_generator = logic_generator
        self.dataloader = dataloader
        self.config = config

        # Major simplification: grab all parameters from the framework
        self.optimizer = AdamW(
            self.framework.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.agents = self.framework.agents
    
    # (_update_agents_after_batch is the same)
    def _update_agents_after_batch(self, rewards_tensor):
        avg_rewards = rewards_tensor.mean(dim=1).detach().cpu().numpy()
        for i, agent in enumerate(self.agents):
            agent.update_fitness(avg_rewards[i], self.config.FITNESS_EMA_BETA)

    def train_epoch(self, epoch_num):
        self.framework.train()
        total_loss = 0
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch_num+1}/{self.config.NUM_EPOCHS}")
        
        for batch in progress_bar:
            # --- START OF POLICY GRADIENT STEP ---
            
            # 1. Generate new candidate logics and their log probabilities for ALL agents
            new_logics = []
            log_probs = []
            
            original_logics = [agent.logic for agent in self.agents] # Store original logics
            
            for agent in self.agents:
                # The logic generator now needs to return log_prob as well
                new_logic, log_prob = self.logic_generator.generate(agent, self.agents, with_log_prob=True)
                new_logics.append(new_logic)
                log_probs.append(log_prob)
                agent.update_logic(new_logic) # Temporarily update agent logic for evaluation

            log_probs_tensor = torch.stack(log_probs)

            # 2. Evaluate the performance WITH THE NEW LOGICS
            # This is the "on-policy" evaluation step
            framework_output = self.framework(batch)
            rewards_new_logic = framework_output["agent_rewards"].mean(dim=1) # Avg reward over batch
            
            # 3. Calculate Advantage
            # Advantage = Reward(new_logic) - Baseline.
            # A simple baseline is the average reward of the group (GRPO).
            baseline = rewards_new_logic.mean()
            advantages = rewards_new_logic - baseline
            advantages = advantages.detach() # Treat advantages as constants in the loss

            # --- END OF POLICY GRADIENT STEP ---
            
            # 4. Calculate the total loss, now including L_PG
            ground_truth = batch["ground_truth"].to(self.config.DEVICE)
            pg_components = {
                "new_logics": new_logics,
                "log_probs": log_probs_tensor,
                "advantages": advantages
            }
            loss_dict = self.loss_fn(framework_output, ground_truth, self.agents, pg_components)
            loss = loss_dict["total_loss"]
            
            # 5. Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 6. Update agent states (fitness) using the new rewards
            self._update_agents_after_batch(framework_output["agent_rewards"])
            
            # 7. Permanently adopt the new logics (or you could have a rule for adoption)
            # For simplicity, we always adopt the generated logic.
            
            # Restore original logics for the next generation step to be clean
            # (or keep them, which means evolution is continuous)
            # For stability, it's often better to start fresh from the updated state.
            # We will keep the new logics, making the process continuous.
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'l_pg': f'{loss_dict["l_pg"]:.4f}',
                'adv_mean': f'{advantages.mean().item():.3f}'
            })
            
        avg_loss = total_loss / len(self.dataloader)
        print(f"\nEpoch {epoch_num+1} Summary: Average Loss = {avg_loss:.4f}")
        print("Final Logics for this epoch:")
        for agent in self.agents:
            print(f"- Agent {agent.id}: Fitness={agent.fitness:.4f}, Logic='{agent.logic}'")

    def train(self):
        # (The train loop is the same, but the _evolve_agent_logics call is removed)
        print("--- Starting Training with Policy Gradient ---")
        for epoch in range(self.config.NUM_EPOCHS):
            self.train_epoch(epoch)
        print("--- Training Finished ---")