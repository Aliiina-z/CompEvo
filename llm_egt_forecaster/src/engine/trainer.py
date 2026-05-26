# File: src/engine/trainer.py (Modified for Policy Gradient)

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime
from typing import Optional
import types
import os
import time
import numpy as np

from llm_egt_forecaster.configs import base_config
from llm_egt_forecaster.src.models.evolutionary_framework import EvolutionaryFramework
from llm_egt_forecaster.src.engine.loss import EvolutionaryLoss
from llm_egt_forecaster.src.models.logic_generator import LogicGenerator
from llm_egt_forecaster.src.engine.training_logger import TrainingLogger

class Trainer:
    # Simplified optimizer by leveraging nn.Module parameter collection
    def __init__(self, framework: EvolutionaryFramework, loss_fn: EvolutionaryLoss, 
                 logic_generator: LogicGenerator, dataloader, val_dataloader, config,
                 checkpoint_dir: str = "checkpoints"):
        self.framework = framework
        self.loss_fn = loss_fn
        self.logic_generator = logic_generator
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # Run identifier for timestamped checkpoints, e.g. 20251217_153012
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.best_checkpoint_path: Optional[str] = None
        self.efficiency_metrics = {
            "train_epoch_times_h": [],
            "avg_train_time_h_per_epoch": None,
            "peak_memory_allocated_gb": None,
            "peak_memory_reserved_gb": None,
        }
        
        # Initialize real-time logger
        log_dir = os.path.join(checkpoint_dir, "training_logs")
        self.training_logger = TrainingLogger(log_dir=log_dir)

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
    
    def _denormalize(self, normalized_values, norm_mean, norm_std):
        """
        Denormalize values back to original scale.
        
        Args:
            normalized_values: Tensor of normalized values (batch_size, ...)
            norm_mean: Tensor of means (batch_size,)
            norm_std: Tensor of stds (batch_size,)
        
        Returns:
            Tensor of denormalized values in original scale
        """
        eps = 1e-8
        # Handle different tensor shapes by unsqueezing norm params
        if normalized_values.dim() > norm_mean.dim():
            # Add dimensions to match normalized_values shape
            for _ in range(normalized_values.dim() - norm_mean.dim()):
                norm_mean = norm_mean.unsqueeze(-1)
                norm_std = norm_std.unsqueeze(-1)
        return normalized_values * (norm_std + eps) + norm_mean

    
    def _compute_ref_log_prob(self, agent, generated_logic: str) -> torch.Tensor:
        """
        Compute reference log probability of generated logic using base model (disabled adapters).
        This serves as the reference distribution for KL divergence constraint.
        
        Args:
            agent: The agent whose base model will be used
            generated_logic: The generated logic string to compute log_prob for
            
        Returns:
            Scalar tensor containing the approximate log probability under base model
        """
        import torch.nn.functional as F
        
        tokenizer = self.logic_generator.tokenizer
        lora_model = agent.lora_model
        
        # Tokenize the generated logic
        inputs = tokenizer(generated_logic, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.config.DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Disable all adapters to use base model
            with lora_model.disable_adapter():
                outputs = lora_model(**inputs)
                logits = outputs.logits
                
                # Compute log probabilities for the input tokens (teacher forcing)
                # Shift logits and labels for next token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = inputs["input_ids"][:, 1:].contiguous()
                
                # Compute log probabilities
                log_probs = F.log_softmax(shift_logits, dim=-1)
                
                # Gather log probs of actual tokens
                token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                
                # Sum log probs (approximate sequence log probability)
                ref_log_prob = token_log_probs.sum()
        
        return ref_log_prob

    def validate(self):
        """Validate the model on the validation set."""
        self.framework.eval()
        total_val_loss = 0
        # Normalized metrics (for model training)
        total_mse_norm = 0
        total_mae_norm = 0
        # Denormalized metrics (for human understanding of actual prediction accuracy)
        total_mse_denorm = 0
        total_mae_denorm = 0
        total_mape_denorm = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                framework_output = self.framework(batch)
                ground_truth = batch["ground_truth"].to(self.config.DEVICE)
                
                # Get normalization parameters for denormalization
                norm_mean = batch["norm_mean"].to(self.config.DEVICE)
                norm_std = batch["norm_std"].to(self.config.DEVICE)
                
                # Calculate validation loss (prediction only, no PG components)
                # Loss is calculated in NORMALIZED space for proper gradient computation
                loss_dict = self.loss_fn(framework_output, ground_truth, self.agents, pg_components=None)
                total_val_loss += loss_dict["total_loss"].item()
                
                # Calculate metrics in NORMALIZED space
                predictions = framework_output["aggregated_prediction"]
                mse_norm = torch.nn.functional.mse_loss(predictions, ground_truth)
                mae_norm = torch.nn.functional.l1_loss(predictions, ground_truth)
                total_mse_norm += mse_norm.item()
                total_mae_norm += mae_norm.item()
                
                # Denormalize for reporting real-world metrics
                predictions_denorm = self._denormalize(predictions, norm_mean, norm_std)
                ground_truth_denorm = self._denormalize(ground_truth, norm_mean, norm_std)
                
                # Calculate metrics in ORIGINAL (denormalized) space
                mse_denorm = torch.nn.functional.mse_loss(predictions_denorm, ground_truth_denorm)
                mae_denorm = torch.nn.functional.l1_loss(predictions_denorm, ground_truth_denorm)
                
                # Calculate MAPE in original space (more meaningful)
                epsilon = 1e-8
                abs_error = torch.abs(predictions_denorm - ground_truth_denorm)
                abs_true = torch.abs(ground_truth_denorm) + epsilon
                mape_denorm = ((abs_error / abs_true) * 100).mean()
                
                total_mse_denorm += mse_denorm.item()
                total_mae_denorm += mae_denorm.item()
                total_mape_denorm += mape_denorm.item()
                num_batches += 1
        
        # Normalized metrics (for training)
        avg_val_loss = total_val_loss / num_batches
        avg_mse_norm = total_mse_norm / num_batches
        avg_mae_norm = total_mae_norm / num_batches
        avg_rmse_norm = avg_mse_norm ** 0.5
        
        # Denormalized metrics (for reporting)
        avg_mse_denorm = total_mse_denorm / num_batches
        avg_mae_denorm = total_mae_denorm / num_batches
        avg_rmse_denorm = avg_mse_denorm ** 0.5
        avg_mape_denorm = total_mape_denorm / num_batches
        
        print(f"\n--- Validation Results ---")
        print(f"Val Loss (normalized): {avg_val_loss:.4f}")
        print(f"\n[Normalized Space Metrics - for model training]")
        print(f"  MSE: {avg_mse_norm:.4f}, RMSE: {avg_rmse_norm:.4f}, MAE: {avg_mae_norm:.4f}")
        print(f"\n[Original Scale Metrics - for human understanding]")
        print(f"  MSE: {avg_mse_denorm:.4f}")
        print(f"  RMSE: {avg_rmse_denorm:.4f}")
        print(f"  MAE: {avg_mae_denorm:.4f}")
        print(f"  MAPE: {avg_mape_denorm:.4f}%")
        
        return {
            "val_loss": avg_val_loss,
            # Normalized metrics (primary for training)
            "val_mse": avg_mse_norm,
            "val_rmse": avg_rmse_norm,
            "val_mae": avg_mae_norm,
            # Denormalized metrics (for reporting)
            "val_mse_original": avg_mse_denorm,
            "val_rmse_original": avg_rmse_denorm,
            "val_mae_original": avg_mae_denorm,
            "val_mape": avg_mape_denorm
        }
    
    def train_epoch(self, epoch_num, total_epochs=None):
        self.framework.train()
        total_loss = 0
        # Log initial logics at the start of first epoch
        if epoch_num == 0:
            self.training_logger.log_initial_logics(self.agents, epoch_num)
        
        display_total_epochs = total_epochs if total_epochs is not None else self.config.NUM_EPOCHS
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch_num+1}/{display_total_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # --- START OF POLICY GRADIENT STEP ---
                
                # 1. Generate new candidate logics and their log probabilities for ALL agents
                new_logics = []
                log_probs = []
                ref_log_probs = []  # Reference log_probs from base model for KL constraint
                
                original_logics = [agent.logic for agent in self.agents] # Store original logics
                
                # Clear cache before generating logics to free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                for i, agent in enumerate(self.agents):
                    try:
                        # The logic generator now needs to return log_prob as well
                        new_logic, log_prob = self.logic_generator.generate(agent, self.agents, with_log_prob=True)
                        new_logics.append(new_logic)
                        log_probs.append(log_prob)
                        
                        # Compute reference log_prob using base model (disabled adapter) for KL constraint
                        # This is approximated by using the generated tokens with disabled LoRA
                        ref_log_prob = self._compute_ref_log_prob(agent, new_logic)
                        ref_log_probs.append(ref_log_prob)
                        
                        agent.update_logic(new_logic) # Temporarily update agent logic for evaluation
                        
                        # Clear cache periodically to prevent memory buildup
                        if torch.cuda.is_available() and (i + 1) % 3 == 0:
                            torch.cuda.empty_cache()
                    except Exception as e:
                        import traceback
                        print(f"\n⚠️ Error generating logic for agent {i} in batch {batch_idx}: {type(e).__name__}: {e}")
                        print(f"Traceback: {traceback.format_exc()}")
                        # Use previous logic as fallback
                        new_logics.append(agent.logic)
                        log_probs.append(torch.tensor(0.0, device=self.config.DEVICE))
                        ref_log_probs.append(torch.tensor(0.0, device=self.config.DEVICE))

                log_probs_tensor = torch.stack(log_probs)
                ref_log_probs_tensor = torch.stack(ref_log_probs)

                # 2. Evaluate the performance WITH THE NEW LOGICS
                # This is the "on-policy" evaluation step
                try:
                    framework_output = self.framework(batch)
                    rewards_new_logic = framework_output["agent_rewards"].mean(dim=1) # Avg reward over batch
                    
                    # Store framework output for logging (every 10 batches)
                    if batch_idx % 10 == 0:
                        self._log_batch_info(
                            batch_idx=batch_idx,
                            epoch_num=epoch_num,
                            framework_output=framework_output,
                            batch=batch,
                            new_logics=new_logics
                        )
                except Exception as e:
                    import traceback
                    print(f"\n⚠️ Error in framework forward pass for batch {batch_idx}: {type(e).__name__}: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    # Skip this batch
                    progress_bar.set_postfix({
                        'loss': 'SKIPPED',
                        'l_pg': 'N/A',
                        'adv_mean': 'N/A',
                        'error': 'framework_error'
                    })
                    continue
                
                # 3. Calculate Advantage
                # Advantage = Reward(new_logic) - Baseline.
                # A simple baseline is the average reward of the group (GRPO).
                baseline = rewards_new_logic.mean()
                advantages = rewards_new_logic - baseline
                advantages = advantages.detach() # Treat advantages as constants in the loss
                
                # Normalize advantages by standard deviation to stabilize l_pg
                # This prevents l_pg from having extremely large values (e.g., millions)
                advantages_std = advantages.std()
                if advantages_std > 1e-8:  # Avoid division by zero
                    advantages = advantages / (advantages_std + 1e-8)
                # After normalization, advantages will have mean≈0, std≈1
                # This makes log_prob × advantages have a more stable scale

                # --- END OF POLICY GRADIENT STEP ---
                
                # 4. Calculate the total loss, now including L_PG and L_KL
                ground_truth = batch["ground_truth"].to(self.config.DEVICE)
                pg_components = {
                    "new_logics": new_logics,
                    "log_probs": log_probs_tensor,
                    "ref_log_probs": ref_log_probs_tensor,  # For KL divergence constraint
                    "advantages": advantages
                }
                loss_dict = self.loss_fn(framework_output, ground_truth, self.agents, pg_components)
                loss = loss_dict["total_loss"]
                
                # 5. Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                max_grad_norm = getattr(self.config, 'MAX_GRAD_NORM', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.framework.parameters(), max_grad_norm)
                
                self.optimizer.step()
                
                # 6. Update agent states (fitness) using the new rewards
                self._update_agents_after_batch(framework_output["agent_rewards"])
                
                # 7. Permanently adopt the new logics (or you could have a rule for adoption)
                # For simplicity, we always adopt the generated logic.
                
                # Restore original logics for the next generation step to be clean
                # (or keep them, which means evolution is continuous)
                # For stability, it's often better to start fresh from the updated state.
                # We will keep the new logics, making the process continuous.
                
                # Calculate metrics for this batch
                aggregated_prediction = framework_output.get("aggregated_prediction")
                if aggregated_prediction is not None:
                    mse = F.mse_loss(aggregated_prediction, ground_truth).item()
                    rmse = np.sqrt(mse)
                    mae = F.l1_loss(aggregated_prediction, ground_truth).item()
                    
                    # Calculate MAPE
                    epsilon = 1e-8
                    abs_error = torch.abs(aggregated_prediction - ground_truth)
                    abs_true = torch.abs(ground_truth) + epsilon
                    mape = (abs_error / abs_true).mean().item() * 100.0
                else:
                    mse = rmse = mae = mape = 0.0
                
                # Real-time logging every 10 batches
                if batch_idx % 10 == 0:
                    metrics_dict = {
                        "loss": loss.item(),
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "mape": mape,
                        "l_pg": loss_dict.get("l_pg", 0.0),
                        "l_diversity": loss_dict.get("l_diversity", 0.0),
                        "l_pruning": loss_dict.get("l_pruning", 0.0)
                    }
                    self.training_logger.log_batch_metrics(
                        batch_idx=batch_idx,
                        epoch_num=epoch_num,
                        agents=self.agents,
                        metrics=metrics_dict,
                        new_logics=new_logics
                    )
                
                total_loss += loss.item()
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'l_pred': f'{loss_dict.get("l_predict", 0.0):.4f}',
                    'l_pg': f'{loss_dict["l_pg"]:.4f}',
                    'l_div': f'{loss_dict.get("l_diversity", 0.0):.4f}',
                    'l_prune': f'{loss_dict.get("l_pruning", 0.0):.4f}',
                    'l_kl': f'{loss_dict.get("l_kl", 0.0):.4f}',
                    'adv_mean': f'{advantages.mean().item():.3f}'
                })
                
            except Exception as e:
                import traceback
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Print detailed error information
                print(f"\n{'='*80}")
                print(f"❌ Critical error in batch {batch_idx}: {error_type}")
                print(f"Error message: {error_msg}")
                print(f"{'='*80}")
                print(f"Full traceback:")
                traceback.print_exc()
                print(f"{'='*80}\n")
                
                # For OSError, provide specific troubleshooting
                if isinstance(e, OSError):
                    print("⚠️ OSError detected. Possible causes:")
                    print("  1. Network/API error (OpenAI API timeout or connection failure)")
                    print("  2. Disk space issue (check available disk space)")
                    print("  3. File permission issue (check file permissions)")
                    print("  4. Memory issue (check system memory)")
                    print("  Training will continue with next batch...\n")
                
                progress_bar.set_postfix({
                    'loss': 'ERROR',
                    'l_pg': 'N/A',
                    'adv_mean': 'N/A',
                    'error': error_type
                })
                # Continue to next batch instead of crashing
                continue
            
        avg_loss = total_loss / len(self.dataloader)
        print(f"\nEpoch {epoch_num+1} Summary: Average Loss = {avg_loss:.4f}")
        print("Final Logics for this epoch:")
        for agent in self.agents:
            print(f"- Agent {agent.id}: Fitness={agent.fitness:.4f}, Logic='{agent.logic}'")
    
    def _log_batch_info(self, batch_idx, epoch_num, framework_output, batch, new_logics):
        """
        Log detailed information for each agent every 10 batches.
        Includes: logic, prediction accuracy, prompt, and response.
        Shows both normalized (for training) and original scale (for human understanding) values.
        """
        print("\n" + "=" * 80)
        print(f"📊 Batch {batch_idx} (Epoch {epoch_num+1}) - Agent Details:")
        print("=" * 80)
        
        ground_truth = batch["ground_truth"].to(self.config.DEVICE)  # Normalized
        agent_predictions = framework_output["agent_predictions"]  # Shape: (num_agents, batch_size, future_steps), Normalized
        agent_prompts = framework_output.get("agent_prompts", [])  # List of lists of prompts
        
        # Get denormalization parameters
        norm_mean = batch["norm_mean"].to(self.config.DEVICE)
        norm_std = batch["norm_std"].to(self.config.DEVICE)
        
        batch_size = agent_predictions.shape[1]
        
        for agent_idx, agent in enumerate(self.agents):
            print(f"\n🤖 Agent {agent.id}:")
            # Use new_logic for this batch if available, otherwise use current agent logic
            current_logic = new_logics[agent_idx] if agent_idx < len(new_logics) else agent.logic
            print(f"  Logic: {current_logic}")
            print(f"  Fitness: {agent.fitness:.4f}")
            
            # Calculate prediction accuracy metrics for this agent (in NORMALIZED space)
            agent_pred = agent_predictions[agent_idx]  # Shape: (batch_size, future_steps)
            mse_norm = torch.nn.functional.mse_loss(agent_pred, ground_truth, reduction='mean')
            rmse_norm = torch.sqrt(mse_norm)
            mae_norm = torch.nn.functional.l1_loss(agent_pred, ground_truth, reduction='mean')
            
            # Denormalize for original scale metrics
            agent_pred_denorm = self._denormalize(agent_pred, norm_mean, norm_std)
            ground_truth_denorm = self._denormalize(ground_truth, norm_mean, norm_std)
            
            mse_denorm = torch.nn.functional.mse_loss(agent_pred_denorm, ground_truth_denorm, reduction='mean')
            rmse_denorm = torch.sqrt(mse_denorm)
            mae_denorm = torch.nn.functional.l1_loss(agent_pred_denorm, ground_truth_denorm, reduction='mean')
            
            # Calculate MAPE in original space (more meaningful)
            epsilon = 1e-8
            abs_error = torch.abs(agent_pred_denorm - ground_truth_denorm)
            abs_true = torch.abs(ground_truth_denorm) + epsilon
            mape = (abs_error / abs_true).mean() * 100.0
            
            # Calculate reward based on configured loss type (using normalized space for consistency)
            loss_type = getattr(self.config, 'PREDICTION_LOSS_TYPE', 'mse').lower()
            if loss_type == 'mae':
                reward = -mae_norm.item()
            elif loss_type == 'mape':
                reward = -(mape.item() / 100.0)  # MAPE is in percentage, convert to ratio
            elif loss_type == 'rmse':
                reward = -rmse_norm.item()
            elif loss_type == 'huber':
                delta = getattr(self.config, 'HUBER_DELTA', 1.0)
                huber = torch.nn.functional.smooth_l1_loss(agent_pred, ground_truth, reduction='mean', beta=delta)
                reward = -huber.item()
            else:
                # Default: MSE
                reward = -mse_norm.item()
            
            print(f"  Prediction Accuracy [Normalized - for training]:")
            print(f"    MSE: {mse_norm.item():.4f}, RMSE: {rmse_norm.item():.4f}, MAE: {mae_norm.item():.4f}")
            print(f"  Prediction Accuracy [Original Scale - for understanding]:")
            print(f"    MSE: {mse_denorm.item():.4f}")
            print(f"    RMSE: {rmse_denorm.item():.4f}")
            print(f"    MAE: {mae_denorm.item():.4f}")
            print(f"    MAPE: {mape.item():.2f}%")
            print(f"    Reward: {reward:.4f}")
            
            # Print prompt and response for first sample in batch
            if agent_prompts and len(agent_prompts) > agent_idx:
                prompts = agent_prompts[agent_idx]
                if prompts and len(prompts) > 0:
                    print(f"\n  📝 Prompt (Sample 0):")
                    prompt_text = prompts[0]
                    # Truncate if too long
                    if len(prompt_text) > 500:
                        print(f"    {prompt_text[:500]}...")
                    else:
                        print(f"    {prompt_text}")
                    
                    # Print response (prediction values from Linear head) - ORIGINAL SCALE
                    print(f"\n  📊 Response (Prediction - Sample 0, Original Scale):")
                    pred_values = agent_pred_denorm[0].detach().cpu().numpy()  # First sample, denormalized
                    # Format as comma-separated values
                    pred_str = ", ".join([f"{v:.1f}" for v in pred_values[:20]])  # Show first 20 values
                    if len(pred_values) > 20:
                        pred_str += f", ... (total {len(pred_values)} values)"
                    print(f"    {pred_str}")
                    
                    # Print ground truth for comparison - ORIGINAL SCALE
                    gt_values = ground_truth_denorm[0].detach().cpu().numpy()
                    gt_str = ", ".join([f"{v:.1f}" for v in gt_values[:20]])
                    if len(gt_values) > 20:
                        gt_str += f", ... (total {len(gt_values)} values)"
                    print(f"\n  ✅ Ground Truth (Sample 0, Original Scale):")
                    print(f"    {gt_str}")
        
        print("=" * 80 + "\n")

    def _serializable_config(self):
        """Return a JSON/pickle-friendly config snapshot (avoid saving module objects)."""
        cfg = self.config
        keys = [
            "MODE", "DEVICE",
            "BASE_LLM_MODEL", "TOKENIZER_PATH", "USE_QUANTIZATION", "MAX_LENGTH",
            "LORA_RANK", "LORA_ALPHA", "LORA_DROPOUT", "LORA_TARGET_MODULES",
            "BATCH_SIZE", "LEARNING_RATE", "NUM_EPOCHS", "NUM_AGENTS", "WEIGHT_DECAY",
            "TEMPERATURE", "FITNESS_EMA_BETA",
            "LAMBDA_STRATEGY", "LAMBDA_DIVERSITY", "LAMBDA_PRUNING",
            "TS_LENGTH", "FUTURE_STEPS",
            "NEWS_SELECTOR_METHOD",
            "NEWS_COSINE_MODEL", "NEWS_COSINE_THRESHOLD", "NEWS_COSINE_TOP_K",
            "MAX_NEWS_TOKENS", "PROMPT_TEMPLATE",
            "SEED",
        ]
        out = {}
        for k in keys:
            if hasattr(cfg, k):
                v = getattr(cfg, k)
                # Avoid saving module objects (e.g., accidentally capturing imported modules)
                if isinstance(v, types.ModuleType):
                    continue
                # Ensure basic python types for lists/tuples
                if isinstance(v, (list, tuple)):
                    v = list(v)
                out[k] = v
        return out

    def save_checkpoint(self, epoch, val_loss, filepath: Optional[str] = None, val_metrics: Optional[dict] = None):
        """Save model checkpoint."""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Timestamped checkpoint path for this run
        if filepath is None:
            filepath = os.path.join(self.checkpoint_dir, f"best_model_{self.run_id}.pt")
        latest_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        
        # Prefer val_loss inside val_metrics if provided
        if isinstance(val_metrics, dict) and "val_loss" in val_metrics:
            val_loss = float(val_metrics["val_loss"])

        # Print what we're about to save (so you can confirm before disk write)
        print("\n" + "=" * 60)
        print("Checkpoint snapshot (about to save)")
        print("=" * 60)
        print(f"run_id: {self.run_id}")
        print(f"epoch: {epoch} (1-based: {epoch + 1})")
        print(f"val_loss: {val_loss:.6f}")
        if isinstance(val_metrics, dict):
            print("val_metrics:")
            for k, v in val_metrics.items():
                try:
                    print(f"  - {k}: {float(v):.6f}")
                except Exception:
                    print(f"  - {k}: {v}")
        print("agents:")
        for agent in self.agents:
            try:
                fitness = float(agent.fitness)
            except Exception:
                fitness = agent.fitness
            print(f"  - Agent {agent.id}: Fitness={fitness}, Logic='{agent.logic}'")
        print("=" * 60 + "\n")

        def _to_py_scalar(x):
            # Convert numpy scalars / torch scalars to plain Python types for safer serialization
            try:
                # numpy / torch scalar
                if hasattr(x, "item") and callable(x.item):
                    return x.item()
            except Exception:
                pass
            return x

        # Normalize types to avoid numpy scalar objects inside checkpoint dict (PyTorch 2.6 weights_only)
        epoch_py = int(_to_py_scalar(epoch))
        val_loss_py = float(_to_py_scalar(val_loss))
        agent_fitness_py = [float(_to_py_scalar(a.fitness)) for a in self.agents]
        agent_logics_py = [str(a.logic) for a in self.agents]

        val_metrics_py = None
        if isinstance(val_metrics, dict):
            val_metrics_py = {}
            for k, v in val_metrics.items():
                vv = _to_py_scalar(v)
                # Keep floats for metric numbers when possible
                if isinstance(vv, (int, float)):
                    val_metrics_py[k] = float(vv)
                else:
                    val_metrics_py[k] = vv

        checkpoint = {
            'epoch': epoch_py,
            'framework_state_dict': self.framework.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss_py,
            'val_metrics': val_metrics_py,
            'config': self._serializable_config(),
            'run_id': str(self.run_id),  # Ensure it's a string
            'agent_logics': agent_logics_py,
            'agent_fitness': agent_fitness_py
        }
        
        # Some environments may still have non-picklable objects sneaking in.
        # If that happens, retry without the config snapshot rather than crashing training.
        try:
            # Try saving with default PyTorch serialization
            torch.save(checkpoint, filepath)
            print(f"💾 Checkpoint saved to {filepath}")
        except (TypeError, AttributeError, ModuleNotFoundError, ImportError) as e:
            print(f"\n⚠️  Checkpoint save failed: {type(e).__name__}: {e}")
            print("⚠️  This might be a PyTorch installation issue.")
            print("⚠️  Retrying save without `config` and `val_metrics` fields...")
            
            # Remove potentially problematic fields
            checkpoint_retry = {
                'epoch': epoch_py,
                'framework_state_dict': self.framework.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss_py,
                'run_id': str(self.run_id),
                'agent_logics': agent_logics_py,
                'agent_fitness': agent_fitness_py
            }
            
            try:
                torch.save(checkpoint_retry, filepath)
                print(f"💾 Checkpoint saved (minimal version) to {filepath}")
            except Exception as e2:
                print(f"\n❌ CRITICAL: Checkpoint save completely failed: {type(e2).__name__}: {e2}")
                print("❌ This indicates a serious PyTorch installation problem.")
                print("❌ Please check your PyTorch installation:")
                print("   1. Run: python -c 'import torch; print(torch.__version__)'")
                print("   2. Try: pip install --upgrade torch")
                print("   3. Or: conda install pytorch -c pytorch")
                print("\n⚠️  Training will continue, but checkpoints will not be saved.")
                print("⚠️  You may lose training progress if training is interrupted.")
                # Don't crash training, just skip checkpoint saving
                return
        
        # Only set best_checkpoint_path if save was successful
        # (check if file exists to confirm)
        if os.path.exists(filepath):
            self.best_checkpoint_path = filepath
            
            # Also update a stable "latest best" checkpoint path for convenience
            if filepath != latest_path:
                try:
                    # Use the same checkpoint dict that was successfully saved
                    if 'config' in checkpoint:
                        # If original save worked, use full checkpoint
                        torch.save(checkpoint, latest_path)
                    else:
                        # If we used minimal checkpoint, use that
                        torch.save(checkpoint_retry, latest_path)
                    print(f"💾 Latest best checkpoint updated at {latest_path}")
                except Exception as e:
                    print(f"⚠️  Failed to save latest checkpoint: {e}")
                    print("⚠️  Main checkpoint saved, but latest checkpoint update failed.")
    
    def load_checkpoint(self, filepath: Optional[str] = None):
        """Load model checkpoint."""
        import os
        if filepath is None:
            filepath = self.best_checkpoint_path or os.path.join(self.checkpoint_dir, "best_model.pt")

        checkpoint = torch.load(filepath, map_location=self.config.DEVICE)
        
        # Filter out quantization-related keys that may not match current model state
        framework_state = checkpoint['framework_state_dict']
        if isinstance(framework_state, dict):
            # Remove quantization state keys (bitsandbytes quantization parameters)
            filtered_state = {}
            for key, value in framework_state.items():
                # Skip quantization-related keys
                if any(quant_key in key for quant_key in [
                    '.absmax', '.quant_map', '.nested_absmax', '.nested_quant_map', 
                    '.quant_state', 'bitsandbytes'
                ]):
                    continue
                filtered_state[key] = value
            framework_state = filtered_state
        
        # Load with strict=False to handle any remaining mismatches
        self.framework.load_state_dict(framework_state, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore agent states
        for i, agent in enumerate(self.agents):
            agent.logic = checkpoint['agent_logics'][i]
            agent.fitness = checkpoint['agent_fitness'][i]

        self.best_checkpoint_path = filepath
        
        print(f"✅ Checkpoint loaded from {filepath}")
        print(f"   Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")
        
        return checkpoint['epoch'], checkpoint['val_loss']

    def train(self, start_epoch=0, num_epochs=None, initial_best_val_loss=None):
        print("--- Starting Training with Policy Gradient ---")
        
        # Print initial agent logics
        print("\n" + "=" * 80)
        print("📋 Initial Agent Logics:")
        print("=" * 80)
        for agent in self.agents:
            print(f"Agent {agent.id}:")
            print(f"  Logic: {agent.logic}")
            print(f"  Fitness: {agent.fitness:.4f}")
            print()
        print("=" * 80 + "\n")
        
        # Print log file paths
        log_paths = self.training_logger.get_log_file_paths()
        print(f"📝 Training logs will be saved to:")
        print(f"   Metrics & Logics log: {log_paths['metrics_log']}")
        print()
        
        epochs_to_run = self.config.NUM_EPOCHS if num_epochs is None else int(num_epochs)
        total_display_epochs = start_epoch + epochs_to_run
        best_val_loss = float('inf') if initial_best_val_loss is None else float(initial_best_val_loss)
        best_epoch = max(0, start_epoch - 1)
        
        if start_epoch > 0:
            print(
                f"Resuming training from epoch {start_epoch + 1}; "
                f"running {epochs_to_run} additional epoch(s)."
            )
            if initial_best_val_loss is not None:
                print(f"Loaded best validation loss baseline: {best_val_loss:.4f}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        for epoch in range(start_epoch, start_epoch + epochs_to_run):
            # Training
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_train_start = time.perf_counter()
            self.train_epoch(epoch, total_epochs=total_display_epochs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_train_time_h = (time.perf_counter() - epoch_train_start) / 3600.0
            self.efficiency_metrics["train_epoch_times_h"].append(epoch_train_time_h)
            print(f"[EFFICIENCY] Epoch {epoch + 1} train_time_h_per_epoch: {epoch_train_time_h:.6f}")
            
            # Validation
            val_metrics = self.validate()
            
            # Track and save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_epoch = epoch
                print(f"✓ New best validation loss: {best_val_loss:.4f}")
                
                # Save best model checkpoint
                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=best_val_loss,
                    filepath=None,
                    val_metrics=val_metrics
                )
            
            print("-" * 50)
        
        if self.efficiency_metrics["train_epoch_times_h"]:
            avg_train_time = sum(self.efficiency_metrics["train_epoch_times_h"]) / len(self.efficiency_metrics["train_epoch_times_h"])
            self.efficiency_metrics["avg_train_time_h_per_epoch"] = avg_train_time
            print(f"[EFFICIENCY] Avg train_time_h_per_epoch: {avg_train_time:.6f}")

        if torch.cuda.is_available():
            peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            self.efficiency_metrics["peak_memory_allocated_gb"] = peak_allocated_gb
            self.efficiency_metrics["peak_memory_reserved_gb"] = peak_reserved_gb
            print(f"[EFFICIENCY] Peak memory allocated_GB: {peak_allocated_gb:.4f}")
            print(f"[EFFICIENCY] Peak memory reserved_GB: {peak_reserved_gb:.4f}")

        print("--- Training Finished ---")
        print(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch + 1})")
        if self.best_checkpoint_path:
            print(f"Best model saved at: {self.best_checkpoint_path}")
        print(f"Latest best model also at: {self.checkpoint_dir}/best_model.pt")
    
