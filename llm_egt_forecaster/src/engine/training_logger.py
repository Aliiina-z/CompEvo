# File: src/engine/training_logger.py
"""
Real-time training metrics logger.
Records agent logics and training metrics to a single JSONL file during training.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


class TrainingLogger:
    """
    Real-time logger for training metrics and agent logics.
    Writes to a single JSONL file immediately (not buffered) to ensure real-time updates.
    """
    
    def __init__(self, log_dir: str = "training_logs"):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create single log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_log_file = self.log_dir / f"training_metrics_{timestamp}.jsonl"  # JSON Lines format
    
    def log_initial_logics(self, agents: List[Any], epoch_num: int = 0):
        """
        Log initial agent logics at the start of training.
        
        Args:
            agents: List of agent objects with .id and .logic attributes
            epoch_num: Epoch number (0-based)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        entry = {
            "timestamp": timestamp,
            "epoch": epoch_num + 1,
            "batch_idx": -1,  # Use -1 to indicate initial state
            "type": "initial_logics",
            "agent_logics": {f"agent_{agent.id}": agent.logic for agent in agents},
            "agent_fitness": {f"agent_{agent.id}": agent.fitness for agent in agents}
        }
        
        with open(self.metrics_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()  # Ensure immediate write
            os.fsync(f.fileno())  # Force write to disk
    
    def log_batch_metrics(self, 
                         batch_idx: int,
                         epoch_num: int,
                         agents: List[Any],
                         metrics: Dict[str, float],
                         new_logics: List[str] = None):
        """
        Log metrics and agent logics for a specific batch.
        
        Args:
            batch_idx: Batch index
            epoch_num: Epoch number (0-based)
            agents: List of agent objects
            metrics: Dictionary containing:
                - loss: Total loss
                - mse: Mean Squared Error
                - rmse: Root Mean Squared Error
                - mae: Mean Absolute Error
                - mape: Mean Absolute Percentage Error
                - l_pg: Policy Gradient Loss
                - l_diversity: Diversity Loss
                - l_pruning: Pruning Loss
            new_logics: Optional list of new logics (if different from agent.logic)
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Collect agent logics (use new_logics if provided, otherwise use agent.logic)
        agent_logics_dict = {}
        for i, agent in enumerate(agents):
            current_logic = new_logics[i] if new_logics and i < len(new_logics) else agent.logic
            agent_logics_dict[f"agent_{agent.id}"] = current_logic
        
        # Create single entry with both metrics and logics
        entry = {
            "timestamp": timestamp,
            "epoch": epoch_num + 1,
            "batch_idx": batch_idx,
            "type": "batch_metrics",
            "metrics": {
                "loss": metrics.get("loss", 0.0),
                "mse": metrics.get("mse", 0.0),
                "rmse": metrics.get("rmse", 0.0),
                "mae": metrics.get("mae", 0.0),
                "mape": metrics.get("mape", 0.0),
                "l_pg": metrics.get("l_pg", 0.0),
                "l_diversity": metrics.get("l_diversity", 0.0),
                "l_pruning": metrics.get("l_pruning", 0.0)
            },
            "agent_logics": agent_logics_dict,
            "agent_fitness": {f"agent_{agent.id}": agent.fitness for agent in agents}
        }
        
        with open(self.metrics_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()  # Ensure immediate write
            os.fsync(f.fileno())  # Force write to disk
    
    def get_log_file_paths(self):
        """Return paths to log files."""
        return {
            "metrics_log": str(self.metrics_log_file)
        }

