# File: src/agent.py (Modified to be an nn.Module)

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import PreTrainedModel

# (Imports for __main__ test are the same)
from llm_egt_forecaster.configs import base_config
from transformers import AutoModelForCausalLM


class Agent(nn.Module):  # --- FIXED: Inherit from nn.Module ---
    """
    Represents a single agent as a proper PyTorch Module.
    This allows it to be seamlessly integrated into nn.ModuleList and ensures
    its parameters (gate and LoRA weights) are automatically collected by optimizers.
    """

    def __init__(self, agent_id: int, base_llm_model: PreTrainedModel, lora_config: LoraConfig, device: str):
        super().__init__()  # --- FIXED: Call super().__init__() ---

        self.id = agent_id
        self.device = device

        # --- Core Attributes (non-parameters) ---
        self.logic = f"Initial analysis logic for Agent {self.id}: Focus on general economic news."
        self.fitness = 0.0
        self.last_reward = 0.0

        # --- Learnable Parameters ---
        # FIXED: Register 'gate' as a proper learnable parameter of the module.
        self.gate = nn.Parameter(torch.tensor(1.0, device=self.device))

        # FIXED: The PeftModel is an nn.Module itself, so assigning it as an attribute
        # automatically registers its parameters with this Agent module.
        self.lora_model: PeftModel = get_peft_model(base_llm_model, lora_config)

        # For logging and debugging purposes
        self.selected_news_indices = []

    def update_fitness(self, reward: float, beta: float):
        """
        Update the agent's long-term fitness using Exponential Moving Average (EMA).
        """
        self.fitness = beta * self.fitness + (1 - beta) * reward
        self.last_reward = reward

    def update_logic(self, new_logic: str):
        """
        Update the agent's strategy logic.
        """
        self.logic = new_logic

    def forward(self, *args, **kwargs):
        """
        The forward pass of an agent is defined by its personalized LoRA model.
        This allows us to call the agent directly like a model.
        """
        return self.lora_model(*args, **kwargs)

    def __repr__(self):
        return (f"Agent(id={self.id}, "
                f"fitness={self.fitness:.4f}, "
                f"gate={self.gate.item():.4f}, "
                f"logic='{self.logic[:50]}...')")


if __name__ == '__main__':
    # --- The test block is updated to reflect the new structure ---
    from peft import prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    print("--- Testing Agent as an nn.Module ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # To test properly, let's simulate the quantized loading process
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_config.BASE_LLM_MODEL,
        quantization_config=quantization_config,
        device_map="auto"
    )
    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=base_config.LORA_RANK, lora_alpha=base_config.LORA_ALPHA,
        target_modules=base_config.LORA_TARGET_MODULES,
        lora_dropout=base_config.LORA_DROPOUT, bias="none", task_type="CAUSAL_LM"
    )

    agent0 = Agent(agent_id=0, base_llm_model=base_model, lora_config=lora_config, device=device)
    print("\nAgent Initial State:")
    print(agent0)

    # --- Verification of nn.Module properties ---
    print("\nVerifying learnable parameters of the Agent module:")
    # Now, we can simply call .parameters() on the agent instance!
    num_trainable_params = sum(p.numel() for p in agent0.parameters() if p.requires_grad)
    print(f"Agent module has {num_trainable_params:,} trainable parameters.")

    # Check if the gate is found
    found_gate = any(p is agent0.gate for p in agent0.parameters())
    print(f"Is agent.gate found in agent.parameters()? {found_gate}")
    assert found_gate, "Gate was not correctly registered as a parameter!"

    # Check LoRA model's own printout
    agent0.lora_model.print_trainable_parameters()

    print("\nAgent nn.Module test completed successfully!")