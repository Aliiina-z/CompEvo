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
    
    Now supports dual LoRA adapters:
    - "default": Prediction LoRA (trained via MSE loss)
    - "logic": Logic generation LoRA (trained via GRPO)
    """

    def __init__(self, agent_id: int, base_llm_model: PreTrainedModel, lora_config: LoraConfig, device: str):
        super().__init__()  # --- FIXED: Call super().__init__() ---

        self.id = agent_id
        self.device = device

        # --- Core Attributes (non-parameters) ---
        # self.logic = (
        #     f"Agent {self.id}: Analyze news related to weather conditions, economic activities, "
        #     "energy policies, infrastructure events, and social behaviors that may impact electricity load consumption."
        # )

        self.logic = (
            f"Agent {self.id}: Select news that impacts regional electricity load consumption. "
            "Focus on: Positive Factors Increasing Load Consumption:- Short-Term:  1. Economic Growth: Increases in commercial activity escalate electricity demand.  2. Technological Advancements: New power-consuming technologies boost demand.  3. Seasonal Factors: Extreme weather conditions necessitate more heating or cooling.  4. Social Events: Large gatherings can temporarily surge energy use.- Long-Term:  1. Population Growth: An increasing population escalates residential energy consumption.  2. Industrial Development: Expanding industries elevate energy requirements.  3. Urbanization: Growing urban areas increase electricity demand.  4. Energy Transition: The shift towards more electrically operated devices and vehicles heightens electricity use.Negative Factors Decreasing Load Consumption:- Short-Term:  1. Economic Downturns: Slumps in industrial activity diminish energy consumption.  2. Efficiency Improvements: Advances in energy-efficient technologies reduce electricity requirements.  3. Weather Patterns: Milder weather can decrease the need for heating and cooling.  4. Public Health Crises: Such events may curtail commercial and industrial energy consumption.- Long-Term:  1. Energy Efficiency: Improved technologies and building efficiencies gradually cut energy demand.  2. Demographic Changes: Trends like aging populations or reduced birth rates can lower demand.  3. Policy and Regulation: Government and international policies promote energy conservation.  4. Technological Innovations: Technological advancements lead to more efficient energy use.Other Influential Factors:- Political Stability: Influences the continuity and direction of energy policies and investments.- Global Market Dynamics: Impact local energy pricing and consumption habits.- Environmental Consciousness: Movements towards sustainability and renewable energy sources alter traditional consumption patterns.This refined logic aids stakeholders, including utility companies, policymakers, and investors, by providing a forward-looking view into the expected changes in energy consumption. It assists in making strategic decisions that align with projected shifts in the energy landscape."
        )
        
        self.fitness = 0.0
        self.last_reward = 0.0

        # --- Learnable Parameters ---
        # FIXED: Register 'gate' as a proper learnable parameter of the module.
        self.gate = nn.Parameter(torch.tensor(1.0, device=self.device))

        # FIXED: The PeftModel is an nn.Module itself, so assigning it as an attribute
        # automatically registers its parameters with this Agent module.
        # This is the "default" adapter for prediction tasks
        self.lora_model: PeftModel = get_peft_model(base_llm_model, lora_config)
        
        # --- Add Logic LoRA Adapter (for GRPO-trained logic generation) ---
        # Uses smaller rank/alpha for minimal initial impact on generation quality
        logic_lora_config = LoraConfig(
            r=base_config.LOGIC_LORA_RANK,
            lora_alpha=base_config.LOGIC_LORA_ALPHA,
            target_modules=base_config.LORA_TARGET_MODULES,
            lora_dropout=base_config.LOGIC_LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.lora_model.add_adapter("logic", logic_lora_config)
        # Ensure we start with the prediction adapter active
        self.lora_model.set_adapter("default")

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