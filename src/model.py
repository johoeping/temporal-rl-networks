"""
Model Creation and Configuration Module

This module provides functionality for creating and configuring RL
models with different architectures (MLP, LSTM, CycleNet) using stable-baselines3.
Supports both online and offline training modes with configurable parameters.
"""

import torch

from enum import Enum
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from src.policies.cyclenet_policy import CycleNetPolicy
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

class NetworkType(Enum):
    """Enumeration of supported neural network architectures."""
    MLP = "mlp"
    LSTM = "lstm"
    CYCLENET = "cyclenet"


class TrainingMode(Enum):
    """Enumeration of supported training modes."""
    ONLINE = "online"
    OFFLINE = "offline"


def create_model(env, config):
    """
    Create a PPO model with the specified environment and configuration.
    
    This function creates and configures a RL model based on the specified
    network type and training parameters. Supports MLP, LSTM, and CycleNet
    architectures with customizable hyperparameters.

    Args:
        env: The Gymnasium environment to train on
        config: Configuration object containing all model parameters including:
            - network_type: Type of neural network architecture to use
            - device: Computing device ('cpu', 'cuda', or 'auto')
            - training_mode: Online or offline training mode
            - Various network-specific parameters
            
    Returns:
        stable_baselines3.PPO or sb3_contrib.RecurrentPPO: Configured RL model
        
    Raises:
        ValueError: If unknown network type or learning rate schedule is specified
    """
    # Auto-detect computing device if not explicitly specified
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configure model architecture and parameters based on network type
    if config.network_type == NetworkType.MLP:
        policy_type = "MlpPolicy"
        policy_kwargs = dict(net_arch=config.mlp_net_arch)
        model_class = PPO
        
    elif config.network_type == NetworkType.LSTM:
        policy_type = "MlpLstmPolicy"
        model_class = RecurrentPPO
        policy_kwargs = dict(
            net_arch=config.lstm_net_arch, 
            lstm_hidden_size=config.lstm_hidden_size
        )
        
    elif config.network_type == NetworkType.CYCLENET:
        policy_kwargs = {
            "seq_len": config.cyclenet_seq_len,
            "enc_in": env.observation_space.shape[0],
            "cycle": config.cyclenet_cycle,
            "model_type": config.cyclenet_model_type,
            "d_model": config.cyclenet_d_model,
            "use_revin": config.cyclenet_use_revin,
        }
        
        policy_type = CycleNetPolicy
        model_class = PPO
        
    else:
        raise ValueError(f"Unknown model type: {config.network_type}")

    # Create learning rate schedule function
    def linear_schedule(initial_value: float):
        """
        Linear learning rate schedule that decreases from initial_value to 0.
        
        Args:
            initial_value: Initial learning rate value
            
        Returns:
            function: Schedule function that takes progress_remaining and returns current LR
        """
        def func(progress_remaining: float) -> float:
            """
            Calculate current learning rate based on training progress.
            
            Args:
                progress_remaining: Training progress (1.0 at start, 0.0 at end)
                
            Returns:
                float: Current learning rate
            """
            return progress_remaining * initial_value
        return func

    # Configure training parameters based on mode
    if config.training_mode == TrainingMode.ONLINE:
        learning_rate = config.online_learning_rate
        lr_schedule = config.online_lr_schedule
        n_steps = config.online_n_steps
        batch_size = config.online_batch_size
        max_grad_norm = config.online_max_grad_norm
    else:  # Offline training
        learning_rate = config.offline_learning_rate
        lr_schedule = config.offline_lr_schedule
        n_steps = 2048  # Standard value for offline training
        batch_size = config.offline_batch_size
        max_grad_norm = config.offline_max_grad_norm
        
    # Apply learning rate schedule
    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)
    elif lr_schedule == "constant":
        # Keep learning_rate as is (constant value)
        pass
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}. Use 'linear' or 'constant'.")

    # Create and configure the PPO model
    model = model_class(
        policy_type,
        env,
        verbose=0,
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs
    )

    return model