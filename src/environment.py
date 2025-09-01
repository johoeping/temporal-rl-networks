"""
Environment Creation and Configuration Module

This module provides utilities for creating and configuring Gymnasium environments
with various wrappers including noise injection, action rescaling, and validation.
"""

import gymnasium as gym

from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.env_checker import check_env

from src.wrappers.noise_wrapper import NoiseWrapper
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

def create_environment(env_name="Ant-v5", render_mode=None, seed=None, 
                      noise_std=0.00, noise_type="gaussian"):
    """
    Create and validate a Gymnasium environment with optional noise injection.
    
    Args:
        env_name: Name of the Gymnasium environment
        render_mode: Rendering mode for the environment  
        seed: Random seed for reproducibility
        noise_std: Standard deviation of noise to add to observations (0 = no noise)
        noise_type: Type of noise ("gaussian" or "uniform")
        
    Returns:
        gym.Env: Configured and validated environment
    """
    def make_env():
        # Create base environment with optional rendering
        env = gym.make(env_name, render_mode=render_mode)
        env = RescaleAction(env, min_action=-1, max_action=1)
        
        # Add noise wrapper if noise is specified
        if noise_std > 0:
            logger.info(f"Adding noise wrapper with std {noise_std} and type {noise_type}.")
            env = NoiseWrapper(env, noise_std=noise_std, noise_type=noise_type, seed=seed)
        
        # Set seeds for reproducibility  
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        return env
    
    env = make_env()
    check_env(env, warn=True)
    return env