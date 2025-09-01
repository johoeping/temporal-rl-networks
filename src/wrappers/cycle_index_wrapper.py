"""
Cycle Index Wrapper for Enhanced RL Environment Observations

This module provides a Gymnasium wrapper that augments environment observations
with cycle index information, enabling CycleNet models to better capture
temporal patterns and periodicities in RL environments.

The wrapper automatically detects observation space structure and appropriately
adds cycle information while maintaining compatibility with various RL algorithms.
"""

import numpy as np
import gymnasium as gym


class CycleIndexWrapper(gym.ObservationWrapper):
    """
    Gymnasium wrapper that adds cycle index information to observations.
    
    This wrapper augments environment observations with a cycle index feature
    that increments with each environment step and wraps around based on the
    specified cycle length. This enables RL agents, particularly CycleNet
    models, to learn periodic patterns in the environment.
    
    The cycle index is added as an additional feature to the observation space:
    - For 1D observations: appends cycle index as last element
    - For sequential observations: appends cycle index to last dimension of all timesteps
    
    Args:
        env: The Gymnasium environment to wrap
        cycle_length: Length of the cycle period (default: 24, suitable for daily cycles)
    
    Attributes:
        cycle_length (int): The period length for cycle index wrapping
        step_counter (int): Current step counter for cycle index calculation
        observation_space: Modified observation space including cycle index
    """
    
    def __init__(self, env, cycle_length: int = 24):
        """
        Initialize the CycleIndexWrapper with enhanced observation space handling.
        
        Modifies the observation space to accommodate the additional cycle index
        feature and sets up internal state tracking.
        
        Args:
            env (gym.Env): The Gymnasium environment to wrap
            cycle_length (int): Period length for cycle index (number of steps before wrapping)
                              Must be positive integer, typically represents daily/weekly cycles
            
        Raises:
            ValueError: If environment has non-Box observation space
        """
        super().__init__(env)
        self.cycle_length = cycle_length
        self.step_counter = 0
        
        # Modify observation space to include cycle index
        if isinstance(env.observation_space, gym.spaces.Box):
            original_shape = env.observation_space.shape
            if len(original_shape) == 1:
                # Flat observation: [obs_dim] -> [obs_dim + 1]
                new_shape = (original_shape[0] + 1,)
                new_low = np.append(env.observation_space.low, 0)
                new_high = np.append(env.observation_space.high, cycle_length - 1)
            else:
                # Sequential observation: [seq_len, obs_dim] -> [seq_len, obs_dim + 1]
                new_shape = original_shape[:-1] + (original_shape[-1] + 1,)
                
                # Create new low/high arrays with correct shape
                new_low = np.full(new_shape, env.observation_space.low.min(), dtype=env.observation_space.dtype)
                new_high = np.full(new_shape, env.observation_space.high.max(), dtype=env.observation_space.dtype)
                
                # Set the cycle index bounds (last feature of each timestep)
                new_low[:, -1] = 0
                new_high[:, -1] = cycle_length - 1
            
            self.observation_space = gym.spaces.Box(
                low=new_low,
                high=new_high,
                shape=new_shape,
                dtype=env.observation_space.dtype
            )
        else:
            raise ValueError("CycleIndexWrapper only supports Box observation spaces")
    
    def observation(self, obs):
        """
        Augment observation with current cycle index.
        
        Adds the current cycle index (step_counter % cycle_length) to the
        observation. The implementation varies based on observation structure:
        - 1D observations: cycle index appended as final element
        - Sequential observations: cycle index added to last dimension of all timesteps
        
        Args:
            obs (np.ndarray): Original environment observation
            
        Returns:
            np.ndarray: Augmented observation including cycle index feature
        """
        cycle_index = self.step_counter % self.cycle_length
        
        if len(obs.shape) == 1:
            # Flat observation: append cycle index
            augmented_obs = np.append(obs, cycle_index)
        else:
            # Sequential observation: append cycle index to last dimension of all timesteps
            cycle_indices = np.full((obs.shape[0], 1), cycle_index, dtype=obs.dtype)
            augmented_obs = np.concatenate([obs, cycle_indices], axis=-1)
        
        return augmented_obs
    
    def step(self, action):
        """
        Execute environment step and update internal cycle counter.
        
        Args:
            action: Action to execute in the environment
            
        Returns:
            tuple: (augmented_observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_counter += 1
        return self.observation(obs), reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """
        Reset environment and initialize cycle counter to zero.
        
        Args:
            **kwargs: Additional arguments passed to environment reset
            
        Returns:
            tuple: (augmented_initial_observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        self.step_counter = 0
        return self.observation(obs), info
