"""
Observation Sequence Wrapper for Sequential RL Models

This module provides a Gymnasium wrapper that maintains a sliding window
of recent observations, creating sequential input suitable for time-series
models like CycleNet and recurrent neural networks in RL.
"""

import numpy as np
import gymnasium as gym
from collections import deque


class ObservationSequenceWrapper(gym.ObservationWrapper):
    """
    Gymnasium wrapper that maintains a sequence of recent observations.
    
    This wrapper collects observations over multiple timesteps and provides
    them as a sequential input, enabling RL agents to leverage temporal
    patterns and dependencies. The wrapper maintains a sliding window of
    observations using a deque buffer.
    
    The observation space is transformed from (obs_dim,) to (seq_len, obs_dim),
    where seq_len is the length of the maintained sequence.
    
    Args:
        env: The Gymnasium environment to wrap
        seq_len: Number of observations to maintain in the sequence
    
    Attributes:
        seq_len (int): Length of the observation sequence
        enc_in (int): Dimension of individual observations
        buffer (deque): Circular buffer maintaining recent observations
        observation_space: Modified observation space for sequential data
    """
    def __init__(self, env, seq_len):
        """
        Initialize the ObservationSequenceWrapper.
        
        Sets up the observation buffer and modifies the observation space
        to accommodate sequential data.
        
        Args:
            env: The Gymnasium environment to wrap
            seq_len: Number of timesteps to include in observation sequence
        """
        super().__init__(env)
        self.seq_len = seq_len
        obs_shape = env.observation_space.shape
        self.enc_in = obs_shape[0]
        self.buffer = deque(maxlen=seq_len)

        self.observation_space = gym.spaces.Box(
            low=np.repeat(env.observation_space.low[None, :], seq_len, axis=0),
            high=np.repeat(env.observation_space.high[None, :], seq_len, axis=0),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        """
        Reset environment and initialize observation sequence.
        
        Clears the observation buffer and initializes it with zero-padded
        observations, placing the actual initial observation at the end.
        
        Args:
            **kwargs: Additional arguments passed to environment reset
            
        Returns:
            tuple: (sequential_observation, info) where sequential_observation
                  has shape (seq_len, obs_dim)
        """
        obs, info = self.env.reset(**kwargs)
        self.buffer.clear()
        for _ in range(self.seq_len):
            self.buffer.append(np.zeros(self.enc_in, dtype=obs.dtype))
        self.buffer[-1] = obs
        return self._get_observation(), info

    def observation(self, obs):
        """
        Add new observation to sequence and return current sequence.
        
        Args:
            obs: New observation from environment
            
        Returns:
            np.ndarray: Current observation sequence with shape (seq_len, obs_dim)
        """
        self.buffer.append(obs)
        return self._get_observation()

    def _get_observation(self):
        """
        Convert buffer contents to numpy array observation.
        
        Returns:
            np.ndarray: Sequential observation with shape (seq_len, obs_dim)
        """
        return np.array(self.buffer)