"""
Noise Wrapper for Robustness Testing

This wrapper adds configurable noise to environment observations to test
the robustness of trained policies against observation uncertainties.
"""

import numpy as np
import gymnasium as gym


class NoiseWrapper(gym.ObservationWrapper):
    """
    Add noise to observations for robustness testing and evaluation.
    
    Args:
        env: The environment to wrap
        noise_std: Standard deviation of the noise (0.0 = no noise)  
        noise_type: Type of noise distribution ("gaussian" or "uniform")
        seed: Random seed for reproducible noise generation
    """
    
    def __init__(self, env, noise_std=0.0, noise_type="gaussian", seed=1):
        super().__init__(env)
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.np_random = np.random.RandomState(seed)

    def observation(self, obs):
        """
        Add noise to the observation based on the configured noise type and magnitude.
        
        Args:
            obs: Original observation from the environment
            
        Returns:
            np.ndarray: Noisy observation with the same shape and dtype as input
        """
        if self.noise_std <= 0:
            return obs
        
        if self.noise_type == "gaussian":
            noise = self.np_random.normal(0, self.noise_std, obs.shape)
        elif self.noise_type == "uniform":
            noise = self.np_random.uniform(-self.noise_std, self.noise_std, obs.shape)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}. Use 'gaussian' or 'uniform'.")
        
        return obs + noise.astype(obs.dtype)