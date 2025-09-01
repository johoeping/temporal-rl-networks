"""
Data Collection Module for Offline RL

This module provides functionality to collect episodes from RL environments
using trained models or random policies, creating datasets suitable for
offline training and analysis.
"""

import os

from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv

from src.datasets.offline_dataset import OfflineDataset
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

class DataCollector:
    """
    Collects episode data from environment interactions for offline RL training.
    
    This class enables data collection using either trained models or random policies,
    generating datasets that can be used for offline training, analysis, and evaluation.
    The collector handles model loading, episode generation, and data storage.
    
    Attributes:
        env: The Gymnasium environment to collect data from
        model: Loaded RL model for policy-based data collection (optional)
    """
    
    def __init__(self, env: GymEnv, model_path: Optional[str] = None):
        """
        Initialize data collector with environment and optional model.
        
        Args:
            env: Gymnasium environment for data collection
            model_path: Path to trained model file (None for random policy only)
        """
        self.env = env
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(model_path, device='cpu')
            logger.data(f"Loaded model from: {model_path}")
            
    def collect_episodes(self, num_episodes: int, use_random_policy: bool = False) -> OfflineDataset:
        """
        Collect multiple episodes using the specified policy type.
        
        This method generates a complete offline dataset by running multiple episodes
        with either a trained model or random actions. Progress is logged periodically
        to track collection status.
        
        Args:
            num_episodes: Number of episodes to collect
            use_random_policy: If True, use random actions; if False, use loaded model
            
        Returns:
            OfflineDataset: Dataset containing all collected episodes and transitions
            
        Raises:
            ValueError: If no model is loaded but use_random_policy is False
        """
        if not use_random_policy and self.model is None:
            raise ValueError("No model loaded - use random policy or provide valid model path")
            
        dataset = OfflineDataset(self.env.observation_space, self.env.action_space)
        policy_type = "random" if use_random_policy else "model"
        
        logger.data(f"Starting data collection with {policy_type} policy for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            episode_data = self._collect_single_episode(use_random_policy)
            dataset.add_episode(episode_data)
            
            # Log progress periodically
            if (episode + 1) % 10 == 0:
                logger.data(f"Collected {episode + 1}/{num_episodes} episodes using {policy_type} policy")
                
        logger.data(f"Data collection completed: {len(dataset)} episodes, {dataset.get_total_transitions()} transitions")
        return dataset
    
    def _collect_single_episode(self, use_random_policy: bool) -> dict:
        """
        Collect data from a single episode interaction.
        
        This method runs one complete episode, collecting all state transitions,
        actions, rewards, and metadata needed for offline training.
        
        Args:
            use_random_policy: Whether to use random actions or loaded model
            
        Returns:
            dict: Episode data containing observations, actions, rewards, etc.
        """
        obs, _ = self.env.reset()
        done = False
        
        # Initialize episode data storage
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': [],
            'infos': []
        }
        
        # Run episode until termination
        while not done:
            # Select action based on policy type
            if use_random_policy:
                action = self.env.action_space.sample()
            else:
                action, _ = self.model.predict(obs, deterministic=False)
            
            # Execute action in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store complete transition
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['next_observations'].append(next_obs)
            episode_data['dones'].append(done)
            episode_data['infos'].append(info)
            
            obs = next_obs
            
        return episode_data
