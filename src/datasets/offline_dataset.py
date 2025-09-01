"""
Offline Dataset Management for RL

This module provides comprehensive dataset management for offline RL training,
including episode-based storage, data validation, serialization, and statistics
computation for behavioral cloning and analysis.
"""

import os
import pickle
import numpy as np
import gymnasium as gym

from typing import Dict, List, Optional, Any

from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

class OfflineDataset:
    """
    Manages offline datasets for RL with episode-based organization.
    
    This class provides a complete solution for storing, managing, and accessing
    offline RL datasets. It organizes data by episodes rather than individual
    transitions, enabling efficient sequence-based training and analysis while
    maintaining data integrity through validation and automatic cleanup.
    
    Attributes:
        observation_space: Gymnasium observation space specification
        action_space: Gymnasium action space specification
        max_episodes: Maximum number of episodes to store
        episodes: List of episode dictionaries containing transition data
    """
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, 
                 max_episodes: int = 10000):
        """
        Initialize empty offline dataset with space specifications.
        
        Args:
            observation_space: Gymnasium observation space defining valid observations
            action_space: Gymnasium action space defining valid actions
            max_episodes: Maximum episodes to store (oldest removed when exceeded)
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.max_episodes = max_episodes
        self.episodes = []
        
    def __len__(self) -> int:
        """Return number of episodes in the dataset."""
        return len(self.episodes)
    
    def get_total_transitions(self) -> int:
        """
        Calculate total number of state-action transitions across all episodes.
        
        Returns:
            int: Total number of transitions in the dataset
        """
        return sum(len(episode['observations']) for episode in self.episodes)
        
    def add_episode(self, episode_data: Dict[str, List]):
        """
        Add a complete episode to the dataset with validation.
        
        This method validates episode data consistency, manages dataset capacity,
        and stores episodes with deep copies to prevent reference issues.
        
        Args:
            episode_data: Dictionary containing episode transitions with keys:
                - observations: List of environment observations
                - actions: List of agent actions
                - rewards: List of received rewards
                - next_observations: List of resulting observations
                - dones: List of episode termination flags
                - infos: List of environment info dictionaries (optional)
                
        Raises:
            ValueError: If episode data has inconsistent lengths between arrays
        """
        # Validate episode data structure and consistency
        required_keys = ['observations', 'actions', 'rewards', 'next_observations', 'dones']
        lengths = [len(episode_data[key]) for key in required_keys]
        
        if not all(length == lengths[0] for length in lengths):
            raise ValueError(f"Episode data arrays have mismatched lengths: {dict(zip(required_keys, lengths))}")

        # Manage dataset capacity by removing oldest episodes
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)
            
        # Store episode with deep copies to prevent reference issues
        episode = {
            'observations': [obs.copy() for obs in episode_data['observations']],
            'actions': [action.copy() for action in episode_data['actions']],
            'rewards': episode_data['rewards'].copy(),
            'next_observations': [next_obs.copy() for next_obs in episode_data['next_observations']],
            'dones': episode_data['dones'].copy(),
            'infos': episode_data.get('infos', [{}] * lengths[0]).copy(),
            'episode_length': lengths[0]
        }
        
        self.episodes.append(episode)
            
    def get_episode(self, episode_idx: int) -> Dict[str, List]:
        """
        Retrieve a specific episode by index.
        
        Args:
            episode_idx: Index of the episode to retrieve
            
        Returns:
            dict: Episode data dictionary
            
        Raises:
            IndexError: If episode index is out of range
        """
        if episode_idx >= len(self.episodes):
            raise IndexError(f"Episode index {episode_idx} out of range (dataset has {len(self.episodes)} episodes)")
        return self.episodes[episode_idx]
    
    def get_episodes_batch(self, start_idx: int = 0, end_idx: Optional[int] = None) -> List[Dict[str, List]]:
        """
        Retrieve a batch of episodes by index range.
        
        Args:
            start_idx: Starting episode index (inclusive)
            end_idx: Ending episode index (exclusive), None for end of dataset
            
        Returns:
            list: List of episode dictionaries
        """
        return self.episodes[start_idx:end_idx]
            
    def save(self, filepath: str):
        """
        Save complete dataset to disk using pickle serialization.
        
        This method creates the output directory if needed and saves all dataset
        components including episodes, space specifications, and metadata.
        
        Args:
            filepath: Path where to save the dataset file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'episodes': self.episodes,
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'max_episodes': self.max_episodes
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        logger.data(f"Dataset saved to {filepath}: {len(self.episodes)} episodes, "
                   f"{self.get_total_transitions()} transitions")
            
    @classmethod
    def load(cls, filepath: str) -> 'OfflineDataset':
        """
        Load dataset from disk file.
        
        Args:
            filepath: Path to the saved dataset file
            
        Returns:
            OfflineDataset: Loaded dataset instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        dataset = cls(data['observation_space'], data['action_space'], data['max_episodes'])
        dataset.episodes = data['episodes']
        
        logger.data(f"Dataset loaded from {filepath}: {len(dataset.episodes)} episodes, "
                   f"{dataset.get_total_transitions()} transitions")
        return dataset
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Compute comprehensive dataset statistics for analysis and reporting.
        
        Returns:
            dict: Statistics including episode counts, reward distributions,
                  episode length statistics, and other summary metrics
        """
        if not self.episodes:
            return {
                "num_episodes": 0, 
                "num_transitions": 0
            }
        
        # Collect all rewards and episode lengths for analysis
        all_rewards = []
        episode_lengths = []
        
        for episode in self.episodes:
            all_rewards.extend(episode['rewards'])
            episode_lengths.append(episode['episode_length'])
            
        return {
            # Basic counts
            "num_episodes": len(self.episodes),
            "num_transitions": len(all_rewards),
            
            # Reward statistics
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "min_reward": np.min(all_rewards),
            "max_reward": np.max(all_rewards),
            "total_reward": np.sum(all_rewards),
            
            # Episode length statistics
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths),
            "min_episode_length": np.min(episode_lengths),
            "max_episode_length": np.max(episode_lengths),
        }
