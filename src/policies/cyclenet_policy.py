"""
CycleNet Policy Integration for Stable-Baselines3

This module provides seamless integration between CycleNet architecture and Stable-Baselines3
RL framework. It implements custom policy classes and feature extractors that
enable the use of CycleNet's cycle-aware neural networks in RL algorithms like PPO.

The module handles both sequential and flat observation spaces and automatically extracts
cycle indices from observations when using CycleIndexWrapper.
"""

import csv
import gymnasium as gym
import matplotlib.pyplot as plt

from types import SimpleNamespace
from CycleNet.models.CycleNet import Model as CycleNetModel
from stable_baselines3.common.policies import ActorCriticPolicy, BaseFeaturesExtractor

from src.utils.paths import DataPaths
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

def save_cycle_data_plot(cycle_data, output_dir):
    """
    Save the learned cycle pattern from CycleNet as visualizations and CSV data.
    
    This function creates comprehensive visualizations of the cycle patterns learned by CycleNet
    during training. It generates both SVG plots and CSV files for further analysis.
    
    Args:
        cycle_data (torch.Tensor): The cycle data tensor from CycleNet with shape [cycle_len, enc_in]
                                 containing the learned cycle patterns for each input feature
        output_dir (str): Directory path where to save SVG plot and CSV data files
    """

    paths = DataPaths(output_dir)
    cycle_np = cycle_data.detach().cpu().numpy()
    cycle_len, enc_in = cycle_np.shape

    # Create and save the plot
    plt.figure(figsize=(12, 6))
    for i in range(enc_in):
        plt.plot(range(cycle_len), cycle_np[:, i], label=f'Feature {i}')
    
    plt.xlim(0, cycle_len - 1)
    plt.title("Learned Cycle Pattern per Feature")
    plt.xlabel("Cycle Time Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save as SVG
    plt.savefig(paths.cycle_plot, format='svg', bbox_inches='tight')
    plt.close()

    # Save as CSV
    with open(paths.cycle_data, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([f"Feature {i}" for i in range(enc_in)])
        # Write data rows
        for row in cycle_np:
            writer.writerow(row)

    logger.training(f"Saved CycleNet cycle parameters to: {paths.cycle_plot} and {paths.cycle_data}.")

class CycleNetFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that integrates CycleNet with Stable-Baselines3.
    
    This class wraps the CycleNet model to function as a feature extractor in SB3 policies.
    It automatically handles different observation space formats and extracts cycle indices
    when using CycleIndexWrapper. The extractor supports both sequential and flat observations.
    
    Attributes:
        model (CycleNetModel): The underlying CycleNet model for feature extraction
        cycle_len (int): Length of the cycle pattern used by CycleNet
        is_sequential (bool): Whether observations are sequential or flat
    
    Args:
        observation_space (gym.Space): The observation space (should include cycle index)
        **kwargs: CycleNet model parameters including:
            - seq_len (int): Sequence length for temporal observations (default: 10)
            - enc_in (int): Number of input features (auto-detected from obs space)
            - cycle (int): Cycle length for periodic patterns (default: 24)
            - model_type (str): CycleNet variant to use (default: "mlp")
            - d_model (int): Model dimension (default: 512)
            - use_revin (bool): Whether to use RevIN normalization (default: False)
    """
    
    def __init__(self, observation_space, **kwargs):
        """
        Features extractor that wraps the CycleNet model to work with stable_baselines3.
        
        Args:
            observation_space: Observation space (should include cycle index from CycleIndexWrapper)
            **kwargs: CycleNet model parameters
        """
        # Get features from observation space
        if isinstance(observation_space, gym.spaces.Box):
            if len(observation_space.shape) == 1:
                # Flat observation: [obs_dim + 1] -> obs_dim
                channels = observation_space.shape[0] - 1
                self.is_sequential = False
            else:
                # Sequential observation: [seq_len, obs_dim + 1] -> obs_dim
                channels = observation_space.shape[-1] - 1
                self.is_sequential = True
        else:
            # Fallback for non-Box spaces
            channels = kwargs.get("enc_in", 17)
            self.is_sequential = False

        # Initialize the superclass with the correct features dimension
        super().__init__(observation_space, features_dim=channels)

        # Create CycleNet configuration with sensible defaults
        config = SimpleNamespace(
            seq_len=kwargs.get("seq_len", 10),
            pred_len=1,
            enc_in=channels,
            cycle=kwargs.get("cycle", 24),
            model_type=kwargs.get("model_type", "mlp"),
            d_model=kwargs.get("d_model", 512),
            use_revin=kwargs.get("use_revin", False)
        )
        
        self.model = CycleNetModel(config)
        self.cycle_len = config.cycle

    def forward(self, observations):
        """
        Extract features from observations using CycleNet.
        
        This method processes batched observations through the CycleNet model to extract
        meaningful features for RL. It automatically handles both sequential
        and flat observation formats and properly extracts cycle indices.
        
        Args:
            observations (torch.Tensor): Batch of observations with embedded cycle index
                - Flat case: [batch_size, obs_dim + 1] where last element is cycle index
                - Sequential case: [batch_size, seq_len, obs_dim + 1] where last feature 
                  of last timestep is cycle index
        
        Returns:
            torch.Tensor: Features extracted by CycleNet [batch_size, enc_in]
        """
        # Extract cycle index from the last element/timestep
        if self.is_sequential:
            # Sequential: cycle index is the last feature of the last timestep
            cycle_index = observations[:, -1, -1].long()
            # Remove cycle index from all timesteps
            actual_observations = observations[:, :, :-1]
        else:
            # Flat: cycle index is the last element
            cycle_index = observations[:, -1].long()
            # Remove cycle index and add sequence dimension for CycleNet
            actual_observations = observations[:, :-1].unsqueeze(1)  # [batch, features] -> [batch, 1, features]
        
        # Process through CycleNet
        output = self.model(actual_observations, cycle_index)  # Returns [batch_size, pred_len=1, enc_in]
        
        # Remove pred_len dimension for RL policy
        return output.squeeze(1)  # [batch_size, enc_in]

def create_cyclenet_policy(base_policy_class):
    """
    Factory function to create CycleNet policies for different algorithms.
    
    Args:
        base_policy_class: The base policy class (ActorCriticPolicy)
        
    Returns:
        A CycleNet policy class that inherits from the specified base class
    """
    class CycleNetPolicy(base_policy_class):
        """
        CycleNet-based policy that leverages cycle-aware feature extraction.
        
        This policy class integrates CycleNet with Stable-Baselines3 algorithms,
        enabling the use of cycle-aware neural networks in RL.
        It automatically extracts CycleNet-specific parameters and configures
        the feature extractor appropriately.
        """
        
        def __init__(self, *args, **kwargs):
            # Extract CycleNet-specific arguments
            cycle_net_kwargs = {
                "seq_len": kwargs.pop("seq_len", 10),
                "pred_len": 1,
                "enc_in": kwargs.pop("enc_in", 105),
                "cycle": kwargs.pop("cycle", 24),
                "model_type": kwargs.pop("model_type", "mlp"),
                "d_model": kwargs.pop("d_model", 512),
                "use_revin": kwargs.pop("use_revin", False),
            }

            # Pass the custom feature extractor to the base class
            super().__init__(
                *args,
                **kwargs,
                features_extractor_class=CycleNetFeaturesExtractor,
                features_extractor_kwargs=cycle_net_kwargs,
            )
    
    return CycleNetPolicy

# Create specific policy class
CycleNetPolicy = create_cyclenet_policy(ActorCriticPolicy)