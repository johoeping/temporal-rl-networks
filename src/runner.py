"""
Training Runner Module

This module provides the TrainingRunner class which orchestrates the complete
training pipeline including environment setup, model creation, wrapper application,
and training execution for both online and offline modes.
"""

import os

from stable_baselines3.common.vec_env import DummyVecEnv

from src.model import create_model
from src.utils.utils import set_seed
from src.environment import create_environment
from src.model import NetworkType, TrainingMode
from src.utils.config_loader import load_config
from src.training.train_online import train_online
from src.training.train_offline import train_offline
from src.datasets.offline_dataset import OfflineDataset
from src.wrappers.cycle_index_wrapper import CycleIndexWrapper
from src.wrappers.sequence_wrapper import ObservationSequenceWrapper
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

class TrainingRunner:
    """
    Orchestrates the complete training pipeline for RL models.
    
    This class handles environment setup, model creation, wrapper application,
    and training execution for various network architectures and training modes.
    It manages output directories and ensures proper cleanup and error handling.
    
    Attributes:
        base_output_dir (str): Base directory for saving training outputs
    """
    
    def __init__(self, base_output_dir: str):
        """
        Initialize the training runner with output directory.
        
        Args:
            base_output_dir: Base directory path for saving all training outputs
        """
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)

    def run_training(self, config_path_or_object):
        """
        Execute the complete training pipeline with the provided configuration.
        
        Args:
            config_path_or_object: Either a path to a YAML config file (str) 
                                 or a Config object instance
                                 
        Raises:
            ValueError: If dataset path doesn't exist for offline training
        """
        logger.training(f"Starting training...")

        # Load configuration from file or use provided Config object
        if isinstance(config_path_or_object, str):
            config = load_config(config_path_or_object)
        else:
            config = config_path_or_object

        # Set random seed for reproducibility
        set_seed(config.seed)
        
        # Generate descriptive run name based on configuration
        network_params = ""
        if config.network_type == NetworkType.MLP:
            network_params = f"_arch{'-'.join(map(str, config.mlp_net_arch))}"
        elif config.network_type == NetworkType.LSTM:
            network_params = f"_arch{'-'.join(map(str, config.lstm_net_arch))}_hidden{config.lstm_hidden_size}"
        elif config.network_type == NetworkType.CYCLENET:
            network_params = f"_seq{config.cyclenet_seq_len}_cycle{config.cyclenet_cycle}_{config.cyclenet_model_type}_dmodel{config.cyclenet_d_model}_revin{config.cyclenet_use_revin}"

        run_name = f"run_{config.env_name.lower()}_{config.network_type.name.lower()}{network_params.lower()}_{config.seed}"
        output_dir = os.path.join(self.base_output_dir, run_name)
        
        # Create and configure the training environment
        env = create_environment(
            config.env_name, 
            render_mode="human" if config.render_env else None, 
            seed=config.seed,
            noise_std=config.noise_std,
            noise_type=config.noise_type
        )
        
        # Apply CycleNet-specific wrappers in the correct order
        if config.network_type == NetworkType.CYCLENET:
            # Step 1: Add cycle index wrapper (augments observations with cycle information)
            env = CycleIndexWrapper(env, cycle_length=getattr(config, 'cyclenet_cycle', 24))
            logger.training(f"Applied CycleIndexWrapper with cycle_length={getattr(config, 'cyclenet_cycle', 24)}.")
            
            # Step 2: Add sequence wrapper if sequence length > 1
            if config.cyclenet_seq_len > 1:
                env = ObservationSequenceWrapper(env, seq_len=config.cyclenet_seq_len)
                logger.training(f"Applied ObservationSequenceWrapper with seq_len={config.cyclenet_seq_len}.")

        # Wrap environment for stable-baselines3 compatibility
        env = DummyVecEnv([lambda: env])
        
        # Create and configure the RL model
        logger.training(f"Creating {config.network_type.name} model...")
        model = create_model(env=env, config=config)
        config.model_class = model.__class__.__name__

        # Execute training based on specified mode
        if config.training_mode == TrainingMode.OFFLINE:
            # Load and validate offline dataset
            dataset_path = config.offline_dataset_path
            if dataset_path and "{seed}" in dataset_path:
                dataset_path = dataset_path.format(seed=config.seed)
                logger.training(f"Using seed-specific dataset: {dataset_path}.")
            if not dataset_path or not os.path.exists(dataset_path):
                raise ValueError(f"Dataset path {dataset_path} does not exist for offline training")
                
            dataset = OfflineDataset.load(dataset_path)
            train_offline(
                model=model,
                dataset=dataset,
                config=config,
                output_dir=output_dir,
                seed=config.seed
            )
        else:
            # Execute online training
            train_online(
                env=env,
                model=model,
                config=config,
                output_dir=output_dir,
                seed=config.seed
            )

        logger.training(f"Training completed successfully!")