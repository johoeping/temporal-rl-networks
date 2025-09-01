"""
Configuration Management Module

This module provides comprehensive configuration management for the CycleNet RL project,
including YAML configuration loading, runtime metadata tracking, progress monitoring,
and command-line argument integration with proper defaults handling.
"""

import yaml
import json

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, asdict

from src.utils.paths import DataPaths
from src.model import NetworkType, TrainingMode
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

@dataclass
class Config:
    """
    Comprehensive configuration class for CycleNet RL training pipeline.
    
    This class encapsulates all configuration parameters needed for training,
    including environment settings, model parameters, training hyperparameters,
    and runtime metadata. Supports both online and offline training modes
    with progress tracking and metadata management.
    
    Attributes:
        # Environment Configuration
        env_name: Name of the Gymnasium environment to use
        render_env: Whether to render the environment during training
        noise_std: Standard deviation of observation noise (0.0 = no noise)
        noise_type: Type of noise distribution ("gaussian" or "uniform")
        
        # Model Architecture
        network_type: Type of neural network (MLP, LSTM, or CYCLENET)
        device: Computing device ("cpu", "cuda", or "auto")
        
        # MLP-specific parameters
        mlp_net_arch: Network architecture for MLP policy
        
        # LSTM-specific parameters  
        lstm_net_arch: Network architecture for LSTM policy
        lstm_hidden_size: Hidden state size for LSTM
        
        # CycleNet-specific parameters
        cyclenet_seq_len: Sequence length for CycleNet input
        cyclenet_cycle: Cycle length for periodic patterns
        cyclenet_model_type: CycleNet model variant ("linear", "mlp", etc.)
        cyclenet_d_model: Model dimension for CycleNet
        cyclenet_use_revin: Whether to use RevIN normalization
        
        # Training Configuration
        training_mode: Training mode (online or offline)
        rolling_length: Window size for rolling statistics
        enable_live_plot: Whether to show live training plots
        
        # Online Training Parameters
        online_total_timesteps: Total timesteps for online training
        online_n_steps: Steps per policy update
        online_batch_size: Batch size for online training
        online_learning_rate: Learning rate for online training
        online_lr_schedule: LR schedule type ("constant" or "linear")
        online_max_grad_norm: Maximum gradient norm for clipping
        
        # Offline Training Parameters
        offline_dataset_path: Path to offline dataset file
        offline_epochs: Number of training epochs for offline mode
        offline_batch_size: Batch size for offline training
        offline_learning_rate: Learning rate for offline training
        offline_lr_schedule: LR schedule type for offline training
        offline_max_grad_norm: Maximum gradient norm for offline training
        
        # Dataset Collection Parameters
        dataset_collection_num_episodes: Episodes to collect for dataset
        dataset_collection_collection_type: Collection type ("model" or "random")
        dataset_collection_model_path: Path to model for dataset collection
        dataset_collection_output_path: Output path for collected dataset
        
        # Experiment Settings
        seed: Random seed for reproducibility
        
        # Runtime Tracking
        trained_timesteps: Current number of trained timesteps
        trained_epochs: Current number of trained epochs
        model_class: Class name of the RL model being used
        start_time: ISO timestamp when training started
        last_updated: ISO timestamp of last update
    """
    # Environment
    env_name: str
    render_env: bool
    
    # Noise settings
    noise_std: float
    noise_type: str
    
    # Model
    network_type: NetworkType
    device: str
    
    # MLP config
    mlp_net_arch: List[int]
    
    # LSTM config
    lstm_net_arch: List[int]
    lstm_hidden_size: int

    # CycleNet config
    cyclenet_seq_len: int
    cyclenet_cycle: int
    cyclenet_model_type: str
    cyclenet_d_model: int
    cyclenet_use_revin: bool
    
    # Common training settings
    rolling_length: int
    enable_live_plot: bool
    
    # Experiment
    seed: int

    # Training mode
    training_mode: TrainingMode

    # Online training settings
    online_total_timesteps: int
    online_n_steps: int
    online_batch_size: int
    online_learning_rate: float
    online_lr_schedule: str
    online_max_grad_norm: float
    
    # Offline training settings
    offline_dataset_path: str
    offline_epochs: int
    offline_batch_size: int
    offline_learning_rate: float
    offline_lr_schedule: str
    offline_max_grad_norm: float
    
    # Dataset collection settings
    dataset_collection_num_episodes: int
    dataset_collection_collection_type: str
    dataset_collection_model_path: str
    dataset_collection_output_path: str

    # Training progress
    trained_timesteps: int
    trained_epochs: int

    # Runtime info
    model_class: str

    # Optional fields
    start_time: Optional[str]
    last_updated: Optional[str]

    def __post_init__(self):
        """Initialize runtime metadata upon object creation."""
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()
        self.update_timestamp()
    
    def update_timestamp(self):
        """Update the last_updated timestamp to current time."""
        self.last_updated = datetime.now().isoformat()
    
    def update_progress(self, timesteps: int = None, epochs: int = None):
        """
        Update training progress and timestamp.
        
        Args:
            timesteps: New timestep count (for online training)
            epochs: New epoch count (for offline training)
        """
        if timesteps is not None:
            self.trained_timesteps = timesteps
        if epochs is not None:
            self.trained_epochs = epochs
        self.update_timestamp()
    
    def is_training_complete(self) -> bool:
        """
        Check if training has reached completion criteria.
        
        Returns:
            bool: True if training is complete based on mode-specific targets
        """
        if self.training_mode == TrainingMode.ONLINE:
            return self.trained_timesteps >= self.online_total_timesteps
        else:
            return self.trained_epochs >= self.offline_epochs
    
    def get_remaining_work(self) -> int:
        """
        Get amount of remaining training work.
        
        Returns:
            int: Remaining timesteps (online) or epochs (offline)
        """
        if self.training_mode == TrainingMode.ONLINE:
            return max(0, self.online_total_timesteps - self.trained_timesteps)
        else:
            return max(0, self.offline_epochs - self.trained_epochs)
    
    def to_dict(self):
        """
        Convert configuration to dictionary for JSON serialization.
        
        Returns:
            dict: Configuration as dictionary with enum values converted to strings
        """
        data = asdict(self)
        # Convert enums to string values for JSON compatibility
        data['network_type'] = self.network_type.value
        data['training_mode'] = self.training_mode.value
        return data
    
    def save_metadata(self, output_dir: str):
        """
        Save configuration metadata as JSON file.
        
        Args:
            output_dir: Directory where to save the metadata file
        """
        paths = DataPaths(output_dir)
        with open(paths.metadata, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


# Centralized default configuration values
# Used by both load_config and create_config_with_overrides functions
CONFIG_DEFAULTS = {
    'env_name': 'Walker2d-v5',
    'render_env': False,
    'noise_std': 0.0,
    'noise_type': 'gaussian',
    'network_type': 'cyclenet',
    'device': 'cpu',
    'mlp_net_arch': [16, 16],
    'lstm_net_arch': [16, 16],
    'lstm_hidden_size': 16,
    'cyclenet_seq_len': 16,
    'cyclenet_cycle': 16,
    'cyclenet_model_type': 'linear',
    'cyclenet_d_model': 16,
    'cyclenet_use_revin': False,
    'training_mode': 'online',
    'rolling_length': 35,
    'enable_live_plot': False,
    'online_total_timesteps': 1000000,
    'online_n_steps': 2048,
    'online_batch_size': 256,
    'online_learning_rate': 0.0003,
    'online_lr_schedule': 'constant',
    'online_max_grad_norm': 0.5,
    'offline_dataset_path': 'data/example_output/custom/run_walker2d-v5_{seed}/dataset.pkl',
    'offline_epochs': 100,
    'offline_batch_size': 256,
    'offline_learning_rate': 0.0001,
    'offline_lr_schedule': 'constant',
    'offline_max_grad_norm': 0.5,
    'dataset_collection_num_episodes': 100,
    'dataset_collection_collection_type': 'model',
    'dataset_collection_model_path': 'data/example_output/custom/run_walker2d-v5_{seed}/model.zip',
    'dataset_collection_output_path': 'data/example_output/custom/run_walker2d-v5_{seed}/dataset.pkl',
    'seed': 2025,
    'trained_timesteps': 0,
    'trained_epochs': 0,
    'model_class': 'PPO',
    'start_time': None,
    'last_updated': None
}


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file with validation and defaults.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config: Populated configuration object with all parameters
        
    Raises:
        ValueError: If learning rate schedule values are invalid
    """
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Validate learning rate schedule settings
    online_lr_schedule = config_data['training'].get('online', {}).get('lr_schedule') or CONFIG_DEFAULTS['online_lr_schedule']
    offline_lr_schedule = config_data['training'].get('offline', {}).get('lr_schedule') or CONFIG_DEFAULTS['offline_lr_schedule']
    
    if online_lr_schedule not in ['linear', 'constant']:
        raise ValueError(f"Invalid online lr_schedule '{online_lr_schedule}'. Must be 'linear' or 'constant'.")
    if offline_lr_schedule not in ['linear', 'constant']:
        raise ValueError(f"Invalid offline lr_schedule '{offline_lr_schedule}'. Must be 'linear' or 'constant'.")
    
    # Create Config object with comprehensive parameter mapping
    return Config(
        # Environment
        env_name=config_data['environment']['name'],
        render_env=config_data['environment']['render'],
        noise_std=config_data['environment'].get('noise_std') or CONFIG_DEFAULTS['noise_std'],
        noise_type=config_data['environment'].get('noise_type') or CONFIG_DEFAULTS['noise_type'],
        
        # Model
        network_type=NetworkType(config_data['model']['network_type']),
        device=config_data['model']['device'],
        
        # MLP
        mlp_net_arch=config_data['model']['mlp']['net_arch'],
        
        # LSTM
        lstm_net_arch=config_data['model']['lstm']['net_arch'],
        lstm_hidden_size=config_data['model']['lstm']['lstm_hidden_size'],
        
        # CycleNet
        cyclenet_seq_len=config_data['model']['cyclenet']['seq_len'],
        cyclenet_cycle=config_data['model']['cyclenet']['cycle'],
        cyclenet_model_type=config_data['model']['cyclenet']['model_type'],
        cyclenet_d_model=config_data['model']['cyclenet']['d_model'],
        cyclenet_use_revin=config_data['model']['cyclenet']['use_revin'],
        
        # Training mode
        training_mode=TrainingMode(config_data['training'].get('mode') or CONFIG_DEFAULTS['training_mode']),
        
        # Common training settings
        rolling_length=config_data['training'].get('rolling_length') or CONFIG_DEFAULTS['rolling_length'],
        enable_live_plot=config_data['training'].get('enable_live_plot') or CONFIG_DEFAULTS['enable_live_plot'],
        
        # Online training settings - apply defaults for missing values
        online_total_timesteps=config_data['training'].get('online', {}).get('total_timesteps') or CONFIG_DEFAULTS['online_total_timesteps'],
        online_n_steps=config_data['training'].get('online', {}).get('n_steps') or CONFIG_DEFAULTS['online_n_steps'],
        online_batch_size=config_data['training'].get('online', {}).get('batch_size') or CONFIG_DEFAULTS['online_batch_size'],
        online_learning_rate=config_data['training'].get('online', {}).get('learning_rate') or CONFIG_DEFAULTS['online_learning_rate'],
        online_lr_schedule=online_lr_schedule,
        online_max_grad_norm=config_data['training'].get('online', {}).get('max_grad_norm') or CONFIG_DEFAULTS['online_max_grad_norm'],
        
        # Offline training settings - apply defaults for missing values
        offline_dataset_path=config_data['training'].get('offline', {}).get('dataset_path') or CONFIG_DEFAULTS['offline_dataset_path'],
        offline_epochs=config_data['training'].get('offline', {}).get('epochs') or CONFIG_DEFAULTS['offline_epochs'],
        offline_batch_size=config_data['training'].get('offline', {}).get('batch_size') or CONFIG_DEFAULTS['offline_batch_size'],
        offline_learning_rate=config_data['training'].get('offline', {}).get('learning_rate') or CONFIG_DEFAULTS['offline_learning_rate'],
        offline_lr_schedule=offline_lr_schedule,
        offline_max_grad_norm=config_data['training'].get('offline', {}).get('max_grad_norm') or CONFIG_DEFAULTS['offline_max_grad_norm'],
        
        # Dataset collection settings - apply defaults for missing values
        dataset_collection_num_episodes=config_data.get('dataset_collection', {}).get('num_episodes') or CONFIG_DEFAULTS['dataset_collection_num_episodes'],
        dataset_collection_collection_type=config_data.get('dataset_collection', {}).get('collection_type') or CONFIG_DEFAULTS['dataset_collection_collection_type'],
        dataset_collection_model_path=config_data.get('dataset_collection', {}).get('model_path') or CONFIG_DEFAULTS['dataset_collection_model_path'],
        dataset_collection_output_path=config_data.get('dataset_collection', {}).get('output_path') or CONFIG_DEFAULTS['dataset_collection_output_path'],
        
        # Experiment configuration
        seed=config_data['experiment']['seed'],
        
        # Runtime progress tracking (initialized to default values)
        trained_timesteps=CONFIG_DEFAULTS['trained_timesteps'],
        trained_epochs=CONFIG_DEFAULTS['trained_epochs'],
        
        # Runtime metadata (initialized to default values)
        model_class=CONFIG_DEFAULTS['model_class'],
        start_time=CONFIG_DEFAULTS['start_time'],
        last_updated=CONFIG_DEFAULTS['last_updated']
    )


def create_config_from_args(args) -> Config:
    """
    Create a Config object directly from command line arguments.
    
    Args:
        args: Parsed command line arguments from argparse
        
    Returns:
        Config: Configuration object with values from args and defaults
    """
    # Enum converters for string-to-enum conversion
    enum_converters = {
        'network_type': NetworkType,
        'training_mode': TrainingMode
    }
    
    # Build configuration dictionary from command line arguments
    config_kwargs = {}
    for field_name in Config.__dataclass_fields__.keys():
        if hasattr(args, field_name):
            value = getattr(args, field_name)
            # Apply enum conversion if needed
            if field_name in enum_converters:
                value = enum_converters[field_name](value)
            config_kwargs[field_name] = value
        else:
            # Use default values for missing fields
            if field_name in CONFIG_DEFAULTS:
                config_kwargs[field_name] = CONFIG_DEFAULTS[field_name]
                if field_name in enum_converters:
                    config_kwargs[field_name] = enum_converters[field_name](CONFIG_DEFAULTS[field_name])
    
    return Config(**config_kwargs)


def create_config_with_overrides(config_path: str, args) -> Config:
    """
    Create a Config object from YAML file with command-line argument overrides.
    
    This function loads a base configuration from a YAML file and then applies
    any command-line argument overrides that were explicitly set by the user.
    
    Args:
        config_path: Path to the base YAML configuration file
        args: Parsed command line arguments that may override config values
        
    Returns:
        Config: Configuration object with base config + user overrides
    """
    # Load base configuration from file
    base_config = load_config(config_path)
    
    # Define converters for enum types
    CONVERTERS = {
        'network_type': NetworkType,
        'training_mode': TrainingMode,
    }
    
    def should_override(field_name: str, arg_value) -> bool:
        """
        Determine if a command-line argument should override the base config.
        
        Args:
            field_name: Name of the configuration field
            arg_value: Value from command line arguments
            
        Returns:
            bool: True if the argument should override the base config
        """
        if not hasattr(args, field_name):
            return False
        
        # Never override with None values (None means "not set")
        if arg_value is None:
            return False
        
        # Use _user_set_fields if available (explicit user override tracking)
        if hasattr(args, '_user_set_fields') and field_name in args._user_set_fields:
            return True
            
        # Fallback: check against known parser defaults
        if field_name in CONFIG_DEFAULTS:
            return arg_value != CONFIG_DEFAULTS[field_name]
        
        # For unknown fields, assume override if value is not None
        return arg_value is not None
    
    # Build final configuration dictionary
    config_kwargs = {}
    
    # Process all configuration fields
    for field_name in base_config.__dataclass_fields__.keys():
        # Skip runtime metadata fields (preserve from base config)
        if field_name in ['trained_timesteps', 'trained_epochs', 'model_class', 
                         'start_time', 'last_updated']:
            config_kwargs[field_name] = getattr(base_config, field_name)
            continue
            
        # Get base value from loaded config
        base_value = getattr(base_config, field_name)
        
        # Check for command-line override
        if hasattr(args, field_name):
            arg_value = getattr(args, field_name)
            if should_override(field_name, arg_value):
                # Apply type converter if needed
                if field_name in CONVERTERS:
                    config_kwargs[field_name] = CONVERTERS[field_name](arg_value)
                else:
                    config_kwargs[field_name] = arg_value
            else:
                config_kwargs[field_name] = base_value
        else:
            config_kwargs[field_name] = base_value
    
    return Config(**config_kwargs)



