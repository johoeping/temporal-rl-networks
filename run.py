"""
Main Training Runner for CycleNet RL Networks

This is the main entry point for training RL models with various
network architectures including MLP, LSTM, and CycleNet. It supports both online
and offline training modes with comprehensive configuration management.
"""

import sys
import os
import argparse

from src.runner import TrainingRunner
from src.utils.config_loader import create_config_from_args, create_config_with_overrides
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

parser = argparse.ArgumentParser(
    description="Train RL models with various network architectures and algorithms",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

# Config file path
parser.add_argument('--config', '--config-path', dest='config_path', 
                    help='Path to the configuration YAML file (arguments can override config values)')

# Required arguments
parser.add_argument('--output-dir', required=True,
                    help='Output directory for training results')

# Environment configuration
parser.add_argument('--env-name', default='Walker2d-v5',
                    help='Environment name (default: Walker2d-v5)')
parser.add_argument('--render-env', action='store_true', default=False,
                    help='Render environment during training')

# Noise configuration
parser.add_argument('--noise-std', type=float, default=0.0,
                    help='Standard deviation of noise to add to observations (default: 0.0)')
parser.add_argument('--noise-type', choices=['gaussian', 'uniform'], default='gaussian',
                    help='Type of noise to add to observations (default: gaussian)')

# Model configuration  
parser.add_argument('--network-type', choices=['mlp', 'lstm', 'cyclenet'], default='cyclenet',
                    help='Network type (default: cyclenet)')
parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu',
                    help='Training device (default: cpu)')

# MLP network arguments
parser.add_argument('--mlp-net-arch', nargs='+', type=int, default=[16, 16],
                    help='MLP network architecture (default: [16, 16])')

# LSTM network arguments
parser.add_argument('--lstm-net-arch', nargs='+', type=int, default=[16, 16],
                    help='LSTM network architecture (default: [16, 16])')
parser.add_argument('--lstm-hidden-size', type=int, default=16,
                    help='LSTM hidden size (default: 16)')

# CycleNet network arguments
parser.add_argument('--cyclenet-seq-len', type=int, default=16,
                    help='CycleNet sequence length (default: 16)')
parser.add_argument('--cyclenet-cycle', type=int, default=16,
                    help='CycleNet cycle length (default: 16)')
parser.add_argument('--cyclenet-model-type', choices=['mlp', 'linear'], default='linear',
                    help='CycleNet model type (default: linear)')
parser.add_argument('--cyclenet-d-model', type=int, default=16,
                    help='CycleNet model dimension (default: 16)')
parser.add_argument('--cyclenet-use-revin', action='store_true', default=False,
                    help='Use RevIN normalization in CycleNet')

# Training configuration
parser.add_argument('--training-mode', choices=['online', 'offline'], default='online',
                    help='Training mode (default: online)')
parser.add_argument('--rolling-length', type=int, default=35,
                    help='Rolling window length for metrics (default: 35)')
parser.add_argument('--enable-live-plot', action='store_true', default=False,
                    help='Enable live plotting during training')

# Online training arguments
parser.add_argument('--online-total-timesteps', type=int, default=1000000,
                    help='Total timesteps for online training (default: 1000000)')
parser.add_argument('--online-batch-size', type=int, default=256,
                    help='Batch size for online training (default: 256)')
parser.add_argument('--online-learning-rate', type=float, default=0.0003,
                    help='Learning rate for online training (default: 0.0003)')
parser.add_argument('--online-lr-schedule', choices=['linear', 'constant'], default='constant',
                    help='Learning rate schedule for online training (default: constant)')
parser.add_argument('--online-n-steps', type=int, default=2048,
                    help='Number of steps per update (PPO) (default: 2048)')
parser.add_argument('--online-max-grad-norm', type=float, default=0.5,
                    help='Maximum gradient norm for online training (default: 0.5)')

# Offline training arguments
parser.add_argument('--offline-dataset-path', 
                    help='Path to dataset for offline training')
parser.add_argument('--offline-epochs', type=int, default=100,
                    help='Number of epochs for offline training (default: 100)')
parser.add_argument('--offline-batch-size', type=int, default=256,
                    help='Batch size for offline training (default: 256)')
parser.add_argument('--offline-learning-rate', type=float, default=0.0001,
                    help='Learning rate for offline training (default: 0.0001)')
parser.add_argument('--offline-lr-schedule', choices=['linear', 'constant'], default='constant',
                    help='Learning rate schedule for offline training (default: constant)')
parser.add_argument('--offline-max-grad-norm', type=float, default=0.5,
                    help='Maximum gradient norm for offline training (default: 0.5)')

# Dataset collection arguments
parser.add_argument('--dataset-collection-num-episodes', type=int, default=100,
                    help='Number of episodes to collect for dataset (default: 100)')
parser.add_argument('--dataset-collection-collection-type', choices=['random', 'model'], default='model',
                    help='Dataset collection type (default: model)')
parser.add_argument('--dataset-collection-model-path',
                    help='Path to model for dataset collection')
parser.add_argument('--dataset-collection-output-path',
                    help='Output path for collected dataset')

# Experiment configuration
parser.add_argument('--seed', type=int, default=2025,
                    help='Random seed for reproducibility (default: 2025)')

args = parser.parse_args()

# Track which arguments were explicitly provided by the user for config overrides
user_provided_args = set()
for i, arg in enumerate(sys.argv[1:]):
    if arg.startswith('--'):
        arg_name = arg[2:].replace('-', '_')
        user_provided_args.add(arg_name)

# Add marker to args for the config loader
args._user_set_fields = user_provided_args

if __name__ == "__main__":
    if args.config_path:
        # Use config file as base with command line overrides
        if not os.path.exists(args.config_path):
            logger.error(f"Configuration file '{args.config_path}' not found.")
            sys.exit(1)
        logger.info(f"Using config file: {args.config_path}.")
        config = create_config_with_overrides(args.config_path, args)
    else:
        # Create config object directly from command line arguments
        logger.info("Using individual command line arguments.")
        config = create_config_from_args(args)

    # Create and run training
    runner = TrainingRunner(args.output_dir)
    runner.run_training(config)