"""
Dataset Creation and Analysis Tool for RL

This script creates datasets from RL environments using trained models or random policies.
Includes data analysis with CSV export and visualization.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.utils import set_seed
from src.model import NetworkType
from src.utils.config_loader import load_config
from src.environment import create_environment
from src.datasets.data_collector import DataCollector
from src.wrappers.cycle_index_wrapper import CycleIndexWrapper
from src.wrappers.sequence_wrapper import ObservationSequenceWrapper
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

def create_analysis(dataset, output_path):
    """
    Create comprehensive analysis of dataset with CSV export and plots.

    Args:
        dataset (Dataset): The dataset to analyze.
        output_path (str): The path to save the analysis results.

    Returns:
        str: The path to the created dataset.
    """
    if not dataset:
        logger.warning("No episodes to analyze.")
        return None

    logger.data(f"Analyzing {len(dataset)} episodes...")

    # Get data structure info from first episode
    first_episode = dataset.get_episode(0)
    actions = np.array(first_episode['actions'])
    observations = np.array(first_episode['observations'])
    
    # Determine dimensions
    action_dims = actions.shape[-1] if len(actions.shape) > 1 else 1
    
    if len(observations.shape) == 3:  # Sequential observations
        obs_dims = observations.shape[-1]
        observations = observations[:, -1, :]  # Use last timestep
    else:
        obs_dims = observations.shape[-1] if len(observations.shape) > 1 else 1
    
    max_length = max(len(dataset.get_episode(i)['actions']) for i in range(len(dataset)))
    
    # Collect all data
    all_actions = np.full((len(dataset), max_length, action_dims), np.nan)
    all_observations = np.full((len(dataset), max_length, obs_dims), np.nan)
    
    for i in range(len(dataset)):
        episode = dataset.get_episode(i)
        ep_actions = np.array(episode['actions'])
        ep_observations = np.array(episode['observations'])
        
        # Handle action dimensions
        if len(ep_actions.shape) == 1:
            ep_actions = ep_actions.reshape(-1, 1)
        length = len(ep_actions)
        all_actions[i, :length] = ep_actions
        
        # Handle observation dimensions
        if len(ep_observations.shape) == 3:
            ep_observations = ep_observations[:, -1, :]  # Last timestep
        elif len(ep_observations.shape) == 1:
            ep_observations = ep_observations.reshape(-1, 1)
        
        obs_length = min(length, len(ep_observations))
        all_observations[i, :obs_length] = ep_observations[:obs_length, :obs_dims]
    
    # Calculate statistics
    mean_actions = np.nanmean(all_actions, axis=0)
    std_actions = np.nanstd(all_actions, axis=0)
    mean_observations = np.nanmean(all_observations, axis=0)
    std_observations = np.nanstd(all_observations, axis=0)
    
    base_name = os.path.splitext(output_path)[0]
    
    # Create cycle_analysis subdirectories for actions and observations
    cycle_analysis_dir = os.path.join(os.path.dirname(base_name), "cycle_analysis")
    actions_dir = os.path.join(cycle_analysis_dir, "actions")
    observations_dir = os.path.join(cycle_analysis_dir, "observations")
    
    os.makedirs(actions_dir, exist_ok=True)
    os.makedirs(observations_dir, exist_ok=True)
    
    # Create and save CSV data
    actions_csv = _create_csv_data(mean_actions, std_actions, max_length, action_dims)
    observations_csv = _create_csv_data(mean_observations, std_observations, max_length, obs_dims)
    
    actions_csv_path = os.path.join(actions_dir, "dataset_actions.csv")
    observations_csv_path = os.path.join(observations_dir, "dataset_observations.csv")
    
    pd.DataFrame(actions_csv).to_csv(actions_csv_path, index=False)
    pd.DataFrame(observations_csv).to_csv(observations_csv_path, index=False)
    
    # Create plots
    action_plot_path = _create_plot(actions_csv, os.path.join(actions_dir, "dataset_actions.svg"), "Actions", action_dims)
    obs_plot_path = _create_plot(observations_csv, os.path.join(observations_dir, "dataset_observations.svg"), "Observations", obs_dims)

    logger.data(f"Finished analysis. Dataset files saved in cycle_analysis subdirectories.")

    return actions_csv_path, observations_csv_path, action_plot_path, obs_plot_path


def _create_csv_data(means, stds, max_length, num_dims):
    """
    Create CSV data from mean and std arrays.

    Args:
        means (np.ndarray): Mean values array.
        stds (np.ndarray): Standard deviation values array.
        max_length (int): Maximum length of the sequences.
        num_dims (int): Number of dimensions (channels).

    Returns:
        list: List of dictionaries containing CSV data.
    """
    csv_data = []
    for step in range(max_length):
        # Only include timesteps with at least some valid data
        if np.any(~np.isnan(means[step])):
            row = {'step': step}
            for dim in range(num_dims):
                row[f'mean_ch{dim}'] = means[step, dim] if not np.isnan(means[step, dim]) else None
                row[f'std_ch{dim}'] = stds[step, dim] if not np.isnan(stds[step, dim]) else None
            csv_data.append(row)
    return csv_data


def _create_plot(csv_data, plot_path, title, num_dims):
    """
    Create visualization plot from CSV data.

    Args:
        csv_data (list): The CSV data to plot.
        plot_path (str): The file path to save the plot.
        title (str): The title of the plot.
        num_dims (int): The number of dimensions (channels) to plot.
    
    Returns:
        str: The file path to the created plot.
    """
    if not csv_data:
        return plot_path
        
    df = pd.DataFrame(csv_data)
    fig, axes = plt.subplots(num_dims, 1, figsize=(15, 4 * num_dims))
    if num_dims == 1:
        axes = [axes]
    
    fig.suptitle(f'Average {title} Values Across Episodes', fontsize=16)
    colors = plt.get_cmap('tab10').colors
    
    for dim in range(num_dims):
        ax = axes[dim]
        mean_col, std_col = f'mean_ch{dim}', f'std_ch{dim}'
        
        if mean_col in df.columns and std_col in df.columns:
            valid_data = df.dropna(subset=[mean_col, std_col]).sort_values('step')
            
            if len(valid_data) > 0:
                steps = valid_data['step'].values
                means = valid_data[mean_col].values
                stds = valid_data[std_col].values
                
                color = colors[dim % 10]
                ax.plot(steps, means, color=color, linewidth=2, label='Mean')
                ax.fill_between(steps, means - stds, means + stds, 
                              alpha=0.3, color=color, label='±1 Std')
        
        ax.set_xlabel('Step')
        ax.set_ylabel(f'{title} Ch{dim}')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(left=0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def create_dataset_from_config(config_path: str, seed: int = None, 
                             make_analysis: bool = True, analysis_only: bool = False):
    """
    Create dataset based on configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
        seed (int, optional): Random seed for reproducibility.
        make_analysis (bool, optional): Whether to create analysis plots.
        analysis_only (bool, optional): If True, only create analysis without saving dataset.
    
    Returns:
        str: The path to the created dataset.
    """
    config = load_config(config_path)
    
    if seed is None:
        seed = config.seed
    set_seed(seed)
    
    logger.data(f"Creating {config.dataset_collection_collection_type} dataset with config {config_path}...")
    
    # Create environment with wrappers
    env = create_environment(config.env_name, render_mode=None, seed=seed, noise_std=config.noise_std, noise_type=config.noise_type)
    
    # Add wrappers for CycleNet in correct order
    if config.network_type == NetworkType.CYCLENET:
        # 1. First add cycle index wrapper (adds cycle_index to each observation)
        env = CycleIndexWrapper(env, cycle_length=getattr(config, 'cyclenet_cycle', 24))
        logger.data(f"Added CycleIndexWrapper with cycle_length {getattr(config, 'cyclenet_cycle', 24)}.")
        
        # 2. Then add sequence wrapper if needed (collects observations with their cycle_indices)
        if config.cyclenet_seq_len > 1:
            env = ObservationSequenceWrapper(env, seq_len=config.cyclenet_seq_len)
            logger.data(f"Added ObservationSequenceWrapper with seq_len {config.cyclenet_seq_len}.")
    
    # Setup data collector
    model_path = config.dataset_collection_model_path
    if model_path and "{seed}" in model_path:
        model_path = model_path.format(seed=seed)
    
    collector = DataCollector(env, model_path)
    
    # Collect data
    use_random = config.dataset_collection_collection_type == "random"
    if config.dataset_collection_collection_type == "model" and (not model_path or not os.path.exists(model_path)):
        raise ValueError(f"Model path {model_path} does not exist")
    
    dataset = collector.collect_episodes(config.dataset_collection_num_episodes, use_random)
    
    # Handle output path with seed templating
    output_path = config.dataset_collection_output_path
    if "{seed}" in output_path:
        output_path = output_path.format(seed=seed)
    else:
        base, ext = os.path.splitext(output_path)
        if not base.endswith(f"_{seed}"):
            output_path = f"{base}_{seed}{ext}"
    
    # Save dataset unless analysis-only
    if not analysis_only:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.save(output_path)
    
    # Create analysis if requested
    if make_analysis:
        create_analysis(dataset, output_path)
    
    # Show stats
    stats = dataset.get_stats()
    logger.data(f"Finished {config.dataset_collection_collection_type} dataset collection for {stats['num_episodes']} episodes with config {config_path}.")
    logger.data(f"Total transitions: {stats['num_transitions']}, mean reward: {stats['mean_reward']:.3f}.")

    logger.data(f"Data collection completed successfully!")

    if not analysis_only:
        return output_path
    return None


def main():
    """
    Command line interface.
    """
    parser = argparse.ArgumentParser(description="Create RL dataset with analysis")
    parser.add_argument('config', help='YAML configuration file path')
    parser.add_argument('--seed', type=int, help='Override config seed')
    parser.add_argument('--no-analysis', action='store_true', help='Skip analysis creation')
    parser.add_argument('--analysis-only', action='store_true', help='Only create analysis, don\'t save dataset')
    
    args = parser.parse_args()

    logger.data(f"Starting data collection...")
    
    create_dataset_from_config(
        args.config, 
        args.seed, 
        not args.no_analysis,
        args.analysis_only
    )


if __name__ == "__main__":
    main()
