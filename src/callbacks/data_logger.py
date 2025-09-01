"""
Training Data Logging and Visualization Callback

This module provides comprehensive data logging and visualization capabilities
for RL training. It tracks episode rewards, training metrics,
generates live plots, and exports data to CSV files for analysis.
"""

import time
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Optional
from stable_baselines3.common.callbacks import BaseCallback

from src.utils.paths import DataPaths
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

class DataLoggerCallback(BaseCallback):
    """
    Comprehensive training data logger with live visualization and CSV export.
    
    This callback provides complete training monitoring capabilities including:
    - Real-time episode reward and length tracking
    - Training metrics extraction (policy loss, value loss, etc.)
    - Live plot updates during training
    - CSV data export for post-training analysis
    - Rolling statistics calculation for trend analysis
    - Configurable update intervals for performance optimization
    
    The callback integrates seamlessly with stable-baselines3 training loops
    and provides both online monitoring and offline analysis capabilities.
    
    Attributes:
        rolling_length (int): Window size for rolling average calculations
        enable_live_plot (bool): Whether to show live training plots
        paths (DataPaths): File path management for outputs
        episode_rewards (List[float]): Collected episode reward values
        episode_lengths (List[int]): Collected episode length values
        training_metrics (Dict): Training metrics from the model logger
    """
    
    # Configuration constants
    PLOT_UPDATE_INTERVAL = 1000
    CSV_SAVE_INTERVAL = 60
    FIGSIZE = (10, 12)
    REWARD_COLOR = "blue"
    LENGTH_COLOR = "orange"
    ALPHA = 0.2
    
    # Mapping of training metrics
    METRIC_MAPPINGS = {
        'policy_loss': ['train/policy_loss', 'train/loss', 'train/clip_loss'],
        'value_loss': ['train/value_loss', 'train/vf_loss'],
        'entropy_loss': ['train/entropy_loss'],
        'learning_rate': ['train/learning_rate'],
        'explained_variance': ['train/explained_variance']
    }

    def __init__(self, rolling_length: int = 10, output_dir: Optional[str] = None, 
                 initial_data: Optional[Dict[str, List]] = None, enable_live_plot: bool = True):
        """
        Initialize the data logging callback with specified configuration.
        
        Args:
            rolling_length: Window size for calculating rolling averages and statistics
            output_dir: Directory path for saving CSV files and plots
            initial_data: Pre-existing data to load (for resuming training)
            enable_live_plot: Whether to enable live plot updates during training
        """
        super().__init__()
        
        self.rolling_length = rolling_length
        self.enable_live_plot = enable_live_plot
        self.paths = DataPaths(output_dir)
        
        # Episode tracking state
        self.current_reward = 0
        self.current_length = 0
        self.last_update_timestep = 0
        self.last_csv_save_time = time.time()
        
        # Data storage containers
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.training_metrics = {
            'policy_losses': [], 'value_losses': [], 'entropy_losses': [],
            'learning_rates': [], 'explained_variances': []
        }
        
        # Rolling statistics caches for efficient computation
        self.rewards_moving_avg: List[float] = []
        self.rewards_std: List[float] = []
        self.lengths_moving_avg: List[float] = []
        self.lengths_std: List[float] = []
        
        # Plot visualization components
        self.fig = None
        self.axs = None
        self.reward_line = None
        self.reward_shading = None
        self.length_line = None
        self.length_shading = None
        
        # Load initial data if provided (for resuming interrupted training)
        if initial_data:
            self.episode_rewards = initial_data.get("episode_rewards", [])
            self.episode_lengths = initial_data.get("episode_lengths", [])
            for key in self.training_metrics:
                if key in initial_data:
                    self.training_metrics[key] = initial_data[key].copy()
            self._recalculate_rolling_stats()
        
        # Initialize live plotting if enabled
        if self.enable_live_plot:
            self._initialize_plot()
    
    def _recalculate_rolling_stats(self) -> None:
        """Recalculate all rolling statistics from existing data for consistency."""
        self._calculate_rolling_stats(self.episode_rewards, self.rewards_moving_avg, self.rewards_std)
        self._calculate_rolling_stats(self.episode_lengths, self.lengths_moving_avg, self.lengths_std)
    
    def _calculate_rolling_stats(self, values: List[float], avg_list: List[float], std_list: List[float]) -> None:
        """
        Calculate rolling statistics for a complete list of values.
        
        Args:
            values: Input values for statistical calculation
            avg_list: Output list for rolling averages (cleared and repopulated)
            std_list: Output list for rolling standard deviations (cleared and repopulated)
        """
        avg_list.clear()
        std_list.clear()
        for i in range(len(values)):
            if i < self.rolling_length:
                window = values[:i + 1]
            else:
                window = values[i - self.rolling_length + 1:i + 1]
            avg_list.append(np.mean(window))
            std_list.append(np.std(window))
    
    def _initialize_plot(self) -> None:
        """
        Initialize matplotlib figure and axes for live training visualization.
        
        Sets up the figure with two subplots for rewards and episode lengths,
        configures appropriate backends for different environments, and
        initializes plot components for efficient updating.
        """
        # Configure matplotlib backend for environment compatibility
        current_backend = matplotlib.get_backend()
        if current_backend.lower() == 'agg' or not self.enable_live_plot:
            matplotlib.use('Agg')
        else:
            plt.ion()  # Enable interactive mode
        
        self.fig, self.axs = plt.subplots(2, 1, figsize=self.FIGSIZE)
        
        # Configure reward subplot
        self.reward_line, = self.axs[0].plot([], [], color=self.REWARD_COLOR)
        self.axs[0].set_title("Episode Rewards")
        self.axs[0].set_xlabel("Timesteps")
        self.axs[0].set_ylabel("Reward per Episode")
        self.axs[0].spines['top'].set_visible(False)
        self.axs[0].spines['right'].set_visible(False)
        
        # Configure length subplot
        self.length_line, = self.axs[1].plot([], [], color=self.LENGTH_COLOR)
        self.axs[1].set_title("Episode Lengths")
        self.axs[1].set_xlabel("Episode")
        self.axs[1].set_ylabel("Steps")
        self.axs[1].spines['top'].set_visible(False)
        self.axs[1].spines['right'].set_visible(False)
        
        plt.tight_layout()
        self._render_plot_data()
    
    def _on_step(self) -> bool:
        """
        Called at each environment step during training.
        
        This method accumulates episode statistics, handles episode completion,
        and triggers periodic updates for plotting and data saving.
        
        Returns:
            bool: Always True to continue training
        """
        self.current_reward += self.locals["rewards"][0]
        self.current_length += 1
        
        # Handle episode completion and update statistics
        if self.locals.get("dones")[0]:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            
            # Update rolling statistics incrementally for efficiency
            self._update_single_rolling_stat(self.episode_rewards, self.rewards_moving_avg, self.rewards_std)
            self._update_single_rolling_stat(self.episode_lengths, self.lengths_moving_avg, self.lengths_std)
            
            self.current_reward = 0
            self.current_length = 0
        
        # Periodic plot updates for live monitoring
        if (self.enable_live_plot and 
            self.num_timesteps - self.last_update_timestep >= self.PLOT_UPDATE_INTERVAL):
            self._update_plot()
            self.last_update_timestep = self.num_timesteps
        
        # Periodic CSV saves for data persistence
        current_time = time.time()
        if current_time - self.last_csv_save_time >= self.CSV_SAVE_INTERVAL:
            self._save_csv_data()
            self.last_csv_save_time = current_time
        
        return True
    
    def _update_single_rolling_stat(self, values: List[float], avg_list: List[float], std_list: List[float]) -> None:
        """
        Update rolling statistics for the most recently added value.
        
        This method efficiently updates rolling statistics without recalculating
        the entire history, improving performance during training.
        
        Args:
            values: Complete list of values including the new one
            avg_list: Rolling averages list to append to
            std_list: Rolling standard deviations list to append to
        """
        i = len(values) - 1
        if i < self.rolling_length:
            window = values[:i + 1]
        else:
            window = values[i - self.rolling_length + 1:i + 1]
        avg_list.append(np.mean(window))
        std_list.append(np.std(window))
    
    def _update_plot(self) -> None:
        """
        Update the live training plot with current data.
        
        Only updates if live plotting is enabled and data is available.
        Handles backend compatibility gracefully.
        """
        if not self.enable_live_plot or self.fig is None or not self.episode_rewards:
            return
        
        self._render_plot_data()
    
    def _render_plot_data(self) -> None:
        """
        Render current training data to the plot visualization.
        
        Updates both reward and episode length plots with current data including
        rolling averages and confidence bands. Handles title updates with
        current training statistics.
        """
        if self.fig is None or not self.episode_rewards:
            return
        
        # Ensure consistent data lengths and handle empty arrays
        if len(self.episode_lengths) == 0 or len(self.rewards_moving_avg) == 0:
            return
            
        # Update reward plot with rolling statistics
        # For offline training, use epoch numbers instead of cumulative timesteps
        if len(self.episode_lengths) != len(self.episode_rewards):
            # Fallback: use episode indices for x-axis
            timesteps = np.arange(1, len(self.episode_rewards) + 1)
        else:
            timesteps = np.cumsum(self.episode_lengths)
            
        avg = np.array(self.rewards_moving_avg)
        std = np.array(self.rewards_std)
        
        # Ensure arrays have same length
        min_length = min(len(timesteps), len(avg), len(std))
        timesteps = timesteps[:min_length]
        avg = avg[:min_length]
        std = std[:min_length]
        
        if min_length > 0:
            self.reward_line.set_data(timesteps, avg)
            if self.reward_shading:
                self.reward_shading.remove()
            self.reward_shading = self.axs[0].fill_between(
                timesteps, avg - std, avg + std, color=self.REWARD_COLOR, alpha=self.ALPHA
            )
            self.axs[0].set_xlim(0, timesteps[-1])
            self.axs[0].relim()
            self.axs[0].autoscale_view()
        
        # Update episode length plot with rolling statistics
        if len(self.episode_lengths) > 0 and len(self.lengths_moving_avg) > 0:
            episodes = np.arange(1, len(self.episode_lengths) + 1)
            avg = np.array(self.lengths_moving_avg)
            std = np.array(self.lengths_std)
            
            # Ensure arrays have same length
            min_length = min(len(episodes), len(avg), len(std))
            episodes = episodes[:min_length]
            avg = avg[:min_length]
            std = std[:min_length]
            
        if min_length > 0:
            self.length_line.set_data(episodes, avg)
            if self.length_shading:
                self.length_shading.remove()
            self.length_shading = self.axs[1].fill_between(
                episodes, avg - std, avg + std, color=self.LENGTH_COLOR, alpha=self.ALPHA
            )
            self.axs[1].set_xlim(0, episodes[-1])
            self.axs[1].relim()
            self.axs[1].autoscale_view()
        
        # Update title with current training statistics
        current_timestep = np.sum(self.episode_lengths)
        current_episode = len(self.episode_rewards)
        avg_reward = self.rewards_moving_avg[-1] if self.rewards_moving_avg else 0.0
        
        # Update display with backend compatibility
        try:
            if matplotlib.get_backend().lower() != 'agg':
                plt.pause(0.01)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except (RuntimeError, AttributeError):
            # Handle non-interactive backends gracefully
            pass
    
    def _save_csv_data(self) -> None:
        """
        Save current training data to CSV files for persistence and analysis.
        
        Exports both episode-level data (rewards, lengths) and detailed training
        metrics (policy loss, learning rate, etc.) to separate CSV files.
        Handles missing data gracefully with appropriate padding.
        """
        try:
            # Save episode-level data
            if self.episode_rewards:
                episode_data = {
                    "Episode": list(range(1, len(self.episode_rewards) + 1)),
                    "Reward": self.episode_rewards,
                    "Length": self.episode_lengths
                }
                pd.DataFrame(episode_data).to_csv(self.paths.episode_data, index=False)
            
            # Save detailed training metrics
            max_length = max(len(lst) for lst in self.training_metrics.values()) if self.training_metrics else 0
            if max_length > 0:
                training_data = {"Rollout": list(range(1, max_length + 1))}
                column_names = {
                    'policy_losses': 'Policy_Loss', 'value_losses': 'Value_Loss',
                    'entropy_losses': 'Entropy_Loss', 'learning_rates': 'Learning_Rate',
                    'explained_variances': 'Explained_Variance'
                }
                for key, col_name in column_names.items():
                    metric_list = self.training_metrics[key]
                    # Pad shorter lists with None values
                    training_data[col_name] = metric_list + [None] * (max_length - len(metric_list))
                pd.DataFrame(training_data).to_csv(self.paths.training_metrics, index=False)
        except Exception as e:
            logger.error(f"Error saving CSV data: {e}")
    
    def _on_rollout_end(self) -> None:
        """
        Extract and store training metrics from the model's logger.
        
        Called at the end of each rollout to collect training metrics like
        policy loss, value loss, learning rate, etc. Maps various metric
        names to standardized storage keys for consistent data handling.
        """
        try:
            if not (hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value')):
                return
            
            logs = self.model.logger.name_to_value
            
            # Map singular metric names to plural storage keys for consistency
            metric_mapping = {
                'policy_loss': 'policy_losses',
                'value_loss': 'value_losses', 
                'entropy_loss': 'entropy_losses',
                'learning_rate': 'learning_rates',
                'explained_variance': 'explained_variances'
            }
            
            # Extract metrics using multiple possible key names
            for metric_name, metric_keys in self.METRIC_MAPPINGS.items():
                storage_key = metric_mapping[metric_name]
                metric_list = self.training_metrics[storage_key]
                for key in metric_keys:
                    if key in logs:
                        metric_list.append(float(logs[key]))
                        break
        except Exception as e:
            logger.error(f"Error extracting training metrics: {e}")
    
    def save_final_data(self) -> None:
        """
        Save all collected data and create final visualization.
        
        This method performs final data export including CSV files, summary
        statistics, and plots. Called at the end of training
        or when training is interrupted.
        """
        self._save_csv_data()
        
        # Generate comprehensive summary statistics
        if self.episode_rewards:
            summary_data = {
                "Metric": [
                    "Total_Episodes", "Total_Timesteps", "Final_Episode_Reward",
                    "Best_Episode_Reward", "Worst_Episode_Reward", "Mean_Episode_Reward",
                    "Std_Episode_Reward", "Final_Policy_Loss", "Final_Value_Loss"
                ],
                "Value": [
                    len(self.episode_rewards),
                    int(np.sum(self.episode_lengths)) if self.episode_lengths else 0,
                    self.episode_rewards[-1], max(self.episode_rewards), min(self.episode_rewards),
                    float(np.mean(self.episode_rewards)), float(np.std(self.episode_rewards)),
                    self.training_metrics['policy_losses'][-1] if self.training_metrics['policy_losses'] else 0,
                    self.training_metrics['value_losses'][-1] if self.training_metrics['value_losses'] else 0
                ]
            }
            pd.DataFrame(summary_data).to_csv(self.paths.summary, index=False)
        
        # Create and save final plot
        if self.fig:
            self._render_plot_data()
            plt.ioff()
            self.fig.savefig(self.paths.training_plot)
            plt.close(self.fig)
        elif not self.enable_live_plot and self.episode_rewards:
            # Create plot specifically for saving when live plotting was disabled
            matplotlib.use('Agg')
            self._initialize_plot()
            self._render_plot_data()
            plt.ioff()
            self.fig.savefig(self.paths.training_plot)
            plt.close(self.fig)
        
        logger.training("Training data and visualizations saved successfully.")