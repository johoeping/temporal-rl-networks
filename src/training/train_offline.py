"""
Offline RL Training Module

This module implements offline training (behavioral cloning) for RL
models using pre-collected datasets. Supports both regular and sequential models with
efficient data processing and comprehensive progress tracking.
"""

import torch
import random
import numpy as np

from src.model import NetworkType
from src.utils.paths import DataPaths
from src.datasets.offline_dataset import OfflineDataset
from src.callbacks.data_logger import DataLoggerCallback
from src.policies.cyclenet_policy import save_cycle_data_plot
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

class OfflineTrainer:
    """
    Implements offline RL training using behavioral cloning.

    This class handles training RL policies on pre-collected datasets through
    behavioral cloning, where the model learns to imitate actions from expert
    demonstrations. Supports both standard and sequential model architectures
    with efficient batch processing and progress tracking.
    
    The trainer implements pure behavioral cloning by minimizing the negative
    log-likelihood of expert actions given observed states, effectively teaching
    the policy to replicate the behavior seen in the dataset.
    
    Attributes:
        model: The RL model to train (PPO-based policy)
        dataset: Offline dataset containing expert transitions
        config: Configuration object with training parameters
        device: PyTorch device for computation (CPU/CUDA)
    """
    
    def __init__(self, model, dataset: OfflineDataset, config):
        """
        Initialize offline trainer with model, dataset, and configuration.
        
        Args:
            model: The RL model to train (PPO policy)
            dataset: Offline dataset containing expert transitions
            config: Configuration object with all training parameters
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = next(model.policy.parameters()).device
        
    def train(self, output_dir: str = None, seed: int = None):
        """
        Execute offline training using behavioral cloning on the dataset.
        
        The training uses pure behavioral cloning, minimizing negative log-likelihood
        of expert actions. For sequential models (CycleNet with seq_len > 1), data
        is generated on-the-fly to handle sequence construction efficiently.
        
        Args:
            output_dir: Directory for saving training outputs and checkpoints
            seed: Random seed for reproducibility tracking and logging
            
        Raises:
            ValueError: If dataset is empty or required paths don't exist
            Exception: Other training errors handled with checkpoint saving
        """
        # Validate dataset
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty - cannot proceed with offline training")
            
        num_epochs = self.config.offline_epochs
        batch_size = self.config.offline_batch_size
        rolling_length = self.config.rolling_length
        
        logger.training(f"Starting offline training with {self.dataset.get_total_transitions()} transitions "
                       f"from {len(self.dataset)} episodes for {num_epochs} epochs.")

        # Check if training is already complete
        if self.config.is_training_complete():
            logger.training(f"Training already complete: {self.config.trained_epochs}/{num_epochs} epochs.")
            return
            
        remaining_epochs = self.config.get_remaining_work()
        logger.training(f"Resuming training from {self.config.trained_epochs} epochs. Remaining epochs: {remaining_epochs}.")

        # Initialize progress tracking callbacks
        callbacks = []
        if output_dir:
            data_logger = DataLoggerCallback(
                rolling_length=rolling_length,
                output_dir=output_dir,
                enable_live_plot=self.config.enable_live_plot
            )
            callbacks.append(data_logger)
        
        try:
            # Determine data processing strategy based on model architecture
            seq_len = self.config.cyclenet_seq_len or 1
            is_sequence_model = seq_len > 1 and self.config.network_type.value == "cyclenet"
            
            # For CycleNet with seq_len=1, use efficient flattened approach
            if seq_len == 1 and self.config.network_type.value == "cyclenet":
                logger.training(f"CycleNet in MLP mode (seq_len=1) - using optimized flattened data processing")
                is_sequence_model = False
            
            # Prepare training data based on model type
            if is_sequence_model:
                logger.training(f"CycleNet sequence model (seq_len={seq_len}) - using on-the-fly sequence generation.")
                obs_sequences = None
                actions_list = None 
            else:
                # Flatten all episodes for non-sequence models
                all_obs = []
                all_actions = []
                
                for episode in self.dataset.episodes:
                    all_obs.extend(episode['observations'])
                    all_actions.extend(episode['actions'])
                
                obs_sequences = torch.from_numpy(np.array(all_obs, dtype=np.float32)).to(self.device)
                actions_list = torch.from_numpy(np.array(all_actions, dtype=np.float32)).to(self.device)
                logger.training(f"Preprocessed flattened data: obs={obs_sequences.shape}, actions={actions_list.shape}.")

            logger.training(f"Training on device: {self.device}.")

            trained_epochs = self.config.trained_epochs

            logger.training(f"Starting training loop (this could take a while)...")

            # Execute epoch-based training loop
            for epoch in range(remaining_epochs):
                epoch_losses = []
                epoch_policy_losses = []
                epoch_learning_rates = []
                
                if is_sequence_model:
                    # Process sequential data with on-the-fly generation
                    num_batches_processed = 0
                    num_samples_processed = 0
                    
                    for batch_obs, batch_actions, _ in self._generate_sequence_batches(seq_len, batch_size):
                        batch_metrics = self._train_batch(batch_obs, batch_actions)
                        
                        epoch_losses.append(batch_metrics['policy_loss'])
                        epoch_policy_losses.append(batch_metrics['policy_loss'])
                        epoch_learning_rates.append(batch_metrics['learning_rate'])
                        num_batches_processed += 1
                        num_samples_processed += len(batch_obs)
                    
                    num_samples = num_samples_processed
                    num_batches = num_batches_processed
                else:
                    # Process pre-flattened data efficiently
                    num_samples = obs_sequences.shape[0]
                    indices = torch.randperm(num_samples)
                    
                    num_batches = 0
                    for i in range(0, num_samples, batch_size):
                        batch_indices = indices[i:i+batch_size]
                        batch_obs = obs_sequences[batch_indices]
                        batch_actions = actions_list[batch_indices]
                        
                        batch_metrics = self._train_batch(batch_obs, batch_actions)
                        
                        epoch_losses.append(batch_metrics['policy_loss'])
                        epoch_policy_losses.append(batch_metrics['policy_loss'])
                        epoch_learning_rates.append(batch_metrics['learning_rate'])
                        num_batches += 1
                
                # Calculate and log epoch statistics
                avg_policy_loss = np.mean(epoch_policy_losses)
                avg_lr = np.mean(epoch_learning_rates)
                
                current_epoch = trained_epochs + epoch + 1

                # Update training progress
                self.config.update_progress(epochs=current_epoch)
                
                # Update callbacks with behavioral cloning metrics
                self._update_callbacks(callbacks, current_epoch, {
                    'policy_loss': avg_policy_loss,
                    'learning_rate': avg_lr
                })

                # Additional print for the last epoch
                if epoch == remaining_epochs - 1:
                    logger.training(f"Epoch {current_epoch} completed: policy_loss={avg_policy_loss}.")
                else:
                    logger.training(f"Epoch {current_epoch} completed: policy_loss={avg_policy_loss}. Continuing...")

        except Exception as e:
            logger.warning(f"Training interrupted by error: {e} Saving model checkpoint...")

        finally:
            # Always attempt to save progress and results
            try:
                if callbacks:
                    callbacks[0].save_final_data()
                    logger.training("Successfully saved training data and visualizations.")
            except Exception as save_error:
                logger.warning(f"Failed to save callback data: {save_error}")

            # Save model and metadata
            try:
                paths = DataPaths(output_dir)
                self.model.save(paths.model_checkpoint)
                
                # Update and save configuration metadata
                self.config.update_progress(epochs=self.config.trained_epochs)
                self.config.save_metadata(output_dir)

                # Generate CycleNet-specific analysis if applicable
                if self.config.network_type == NetworkType.CYCLENET:
                    try:
                        save_cycle_data_plot(self.model.policy.features_extractor.model.cycleQueue.data, output_dir)
                    except Exception as e:
                        logger.warning(f"Could not save CycleNet cycle data plot: {e}")

                logger.training(f"Offline training completed. Model saved to {output_dir}.")
            except Exception as e:
                logger.error(f"Error saving model: {e}")

    def _train_batch(self, obs, actions):
        """
        Execute a single training batch using behavioral cloning.
        
        This method implements pure behavioral cloning by:
        1. Getting action distribution from current policy given observations
        2. Calculating negative log-likelihood of expert actions
        3. Performing gradient descent to minimize this loss
        
        Args:
            obs: Batch of observations (torch.Tensor)
            actions: Batch of expert actions (torch.Tensor)
            
        Returns:
            dict: Training metrics including policy loss and learning rate
        """
        try:
            self.model.policy.train()
            current_lr = self.model.policy.optimizer.param_groups[0]['lr']
            
            with torch.set_grad_enabled(True):
                # Get action distribution from current policy
                distribution = self.model.policy.get_distribution(obs)
                # Calculate negative log-likelihood of expert actions
                log_probs = distribution.log_prob(actions)
                policy_loss = -log_probs.mean()
            
            # Perform gradient update
            self.model.policy.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 0.5)
            self.model.policy.optimizer.step()
            
            return {
                'policy_loss': policy_loss.item(),
                'learning_rate': current_lr
            }
            
        except Exception as e:
            logger.error(f"Error in batch training: {e}")
            return {'policy_loss': 0.0, 'learning_rate': 0.0}

    def _update_callbacks(self, callbacks, epoch, metrics):
        """
        Update progress tracking callbacks with current training metrics.
        
        This method updates the data logger callback with behavioral cloning
        metrics, treating the negative policy loss as a reward-like metric
        for consistency with online training visualizations.
        
        Args:
            callbacks: List of callback objects to update
            epoch: Current training epoch number
            metrics: Dictionary containing training metrics (policy_loss, learning_rate)
        """
        if not callbacks:
            return
            
        # Update callback data structures with metrics
        for callback in callbacks:
            if hasattr(callback, 'episode_rewards'):
                # Use negative policy loss as reward-like metric
                callback.episode_rewards.append(-metrics['policy_loss'])
            if hasattr(callback, 'episode_lengths'):
                # For offline training, use a consistent length (e.g., epoch number)
                callback.episode_lengths.append(epoch)
            
            # Update rolling statistics to ensure consistency between rewards and lengths
            if (hasattr(callback, 'episode_rewards') and hasattr(callback, 'episode_lengths') and
                len(callback.episode_rewards) == len(callback.episode_lengths)):
                # Only update rolling stats if arrays have consistent lengths
                if hasattr(callback, '_update_single_rolling_stat'):
                    callback._update_single_rolling_stat(
                        callback.episode_rewards, callback.rewards_moving_avg, callback.rewards_std
                    )
                    callback._update_single_rolling_stat(
                        callback.episode_lengths, callback.lengths_moving_avg, callback.lengths_std
                    )
            
            # Update training metrics if callback has these attributes
            if hasattr(callback, 'training_metrics'):
                if 'policy_losses' in callback.training_metrics:
                    callback.training_metrics['policy_losses'].append(metrics['policy_loss'])
                if 'learning_rates' in callback.training_metrics:
                    callback.training_metrics['learning_rates'].append(metrics['learning_rate'])
                
            # Trigger data saving and plot updates
            if hasattr(callback, '_save_csv_data'):
                callback._save_csv_data()
                
            if hasattr(callback, 'enable_live_plot') and callback.enable_live_plot:
                if hasattr(callback, '_update_plot'):
                    callback._update_plot()

    def _generate_sequence_batches(self, seq_len: int, batch_size: int):
        """
        Generator that yields batches of sequential data for CycleNet training.
        
        This method generates training batches for sequence-based models by:
        1. Creating sequences from episode data with proper temporal ordering
        2. Adding cycle indices based on configured cycle length
        3. Handling sequence padding for timesteps before episode start
        4. Yielding batches of the specified size for efficient training
        
        The generator processes episodes in shuffled order and constructs sequences
        on-the-fly to handle memory efficiently for large datasets.
        
        Args:
            seq_len: Length of sequences to generate for each sample
            batch_size: Number of sequences per batch
            
        Yields:
            tuple: (observations_batch, actions_batch, None) where:
                - observations_batch: [batch_size, seq_len, obs_dim] tensor
                - actions_batch: [batch_size, action_dim] tensor
                - None: Placeholder for compatibility with standard batch format
        """
        large_batch_size = min(batch_size * 4, 8192)  # Process in larger chunks for efficiency
        batch_obs_sequences = []
        batch_actions = []
        
        # Randomize episode processing order for better training dynamics
        episodes_shuffled = list(enumerate(self.dataset.episodes))
        random.shuffle(episodes_shuffled)
        
        for episode_idx, episode in episodes_shuffled:
            obs_list = episode['observations']
            actions_list = episode['actions']
            episode_length = len(obs_list)
            
            cycle_length = self.config.cyclenet_cycle or seq_len
            
            # Generate sequences for each timestep in the episode
            for t in range(episode_length):
                sequence = []
                
                # Build sequence by looking back seq_len timesteps
                for i in range(seq_len):
                    seq_idx = t - (seq_len - 1 - i)  # Calculate actual timestep index
                    
                    if seq_idx >= 0:
                        # Use actual observation from episode
                        obs = np.array(obs_list[seq_idx], dtype=np.float32)
                        cycle_index = seq_idx % cycle_length
                    else:
                        # Pad with zeros for timesteps before episode start
                        obs_shape = np.array(obs_list[0]).shape
                        obs = np.zeros(obs_shape, dtype=np.float32)
                        cycle_index = 0
                    
                    # Append cycle index to observation
                    obs_with_cycle = np.append(obs, cycle_index)
                    sequence.append(obs_with_cycle)
                
                # Store completed sequence and corresponding action
                sequence = np.array(sequence, dtype=np.float32)
                batch_obs_sequences.append(sequence)
                batch_actions.append(np.array(actions_list[t], dtype=np.float32))
                
                # Yield batch when large_batch_size is reached
                if len(batch_obs_sequences) >= large_batch_size:
                    obs_tensor = torch.from_numpy(np.array(batch_obs_sequences, dtype=np.float32)).to(self.device, non_blocking=True)
                    actions_tensor = torch.from_numpy(np.array(batch_actions, dtype=np.float32)).to(self.device, non_blocking=True)
                    
                    # Split large batch into smaller training batches
                    for start_idx in range(0, len(batch_obs_sequences), batch_size):
                        end_idx = min(start_idx + batch_size, len(batch_obs_sequences))
                        yield (obs_tensor[start_idx:end_idx], 
                               actions_tensor[start_idx:end_idx], 
                               None)
                    
                    # Clear batch for next iteration
                    batch_obs_sequences = []
                    batch_actions = []
        
        # Process remaining sequences that didn't fill a complete large batch
        if len(batch_obs_sequences) > 0:
            obs_tensor = torch.from_numpy(np.array(batch_obs_sequences, dtype=np.float32)).to(self.device, non_blocking=True)
            actions_tensor = torch.from_numpy(np.array(batch_actions, dtype=np.float32)).to(self.device, non_blocking=True)
            
            for start_idx in range(0, len(batch_obs_sequences), batch_size):
                end_idx = min(start_idx + batch_size, len(batch_obs_sequences))
                yield (obs_tensor[start_idx:end_idx], 
                       actions_tensor[start_idx:end_idx], 
                       None)


def train_offline(model, dataset: OfflineDataset, config, output_dir: str, seed: int):
    """
    Execute offline training using behavioral cloning on a pre-collected dataset.
    
    This function provides a convenient interface for offline training by creating
    an OfflineTrainer instance and delegating the training process to it.
    
    Args:
        model: The RL model to train (PPO policy)
        dataset: Offline dataset containing expert demonstrations
        config: Configuration object with all training parameters
        output_dir: Directory for saving training outputs and checkpoints
        seed: Random seed for reproducibility and tracking
    """
    trainer = OfflineTrainer(model, dataset, config)
    trainer.train(output_dir=output_dir, seed=seed)
