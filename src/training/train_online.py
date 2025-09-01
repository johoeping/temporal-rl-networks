"""
Online RL Training Module

This module implements online training for RL models,
where the agent learns by interacting directly with the environment.
Includes progress tracking, data logging, and specialized CycleNet support.
"""

from src.model import NetworkType
from src.utils.paths import DataPaths
from src.callbacks.data_logger import DataLoggerCallback
from src.policies.cyclenet_policy import save_cycle_data_plot
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

def train_online(env, model, config, output_dir: str, seed: int):
    """
    Execute online training where the agent learns by interacting with the environment.
    
    This function implements the standard RL training loop where
    the model learns through trial and error by taking actions in the environment
    and receiving feedback through rewards. Includes comprehensive logging, progress
    tracking, and error handling with graceful cleanup.
    
    Args:
        env: Vectorized Gymnasium environment for training
        model: RL model to train (PPO or RecurrentPPO from stable-baselines3)
        config: Configuration object containing training parameters including:
            - online_total_timesteps: Number of environment steps to train for
            - rolling_length: Window size for rolling average calculations
            - enable_live_plot: Whether to show live training progress plots
            - network_type: Type of network architecture being used
        output_dir: Directory path for saving training outputs and results
        seed: Random seed for reproducible training runs
        
    Raises:
        Exception: Training interruption (handled gracefully with progress saving)
    """
    # Initialize environment with reproducible seed
    env.seed(seed)
    total_timesteps = config.online_total_timesteps
    rolling_length = config.rolling_length
    
    logger.training(f"Starting online training for {total_timesteps} timesteps...")
    
    # Set up data logging and progress tracking
    data_logger = DataLoggerCallback(
        rolling_length=rolling_length,
        output_dir=output_dir,
        enable_live_plot=config.enable_live_plot
    )
    callbacks = [data_logger]
    
    try:
        # Execute the training loop
        model.verbose = 0  # Suppress detailed output for cleaner logs
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # Clean up environment resources
        env.close()
        
        # Update configuration with actual training progress
        actual_timesteps = getattr(model, 'num_timesteps', total_timesteps)
        config.update_progress(timesteps=actual_timesteps)
        config.save_metadata(output_dir)

        # Save final training data and visualizations
        data_logger.save_final_data()

        # Save trained model checkpoint
        paths = DataPaths(output_dir)
        model.save(paths.model_checkpoint)

        # Generate CycleNet-specific analysis if applicable
        if config.network_type == NetworkType.CYCLENET:
            try:
                save_cycle_data_plot(model.policy.features_extractor.model.cycleQueue.data, output_dir)
            except Exception as e:
                logger.warning(f"Could not save CycleNet cycle data plot: {e}")

        logger.training(f"Online training completed. Total timesteps: {actual_timesteps}.")

    except Exception as e:
        # Graceful handling of training interruptions
        logger.warning(f"Training interrupted: {e} Saving current progress...")
        
        # Save progress even if training was interrupted
        actual_timesteps = getattr(model, 'num_timesteps', 0)
        config.update_progress(timesteps=actual_timesteps)
        config.save_metadata(output_dir)

        # Save available training data
        data_logger.save_final_data()

        # Save model checkpoint
        paths = DataPaths(output_dir)
        model.save(paths.model_checkpoint)

        logger.training(f"Progress saved at {actual_timesteps} timesteps.")
