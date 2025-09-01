"""
Data Path Management Module

This module provides centralized path management for all data files generated
during training, including episode data, training metrics, model checkpoints,
plots, and metadata. Ensures consistent file naming and directory structure.
"""

from pathlib import Path


class DataPaths:
    """
    Centralized management of all data file paths for training outputs.
    
    This class provides a single point of control for all file paths used
    during training, ensuring consistent naming conventions and proper
    directory structure creation.
    
    Attributes:
        output_dir (Path): Base output directory for all files
    """
    
    def __init__(self, output_dir):
        """
        Initialize path manager and ensure output directory exists.
        
        Args:
            output_dir: Base directory path for all training outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def episode_data(self):
        """Path to CSV file containing episode-wise training data."""
        return self.output_dir / "data_episodes.csv"
    
    @property
    def training_metrics(self):
        """Path to CSV file containing detailed training metrics."""
        return self.output_dir / "data_training_metrics.csv"
    
    @property
    def summary(self):
        """Path to CSV file containing training summary statistics."""
        return self.output_dir / "data_summary.csv"
    
    @property
    def training_plot(self):
        """Path to SVG file containing training progress visualization."""
        return self.output_dir / "training_plot.svg"
    
    @property
    def cycle_data(self):
        """Path to CSV file containing CycleNet parameter data."""
        return self.output_dir / "cyclenet_parameter_data.csv"

    @property
    def cycle_plot(self):
        """Path to SVG file containing CycleNet cycle pattern visualization."""
        return self.output_dir / "cyclenet_parameter_plot.svg"
    
    @property
    def model_checkpoint(self):
        """Path to ZIP file containing saved model checkpoint."""
        return self.output_dir / "model.zip"
    
    @property
    def metadata(self):
        """Path to JSON file containing run metadata and configuration."""
        return self.output_dir / "metadata.json"