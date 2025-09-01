"""
Utility Functions for CycleNet RL Networks

This module provides common utility functions used throughout the project,
including GPU checking, file path management, and reproducibility utilities.
"""

import torch
import random
import numpy as np
from typing import Optional


def set_seed(seed: Optional[int]) -> None:
    """
    Set random seeds for reproducibility across all libraries and frameworks.
    
    Args:
        seed: Random seed value for reproducible results. If None, no seeding is performed.
    """
    if seed is None:
        return
        
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False