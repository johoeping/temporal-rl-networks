"""
Centralized Logging Configuration Module

This module provides a unified logging system for the entire CycleNet RL project
with custom log levels, colored output formatting, and specialized logging methods
for different components like training, cycle analysis, and data collection.
"""

import logging


# Global flag to ensure logging is configured only once
_logging_configured = False

class CustomFormatter(logging.Formatter):
    """
    Custom formatter that provides colored output for different log levels.
    
    This formatter uses ANSI color codes to provide visual distinction between
    different types of log messages, making it easier to identify important
    information during training and development.
    """

    # Standard log level colors
    cyan = "\033[0;30m"      # DEBUG
    green = "\033[0;30m"     # INFO
    yellow = "\033[0;33m"    # WARNING
    red = "\033[0;31m"       # ERROR
    blue = "\033[1;34m"      # CRITICAL
    
    # Custom log level colors for specialized logging
    purple = "\033[1;35m"    # IMPORTANT (level 25)
    orange = "\033[0;36m"    # TRAINING (level 35)
    pink = "\033[0;34m"      # CYCLE ANALYSIS (level 45)
    bright_cyan = "\033[0;35m"   # DATA COLLECTION (level 65)
    
    reset = "\033[0m"
    format = "%(asctime)s - %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: cyan + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: blue + format + reset,
        25: purple + format + reset,      # IMPORTANT
        35: orange + format + reset,      # TRAINING
        45: pink + format + reset,        # CYCLE ANALYSIS
        65: bright_cyan + format + reset,  # DATA COLLECTION
    }

    def format(self, record):
        """
        Format log record with appropriate color scheme.
        
        Args:
            record: LogRecord instance to format
            
        Returns:
            str: Formatted log message with color codes
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def get_logger(name=None, level=logging.INFO):
    """
    Set up logging configuration and return a configured logger instance.
    
    This function ensures logging is configured only once globally and provides
    specialized logging methods for different project components. The logger
    includes custom log levels for training, cycle analysis, and data collection.
    
    Args:
        name: Logger name (defaults to caller's module name if None)
        level: Base logging level (default: logging.INFO)
    
    Returns:
        logging.Logger: Configured logger with custom methods and formatting
    """
    global _logging_configured
    
    # Configure logging system only once
    if not _logging_configured:
        # Register custom log levels with descriptive names
        logging.addLevelName(25, 'INFO')
        logging.addLevelName(35, 'TRAINING')
        logging.addLevelName(45, 'CYCLE ANALYSIS')
        logging.addLevelName(65, 'DATA COLLECTION')
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Remove existing handlers to prevent duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create and configure console handler with custom formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = CustomFormatter()
        console_handler.setFormatter(formatter)
        
        # Attach handler to root logger
        root_logger.addHandler(console_handler)
        
        _logging_configured = True
    
    # Create logger for specified name or use caller's module
    if name is None:
        name = __name__
    
    logger = logging.getLogger(name)
    
    # Add specialized logging methods to logger instance
    def important(message, *args, **kwargs):
        """Log important information messages (level 25)."""
        logger.log(25, message, *args, **kwargs)
    
    def training(message, *args, **kwargs):
        """Log training-related messages (level 35)."""
        logger.log(35, message, *args, **kwargs)
    
    def cycle(message, *args, **kwargs):
        """Log cycle analysis messages (level 45)."""
        logger.log(45, message, *args, **kwargs)
    
    def data(message, *args, **kwargs):
        """Log data collection messages (level 65)."""
        logger.log(65, message, *args, **kwargs)
    
    # Bind specialized methods to logger instance
    logger.important = important
    logger.training = training
    logger.cycle = cycle
    logger.data = data
    
    return logger
