import logging
import os
import sys
from utils.config import LoggingConfig

def setup_logger(name: str, config: LoggingConfig) -> logging.Logger:
    """Sets up a structured logger using parameters from configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    
    # Avoid duplicate handler definitions if called multiple times
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(config.format)
    
    # Console output handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File output handler
    if config.file_path:
        log_dir = os.path.dirname(config.file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(config.file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
