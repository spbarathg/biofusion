import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        self.fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        super().__init__(self.fmt)

    def format(self, record):
        if not self.use_colors:
            return super().format(record)

        # Save original values
        orig_levelname = record.levelname
        orig_msg = record.msg

        # Apply colors
        if record.levelno == logging.DEBUG:
            record.levelname = f"{self.grey}{record.levelname}{self.reset}"
            record.msg = f"{self.grey}{record.msg}{self.reset}"
        elif record.levelno == logging.INFO:
            record.levelname = f"{self.blue}{record.levelname}{self.reset}"
            record.msg = f"{self.blue}{record.msg}{self.reset}"
        elif record.levelno == logging.WARNING:
            record.levelname = f"{self.yellow}{record.levelname}{self.reset}"
            record.msg = f"{self.yellow}{record.msg}{self.reset}"
        elif record.levelno == logging.ERROR:
            record.levelname = f"{self.red}{record.levelname}{self.reset}"
            record.msg = f"{self.red}{record.msg}{self.reset}"
        elif record.levelno == logging.CRITICAL:
            record.levelname = f"{self.bold_red}{record.levelname}{self.reset}"
            record.msg = f"{self.bold_red}{record.msg}{self.reset}"

        # Format with colors
        result = super().format(record)

        # Restore original values
        record.levelname = orig_levelname
        record.msg = orig_msg

        return result

def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with both file and console handlers
    
    Args:
        name: Name of the logger
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger

    # Create logs directory if it doesn't exist
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Create handlers
    # File handler with rotation
    log_file = logs_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter(
        use_colors=sys.stdout.isatty()  # Only use colors if output is a terminal
    ))

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance. If no name is provided, returns the root logger.
    
    Args:
        name: Optional name for the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    if name is None:
        return logging.getLogger()
    return setup_logger(name, os.getenv('LOG_LEVEL', 'INFO'))

# Set up root logger
setup_logger('root', os.getenv('LOG_LEVEL', 'INFO')) 