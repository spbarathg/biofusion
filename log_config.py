import sys
from loguru import logger

def setup_logging(component_name, log_file=None):
    """Set up comprehensive logging for easier debugging and implementation."""
    
    # Remove default handler
    logger.remove()
    
    # Add stdout handler with detailed formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG"  # Capture all log levels to stdout
    )
    
    # Add file handler for the specific component
    if log_file:
        logger.add(
            f"logs/{log_file}",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
            level="DEBUG"  # Capture all log levels to file
        )
    
    # Add a general debug log file with all logs from all components
    logger.add(
        "logs/debug.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="50 MB",
        retention="30 days",
        compression="zip",
        level="DEBUG"
    )
    
    # Add a separate error log file for critical issues
    logger.add(
        "logs/error.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="60 days",
        compression="zip",
        level="ERROR"  # Only capture ERROR and CRITICAL
    )
    
    logger.info(f"Logging initialized for {component_name}")
    return logger 