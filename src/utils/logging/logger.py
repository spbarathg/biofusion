import os
import sys
from loguru import logger

def setup_logging(component_name, log_file=None):
    """
    Set up comprehensive logging for easier debugging and implementation.
    
    Args:
        component_name: Name of the component being logged
        log_file: Optional specific log file name
        
    Returns:
        Configured logger instance
    """
    # Ensure log directory exists
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add stdout handler with detailed formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"  # Default level for stdout
    )
    
    # Add file handler for the specific component
    if log_file:
        logger.add(
            os.path.join(log_dir, log_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
            level="DEBUG"  # More detailed for file logs
        )
    
    # Add a general debug log file with all logs from all components
    logger.add(
        os.path.join(log_dir, "debug.log"),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="50 MB",
        retention="30 days",
        compression="zip",
        level="DEBUG"
    )
    
    # Add a separate error log file for critical issues
    logger.add(
        os.path.join(log_dir, "error.log"),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="60 days",
        compression="zip",
        level="ERROR"  # Only capture ERROR and CRITICAL
    )
    
    logger.info(f"Logging initialized for {component_name}")
    return logger

if __name__ == "__main__":
    # Test the logger
    test_logger = setup_logging("logger_test", "test.log")
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message") 