import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger

from utils.config_loader import get_config

# Configure default logger
def setup_logger(log_dir: str = "logs"):
    """
    Configure the default logger settings based on configuration.
    Creates necessary log directory and sets up log handlers.
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Get logging configuration
    log_level = get_config("monitoring.log_level", "INFO")
    log_rotation = get_config("monitoring.log_rotation", "1 day")
    log_retention = get_config("monitoring.log_retention", "7 days")
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # Add file handlers for different log types
    logger.add(
        os.path.join(log_dir, "trades_{time}.log"),
        rotation=log_rotation,
        retention=log_retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level=log_level,
        filter=lambda record: "trade" in record["extra"].get("category", "")
    )
    
    logger.add(
        os.path.join(log_dir, "wallets_{time}.log"),
        rotation=log_rotation,
        retention=log_retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level=log_level,
        filter=lambda record: "wallet" in record["extra"].get("category", "")
    )
    
    logger.add(
        os.path.join(log_dir, "capital_{time}.log"),
        rotation=log_rotation,
        retention=log_retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level=log_level,
        filter=lambda record: "capital" in record["extra"].get("category", "")
    )
    
    logger.add(
        os.path.join(log_dir, "alerts_{time}.log"),
        rotation=log_rotation,
        retention=log_retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="WARNING",
        filter=lambda record: "alert" in record["extra"].get("category", "")
    )
    
    logger.add(
        os.path.join(log_dir, "errors_{time}.log"),
        rotation=log_rotation,
        retention=log_retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="ERROR",
        filter=lambda record: record["level"].name == "ERROR"
    )
    
    # Add a catch-all log file
    logger.add(
        os.path.join(log_dir, "all_{time}.log"),
        rotation=log_rotation,
        retention=log_retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG"
    )
    
    return logger

# Configure component-specific logger
def get_component_logger(component_name: str):
    """
    Get a logger configured for a specific component.
    
    Args:
        component_name: Name of the component (queen, princess, worker, etc.)
        
    Returns:
        Configured logger instance
    """
    log_level = get_config("monitoring.log_level", "INFO")
    log_rotation = get_config("monitoring.log_rotation", "1 day")
    log_retention = get_config("monitoring.log_retention", "7 days")
    
    # Add component-specific file handler
    component_logger = logger.bind(component=component_name)
    
    # Component log file
    log_file = os.path.join("logs", f"{component_name}.log")
    
    # Return the configured logger
    return component_logger

class AntLogger:
    """
    Central logging system for the entire bot.
    Logs trades, wallet status, capital events, and alerts.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        
        # Ensure logger is configured
        setup_logger(log_dir)
    
    def log_trade(self, worker_id: str, trade_data: Dict[str, Any]):
        """
        Log a trade
        
        Args:
            worker_id: ID of the worker that executed the trade
            trade_data: Trade data including tokens, amounts, profit, etc.
        """
        trade_data["worker_id"] = worker_id
        trade_data["timestamp"] = datetime.now().isoformat()
        
        logger.info(
            f"Trade executed by Worker {worker_id}: {json.dumps(trade_data)}",
            category="trade"
        )
        
        # Save to trades JSON file
        self._append_to_json_file("trades.json", trade_data)
    
    def log_wallet(self, wallet_type: str, wallet_address: str, event: str, data: Dict[str, Any]):
        """
        Log a wallet event
        
        Args:
            wallet_type: Type of wallet (queen, princess, worker)
            wallet_address: Wallet address
            event: Event type (created, funded, transitioned, etc.)
            data: Additional data
        """
        log_data = {
            "wallet_type": wallet_type,
            "wallet_address": wallet_address,
            "event": event,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        
        logger.info(
            f"Wallet event: {wallet_type} {wallet_address} - {event}",
            category="wallet"
        )
        
        # Save to wallets JSON file
        self._append_to_json_file("wallets.json", log_data)
    
    def log_capital(self, event: str, data: Dict[str, Any]):
        """
        Log a capital event
        
        Args:
            event: Event type (compounded, saved, withdrawn, etc.)
            data: Event data
        """
        log_data = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        
        logger.info(
            f"Capital event: {event}",
            category="capital"
        )
        
        # Save to capital JSON file
        self._append_to_json_file("capital.json", log_data)
    
    def log_alert(self, alert_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """
        Log an alert
        
        Args:
            alert_type: Type of alert (warning, error, critical)
            message: Alert message
            data: Additional data
        """
        log_data = {
            "alert_type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            **(data or {})
        }
        
        if alert_type == "warning":
            logger.warning(message, category="alert")
        elif alert_type == "error":
            logger.error(message, category="alert")
        elif alert_type == "critical":
            logger.critical(message, category="alert")
        else:
            logger.info(message, category="alert")
        
        # Save to alerts JSON file
        self._append_to_json_file("alerts.json", log_data)
    
    def _append_to_json_file(self, filename: str, data: Dict[str, Any]):
        """
        Append data to a JSON file
        
        Args:
            filename: Name of the file
            data: Data to append
        """
        filepath = os.path.join(self.log_dir, filename)
        
        # Create file with empty list if it doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                json.dump([], f)
        
        # Read existing data
        with open(filepath, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
        
        # Append new data
        existing_data.append(data)
        
        # Write back to file
        with open(filepath, "w") as f:
            json.dump(existing_data, f, indent=2)

# Initialize the default logger
setup_logger()

# Create a singleton instance
ant_logger = AntLogger()

# Convenience functions
def log_trade(worker_id: str, trade_data: Dict[str, Any]):
    ant_logger.log_trade(worker_id, trade_data)

def log_wallet(wallet_type: str, wallet_address: str, event: str, data: Dict[str, Any]):
    ant_logger.log_wallet(wallet_type, wallet_address, event, data)

def log_capital(event: str, data: Dict[str, Any]):
    ant_logger.log_capital(event, data)

def log_alert(alert_type: str, message: str, data: Optional[Dict[str, Any]] = None):
    ant_logger.log_alert(alert_type, message, data) 