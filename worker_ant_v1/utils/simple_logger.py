"""
Smart Ape Mode - Simplified Logger
==================================

Lightweight logging without external dependencies.
For production use, replace with the full logger.py after installing dependencies.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Setup a basic logger without database dependencies"""
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if logs directory exists)
    logs_dir = Path("logs")
    if logs_dir.exists():
        file_handler = logging.FileHandler(
            logs_dir / f"{name.lower()}.log",
            mode='a',
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


class TradingLogger:
    """Simplified trading logger class"""
    
    def __init__(self, name: str = "TradingLogger"):
        self.logger = setup_logger(name)
        self.trade_count = 0
        self.session_start = datetime.now()
    
    def log_trade(self, trade_type: str, token_address: str, amount: float, price: float, success: bool = True):
        """Log a trade execution"""
        self.trade_count += 1
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.logger.info(
            f"TRADE #{self.trade_count} | {trade_type.upper()} | "
            f"Token: {token_address[:8]}... | "
            f"Amount: {amount:.4f} | Price: ${price:.6f} | {status}"
        )
    
    def log_signal(self, signal_type: str, token_address: str, strength: float, confidence: float):
        """Log a trading signal"""
        self.logger.info(
            f"SIGNAL | {signal_type.upper()} | "
            f"Token: {token_address[:8]}... | "
            f"Strength: {strength:.2f} | Confidence: {confidence:.2f}"
        )
    
    def log_performance(self, capital: float, profit: float, roi: float, trades: int):
        """Log performance metrics"""
        self.logger.info(
            f"PERFORMANCE | Capital: ${capital:.2f} | "
            f"Profit: ${profit:.2f} | ROI: {roi:.1f}% | "
            f"Trades: {trades}"
        )
    
    def log_error(self, operation: str, error: str, context: Optional[str] = None):
        """Log an error"""
        context_str = f" | Context: {context}" if context else ""
        self.logger.error(f"ERROR | {operation} | {error}{context_str}")
    
    def log_warning(self, message: str, details: Optional[str] = None):
        """Log a warning"""
        details_str = f" | {details}" if details else ""
        self.logger.warning(f"WARNING | {message}{details_str}")
    
    def log_system_event(self, event: str, details: Optional[str] = None):
        """Log a system event"""
        details_str = f" | {details}" if details else ""
        self.logger.info(f"SYSTEM | {event}{details_str}")


# Global logger instances
_loggers = {}


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger instance"""
    if name not in _loggers:
        _loggers[name] = setup_logger(name)
    return _loggers[name]


def get_trading_logger(name: str = "TradingLogger") -> TradingLogger:
    """Get or create a trading logger instance"""
    key = f"trading_{name}"
    if key not in _loggers:
        _loggers[key] = TradingLogger(name)
    return _loggers[key]


def set_log_level(level: str):
    """Set log level for all loggers"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    for logger_obj in _loggers.values():
        if isinstance(logger_obj, logging.Logger):
            logger_obj.setLevel(log_level)
            for handler in logger_obj.handlers:
                handler.setLevel(log_level)
        elif hasattr(logger_obj, 'logger'):
            logger_obj.logger.setLevel(log_level)
            for handler in logger_obj.logger.handlers:
                handler.setLevel(log_level)


def create_logs_directory():
    """Create logs directory if it doesn't exist"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


# Initialize logs directory on import
create_logs_directory()

# Alias for backward compatibility
trading_logger = get_trading_logger("TradingLogger") 