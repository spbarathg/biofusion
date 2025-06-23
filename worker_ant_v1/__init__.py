"""
Worker Ant V1 - Simplified MVP Trading Bot
==========================================

A focused, profitable memecoin trading bot for Solana.
Strips away all complexity - just finds new tokens and trades them profitably.

Core Components:
- scanner.py: Detects new token listings
- buyer.py: Fast trade execution  
- seller.py: Exit strategy logic
- logger.py: Performance tracking
- config.py: Configuration management
"""

__version__ = "1.0.0"
__all__ = ["scanner", "buyer", "seller", "logger", "config"] 