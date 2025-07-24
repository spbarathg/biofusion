"""
Smart Ape Mode - Trading Module
===============================

Trading execution components including:
- Market scanning and opportunity detection
- Unified trading execution (replacing redundant order modules)
- Trade execution coordination
- Position management
"""

from worker_ant_v1.trading.market_scanner import RealMarketScanner, ScanResult
from worker_ant_v1.utils.market_data_fetcher import MarketOpportunity
from worker_ant_v1.core.unified_trading_engine import UnifiedTradingEngine, TradeOrder, TradeResult

__all__ = [
    'RealMarketScanner',
    'MarketOpportunity',
    'ScanResult', 
    'UnifiedTradingEngine',
    'TradeOrder',
    'TradeResult'
]
