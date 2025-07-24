"""
LEGACY TRADING BOT WRAPPER - SIMPLIFIED
=======================================

Legacy wrapper that delegates to the new SimplifiedTradingBot.
Maintains backward compatibility while eliminating complexity.
"""

import asyncio
from typing import Dict, Any
from worker_ant_v1.trading.simplified_trading_bot import SimplifiedTradingBot, SimplifiedConfig
from worker_ant_v1.utils.logger import get_logger


class HyperIntelligentTradingSwarm:
    """Legacy wrapper for backward compatibility - delegates to SimplifiedTradingBot"""
    
    def __init__(self, config_file: str = None, initial_capital: float = 300.0):
        """Initialize legacy wrapper."""
        self.logger = get_logger("LegacyTradingSwarm")
        self.initial_capital = initial_capital
        
        # Convert to SOL (assuming $200/SOL)
        initial_capital_sol = initial_capital / 200.0
        
        # Create simplified config
        config = SimplifiedConfig(initial_capital_sol=initial_capital_sol)
        
        # Create simplified bot
        self.simplified_bot = SimplifiedTradingBot(config)
        
        self.logger.info(f"🔄 Legacy wrapper initialized - delegating to SimplifiedTradingBot")
    
    async def initialize_all_systems(self) -> bool:
        """Initialize systems (legacy interface)"""
        return await self.simplified_bot.initialize()
    
    async def initialize(self) -> bool:
        """Initialize (alternative legacy interface)"""
        return await self.simplified_bot.initialize()
    
    async def run(self):
        """Run the trading system (legacy interface)"""
        await self.simplified_bot.run()
    
    async def shutdown(self):
        """Shutdown (legacy interface)"""
        await self.simplified_bot.shutdown()
    
    async def emergency_shutdown(self):
        """Emergency shutdown (legacy interface)"""
        await self.simplified_bot.emergency_shutdown()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status (legacy interface)"""
        return self.simplified_bot.get_status()


# Legacy compatibility class alias
MemecoinTradingBot = HyperIntelligentTradingSwarm

