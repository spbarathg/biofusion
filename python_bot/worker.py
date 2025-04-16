import os
import yaml
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

class WorkerConfig:
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            worker_config = config.get("worker", {})
            
            self.max_trades_per_hour = worker_config.get("max_trades_per_hour", 10)
            self.min_profit_threshold = worker_config.get("min_profit_threshold", 0.01)
            self.max_slippage = worker_config.get("max_slippage", 0.02)
            self.max_trade_size = worker_config.get("max_trade_size", 1.0)
            self.min_liquidity = worker_config.get("min_liquidity", 1000.0)
            self.trading_pairs = worker_config.get("trading_pairs", [])
            self.dex_preferences = worker_config.get("dex_preferences", {})

class Worker:
    def __init__(self, worker_id: str, config_path: str = "config/settings.yaml"):
        self.worker_id = worker_id
        self.config = WorkerConfig(config_path)
        self.is_active = True
        self.trades_executed = 0
        self.total_profit = 0.0
        self.last_trade_time = None
        
        # Setup logging
        logger.add(
            f"logs/worker_{worker_id}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
        
    async def start(self):
        """Start the worker's trading cycle"""
        logger.info(f"Worker {self.worker_id} starting...")
        
        while self.is_active:
            try:
                await self.execute_trading_cycle()
                self.trades_executed += 1
                
                # Check if we've hit the hourly limit
                if self.trades_executed >= self.config.max_trades_per_hour:
                    logger.info("Hourly trade limit reached, waiting...")
                    await asyncio.sleep(3600)  # Wait an hour
                    self.trades_executed = 0
                
                # Sleep between cycles
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def stop(self):
        """Stop the worker"""
        logger.info(f"Worker {self.worker_id} stopping...")
        self.is_active = False
    
    async def execute_trading_cycle(self):
        """Execute a single trading cycle"""
        # Get market data
        market_data = await self.get_market_data()
        
        # Find trading opportunities
        opportunities = await self.find_opportunities(market_data)
        
        # Filter and rank opportunities
        ranked_opportunities = self.rank_opportunities(opportunities)
        
        # Execute best opportunity if found
        if ranked_opportunities:
            best_opportunity = ranked_opportunities[0]
            await self.execute_trade(best_opportunity)
    
    async def get_market_data(self) -> Dict:
        """Get current market data for configured trading pairs"""
        # This would interact with the Rust core to get market data
        # Placeholder implementation
        return {
            "timestamp": datetime.now().isoformat(),
            "pairs": {}
        }
    
    async def find_opportunities(self, market_data: Dict) -> List[Dict]:
        """Find trading opportunities based on market data"""
        opportunities = []
        
        # This would use the Rust core's pathfinder to find opportunities
        # Placeholder implementation
        return opportunities
    
    def rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Rank trading opportunities by expected profit"""
        ranked = sorted(
            opportunities,
            key=lambda x: x.get("expected_profit", 0),
            reverse=True
        )
        
        # Filter out opportunities that don't meet criteria
        filtered = [
            opp for opp in ranked
            if opp.get("expected_profit", 0) >= self.config.min_profit_threshold
            and opp.get("slippage", 1.0) <= self.config.max_slippage
            and opp.get("liquidity", 0) >= self.config.min_liquidity
        ]
        
        return filtered
    
    async def execute_trade(self, opportunity: Dict):
        """Execute a trade based on the opportunity"""
        try:
            # This would interact with the Rust core to execute the trade
            # Placeholder implementation
            logger.info(f"Executing trade: {json.dumps(opportunity)}")
            
            # Update metrics
            self.last_trade_time = datetime.now()
            self.total_profit += opportunity.get("expected_profit", 0)
            
            logger.info(f"Trade executed successfully. Profit: {opportunity.get('expected_profit', 0)}")
            
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict:
        """Get worker metrics"""
        return {
            "worker_id": self.worker_id,
            "trades_executed": self.trades_executed,
            "total_profit": self.total_profit,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "is_active": self.is_active
        }

async def main():
    # Example usage
    worker = Worker("worker_1")
    try:
        await worker.start()
    except KeyboardInterrupt:
        await worker.stop()

if __name__ == "__main__":
    asyncio.run(main()) 