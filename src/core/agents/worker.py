import os
import yaml
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from src.utils.logging.logger import setup_logging

class WorkerConfig:
    """Configuration for worker agents"""
    def __init__(self, config_path: str = "config/settings.yaml"):
        # If not absolute path, convert to absolute path
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), config_path)
            
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            worker_config = config.get("worker", {})
            
            # Load worker configuration with defaults
            self.max_trades_per_hour = worker_config.get("max_trades_per_hour", 10)
            self.min_profit_threshold = worker_config.get("min_profit_threshold", 0.01)
            self.max_slippage = worker_config.get("max_slippage", 0.02)
            self.max_trade_size = worker_config.get("max_trade_size", 1.0)
            self.min_liquidity = worker_config.get("min_liquidity", 1000.0)
            self.max_hold_time = worker_config.get("max_hold_time", 30)

class Worker:
    """
    Worker agent that executes trades.
    Handles all aspects of trading execution and reporting status back to the Queen.
    """
    def __init__(self, worker_id: str, config_path: str = "config/settings.yaml"):
        # Convert config_path to absolute path if needed
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), config_path)
            
        self.worker_id = worker_id
        self.config = WorkerConfig(config_path)
        self.is_active = False
        self.trades_executed = 0
        self.total_profit = 0.0
        self.last_trade_time = None
        self.wallet_address = None
        self.capital = 0.0
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for the worker agent"""
        setup_logging(f"worker_{self.worker_id}", f"worker_{self.worker_id}.log")
        
    async def initialize(self, wallet_address: str, capital: float) -> bool:
        """Initialize the worker with a wallet and capital"""
        logger.info(f"Initializing worker {self.worker_id} with {capital} SOL")
        self.wallet_address = wallet_address
        self.capital = capital
        self.is_active = True
        return True
        
    async def start(self):
        """Start the worker's trading cycle"""
        if not self.is_active:
            logger.warning(f"Cannot start worker {self.worker_id} - not initialized")
            return False
            
        logger.info(f"Worker {self.worker_id} starting...")
        
        try:
            # Start trading loop in a separate task
            asyncio.create_task(self._trading_loop())
            return True
        except Exception as e:
            logger.error(f"Failed to start worker {self.worker_id}: {str(e)}")
            return False
    
    async def stop(self):
        """Stop the worker"""
        logger.info(f"Worker {self.worker_id} stopping...")
        self.is_active = False
        return True
    
    async def _trading_loop(self):
        """Main trading loop that runs until the worker is stopped"""
        while self.is_active:
            try:
                await self.execute_trading_cycle()
                
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
    
    async def execute_trading_cycle(self):
        """Execute a single trading cycle"""
        # 1. Get market data
        market_data = await self.get_market_data()
        
        # 2. Find trading opportunities
        opportunities = await self.find_opportunities(market_data)
        
        # 3. Filter and rank opportunities
        ranked_opportunities = self.rank_opportunities(opportunities)
        
        # 4. Execute best opportunity if found
        if ranked_opportunities:
            best_opportunity = ranked_opportunities[0]
            success = await self.execute_trade(best_opportunity)
            
            if success:
                self.trades_executed += 1
                logger.info(f"Successfully executed trade, total trades this hour: {self.trades_executed}")
    
    async def get_market_data(self) -> Dict:
        """Get current market data for trading"""
        # This would interact with the Rust core to get market data
        # Simple placeholder implementation
        return {
            "timestamp": datetime.now().isoformat(),
            "pairs": {}
        }
    
    async def find_opportunities(self, market_data: Dict) -> List[Dict]:
        """Find trading opportunities based on market data"""
        # This would use the Rust core's pathfinder to find opportunities
        # Simple placeholder implementation
        opportunities = []
        
        # Mock opportunity for demonstration
        if self.trades_executed < self.config.max_trades_per_hour:
            opportunities.append({
                "pair": "SOL/USDC",
                "expected_profit": 0.02,  # 2% profit
                "slippage": 0.01,
                "liquidity": 5000.0,
                "trade_size": min(self.capital * 0.1, self.config.max_trade_size),
                "execution_path": ["jupiter"]
            })
        
        return opportunities
    
    def rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Rank trading opportunities by expected profit"""
        # Sort by expected profit (descending)
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
    
    async def execute_trade(self, opportunity: Dict) -> bool:
        """Execute a trade based on the opportunity"""
        try:
            # This would interact with the Rust core to execute the trade
            # Simple placeholder implementation
            logger.info(f"Executing trade: {json.dumps(opportunity)}")
            
            # Simulate trade execution with a delay
            await asyncio.sleep(0.5)
            
            # Calculate profit (would come from actual trade result)
            profit = opportunity.get("trade_size") * opportunity.get("expected_profit", 0)
            
            # Update metrics
            self.last_trade_time = datetime.now()
            self.total_profit += profit
            
            logger.info(f"Trade executed successfully. Profit: {profit:.6f} SOL")
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")
            return False
    
    def get_status(self) -> Dict:
        """Get worker status and metrics"""
        return {
            "worker_id": self.worker_id,
            "wallet_address": self.wallet_address,
            "capital": self.capital,
            "trades_executed": self.trades_executed,
            "total_profit": self.total_profit,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "is_running": self.is_active
        }

async def main():
    # Example usage
    worker = Worker("worker_1")
    await worker.initialize("ExampleWalletAddress123", 1.0)
    
    try:
        await worker.start()
        # Keep running for demo purposes
        await asyncio.sleep(60)
    except KeyboardInterrupt:
        await worker.stop()

if __name__ == "__main__":
    asyncio.run(main()) 