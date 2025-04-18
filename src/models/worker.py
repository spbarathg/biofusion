import os
import yaml
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger

# Import paths module
from src.core.paths import CONFIG_PATH, LOGS_DIR
from src.bindings.worker_bridge import WorkerBridge
from src.core.wallet_manager import WalletManager

class WorkerConfig:
    def __init__(self, config_path: str = None):
        # Use provided config path or default
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
            
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
            worker_config = config.get("worker", {})
            
            self.max_trades_per_hour = worker_config.get("max_trades_per_hour", 10)
            self.min_profit_threshold = worker_config.get("min_profit_threshold", 0.01)
            self.max_slippage = worker_config.get("max_slippage", 0.02)
            self.max_trade_size = worker_config.get("max_trade_size", 1.0)
            self.min_liquidity = worker_config.get("min_liquidity", 1000.0)
            self.trading_pairs = worker_config.get("trading_pairs", [])
            self.dex_preferences = worker_config.get("dex_preferences", {})
            self.target_trades_per_minute = worker_config.get("target_trades_per_minute", 1)
            self.max_concurrent_trades = worker_config.get("max_concurrent_trades", 5)

class Worker:
    def __init__(self, worker_id: str, config_path: str = None, wallet_id: str = None):
        # Use provided config path or default
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self.worker_id = worker_id
        self.wallet_id = wallet_id
        self.config = WorkerConfig(self.config_path)
        self.is_active = True
        self.trades_executed = 0
        self.total_profit = 0.0
        self.last_trade_time = None
        
        # Setup logging
        logger.add(
            LOGS_DIR / f"worker_{worker_id}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
        
        # Initialize bridge to Rust trading engine
        self.bridge = WorkerBridge(str(self.config_path))
        
        # Initialize wallet manager
        self.wallet_manager = WalletManager(str(self.config_path))
        
    async def start(self):
        """Start the worker's trading cycle"""
        logger.info(f"Worker {self.worker_id} starting...")
        
        # Make sure we have a wallet
        if not self.wallet_id:
            logger.warning(f"Worker {self.worker_id} has no wallet, creating one...")
            self.wallet_id = self.wallet_manager.create_wallet(
                f"worker_{self.worker_id}", 
                "worker"
            )
            logger.info(f"Created wallet {self.wallet_id} for worker {self.worker_id}")
        
        # Get wallet balance
        try:
            wallet_info = self.wallet_manager.wallets.get(self.wallet_id, {})
            if not wallet_info:
                logger.error(f"Wallet {self.wallet_id} not found for worker {self.worker_id}")
                return
                
            wallet_public_key = wallet_info.get('public_key')
            if not wallet_public_key:
                logger.error(f"No public key found for worker wallet {self.wallet_id}")
                return
                
            balance = await self.wallet_manager.get_balance(self.wallet_id)
            logger.info(f"Worker {self.worker_id} balance: {balance} SOL")
            
            if balance <= 0.01:
                logger.warning(f"Worker {self.worker_id} has insufficient balance: {balance} SOL")
                return
                
            # Start the worker ant in the Rust engine
            success = await self.bridge.start_worker(
                self.worker_id,
                wallet_public_key,
                balance
            )
            
            if not success:
                logger.error(f"Failed to start worker {self.worker_id} in Rust engine")
                return
                
            logger.info(f"Worker {self.worker_id} started in Rust engine with {balance} SOL")
            
            # Monitor worker status
            while self.is_active:
                try:
                    # Get status from Rust engine
                    status = await self.bridge.get_worker_status(self.worker_id)
                    
                    if not status or not status.get("is_running", False):
                        logger.warning(f"Worker {self.worker_id} is no longer running in Rust engine")
                        break
                        
                    # Update metrics
                    self.trades_executed = status.get("trades_executed", self.trades_executed)
                    self.total_profit = status.get("total_profit", self.total_profit)
                    
                    # Log status
                    logger.info(f"Worker {self.worker_id} status: {json.dumps(status)}")
                    
                    # Sleep for a while
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error monitoring worker {self.worker_id}: {str(e)}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Error starting worker {self.worker_id}: {str(e)}")
    
    async def stop(self):
        """Stop the worker"""
        logger.info(f"Worker {self.worker_id} stopping...")
        self.is_active = False
        
        # Stop the worker in the Rust engine
        try:
            success = await self.bridge.stop_worker(self.worker_id)
            
            if not success:
                logger.error(f"Failed to stop worker {self.worker_id} in Rust engine")
            else:
                logger.info(f"Worker {self.worker_id} stopped in Rust engine")
                
        except Exception as e:
            logger.error(f"Error stopping worker {self.worker_id}: {str(e)}")
    
    async def execute_trading_cycle(self):
        """
        Execute a single trading cycle
        Note: This is now handled by the Rust engine
        """
        # This method is kept for compatibility but actual trading is done in Rust
        pass
    
    async def get_market_data(self) -> Dict:
        """
        Get current market data for configured trading pairs
        Note: This is now handled by the Rust engine
        """
        # This method is kept for compatibility but market data is handled in Rust
        return {
            "timestamp": datetime.now().isoformat(),
            "pairs": {}
        }
    
    async def find_opportunities(self, market_data: Dict) -> List[Dict]:
        """
        Find trading opportunities based on market data
        Note: This is now handled by the Rust engine
        """
        # This method is kept for compatibility but opportunity finding is done in Rust
        return []
    
    def rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Rank trading opportunities by expected profit
        Note: This is now handled by the Rust engine
        """
        # This method is kept for compatibility but ranking is done in Rust
        return []
    
    async def execute_trade(self, opportunity: Dict):
        """
        Execute a trade based on the opportunity
        Note: This is now handled by the Rust engine
        """
        # This method is kept for compatibility but trade execution is done in Rust
        pass
    
    def get_metrics(self) -> Dict:
        """Get worker metrics"""
        return {
            "worker_id": self.worker_id,
            "wallet_id": self.wallet_id,
            "trades_executed": self.trades_executed,
            "total_profit": self.total_profit,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "is_active": self.is_active
        }

async def main():
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Worker Ant")
    parser.add_argument("--id", type=str, help="Worker ID", default="worker_1")
    parser.add_argument("--wallet", type=str, help="Wallet ID", default=None)
    args = parser.parse_args()
    
    worker = Worker(args.id, wallet_id=args.wallet)
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        await worker.stop()

if __name__ == "__main__":
    asyncio.run(main()) 