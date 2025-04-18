import os
import yaml
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from src.utils.logging.logger import setup_logging
from src.core.wallet.wallet_manager import WalletManager
from src.core.capital.capital_manager import CapitalManager
from src.bindings.worker_bridge import WorkerBridge

@dataclass
class ColonyConfig:
    """Configuration for the colony structure"""
    initial_capital: float
    min_princess_capital: float
    max_workers: int
    min_workers: int
    worker_timeout: int
    health_check_interval: int

class Queen:
    """
    Queen agent that manages the entire colony of trading agents.
    Responsible for capital allocation, worker spawning and monitoring.
    """
    def __init__(self, config_path: str = "config/settings.yaml"):
        # Convert config_path to absolute path if needed
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), config_path)
        
        self.config_path = config_path
        self.config = self._load_config(self.config_path)
        
        # Load savings ratio from config
        with open(self.config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            self.savings_ratio = full_config.get('capital', {}).get('savings_ratio', 0.90)
            
        self.colony_state = {
            "queen_wallet": None,
            "princess_wallets": [],
            "worker_wallets": [],
            "savings_wallet": None,
            "total_capital": 0.0,
            "total_profits": 0.0,
        }
        
        # Initialize components
        self._setup_logging()
        self.wallet_manager = WalletManager(config_path)
        self.capital_manager = CapitalManager(config_path)
        self.worker_bridge = WorkerBridge(config_path)
        
        # Worker state tracking
        self.active_workers = {}

    def _setup_logging(self):
        """Set up logging for the Queen agent"""
        setup_logging("queen", "queen.log")

    def _load_config(self, config_path: str) -> ColonyConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        colony_config = config.get('colony', {})
        return ColonyConfig(
            initial_capital=colony_config.get('initial_capital', 10.0),
            min_princess_capital=colony_config.get('min_princess_capital', 5.0),
            max_workers=colony_config.get('max_workers', 10),
            min_workers=colony_config.get('min_workers', 3),
            worker_timeout=colony_config.get('worker_timeout', 300),
            health_check_interval=colony_config.get('health_check_interval', 60)
        )

    async def initialize_colony(self, initial_capital: float) -> None:
        """Initialize the colony with founding capital."""
        logger.info(f"Initializing colony with {initial_capital} SOL")
        
        # Create queen wallet if it doesn't exist yet
        if not self.colony_state["queen_wallet"]:
            self.colony_state["queen_wallet"] = await self._create_wallet("queen")
            logger.info(f"Created queen wallet: {self.colony_state['queen_wallet']}")
        
        # Create savings wallet if it doesn't exist yet
        if not self.colony_state["savings_wallet"]:
            self.colony_state["savings_wallet"] = await self._create_wallet("savings")
            logger.info(f"Created savings wallet: {self.colony_state['savings_wallet']}")
        
        # Fund queen wallet with initial capital
        # In a real scenario, this would be done by transferring from an external wallet
        logger.info(f"For initial funding, please transfer {initial_capital} SOL to:")
        logger.info(f"Queen wallet public key: {self.wallet_manager.get_public_key(self.colony_state['queen_wallet'])}")
        
        self.colony_state["total_capital"] = initial_capital
        logger.info("Colony initialized successfully")
        
        # Start the health check monitoring
        asyncio.create_task(self._health_check_loop())

    async def manage_workers(self) -> None:
        """Manage worker deployment and scaling."""
        current_workers = len(self.colony_state["worker_wallets"])
        
        # Check if we need to spawn more workers
        if current_workers < self.config.min_workers:
            logger.info(f"Colony has {current_workers} workers, below minimum {self.config.min_workers}. Spawning more...")
            await self._spawn_workers(self.config.min_workers - current_workers)
        
        # Check if we can spawn more workers
        elif current_workers < self.config.max_workers:
            # Check if we have enough capital to spawn new workers
            available_capital = await self._get_available_capital()
            capital_per_worker = available_capital / (self.config.max_workers - current_workers)
            
            # Only spawn workers if we have enough capital
            if capital_per_worker >= 0.1:  # Minimum 0.1 SOL per worker
                additional_workers = min(
                    self.config.max_workers - current_workers,
                    int(available_capital / 0.1)
                )
                
                if additional_workers > 0:
                    logger.info(f"Spawning {additional_workers} additional workers")
                    await self._spawn_workers(additional_workers)
        
        # Check if existing workers are healthy
        await self._check_worker_health()

    async def _spawn_workers(self, count: int) -> None:
        """Spawn a specified number of worker wallets."""
        available_capital = await self._get_available_capital()
        
        if available_capital < 0.1 * count:
            logger.warning(f"Not enough capital to spawn {count} workers. Need {0.1 * count} SOL, have {available_capital} SOL")
            return
        
        capital_per_worker = min(1.0, available_capital / count)  # Maximum 1 SOL per worker
        
        for i in range(count):
            worker_wallet = await self._create_wallet("worker")
            self.colony_state["worker_wallets"].append(worker_wallet)
            
            # Allocate capital to worker
            await self._transfer_capital(
                self.colony_state["queen_wallet"],
                worker_wallet,
                capital_per_worker
            )
            
            logger.info(f"Spawned new worker wallet: {worker_wallet} with {capital_per_worker} SOL")
            
            # Start the worker in the Rust engine
            wallet_info = self.wallet_manager.get_wallet_info(worker_wallet)
            worker_id = f"worker_{len(self.colony_state['worker_wallets'])}"
            
            success = await self.worker_bridge.start_worker(
                worker_id,
                wallet_info['public_key'],
                capital_per_worker
            )
            
            if success:
                logger.info(f"Started worker {worker_id} in Rust engine")
                # Track active worker
                self.active_workers[worker_id] = {
                    "wallet_id": worker_wallet,
                    "start_time": asyncio.get_event_loop().time(),
                    "last_check": asyncio.get_event_loop().time()
                }
            else:
                logger.error(f"Failed to start worker {worker_id} in Rust engine")

    async def _check_worker_health(self) -> None:
        """Check the health of all active workers."""
        for worker_id, info in list(self.active_workers.items()):
            try:
                status = await self.worker_bridge.get_worker_status(worker_id)
                
                if not status or not status.get("is_running", False):
                    logger.warning(f"Worker {worker_id} is not running. Removing from active workers.")
                    await self.worker_bridge.stop_worker(worker_id)
                    del self.active_workers[worker_id]
                else:
                    # Update last check time
                    self.active_workers[worker_id]["last_check"] = asyncio.get_event_loop().time()
                    
                    # Check profit and update metrics
                    if "total_profit" in status:
                        profit = float(status["total_profit"])
                        if profit > 0:
                            logger.info(f"Worker {worker_id} has earned {profit} SOL in profit")
                            
                            # If profit is significant, collect it
                            if profit >= 0.05:  # Minimum 0.05 SOL to collect
                                await self._collect_worker_profits(worker_id, info["wallet_id"])
            
            except Exception as e:
                logger.error(f"Error checking health of worker {worker_id}: {str(e)}")

    async def _health_check_loop(self) -> None:
        """Run a continuous health check loop."""
        while True:
            try:
                await self.manage_workers()
                await self.collect_profits()
                
                # Sleep for the configured interval
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute on error

    async def collect_profits(self) -> None:
        """Collect and manage profits from workers."""
        total_profits = 0.0
        
        # Collect from workers
        for worker_id, info in self.active_workers.items():
            wallet_id = info["wallet_id"]
            profits = await self._collect_wallet_profits(wallet_id)
            total_profits += profits
        
        if total_profits > 0:
            logger.info(f"Collected total profits: {total_profits} SOL")
            
            # Split profits between savings and reinvestment
            savings_amount = total_profits * self.savings_ratio
            
            # Send to savings
            await self._transfer_capital(
                self.colony_state["queen_wallet"],
                self.colony_state["savings_wallet"],
                savings_amount
            )
            
            # Update metrics
            self.colony_state["total_profits"] += total_profits
            logger.info(f"Added {savings_amount} SOL to savings")

    async def _collect_worker_profits(self, worker_id: str, wallet_id: str) -> float:
        """Collect profits from a specific worker and transfer to queen."""
        try:
            # Get worker wallet balance
            worker_balance = await self._get_wallet_balance(wallet_id)
            
            # Get worker metrics from bridge
            status = await self.worker_bridge.get_worker_status(worker_id)
            reported_profit = float(status.get("total_profit", 0))
            
            if reported_profit > 0 and worker_balance > 0.1:  # Keep a minimum balance in worker
                # Calculate amount to transfer
                transfer_amount = min(reported_profit, worker_balance - 0.1)
                
                if transfer_amount > 0:
                    # Transfer to queen
                    await self._transfer_capital(wallet_id, self.colony_state["queen_wallet"], transfer_amount)
                    logger.info(f"Collected {transfer_amount} SOL profit from worker {worker_id}")
                    
                    # Reset profit counter in bridge
                    await self.worker_bridge.update_worker_metrics(worker_id, 0, 0)
                    
                    return transfer_amount
            
            return 0.0
        except Exception as e:
            logger.error(f"Error collecting profits from worker {worker_id}: {str(e)}")
            return 0.0

    async def _create_wallet(self, wallet_type: str) -> str:
        """Create a new wallet of specified type."""
        wallet_name = f"{wallet_type}_{int(self.colony_state['total_profits'])}"
        wallet_id = await self.wallet_manager.create_wallet(wallet_name, wallet_type)
        return wallet_id

    async def _get_wallet_balance(self, wallet: str) -> float:
        """Get balance of specified wallet."""
        return await self.wallet_manager.get_balance(wallet)

    async def _transfer_capital(self, from_wallet: str, to_wallet: str, amount: float) -> None:
        """Transfer capital between wallets."""
        await self.wallet_manager.transfer_sol(from_wallet, to_wallet, amount)

    async def _get_available_capital(self) -> float:
        """Get total available capital for new workers."""
        queen_balance = await self._get_wallet_balance(self.colony_state["queen_wallet"])
        return queen_balance * 0.5  # Use 50% of queen's balance for worker allocation

    async def _collect_wallet_profits(self, wallet: str) -> float:
        """Collect profits from a wallet based on trading activity."""
        try:
            # In a real implementation, this would analyze the wallet's trading history
            # and determine profits to collect
            # For now, we'll check if balance exceeds a threshold and collect the excess
            balance = await self._get_wallet_balance(wallet)
            
            # Keep a minimum balance in the wallet
            min_balance = 0.1
            
            if balance > min_balance:
                profit = balance - min_balance
                
                if profit > 0.01:  # Only collect if profit is significant
                    # Transfer to queen
                    await self._transfer_capital(wallet, self.colony_state["queen_wallet"], profit)
                    logger.info(f"Collected {profit} SOL from wallet {wallet}")
                    return profit
            
            return 0.0
        except Exception as e:
            logger.error(f"Error collecting profits from wallet {wallet}: {str(e)}")
            return 0.0

    async def get_colony_state(self) -> Dict:
        """Get current state of the colony including wallet balances."""
        state = {
            "total_capital": self.colony_state["total_capital"],
            "total_profits": self.colony_state["total_profits"],
            "wallets": {},
            "workers": {}
        }
        
        # Queen wallet
        if self.colony_state["queen_wallet"]:
            state["wallets"]["queen"] = {
                "id": self.colony_state["queen_wallet"],
                "balance": await self._get_wallet_balance(self.colony_state["queen_wallet"])
            }
        
        # Savings wallet
        if self.colony_state["savings_wallet"]:
            state["wallets"]["savings"] = {
                "id": self.colony_state["savings_wallet"],
                "balance": await self._get_wallet_balance(self.colony_state["savings_wallet"])
            }
        
        # Worker wallets & status
        for worker_id, info in self.active_workers.items():
            wallet_id = info["wallet_id"]
            status = await self.worker_bridge.get_worker_status(worker_id)
            
            state["workers"][worker_id] = {
                "wallet_id": wallet_id,
                "balance": await self._get_wallet_balance(wallet_id),
                "status": status
            }
        
        return state

    async def backup_wallets(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of all wallets."""
        return await self.wallet_manager.create_backup(backup_path)

    async def stop_colony(self) -> None:
        """Stop all workers and shut down the colony."""
        logger.info("Stopping colony...")
        
        # Stop all workers
        for worker_id in list(self.active_workers.keys()):
            try:
                logger.info(f"Stopping worker {worker_id}")
                await self.worker_bridge.stop_worker(worker_id)
            except Exception as e:
                logger.error(f"Error stopping worker {worker_id}: {str(e)}")
        
        self.active_workers = {}
        logger.info("Colony stopped")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ant Bot Queen")
    parser.add_argument("--init-capital", type=float, help="Initial capital in SOL")
    parser.add_argument("--state", action="store_true", help="Show colony state")
    parser.add_argument("--backup", action="store_true", help="Backup wallets")
    parser.add_argument("--stop", action="store_true", help="Stop the colony")
    
    async def main():
        args = parser.parse_args()
        queen = Queen()
        
        if args.init_capital:
            await queen.initialize_colony(args.init_capital)
        
        if args.state:
            state = await queen.get_colony_state()
            import json
            print(json.dumps(state, indent=2))
        
        if args.backup:
            backup_path = await queen.backup_wallets()
            print(f"Wallets backed up to: {backup_path}")
        
        if args.stop:
            await queen.stop_colony()
        
        # If no arguments are provided, initialize and run the colony
        if not (args.init_capital or args.state or args.backup or args.stop):
            balance = 10.0  # Default balance
            await queen.initialize_colony(balance)
            try:
                # Run forever until interrupted
                while True:
                    await asyncio.sleep(60)
            except KeyboardInterrupt:
                await queen.stop_colony()
    
    if __name__ == "__main__":
        asyncio.run(main()) 