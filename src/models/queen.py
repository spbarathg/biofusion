import os
import yaml
import logging
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

# Fix imports
from src.logging.log_config import setup_logging
from src.core.wallet_manager import WalletManager
from src.core.paths import CONFIG_DIR, QUEEN_CONFIG_PATH
from src.bindings.worker_bridge import WorkerBridge
from src.models.capital_manager import CapitalManager

def todo(message: str):
    logger.warning(f"TODO: {message}")
    pass

@dataclass
class ColonyConfig:
    initial_capital: float
    min_princess_capital: float
    max_workers_per_vps: int
    max_workers: int
    min_workers: int
    worker_timeout: int
    health_check_interval: int

class Queen:
    def __init__(self, config_path: str = None):
        # Use provided config path or default
        self.config_path = Path(config_path) if config_path else QUEEN_CONFIG_PATH
        self.config = self._load_config(self.config_path)
        
        # Load savings ratio from config
        with open(self.config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            self.savings_ratio = full_config['capital']['savings_ratio']
            
        self.colony_state = {
            "queen_wallet": None,
            "princess_wallets": [],
            "worker_wallets": [],
            "savings_wallet": None,
            "total_capital": 0.0,
            "total_profits": 0.0,
        }
        self._setup_logging()
        self.wallet_manager = WalletManager(config_path)
        self.worker_bridge = WorkerBridge(str(self.config_path))
        self.capital_manager = CapitalManager(str(self.config_path))
        
        # Worker state tracking
        self.active_workers = {}

    def _setup_logging(self):
        setup_logging("queen", "queen.log")

    def _load_config(self, config_path: str) -> ColonyConfig:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return ColonyConfig(**config['colony'])

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
        logger.info(f"Queen wallet public key: {self.wallet_manager.wallets[self.colony_state['queen_wallet']]['public_key']}")
        
        self.colony_state["total_capital"] = initial_capital
        logger.info("Colony initialized successfully")
        
        # Start the health check monitoring
        asyncio.create_task(self._health_check_loop())

    async def spawn_princess(self) -> None:
        """Spawn a new princess wallet when conditions are met."""
        queen_balance = await self._get_wallet_balance(
            self.colony_state["queen_wallet"]
        )
        
        if queen_balance >= self.config.min_princess_capital:
            princess_wallet = await self._create_wallet("princess")
            self.colony_state["princess_wallets"].append(princess_wallet)
            
            # Split capital
            split_amount = queen_balance / 2
            await self._transfer_capital(
                self.colony_state["queen_wallet"],
                princess_wallet,
                split_amount
            )
            
            logger.info(f"Spawned new princess wallet: {princess_wallet}")
            
            # Start a princess manager for the new wallet
            # TODO: Implement princess management logic

    async def manage_workers(self) -> None:
        """Manage worker ant deployment and scaling."""
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
                logger.info(f"Colony has {current_workers} workers, can spawn more with {available_capital} SOL available")
                
                # Calculate how many more workers we can spawn
                additional_workers = min(
                    self.config.max_workers - current_workers,
                    int(available_capital / 0.1)  # Minimum 0.1 SOL per worker
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
            wallet_info = self.wallet_manager.wallets[worker_wallet]
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
                
                # Check if we should spawn a princess
                await self.spawn_princess()
                
                # Sleep for the configured interval
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute on error

    async def collect_profits(self) -> None:
        """Collect and manage profits from workers and princesses."""
        total_profits = 0.0
        
        # Collect from workers
        for worker_id, info in self.active_workers.items():
            wallet_id = info["wallet_id"]
            profits = await self._collect_wallet_profits(wallet_id)
            total_profits += profits
        
        # Collect from princesses
        for princess in self.colony_state["princess_wallets"]:
            profits = await self._collect_wallet_profits(princess)
            total_profits += profits
        
        if total_profits > 0:
            logger.info(f"Collected total profits: {total_profits} SOL")
            
            # Process profits through capital manager
            await self.capital_manager.process_profits(
                total_profits,
                self.colony_state["queen_wallet"]
            )
            
            # Update total profits metric
            self.colony_state["total_profits"] += total_profits
            
            # Check if we should compound savings
            await self._compound_savings()

    async def _compound_savings(self) -> None:
        """Compound savings by reinvesting a portion back into trading."""
        compound_amount = await self.capital_manager.compound_savings(
            self.colony_state["queen_wallet"]
        )
        
        if compound_amount > 0:
            logger.info(f"Compounded {compound_amount} SOL from savings")
            self.colony_state["total_capital"] += compound_amount

    async def _collect_worker_profits(self, worker_id: str, wallet_id: str) -> float:
        """Collect profits from a specific worker."""
        try:
            # Get worker status from Rust engine
            status = await self.worker_bridge.get_worker_status(worker_id)
            
            if not status:
                logger.warning(f"Worker {worker_id} status not available")
                return 0.0
            
            profit = float(status.get("total_profit", 0))
            
            if profit > 0.05:  # Minimum 0.05 SOL to collect
                # Get current balance
                balance = await self._get_wallet_balance(wallet_id)
                
                # Calculate amount to collect (leave some for gas)
                collect_amount = balance - 0.05  # Leave 0.05 SOL for gas
                
                if collect_amount > 0:
                    # Transfer to queen wallet
                    await self._transfer_capital(
                        wallet_id,
                        self.colony_state["queen_wallet"],
                        collect_amount
                    )
                    
                    logger.info(f"Collected {collect_amount} SOL from worker {worker_id}")
                    
                    # Reset profit counter in Rust engine
                    await self.worker_bridge.update_worker_metrics(worker_id, 0, 0.0)
                    
                    return collect_amount
            
            return 0.0
                
        except Exception as e:
            logger.error(f"Error collecting profits from worker {worker_id}: {str(e)}")
            return 0.0

    async def _create_wallet(self, wallet_type: str) -> str:
        """Create a new wallet of specified type."""
        wallet_name = f"{wallet_type}_{int(self.colony_state['total_profits'])}"
        wallet_id = self.wallet_manager.create_wallet(wallet_name, wallet_type)
        return wallet_id

    async def _get_wallet_balance(self, wallet: str) -> float:
        """Get balance of specified wallet."""
        return await self.wallet_manager.get_balance(wallet)

    async def _transfer_capital(
        self,
        from_wallet: str,
        to_wallet: str,
        amount: float
    ) -> None:
        """Transfer capital between wallets."""
        await self.wallet_manager.transfer_sol(from_wallet, to_wallet, amount)

    async def _get_available_capital(self) -> float:
        """Get total available capital for new workers."""
        queen_balance = await self._get_wallet_balance(self.colony_state["queen_wallet"])
        return queen_balance * 0.5  # Use 50% of queen's balance for worker allocation

    async def _collect_wallet_profits(self, wallet: str) -> float:
        """Collect profits from a wallet."""
        try:
            # Get current balance
            balance = await self._get_wallet_balance(wallet)
            
            # We consider profit if balance is above initial allocation
            # This is a simplification - in a real implementation we would track deposits and withdrawals
            initial_allocation = 0.1  # Assume 0.1 SOL initial allocation
            profit = max(0, balance - initial_allocation)
            
            if profit > 0.01:  # Only collect if profit is significant
                # Calculate amount to collect (leave some for gas)
                collect_amount = profit - 0.01  # Leave 0.01 SOL for gas
                
                # Transfer to queen wallet
                await self._transfer_capital(
                    wallet,
                    self.colony_state["queen_wallet"],
                    collect_amount
                )
                
                logger.info(f"Collected {collect_amount} SOL profit from wallet {wallet}")
                return collect_amount
            
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
            "active_workers": len(self.active_workers)
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
        
        # Princess wallets
        state["wallets"]["princesses"] = []
        for wallet_id in self.colony_state["princess_wallets"]:
            state["wallets"]["princesses"].append({
                "id": wallet_id,
                "balance": await self._get_wallet_balance(wallet_id)
            })
        
        # Worker wallets
        state["wallets"]["workers"] = []
        for worker_id, info in self.active_workers.items():
            wallet_id = info["wallet_id"]
            status = await self.worker_bridge.get_worker_status(worker_id)
            
            worker_state = {
                "id": wallet_id,
                "worker_id": worker_id,
                "balance": await self._get_wallet_balance(wallet_id),
                "is_active": bool(status and status.get("is_running", False)),
                "trades_executed": status.get("trades_executed", 0) if status else 0,
                "profit": status.get("total_profit", 0.0) if status else 0.0
            }
            
            state["wallets"]["workers"].append(worker_state)
        
        return state

    async def backup_wallets(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of all wallets."""
        return await self.wallet_manager.create_backup(backup_path)

    async def stop_colony(self) -> None:
        """Stop all workers and prepare for shutdown."""
        logger.info("Stopping colony...")
        
        # Stop all workers
        for worker_id in list(self.active_workers.keys()):
            logger.info(f"Stopping worker {worker_id}")
            await self.worker_bridge.stop_worker(worker_id)
        
        self.active_workers = {}
        logger.info("Colony stopped successfully")

if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Ant Bot Queen")
    parser.add_argument(
        "--init-capital",
        type=float,
        help="Initial capital in SOL"
    )
    parser.add_argument(
        "--state",
        action="store_true",
        help="Show colony state"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup wallets"
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the queen service"
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the queen service"
    )
    args = parser.parse_args()

    async def main():
        queen = Queen()
        
        if args.init_capital:
            await queen.initialize_colony(args.init_capital)
            
        if args.state:
            state = await queen.get_colony_state()
            print(f"Colony State: {state}")
            
        if args.backup:
            backup_path = await queen.backup_wallets()
            print(f"Backup created at: {backup_path}")
            
        if args.start:
            await queen.initialize_colony(args.init_capital or 1.0)
            await queen.manage_workers()
            
            # Run indefinitely
            while True:
                await asyncio.sleep(60)
                
        if args.stop:
            await queen.stop_colony()

    asyncio.run(main()) 