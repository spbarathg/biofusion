import os
import yaml
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
# Direct import that works in both development and production
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log_config import setup_logging
from utils.wallet_manager import WalletManager

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
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        with open(config_path, 'r') as f:
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

    def _setup_logging(self):
        setup_logging("queen", "queen.log")

    def _load_config(self, config_path: str) -> ColonyConfig:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return ColonyConfig(**config['colony'])

    async def initialize_colony(self, initial_capital: float) -> None:
        """Initialize the colony with founding capital."""
        logger.info(f"Initializing colony with {initial_capital} SOL")
        
        # Create queen wallet
        self.colony_state["queen_wallet"] = await self._create_wallet("queen")
        
        # Create savings wallet
        self.colony_state["savings_wallet"] = await self._create_wallet("savings")
        
        # Fund queen wallet with initial capital
        # In a real scenario, this would be done by transferring from an external wallet
        logger.info(f"For initial funding, please transfer {initial_capital} SOL to:")
        logger.info(f"Queen wallet public key: {self.wallet_manager.wallets[self.colony_state['queen_wallet']]['public_key']}")
        
        self.colony_state["total_capital"] = initial_capital
        logger.info("Colony initialized successfully")

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

    async def manage_workers(self) -> None:
        """Manage worker ant deployment and scaling."""
        current_workers = len(self.colony_state["worker_wallets"])
        
        if current_workers < self.config.max_workers_per_vps:
            # Check if we have enough capital to spawn new workers
            available_capital = await self._get_available_capital()
            
            if available_capital >= self.config.min_princess_capital:
                worker_wallet = await self._create_wallet("worker")
                self.colony_state["worker_wallets"].append(worker_wallet)
                
                # Allocate capital to worker
                worker_capital = available_capital * 0.1  # 10% of available capital
                await self._transfer_capital(
                    self.colony_state["queen_wallet"],
                    worker_wallet,
                    worker_capital
                )
                
                logger.info(f"Spawned new worker wallet: {worker_wallet}")

    async def collect_profits(self) -> None:
        """Collect and manage profits from workers and princesses."""
        total_profits = 0.0
        
        # Collect from workers
        for worker in self.colony_state["worker_wallets"]:
            profits = await self._collect_wallet_profits(worker)
            total_profits += profits
        
        # Collect from princesses
        for princess in self.colony_state["princess_wallets"]:
            profits = await self._collect_wallet_profits(princess)
            total_profits += profits
        
        # Split profits between savings and reinvestment
        savings_amount = total_profits * self.savings_ratio
        reinvestment = total_profits * (1 - self.savings_ratio)
        
        # Send to savings
        await self._transfer_capital(
            self.colony_state["queen_wallet"],
            self.colony_state["savings_wallet"],
            savings_amount
        )
        
        # Reinvest remainder
        self.colony_state["total_capital"] += reinvestment
        self.colony_state["total_profits"] += total_profits
        
        logger.info(f"Collected profits: {total_profits} SOL")

    async def _create_wallet(self, wallet_type: str) -> str:
        """Create a new wallet of specified type."""
        wallet_name = f"{wallet_type}_{int(self.colony_state['total_profits'])}"
        wallet_id = await self.wallet_manager.create_wallet(wallet_name, wallet_type)
        return wallet_id

    async def _fund_wallet(self, wallet: str, amount: float) -> None:
        """Fund a wallet with specified amount."""
        # In a real scenario, this would involve transferring funds from an external source
        # For now, we'll just log it
        logger.info(f"To fund wallet, please transfer {amount} SOL to:")
        wallet_info = self.wallet_manager.wallets[wallet]
        logger.info(f"Wallet public key: {wallet_info['public_key']}")

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
        # In a real implementation, this would analyze the wallet's trading history
        # and determine profits to collect
        # For now, it's a stub returning 0
        return 0.0

    async def get_colony_state(self) -> Dict:
        """Get current state of the colony including wallet balances."""
        state = {
            "total_capital": self.colony_state["total_capital"],
            "total_profits": self.colony_state["total_profits"],
            "wallets": {}
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
        for wallet_id in self.colony_state["worker_wallets"]:
            state["wallets"]["workers"].append({
                "id": wallet_id,
                "balance": await self._get_wallet_balance(wallet_id)
            })
        
        return state

    async def backup_wallets(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of all wallets."""
        return await self.wallet_manager.create_backup(backup_path)

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
        "--backup-path",
        type=str,
        help="Path for wallet backup"
    )
    args = parser.parse_args()

    async def main():
        queen = Queen()
        
        if args.init_capital:
            await queen.initialize_colony(args.init_capital)
        
        if args.state:
            state = await queen.get_colony_state()
            print("Colony State:")
            print(f"Total Capital: {state['total_capital']} SOL")
            print(f"Total Profits: {state['total_profits']} SOL")
            print("Wallets:")
            if "queen" in state["wallets"]:
                print(f"  Queen: {state['wallets']['queen']['balance']} SOL")
            if "savings" in state["wallets"]:
                print(f"  Savings: {state['wallets']['savings']['balance']} SOL")
            print(f"  Princesses: {len(state['wallets'].get('princesses', []))}")
            print(f"  Workers: {len(state['wallets'].get('workers', []))}")
        
        if args.backup:
            backup_path = await queen.backup_wallets(args.backup_path)
            print(f"Created wallet backup: {backup_path}")

    asyncio.run(main()) 