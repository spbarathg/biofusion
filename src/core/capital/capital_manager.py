import os
import yaml
import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from src.utils.logging.logger import setup_logging
from src.core.wallet.wallet_manager import WalletManager

@dataclass
class CapitalConfig:
    """Configuration for capital management"""
    savings_ratio: float
    reinvestment_ratio: float
    compound_frequency: int
    min_savings: float
    max_withdrawal: float
    emergency_reserve: float

class CapitalManager:
    """
    Capital Manager for handling profit allocation, savings and reinvestment.
    """
    def __init__(self, config_path: str = "config/settings.yaml"):
        # Convert config_path to absolute path if needed
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), config_path)
        
        self.config_path = config_path
        self.config = self._load_config(self.config_path)
        self.savings_metrics = {
            "total_saved": 0.0,
            "total_reinvested": 0.0,
            "compound_events": 0,
            "last_compound_time": time.time(),
        }
        self._setup_logging()
        self.wallet_manager = WalletManager(config_path)

    def _setup_logging(self):
        """Set up logging for the capital manager"""
        setup_logging("capital_manager", "capital_manager.log")

    def _load_config(self, config_path: str) -> CapitalConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            capital_config = config.get("capital", {})
            
        return CapitalConfig(
            savings_ratio=capital_config.get("savings_ratio", 0.90),
            reinvestment_ratio=capital_config.get("reinvestment_ratio", 0.80),
            compound_frequency=capital_config.get("compound_frequency", 24),
            min_savings=capital_config.get("min_savings", 1.0),
            max_withdrawal=capital_config.get("max_withdrawal", 0.5),
            emergency_reserve=capital_config.get("emergency_reserve", 100.0)
        )

    async def process_profits(self, profits: float, source_wallet: str) -> Dict[str, float]:
        """Process profits and allocate between savings and reinvestment."""
        logger.info(f"Processing {profits} SOL in profits from wallet {source_wallet[:8]}")
        
        # Split profits
        savings_amount = profits * self.config.savings_ratio
        trading_amount = profits * (1 - self.config.savings_ratio)
        
        # Update metrics
        self.savings_metrics["total_saved"] += savings_amount
        
        # Get or create a savings wallet
        savings_wallet_id = await self._get_savings_wallet_id()
        if not savings_wallet_id:
            logger.warning("No savings wallet found, creating one")
            savings_wallet_id = await self.wallet_manager.create_wallet("savings_main", "savings")
        
        # Send to savings - using the wallet manager
        if savings_amount > 0.01:  # Only transfer if significant
            await self.wallet_manager.transfer_sol(source_wallet, savings_wallet_id, savings_amount)
            logger.info(f"Transferred {savings_amount} SOL to savings wallet")
        
        logger.info(f"Allocated {savings_amount} SOL to savings, {trading_amount} SOL for trading")
        return {
            "savings": savings_amount,
            "trading": trading_amount,
        }

    async def compound_savings(self, queen_wallet_id: str) -> float:
        """Compound savings by reinvesting a portion back into trading."""
        current_time = time.time()
        hours_since_last_compound = (current_time - self.savings_metrics["last_compound_time"]) / 3600
        
        if hours_since_last_compound < self.config.compound_frequency:
            logger.info(f"Not time to compound yet. {self.config.compound_frequency - hours_since_last_compound:.1f} hours remaining")
            return 0.0
        
        # Get current savings balance
        savings_wallet_id = await self._get_savings_wallet_id()
        if not savings_wallet_id:
            logger.warning("No savings wallet found for compounding")
            return 0.0
            
        savings_balance = await self._get_savings_balance(savings_wallet_id)
        
        if savings_balance < self.config.min_savings:
            logger.info(f"Savings balance {savings_balance} SOL below minimum, skipping compound")
            return 0.0
        
        # Calculate compound amount
        compound_amount = min(
            savings_balance * self.config.reinvestment_ratio,
            savings_balance - self.config.emergency_reserve
        )
        
        # Ensure we're not going below the minimum
        if savings_balance - compound_amount < self.config.min_savings:
            compound_amount = savings_balance - self.config.min_savings
            
        if compound_amount <= 0:
            logger.info("No funds available for compounding")
            return 0.0
        
        # Withdraw from savings and send to queen for redistribution
        await self._withdraw_from_savings(savings_wallet_id, queen_wallet_id, compound_amount)
        
        # Update metrics
        self.savings_metrics["total_reinvested"] += compound_amount
        self.savings_metrics["compound_events"] += 1
        self.savings_metrics["last_compound_time"] = current_time
        
        logger.info(f"Compounded {compound_amount} SOL from savings")
        return compound_amount

    async def get_savings_metrics(self) -> Dict[str, float]:
        """Get current savings metrics."""
        savings_wallet_id = await self._get_savings_wallet_id()
        savings_balance = 0.0
        
        if savings_wallet_id:
            savings_balance = await self._get_savings_balance(savings_wallet_id)
        
        return {
            "current_balance": savings_balance,
            "total_saved": self.savings_metrics["total_saved"],
            "total_reinvested": self.savings_metrics["total_reinvested"],
            "compound_events": self.savings_metrics["compound_events"],
            "compound_rate": self.savings_metrics["total_reinvested"] / max(1, self.savings_metrics["total_saved"]),
            "last_compound_time": self.savings_metrics["last_compound_time"],
            "hours_since_last_compound": (time.time() - self.savings_metrics["last_compound_time"]) / 3600,
            "next_compound_in_hours": max(0, self.config.compound_frequency - (time.time() - self.savings_metrics["last_compound_time"]) / 3600)
        }

    async def _send_to_savings(self, source_wallet_id: str, amount: float) -> None:
        """Send funds to savings wallet."""
        savings_wallet_id = await self._get_savings_wallet_id()
        if not savings_wallet_id:
            logger.warning("No savings wallet found, creating one")
            savings_wallet_id = await self.wallet_manager.create_wallet("savings_main", "savings")
        
        await self.wallet_manager.transfer_sol(source_wallet_id, savings_wallet_id, amount)
        logger.info(f"Transferred {amount} SOL to savings wallet")

    async def _withdraw_from_savings(self, savings_wallet_id: str, destination_wallet_id: str, amount: float) -> None:
        """Withdraw funds from savings wallet."""
        await self.wallet_manager.transfer_sol(savings_wallet_id, destination_wallet_id, amount)
        logger.info(f"Withdrawn {amount} SOL from savings wallet")

    async def _get_savings_balance(self, savings_wallet_id: str) -> float:
        """Get current savings wallet balance."""
        return await self.wallet_manager.get_balance(savings_wallet_id)
        
    async def _get_savings_wallet_id(self) -> Optional[str]:
        """Get the ID of the savings wallet."""
        savings_wallets = await self.wallet_manager.list_wallets(wallet_type="savings")
        if not savings_wallets:
            return None
        
        # Use the first savings wallet
        return savings_wallets[0]["id"]
        
    async def redistribute_capital(self, queen_wallet_id: str, worker_allocation: float = 0.7) -> Dict[str, float]:
        """
        Redistribute capital for efficient allocation to workers
        
        Args:
            queen_wallet_id: Queen wallet ID to distribute from
            worker_allocation: Percentage to allocate to workers (default 70%)
            
        Returns:
            Dict with allocated amounts
        """
        # Get queen balance
        queen_balance = await self.wallet_manager.get_balance(queen_wallet_id)
        
        # Calculate allocations
        worker_amount = queen_balance * worker_allocation
        queen_remaining = queen_balance - worker_amount
        
        logger.info(f"Redistributing capital: {worker_amount} SOL to workers, {queen_remaining} SOL remains with queen")
        
        return {
            "worker_allocation": worker_amount,
            "queen_remaining": queen_remaining
        }

if __name__ == "__main__":
    # Example usage
    async def main():
        manager = CapitalManager()
        metrics = await manager.get_savings_metrics()
        print(f"Savings Metrics: {metrics}")
        
    asyncio.run(main()) 