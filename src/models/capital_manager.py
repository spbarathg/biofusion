import os
import yaml
import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

# Import modules from the correct paths
from src.core.paths import CONFIG_PATH, LOGS_DIR
from src.core.wallet_manager import WalletManager
from src.logging.log_config import setup_logging

def todo(message: str):
    logger.warning(f"TODO: {message}")
    pass

@dataclass
class CapitalConfig:
    savings_ratio: float
    reinvestment_ratio: float
    compound_frequency: int
    min_savings: float
    max_withdrawal: float
    emergency_reserve: float

class CapitalManager:
    def __init__(self, config_path: str = None):
        # Use provided config path or default
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
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
        setup_logging("capital_manager", "capital_manager.log")

    def _load_config(self, config_path: str) -> CapitalConfig:
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
        logger.info(f"Processing {profits} SOL in profits from {source_wallet[:8]}")
        
        # Split profits
        savings_amount = profits * self.config.savings_ratio
        trading_amount = profits * (1 - self.config.savings_ratio)
        
        # Update metrics
        self.savings_metrics["total_saved"] += savings_amount
        
        # Get or create a savings wallet
        savings_wallet_id = await self._get_savings_wallet_id()
        if not savings_wallet_id:
            logger.warning("No savings wallet found, creating one")
            savings_wallet_id = self.wallet_manager.create_wallet("savings_main", "savings")
        
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

    async def withdraw_savings(self, amount: float, destination_wallet_id: str) -> float:
        """Withdraw a portion of savings for external use."""
        savings_wallet_id = await self._get_savings_wallet_id()
        if not savings_wallet_id:
            logger.warning("No savings wallet found for withdrawal")
            return 0.0
            
        savings_balance = await self._get_savings_balance(savings_wallet_id)
        
        # Limit withdrawal to maximum allowed
        if amount > savings_balance * self.config.max_withdrawal:
            logger.warning(f"Withdrawal amount {amount} exceeds maximum, adjusting")
            amount = savings_balance * self.config.max_withdrawal
        
        # Ensure minimum savings remains and respect emergency reserve
        min_required = max(self.config.min_savings, self.config.emergency_reserve)
        if savings_balance - amount < min_required:
            logger.warning(f"Withdrawal would leave insufficient savings, adjusting")
            amount = max(0, savings_balance - min_required)
        
        if amount <= 0:
            logger.info("No savings available for withdrawal")
            return 0.0
        
        # Process withdrawal
        await self._withdraw_from_savings(savings_wallet_id, destination_wallet_id, amount)
        
        logger.info(f"Withdrawn {amount} SOL from savings")
        return amount

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
            savings_wallet_id = self.wallet_manager.create_wallet("savings_main", "savings")
        
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
        savings_wallets = self.wallet_manager.list_wallets(wallet_type="savings")
        if not savings_wallets:
            return None
        
        # Use the first savings wallet
        return savings_wallets[0]["id"]
        
    async def redistribute_capital(self, queen_wallet_id: str, worker_allocation: float, princess_allocation: float) -> Dict[str, float]:
        """
        Redistribute capital for efficient allocation to workers and princesses
        
        Args:
            queen_wallet_id: Queen wallet ID to distribute from
            worker_allocation: Percentage to allocate to workers
            princess_allocation: Percentage to allocate to princesses
            
        Returns:
            Dict with allocated amounts
        """
        # Get queen balance
        queen_balance = await self.wallet_manager.get_balance(queen_wallet_id)
        
        # Calculate allocations
        worker_amount = queen_balance * worker_allocation
        princess_amount = queen_balance * princess_allocation
        queen_remaining = queen_balance - worker_amount - princess_amount
        
        logger.info(f"Redistributing capital: {worker_amount} SOL to workers, {princess_amount} SOL to princesses")
        
        return {
            "worker_allocation": worker_amount,
            "princess_allocation": princess_amount,
            "queen_remaining": queen_remaining
        }
        
    async def emergency_recovery(self, destination_wallet_id: str) -> float:
        """
        Emergency recovery to withdraw emergency funds
        
        Args:
            destination_wallet_id: Destination wallet for emergency funds
            
        Returns:
            Amount recovered
        """
        # Get savings wallet
        savings_wallet_id = await self._get_savings_wallet_id()
        if not savings_wallet_id:
            logger.error("No savings wallet found for emergency recovery")
            return 0.0
            
        # Get balance
        savings_balance = await self._get_savings_balance(savings_wallet_id)
        
        # Calculate emergency amount (up to 90% of savings in an emergency)
        emergency_amount = min(
            savings_balance * 0.9,
            savings_balance - 0.1  # Leave at least 0.1 SOL
        )
        
        if emergency_amount <= 0:
            logger.warning("No funds available for emergency recovery")
            return 0.0
            
        # Transfer emergency funds
        await self._withdraw_from_savings(savings_wallet_id, destination_wallet_id, emergency_amount)
        
        logger.warning(f"EMERGENCY RECOVERY: Withdrawn {emergency_amount} SOL from savings")
        return emergency_amount

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ant Bot Capital Manager")
    parser.add_argument(
        "--action",
        choices=["process", "compound", "withdraw", "metrics", "emergency", "redistribute"],
        required=True,
        help="Action to perform"
    )
    parser.add_argument(
        "--amount",
        type=float,
        help="Amount for process/withdraw actions"
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Source wallet name for process action"
    )
    parser.add_argument(
        "--destination",
        type=str,
        help="Destination wallet name for withdraw action"
    )
    parser.add_argument(
        "--worker-allocation",
        type=float,
        default=0.4,
        help="Percentage to allocate to workers"
    )
    parser.add_argument(
        "--princess-allocation",
        type=float,
        default=0.4,
        help="Percentage to allocate to princesses"
    )
    args = parser.parse_args()

    async def main():
        manager = CapitalManager()
        
        if args.action == "process":
            if not args.amount or not args.source:
                print("Amount and source required for process action")
                return
                
            source_wallet_id = manager.wallet_manager.get_wallet_by_name(args.source)
            if not source_wallet_id:
                print(f"Source wallet {args.source} not found")
                return
                
            result = await manager.process_profits(args.amount, source_wallet_id)
            print(f"Processed profits: {result}")
        
        elif args.action == "compound":
            # Find queen wallet
            queen_wallets = manager.wallet_manager.list_wallets(wallet_type="queen")
            if not queen_wallets:
                print("No queen wallet found for compounding")
                return
                
            queen_wallet_id = queen_wallets[0]["id"]
            amount = await manager.compound_savings(queen_wallet_id)
            print(f"Compounded amount: {amount} SOL")
        
        elif args.action == "withdraw":
            if not args.amount or not args.destination:
                print("Amount and destination required for withdraw action")
                return
                
            destination_wallet_id = manager.wallet_manager.get_wallet_by_name(args.destination)
            if not destination_wallet_id:
                print(f"Destination wallet {args.destination} not found")
                return
                
            amount = await manager.withdraw_savings(args.amount, destination_wallet_id)
            print(f"Withdrawn amount: {amount} SOL")
        
        elif args.action == "metrics":
            metrics = await manager.get_savings_metrics()
            print("Savings metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
                
        elif args.action == "emergency":
            if not args.destination:
                print("Destination required for emergency action")
                return
                
            destination_wallet_id = manager.wallet_manager.get_wallet_by_name(args.destination)
            if not destination_wallet_id:
                print(f"Destination wallet {args.destination} not found")
                return
                
            amount = await manager.emergency_recovery(destination_wallet_id)
            print(f"Emergency recovery amount: {amount} SOL")
            
        elif args.action == "redistribute":
            # Find queen wallet
            queen_wallets = manager.wallet_manager.list_wallets(wallet_type="queen")
            if not queen_wallets:
                print("No queen wallet found for redistribution")
                return
                
            queen_wallet_id = queen_wallets[0]["id"]
            result = await manager.redistribute_capital(
                queen_wallet_id,
                args.worker_allocation,
                args.princess_allocation
            )
            print(f"Capital redistribution: {result}")

    asyncio.run(main()) 