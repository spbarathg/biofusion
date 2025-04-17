import os
import yaml
import asyncio
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
class CapitalConfig:
    savings_ratio: float
    reinvestment_ratio: float
    compound_frequency: int
    min_savings: float
    max_withdrawal: float

class CapitalManager:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.savings_metrics = {
            "total_saved": 0.0,
            "total_reinvested": 0.0,
            "compound_events": 0,
            "last_compound_time": 0,
        }
        self._setup_logging()
        self.wallet_manager = WalletManager(config_path)

    def _setup_logging(self):
        setup_logging("capital_manager", "capital_manager.log")

    def _load_config(self, config_path: str) -> CapitalConfig:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return CapitalConfig(
            savings_ratio=0.90,     # 90% to savings
            reinvestment_ratio=0.80, # 80% of savings reinvested
            compound_frequency=24,   # Compound every 24 hours
            min_savings=1.0,        # Minimum 1 SOL in savings
            max_withdrawal=0.5,     # Maximum 50% withdrawal at once
        )

    async def process_profits(self, profits: float, source_wallet: str) -> Dict[str, float]:
        """Process profits and allocate between savings and reinvestment."""
        logger.info(f"Processing {profits} SOL in profits from {source_wallet[:8]}")
        
        # Split profits
        savings_amount = profits * self.config.savings_ratio
        trading_amount = profits * (1 - self.config.savings_ratio)
        
        # Update metrics
        self.savings_metrics["total_saved"] += savings_amount
        
        # Send to savings - using the wallet manager
        await self._send_to_savings(source_wallet, savings_amount)
        
        logger.info(f"Allocated {savings_amount} SOL to savings, {trading_amount} SOL for trading")
        return {
            "savings": savings_amount,
            "trading": trading_amount,
        }

    async def compound_savings(self, queen_wallet_id: str) -> float:
        """Compound savings by reinvesting a portion back into trading."""
        current_time = asyncio.get_event_loop().time()
        hours_since_last_compound = (current_time - self.savings_metrics["last_compound_time"]) / 3600
        
        if hours_since_last_compound < self.config.compound_frequency:
            logger.info(f"Not time to compound yet. {self.config.compound_frequency - hours_since_last_compound:.1f} hours remaining")
            return 0.0
        
        # Get current savings balance - get all savings wallets
        savings_wallet_id = await self._get_savings_wallet_id()
        if not savings_wallet_id:
            logger.warning("No savings wallet found for compounding")
            return 0.0
            
        savings_balance = await self._get_savings_balance(savings_wallet_id)
        
        if savings_balance < self.config.min_savings:
            logger.info(f"Savings balance {savings_balance} SOL below minimum, skipping compound")
            return 0.0
        
        # Calculate compound amount
        compound_amount = savings_balance * self.config.reinvestment_ratio
        
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
        
        # Ensure minimum savings remains
        if savings_balance - amount < self.config.min_savings:
            logger.warning(f"Withdrawal would leave insufficient savings, adjusting")
            amount = savings_balance - self.config.min_savings
        
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
        savings_wallets = self.wallet_manager.list_wallets(wallet_type="savings")
        if not savings_wallets:
            return None
        
        # Use the first savings wallet
        return savings_wallets[0]["id"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ant Bot Capital Manager")
    parser.add_argument(
        "--action",
        choices=["process", "compound", "withdraw", "metrics"],
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

    asyncio.run(main()) 