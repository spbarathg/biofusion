import os
import json
import uuid
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

from src.utils.logging.logger import setup_logging

class WalletManager:
    """
    Manages Solana wallets for the trading system.
    Handles wallet creation, storage, backup and transactions.
    """
    def __init__(self, config_path: str = "config/settings.yaml"):
        # Convert config_path to absolute path if needed
        if not os.path.isabs(config_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            config_path = os.path.join(base_dir, config_path)
            
        self.config_path = config_path
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.wallets_dir = os.path.join(self.base_dir, "data", "wallets")
        self.backup_dir = os.path.join(self.base_dir, "data", "backups")
        
        # Ensure directories exist
        os.makedirs(self.wallets_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Initialize wallet storage
        self.wallets = {}
        self._load_wallets()
        
        # Setup logging
        setup_logging("wallet_manager", "wallet_manager.log")
        
    def _load_wallets(self) -> None:
        """Load wallets from disk"""
        wallet_files = [f for f in os.listdir(self.wallets_dir) if f.endswith('.json')]
        
        for wallet_file in wallet_files:
            try:
                with open(os.path.join(self.wallets_dir, wallet_file), 'r') as f:
                    wallet_data = json.load(f)
                    wallet_id = wallet_data.get('id', os.path.splitext(wallet_file)[0])
                    self.wallets[wallet_id] = wallet_data
            except Exception as e:
                logger.error(f"Error loading wallet from {wallet_file}: {str(e)}")
                
        logger.info(f"Loaded {len(self.wallets)} wallets")
        
    def _save_wallet(self, wallet_id: str) -> None:
        """Save wallet to disk"""
        if wallet_id not in self.wallets:
            logger.error(f"Cannot save non-existent wallet: {wallet_id}")
            return
            
        wallet_data = self.wallets[wallet_id]
        wallet_path = os.path.join(self.wallets_dir, f"{wallet_id}.json")
        
        with open(wallet_path, 'w') as f:
            json.dump(wallet_data, f, indent=2)
            
    async def create_wallet(self, name: str, wallet_type: str) -> str:
        """
        Create a new wallet
        
        Args:
            name: Name for the wallet
            wallet_type: Type of wallet (queen, worker, savings, etc)
            
        Returns:
            Wallet ID
        """
        # Generate a unique ID
        wallet_id = str(uuid.uuid4())
        
        # In a real implementation, this would create an actual Solana wallet
        # For now, we'll just simulate it
        public_key = f"SIMULATED_{wallet_id[:8]}"
        private_key = f"SIMULATED_PRIVATE_{wallet_id[:8]}"
        
        # Create wallet record
        wallet_data = {
            "id": wallet_id,
            "name": name,
            "type": wallet_type,
            "public_key": public_key,
            "private_key": private_key,  # In real implementation, this would be encrypted
            "created_at": time.time(),
            "balance": 0.0,
            "last_updated": time.time()
        }
        
        # Store wallet
        self.wallets[wallet_id] = wallet_data
        self._save_wallet(wallet_id)
        
        logger.info(f"Created {wallet_type} wallet: {name} ({wallet_id[:8]})")
        return wallet_id
        
    async def get_balance(self, wallet_id: str) -> float:
        """
        Get wallet balance
        
        Args:
            wallet_id: Wallet ID
            
        Returns:
            Balance in SOL
        """
        if wallet_id not in self.wallets:
            logger.error(f"Unknown wallet: {wallet_id}")
            return 0.0
            
        # In a real implementation, this would query the Solana blockchain
        # For now, we'll just return the cached balance
        return float(self.wallets[wallet_id].get("balance", 0.0))
        
    async def transfer_sol(self, from_wallet: str, to_wallet: str, amount: float) -> bool:
        """
        Transfer SOL between wallets
        
        Args:
            from_wallet: Source wallet ID
            to_wallet: Destination wallet ID
            amount: Amount in SOL
            
        Returns:
            Success status
        """
        # Validate inputs
        if from_wallet not in self.wallets:
            logger.error(f"Unknown source wallet: {from_wallet}")
            return False
            
        if to_wallet not in self.wallets:
            logger.error(f"Unknown destination wallet: {to_wallet}")
            return False
            
        if amount <= 0:
            logger.error(f"Invalid transfer amount: {amount}")
            return False
            
        # Check balance
        from_balance = await self.get_balance(from_wallet)
        if from_balance < amount:
            logger.error(f"Insufficient balance: {from_balance} SOL, trying to transfer {amount} SOL")
            return False
            
        # In a real implementation, this would execute a Solana transaction
        # For now, we'll just update the cached balances
        
        # Update source wallet
        self.wallets[from_wallet]["balance"] = from_balance - amount
        self.wallets[from_wallet]["last_updated"] = time.time()
        self._save_wallet(from_wallet)
        
        # Update destination wallet
        to_balance = await self.get_balance(to_wallet)
        self.wallets[to_wallet]["balance"] = to_balance + amount
        self.wallets[to_wallet]["last_updated"] = time.time()
        self._save_wallet(to_wallet)
        
        logger.info(f"Transferred {amount} SOL from {from_wallet[:8]} to {to_wallet[:8]}")
        return True
        
    async def list_wallets(self, wallet_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List wallets, optionally filtered by type
        
        Args:
            wallet_type: Optional type filter
            
        Returns:
            List of wallet data
        """
        result = []
        
        for wallet_id, wallet_data in self.wallets.items():
            if wallet_type is None or wallet_data.get("type") == wallet_type:
                # Make a copy with sensitive data removed
                wallet_copy = wallet_data.copy()
                if "private_key" in wallet_copy:
                    del wallet_copy["private_key"]
                result.append(wallet_copy)
                
        return result
        
    def get_public_key(self, wallet_id: str) -> str:
        """Get wallet's public key"""
        if wallet_id not in self.wallets:
            logger.error(f"Unknown wallet: {wallet_id}")
            return ""
            
        return self.wallets[wallet_id].get("public_key", "")
        
    def get_wallet_info(self, wallet_id: str) -> Dict[str, Any]:
        """Get wallet info (excluding private key)"""
        if wallet_id not in self.wallets:
            logger.error(f"Unknown wallet: {wallet_id}")
            return {}
            
        # Make a copy with sensitive data removed
        wallet_copy = self.wallets[wallet_id].copy()
        if "private_key" in wallet_copy:
            del wallet_copy["private_key"]
            
        return wallet_copy
        
    async def create_backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of all wallets
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to backup file
        """
        if not backup_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"wallet_backup_{timestamp}.json")
            
        # Create a backup with encrypted private keys
        backup_data = {
            "timestamp": time.time(),
            "wallets": self.wallets
        }
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
            
        logger.info(f"Created wallet backup at {backup_path}")
        return backup_path
        
    async def restore_from_backup(self, backup_path: str) -> bool:
        """
        Restore wallets from backup
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Success status
        """
        try:
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
                
            if "wallets" not in backup_data:
                logger.error(f"Invalid backup format in {backup_path}")
                return False
                
            # Replace current wallets
            self.wallets = backup_data["wallets"]
            
            # Save all wallets to disk
            for wallet_id in self.wallets:
                self._save_wallet(wallet_id)
                
            logger.info(f"Restored {len(self.wallets)} wallets from backup")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from backup {backup_path}: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    async def main():
        manager = WalletManager()
        
        # Create wallets
        queen_id = await manager.create_wallet("queen_main", "queen")
        worker_id = await manager.create_wallet("worker_1", "worker")
        
        # List wallets
        wallets = await manager.list_wallets()
        print(f"Created wallets: {wallets}")
        
        # Simulate funding
        # In a real implementation, this would involve external transfers
        manager.wallets[queen_id]["balance"] = 10.0
        manager._save_wallet(queen_id)
        
        # Transfer funds
        await manager.transfer_sol(queen_id, worker_id, 1.0)
        
        # Check balances
        queen_balance = await manager.get_balance(queen_id)
        worker_balance = await manager.get_balance(worker_id)
        print(f"Queen balance: {queen_balance}")
        print(f"Worker balance: {worker_balance}")
        
        # Create backup
        backup_path = await manager.create_backup()
        print(f"Created backup at {backup_path}")
        
    asyncio.run(main()) 