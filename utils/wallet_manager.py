import os
import json
import uuid
import time
import base64
import secrets
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from loguru import logger
import yaml

# Encryption imports
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend

class WalletManager:
    """
    Basic manager for Solana wallet operations including:
    - Wallet generation and secure storage
    - Balance checking (mock)
    - Transfer operations (mock)
    
    Note: This implementation doesn't make actual blockchain transactions
    but demonstrates the structure for wallet management.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize the wallet manager with configuration"""
        self.wallets = {}
        self.wallet_dir = Path("wallets")
        self.wallet_dir.mkdir(exist_ok=True)
        
        # Load config if file exists
        self.config = {}
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                try:
                    self.config = yaml.safe_load(f)
                except Exception as e:
                    logger.error(f"Error loading config: {e}")
                    self.config = {}
        
        # Generate encryption key if not exists
        self.encryption_key_path = Path(".encryption_key")
        if not self.encryption_key_path.exists():
            self._generate_encryption_key()
        
        # Load existing wallets
        self._load_wallets()
    
    def _generate_encryption_key(self) -> None:
        """Generate a secure encryption key for wallet data"""
        # Generate a random salt
        salt = os.urandom(16)
        
        # Generate a random password
        password = os.urandom(32)
        
        # Create a key derivation function
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        # Derive the key
        key = kdf.derive(password)
        
        # Save the key and salt
        with open(self.encryption_key_path, "wb") as f:
            f.write(salt + key)
        
        logger.info("Generated new encryption key")
    
    def _get_encryption_key(self) -> Tuple[bytes, bytes]:
        """Get the encryption key and salt"""
        with open(self.encryption_key_path, "rb") as f:
            data = f.read()
        
        salt = data[:16]
        key = data[16:]
        
        return salt, key
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt data using AES-GCM"""
        _, key = self._get_encryption_key()
        
        # Generate a random nonce
        nonce = os.urandom(12)
        
        # Create an AES-GCM cipher
        aesgcm = AESGCM(key)
        
        # Encrypt the data
        ciphertext = aesgcm.encrypt(nonce, data.encode(), None)
        
        # Encode the nonce and ciphertext in base64
        encoded = base64.b64encode(nonce + ciphertext).decode()
        
        return encoded
    
    def _decrypt_data(self, encoded: str) -> str:
        """Decrypt data using AES-GCM"""
        _, key = self._get_encryption_key()
        
        # Decode the base64 data
        data = base64.b64decode(encoded)
        
        # Extract the nonce and ciphertext
        nonce = data[:12]
        ciphertext = data[12:]
        
        # Create an AES-GCM cipher
        aesgcm = AESGCM(key)
        
        # Decrypt the data
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        
        return plaintext.decode()
    
    def _load_wallets(self) -> None:
        """Load all wallets from storage"""
        for wallet_file in self.wallet_dir.glob("*.json"):
            try:
                with open(wallet_file, "r") as f:
                    wallet_data = json.load(f)
                
                wallet_id = wallet_file.stem
                
                # Add to wallet dictionary
                self.wallets[wallet_id] = {
                    "public_key": wallet_data["public_key"],
                    "name": wallet_data["name"],
                    "wallet_type": wallet_data["wallet_type"],
                    "created_at": wallet_data["created_at"]
                }
                
                logger.debug(f"Loaded wallet {wallet_id}: {wallet_data['name']}")
            except Exception as e:
                logger.error(f"Error loading wallet {wallet_file}: {e}")
    
    def create_wallet(self, name: str, wallet_type: str) -> str:
        """
        Create a new Solana wallet
        
        Args:
            name: User-friendly name for the wallet
            wallet_type: Type of wallet (queen, princess, worker, savings)
            
        Returns:
            wallet_id: Unique identifier for the wallet
        """
        # Generate a new keypair (using secrets for this demo)
        private_key = secrets.token_bytes(32)
        # In a real implementation, this would properly derive a public key
        public_key = base64.b64encode(private_key[:16]).decode()
        
        # Generate a unique ID
        wallet_id = str(uuid.uuid4())
        
        # Store the wallet data
        wallet_data = {
            "name": name,
            "wallet_type": wallet_type,
            "public_key": public_key,
            "encrypted_private_key": self._encrypt_data(base64.b64encode(private_key).decode()),
            "created_at": int(time.time())
        }
        
        # Save to file
        wallet_path = self.wallet_dir / f"{wallet_id}.json"
        with open(wallet_path, "w") as f:
            json.dump(wallet_data, f, indent=2)
        
        # Add to wallets dictionary
        self.wallets[wallet_id] = {
            "public_key": public_key,
            "name": name,
            "wallet_type": wallet_type,
            "created_at": wallet_data["created_at"]
        }
        
        logger.info(f"Created new {wallet_type} wallet: {name} ({wallet_id[:8]})")
        return wallet_id
    
    def get_balance(self, wallet_id: str) -> float:
        """
        Get the SOL balance of a wallet (mock implementation)
        
        Args:
            wallet_id: Unique identifier for the wallet
            
        Returns:
            balance: Balance in SOL
        """
        if wallet_id not in self.wallets:
            raise ValueError(f"Wallet {wallet_id} not found")
        
        # In a real implementation, this would query the Solana blockchain
        # For demo purposes, we'll return a mock balance
        return 10.0
    
    def transfer_sol(
        self, 
        from_wallet_id: str, 
        to_wallet_id: str, 
        amount_sol: float
    ) -> str:
        """
        Transfer SOL between wallets (mock implementation)
        
        Args:
            from_wallet_id: Source wallet ID
            to_wallet_id: Destination wallet ID
            amount_sol: Amount to transfer in SOL
            
        Returns:
            transaction_signature: The transaction signature
        """
        if from_wallet_id not in self.wallets:
            raise ValueError(f"Source wallet {from_wallet_id} not found")
        if to_wallet_id not in self.wallets:
            raise ValueError(f"Destination wallet {to_wallet_id} not found")
        
        from_wallet = self.wallets[from_wallet_id]
        to_wallet = self.wallets[to_wallet_id]
        
        # In a real implementation, this would create and sign a Solana transaction
        # For demo purposes, we'll return a mock signature
        mock_signature = base64.b64encode(os.urandom(32)).decode()
        
        logger.info(f"Transferred {amount_sol} SOL from {from_wallet['name']} to {to_wallet['name']} ({mock_signature[:8]})")
        return mock_signature
    
    def transfer_sol_to_external(
        self, 
        from_wallet_id: str, 
        to_pubkey_str: str, 
        amount_sol: float
    ) -> str:
        """
        Transfer SOL to an external wallet address (mock implementation)
        
        Args:
            from_wallet_id: Source wallet ID
            to_pubkey_str: Destination public key as string
            amount_sol: Amount to transfer in SOL
            
        Returns:
            transaction_signature: The transaction signature
        """
        if from_wallet_id not in self.wallets:
            raise ValueError(f"Source wallet {from_wallet_id} not found")
        
        from_wallet = self.wallets[from_wallet_id]
        
        # In a real implementation, this would create and sign a Solana transaction
        # For demo purposes, we'll return a mock signature
        mock_signature = base64.b64encode(os.urandom(32)).decode()
        
        logger.info(f"Transferred {amount_sol} SOL from {from_wallet['name']} to external wallet ({mock_signature[:8]})")
        return mock_signature
    
    def list_wallets(self, wallet_type: Optional[str] = None) -> List[Dict]:
        """
        List all wallets or wallets of a specific type
        
        Args:
            wallet_type: Optional filter for wallet type
            
        Returns:
            List of wallet information dictionaries
        """
        result = []
        
        for wallet_id, wallet in self.wallets.items():
            if wallet_type is None or wallet["wallet_type"] == wallet_type:
                result.append({
                    "id": wallet_id,
                    "name": wallet["name"],
                    "type": wallet["wallet_type"],
                    "public_key": wallet["public_key"],
                    "created_at": wallet["created_at"]
                })
        
        return result
    
    def get_wallet_by_name(self, name: str) -> Optional[str]:
        """
        Get wallet ID by name
        
        Args:
            name: Name of the wallet
            
        Returns:
            wallet_id: ID of the wallet or None if not found
        """
        for wallet_id, wallet in self.wallets.items():
            if wallet["name"] == name:
                return wallet_id
        
        return None
    
    def create_backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create an encrypted backup of all wallets
        
        Args:
            backup_path: Optional path to save the backup
            
        Returns:
            backup_path: Path to the backup file
        """
        if backup_path is None:
            backup_path = f"backup_{int(time.time())}.json"
        
        backup_data = {
            "wallets": {},
            "created_at": int(time.time())
        }
        
        # Collect wallet data
        for wallet_file in self.wallet_dir.glob("*.json"):
            with open(wallet_file, "r") as f:
                wallet_data = json.load(f)
            
            backup_data["wallets"][wallet_file.stem] = wallet_data
        
        # Encrypt and save the backup
        encrypted_backup = self._encrypt_data(json.dumps(backup_data))
        
        with open(backup_path, "w") as f:
            f.write(encrypted_backup)
        
        logger.info(f"Created encrypted wallet backup: {backup_path}")
        return backup_path
    
    def restore_from_backup(self, backup_path: str) -> int:
        """
        Restore wallets from an encrypted backup
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            count: Number of wallets restored
        """
        try:
            with open(backup_path, "r") as f:
                encrypted_backup = f.read()
            
            # Decrypt the backup
            backup_json = self._decrypt_data(encrypted_backup)
            backup_data = json.loads(backup_json)
            
            # Restore wallets
            count = 0
            for wallet_id, wallet_data in backup_data["wallets"].items():
                wallet_path = self.wallet_dir / f"{wallet_id}.json"
                
                if not wallet_path.exists():
                    with open(wallet_path, "w") as f:
                        json.dump(wallet_data, f, indent=2)
                    count += 1
            
            # Reload wallets
            self._load_wallets()
            
            logger.info(f"Restored {count} wallets from backup")
            return count
        
        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            raise

# Simple test if run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Solana Wallet Manager")
    parser.add_argument("--create", action="store_true", help="Create a new wallet")
    parser.add_argument("--name", type=str, help="Wallet name")
    parser.add_argument("--type", type=str, help="Wallet type (queen, princess, worker, savings)")
    parser.add_argument("--list", action="store_true", help="List all wallets")
    parser.add_argument("--backup", action="store_true", help="Create wallet backup")
    parser.add_argument("--backup-path", type=str, help="Backup file path")
    parser.add_argument("--restore", type=str, help="Restore from backup file")
    parser.add_argument("--balance", type=str, help="Get wallet balance by name")
    parser.add_argument("--transfer", action="store_true", help="Transfer SOL between wallets")
    parser.add_argument("--from-wallet", type=str, help="Source wallet name")
    parser.add_argument("--to-wallet", type=str, help="Destination wallet name")
    parser.add_argument("--amount", type=float, help="Amount to transfer in SOL")
    
    args = parser.parse_args()
    
    async def main():
        manager = WalletManager()
        
        if args.create and args.name and args.type:
            wallet_id = await manager.create_wallet(args.name, args.type)
            print(f"Created wallet {args.name} ({wallet_id})")
        
        elif args.list:
            wallets = manager.list_wallets()
            for wallet in wallets:
                print(f"{wallet['name']} ({wallet['type']}): {wallet['public_key']}")
        
        elif args.backup:
            backup_path = await manager.create_backup(args.backup_path)
            print(f"Created backup: {backup_path}")
        
        elif args.restore:
            count = await manager.restore_from_backup(args.restore)
            print(f"Restored {count} wallets from backup")
        
        elif args.balance:
            wallet_id = manager.get_wallet_by_name(args.balance)
            if wallet_id:
                balance = await manager.get_balance(wallet_id)
                print(f"Balance for {args.balance}: {balance} SOL")
            else:
                print(f"Wallet {args.balance} not found")
        
        elif args.transfer and args.from_wallet and args.to_wallet and args.amount:
            from_id = manager.get_wallet_by_name(args.from_wallet)
            to_id = manager.get_wallet_by_name(args.to_wallet)
            
            if from_id and to_id:
                signature = await manager.transfer_sol(from_id, to_id, args.amount)
                print(f"Transferred {args.amount} SOL from {args.from_wallet} to {args.to_wallet}")
                print(f"Transaction signature: {signature}")
            else:
                if not from_id:
                    print(f"Source wallet {args.from_wallet} not found")
                if not to_id:
                    print(f"Destination wallet {args.to_wallet} not found")
    
    asyncio.run(main()) 