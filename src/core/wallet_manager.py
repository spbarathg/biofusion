import os
import json
import uuid
import time
import base64
from typing import Dict, List, Optional, Any
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.rpc.api import Client as SolanaClient
from solana.transaction import Transaction
from solana.system_program import TransferParams, transfer
import loguru

# Import paths module
from .paths import WALLETS_DIR, BACKUPS_DIR, ENCRYPTION_KEY_PATH, CONFIG_PATH

class WalletManager:
    """
    Manages wallets for the AntBot system, including creation, storage, and SOL transfers.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize wallet manager with configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Use default config path if not specified
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        
        # Set up paths
        self.wallets_dir = WALLETS_DIR
        
        # Load encryption key
        self.key_path = ENCRYPTION_KEY_PATH
        self._load_or_create_key()
        
        # Initialize wallet dictionary
        self.wallets = {}
        
        # Load all wallets
        self._load_wallets()
        
        # Set up Solana client
        self.client = SolanaClient("https://api.mainnet-beta.solana.com")
        # Can be overridden with testnet or devnet
        # self.client = SolanaClient("https://api.devnet.solana.com")
    
    def _load_or_create_key(self):
        """Load existing encryption key or create a new one."""
        try:
            if self.key_path.exists():
                with open(self.key_path, 'rb') as f:
                    self.key = f.read()
            else:
                # Generate a new key
                self.key = Fernet.generate_key()
                with open(self.key_path, 'wb') as f:
                    f.write(self.key)
            
            self.fernet = Fernet(self.key)
            
        except Exception as e:
            loguru.logger.error(f"Error loading/creating encryption key: {str(e)}")
            raise
    
    def _load_wallets(self):
        """Load all wallets from file system."""
        try:
            for wallet_file in self.wallets_dir.glob('*.json'):
                try:
                    with open(wallet_file, 'r') as f:
                        wallet_data = json.load(f)
                    
                    # Decrypt private key
                    encrypted_key = wallet_data.pop('encrypted_private_key', None)
                    if encrypted_key:
                        # Store the decrypted key in memory only (not saved to disk)
                        wallet_data['private_key'] = self.fernet.decrypt(
                            encrypted_key.encode()
                        ).decode()
                    
                    wallet_id = wallet_data.get('id')
                    if wallet_id:
                        self.wallets[wallet_id] = wallet_data
                
                except Exception as e:
                    loguru.logger.error(f"Error loading wallet {wallet_file.name}: {str(e)}")
        
        except Exception as e:
            loguru.logger.error(f"Error loading wallets: {str(e)}")
    
    def create_wallet(self, name: str, wallet_type: str) -> str:
        """
        Create a new wallet.
        
        Args:
            name: Wallet name
            wallet_type: Type of wallet (queen, princess, worker, savings)
            
        Returns:
            Wallet ID
        """
        # Generate new keypair
        keypair = Keypair()
        private_key = base64.b64encode(keypair.seed).decode('ascii')
        public_key = str(keypair.public_key)
        
        # Generate unique ID
        wallet_id = str(uuid.uuid4())
        
        # Create wallet object
        wallet = {
            'id': wallet_id,
            'name': name,
            'type': wallet_type,
            'public_key': public_key,
            'private_key': private_key,
            'created_at': time.time()
        }
        
        # Store in memory
        self.wallets[wallet_id] = wallet
        
        # Save to disk (encrypted)
        self._save_wallet(wallet_id)
        
        return wallet_id
    
    def _save_wallet(self, wallet_id: str):
        """
        Save wallet to disk with encrypted private key.
        
        Args:
            wallet_id: ID of wallet to save
        """
        if wallet_id not in self.wallets:
            raise ValueError(f"Wallet {wallet_id} not found")
        
        wallet = self.wallets[wallet_id].copy()
        
        # Encrypt private key
        if 'private_key' in wallet:
            wallet['encrypted_private_key'] = self.fernet.encrypt(
                wallet['private_key'].encode()
            ).decode()
            del wallet['private_key']  # Don't save unencrypted key
        
        # Save to file
        wallet_path = self.wallets_dir / f"{wallet_id}.json"
        with open(wallet_path, 'w') as f:
            json.dump(wallet, f, indent=2)
    
    def list_wallets(self, wallet_type: Optional[str] = None) -> List[Dict]:
        """
        List all wallets, optionally filtered by type.
        
        Args:
            wallet_type: Optional filter by wallet type
            
        Returns:
            List of wallet dictionaries (without private keys)
        """
        result = []
        
        for wallet_id, wallet in self.wallets.items():
            if wallet_type and wallet.get('type') != wallet_type:
                continue
                
            # Create a copy without private key
            safe_wallet = wallet.copy()
            if 'private_key' in safe_wallet:
                del safe_wallet['private_key']
            
            result.append(safe_wallet)
            
        return result
    
    def get_wallet_by_name(self, name: str) -> Optional[str]:
        """
        Find wallet ID by name.
        
        Args:
            name: Wallet name to look for
            
        Returns:
            Wallet ID if found, None otherwise
        """
        for wallet_id, wallet in self.wallets.items():
            if wallet.get('name') == name:
                return wallet_id
        
        return None
    
    def get_balance(self, wallet_id: str) -> float:
        """
        Get balance of a wallet in SOL.
        
        Args:
            wallet_id: Wallet ID
            
        Returns:
            Balance in SOL
        """
        if wallet_id not in self.wallets:
            raise ValueError(f"Wallet {wallet_id} not found")
        
        public_key = self.wallets[wallet_id]['public_key']
        
        try:
            response = self.client.get_balance(PublicKey(public_key))
            lamports = response['result']['value']
            sol = lamports / 1_000_000_000  # Convert from lamports to SOL
            return sol
            
        except Exception as e:
            loguru.logger.error(f"Error getting balance for wallet {wallet_id}: {str(e)}")
            raise
    
    def transfer_sol(self, from_id: str, to_id: str, amount: float) -> str:
        """
        Transfer SOL between two managed wallets.
        
        Args:
            from_id: Source wallet ID
            to_id: Destination wallet ID
            amount: Amount of SOL to transfer
            
        Returns:
            Transaction signature
        """
        if from_id not in self.wallets:
            raise ValueError(f"Source wallet {from_id} not found")
        
        if to_id not in self.wallets:
            raise ValueError(f"Destination wallet {to_id} not found")
            
        sender_keypair = self._get_keypair(from_id)
        recipient_pubkey = PublicKey(self.wallets[to_id]['public_key'])
        
        return self._execute_transfer(sender_keypair, recipient_pubkey, amount)
    
    def transfer_sol_to_external(self, from_id: str, to_address: str, amount: float) -> str:
        """
        Transfer SOL to an external wallet.
        
        Args:
            from_id: Source wallet ID
            to_address: Destination wallet public key
            amount: Amount of SOL to transfer
            
        Returns:
            Transaction signature
        """
        if from_id not in self.wallets:
            raise ValueError(f"Source wallet {from_id} not found")
            
        sender_keypair = self._get_keypair(from_id)
        recipient_pubkey = PublicKey(to_address)
        
        return self._execute_transfer(sender_keypair, recipient_pubkey, amount)
    
    def _get_keypair(self, wallet_id: str) -> Keypair:
        """
        Get Solana keypair for a wallet.
        
        Args:
            wallet_id: Wallet ID
            
        Returns:
            Solana Keypair object
        """
        if wallet_id not in self.wallets:
            raise ValueError(f"Wallet {wallet_id} not found")
            
        if 'private_key' not in self.wallets[wallet_id]:
            raise ValueError(f"Private key not available for wallet {wallet_id}")
        
        private_key = base64.b64decode(self.wallets[wallet_id]['private_key'])
        return Keypair.from_seed(private_key)
    
    def _execute_transfer(self, sender_keypair: Keypair, recipient_pubkey: PublicKey, 
                          amount: float) -> str:
        """
        Execute a SOL transfer.
        
        Args:
            sender_keypair: Sender's Keypair object
            recipient_pubkey: Recipient's PublicKey object
            amount: Amount in SOL to transfer
            
        Returns:
            Transaction signature
        """
        try:
            lamports = int(amount * 1_000_000_000)  # Convert SOL to lamports
            
            # Create transfer instruction
            transfer_params = TransferParams(
                from_pubkey=sender_keypair.public_key,
                to_pubkey=recipient_pubkey,
                lamports=lamports
            )
            instruction = transfer(transfer_params)
            
            # Create and sign transaction
            transaction = Transaction().add(instruction)
            response = self.client.send_transaction(
                transaction, sender_keypair
            )
            
            signature = response['result']
            return signature
            
        except Exception as e:
            loguru.logger.error(f"Error executing transfer: {str(e)}")
            raise
    
    def create_backup(self, backup_path: Optional[str] = None) -> str:
        """
        Create an encrypted backup of all wallets.
        
        Args:
            backup_path: Optional path for backup file
            
        Returns:
            Path to the created backup file
        """
        # Use default path if not specified
        if not backup_path:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            backup_path = BACKUPS_DIR / f'wallet_backup_{timestamp}.json'
        else:
            backup_path = Path(backup_path)
        
        # Ensure the backup directory exists
        backup_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Create backup data
        backup_data = []
        for wallet_id, wallet in self.wallets.items():
            wallet_backup = wallet.copy()
            
            # Include the encrypted private key
            if 'private_key' in wallet:
                wallet_backup['encrypted_private_key'] = self.fernet.encrypt(
                    wallet['private_key'].encode()
                ).decode()
                del wallet_backup['private_key']
                
            backup_data.append(wallet_backup)
        
        # Write the encrypted backup
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        return str(backup_path)
    
    def restore_from_backup(self, backup_path: str) -> int:
        """
        Restore wallets from an encrypted backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Number of wallets restored
        """
        try:
            backup_path = Path(backup_path)
            with open(backup_path, 'r') as f:
                backup_data = json.load(f)
            
            count = 0
            for wallet_data in backup_data:
                wallet_id = wallet_data.get('id')
                if not wallet_id:
                    continue
                    
                # Decrypt the private key if present
                if 'encrypted_private_key' in wallet_data:
                    try:
                        wallet_data['private_key'] = self.fernet.decrypt(
                            wallet_data['encrypted_private_key'].encode()
                        ).decode()
                    except Exception as e:
                        loguru.logger.error(f"Error decrypting wallet {wallet_id}: {str(e)}")
                        continue
                
                # Store the wallet
                self.wallets[wallet_id] = wallet_data
                self._save_wallet(wallet_id)
                count += 1
            
            return count
            
        except Exception as e:
            loguru.logger.error(f"Error restoring from backup: {str(e)}")
            raise
