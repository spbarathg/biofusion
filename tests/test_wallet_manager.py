import unittest
import os
import tempfile
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
import base64

# Import the module to test
from src.core.wallet_manager import WalletManager
from solana.keypair import Keypair
from solana.publickey import PublicKey

class TestWalletManager(unittest.TestCase):
    """Test cases for the WalletManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory structure
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temp_dir.name)
        
        # Create directory structure
        self.wallets_dir = self.test_root / "wallets"
        self.config_dir = self.test_root / "config"
        self.backups_dir = self.test_root / "backups"
        
        # Create the directories
        self.wallets_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        self.backups_dir.mkdir(exist_ok=True)
        
        # Create a test config file
        self.config_path = self.config_dir / "settings.yaml"
        with open(self.config_path, 'w') as f:
            f.write("# Test config file")
        
        # Create a test encryption key
        self.key_path = self.config_dir / "encryption.key"
        
        # Set up patches for the paths module
        self.paths_patch = patch.multiple(
            'src.core.wallet_manager',
            WALLETS_DIR=self.wallets_dir,
            BACKUPS_DIR=self.backups_dir,
            ENCRYPTION_KEY_PATH=self.key_path,
            CONFIG_PATH=self.config_path
        )
        self.paths_patch.start()
        
        # Set up a mock for the Solana client
        self.client_patch = patch('src.core.wallet_manager.SolanaClient')
        self.mock_client = self.client_patch.start()
        
        # Configure mock client
        self.mock_client_instance = MagicMock()
        self.mock_client.return_value = self.mock_client_instance
        
        # Set up balance response
        self.mock_client_instance.get_balance = MagicMock(return_value={
            'result': {'value': 1000000000}  # 1 SOL = 10^9 lamports
        })
        
        # Initialize the wallet manager
        self.wallet_manager = WalletManager(str(self.config_path))
    
    def tearDown(self):
        """Tear down test fixtures"""
        # Stop all patches
        self.paths_patch.stop()
        self.client_patch.stop()
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_create_wallet(self):
        """Test wallet creation"""
        # Create a wallet
        wallet_id = self.wallet_manager.create_wallet(
            name="Test Wallet",
            wallet_type="queen"
        )
        
        # Verify wallet was created
        self.assertIsNotNone(wallet_id)
        self.assertIn(wallet_id, self.wallet_manager.wallets)
        
        # Verify wallet properties
        wallet = self.wallet_manager.wallets[wallet_id]
        self.assertEqual(wallet['name'], "Test Wallet")
        self.assertEqual(wallet['type'], "queen")
        self.assertIn('public_key', wallet)
        self.assertIn('private_key', wallet)
        
        # Verify wallet file was created
        wallet_file = self.wallets_dir / f"{wallet_id}.json"
        self.assertTrue(wallet_file.exists())
        
        # Verify file contents
        with open(wallet_file, 'r') as f:
            wallet_data = json.load(f)
        
        self.assertEqual(wallet_data['name'], "Test Wallet")
        self.assertEqual(wallet_data['type'], "queen")
        self.assertIn('public_key', wallet_data)
        self.assertIn('encrypted_private_key', wallet_data)
        self.assertNotIn('private_key', wallet_data)  # Private key should be encrypted
    
    def test_list_wallets(self):
        """Test listing wallets"""
        # Create a few wallets of different types
        queen_id = self.wallet_manager.create_wallet("Queen", "queen")
        worker1_id = self.wallet_manager.create_wallet("Worker 1", "worker")
        worker2_id = self.wallet_manager.create_wallet("Worker 2", "worker")
        princess_id = self.wallet_manager.create_wallet("Princess", "princess")
        
        # List all wallets
        all_wallets = self.wallet_manager.list_wallets()
        self.assertEqual(len(all_wallets), 4)
        
        # Check that private keys are not included
        for wallet in all_wallets:
            self.assertNotIn('private_key', wallet)
        
        # List by type
        worker_wallets = self.wallet_manager.list_wallets(wallet_type="worker")
        self.assertEqual(len(worker_wallets), 2)
        for wallet in worker_wallets:
            self.assertEqual(wallet['type'], "worker")
    
    def test_get_wallet_by_name(self):
        """Test finding a wallet by name"""
        # Create a wallet
        wallet_id = self.wallet_manager.create_wallet(
            name="Unique Name",
            wallet_type="queen"
        )
        
        # Find by name
        found_id = self.wallet_manager.get_wallet_by_name("Unique Name")
        self.assertEqual(found_id, wallet_id)
        
        # Try with non-existent name
        not_found = self.wallet_manager.get_wallet_by_name("Does Not Exist")
        self.assertIsNone(not_found)
    
    @patch('src.core.wallet_manager.Keypair')
    async def test_get_balance(self, mock_keypair):
        """Test getting a wallet balance"""
        # Create a wallet with known keypair
        public_key = "DYw8jCTfwHNRJhhmFcbXvVDTqWMEVFBX6ZKUmG5CNSKK"
        # Mock keypair
        mock_keypair_instance = MagicMock()
        mock_keypair_instance.public_key = PublicKey(public_key)
        mock_keypair.return_value = mock_keypair_instance
        
        # Create wallet
        wallet_id = self.wallet_manager.create_wallet(
            name="Balance Test",
            wallet_type="worker"
        )
        
        # Get balance
        balance = await self.wallet_manager.get_balance(wallet_id)
        
        # Verify balance calculation (1 SOL = 10^9 lamports)
        self.assertEqual(balance, 1.0)
        
        # Verify client method was called
        self.mock_client_instance.get_balance.assert_called()
    
    @patch('src.core.wallet_manager.transfer')
    @patch('src.core.wallet_manager.Transaction')
    async def test_transfer_sol(self, mock_transaction, mock_transfer):
        """Test transferring SOL between wallets"""
        # Mock the transaction
        mock_txn = MagicMock()
        mock_transaction.return_value = mock_txn
        
        # Mock transfer instruction
        mock_transfer_instruction = MagicMock()
        mock_transfer.return_value = mock_transfer_instruction
        
        # Mock transaction result
        self.mock_client_instance.send_transaction.return_value = {
            'result': 'test_signature'
        }
        
        # Create sender and recipient wallets
        sender_id = self.wallet_manager.create_wallet("Sender", "queen")
        recipient_id = self.wallet_manager.create_wallet("Recipient", "worker")
        
        # Transfer SOL
        signature = await self.wallet_manager.transfer_sol(
            from_id=sender_id,
            to_id=recipient_id,
            amount=0.5
        )
        
        # Verify signature
        self.assertEqual(signature, 'test_signature')
        
        # Verify transaction was sent
        self.mock_client_instance.send_transaction.assert_called_once()
        
        # Verify transfer instruction was added
        mock_txn.add.assert_called_once_with(mock_transfer_instruction)
    
    async def test_create_and_restore_backup(self):
        """Test creating and restoring wallet backups"""
        # Create a few wallets
        wallet_ids = []
        for i in range(3):
            wallet_id = self.wallet_manager.create_wallet(
                name=f"Backup Test {i}",
                wallet_type="worker"
            )
            wallet_ids.append(wallet_id)
        
        # Create a backup
        backup_path = await self.wallet_manager.create_backup()
        self.assertTrue(os.path.exists(backup_path))
        
        # Create a new wallet manager instance
        with patch('src.core.wallet_manager.SolanaClient'):
            # Clear wallets directory first
            for file in self.wallets_dir.glob('*.json'):
                file.unlink()
                
            # Create new instance
            new_manager = WalletManager(str(self.config_path))
            self.assertEqual(len(new_manager.wallets), 0)
            
            # Restore from backup
            restored_count = await new_manager.restore_from_backup(backup_path)
            
            # Verify all wallets were restored
            self.assertEqual(restored_count, 3)
            self.assertEqual(len(new_manager.wallets), 3)
            
            # Verify wallet IDs match
            for wallet_id in wallet_ids:
                self.assertIn(wallet_id, new_manager.wallets)

if __name__ == '__main__':
    # Run with asyncio support
    loop = asyncio.get_event_loop()
    loop.run_until_complete(unittest.main()) 