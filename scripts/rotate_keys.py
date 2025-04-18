#!/usr/bin/env python3
"""
Key Rotation Script for AntBot

This script provides a secure way to rotate encryption keys:
- Generates a new encryption key
- Re-encrypts all wallet data with the new key
- Updates key storage securely
- Creates a backup of the old data

Usage:
    python rotate_keys.py [--force] [--dry-run] [--backup-dir PATH]
"""

import os
import sys
import json
import time
import base64
import shutil
import argparse
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from src.utils.secrets import SecretManager, get_encryption_key
from src.core.paths import WALLETS_DIR, BACKUPS_DIR, DATA_DIR
from src.utils.logging.logger import setup_logging

# Import metrics utilities
try:
    from src.utils.monitoring.metrics import update_key_rotation_metrics
    metrics_available = True
except ImportError:
    metrics_available = False

# Set up logging
logger = setup_logging("key_rotation", "key_rotation.log")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Rotate encryption keys for AntBot")
    parser.add_argument("--force", action="store_true", help="Force rotation without confirmation")
    parser.add_argument("--dry-run", action="store_true", help="Simulate rotation without making changes")
    parser.add_argument("--backup-dir", type=str, help="Directory to store backup of old data")
    parser.add_argument("--verification", action="store_true", help="Verify all files after rotation")
    return parser.parse_args()

def backup_current_data(backup_dir: Optional[Path] = None) -> Path:
    """
    Create a backup of current wallet data
    
    Args:
        backup_dir: Optional custom backup directory
        
    Returns:
        Path to backup directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if backup_dir is None:
        backup_dir = BACKUPS_DIR / f"key_rotation_{timestamp}"
    
    try:
        # Create backup directory
        backup_dir.mkdir(exist_ok=True, parents=True)
        
        # Backup wallets directory
        backup_wallets_dir = backup_dir / "wallets"
        backup_wallets_dir.mkdir(exist_ok=True)
        
        for wallet_file in WALLETS_DIR.glob("**/*"):
            if wallet_file.is_file():
                rel_path = wallet_file.relative_to(WALLETS_DIR)
                dest_path = backup_wallets_dir / rel_path
                dest_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(wallet_file, dest_path)
        
        # Backup encryption key
        key_path = DATA_DIR / ".encryption_key"
        if key_path.exists():
            shutil.copy2(key_path, backup_dir / ".encryption_key")
        
        logger.info(f"Backup created at {backup_dir}")
        return backup_dir
        
    except Exception as e:
        logger.error(f"Backup failed: {str(e)}")
        raise

def generate_new_key() -> bytes:
    """
    Generate a new encryption key
    
    Returns:
        New encryption key as bytes
    """
    try:
        # Create a new SecretManager instance
        secret_manager = SecretManager()
        
        # Generate a new key (don't save it to the default location yet)
        new_key = secret_manager.generate_encryption_key()
        
        logger.info("New encryption key generated")
        return new_key
        
    except Exception as e:
        logger.error(f"Key generation failed: {str(e)}")
        raise

def decrypt_wallet_file(wallet_path: Path, old_key: bytes) -> Dict[str, Any]:
    """
    Decrypt a wallet file with the old key
    
    Args:
        wallet_path: Path to the wallet file
        old_key: Old encryption key
        
    Returns:
        Decrypted wallet data
    """
    try:
        from cryptography.fernet import Fernet
        f = Fernet(old_key)
        
        with open(wallet_path, "rb") as file:
            encrypted_data = file.read()
            decrypted_data = f.decrypt(encrypted_data)
            
        return json.loads(decrypted_data.decode())
        
    except Exception as e:
        logger.error(f"Failed to decrypt {wallet_path}: {str(e)}")
        raise

def encrypt_wallet_file(wallet_data: Dict[str, Any], wallet_path: Path, new_key: bytes) -> None:
    """
    Encrypt wallet data with the new key and save it
    
    Args:
        wallet_data: Wallet data to encrypt
        wallet_path: Target path to save encrypted wallet
        new_key: New encryption key
    """
    try:
        from cryptography.fernet import Fernet
        f = Fernet(new_key)
        
        # Create a temporary file first
        fd, temp_path = tempfile.mkstemp(dir=wallet_path.parent)
        try:
            json_data = json.dumps(wallet_data).encode()
            encrypted_data = f.encrypt(json_data)
            
            with os.fdopen(fd, "wb") as file:
                file.write(encrypted_data)
                
            # Atomic replace of original file
            os.replace(temp_path, wallet_path)
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
            
    except Exception as e:
        logger.error(f"Failed to encrypt {wallet_path}: {str(e)}")
        raise

def install_new_key(new_key: bytes) -> None:
    """
    Install the new encryption key
    
    Args:
        new_key: New encryption key to install
    """
    try:
        key_path = DATA_DIR / ".encryption_key"
        
        # Create a temporary file first
        fd, temp_path = tempfile.mkstemp(dir=key_path.parent)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(new_key)
                
            # Atomic replace of original file
            os.replace(temp_path, key_path)
            
            logger.info(f"New encryption key installed at {key_path}")
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
            
    except Exception as e:
        logger.error(f"Failed to install new key: {str(e)}")
        raise

def verify_encryption(wallet_path: Path, new_key: bytes) -> bool:
    """
    Verify that a wallet file can be decrypted with the new key
    
    Args:
        wallet_path: Path to the wallet file
        new_key: New encryption key
        
    Returns:
        True if verification succeeds, False otherwise
    """
    try:
        from cryptography.fernet import Fernet
        f = Fernet(new_key)
        
        with open(wallet_path, "rb") as file:
            encrypted_data = file.read()
            
        # Attempt to decrypt
        decrypted_data = f.decrypt(encrypted_data)
        json.loads(decrypted_data.decode())  # Ensure it's valid JSON
        
        return True
        
    except Exception as e:
        logger.error(f"Verification failed for {wallet_path}: {str(e)}")
        return False

def rotate_keys(args):
    """Main key rotation process"""
    start_time = time.time()
    success = False
    
    try:
        logger.info("Starting encryption key rotation")
        
        # Get confirmation unless --force is used
        if not args.force:
            confirm = input("This will rotate all encryption keys. Are you sure? (y/n): ")
            if confirm.lower() != "y":
                logger.info("Key rotation cancelled by user")
                return
        
        # Create backup
        backup_dir = backup_current_data(
            Path(args.backup_dir) if args.backup_dir else None
        )
        
        # Get the old key
        old_key = get_encryption_key()
        
        # Generate new key
        new_key = generate_new_key()
        
        # In dry run mode, stop here
        if args.dry_run:
            logger.info("Dry run - stopping before re-encryption")
            return
        
        # Process all wallet files
        wallet_count = 0
        wallet_files = list(WALLETS_DIR.glob("**/*.json"))
        
        logger.info(f"Found {len(wallet_files)} wallet files to process")
        
        for wallet_path in wallet_files:
            try:
                # Decrypt with old key
                wallet_data = decrypt_wallet_file(wallet_path, old_key)
                
                # Re-encrypt with new key
                encrypt_wallet_file(wallet_data, wallet_path, new_key)
                
                wallet_count += 1
                logger.info(f"Re-encrypted {wallet_path}")
                
            except Exception as e:
                logger.error(f"Failed to process {wallet_path}: {str(e)}")
                # Continue to next file on error
        
        # Install the new key
        install_new_key(new_key)
        
        # Verify all files if requested
        if args.verification:
            verification_failures = 0
            for wallet_path in wallet_files:
                if not verify_encryption(wallet_path, new_key):
                    verification_failures += 1
            
            if verification_failures > 0:
                logger.error(f"{verification_failures} files failed verification")
                logger.error(f"Backup location: {backup_dir}")
                success = False
                sys.exit(1)
        
        logger.info(f"Key rotation completed successfully. {wallet_count} wallets re-encrypted.")
        logger.info(f"Backup location: {backup_dir}")
        success = True
        
    except Exception as e:
        logger.error(f"Key rotation failed: {str(e)}")
        logger.error("Please restore from backup or contact system administrator")
        success = False
        sys.exit(1)
    finally:
        # Record metrics if available
        duration = time.time() - start_time
        if metrics_available:
            try:
                update_key_rotation_metrics(success, duration)
                logger.info("Key rotation metrics updated")
            except Exception as e:
                logger.error(f"Failed to update metrics: {str(e)}")

def main():
    args = parse_args()
    rotate_keys(args)

if __name__ == "__main__":
    main() 