#!/usr/bin/env python3
"""
Script to repair wallet files that are in an incorrect format.
"""

import os
import sys
import json
import uuid
import base64
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cryptography.fernet import Fernet
from loguru import logger

# Import paths
from src.core.paths import WALLETS_DIR, ENCRYPTION_KEY_PATH

def repair_wallet_file(wallet_path: Path, fernet: Fernet) -> bool:
    """
    Repair a single wallet file.
    
    Args:
        wallet_path: Path to the wallet file
        fernet: Fernet instance for encryption/decryption
        
    Returns:
        bool: True if repair was successful
    """
    try:
        # Read the wallet file
        with open(wallet_path, 'r') as f:
            wallet_data = json.load(f)
        
        # Check if file needs repair
        needs_repair = False
        
        # Add missing id field
        if 'id' not in wallet_data:
            wallet_data['id'] = str(uuid.uuid4())
            needs_repair = True
        
        # Fix wallet type field
        if 'wallet_type' in wallet_data and 'type' not in wallet_data:
            wallet_data['type'] = wallet_data.pop('wallet_type')
            needs_repair = True
        
        # Fix public key format
        if 'public_key' in wallet_data:
            try:
                # Try to decode and re-encode to ensure proper format
                decoded = base64.b64decode(wallet_data['public_key'])
                wallet_data['public_key'] = base64.b64encode(decoded).decode('ascii')
                needs_repair = True
            except:
                pass
        
        # Fix encrypted private key format
        if 'encrypted_private_key' in wallet_data:
            try:
                # Try to decrypt and re-encrypt to ensure proper format
                decrypted = fernet.decrypt(wallet_data['encrypted_private_key'].encode())
                wallet_data['encrypted_private_key'] = fernet.encrypt(decrypted).decode()
                needs_repair = True
            except:
                pass
        
        # Save repaired wallet if needed
        if needs_repair:
            with open(wallet_path, 'w') as f:
                json.dump(wallet_data, f, indent=2)
            logger.info(f"Repaired wallet file: {wallet_path.name}")
            return True
        
        return True
        
    except Exception as e:
        logger.error(f"Error repairing wallet {wallet_path.name}: {str(e)}")
        return False

def main():
    """Main function to repair all wallet files."""
    try:
        # Create wallets directory if it doesn't exist
        WALLETS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load or create encryption key
        if ENCRYPTION_KEY_PATH.exists():
            with open(ENCRYPTION_KEY_PATH, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            ENCRYPTION_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ENCRYPTION_KEY_PATH, 'wb') as f:
                f.write(key)
        
        fernet = Fernet(key)
        
        # Get all wallet files
        wallet_files = list(WALLETS_DIR.glob('*.json'))
        
        if not wallet_files:
            logger.info("No wallet files found to repair")
            return
        
        # Repair each wallet file
        repaired = 0
        for wallet_path in wallet_files:
            if repair_wallet_file(wallet_path, fernet):
                repaired += 1
        
        logger.info(f"Repaired {repaired} out of {len(wallet_files)} wallet files")
        
    except Exception as e:
        logger.error(f"Error repairing wallets: {str(e)}")

if __name__ == '__main__':
    main() 