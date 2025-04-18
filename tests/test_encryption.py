#!/usr/bin/env python3
import os
import sys
from cryptography.fernet import Fernet
from pathlib import Path

# Add src to the path so we can import modules from there
sys.path.append(os.path.abspath("."))

# Import the WalletManager to test it
from src.core.wallet_manager import WalletManager

def test_encryption_key():
    """Test that the encryption key can be loaded properly"""
    # Path to the encryption key
    key_path = os.environ.get("ENCRYPTION_KEY_PATH", "config/secrets/.encryption_key")
    
    print(f"Checking for encryption key at: {key_path}")
    
    # Check if the key file exists
    if not os.path.exists(key_path):
        print(f"Error: Encryption key file not found at {key_path}")
        return False
    
    try:
        # Try to load the key and verify it's a valid Fernet key
        with open(key_path, "rb") as f:
            key_data = f.read()
            
        # Attempt to create a Fernet instance with the key
        fernet = Fernet(key_data)
        test_data = b"Test encryption"
        encrypted = fernet.encrypt(test_data)
        decrypted = fernet.decrypt(encrypted)
        
        if decrypted == test_data:
            print("Encryption key test passed! Key is valid.")
            return True
        else:
            print("Encryption test failed: Data mismatch after decryption")
            return False
            
    except Exception as e:
        print(f"Error testing encryption key: {str(e)}")
        return False

def test_wallet_manager():
    """Test that the WalletManager can initialize properly"""
    try:
        # Set the encryption key path environment variable
        os.environ["ENCRYPTION_KEY_PATH"] = "config/secrets/.encryption_key"
        
        # Create the wallet manager
        wallet_manager = WalletManager()
        print("WalletManager initialization successful!")
        return True
    except Exception as e:
        print(f"Error initializing WalletManager: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Testing Encryption Key ===")
    key_result = test_encryption_key()
    
    print("\n=== Testing WalletManager Initialization ===")
    wallet_result = test_wallet_manager()
    
    if key_result and wallet_result:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nOne or more tests failed.")
        sys.exit(1) 