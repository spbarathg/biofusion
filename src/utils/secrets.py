"""
Secret Management for AntBot

This module provides utilities for managing secrets securely.
It supports environment variables with a fallback to HashiCorp Vault 
for production environments.
"""

import os
import logging
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Import dotenv for development environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional Vault integration for production
try:
    import hvac
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False

logger = logging.getLogger(__name__)

class SecretManager:
    """Manages secrets for the AntBot system"""
    
    def __init__(self, environment: str = None):
        """
        Initialize the secret manager
        
        Args:
            environment: The deployment environment (dev, staging, prod)
        """
        self.environment = environment or os.getenv("APP_ENV", "dev")
        self.vault_client = None
        
        # Initialize Vault client if configured for production
        if self.environment == "prod" and VAULT_AVAILABLE:
            self._init_vault()
    
    def _init_vault(self) -> None:
        """Initialize the Vault client if credentials are available"""
        vault_addr = os.getenv("VAULT_ADDR")
        vault_token = os.getenv("VAULT_TOKEN")
        
        if vault_addr and vault_token:
            try:
                self.vault_client = hvac.Client(url=vault_addr, token=vault_token)
                if not self.vault_client.is_authenticated():
                    logger.warning("Vault client failed to authenticate")
                    self.vault_client = None
                else:
                    logger.info("Vault client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Vault client: {e}")
                self.vault_client = None
    
    def get_secret(self, key: str, default: Any = None) -> Any:
        """
        Get a secret value with fallback mechanism
        
        Priority:
        1. Environment variable
        2. HashiCorp Vault (prod only)
        3. Default value
        
        Args:
            key: Secret key name
            default: Default value if not found
            
        Returns:
            The secret value or default
        """
        # Try environment variable first (standard format or prefixed)
        env_value = os.getenv(key) or os.getenv(f"ANTBOT_{key.upper()}")
        if env_value:
            return env_value
        
        # Try Vault for production
        if self.environment == "prod" and self.vault_client is not None:
            try:
                secret_path = f"secret/antbot/{key}"
                secret = self.vault_client.secrets.kv.v2.read_secret_version(path=secret_path)
                if secret and "data" in secret and "data" in secret["data"]:
                    return secret["data"]["data"].get(key)
            except Exception as e:
                logger.error(f"Error retrieving secret from Vault: {e}")
        
        return default
    
    def get_encryption_key(self) -> bytes:
        """
        Get the encryption key for securing sensitive data
        
        Returns:
            Bytes representation of the encryption key
        """
        # Try to get from environment
        key_str = self.get_secret("ENCRYPTION_KEY")
        
        # If not in environment, try file
        if not key_str:
            key_path = Path(os.getenv("ENCRYPTION_KEY_PATH", "config/secrets/.encryption_key"))
            if key_path.exists():
                try:
                    with open(key_path, "r") as f:
                        key_str = f.read().strip()
                except Exception as e:
                    logger.error(f"Failed to read encryption key from file: {e}")
        
        # If we have a key string, decode it
        if key_str:
            try:
                # Assume base64 encoding for the key
                return base64.b64decode(key_str)
            except Exception as e:
                logger.error(f"Failed to decode encryption key: {e}")
        
        # No valid key found
        raise ValueError("No valid encryption key found. Please set ENCRYPTION_KEY environment variable or create key file.")
    
    def generate_encryption_key(self) -> bytes:
        """
        Generate a new encryption key
        
        Returns:
            Newly generated encryption key as bytes
        """
        import secrets
        key = secrets.token_bytes(32)  # 256-bit key
        
        # Save to file if in development mode
        if self.environment in ("dev", "test"):
            key_path = Path(os.getenv("ENCRYPTION_KEY_PATH", "config/secrets/.encryption_key"))
            key_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(key_path, "wb") as f:
                f.write(key)
            
            logger.info(f"New encryption key generated and saved to {key_path}")
        
        return key
    
    def rotate_encryption_key(self, old_key: bytes = None) -> bytes:
        """
        Generate a new encryption key, potentially re-encrypting existing data
        
        Args:
            old_key: The previous encryption key for re-encryption
            
        Returns:
            The new encryption key
        """
        # Generate new key
        new_key = self.generate_encryption_key()
        
        # Here you would implement logic to re-encrypt all data
        # with the new key if old_key is provided
        
        return new_key

# Global instance
secret_manager = SecretManager()

def get_secret(key: str, default: Any = None) -> Any:
    """Convenience function to get a secret"""
    return secret_manager.get_secret(key, default)

def get_encryption_key() -> bytes:
    """Convenience function to get the encryption key"""
    return secret_manager.get_encryption_key() 