"""
Solana Compatibility Layer
==========================

This module provides compatibility for solana imports when the main solana package
is not available (e.g., on Python 3.13).
"""

import asyncio
import json
import base58
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# Mock classes for compatibility
class Commitment(Enum):
    PROCESSED = "processed"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"

class Confirmed(Enum):
    CONFIRMED = "confirmed"

class Finalized(Enum):
    FINALIZED = "finalized"

class Processed(Enum):
    PROCESSED = "processed"

@dataclass
class Keypair:
    """Mock Keypair class"""
    public_key: str
    secret_key: bytes
    
    @classmethod
    def from_secret_key(cls, secret_key: bytes):
        """Create keypair from secret key"""
        return cls(public_key="mock_public_key", secret_key=secret_key)
    
    @classmethod
    def generate(cls):
        """Generate new keypair"""
        return cls(public_key="mock_public_key", secret_key=b"mock_secret")

@dataclass
class PublicKey:
    """Mock PublicKey class"""
    public_key: str
    
    def __init__(self, public_key: str):
        self.public_key = public_key
    
    def __str__(self):
        return self.public_key

@dataclass
class TransferParams:
    """Mock TransferParams class"""
    from_pubkey: PublicKey
    to_pubkey: PublicKey
    lamports: int

def transfer(params: TransferParams) -> Dict[str, Any]:
    """Mock transfer function"""
    return {"signature": "mock_signature"}

class Transaction:
    """Mock Transaction class"""
    def __init__(self):
        self.instructions = []
        self.signatures = []
    
    def add(self, instruction):
        """Add instruction to transaction"""
        self.instructions.append(instruction)
    
    def sign(self, *keypairs):
        """Sign transaction"""
        for keypair in keypairs:
            self.signatures.append(f"mock_signature_{keypair.public_key}")

class AsyncClient:
    """Mock AsyncClient class"""
    def __init__(self, endpoint: str = "https://api.mainnet-beta.solana.com"):
        self.endpoint = endpoint
        self.connected = False
    
    async def connect(self):
        """Connect to Solana network"""
        self.connected = True
        return {"result": "connected"}
    
    async def disconnect(self):
        """Disconnect from Solana network"""
        self.connected = False
        return {"result": "disconnected"}
    
    async def get_balance(self, public_key: str, commitment: Commitment = Commitment.CONFIRMED):
        """Get balance for public key"""
        return {"result": {"value": 1000000}}  # Mock balance
    
    async def get_account_info(self, public_key: str, commitment: Commitment = Commitment.CONFIRMED):
        """Get account info"""
        return {"result": {"value": {"data": "mock_data"}}}
    
    async def send_transaction(self, transaction: Transaction, *keypairs):
        """Send transaction"""
        return {"result": "mock_signature"}
    
    async def confirm_transaction(self, signature: str, commitment: Commitment = Commitment.CONFIRMED):
        """Confirm transaction"""
        return {"result": {"value": True}}

class Client:
    """Mock synchronous Client class"""
    def __init__(self, endpoint: str = "https://api.mainnet-beta.solana.com"):
        self.endpoint = endpoint
    
    def get_balance(self, public_key: str, commitment: Commitment = Commitment.CONFIRMED):
        """Get balance for public key"""
        return {"result": {"value": 1000000}}  # Mock balance

# Mock RPC types
class TxOpts:
    """Mock transaction options"""
    def __init__(self, **kwargs):
        self.skip_preflight = kwargs.get('skip_preflight', False)
        self.preflight_commitment = kwargs.get('preflight_commitment', Commitment.CONFIRMED)

# Export all the mock classes
__all__ = [
    'AsyncClient', 'Client', 'Keypair', 'PublicKey', 'Transaction',
    'TransferParams', 'transfer', 'Commitment', 'Confirmed', 'Finalized',
    'Processed', 'TxOpts'
] 