"""
Mock Solana module for testing and development.
This module provides mock implementations of Solana classes and functions.
"""

class GetBalanceResp:
    """Mock GetBalanceResp class for Solana balance responses."""
    
    def __init__(self, value=0):
        """Initialize a mock balance response."""
        self.value = value
        
    def __getitem__(self, key):
        """Allow dictionary-style access to match real Solana response."""
        if key == 'result':
            return {'value': self.value}
        raise KeyError(key)

class Keypair:
    """Mock Keypair class for Solana transactions."""
    
    def __init__(self, secret_key=None):
        """Initialize a mock keypair."""
        self.secret_key = secret_key or b'0' * 64
        self.public_key = b'0' * 32
        self.seed = secret_key or b'0' * 64
        
    @classmethod
    def from_secret_key(cls, secret_key):
        """Create a keypair from a secret key."""
        return cls(secret_key)
        
    def sign(self, message):
        """Sign a message."""
        return b'0' * 64

class PublicKey:
    """Mock PublicKey class for Solana addresses."""
    
    def __init__(self, value=None):
        """Initialize a mock public key."""
        self.value = value or b'0' * 32
        
    def __str__(self):
        """Return a string representation of the public key."""
        return "MockPublicKey"

class SystemProgram:
    """Mock SystemProgram class for Solana system instructions."""
    
    @staticmethod
    def transfer():
        """Return a mock transfer instruction."""
        return "MockTransferInstruction"

class Transaction:
    """Mock Transaction class for Solana transactions."""
    
    def __init__(self):
        """Initialize a mock transaction."""
        self.instructions = []
        
    def add(self, instruction):
        """Add an instruction to the transaction."""
        self.instructions.append(instruction)
        return self 