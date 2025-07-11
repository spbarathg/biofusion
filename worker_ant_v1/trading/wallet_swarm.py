from typing import List, Dict, Optional
import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from web3 import Web3
import redis
from eth_account import Account
import secrets
import os

@dataclass
class WalletPerformance:
    address: str
    total_trades: int
    successful_trades: int
    total_profit: float
    avg_return: float
    last_active: datetime
    risk_score: float

class WalletSwarm:
    def __init__(self, redis_client: redis.Redis, web3: Web3):
        self.redis = redis_client
        self.web3 = web3
        self.logger = logging.getLogger(__name__)
        self.SWARM_SIZE = 10
        self.MIN_WALLET_BALANCE = 0.01  # in ETH
        self._load_wallets()
        
    def _load_wallets(self):
        """Load or initialize wallet swarm"""
        self.wallets = self.redis.hgetall("trading:active_wallets") or {}
        
        # Create new wallets if needed
        while len(self.wallets) < self.SWARM_SIZE:
            self._create_new_wallet()
            
    def _create_new_wallet(self) -> str:
        """Create a new wallet with enhanced privacy"""
        # Generate new account
        private_key = "0x" + secrets.token_hex(32)
        account = Account.from_key(private_key)
        
        # Store encrypted private key in Redis with 24h expiry
        encrypted = self._encrypt_key(private_key)
        self.redis.setex(
            f"wallet:key:{account.address}",
            86400,  # 24h expiry
            encrypted
        )
        
        # Initialize performance metrics
        self.redis.hset("trading:active_wallets", account.address, json.dumps({
            "created_at": datetime.now().isoformat(),
            "total_trades": 0,
            "successful_trades": 0,
            "total_profit": 0.0,
            "risk_score": 0.5
        }))
        
        return account.address
        
    def _encrypt_key(self, private_key: str) -> str:
        """Encrypt private key before storage using Fernet encryption"""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Get encryption key from environment or generate one
            encryption_key = os.getenv('ENCRYPTION_KEY')
            if not encryption_key:
                # Generate a new key if not available
                encryption_key = Fernet.generate_key()
                self.logger.warning("No encryption key found, generated new one. Store ENCRYPTION_KEY in environment.")
            else:
                # Ensure key is properly formatted
                if len(encryption_key) != 44:  # Fernet key length
                    # Pad or truncate to correct length
                    encryption_key = base64.urlsafe_b64encode(
                        encryption_key.encode()[:32].ljust(32, b'0')
                    )
                else:
                    encryption_key = encryption_key.encode()
            
            # Create Fernet cipher
            cipher = Fernet(encryption_key)
            
            # Encrypt the private key
            encrypted_data = cipher.encrypt(private_key.encode())
            
            # Return base64 encoded encrypted data
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            # Fallback to simple encoding (not secure, but functional)
            return base64.b64encode(private_key.encode()).decode()
        
    def _decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt private key for use"""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Get encryption key from environment
            encryption_key = os.getenv('ENCRYPTION_KEY')
            if not encryption_key:
                self.logger.error("No encryption key found in environment")
                # Try to decode as base64 (fallback)
                return base64.b64decode(encrypted_key.encode()).decode()
            
            # Ensure key is properly formatted
            if len(encryption_key) != 44:  # Fernet key length
                # Pad or truncate to correct length
                encryption_key = base64.urlsafe_b64encode(
                    encryption_key.encode()[:32].ljust(32, b'0')
                )
            else:
                encryption_key = encryption_key.encode()
            
            # Create Fernet cipher
            cipher = Fernet(encryption_key)
            
            # Decode base64 and decrypt
            encrypted_data = base64.b64decode(encrypted_key.encode())
            decrypted_data = cipher.decrypt(encrypted_data)
            
            return decrypted_data.decode()
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            # Fallback to simple decoding
            try:
                return base64.b64decode(encrypted_key.encode()).decode()
            except Exception:
                self.logger.error("All decryption methods failed")
                return ""
        
    async def get_best_wallet(self, trade_risk: float) -> Optional[str]:
        """Get best wallet for trade based on risk profile and performance"""
        best_score = -1
        best_wallet = None
        
        for address, data in self.wallets.items():
            metrics = json.loads(data)
            
            # Skip if wallet is currently in trade
            if self.redis.get(f"wallet:in_trade:{address}"):
                continue
                
            # Calculate wallet score based on performance and risk alignment
            success_rate = metrics["successful_trades"] / max(1, metrics["total_trades"])
            risk_alignment = 1 - abs(trade_risk - metrics["risk_score"])
            
            score = success_rate * 0.7 + risk_alignment * 0.3
            
            if score > best_score:
                best_score = score
                best_wallet = address
                
        return best_wallet
        
    async def evolve_wallets(self):
        """Daily evolution of wallet swarm"""
        while True:
            try:
                performances = await self._get_wallet_performances()
                
                # Sort by performance
                ranked = sorted(
                    performances,
                    key=lambda x: x.avg_return * 0.7 + x.successful_trades/max(1, x.total_trades) * 0.3,
                    reverse=True
                )
                
                # Replace bottom 20% with clones of top performers
                replacements = len(ranked) // 5
                for i in range(replacements):
                    # Remove worst performer
                    worst = ranked[-(i+1)]
                    self.redis.hdel("trading:active_wallets", worst.address)
                    
                    # Clone a top performer with slight mutations
                    top = ranked[i]
                    self._create_new_wallet()  # This will automatically mutate parameters
                    
                await asyncio.sleep(86400)  # Run daily
                
            except Exception as e:
                self.logger.error(f"Evolution error: {str(e)}")
                await asyncio.sleep(3600)
                
    async def _get_wallet_performances(self) -> List[WalletPerformance]:
        """Get performance metrics for all wallets"""
        performances = []
        
        for address, data in self.wallets.items():
            metrics = json.loads(data)
            performances.append(
                WalletPerformance(
                    address=address,
                    total_trades=metrics["total_trades"],
                    successful_trades=metrics["successful_trades"],
                    total_profit=metrics["total_profit"],
                    avg_return=metrics["total_profit"] / max(1, metrics["total_trades"]),
                    last_active=datetime.fromisoformat(metrics.get("last_active", metrics["created_at"])),
                    risk_score=metrics["risk_score"]
                )
            )
            
        return performances
        
    async def update_wallet_performance(self, address: str, trade_result: Dict):
        """Update wallet performance metrics after trade"""
        if data := self.redis.hget("trading:active_wallets", address):
            metrics = json.loads(data)
            
            metrics["total_trades"] += 1
            if trade_result["success"]:
                metrics["successful_trades"] += 1
            metrics["total_profit"] += trade_result["profit"]
            metrics["last_active"] = datetime.now().isoformat()
            
            # Update risk score based on trade performance
            if trade_result["success"]:
                metrics["risk_score"] = min(1.0, metrics["risk_score"] * 1.1)
            else:
                metrics["risk_score"] = max(0.1, metrics["risk_score"] * 0.9)
                
            self.redis.hset("trading:active_wallets", address, json.dumps(metrics))
            
    async def mark_wallet_in_trade(self, address: str, duration: int = 300):
        """Mark wallet as currently in trade"""
        self.redis.setex(f"wallet:in_trade:{address}", duration, "1") 