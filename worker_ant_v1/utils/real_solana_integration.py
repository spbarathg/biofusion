"""
Production Solana Integration
===========================

Real Solana blockchain integration with RPC endpoints and transaction handling.
"""

import os
import json
import base58
import asyncio
from typing import Dict, List, Optional, Union
from solana.rpc.async_api import AsyncClient
from solana.transaction import Transaction
from solana.system_program import TransactionInstruction
from solders.pubkey import Pubkey
from solders.signature import Signature

class ProductionSolanaClient:
    """Production-ready Solana client with failover and monitoring"""
    
    def __init__(self):
        self.primary_endpoint = os.getenv('SOLANA_RPC_URL')
        self.backup_endpoints = [
            os.getenv('SOLANA_RPC_BACKUP_1'),
            os.getenv('SOLANA_RPC_BACKUP_2')
        ]
        
        self.current_endpoint = self.primary_endpoint
        self.client = AsyncClient(self.current_endpoint)
        self.health_check_interval = 60  # seconds
        self.last_error_time = None
        self.error_count = 0
        
    async def initialize(self):
        """Initialize the client and verify connection"""
        try:
            await self.client.is_connected()
            print(f"✅ Connected to Solana RPC: {self.current_endpoint}")
        except Exception as e:
            print(f"❌ Failed to connect to primary RPC: {e}")
            await self._try_backup_endpoints()
    
    async def _try_backup_endpoints(self):
        """Attempt to connect to backup endpoints"""
        for endpoint in self.backup_endpoints:
            if not endpoint:
                continue
                
            try:
                self.client = AsyncClient(endpoint)
                await self.client.is_connected()
                self.current_endpoint = endpoint
                print(f"✅ Connected to backup RPC: {endpoint}")
                return
            except Exception as e:
                print(f"❌ Failed to connect to backup RPC {endpoint}: {e}")
        
        raise Exception("❌ All RPC endpoints failed")
    
    async def get_balance(self, pubkey: Union[str, Pubkey]) -> float:
        """Get SOL balance for an address"""
        try:
            if isinstance(pubkey, str):
                pubkey = Pubkey.from_string(pubkey)
            
            response = await self.client.get_balance(pubkey)
            if response.value is not None:
                return float(response.value) / 1e9  # Convert lamports to SOL
            return 0.0
            
        except Exception as e:
            print(f"❌ Failed to get balance: {e}")
            await self._handle_error()
            return 0.0
    
    async def send_transaction(
        self,
        transaction: Transaction,
        signers: List[str],
        max_retries: int = 3
    ) -> Optional[str]:
        """Send a transaction with retry logic"""
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Sign transaction
                for signer in signers:
                    transaction.sign(signer)
                
                # Send transaction
                result = await self.client.send_transaction(
                    transaction,
                    opts={"skip_preflight": False, "max_retries": 3}
                )
                
                if result.value:
                    signature = str(result.value)
                    print(f"✅ Transaction sent: {signature}")
                    return signature
                    
                print(f"❌ Transaction failed: {result.error}")
                return None
                
            except Exception as e:
                print(f"❌ Transaction error (attempt {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                await asyncio.sleep(1)
                
                if retry_count == max_retries:
                    await self._handle_error()
                    return None
    
    async def confirm_transaction(self, signature: Union[str, Signature]) -> bool:
        """Wait for transaction confirmation"""
        try:
            if isinstance(signature, str):
                signature = Signature.from_string(signature)
            
            result = await self.client.confirm_transaction(signature)
            return result.value
            
        except Exception as e:
            print(f"❌ Failed to confirm transaction: {e}")
            await self._handle_error()
            return False
    
    async def get_token_accounts(self, owner: Union[str, Pubkey]) -> List[Dict]:
        """Get all token accounts for an owner"""
        try:
            if isinstance(owner, str):
                owner = Pubkey.from_string(owner)
            
            response = await self.client.get_token_accounts_by_owner(
                owner,
                {"programId": Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")}
            )
            
            if response.value:
                return [
                    {
                        'pubkey': str(account.pubkey),
                        'mint': str(account.account.data.parsed['info']['mint']),
                        'amount': int(account.account.data.parsed['info']['tokenAmount']['amount'])
                    }
                    for account in response.value
                ]
            return []
            
        except Exception as e:
            print(f"❌ Failed to get token accounts: {e}")
            await self._handle_error()
            return []
    
    async def _handle_error(self):
        """Handle RPC errors and attempt recovery"""
        self.error_count += 1
        
        if self.error_count >= 3:
            print("🔄 Too many errors, attempting to switch RPC endpoint")
            await self._try_backup_endpoints()
            self.error_count = 0 