"""
SURGICAL TRADE EXECUTOR - REAL DEX INTEGRATION
=============================================

Executes trades with surgical precision using Jupiter DEX integration.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import time

from worker_ant_v1.utils.logger import setup_logger

@dataclass
class ExecutionResult:
    success: bool
    signature: Optional[str] = None
    amount_sol: float = 0.0
    amount_tokens: float = 0.0
    price: float = 0.0
    slippage_percent: float = 0.0
    latency_ms: int = 0
    error: Optional[str] = None

class SurgicalTradeExecutor:
    """Surgical trade executor with Jupiter DEX integration"""
    
    def __init__(self):
        self.logger = setup_logger("SurgicalTradeExecutor")
        
        # Jupiter API endpoints
        self.jupiter_api = "https://quote-api.jup.ag/v6"
        self.jupiter_swap_api = "https://quote-api.jup.ag/v6/swap"
        
        # RPC client (will be set during initialization)
        self.rpc_client = None
        self.wallet_manager = None
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_volume_sol = 0.0
        self.avg_execution_time_ms = 0.0
        
    async def initialize(self, rpc_client, wallet_manager):
        """Initialize trade executor"""
        self.rpc_client = rpc_client
        self.wallet_manager = wallet_manager
        self.logger.info("✅ Surgical trade executor initialized")
        
    async def execute_buy(
        self,
        token_address: str,
        amount_sol: float,
        wallet: str,
        max_slippage: float = 2.0
    ) -> ExecutionResult:
        """Execute a buy order with surgical precision"""
        
        start_time = time.time()
        result = ExecutionResult(success=False)
        
        try:
            self.logger.info(f"🔵 Executing buy: {amount_sol} SOL -> {token_address[:8]}...")
            
            # Get wallet keypair
            wallet_keypair = await self.wallet_manager.get_wallet_keypair(wallet)
            if not wallet_keypair:
                result.error = "Wallet not found"
                return result
                
            # Get quote from Jupiter
            quote = await self._get_jupiter_quote(
                input_mint="So11111111111111111111111111111111111111112",  # SOL
                output_mint=token_address,
                amount_sol=amount_sol,
                slippage_bps=int(max_slippage * 100)
            )
            
            if not quote:
                result.error = "Failed to get quote"
                return result
                
            # Execute swap
            swap_result = await self._execute_jupiter_swap(
                quote=quote,
                wallet_keypair=wallet_keypair
            )
            
            if swap_result:
                result.success = True
                result.signature = swap_result['signature']
                result.amount_sol = amount_sol
                result.amount_tokens = float(quote['outAmount'])
                result.price = amount_sol / float(quote['outAmount']) if float(quote['outAmount']) > 0 else 0
                result.slippage_percent = max_slippage
                
                # Update metrics
                self.total_trades += 1
                self.successful_trades += 1
                self.total_volume_sol += amount_sol
                
                self.logger.info(f"✅ Buy executed: {amount_sol} SOL -> {result.amount_tokens} tokens")
            else:
                result.error = "Swap execution failed"
                
        except Exception as e:
            result.error = str(e)
            self.failed_trades += 1
            self.logger.error(f"❌ Buy execution failed: {e}")
            
        finally:
            result.latency_ms = int((time.time() - start_time) * 1000)
            self._update_execution_metrics(result.latency_ms)
            
        return result
        
    async def execute_sell(
        self,
        token_address: str,
        amount_tokens: float,
        wallet: str,
        max_slippage: float = 2.0
    ) -> ExecutionResult:
        """Execute a sell order with surgical precision"""
        
        start_time = time.time()
        result = ExecutionResult(success=False)
        
        try:
            self.logger.info(f"🔴 Executing sell: {amount_tokens} tokens -> SOL...")
            
            # Get wallet keypair
            wallet_keypair = await self.wallet_manager.get_wallet_keypair(wallet)
            if not wallet_keypair:
                result.error = "Wallet not found"
                return result
                
            # Get quote from Jupiter
            quote = await self._get_jupiter_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111111112",  # SOL
                amount_tokens=amount_tokens,
                slippage_bps=int(max_slippage * 100)
            )
            
            if not quote:
                result.error = "Failed to get quote"
                return result
                
            # Execute swap
            swap_result = await self._execute_jupiter_swap(
                quote=quote,
                wallet_keypair=wallet_keypair
            )
            
            if swap_result:
                result.success = True
                result.signature = swap_result['signature']
                result.amount_sol = float(quote['outAmount']) / 1e9  # Convert from lamports
                result.amount_tokens = amount_tokens
                result.price = result.amount_sol / amount_tokens if amount_tokens > 0 else 0
                result.slippage_percent = max_slippage
                
                # Update metrics
                self.total_trades += 1
                self.successful_trades += 1
                self.total_volume_sol += result.amount_sol
                
                self.logger.info(f"✅ Sell executed: {amount_tokens} tokens -> {result.amount_sol} SOL")
            else:
                result.error = "Swap execution failed"
                
        except Exception as e:
            result.error = str(e)
            self.failed_trades += 1
            self.logger.error(f"❌ Sell execution failed: {e}")
            
        finally:
            result.latency_ms = int((time.time() - start_time) * 1000)
            self._update_execution_metrics(result.latency_ms)
            
        return result
        
    async def _get_jupiter_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount_sol: float = None,
        amount_tokens: float = None,
        slippage_bps: int = 50
    ) -> Optional[Dict]:
        """Get quote from Jupiter API"""
        try:
            # Determine amount and decimals
            if amount_sol:
                amount = str(int(amount_sol * 1e9))  # Convert SOL to lamports
            elif amount_tokens:
                amount = str(int(amount_tokens))  # Assume 0 decimals for tokens
            else:
                return None
                
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": amount,
                "slippageBps": str(slippage_bps),
                "onlyDirectRoutes": "false",
                "asLegacyTransaction": "false"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.jupiter_api}/quote", params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'data' in data:
                            return data['data']
                    else:
                        self.logger.warning(f"Jupiter quote API error: {resp.status}")
                        
        except Exception as e:
            self.logger.error(f"Quote fetch error: {e}")
            
        return None
        
    async def _execute_jupiter_swap(
        self,
        quote: Dict,
        wallet_keypair: Any
    ) -> Optional[Dict]:
        """Execute swap using Jupiter API"""
        try:
            # Get swap transaction
            swap_data = {
                "quoteResponse": quote,
                "userPublicKey": str(wallet_keypair.pubkey()),
                "wrapUnwrapSOL": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.jupiter_swap_api}", json=swap_data) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if 'swapTransaction' in data:
                            # Decode transaction
                            tx_data = data['swapTransaction']
                            
                            # Send transaction
                            signature = await self._send_transaction(tx_data, wallet_keypair)
                            if signature:
                                return {"signature": signature}
                    else:
                        self.logger.warning(f"Jupiter swap API error: {resp.status}")
                        
        except Exception as e:
            self.logger.error(f"Swap execution error: {e}")
            
        return None
        
    async def _send_transaction(self, tx_data: str, wallet_keypair: Any) -> Optional[str]:
        """Send transaction to Solana network"""
        try:
            from solana.transaction import Transaction
            from solana.rpc.commitment import Confirmed
            
            # Deserialize transaction
            transaction = Transaction.deserialize(bytes(tx_data))
            
            # Sign transaction
            transaction.sign(wallet_keypair)
            
            # Send transaction
            result = await self.rpc_client.send_transaction(
                transaction,
                opts={"skip_preflight": True, "preflight_commitment": Confirmed}
            )
            
            # Wait for confirmation
            await self.rpc_client.confirm_transaction(
                result.value,
                commitment=Confirmed
            )
            
            return str(result.value)
            
        except Exception as e:
            self.logger.error(f"Transaction send error: {e}")
            return None
            
    def _update_execution_metrics(self, latency_ms: int):
        """Update execution performance metrics"""
        if self.avg_execution_time_ms == 0:
            self.avg_execution_time_ms = latency_ms
        else:
            alpha = 0.1  # Smoothing factor
            self.avg_execution_time_ms = (
                alpha * latency_ms + (1 - alpha) * self.avg_execution_time_ms
            )
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "total_trades": self.total_trades,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "success_rate": (self.successful_trades / max(1, self.total_trades)) * 100,
            "total_volume_sol": self.total_volume_sol,
            "avg_execution_time_ms": self.avg_execution_time_ms
        } 