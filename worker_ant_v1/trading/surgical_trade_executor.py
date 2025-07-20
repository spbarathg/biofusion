"""
SURGICAL TRADE EXECUTOR - REAL DEX INTEGRATION
=============================================

Executes trades with surgical precision using Jupiter DEX integration.
Enhanced with Devil's Advocate Synapse for pre-mortem analysis.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from datetime import datetime
import time

from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.trading.devils_advocate_synapse import DevilsAdvocateSynapse, PreMortemAnalysis

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
    
    # Devil's Advocate analysis results
    pre_mortem_analysis: Optional[PreMortemAnalysis] = None
    veto_issued: bool = False
    veto_reasons: List[str] = None

class SurgicalTradeExecutor:
    """Surgical trade executor with Jupiter DEX integration and Devil's Advocate pre-mortem analysis"""
    
    def __init__(self):
        self.logger = setup_logger("SurgicalTradeExecutor")
        
        # Jupiter API endpoints
        self.jupiter_api = "https://quote-api.jup.ag/v6"
        self.jupiter_swap_api = "https://quote-api.jup.ag/v6/swap"
        
        # Core components
        self.rpc_client = None
        self.wallet_manager = None
        self.devils_advocate = DevilsAdvocateSynapse()  # Pre-mortem analysis system
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.vetoed_trades = 0
        self.total_volume_sol = 0.0
        self.avg_execution_time_ms = 0.0
        
    async def initialize(self, rpc_client, wallet_manager):
        """Initialize trade executor with Devil's Advocate Synapse"""
        self.rpc_client = rpc_client
        self.wallet_manager = wallet_manager
        
        # Initialize Devil's Advocate Synapse
        await self.devils_advocate.initialize()
        
        self.logger.info("✅ Surgical trade executor initialized with Devil's Advocate protection")
    
    async def prepare_trade(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare trade parameters for execution with pre-mortem analysis"""
        try:
            prepared_trade = {
                'token_address': trade_params.get('token_address', ''),
                'amount': float(trade_params.get('amount', 0.0)),
                'wallet': trade_params.get('wallet', ''),
                'order_type': trade_params.get('order_type', 'buy'),
                'max_slippage': float(trade_params.get('max_slippage', 2.0)),
                'prepared_at': datetime.now().isoformat(),
                'status': 'prepared'
            }
            
            # Validate parameters
            if not prepared_trade['token_address']:
                raise ValueError("Token address is required")
            if prepared_trade['amount'] <= 0:
                raise ValueError("Amount must be positive")
            if not prepared_trade['wallet']:
                raise ValueError("Wallet is required")
            
            # Conduct Devil's Advocate pre-mortem analysis
            self.logger.info(f"🕵️ Running pre-mortem analysis for {prepared_trade['order_type']} {prepared_trade['amount']} SOL")
            pre_mortem_analysis = await self.devils_advocate.conduct_pre_mortem_analysis(trade_params)
            
            # Check if trade should be vetoed
            if pre_mortem_analysis.veto_recommended:
                prepared_trade.update({
                    'status': 'vetoed',
                    'veto_reasons': [reason.value for reason in pre_mortem_analysis.veto_reasons],
                    'failure_probability': pre_mortem_analysis.overall_failure_probability,
                    'pre_mortem_analysis': pre_mortem_analysis
                })
                self.vetoed_trades += 1
                return prepared_trade
            
            # Trade cleared pre-mortem analysis
            prepared_trade['pre_mortem_analysis'] = pre_mortem_analysis
            prepared_trade['devils_advocate_cleared'] = True
            
            self.logger.info(f"📋 Trade prepared and cleared: {prepared_trade['order_type']} {prepared_trade['amount']} SOL | Risk: {pre_mortem_analysis.overall_failure_probability:.1%}")
            return prepared_trade
            
        except Exception as e:
            self.logger.error(f"Error preparing trade: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
        
    async def execute_buy(
        self,
        token_address: str,
        amount_sol: float,
        wallet: str,
        max_slippage: float = 2.0,
        trade_params: Dict[str, Any] = None
    ) -> ExecutionResult:
        """Execute a buy order with surgical precision and Devil's Advocate protection"""
        
        start_time = time.time()
        result = ExecutionResult(success=False)
        
        try:
            self.logger.info(f"🔵 Executing buy: {amount_sol} SOL -> {token_address[:8]}...")
            
            # Prepare trade parameters for Devil's Advocate analysis
            if not trade_params:
                trade_params = {
                    'token_address': token_address,
                    'amount': amount_sol,
                    'wallet_id': wallet,
                    'order_type': 'buy',
                    'max_slippage': max_slippage
                }
            
            # Conduct pre-mortem analysis if not already done
            if 'pre_mortem_analysis' not in trade_params:
                self.logger.info("🕵️ Conducting final pre-mortem analysis...")
                pre_mortem_analysis = await self.devils_advocate.conduct_pre_mortem_analysis(trade_params)
                
                # Check for veto
                if pre_mortem_analysis.veto_recommended:
                    result.pre_mortem_analysis = pre_mortem_analysis
                    result.veto_issued = True
                    result.veto_reasons = [reason.value for reason in pre_mortem_analysis.veto_reasons]
                    result.error = f"Trade vetoed by Devil's Advocate: {', '.join(result.veto_reasons)}"
                    
                    self.vetoed_trades += 1
                    self.logger.warning(f"🚫 Trade execution halted by Devil's Advocate veto")
                    return result
                
                trade_params['pre_mortem_analysis'] = pre_mortem_analysis
            else:
                pre_mortem_analysis = trade_params['pre_mortem_analysis']
            
            # Proceed with trade execution - Devil's Advocate has cleared the trade
            result.pre_mortem_analysis = pre_mortem_analysis
            
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
                
                self.logger.info(f"✅ Buy executed with Devil's Advocate approval: {amount_sol} SOL -> {result.amount_tokens} tokens | Risk assessed: {pre_mortem_analysis.overall_failure_probability:.1%}")
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
            try:
                from solana.transaction import Transaction
                from solana.rpc.commitment import Confirmed
            except ImportError:
                from ..utils.solana_compat import Transaction, Confirmed
            
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
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get comprehensive execution status including Devil's Advocate metrics"""
        total_attempts = self.total_trades + self.vetoed_trades
        
        return {
            'total_trade_attempts': total_attempts,
            'executed_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'vetoed_trades': self.vetoed_trades,
            'execution_success_rate': self.successful_trades / max(self.total_trades, 1),
            'devils_advocate_veto_rate': self.vetoed_trades / max(total_attempts, 1),
            'total_volume_sol': self.total_volume_sol,
            'avg_execution_time_ms': self.avg_execution_time_ms,
            'devils_advocate_status': self.devils_advocate.get_synapse_status(),
            'protection_active': True,
            'surgical_precision_mode': True
        } 