"""
Buyer Module for Worker Ant V1
==============================

Handles ultra-fast buy execution with proper slippage control and safety checks.
"""

import asyncio
import time
from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    import solana
    from solana.rpc.async_api import AsyncClient
    from solana.rpc.commitment import Confirmed
    from solana.transaction import Transaction
    from solders.keypair import Keypair  # Updated import path
    from solders.pubkey import Pubkey as PublicKey  # Updated import path
    from spl.token.constants import TOKEN_PROGRAM_ID
    from spl.token.instructions import get_associated_token_address
    SOLANA_AVAILABLE = True
except ImportError:
    # Fallback for systems without Solana SDK
    SOLANA_AVAILABLE = False
    print("Warning: Solana SDK not available - using simulation mode")

from worker_ant_v1.config import trading_config, wallet_config
from worker_ant_v1.logger import trading_logger, TradeResult
from worker_ant_v1.scanner import TokenOpportunity


@dataclass
class BuyResult:
    """Result of a buy operation"""
    
    success: bool
    signature: Optional[str] = None
    error_message: Optional[str] = None
    
    # Trade details
    amount_sol: float = 0.0
    amount_tokens: float = 0.0
    price: float = 0.0
    slippage_percent: float = 0.0
    latency_ms: int = 0
    
    # On-chain details
    tx_fee: float = 0.0
    block_height: Optional[int] = None


class TradeBuyer:
    """Fast trade execution engine"""
    
    def __init__(self):
        self.client: Optional[AsyncClient] = None
        self.wallet: Optional[Keypair] = None
        self.jupiter_url = "https://quote-api.jup.ag/v6"
        self.setup_complete = False
        
    async def setup(self):
        """Initialize the buyer with wallet and RPC connection"""
        
        # Setup Solana RPC client
        self.client = AsyncClient(
            trading_config.rpc_url,
            commitment=Confirmed,
            timeout=trading_config.rpc_timeout_seconds
        )
        
        # Load or create wallet
        await self._setup_wallet()
        
        # Test connection
        try:
            response = await self.client.get_health()
            if response.value != "ok":
                raise Exception("RPC connection unhealthy")
            trading_logger.logger.info("RPC connection established")
        except Exception as e:
            trading_logger.logger.error(f"RPC connection failed: {e}")
            raise
            
        self.setup_complete = True
        trading_logger.logger.info("Buyer setup complete")
        
    async def _setup_wallet(self):
        """Load or create trading wallet"""
        
        if wallet_config.wallet_private_key:
            # Load from private key
            try:
                # Assuming base58 encoded private key
                import base58
                private_key_bytes = base58.b58decode(wallet_config.wallet_private_key)
                self.wallet = Keypair.from_secret_key(private_key_bytes)
                trading_logger.logger.info(f"Loaded wallet: {self.wallet.public_key}")
            except Exception as e:
                trading_logger.logger.error(f"Failed to load wallet from private key: {e}")
                raise
        else:
            # Create new wallet
            self.wallet = Keypair()
            trading_logger.logger.warning(f"Created new wallet: {self.wallet.public_key}")
            trading_logger.logger.warning("Fund this wallet with SOL before trading!")
            
        # Check wallet balance
        try:
            balance = await self.get_sol_balance()
            trading_logger.logger.info(f"Wallet balance: {balance:.4f} SOL")
            
            if balance < trading_config.trade_amount_sol:
                trading_logger.logger.warning(
                    f"Low wallet balance: {balance:.4f} SOL < {trading_config.trade_amount_sol:.4f} SOL"
                )
        except Exception as e:
            trading_logger.logger.error(f"Failed to check wallet balance: {e}")
            
    async def get_sol_balance(self) -> float:
        """Get current SOL balance"""
        
        if not self.client or not self.wallet:
            return 0.0
            
        try:
            response = await self.client.get_balance(self.wallet.public_key)
            return response.value / 1e9  # Convert lamports to SOL
        except Exception as e:
            trading_logger.logger.error(f"Error getting balance: {e}")
            return 0.0
            
    async def execute_buy(self, opportunity: TokenOpportunity) -> BuyResult:
        """Execute a buy trade for the given opportunity"""
        
        if not self.setup_complete:
            return BuyResult(
                success=False,
                error_message="Buyer not properly setup"
            )
            
        start_time = time.time()
        
        try:
            # Pre-trade safety checks
            if not await self._pre_trade_checks(opportunity):
                return BuyResult(
                    success=False,
                    error_message="Pre-trade safety checks failed"
                )
                
            # Get quote from Jupiter
            quote = await self._get_jupiter_quote(
                from_token="So11111111111111111111111111111111111112",  # SOL
                to_token=opportunity.token_address,
                amount_sol=trading_config.trade_amount_sol
            )
            
            if not quote:
                return BuyResult(
                    success=False,
                    error_message="Failed to get price quote"
                )
                
            # Check slippage
            expected_slippage = quote.get('priceImpactPct', 0)
            if expected_slippage > trading_config.max_slippage_percent:
                return BuyResult(
                    success=False,
                    error_message=f"Slippage too high: {expected_slippage:.2f}%"
                )
                
            # Execute the swap
            swap_result = await self._execute_jupiter_swap(quote)
            
            # Calculate metrics
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            if swap_result['success']:
                # Calculate trade details
                amount_tokens = float(quote.get('outAmount', 0)) / (10 ** 6)  # Assuming 6 decimals
                price = trading_config.trade_amount_sol / amount_tokens if amount_tokens > 0 else 0
                
                result = BuyResult(
                    success=True,
                    signature=swap_result['signature'],
                    amount_sol=trading_config.trade_amount_sol,
                    amount_tokens=amount_tokens,
                    price=price,
                    slippage_percent=expected_slippage,
                    latency_ms=latency_ms,
                    tx_fee=swap_result.get('fee', 0)
                )
                
                # Log the trade
                trade_record = TradeResult(
                    timestamp=datetime.utcnow().isoformat(),
                    token_address=opportunity.token_address,
                    token_symbol=opportunity.token_symbol,
                    trade_type='BUY',
                    amount_sol=result.amount_sol,
                    amount_tokens=result.amount_tokens,
                    price=result.price,
                    slippage_percent=result.slippage_percent,
                    latency_ms=result.latency_ms,
                    tx_signature=result.signature,
                    success=True
                )
                
                trading_logger.log_trade(trade_record)
                trading_logger.update_last_trade_time()
                
                return result
                
            else:
                return BuyResult(
                    success=False,
                    error_message=swap_result.get('error', 'Unknown swap error'),
                    latency_ms=latency_ms
                )
                
        except Exception as e:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            error_msg = f"Buy execution error: {str(e)}"
            trading_logger.logger.error(error_msg)
            
            # Log failed trade
            trade_record = TradeResult(
                timestamp=datetime.utcnow().isoformat(),
                token_address=opportunity.token_address,
                token_symbol=opportunity.token_symbol,
                trade_type='BUY',
                                    amount_sol=trading_config.trade_amount_sol,
                amount_tokens=0,
                price=0,
                slippage_percent=0,
                latency_ms=latency_ms,
                tx_signature="",
                success=False,
                error_message=error_msg
            )
            
            trading_logger.log_trade(trade_record)
            
            return BuyResult(
                success=False,
                error_message=error_msg,
                latency_ms=latency_ms
            )
            
    async def _pre_trade_checks(self, opportunity: TokenOpportunity) -> bool:
        """Run safety checks before executing trade"""
        
        # Check safety limits
        if not trading_logger.check_safety_limits():
            trading_logger.logger.warning("Safety limits exceeded, skipping trade")
            return False
            
        # Check wallet balance
        balance = await self.get_sol_balance()
        if balance < trading_config.trade_amount_sol * 1.1:  # 10% buffer for fees
            trading_logger.logger.warning(f"Insufficient balance: {balance:.4f} SOL")
            return False
            
        # Check if token is blacklisted
        from worker_ant_v1.config import scanner_config
        if opportunity.token_address in scanner_config.blacklisted_tokens:
            trading_logger.logger.warning(f"Token blacklisted: {opportunity.token_address}")
            return False
            
        # Check liquidity
        if opportunity.liquidity_sol < trading_config.min_liquidity_sol:
            trading_logger.logger.warning(f"Insufficient liquidity: {opportunity.liquidity_sol:.2f} SOL")
            return False
            
        return True
        
    async def _get_jupiter_quote(self, from_token: str, to_token: str, amount_sol: float) -> Optional[dict]:
        """Get price quote from Jupiter aggregator"""
        
        try:
            import aiohttp
            
            # Convert SOL amount to lamports
            amount_lamports = int(amount_sol * 1e9)
            
            url = f"{self.jupiter_url}/quote"
            params = {
                'inputMint': from_token,
                'outputMint': to_token,
                'amount': amount_lamports,
                'slippageBps': int(trading_config.max_slippage_percent * 100),  # Convert to basis points
                'onlyDirectRoutes': True,  # Faster execution
                'asLegacyTransaction': False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        trading_logger.logger.error(f"Jupiter quote error: {resp.status}")
                        return None
                        
        except Exception as e:
            trading_logger.logger.error(f"Error getting Jupiter quote: {e}")
            return None
            
    async def _execute_jupiter_swap(self, quote: dict) -> dict:
        """Execute the swap using Jupiter"""
        
        try:
            import aiohttp
            import json
            
            # Get swap transaction from Jupiter
            swap_url = f"{self.jupiter_url}/swap"
            swap_data = {
                'quoteResponse': quote,
                'userPublicKey': str(self.wallet.public_key),
                'wrapAndUnwrapSol': True,
                'useSharedAccounts': True,
                'feeAccount': None,  # No fee account for MVP
                'computeUnitPriceMicroLamports': 'auto'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    swap_url, 
                    json=swap_data,
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status != 200:
                        return {
                            'success': False,
                            'error': f'Jupiter swap API error: {resp.status}'
                        }
                        
                    swap_response = await resp.json()
                    
            # Get the transaction
            swap_transaction = swap_response.get('swapTransaction')
            if not swap_transaction:
                return {
                    'success': False,
                    'error': 'No swap transaction returned'
                }
                
            # Deserialize and sign transaction
            import base64
            from solana.transaction import Transaction
            
            tx_bytes = base64.b64decode(swap_transaction)
            transaction = Transaction.deserialize(tx_bytes)
            
            # Sign the transaction
            transaction.sign(self.wallet)
            
            # Submit transaction with high priority
            try:
                response = await self.client.send_transaction(
                    transaction,
                    opts=solana.rpc.types.TxOpts(
                        skip_preflight=False,
                        preflight_commitment=Confirmed,
                        max_retries=3
                    )
                )
                
                if response.value:
                    # Wait for confirmation (with timeout)
                    confirmed = await self._wait_for_confirmation(response.value, timeout=30)
                    
                    if confirmed:
                        return {
                            'success': True,
                            'signature': response.value,
                            'fee': 0.000005  # Estimate
                        }
                    else:
                        return {
                            'success': False,
                            'error': 'Transaction confirmation timeout'
                        }
                else:
                    return {
                        'success': False,
                        'error': 'Transaction submission failed'
                    }
                    
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Transaction submission error: {str(e)}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Swap execution error: {str(e)}'
            }
            
    async def _wait_for_confirmation(self, signature: str, timeout: int = 30) -> bool:
        """Wait for transaction confirmation"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = await self.client.get_signature_statuses([signature])
                
                if response.value and response.value[0]:
                    status = response.value[0]
                    
                    if status.confirmation_status in ['confirmed', 'finalized']:
                        return True
                    elif status.err:
                        trading_logger.logger.error(f"Transaction failed: {status.err}")
                        return False
                        
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                trading_logger.logger.error(f"Error checking confirmation: {e}")
                await asyncio.sleep(2)
                
        return False
        
    async def close(self):
        """Clean up resources"""
        if self.client:
            await self.client.close()


# Global buyer instance
trade_buyer = TradeBuyer() 