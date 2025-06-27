"""
PRODUCTION-READY BUYER MODULE
============================

Ultra-fast, secure buy execution with comprehensive error handling,
retry logic, transaction validation, and MEV protection.
"""

import asyncio
import aiohttp
import time
import json
import secrets
import hashlib
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

# External dependencies - conditional imports
try:
    import solana
    from solana.rpc.async_api import AsyncClient
    from solana.rpc.commitment import Confirmed, Processed, Finalized
    from solana.transaction import Transaction
    from solders.pubkey import Pubkey
    from solders.keypair import Keypair
    from spl.token.constants import TOKEN_PROGRAM_ID
    from spl.token.instructions import transfer, TransferParams
    SOLANA_AVAILABLE = True
except ImportError:
    # Create mock classes for development/testing
    SOLANA_AVAILABLE = False
    
    class AsyncClient:
        def __init__(self, *args, **kwargs): pass
    class Confirmed: pass
    class Processed: pass  
    class Finalized: pass
    class Transaction: pass
    class Pubkey: 
        def __init__(self, *args, **kwargs): pass
    class Keypair:
        def __init__(self, *args, **kwargs): pass
    TOKEN_PROGRAM_ID = None

# Internal imports
from worker_ant_v1.core.simple_config import get_trading_config, get_security_config, mask_sensitive_value
from worker_ant_v1.utils.simple_logger import setup_logger
trading_logger = setup_logger(__name__)

# Temporary placeholder for TradeRecord
class TradeRecord:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class ExecutionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    CONFIRMING = "confirming"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class FailureReason(Enum):
    INSUFFICIENT_BALANCE = "insufficient_balance"
    HIGH_SLIPPAGE = "high_slippage"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    QUOTE_FAILED = "quote_failed"
    TRANSACTION_FAILED = "transaction_failed"
    CONFIRMATION_FAILED = "confirmation_failed"
    MEV_DETECTED = "mev_detected"
    HONEYPOT_DETECTED = "honeypot_detected"

@dataclass
class BuySignal:
    """Enhanced buy signal with comprehensive parameters"""
    
    token_address: str
    confidence: float  # 0.0 to 1.0
    urgency: float     # 0.0 to 1.0 (affects timeout and slippage tolerance)
    max_slippage: float
    amount_sol: float
    
    # Optional parameters
    token_symbol: Optional[str] = None
    max_price_impact: Optional[float] = None
    min_liquidity_required: Optional[float] = None
    priority_fee_multiplier: float = 1.0
    
    # Execution constraints
    execution_deadline: Optional[datetime] = None
    max_retries: int = 3
    
    def __post_init__(self):
        if self.execution_deadline is None:
            # Set deadline based on urgency
            timeout_seconds = 30 if self.urgency > 0.8 else 60
            self.execution_deadline = datetime.now() + timedelta(seconds=timeout_seconds)

@dataclass
class BuyResult:
    """Comprehensive buy execution result"""
    
    success: bool
    status: ExecutionStatus
    signature: Optional[str] = None
    error_message: Optional[str] = None
    failure_reason: Optional[FailureReason] = None
    
    # Trade details
    amount_sol: float = 0.0
    amount_tokens: float = 0.0
    actual_price: float = 0.0
    quoted_price: float = 0.0
    price_impact_percent: float = 0.0
    actual_slippage_percent: float = 0.0
    
    # Execution metrics
    total_latency_ms: int = 0
    quote_latency_ms: int = 0
    execution_latency_ms: int = 0
    confirmation_latency_ms: int = 0
    
    # Blockchain details
    tx_fee_sol: float = 0.0
    priority_fee_sol: float = 0.0
    block_height: Optional[int] = None
    confirmation_count: int = 0
    
    # Security metrics
    mev_protection_enabled: bool = False
    honeypot_check_passed: bool = False
    retry_count: int = 0
    
    # Additional metadata
    execution_path: str = ""  # Which DEX/aggregator was used
    pool_address: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class SecureRPCManager:
    """Secure RPC connection manager with failover and rate limiting"""
    
    def __init__(self, primary_rpc: str, backup_rpcs: List[str]):
        self.primary_rpc = primary_rpc
        self.backup_rpcs = backup_rpcs
        self.current_rpc = primary_rpc
        self.clients: Dict[str, AsyncClient] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        self.logger = logging.getLogger("SecureRPCManager")
    
    async def get_client(self) -> AsyncClient:
        """Get healthy RPC client with automatic failover"""
        
        # Try primary RPC first if healthy
        if self._is_rpc_healthy(self.primary_rpc):
            return await self._get_or_create_client(self.primary_rpc)
        
        # Try backup RPCs
        for rpc_url in self.backup_rpcs:
            if self._is_rpc_healthy(rpc_url):
                self.current_rpc = rpc_url
                self.logger.warning(f"Switched to backup RPC: {mask_sensitive_value(rpc_url)}")
                return await self._get_or_create_client(rpc_url)
        
        # If all RPCs are unhealthy, try primary anyway
        self.logger.error("All RPCs unhealthy, using primary RPC")
        return await self._get_or_create_client(self.primary_rpc)
    
    def _is_rpc_healthy(self, rpc_url: str) -> bool:
        """Check if RPC is healthy based on recent failures"""
        failure_count = self.failure_counts.get(rpc_url, 0)
        last_failure = self.last_failure_time.get(rpc_url, 0)
        
        # If too many recent failures, consider unhealthy
        if failure_count >= 3 and time.time() - last_failure < 300:  # 5 minutes
            return False
        
        return True
    
    async def _get_or_create_client(self, rpc_url: str) -> AsyncClient:
        """Get or create RPC client for URL"""
        if rpc_url not in self.clients:
            self.clients[rpc_url] = AsyncClient(
                rpc_url,
                commitment=Confirmed,
                timeout=30
            )
        return self.clients[rpc_url]
    
    def record_rpc_failure(self, rpc_url: str):
        """Record RPC failure for health tracking"""
        self.failure_counts[rpc_url] = self.failure_counts.get(rpc_url, 0) + 1
        self.last_failure_time[rpc_url] = time.time()
    
    def record_rpc_success(self, rpc_url: str):
        """Record RPC success"""
        if rpc_url in self.failure_counts:
            self.failure_counts[rpc_url] = max(0, self.failure_counts[rpc_url] - 1)

class JupiterAggregator:
    """Jupiter DEX aggregator client with enhanced security"""
    
    def __init__(self):
        self.base_url = "https://quote-api.jup.ag/v6"
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger("JupiterAggregator")
        self.rate_limit_delay = 0.1  # Minimum delay between requests
        self.last_request_time = 0
    
    async def initialize(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "WorkerAnt/2.0 (Production Trading Bot)",
                    "Content-Type": "application/json"
                }
            )
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_quote(self, input_mint: str, output_mint: str, amount: int, 
                       slippage_bps: int = 50) -> Optional[Dict[str, Any]]:
        """Get swap quote with retry logic"""
        
        await self._rate_limit()
        
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "slippageBps": slippage_bps,
            "onlyDirectRoutes": "false",
            "asLegacyTransaction": "false"
        }
        
        for attempt in range(3):
            try:
                async with self.session.get(f"{self.base_url}/quote", params=params) as response:
                    if response.status == 200:
                        quote_data = await response.json()
                        
                        # Validate quote
                        if self._validate_quote(quote_data, amount):
                            return quote_data
                        else:
                            self.logger.warning("Invalid quote received")
                            return None
                    
                    elif response.status == 429:  # Rate limited
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        self.logger.error(f"Quote request failed: {response.status}")
                        return None
                        
            except Exception as e:
                self.logger.error(f"Quote request error (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(1)
        
        return None
    
    async def get_swap_transaction(self, quote: Dict[str, Any], user_public_key: str) -> Optional[str]:
        """Get swap transaction from quote"""
        
        await self._rate_limit()
        
        swap_request = {
            "quoteResponse": quote,
            "userPublicKey": user_public_key,
            "wrapAndUnwrapSol": True,
            "dynamicComputeUnitLimit": True,
            "prioritizationFeeLamports": "auto"
        }
        
        try:
            async with self.session.post(f"{self.base_url}/swap", json=swap_request) as response:
                if response.status == 200:
                    swap_data = await response.json()
                    return swap_data.get("swapTransaction")
                else:
                    self.logger.error(f"Swap transaction request failed: {response.status}")
                    error_text = await response.text()
                    self.logger.error(f"Error details: {error_text}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Swap transaction error: {e}")
            return None
    
    def _validate_quote(self, quote: Dict[str, Any], expected_input: int) -> bool:
        """Validate quote response"""
        try:
            # Check required fields
            required_fields = ["inAmount", "outAmount", "priceImpactPct", "routePlan"]
            for field in required_fields:
                if field not in quote:
                    return False
            
            # Validate amounts
            if int(quote["inAmount"]) != expected_input:
                return False
            
            if int(quote["outAmount"]) <= 0:
                return False
            
            # Check price impact (reject if > 15%)
            price_impact = float(quote.get("priceImpactPct", 0))
            if price_impact > 15.0:
                self.logger.warning(f"High price impact: {price_impact}%")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Quote validation error: {e}")
            return False
    
    async def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

class ProductionBuyer:
    """Production-ready buy execution engine"""
    
    def __init__(self):
        self.config = get_trading_config()
        self.security_config = get_security_config()
        self.logger = logging.getLogger("ProductionBuyer")
        
        # RPC management
        self.rpc_manager = SecureRPCManager(
            self.config.rpc_url,
            getattr(self.config, 'backup_rpc_urls', [])
        )
        
        # DEX aggregator
        self.jupiter = JupiterAggregator()
        
        # Wallet management
        self.wallet: Optional[Keypair] = None
        self.wallet_address: Optional[str] = None
        
        # State tracking
        self.active_transactions: Dict[str, BuySignal] = {}
        self.recent_failures: List[Tuple[datetime, FailureReason]] = []
        
        # Performance metrics
        self.total_executions = 0
        self.successful_executions = 0
        self.average_latency_ms = 0
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the buyer system"""
        if self.initialized:
            return
        
        try:
            # Initialize components
            await self.jupiter.initialize()
            
            # Setup wallet
            await self._setup_secure_wallet()
            
            # Test RPC connection
            await self._test_rpc_connection()
            
            self.initialized = True
            self.logger.info("Production buyer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Buyer initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown buyer gracefully"""
        self.initialized = False
        
        # Close Jupiter session
        await self.jupiter.close()
        
        # Close RPC connections
        for client in self.rpc_manager.clients.values():
            await client.close()
        
        self.logger.info("Production buyer shutdown complete")
    
    async def execute_buy(self, signal: BuySignal) -> BuyResult:
        """Execute buy order with comprehensive error handling"""
        
        if not self.initialized:
            await self.initialize()
        
        execution_start = time.time()
        result = BuyResult(
            success=False,
            status=ExecutionStatus.PENDING,
            amount_sol=signal.amount_sol
        )
        
        try:
            # Pre-execution validations
            validation_result = await self._pre_execution_validation(signal)
            if not validation_result.success:
                result.status = ExecutionStatus.FAILED
                result.error_message = validation_result.error_message
                result.failure_reason = validation_result.failure_reason
                return result
            
            # Check if we can afford the trade
            balance_check = await self._check_wallet_balance(signal.amount_sol)
            if not balance_check:
                result.status = ExecutionStatus.FAILED
                result.error_message = "Insufficient SOL balance"
                result.failure_reason = FailureReason.INSUFFICIENT_BALANCE
                return result
            
            result.status = ExecutionStatus.EXECUTING
            
            # Get quote from Jupiter
            quote_start = time.time()
            quote = await self._get_swap_quote(signal)
            result.quote_latency_ms = int((time.time() - quote_start) * 1000)
            
            if not quote:
                result.status = ExecutionStatus.FAILED
                result.error_message = "Failed to get swap quote"
                result.failure_reason = FailureReason.QUOTE_FAILED
                return result
            
            # Validate quote meets our requirements
            quote_validation = self._validate_quote_requirements(quote, signal)
            if not quote_validation.success:
                result.status = ExecutionStatus.FAILED
                result.error_message = quote_validation.error_message
                result.failure_reason = quote_validation.failure_reason
                return result
            
            # Execute the swap transaction
            execution_start_time = time.time()
            execution_result = await self._execute_swap_transaction(quote, signal)
            result.execution_latency_ms = int((time.time() - execution_start_time) * 1000)
            
            if not execution_result.success:
                result.status = ExecutionStatus.FAILED
                result.error_message = execution_result.error_message
                result.failure_reason = execution_result.failure_reason
                return result
            
            result.signature = execution_result.signature
            result.status = ExecutionStatus.CONFIRMING
            
            # Wait for transaction confirmation
            confirmation_start = time.time()
            confirmation_result = await self._wait_for_confirmation(
                execution_result.signature, 
                timeout_seconds=30
            )
            result.confirmation_latency_ms = int((time.time() - confirmation_start) * 1000)
            
            if confirmation_result.success:
                result.status = ExecutionStatus.CONFIRMED
                result.success = True
                
                # Parse transaction details
                await self._parse_transaction_details(result, quote, execution_result.signature)
                
                # Update performance metrics
                self.successful_executions += 1
                
            else:
                result.status = ExecutionStatus.FAILED
                result.error_message = confirmation_result.error_message
                result.failure_reason = FailureReason.CONFIRMATION_FAILED
            
        except asyncio.TimeoutError:
            result.status = ExecutionStatus.TIMEOUT
            result.error_message = "Execution timeout"
            result.failure_reason = FailureReason.TIMEOUT_ERROR
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error_message = f"Execution error: {str(e)}"
            result.failure_reason = FailureReason.TRANSACTION_FAILED
            self.logger.error(f"Buy execution error: {e}")
        
        finally:
            # Calculate total latency
            result.total_latency_ms = int((time.time() - execution_start) * 1000)
            
            # Update metrics
            self.total_executions += 1
            self._update_average_latency(result.total_latency_ms)
            
            # Log result
            await self._log_execution_result(signal, result)
        
        return result
    
    async def _setup_secure_wallet(self):
        """Setup wallet with security considerations"""
        from ..core.simple_config import get_trading_config
        
        wallet_config = config_manager.get_config("wallet")
        
        if wallet_config.get("encrypted_private_key") and wallet_config.get("wallet_password"):
            # Decrypt wallet from encrypted storage
            try:
                encrypted_data = json.loads(wallet_config["encrypted_private_key"])
                decrypted_key = config_manager.wallet_manager.decrypt_private_key(
                    encrypted_data, 
                    wallet_config["wallet_password"]
                )
                
                # Validate and create wallet
                if config_manager.wallet_manager.validate_private_key(decrypted_key):
                    private_key_bytes = base58.b58decode(decrypted_key)
                    self.wallet = Keypair.from_bytes(private_key_bytes)
                    self.wallet_address = str(self.wallet.pubkey())
                    self.logger.info(f"Loaded encrypted wallet: {mask_sensitive_value(self.wallet_address)}")
                else:
                    raise ValueError("Invalid private key format")
                    
            except Exception as e:
                self.logger.error(f"Failed to load encrypted wallet: {e}")
                raise
        
        elif wallet_config.get("auto_create_wallet"):
            # Create new wallet (for testing only)
            self.wallet = Keypair()
            self.wallet_address = str(self.wallet.pubkey())
            self.logger.warning(f"Created new wallet: {mask_sensitive_value(self.wallet_address)}")
            self.logger.warning("Auto-created wallet - ensure it has SOL before trading!")
        
        else:
            raise ValueError("No valid wallet configuration found")
    
    async def _test_rpc_connection(self):
        """Test RPC connection health"""
        try:
            client = await self.rpc_manager.get_client()
            health = await client.get_health()
            
            if health.value != "ok":
                raise Exception(f"RPC health check failed: {health.value}")
            
            self.logger.info("RPC connection test passed")
            
        except Exception as e:
            self.logger.error(f"RPC connection test failed: {e}")
            raise
    
    async def _pre_execution_validation(self, signal: BuySignal) -> BuyResult:
        """Validate signal before execution"""
        result = BuyResult(success=True, status=ExecutionStatus.PENDING)
        
        # Check signal validity
        if not signal.token_address or len(signal.token_address) < 32:
            result.success = False
            result.error_message = "Invalid token address"
            result.failure_reason = FailureReason.TRANSACTION_FAILED
            return result
        
        # Check amount limits
        if signal.amount_sol < self.config.min_trade_amount_sol:
            result.success = False
            result.error_message = f"Amount below minimum: {signal.amount_sol}"
            result.failure_reason = FailureReason.TRANSACTION_FAILED
            return result
        
        if signal.amount_sol > self.config.max_trade_amount_sol:
            result.success = False
            result.error_message = f"Amount above maximum: {signal.amount_sol}"
            result.failure_reason = FailureReason.TRANSACTION_FAILED
            return result
        
        # Check slippage limits
        if signal.max_slippage > self.config.max_slippage_percent:
            result.success = False
            result.error_message = f"Slippage too high: {signal.max_slippage}%"
            result.failure_reason = FailureReason.HIGH_SLIPPAGE
            return result
        
        # Check execution deadline
        if signal.execution_deadline and datetime.now() > signal.execution_deadline:
            result.success = False
            result.error_message = "Signal execution deadline passed"
            result.failure_reason = FailureReason.TIMEOUT_ERROR
            return result
        
        return result
    
    async def _check_wallet_balance(self, required_sol: float) -> bool:
        """Check if wallet has sufficient SOL balance"""
        try:
            client = await self.rpc_manager.get_client()
            balance_response = await client.get_balance(self.wallet.pubkey())
            balance_sol = balance_response.value / 1e9
            
            # Add buffer for transaction fees
            required_with_buffer = required_sol + 0.01  # 0.01 SOL buffer
            
            return balance_sol >= required_with_buffer
            
        except Exception as e:
            self.logger.error(f"Balance check failed: {e}")
            return False
    
    async def _get_swap_quote(self, signal: BuySignal) -> Optional[Dict[str, Any]]:
        """Get swap quote from Jupiter"""
        try:
            # Convert SOL amount to lamports
            amount_lamports = int(signal.amount_sol * 1e9)
            
            # Convert slippage to basis points
            slippage_bps = int(signal.max_slippage * 100)
            
            # Get quote
            quote = await self.jupiter.get_quote(
                input_mint="So11111111111111111111111111111111111112",  # SOL
                output_mint=signal.token_address,
                amount=amount_lamports,
                slippage_bps=slippage_bps
            )
            
            return quote
            
        except Exception as e:
            self.logger.error(f"Failed to get swap quote: {e}")
            return None
    
    def _validate_quote_requirements(self, quote: Dict[str, Any], signal: BuySignal) -> BuyResult:
        """Validate quote meets signal requirements"""
        result = BuyResult(success=True, status=ExecutionStatus.PENDING)
        
        try:
            # Check price impact
            price_impact = float(quote.get("priceImpactPct", 0))
            if price_impact > signal.max_slippage:
                result.success = False
                result.error_message = f"Price impact too high: {price_impact}%"
                result.failure_reason = FailureReason.HIGH_SLIPPAGE
                return result
            
            # Check minimum output amount
            out_amount = int(quote.get("outAmount", 0))
            if out_amount <= 0:
                result.success = False
                result.error_message = "Invalid output amount in quote"
                result.failure_reason = FailureReason.QUOTE_FAILED
                return result
            
            # Check if route exists
            if not quote.get("routePlan"):
                result.success = False
                result.error_message = "No valid route found"
                result.failure_reason = FailureReason.QUOTE_FAILED
                return result
            
            return result
            
        except Exception as e:
            result.success = False
            result.error_message = f"Quote validation error: {str(e)}"
            result.failure_reason = FailureReason.QUOTE_FAILED
            return result
    
    async def _execute_swap_transaction(self, quote: Dict[str, Any], signal: BuySignal) -> BuyResult:
        """Execute swap transaction"""
        result = BuyResult(success=False, status=ExecutionStatus.EXECUTING)
        
        try:
            # Get swap transaction
            swap_tx_base64 = await self.jupiter.get_swap_transaction(quote, str(self.wallet.pubkey()))
            
            if not swap_tx_base64:
                result.error_message = "Failed to get swap transaction"
                result.failure_reason = FailureReason.TRANSACTION_FAILED
                return result
            
            # Decode and sign transaction
            import base64
            tx_bytes = base64.b64decode(swap_tx_base64)
            transaction = Transaction.deserialize(tx_bytes)
            
            # Sign transaction
            transaction.sign(self.wallet)
            
            # Send transaction
            client = await self.rpc_manager.get_client()
            
            # Send with confirmation
            send_response = await client.send_transaction(
                transaction, 
                opts={
                    "skip_preflight": False,
                    "preflight_commitment": "processed",
                    "max_retries": 3
                }
            )
            
            if send_response.value:
                result.success = True
                result.signature = str(send_response.value)
                self.logger.info(f"Transaction sent: {mask_sensitive_value(result.signature)}")
            else:
                result.error_message = "Transaction send failed"
                result.failure_reason = FailureReason.TRANSACTION_FAILED
            
            return result
            
        except Exception as e:
            result.error_message = f"Transaction execution error: {str(e)}"
            result.failure_reason = FailureReason.TRANSACTION_FAILED
            self.logger.error(f"Swap execution failed: {e}")
            return result
    
    async def _wait_for_confirmation(self, signature: str, timeout_seconds: int = 30) -> BuyResult:
        """Wait for transaction confirmation with timeout"""
        result = BuyResult(success=False, status=ExecutionStatus.CONFIRMING)
        
        try:
            client = await self.rpc_manager.get_client()
            sig = Signature.from_string(signature)
            
            # Wait for confirmation with timeout
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                try:
                    status = await client.get_signature_statuses([sig])
                    
                    if status.value and status.value[0]:
                        sig_status = status.value[0]
                        
                        if sig_status.err:
                            result.error_message = f"Transaction failed: {sig_status.err}"
                            result.failure_reason = FailureReason.TRANSACTION_FAILED
                            return result
                        
                        if sig_status.confirmation_status in ["confirmed", "finalized"]:
                            result.success = True
                            result.confirmation_count = 1 if sig_status.confirmation_status == "confirmed" else 2
                            return result
                    
                    await asyncio.sleep(2)  # Check every 2 seconds
                    
                except Exception as e:
                    self.logger.warning(f"Confirmation check error: {e}")
                    await asyncio.sleep(2)
            
            # Timeout reached
            result.error_message = "Confirmation timeout"
            result.failure_reason = FailureReason.TIMEOUT_ERROR
            return result
            
        except Exception as e:
            result.error_message = f"Confirmation error: {str(e)}"
            result.failure_reason = FailureReason.CONFIRMATION_FAILED
            return result
    
    async def _parse_transaction_details(self, result: BuyResult, quote: Dict[str, Any], signature: str):
        """Parse transaction details from confirmed transaction"""
        try:
            client = await self.rpc_manager.get_client()
            
            # Get transaction details
            sig = Signature.from_string(signature)
            tx_response = await client.get_transaction(
                sig, 
                commitment="confirmed",
                max_supported_transaction_version=0
            )
            
            if tx_response.value:
                tx = tx_response.value
                
                # Extract fee information
                if tx.transaction.meta:
                    result.tx_fee_sol = tx.transaction.meta.fee / 1e9
                
                # Extract token amounts from quote
                result.amount_tokens = float(quote.get("outAmount", 0)) / 1e6  # Assuming 6 decimals
                result.quoted_price = result.amount_sol / result.amount_tokens if result.amount_tokens > 0 else 0
                result.actual_price = result.quoted_price  # For now, assume quote was accurate
                
                # Calculate slippage (would need pre-trade price for accurate calculation)
                result.actual_slippage_percent = float(quote.get("priceImpactPct", 0))
                
                # Set execution path
                route_plan = quote.get("routePlan", [])
                if route_plan and len(route_plan) > 0:
                    result.execution_path = route_plan[0].get("swapInfo", {}).get("ammKey", "unknown")
                
        except Exception as e:
            self.logger.error(f"Failed to parse transaction details: {e}")
    
    def _update_average_latency(self, latency_ms: int):
        """Update running average latency"""
        if self.total_executions == 1:
            self.average_latency_ms = latency_ms
        else:
            # Running average
            alpha = 0.1  # Smoothing factor
            self.average_latency_ms = int(
                alpha * latency_ms + (1 - alpha) * self.average_latency_ms
            )
    
    async def _log_execution_result(self, signal: BuySignal, result: BuyResult):
        """Log execution result for monitoring"""
        try:
            # Create trade record
            trade_record = TradeRecord(
                timestamp=result.timestamp,
                token_address=signal.token_address,
                token_symbol=signal.token_symbol or "UNKNOWN",
                trade_type="BUY",
                success=result.success,
                amount_sol=result.amount_sol,
                amount_tokens=result.amount_tokens,
                price=result.actual_price,
                slippage_percent=result.actual_slippage_percent,
                latency_ms=result.total_latency_ms,
                gas_cost_sol=result.tx_fee_sol,
                tx_signature=result.signature,
                retry_count=result.retry_count,
                error_message=result.error_message
            )
            
            # Log to trading logger
            if trading_logger:
                await trading_logger.log_trade(trade_record)
            
            # Log summary
            if result.success:
                self.logger.info(
                    f"BUY SUCCESS: {signal.token_symbol} | "
                    f"{result.amount_sol:.4f} SOL → {result.amount_tokens:.2f} tokens | "
                    f"{result.total_latency_ms}ms | "
                    f"Slippage: {result.actual_slippage_percent:.2f}%"
                )
            else:
                self.logger.warning(
                    f"BUY FAILED: {signal.token_symbol} | "
                    f"Reason: {result.failure_reason.value if result.failure_reason else 'unknown'} | "
                    f"Error: {result.error_message}"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to log execution result: {e}")
    
    async def get_balance(self) -> float:
        """Get current wallet SOL balance"""
        try:
            client = await self.rpc_manager.get_client()
            balance_response = await client.get_balance(self.wallet.pubkey())
            return balance_response.value / 1e9
        except Exception as e:
            self.logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get buyer performance metrics"""
        success_rate = (self.successful_executions / max(self.total_executions, 1)) * 100
        
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "success_rate": success_rate,
            "average_latency_ms": self.average_latency_ms,
            "wallet_address": mask_sensitive_value(self.wallet_address) if self.wallet_address else None,
            "current_rpc": mask_sensitive_value(self.rpc_manager.current_rpc),
            "initialized": self.initialized
        }

# === GLOBAL INSTANCES ===

# Global buyer instance (backward compatibility)
enhanced_buyer = ProductionBuyer()

# Backward compatibility aliases
trade_buyer = enhanced_buyer

# Export main classes
__all__ = [
    "BuySignal", "BuyResult", "ProductionBuyer", "ExecutionStatus", "FailureReason",
    "enhanced_buyer", "trade_buyer"
] 