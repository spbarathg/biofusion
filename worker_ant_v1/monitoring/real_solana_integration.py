"""
REAL SOLANA INTEGRATION - PRODUCTION BLOCKCHAIN OPERATIONS
==========================================================

Production-ready Solana blockchain integration with comprehensive error handling,
retry logic, and real transaction processing. No mocks - only real operations.
"""

import asyncio
import aiohttp
import time
import json
import base58
import base64
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging


try:
    from solders.keypair import Keypair
    from solana.rpc.async_api import AsyncClient
    from solana.rpc.commitment import Commitment, Confirmed, Finalized
    from solana.rpc.types import TxOpts
    from solders.transaction import Transaction
    from solders.system_program import transfer
    from solders.pubkey import Pubkey as PublicKey
    from solana.rpc.api import Client
    from solders.signature import Signature
    from solders.transaction import VersionedTransaction
    from spl.token.client import Token
    from spl.token.constants import TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID
    from spl.token.instructions import get_associated_token_address
except ImportError as e:
    raise ImportError(
        f"Critical Solana dependencies missing: {e}. "
        "Install with: pip install solana spl-token solders"
    ) from e

from worker_ant_v1.core.unified_config import get_trading_config, get_security_config, get_network_config
from worker_ant_v1.utils.logger import setup_logger

class TransactionStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class TransactionResult:
    """Comprehensive transaction result tracking"""
    
    signature: str
    status: TransactionStatus
    slot: Optional[int] = None
    confirmation_time: Optional[datetime] = None
    gas_used: Optional[int] = None
    error: Optional[str] = None
    
    
    submission_time: datetime = field(default_factory=datetime.now)
    confirmation_latency_ms: Optional[float] = None
    
    
    from_address: Optional[str] = None
    to_address: Optional[str] = None
    amount: Optional[float] = None
    token_mint: Optional[str] = None

@dataclass
class RPCEndpoint:
    """RPC endpoint configuration with health tracking"""
    
    url: str
    name: str
    priority: int = 1  # 1 = highest priority
    max_requests_per_second: int = 50
    timeout_seconds: float = 5.0
    
    
    is_healthy: bool = True
    last_request: Optional[datetime] = None
    consecutive_failures: int = 0
    avg_response_time_ms: float = 0.0
    success_rate_24h: float = 100.0
    
    
    request_count_current_second: int = 0
    current_second: int = 0

class ProductionSolanaClient:
    """Production-grade Solana client with redundancy and error handling"""
    
    def __init__(self):
        self.logger = setup_logger("ProductionSolanaClient")
        self.config = get_trading_config()
        self.network_config = get_network_config()
        
        
        self.rpc_endpoints: List[RPCEndpoint] = []
        self.current_endpoint_index = 0
        self.clients: Dict[str, AsyncClient] = {}
        
        
        self.connection_pool_size = 10
        self.max_retries = 3
        self.retry_delay_base = 1.0  # exponential backoff base
        
        
        self.transaction_history: List[TransactionResult] = []
        self.performance_metrics = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'failed_transactions': 0,
            'avg_confirmation_time_ms': 0.0,
            'rpc_switch_count': 0
        }
        
        self._initialize_endpoints()
    
    def _initialize_endpoints(self):
        """Initialize RPC endpoints with production configuration"""
        
        
        primary_endpoints = [
            RPCEndpoint(
                url="https://api.mainnet-beta.solana.com",
                name="Solana_Labs_Primary",
                priority=1,
                max_requests_per_second=40
            ),
            RPCEndpoint(
                url="https://solana-api.projectserum.com",
                name="Serum_API",
                priority=2,
                max_requests_per_second=30
            )
        ]
        
        
        custom_rpc = self.network_config.rpc_url
        if custom_rpc and custom_rpc not in [ep.url for ep in primary_endpoints]:
            primary_endpoints.insert(0, RPCEndpoint(
                url=custom_rpc,
                name="Custom_RPC",
                priority=1,
                max_requests_per_second=100  # Assume higher limits for paid RPC
            ))
        
        
        backup_endpoints = [
            RPCEndpoint(
                url="https://rpc.ankr.com/solana",
                name="Ankr_Backup",
                priority=3,
                max_requests_per_second=20
            ),
            RPCEndpoint(
                url="https://solana.public-rpc.com",
                name="Public_RPC_Backup",
                priority=4,
                max_requests_per_second=15
            )
        ]
        
        self.rpc_endpoints = primary_endpoints + backup_endpoints
        self.logger.info(f"Initialized {len(self.rpc_endpoints)} RPC endpoints")
    
    async def initialize(self) -> bool:
        """Initialize all RPC connections with health checks"""
        
        self.logger.info("🔗 Initializing production Solana connections...")
        
        try:
            for endpoint in self.rpc_endpoints:
                client = AsyncClient(
                    endpoint.url,
                    commitment=Confirmed,
                    timeout=endpoint.timeout_seconds
                )
                self.clients[endpoint.name] = client
            
            
            await self._health_check_all_endpoints()
            
            
            healthy_endpoints = [ep for ep in self.rpc_endpoints if ep.is_healthy]
            if not healthy_endpoints:
                raise RuntimeError("No healthy RPC endpoints available")
            
            self.logger.info(f"✅ {len(healthy_endpoints)}/{len(self.rpc_endpoints)} RPC endpoints healthy")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Solana connections: {e}")
            return False
    
    async def _health_check_all_endpoints(self):
        """Perform health checks on all RPC endpoints"""
        
        health_check_tasks = []
        for endpoint in self.rpc_endpoints:
            task = asyncio.create_task(self._health_check_endpoint(endpoint))
            health_check_tasks.append(task)
        
        await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _health_check_endpoint(self, endpoint: RPCEndpoint):
        """Health check individual RPC endpoint"""
        
        try:
            client = self.clients[endpoint.name]
            start_time = time.time()
            
            
            response = await client.get_latest_blockhash()
            
            
            response_time_ms = (time.time() - start_time) * 1000
            endpoint.avg_response_time_ms = response_time_ms
            
            if response.value:
                endpoint.is_healthy = True
                endpoint.consecutive_failures = 0
                self.logger.debug(f"✅ {endpoint.name}: {response_time_ms:.1f}ms")
            else:
                raise Exception("Invalid response from RPC")
                
        except Exception as e:
            endpoint.is_healthy = False
            endpoint.consecutive_failures += 1
            self.logger.warning(f"⚠️ {endpoint.name} health check failed: {e}")
    
    async def get_current_client(self) -> AsyncClient:
        """Get current healthy RPC client with automatic failover"""
        
        
        current_endpoint = self.rpc_endpoints[self.current_endpoint_index]
        
        if current_endpoint.is_healthy and await self._check_rate_limit(current_endpoint):
            return self.clients[current_endpoint.name]
        
        
        for i, endpoint in enumerate(self.rpc_endpoints):
            if endpoint.is_healthy and await self._check_rate_limit(endpoint):
                if i != self.current_endpoint_index:
                    self.logger.info(f"🔄 Switching to RPC: {endpoint.name}")
                    self.current_endpoint_index = i
                    self.performance_metrics['rpc_switch_count'] += 1
                
                return self.clients[endpoint.name]
        
        
        raise RuntimeError("No healthy RPC endpoints available")
    
    async def _check_rate_limit(self, endpoint: RPCEndpoint) -> bool:
        """Check if endpoint is within rate limits"""
        
        current_second = int(time.time())
        
        if endpoint.current_second != current_second:
            endpoint.current_second = current_second
            endpoint.request_count_current_second = 0
        
        if endpoint.request_count_current_second >= endpoint.max_requests_per_second:
            return False
        
        endpoint.request_count_current_second += 1
        endpoint.last_request = datetime.now()
        return True
    
    async def send_transaction_with_retry(self, 
                                         transaction: VersionedTransaction,
                                         max_retries: Optional[int] = None) -> TransactionResult:
        """Send transaction with comprehensive retry logic and monitoring"""
        
        max_retries = max_retries or self.max_retries
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                client = await self.get_current_client()
                
                
                start_time = time.time()
                result = await client.send_transaction(
                    transaction,
                    opts=TxOpts(
                        skip_confirmation=False,
                        skip_preflight=False,
                        max_retries=1
                    )
                )
                
                if result.value:
                if result.value:
                    signature = str(result.value)
                    tx_result = await self._monitor_transaction_confirmation(
                        signature, start_time
                    )
                    
                    self.performance_metrics['total_transactions'] += 1
                    if tx_result.status in [TransactionStatus.CONFIRMED, TransactionStatus.FINALIZED]:
                        self.performance_metrics['successful_transactions'] += 1
                    else:
                        self.performance_metrics['failed_transactions'] += 1
                    
                    return tx_result
                else:
                    raise Exception("Transaction submission failed - no signature returned")
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"Transaction attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries:
                if attempt < max_retries:
                    delay = self.retry_delay_base * (2 ** attempt)
                    await asyncio.sleep(delay)
                    
                    
                    await self._health_check_all_endpoints()
        
        
        self.performance_metrics['total_transactions'] += 1
        self.performance_metrics['failed_transactions'] += 1
        
        return TransactionResult(
            signature="",
            status=TransactionStatus.FAILED,
            error=str(last_error),
            confirmation_latency_ms=None
        )
    
    async def _monitor_transaction_confirmation(self, 
                                              signature: str, 
                                              start_time: float) -> TransactionResult:
        """Monitor transaction confirmation with timeout"""
        
        confirmation_timeout = 30.0  # 30 seconds
        poll_interval = 1.0  # Check every second
        
        try:
            client = await self.get_current_client()
            
            while time.time() - start_time < confirmation_timeout:
                try:
                    status_result = await client.get_signature_statuses([Signature.from_string(signature)])
                    
                    if status_result.value and status_result.value[0]:
                        status_info = status_result.value[0]
                        
                        if status_info.confirmation_status:
                            confirmation_time = datetime.now()
                            latency_ms = (time.time() - start_time) * 1000
                            
                            if status_info.confirmation_status == "finalized":
                                return TransactionResult(
                                    signature=signature,
                                    status=TransactionStatus.FINALIZED,
                                    slot=status_info.slot,
                                    confirmation_time=confirmation_time,
                                    confirmation_latency_ms=latency_ms
                                )
                            elif status_info.confirmation_status in ["confirmed", "processed"]:
                                return TransactionResult(
                                    signature=signature,
                                    status=TransactionStatus.CONFIRMED,
                                    slot=status_info.slot,
                                    confirmation_time=confirmation_time,
                                    confirmation_latency_ms=latency_ms
                                )
                        
                        if status_info.err:
                            return TransactionResult(
                                signature=signature,
                                status=TransactionStatus.FAILED,
                                error=str(status_info.err),
                                confirmation_latency_ms=(time.time() - start_time) * 1000
                            )
                
                except Exception as e:
                    self.logger.debug(f"Status check error: {e}")
                
                await asyncio.sleep(poll_interval)
            
            
            return TransactionResult(
                signature=signature,
                status=TransactionStatus.TIMEOUT,
                confirmation_latency_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            return TransactionResult(
                signature=signature,
                status=TransactionStatus.FAILED,
                error=str(e),
                confirmation_latency_ms=(time.time() - start_time) * 1000
            )
    
    async def get_token_balance(self, wallet_address: str, token_mint: str) -> float:
        """Get token balance with error handling and retries"""
        
        for attempt in range(self.max_retries):
            try:
                client = await self.get_current_client()
                
                
                token_account = get_associated_token_address(
                    PublicKey(wallet_address),
                    PublicKey(token_mint)
                )
                
                
                response = await client.get_token_account_balance(token_account)
                
                if response.value:
                if response.value:
                    amount = response.value.amount
                    decimals = response.value.decimals
                    return float(amount) / (10 ** decimals)
                
                return 0.0
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Failed to get token balance: {e}")
                    return 0.0
                await asyncio.sleep(self.retry_delay_base * (attempt + 1))
        
        return 0.0
    
    async def get_sol_balance(self, wallet_address: str) -> float:
        """Get SOL balance with error handling"""
        
        for attempt in range(self.max_retries):
            try:
                client = await self.get_current_client()
                response = await client.get_balance(PublicKey(wallet_address))
                
                if response.value is not None:
                if response.value is not None:
                    return response.value / 1_000_000_000
                
                return 0.0
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Failed to get SOL balance: {e}")
                    return 0.0
                await asyncio.sleep(self.retry_delay_base * (attempt + 1))
        
        return 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        healthy_endpoints = [ep for ep in self.rpc_endpoints if ep.is_healthy]
        
        return {
            'endpoints': {
                'total': len(self.rpc_endpoints),
                'healthy': len(healthy_endpoints),
                'current': self.rpc_endpoints[self.current_endpoint_index].name,
                'avg_response_time_ms': sum(ep.avg_response_time_ms for ep in healthy_endpoints) / len(healthy_endpoints) if healthy_endpoints else 0
            },
            'transactions': self.performance_metrics,
            'success_rate': (
                self.performance_metrics['successful_transactions'] / 
                max(self.performance_metrics['total_transactions'], 1)
            ) * 100
        }
    
    async def shutdown(self):
        """Graceful shutdown of all connections"""
        
        self.logger.info("🔽 Shutting down Solana connections...")
        
        
        for client in self.clients.values():
            try:
                await client.close()
            except Exception as e:
                self.logger.debug(f"Error closing client: {e}")
        
        self.clients.clear()
        self.logger.info("✅ Solana connections closed")


production_solana_client = ProductionSolanaClient() 