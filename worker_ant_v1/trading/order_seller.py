"""
PRODUCTION-READY SELLER MODULE
=============================

Ultra-fast, secure sell execution with intelligent exit strategies,
profit optimization, emergency exits, and comprehensive monitoring.
"""

import asyncio
import aiohttp
import time
import json
import hashlib
from typing import Optional, Tuple, Dict, List, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

# External dependencies - conditional imports
try:
    from solana.rpc.async_api import AsyncClient
    from solana.rpc.commitment import Confirmed, Processed, Finalized
    from solana.transaction import Transaction
    from solders.pubkey import Pubkey
    from solders.keypair import Keypair
    from spl.token.constants import TOKEN_PROGRAM_ID
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
from worker_ant_v1.trading.order_buyer import SecureRPCManager, JupiterAggregator

class SellTrigger(Enum):
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIMEOUT = "timeout"
    EMERGENCY = "emergency"
    MANUAL = "manual"
    MARKET_CONDITION = "market_condition"
    LIQUIDITY_LOW = "liquidity_low"
    HONEYPOT_DETECTED = "honeypot_detected"

class SellStrategy(Enum):
    IMMEDIATE = "immediate"           # Sell immediately at market price
    PROGRESSIVE = "progressive"       # Gradual sell to minimize impact
    LIMIT_ORDER = "limit_order"      # Try to sell at specific price
    BEST_EFFORT = "best_effort"      # Try multiple strategies
    EMERGENCY_EXIT = "emergency_exit" # Fast exit, ignore slippage

class SellStatus(Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    EXECUTING = "executing"
    CONFIRMING = "confirming"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

@dataclass
class Position:
    """Active trading position tracking"""
    
    token_address: str
    token_symbol: str
    entry_timestamp: datetime
    entry_signature: str
    
    # Position details
    amount_sol_invested: float
    amount_tokens_held: float
    entry_price: float
    current_price: float = 0.0
    
    # P&L tracking
    unrealized_pnl_sol: float = 0.0
    unrealized_pnl_percent: float = 0.0
    
    # Strategy settings
    profit_target_percent: float = 8.0
    stop_loss_percent: float = 7.0
    timeout_seconds: int = 180
    
    # Status tracking
    last_price_update: datetime = field(default_factory=datetime.utcnow)
    price_checks: int = 0
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    
    # Exit conditions
    trailing_stop_enabled: bool = False
    trailing_stop_percent: float = 3.0
    max_hold_time_seconds: int = 300
    
    def calculate_pnl(self, current_price: float) -> Tuple[float, float]:
        """Calculate current P&L"""
        if self.amount_tokens_held > 0:
            current_value_sol = self.amount_tokens_held * current_price
            pnl_sol = current_value_sol - self.amount_sol_invested
            pnl_percent = (pnl_sol / self.amount_sol_invested) * 100
            return pnl_sol, pnl_percent
        return 0.0, 0.0
    
    def should_sell_profit_target(self) -> bool:
        """Check if profit target reached"""
        return self.unrealized_pnl_percent >= self.profit_target_percent
    
    def should_sell_stop_loss(self) -> bool:
        """Check if stop loss triggered"""
        return self.unrealized_pnl_percent <= -self.stop_loss_percent
    
    def should_sell_timeout(self) -> bool:
        """Check if position timeout reached"""
        elapsed = (datetime.utcnow() - self.entry_timestamp).total_seconds()
        return elapsed >= self.timeout_seconds
    
    def should_sell_trailing_stop(self, current_price: float) -> bool:
        """Check if trailing stop triggered"""
        if not self.trailing_stop_enabled or self.highest_price == 0:
            return False
        
        # Calculate drop from highest price
        price_drop_percent = ((self.highest_price - current_price) / self.highest_price) * 100
        return price_drop_percent >= self.trailing_stop_percent
    
    def update_price(self, new_price: float):
        """Update position with new price data"""
        self.current_price = new_price
        self.last_price_update = datetime.utcnow()
        self.price_checks += 1
        
        # Track highest and lowest prices
        self.highest_price = max(self.highest_price, new_price)
        self.lowest_price = min(self.lowest_price, new_price)
        
        # Update P&L
        self.unrealized_pnl_sol, self.unrealized_pnl_percent = self.calculate_pnl(new_price)

@dataclass
class SellSignal:
    """Sell signal with comprehensive parameters"""
    
    position: Position
    trigger: SellTrigger
    strategy: SellStrategy
    urgency: float  # 0.0 to 1.0
    
    # Execution parameters
    max_slippage: float = 5.0
    min_amount_out: Optional[float] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    
    # Strategy-specific settings
    progressive_chunks: int = 1  # For progressive selling
    price_limit: Optional[float] = None  # For limit orders
    
    # Emergency settings
    emergency_exit: bool = False
    ignore_slippage: bool = False

@dataclass
class SellResult:
    """Comprehensive sell execution result"""
    
    success: bool
    status: SellStatus
    signature: Optional[str] = None
    error_message: Optional[str] = None
    
    # Trade details
    amount_sol_received: float = 0.0
    amount_tokens_sold: float = 0.0
    average_sell_price: float = 0.0
    actual_slippage_percent: float = 0.0
    
    # P&L details
    profit_loss_sol: float = 0.0
    profit_loss_percent: float = 0.0
    hold_time_seconds: int = 0
    
    # Execution metrics
    total_latency_ms: int = 0
    quote_latency_ms: int = 0
    execution_latency_ms: int = 0
    confirmation_latency_ms: int = 0
    
    # Fees and costs
    tx_fee_sol: float = 0.0
    priority_fee_sol: float = 0.0
    slippage_cost_sol: float = 0.0
    
    # Additional details
    trigger_reason: str = ""
    execution_path: str = ""
    blocks_waited: int = 0
    retry_count: int = 0
    
    # Performance metrics
    price_at_signal: float = 0.0
    price_at_execution: float = 0.0
    price_impact_percent: float = 0.0
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class PriceMonitor:
    """Real-time price monitoring for positions"""
    
    def __init__(self, jupiter: JupiterAggregator):
        self.jupiter = jupiter
        self.logger = logging.getLogger("PriceMonitor")
        self.price_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl_seconds = 5  # Price cache TTL
        
    async def get_current_price(self, token_address: str, reference_amount: float = 1.0) -> Optional[float]:
        """Get current token price in SOL"""
        
        # Check cache first
        cache_key = token_address
        if cache_key in self.price_cache:
            cached_data = self.price_cache[cache_key]
            if time.time() - cached_data["timestamp"] < self.cache_ttl_seconds:
                return cached_data["price"]
        
        try:
            # Use a small amount for price discovery
            amount_tokens = int(reference_amount * 1e6)  # Assuming 6 decimals
            
            # Get quote for selling tokens -> SOL
            quote = await self.jupiter.get_quote(
                input_mint=token_address,
                output_mint="So11111111111111111111111111111111111112",  # SOL
                amount=amount_tokens,
                slippage_bps=100  # 1% for price discovery
            )
            
            if quote and quote.get("outAmount"):
                sol_amount = float(quote["outAmount"]) / 1e9  # Convert lamports to SOL
                price_per_token = sol_amount / reference_amount
                
                # Cache the result
                self.price_cache[cache_key] = {
                    "price": price_per_token,
                    "timestamp": time.time()
                }
                
                return price_per_token
        
        except Exception as e:
            self.logger.error(f"Failed to get price for {mask_sensitive_value(token_address)}: {e}")
        
        return None
    
    async def get_multiple_prices(self, token_addresses: List[str]) -> Dict[str, Optional[float]]:
        """Get prices for multiple tokens efficiently"""
        
        tasks = []
        for token_address in token_addresses:
            task = asyncio.create_task(self.get_current_price(token_address))
            tasks.append((token_address, task))
        
        results = {}
        for token_address, task in tasks:
            try:
                price = await task
                results[token_address] = price
            except Exception as e:
                self.logger.error(f"Failed to get price for {mask_sensitive_value(token_address)}: {e}")
                results[token_address] = None
        
        return results

class PositionManager:
    """Manages active trading positions"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.logger = logging.getLogger("PositionManager")
        self.config = get_trading_config()
    
    def add_position(self, position: Position):
        """Add new position to tracking"""
        self.positions[position.token_address] = position
        self.logger.info(f"Added position: {position.token_symbol} @ {position.entry_price:.8f} SOL")
    
    def remove_position(self, token_address: str) -> Optional[Position]:
        """Remove and return position"""
        return self.positions.pop(token_address, None)
    
    def get_position(self, token_address: str) -> Optional[Position]:
        """Get position by token address"""
        return self.positions.get(token_address)
    
    def get_all_positions(self) -> List[Position]:
        """Get all active positions"""
        return list(self.positions.values())
    
    def update_position_price(self, token_address: str, new_price: float):
        """Update position with new price"""
        if token_address in self.positions:
            self.positions[token_address].update_price(new_price)
    
    def get_positions_needing_exit(self) -> List[Tuple[Position, SellTrigger]]:
        """Get positions that need to be sold"""
        exit_positions = []
        
        for position in self.positions.values():
            # Check various exit conditions
            if position.should_sell_profit_target():
                exit_positions.append((position, SellTrigger.PROFIT_TARGET))
            elif position.should_sell_stop_loss():
                exit_positions.append((position, SellTrigger.STOP_LOSS))
            elif position.should_sell_timeout():
                exit_positions.append((position, SellTrigger.TIMEOUT))
            elif position.should_sell_trailing_stop(position.current_price):
                exit_positions.append((position, SellTrigger.MARKET_CONDITION))
        
        return exit_positions
    
    def get_portfolio_value(self) -> Tuple[float, float]:
        """Get total portfolio value and P&L"""
        total_invested = 0.0
        total_current_value = 0.0
        
        for position in self.positions.values():
            total_invested += position.amount_sol_invested
            if position.current_price > 0:
                current_value = position.amount_tokens_held * position.current_price
                total_current_value += current_value
            else:
                total_current_value += position.amount_sol_invested  # Fallback to entry value
        
        total_pnl = total_current_value - total_invested
        return total_current_value, total_pnl

class ProductionSeller:
    """Production-ready sell execution engine"""
    
    def __init__(self):
        self.config = get_trading_config()
        self.security_config = get_security_config()
        self.logger = logging.getLogger("ProductionSeller")
        
        # Component managers
        self.rpc_manager = SecureRPCManager(
            self.config.rpc_url,
            getattr(self.config, 'backup_rpc_urls', [])
        )
        self.jupiter = JupiterAggregator()
        self.price_monitor = PriceMonitor(self.jupiter)
        self.position_manager = PositionManager()
        
        # Wallet management
        self.wallet: Optional[Keypair] = None
        self.wallet_address: Optional[str] = None
        
        # Performance tracking
        self.total_sells = 0
        self.successful_sells = 0
        self.total_profit_sol = 0.0
        self.average_hold_time = 0.0
        
        # Background monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_active = False
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize the seller system"""
        if self.initialized:
            return
        
        try:
            # Initialize components
            await self.jupiter.initialize()
            
            # Setup wallet (shared with buyer)
            await self._setup_wallet()
            
            # Start position monitoring
            await self._start_position_monitoring()
            
            self.initialized = True
            self.logger.info("Production seller initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Seller initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown seller gracefully"""
        self.initialized = False
        self.monitoring_active = False
        
        # Cancel monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Close Jupiter session
        await self.jupiter.close()
        
        self.logger.info("Production seller shutdown complete")
    
    async def add_position_from_buy(self, token_address: str, token_symbol: str, 
                                   buy_result: Any) -> Position:
        """Add position from successful buy"""
        
        position = Position(
            token_address=token_address,
            token_symbol=token_symbol,
            entry_timestamp=datetime.utcnow(),
            entry_signature=buy_result.signature,
            amount_sol_invested=buy_result.amount_sol,
            amount_tokens_held=buy_result.amount_tokens,
            entry_price=buy_result.actual_price,
            profit_target_percent=self.config.base_profit_target_percent,
            stop_loss_percent=self.config.base_stop_loss_percent,
            timeout_seconds=self.config.timeout_exit_seconds
        )
        
        # Get initial price
        current_price = await self.price_monitor.get_current_price(token_address)
        if current_price:
            position.update_price(current_price)
        
        self.position_manager.add_position(position)
        
        self.logger.info(
            f"Added position: {token_symbol} | "
            f"Invested: {position.amount_sol_invested:.4f} SOL | "
            f"Tokens: {position.amount_tokens_held:.2f} | "
            f"Target: +{position.profit_target_percent:.1f}% | "
            f"Stop: -{position.stop_loss_percent:.1f}%"
        )
        
        return position
    
    async def execute_sell(self, sell_signal: SellSignal) -> SellResult:
        """Execute sell order with comprehensive handling"""
        
        if not self.initialized:
            await self.initialize()
        
        execution_start = time.time()
        position = sell_signal.position
        
        result = SellResult(
            success=False,
            status=SellStatus.PENDING,
            trigger_reason=sell_signal.trigger.value,
            price_at_signal=position.current_price
        )
        
        try:
            result.status = SellStatus.ANALYZING
            
            # Pre-sell validation
            validation_result = await self._validate_sell_signal(sell_signal)
            if not validation_result:
                result.status = SellStatus.FAILED
                result.error_message = "Sell signal validation failed"
                return result
            
            result.status = SellStatus.EXECUTING
            
            # Get current price for execution
            current_price = await self.price_monitor.get_current_price(position.token_address)
            if not current_price:
                result.status = SellStatus.FAILED
                result.error_message = "Failed to get current price"
                return result
            
            result.price_at_execution = current_price
            position.update_price(current_price)
            
            # Execute sell based on strategy
            if sell_signal.strategy == SellStrategy.EMERGENCY_EXIT:
                execution_result = await self._execute_emergency_sell(sell_signal, current_price)
            elif sell_signal.strategy == SellStrategy.PROGRESSIVE:
                execution_result = await self._execute_progressive_sell(sell_signal, current_price)
            elif sell_signal.strategy == SellStrategy.LIMIT_ORDER:
                execution_result = await self._execute_limit_sell(sell_signal, current_price)
            else:  # IMMEDIATE or BEST_EFFORT
                execution_result = await self._execute_immediate_sell(sell_signal, current_price)
            
            if execution_result.success:
                result.status = SellStatus.CONFIRMING
                result.signature = execution_result.signature
                
                # Wait for confirmation
                confirmation_start = time.time()
                confirmed = await self._wait_for_sell_confirmation(execution_result.signature)
                result.confirmation_latency_ms = int((time.time() - confirmation_start) * 1000)
                
                if confirmed:
                    result.status = SellStatus.COMPLETED
                    result.success = True
                    
                    # Parse transaction details
                    await self._parse_sell_transaction_details(result, position, execution_result.signature)
                    
                    # Calculate P&L
                    result.profit_loss_sol = result.amount_sol_received - position.amount_sol_invested
                    result.profit_loss_percent = (result.profit_loss_sol / position.amount_sol_invested) * 100
                    result.hold_time_seconds = int((datetime.utcnow() - position.entry_timestamp).total_seconds())
                    
                    # Update metrics
                    self.successful_sells += 1
                    self.total_profit_sol += result.profit_loss_sol
                    self._update_average_hold_time(result.hold_time_seconds)
                    
                    # Remove position from tracking
                    self.position_manager.remove_position(position.token_address)
                    
                else:
                    result.status = SellStatus.FAILED
                    result.error_message = "Transaction confirmation failed"
            else:
                result.status = SellStatus.FAILED
                result.error_message = execution_result.error_message
        
        except Exception as e:
            result.status = SellStatus.FAILED
            result.error_message = f"Sell execution error: {str(e)}"
            self.logger.error(f"Sell execution failed: {e}")
        
        finally:
            # Calculate total latency
            result.total_latency_ms = int((time.time() - execution_start) * 1000)
            
            # Update metrics
            self.total_sells += 1
            
            # Log result
            await self._log_sell_result(sell_signal, result)
        
        return result
    
    async def emergency_exit_all(self, reason: str = "Emergency exit") -> List[SellResult]:
        """Emergency exit all positions"""
        
        self.logger.warning(f"EMERGENCY EXIT ALL POSITIONS: {reason}")
        
        results = []
        positions = self.position_manager.get_all_positions()
        
        # Create emergency sell signals for all positions
        sell_tasks = []
        for position in positions:
            sell_signal = SellSignal(
                position=position,
                trigger=SellTrigger.EMERGENCY,
                strategy=SellStrategy.EMERGENCY_EXIT,
                urgency=1.0,
                max_slippage=20.0,  # Allow high slippage for emergency
                emergency_exit=True,
                ignore_slippage=True,
                timeout_seconds=15
            )
            
            task = asyncio.create_task(self.execute_sell(sell_signal))
            sell_tasks.append(task)
        
        # Execute all sells concurrently
        if sell_tasks:
            results = await asyncio.gather(*sell_tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = [r for r in results if isinstance(r, SellResult)]
            
            # Log emergency exit summary
            successful_exits = sum(1 for r in valid_results if r.success)
            total_positions = len(positions)
            
            self.logger.warning(
                f"Emergency exit completed: {successful_exits}/{total_positions} positions exited"
            )
        
        return results
    
    async def _setup_wallet(self):
        """Setup wallet (shared with buyer)"""
        from worker_ant_v1.core.simple_config import get_trading_config
        
        wallet_config = config_manager.get_config("wallet")
        
        if wallet_config.get("encrypted_private_key") and wallet_config.get("wallet_password"):
            try:
                encrypted_data = json.loads(wallet_config["encrypted_private_key"])
                decrypted_key = config_manager.wallet_manager.decrypt_private_key(
                    encrypted_data, 
                    wallet_config["wallet_password"]
                )
                
                if config_manager.wallet_manager.validate_private_key(decrypted_key):
                    private_key_bytes = base58.b58decode(decrypted_key)
                    self.wallet = Keypair.from_bytes(private_key_bytes)
                    self.wallet_address = str(self.wallet.pubkey())
                else:
                    raise ValueError("Invalid private key format")
                    
            except Exception as e:
                self.logger.error(f"Failed to load encrypted wallet: {e}")
                raise
        else:
            raise ValueError("No valid wallet configuration found")
    
    async def _start_position_monitoring(self):
        """Start background position monitoring"""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._position_monitoring_loop())
    
    async def _position_monitoring_loop(self):
        """Background loop to monitor positions"""
        
        while self.monitoring_active:
            try:
                positions = self.position_manager.get_all_positions()
                
                if positions:
                    # Get current prices for all positions
                    token_addresses = [p.token_address for p in positions]
                    prices = await self.price_monitor.get_multiple_prices(token_addresses)
                    
                    # Update positions with new prices
                    for position in positions:
                        if prices.get(position.token_address):
                            self.position_manager.update_position_price(
                                position.token_address, 
                                prices[position.token_address]
                            )
                    
                    # Check for positions that need to exit
                    exit_positions = self.position_manager.get_positions_needing_exit()
                    
                    # Execute sells for positions that need exit
                    for position, trigger in exit_positions:
                        try:
                            # Determine strategy based on trigger
                            if trigger == SellTrigger.STOP_LOSS:
                                strategy = SellStrategy.EMERGENCY_EXIT
                                urgency = 0.9
                                max_slippage = 10.0
                            elif trigger == SellTrigger.TIMEOUT:
                                strategy = SellStrategy.IMMEDIATE
                                urgency = 0.8
                                max_slippage = 5.0
                            else:  # PROFIT_TARGET or MARKET_CONDITION
                                strategy = SellStrategy.BEST_EFFORT
                                urgency = 0.6
                                max_slippage = 3.0
                            
                            sell_signal = SellSignal(
                                position=position,
                                trigger=trigger,
                                strategy=strategy,
                                urgency=urgency,
                                max_slippage=max_slippage
                            )
                            
                            # Execute sell asynchronously
                            asyncio.create_task(self.execute_sell(sell_signal))
                            
                        except Exception as e:
                            self.logger.error(f"Error creating sell signal for {position.token_symbol}: {e}")
                
                # Wait before next monitoring cycle
                await asyncio.sleep(2)  # Monitor every 2 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(5)  # Longer delay on error
    
    async def _validate_sell_signal(self, sell_signal: SellSignal) -> bool:
        """Validate sell signal before execution"""
        try:
            position = sell_signal.position
            
            # Check if position exists
            tracked_position = self.position_manager.get_position(position.token_address)
            if not tracked_position:
                self.logger.warning(f"Position not found: {position.token_address}")
                return False
            
            # Check if position has tokens to sell
            if tracked_position.amount_tokens_held <= 0:
                self.logger.warning(f"No tokens to sell: {position.token_symbol}")
                return False
            
            # Check slippage limits
            if sell_signal.max_slippage > 50.0 and not sell_signal.emergency_exit:
                self.logger.warning(f"Slippage too high: {sell_signal.max_slippage}%")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sell signal validation error: {e}")
            return False
    
    async def _execute_immediate_sell(self, sell_signal: SellSignal, current_price: float) -> SellResult:
        """Execute immediate market sell"""
        
        position = sell_signal.position
        result = SellResult(success=False, status=SellStatus.EXECUTING)
        
        try:
            # Calculate amount to sell (in token's smallest unit)
            amount_tokens = int(position.amount_tokens_held * 1e6)  # Assuming 6 decimals
            
            # Get quote for selling
            quote_start = time.time()
            quote = await self.jupiter.get_quote(
                input_mint=position.token_address,
                output_mint="So11111111111111111111111111111111111112",  # SOL
                amount=amount_tokens,
                slippage_bps=int(sell_signal.max_slippage * 100)
            )
            result.quote_latency_ms = int((time.time() - quote_start) * 1000)
            
            if not quote:
                result.error_message = "Failed to get sell quote"
                return result
            
            # Validate quote
            price_impact = float(quote.get("priceImpactPct", 0))
            if price_impact > sell_signal.max_slippage and not sell_signal.ignore_slippage:
                result.error_message = f"Price impact too high: {price_impact:.2f}%"
                return result
            
            # Get swap transaction
            swap_tx = await self.jupiter.get_swap_transaction(quote, str(self.wallet.pubkey()))
            if not swap_tx:
                result.error_message = "Failed to get swap transaction"
                return result
            
            # Execute transaction
            execution_start = time.time()
            signature = await self._send_transaction(swap_tx)
            result.execution_latency_ms = int((time.time() - execution_start) * 1000)
            
            if signature:
                result.success = True
                result.signature = signature
                result.amount_tokens_sold = position.amount_tokens_held
                result.amount_sol_received = float(quote.get("outAmount", 0)) / 1e9
                result.average_sell_price = result.amount_sol_received / position.amount_tokens_held
                result.actual_slippage_percent = price_impact
            else:
                result.error_message = "Transaction failed to send"
            
            return result
            
        except Exception as e:
            result.error_message = f"Immediate sell error: {str(e)}"
            return result
    
    async def _execute_emergency_sell(self, sell_signal: SellSignal, current_price: float) -> SellResult:
        """Execute emergency sell with minimal checks"""
        
        # Emergency sell is similar to immediate sell but with relaxed constraints
        emergency_signal = SellSignal(
            position=sell_signal.position,
            trigger=sell_signal.trigger,
            strategy=SellStrategy.IMMEDIATE,
            urgency=1.0,
            max_slippage=min(sell_signal.max_slippage, 25.0),  # Cap at 25%
            ignore_slippage=True,
            timeout_seconds=10  # Very short timeout
        )
        
        return await self._execute_immediate_sell(emergency_signal, current_price)
    
    async def _execute_progressive_sell(self, sell_signal: SellSignal, current_price: float) -> SellResult:
        """Execute progressive sell in chunks to minimize market impact"""
        
        total_tokens = sell_signal.position.amount_tokens_held
        chunk_count = max(1, min(sell_signal.progressive_chunks, 5))  # Limit to 5 chunks max
        chunk_size = total_tokens / chunk_count
        
        results = []
        total_received = 0.0
        total_sold = 0.0
        
        self.logger.info(f"Starting progressive sell: {total_tokens:.2f} tokens in {chunk_count} chunks")
        
        for i in range(chunk_count):
            # Calculate chunk size (last chunk gets remainder)
            if i == chunk_count - 1:
                chunk_tokens = total_tokens - total_sold
            else:
                chunk_tokens = chunk_size
            
            # Create chunk sell signal
            chunk_signal = SellSignal(
                position=sell_signal.position,
                trigger=sell_signal.trigger,
                strategy=SellStrategy.IMMEDIATE,
                urgency=sell_signal.urgency,
                max_slippage=sell_signal.max_slippage,
                timeout_seconds=sell_signal.timeout_seconds
            )
            
            # Temporarily adjust position for this chunk
            original_tokens = sell_signal.position.amount_tokens_held
            sell_signal.position.amount_tokens_held = chunk_tokens
            
            try:
                chunk_result = await self._execute_immediate_sell(chunk_signal, current_price)
                results.append(chunk_result)
                
                if chunk_result.success:
                    total_received += chunk_result.amount_sol_received
                    total_sold += chunk_result.amount_tokens_sold
                else:
                    self.logger.warning(f"Progressive sell chunk {i+1} failed: {chunk_result.error_message}")
                    break
                
                # Brief delay between chunks to reduce market impact
                if i < chunk_count - 1:
                    await asyncio.sleep(2)
                    
            finally:
                # Restore original position
                sell_signal.position.amount_tokens_held = original_tokens
        
        # Calculate aggregated results
        success_rate = len([r for r in results if r.success]) / len(results) if results else 0
        all_successful = success_rate == 1.0
        
        average_price = total_received / total_sold if total_sold > 0 else 0
        
        return SellResult(
            success=all_successful,
            status=SellStatus.COMPLETED if all_successful else SellStatus.PARTIAL,
            amount_sol_received=total_received,
            amount_tokens_sold=total_sold,
            average_sell_price=average_price,
            trigger_reason=f"Progressive sell: {len(results)} chunks, {success_rate:.1%} success rate",
            total_latency_ms=sum(r.total_latency_ms for r in results),
            tx_fee_sol=sum(r.tx_fee_sol for r in results)
        )
    
    async def _execute_limit_sell(self, sell_signal: SellSignal, current_price: float) -> SellResult:
        """Execute limit sell with price monitoring and timeout"""
        
        start_time = time.time()
        target_price = sell_signal.price_limit or current_price * 1.02  # 2% above current if no limit set
        timeout_seconds = sell_signal.timeout_seconds
        check_interval = 5  # Check price every 5 seconds
        
        self.logger.info(f"Starting limit sell: target {target_price:.8f} SOL, timeout {timeout_seconds}s")
        
        # If current price already meets target, execute immediately
        if current_price >= target_price:
            self.logger.info(f"Limit price already met: {current_price:.8f} >= {target_price:.8f}")
            return await self._execute_immediate_sell(sell_signal, current_price)
        
        # Monitor price until target is reached or timeout
        while time.time() - start_time < timeout_seconds:
            try:
                # Get fresh price
                fresh_price = await self.price_monitor.get_current_price(
                    sell_signal.position.token_address
                )
                
                if fresh_price is None:
                    self.logger.warning("Failed to get current price for limit order")
                    await asyncio.sleep(check_interval)
                    continue
                
                # Update position with latest price
                sell_signal.position.update_price(fresh_price)
                
                # Check if target price reached
                if fresh_price >= target_price:
                    self.logger.info(f"Limit price reached: {fresh_price:.8f} >= {target_price:.8f}")
                    return await self._execute_immediate_sell(sell_signal, fresh_price)
                
                # Check for emergency exit conditions while waiting
                position = sell_signal.position
                if (position.should_sell_stop_loss() or 
                    position.should_sell_trailing_stop(fresh_price)):
                    self.logger.warning("Stop loss triggered during limit order, executing emergency exit")
                    emergency_signal = SellSignal(
                        position=position,
                        trigger=SellTrigger.STOP_LOSS,
                        strategy=SellStrategy.EMERGENCY_EXIT,
                        urgency=1.0,
                        emergency_exit=True
                    )
                    return await self._execute_emergency_sell(emergency_signal, fresh_price)
                
                # Wait before next check
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error during limit sell monitoring: {e}")
                await asyncio.sleep(check_interval)
        
        # Timeout reached - decide what to do
        elapsed_time = time.time() - start_time
        
        if sell_signal.emergency_exit:
            # If emergency exit is enabled, sell at current price
            self.logger.warning(f"Limit sell timeout ({elapsed_time:.1f}s), executing emergency exit")
            final_price = await self.price_monitor.get_current_price(
                sell_signal.position.token_address
            ) or current_price
            return await self._execute_immediate_sell(sell_signal, final_price)
        else:
            # Return failure if no emergency exit
            return SellResult(
                success=False,
                status=SellStatus.FAILED,
                error_message=f"Limit sell timeout after {elapsed_time:.1f}s - target price {target_price:.8f} not reached",
                trigger_reason="Limit order timeout",
                total_latency_ms=int(elapsed_time * 1000)
            )
    
    async def _send_transaction(self, transaction_base64: str) -> Optional[str]:
        """Send transaction to network"""
        try:
            import base64
            
            # Decode and sign transaction
            tx_bytes = base64.b64decode(transaction_base64)
            transaction = Transaction.deserialize(tx_bytes)
            transaction.sign(self.wallet)
            
            # Send transaction
            client = await self.rpc_manager.get_client()
            response = await client.send_transaction(
                transaction,
                opts={
                    "skip_preflight": False,
                    "preflight_commitment": "processed",
                    "max_retries": 2
                }
            )
            
            if response.value:
                return str(response.value)
            
        except Exception as e:
            self.logger.error(f"Transaction send error: {e}")
        
        return None
    
    async def _wait_for_sell_confirmation(self, signature: str, timeout_seconds: int = 30) -> bool:
        """Wait for sell transaction confirmation"""
        try:
            from solders.signature import Signature as SolanaSignature
            
            client = await self.rpc_manager.get_client()
            sig = SolanaSignature.from_string(signature)
            
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                try:
                    status = await client.get_signature_statuses([sig])
                    
                    if status.value and status.value[0]:
                        sig_status = status.value[0]
                        
                        if sig_status.err:
                            self.logger.error(f"Sell transaction failed: {sig_status.err}")
                            return False
                        
                        if sig_status.confirmation_status in ["confirmed", "finalized"]:
                            return True
                    
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    self.logger.warning(f"Confirmation check error: {e}")
                    await asyncio.sleep(2)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Confirmation wait error: {e}")
            return False
    
    async def _parse_sell_transaction_details(self, result: SellResult, position: Position, signature: str):
        """Parse sell transaction details"""
        try:
            from solders.signature import Signature as SolanaSignature
            
            client = await self.rpc_manager.get_client()
            sig = SolanaSignature.from_string(signature)
            
            tx_response = await client.get_transaction(
                sig,
                commitment="confirmed",
                max_supported_transaction_version=0
            )
            
            if tx_response.value and tx_response.value.transaction.meta:
                meta = tx_response.value.transaction.meta
                result.tx_fee_sol = meta.fee / 1e9
                
                # Additional parsing could be added here for more detailed analysis
                
        except Exception as e:
            self.logger.error(f"Failed to parse sell transaction details: {e}")
    
    def _update_average_hold_time(self, hold_time_seconds: int):
        """Update running average hold time"""
        if self.successful_sells == 1:
            self.average_hold_time = hold_time_seconds
        else:
            # Running average
            alpha = 0.1
            self.average_hold_time = alpha * hold_time_seconds + (1 - alpha) * self.average_hold_time
    
    async def _log_sell_result(self, sell_signal: SellSignal, result: SellResult):
        """Log sell result for monitoring"""
        try:
            position = sell_signal.position
            
            # Create trade record
            trade_record = TradeRecord(
                timestamp=result.timestamp,
                token_address=position.token_address,
                token_symbol=position.token_symbol,
                trade_type="SELL",
                success=result.success,
                amount_sol=result.amount_sol_received,
                amount_tokens=result.amount_tokens_sold,
                price=result.average_sell_price,
                slippage_percent=result.actual_slippage_percent,
                latency_ms=result.total_latency_ms,
                gas_cost_sol=result.tx_fee_sol,
                profit_loss_sol=result.profit_loss_sol,
                profit_loss_percent=result.profit_loss_percent,
                hold_time_seconds=result.hold_time_seconds,
                tx_signature=result.signature,
                exit_reason=result.trigger_reason,
                error_message=result.error_message
            )
            
            # Log to trading logger
            if trading_logger:
                await trading_logger.log_trade(trade_record)
            
            # Log summary
            if result.success:
                self.logger.info(
                    f"SELL SUCCESS: {position.token_symbol} | "
                    f"{result.amount_tokens_sold:.2f} tokens → {result.amount_sol_received:.4f} SOL | "
                    f"P&L: {result.profit_loss_sol:+.4f} SOL ({result.profit_loss_percent:+.1f}%) | "
                    f"Hold: {result.hold_time_seconds}s | Trigger: {result.trigger_reason}"
                )
            else:
                self.logger.warning(
                    f"SELL FAILED: {position.token_symbol} | "
                    f"Trigger: {result.trigger_reason} | "
                    f"Error: {result.error_message}"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to log sell result: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get seller performance metrics"""
        success_rate = (self.successful_sells / max(self.total_sells, 1)) * 100
        portfolio_value, portfolio_pnl = self.position_manager.get_portfolio_value()
        
        return {
            "total_sells": self.total_sells,
            "successful_sells": self.successful_sells,
            "success_rate": success_rate,
            "total_profit_sol": self.total_profit_sol,
            "average_hold_time": self.average_hold_time,
            "active_positions": len(self.position_manager.positions),
            "portfolio_value_sol": portfolio_value,
            "portfolio_pnl_sol": portfolio_pnl,
            "monitoring_active": self.monitoring_active,
            "initialized": self.initialized
        }
    
    def get_active_positions_summary(self) -> List[Dict[str, Any]]:
        """Get summary of active positions"""
        summaries = []
        
        for position in self.position_manager.get_all_positions():
            summaries.append({
                "token_symbol": position.token_symbol,
                "token_address": mask_sensitive_value(position.token_address),
                "entry_time": position.entry_timestamp.isoformat(),
                "amount_sol_invested": position.amount_sol_invested,
                "amount_tokens_held": position.amount_tokens_held,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "unrealized_pnl_sol": position.unrealized_pnl_sol,
                "unrealized_pnl_percent": position.unrealized_pnl_percent,
                "profit_target": position.profit_target_percent,
                "stop_loss": position.stop_loss_percent,
                "hold_time_seconds": int((datetime.utcnow() - position.entry_timestamp).total_seconds())
            })
        
        return summaries

# === GLOBAL INSTANCES ===

# Global seller instance
production_seller = ProductionSeller()

# Backward compatibility aliases
trade_seller = production_seller

# Export main classes
__all__ = [
    "Position", "SellSignal", "SellResult", "SellTrigger", "SellStrategy", "SellStatus",
    "ProductionSeller", "PositionManager", "PriceMonitor",
    "production_seller", "trade_seller"
] 