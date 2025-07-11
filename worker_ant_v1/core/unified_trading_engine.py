"""
UNIFIED TRADING ENGINE - CORE TRADING COORDINATOR
===============================================

Core trading engine that coordinates all trading operations across
the swarm, manages order execution, and provides unified trading interface.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
from worker_ant_v1.core.vault_wallet_system import VaultWalletSystem
from worker_ant_v1.intelligence.token_intelligence_system import TokenIntelligenceSystem
from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch

class OrderType(Enum):
    """Order types"""
    BUY = "buy"
    SELL = "sell"
    SWAP = "swap"
    STAKE = "stake"
    UNSTAKE = "unstake"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TradeOrder:
    """Trade order structure"""
    order_id: str
    token_address: str
    order_type: OrderType
    amount: float
    wallet_id: str
    status: OrderStatus
    created_at: datetime
    executed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradeResult:
    """Trade execution result"""
    success: bool
    order_id: str
    token_address: str
    amount_executed: float
    profit_sol: float
    gas_fee: float
    slippage: float
    execution_time_ms: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedTradingEngine:
    """Unified trading engine for coordinating all trading operations"""
    
    def __init__(self):
        self.logger = setup_logger("UnifiedTradingEngine")
        
        # Core systems
        self.wallet_manager: Optional[UnifiedWalletManager] = None
        self.vault_system: Optional[VaultWalletSystem] = None
        self.token_intelligence: Optional[TokenIntelligenceSystem] = None
        self.kill_switch: Optional[EnhancedKillSwitch] = None
        
        # Order management
        self.active_orders: Dict[str, TradeOrder] = {}
        self.order_history: List[TradeOrder] = []
        self.order_queue: asyncio.Queue = asyncio.Queue()
        
        # Performance tracking
        self.trading_metrics = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_profit': 0.0,
            'total_volume': 0.0,
            'avg_execution_time_ms': 0.0
        }
        
        # Configuration
        self.config = {
            'max_concurrent_orders': 5,
            'order_timeout_seconds': 30,
            'max_slippage': 0.05,  # 5%
            'min_profit_threshold': 0.001,  # 0.001 SOL
            'retry_failed_orders': True,
            'max_retries': 3
        }
        
        # System state
        self.initialized = False
        self.trading_active = False
        
    async def initialize(self) -> bool:
        """Initialize the unified trading engine"""
        try:
            self.logger.info("🚀 Initializing Unified Trading Engine...")
            
            # Get core systems
            from worker_ant_v1.core.wallet_manager import get_wallet_manager
            from worker_ant_v1.core.vault_wallet_system import get_vault_system
            from worker_ant_v1.intelligence.token_intelligence_system import get_token_intelligence_system
            
            self.wallet_manager = await get_wallet_manager()
            self.vault_system = await get_vault_system()
            self.token_intelligence = await get_token_intelligence_system()
            
            # Initialize kill switch
            self.kill_switch = EnhancedKillSwitch()
            await self.kill_switch.initialize()
            
            # Start background tasks
            asyncio.create_task(self._order_processing_loop())
            asyncio.create_task(self._order_monitoring_loop())
            asyncio.create_task(self._performance_tracking_loop())
            
            self.initialized = True
            self.trading_active = True
            self.logger.info("✅ Unified Trading Engine initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trading engine: {e}")
            return False
    
    async def execute_trade(self, token_address: str, trade_params: Dict[str, Any]) -> TradeResult:
        """Execute a trade with the given parameters"""
        try:
            # Check if trading is active
            if not self.trading_active:
                return TradeResult(
                    success=False,
                    order_id="",
                    token_address=token_address,
                    amount_executed=0.0,
                    profit_sol=0.0,
                    gas_fee=0.0,
                    slippage=0.0,
                    execution_time_ms=0,
                    error_message="Trading engine not active"
                )
            
            # Check kill switch
            if self.kill_switch and self.kill_switch.is_triggered():
                return TradeResult(
                    success=False,
                    order_id="",
                    token_address=token_address,
                    amount_executed=0.0,
                    profit_sol=0.0,
                    gas_fee=0.0,
                    slippage=0.0,
                    execution_time_ms=0,
                    error_message="Kill switch triggered"
                )
            
            # Validate trade parameters
            validation_result = await self._validate_trade_params(token_address, trade_params)
            if not validation_result['valid']:
                return TradeResult(
                    success=False,
                    order_id="",
                    token_address=token_address,
                    amount_executed=0.0,
                    profit_sol=0.0,
                    gas_fee=0.0,
                    slippage=0.0,
                    execution_time_ms=0,
                    error_message=validation_result['error']
                )
            
            # Create trade order
            order = await self._create_trade_order(token_address, trade_params)
            
            # Add to queue
            await self.order_queue.put(order)
            
            # Wait for execution
            result = await self._wait_for_order_completion(order.order_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return TradeResult(
                success=False,
                order_id="",
                token_address=token_address,
                amount_executed=0.0,
                profit_sol=0.0,
                gas_fee=0.0,
                slippage=0.0,
                execution_time_ms=0,
                error_message=str(e)
            )
    
    async def _validate_trade_params(self, token_address: str, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade parameters"""
        try:
            # Check required parameters
            required_params = ['wallet_id', 'amount']
            for param in required_params:
                if param not in trade_params:
                    return {'valid': False, 'error': f"Missing required parameter: {param}"}
            
            wallet_id = trade_params['wallet_id']
            amount = trade_params['amount']
            
            # Validate wallet
            if not self.wallet_manager or wallet_id not in self.wallet_manager.wallets:
                return {'valid': False, 'error': f"Invalid wallet ID: {wallet_id}"}
            
            # Validate amount
            if amount <= 0:
                return {'valid': False, 'error': "Amount must be positive"}
            
            # Check wallet balance
            wallet = self.wallet_manager.wallets[wallet_id]
            if hasattr(wallet, 'balance') and wallet.balance < amount:
                return {'valid': False, 'error': f"Insufficient balance: {wallet.balance} < {amount}"}
            
            # Validate token (basic check)
            if not token_address or len(token_address) < 10:
                return {'valid': False, 'error': "Invalid token address"}
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            self.logger.error(f"Error validating trade params: {e}")
            return {'valid': False, 'error': f"Validation error: {str(e)}"}
    
    async def _create_trade_order(self, token_address: str, trade_params: Dict[str, Any]) -> TradeOrder:
        """Create a trade order"""
        try:
            order_id = f"order_{uuid.uuid4().hex[:8]}"
            
            order = TradeOrder(
                order_id=order_id,
                token_address=token_address,
                order_type=OrderType.BUY,  # Default to buy
                amount=trade_params['amount'],
                wallet_id=trade_params['wallet_id'],
                status=OrderStatus.PENDING,
                created_at=datetime.now(),
                metadata=trade_params
            )
            
            # Store order
            self.active_orders[order_id] = order
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error creating trade order: {e}")
            raise
    
    async def _wait_for_order_completion(self, order_id: str, timeout_seconds: int = 30) -> TradeResult:
        """Wait for order completion"""
        try:
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < timeout_seconds:
                if order_id in self.active_orders:
                    order = self.active_orders[order_id]
                    
                    if order.status == OrderStatus.COMPLETED:
                        return self._create_trade_result(order, True)
                    elif order.status == OrderStatus.FAILED:
                        return self._create_trade_result(order, False)
                    
                    await asyncio.sleep(0.1)  # Wait 100ms
                else:
                    # Order completed and removed from active orders
                    return self._create_trade_result(None, True)
            
            # Timeout
            return TradeResult(
                success=False,
                order_id=order_id,
                token_address="",
                amount_executed=0.0,
                profit_sol=0.0,
                gas_fee=0.0,
                slippage=0.0,
                execution_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                error_message="Order execution timeout"
            )
            
        except Exception as e:
            self.logger.error(f"Error waiting for order completion: {e}")
            return TradeResult(
                success=False,
                order_id=order_id,
                token_address="",
                amount_executed=0.0,
                profit_sol=0.0,
                gas_fee=0.0,
                slippage=0.0,
                execution_time_ms=0,
                error_message=str(e)
            )
    
    def _create_trade_result(self, order: Optional[TradeOrder], success: bool) -> TradeResult:
        """Create trade result from order"""
        try:
            if not order:
                return TradeResult(
                    success=False,
                    order_id="",
                    token_address="",
                    amount_executed=0.0,
                    profit_sol=0.0,
                    gas_fee=0.0,
                    slippage=0.0,
                    execution_time_ms=0,
                    error_message="No order data"
                )
            
            execution_time = (order.executed_at - order.created_at).total_seconds() * 1000 if order.executed_at else 0
            
            if success and order.result:
                return TradeResult(
                    success=True,
                    order_id=order.order_id,
                    token_address=order.token_address,
                    amount_executed=order.result.get('amount_executed', order.amount),
                    profit_sol=order.result.get('profit_sol', 0.0),
                    gas_fee=order.result.get('gas_fee', 0.0),
                    slippage=order.result.get('slippage', 0.0),
                    execution_time_ms=int(execution_time),
                    metadata=order.result
                )
            else:
                return TradeResult(
                    success=False,
                    order_id=order.order_id,
                    token_address=order.token_address,
                    amount_executed=0.0,
                    profit_sol=0.0,
                    gas_fee=0.0,
                    slippage=0.0,
                    execution_time_ms=int(execution_time),
                    error_message=order.result.get('error_message', 'Unknown error') if order.result else 'Order failed'
                )
                
        except Exception as e:
            self.logger.error(f"Error creating trade result: {e}")
            return TradeResult(
                success=False,
                order_id=order.order_id if order else "",
                token_address=order.token_address if order else "",
                amount_executed=0.0,
                profit_sol=0.0,
                gas_fee=0.0,
                slippage=0.0,
                execution_time_ms=0,
                error_message=str(e)
            )
    
    async def _order_processing_loop(self):
        """Main order processing loop"""
        while self.trading_active:
            try:
                # Process orders from queue
                while not self.order_queue.empty() and len(self.active_orders) < self.config['max_concurrent_orders']:
                    order = await self.order_queue.get()
                    asyncio.create_task(self._process_order(order))
                
                await asyncio.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                self.logger.error(f"Order processing loop error: {e}")
                await asyncio.sleep(1)
    
    async def _process_order(self, order: TradeOrder):
        """Process a single order"""
        try:
            self.logger.info(f"🔄 Processing order {order.order_id} for {order.token_address}")
            
            # Update order status
            order.status = OrderStatus.EXECUTING
            
            # Execute the trade
            result = await self._execute_order(order)
            
            # Update order with result
            order.result = result
            order.executed_at = datetime.now()
            order.status = OrderStatus.COMPLETED if result.get('success', False) else OrderStatus.FAILED
            
            # Update metrics
            self._update_trading_metrics(order, result)
            
            # Move to history
            self.order_history.append(order)
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
            
            # Keep only last 1000 orders in history
            if len(self.order_history) > 1000:
                self.order_history = self.order_history[-1000:]
            
            self.logger.info(f"✅ Order {order.order_id} completed: {result.get('success', False)}")
            
        except Exception as e:
            self.logger.error(f"Error processing order {order.order_id}: {e}")
            
            # Mark order as failed
            order.result = {'success': False, 'error_message': str(e)}
            order.executed_at = datetime.now()
            order.status = OrderStatus.FAILED
            
            # Update metrics
            self.trading_metrics['failed_orders'] += 1
    
    async def _execute_order(self, order: TradeOrder) -> Dict[str, Any]:
        """Execute a single order"""
        try:
            start_time = datetime.now()
            
            # Get token analysis
            token_analysis = await self.token_intelligence.analyze_token(order.token_address)
            
            # Check if we should proceed based on analysis
            if token_analysis.opportunity_level.value == 'avoid':
                return {
                    'success': False,
                    'error_message': 'Token flagged as avoid by intelligence system'
                }
            
            # Simulate trade execution (placeholder)
            # In a real implementation, this would execute the actual trade
            await asyncio.sleep(0.1)  # Simulate execution time
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Simulate trade result
            success = True  # 90% success rate for demo
            if success:
                profit = order.amount * 0.05  # 5% profit
                gas_fee = order.amount * 0.001  # 0.1% gas fee
                slippage = 0.01  # 1% slippage
                
                return {
                    'success': True,
                    'amount_executed': order.amount,
                    'profit_sol': profit,
                    'gas_fee': gas_fee,
                    'slippage': slippage,
                    'execution_time_ms': int(execution_time),
                    'token_analysis': {
                        'confidence_score': token_analysis.confidence_score,
                        'risk_score': token_analysis.risk_score,
                        'recommendation': token_analysis.recommendation
                    }
                }
            else:
                return {
                    'success': False,
                    'error_message': 'Trade execution failed',
                    'execution_time_ms': int(execution_time)
                }
            
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }
    
    def _update_trading_metrics(self, order: TradeOrder, result: Dict[str, Any]):
        """Update trading metrics"""
        try:
            self.trading_metrics['total_orders'] += 1
            
            if result.get('success', False):
                self.trading_metrics['successful_orders'] += 1
                self.trading_metrics['total_profit'] += result.get('profit_sol', 0.0)
                self.trading_metrics['total_volume'] += result.get('amount_executed', 0.0)
            else:
                self.trading_metrics['failed_orders'] += 1
            
            # Update average execution time
            execution_time = result.get('execution_time_ms', 0)
            if execution_time > 0:
                current_avg = self.trading_metrics['avg_execution_time_ms']
                total_orders = self.trading_metrics['total_orders']
                self.trading_metrics['avg_execution_time_ms'] = (
                    (current_avg * (total_orders - 1) + execution_time) / total_orders
                )
            
        except Exception as e:
            self.logger.error(f"Error updating trading metrics: {e}")
    
    async def _order_monitoring_loop(self):
        """Monitor active orders for timeouts"""
        while self.trading_active:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                current_time = datetime.now()
                timeout_orders = []
                
                for order_id, order in self.active_orders.items():
                    if order.status == OrderStatus.EXECUTING:
                        age_seconds = (current_time - order.created_at).total_seconds()
                        if age_seconds > self.config['order_timeout_seconds']:
                            timeout_orders.append(order_id)
                
                # Handle timeout orders
                for order_id in timeout_orders:
                    order = self.active_orders[order_id]
                    order.status = OrderStatus.FAILED
                    order.result = {
                        'success': False,
                        'error_message': 'Order execution timeout'
                    }
                    order.executed_at = current_time
                    
                    self.logger.warning(f"⏰ Order {order_id} timed out")
                    
                    # Move to history
                    self.order_history.append(order)
                    del self.active_orders[order_id]
                    
                    # Update metrics
                    self.trading_metrics['failed_orders'] += 1
                
            except Exception as e:
                self.logger.error(f"Order monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _performance_tracking_loop(self):
        """Track and log performance metrics"""
        while self.trading_active:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Log performance summary
                await self._log_performance_summary()
                
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(300)
    
    async def _log_performance_summary(self):
        """Log performance summary"""
        try:
            # Log every 30 minutes
            if hasattr(self, '_last_performance_log'):
                if (datetime.now() - self._last_performance_log).total_seconds() < 1800:
                    return
            
            self._last_performance_log = datetime.now()
            
            success_rate = 0.0
            if self.trading_metrics['total_orders'] > 0:
                success_rate = self.trading_metrics['successful_orders'] / self.trading_metrics['total_orders']
            
            self.logger.info("📊 Trading Performance Summary:")
            self.logger.info(f"   Total Orders: {self.trading_metrics['total_orders']}")
            self.logger.info(f"   Success Rate: {success_rate:.1%}")
            self.logger.info(f"   Total Profit: {self.trading_metrics['total_profit']:.6f} SOL")
            self.logger.info(f"   Total Volume: {self.trading_metrics['total_volume']:.6f} SOL")
            self.logger.info(f"   Avg Execution Time: {self.trading_metrics['avg_execution_time_ms']:.1f}ms")
            self.logger.info(f"   Active Orders: {len(self.active_orders)}")
            
        except Exception as e:
            self.logger.error(f"Error logging performance summary: {e}")
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get trading engine status"""
        try:
            success_rate = 0.0
            if self.trading_metrics['total_orders'] > 0:
                success_rate = self.trading_metrics['successful_orders'] / self.trading_metrics['total_orders']
            
            return {
                'initialized': self.initialized,
                'trading_active': self.trading_active,
                'active_orders': len(self.active_orders),
                'queue_size': self.order_queue.qsize(),
                'total_orders': self.trading_metrics['total_orders'],
                'success_rate': success_rate,
                'total_profit': self.trading_metrics['total_profit'],
                'total_volume': self.trading_metrics['total_volume'],
                'avg_execution_time_ms': self.trading_metrics['avg_execution_time_ms']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading status: {e}")
            return {}
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order.status = OrderStatus.CANCELLED
                order.executed_at = datetime.now()
                order.result = {
                    'success': False,
                    'error_message': 'Order cancelled by user'
                }
                
                # Move to history
                self.order_history.append(order)
                del self.active_orders[order_id]
                
                self.logger.info(f"❌ Order {order_id} cancelled")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the trading engine"""
        try:
            self.logger.info("🛑 Shutting down trading engine...")
            self.trading_active = False
            
            # Cancel all active orders
            for order_id in list(self.active_orders.keys()):
                await self.cancel_order(order_id)
            
            self.logger.info("✅ Trading engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Global instance
_trading_engine = None

async def get_trading_engine() -> UnifiedTradingEngine:
    """Get global trading engine instance"""
    global _trading_engine
    if _trading_engine is None:
        _trading_engine = UnifiedTradingEngine()
        await _trading_engine.initialize()
    return _trading_engine 