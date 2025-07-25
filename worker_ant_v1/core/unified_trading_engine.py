"""
UNIFIED TRADING ENGINE - PRODUCTION READY
========================================

Real trading execution engine with Jupiter DEX integration,
actual transaction signing, and live position tracking.
"""

import asyncio
import time
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import logging
import json

try:
    from solana.rpc.async_api import AsyncClient
    from solana.transaction import Transaction
    from solana.keypair import Keypair
    from solana.rpc.commitment import Commitment
except ImportError:
    from ..utils.solana_compat import AsyncClient, Transaction, Keypair, Commitment

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.core.unified_config import get_trading_config

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
    transaction_signature: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedTradingEngine:
    """Production-ready unified trading engine with real DEX integration"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_trading_config()
        
        # Core systems
        self.wallet_manager = None
        self.vault_system = None
        self.token_intelligence = None
        self.kill_switch = None
        
        # Jupiter DEX integration
        self.jupiter_base_url = "https://quote-api.jup.ag/v6"
        self.jupiter_quote_url = f"{self.jupiter_base_url}/quote"
        self.jupiter_swap_url = f"{self.jupiter_base_url}/swap"
        self.jupiter_price_url = f"{self.jupiter_base_url}/price"
        
        # Solana RPC client
        self.rpc_client = AsyncClient(self.config.get('solana_rpc_url', 'https://api.mainnet-beta.solana.com'))
        
        # Trading state
        self.initialized = False
        self.trading_active = False
        self.active_orders: Dict[str, TradeOrder] = {}
        self.order_history: List[TradeOrder] = []
        self.order_queue = asyncio.Queue()
        
        # Performance tracking
        self.trading_metrics = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_profit': 0.0,
            'total_volume': 0.0,
            'avg_execution_time_ms': 0.0
        }
        
        # Rate limiting
        self.last_request_time = 0.0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Token mints for common tokens
        self.token_mints = {
            'SOL': 'So11111111111111111111111111111111111111112',
            'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
            'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
            'WSOL': 'So11111111111111111111111111111111111111112'
        }
        
        self.logger.info("🚀 Unified Trading Engine initialized")

    async def initialize(self) -> bool:
        """Initialize the unified trading engine"""
        try:
            self.logger.info("🚀 Initializing Unified Trading Engine...")
            
            # Get core systems
            from worker_ant_v1.core.wallet_manager import get_wallet_manager
            from worker_ant_v1.safety.vault_wallet_system import get_vault_system
            from worker_ant_v1.intelligence.token_intelligence_system import get_token_intelligence_system
            
            self.wallet_manager = await get_wallet_manager()
            self.vault_system = await get_vault_system()
            self.token_intelligence = await get_token_intelligence_system()
            
            # Initialize kill switch
            from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch
            self.kill_switch = EnhancedKillSwitch()
            await self.kill_switch.initialize()
            
            # Test RPC connection
            try:
                await self.rpc_client.get_health()
                self.logger.info("✅ Solana RPC connection established")
            except Exception as e:
                self.logger.error(f"❌ Solana RPC connection failed: {e}")
                return False
            
            # Start background tasks
            asyncio.create_task(self._order_processing_loop())
            asyncio.create_task(self._order_monitoring_loop())
            asyncio.create_task(self._performance_tracking_loop())
            
            self.initialized = True
            self.trading_active = True
            
            self.logger.info("✅ Unified Trading Engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Trading engine initialization failed: {e}")
            return False

    async def execute_trade(self, token_address: str, trade_params: Dict[str, Any]) -> TradeResult:
        """Execute a real trade via Jupiter DEX"""
        try:
            # Validate trade parameters
            validation_result = await self._validate_trade_params(token_address, trade_params)
            if not validation_result['valid']:
                return self._create_trade_result(None, False, error_message=validation_result['error'])
            
            # Check kill switch
            if self.kill_switch and self.kill_switch.is_triggered:
                return self._create_trade_result(None, False, error_message="Kill switch triggered")
            
            # Get wallet for trade execution
            wallet_id = trade_params.get('wallet_id')
            if not wallet_id or not self.wallet_manager:
                return self._create_trade_result(None, False, error_message="No wallet specified or wallet manager not available")
            
            # Get wallet and perform self-assessment
            wallet = await self.wallet_manager.get_wallet_info(wallet_id)
            if not wallet:
                return self._create_trade_result(None, False, error_message=f"Wallet {wallet_id} not found")
            
            # Check for active squad ruleset first, then personal risk profile
            if wallet.get('active_squad_ruleset'):
                # Use squad rules for assessment
                if not self._assess_squad_trade(wallet, trade_params):
                    return self._create_trade_result(None, False, error_message=f"Wallet {wallet_id} vetoed trade due to squad rules")
            else:
                # Use personal risk profile for assessment
                if not self._assess_personal_trade(wallet, trade_params):
                    return self._create_trade_result(None, False, error_message=f"Wallet {wallet_id} vetoed trade due to personal risk profile")
            
            # Create trade order
            order = await self._create_trade_order(token_address, trade_params)
            self.active_orders[order.order_id] = order
            
            # Wait for order completion
            result = await self._wait_for_order_completion(order.order_id, timeout_seconds=30)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return self._create_trade_result(None, False, error_message=str(e))
    
    async def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data for trading decisions"""
        try:
            processed_data = {
                'token_address': market_data.get('token_address', ''),
                'symbol': market_data.get('symbol', ''),
                'price': float(market_data.get('price', 0.0)),
                'volume_24h': float(market_data.get('volume_24h', 0.0)),
                'liquidity': float(market_data.get('liquidity', 0.0)),
                'price_change_24h': float(market_data.get('price_change_24h', 0.0)),
                'price_change_1h': float(market_data.get('price_change_1h', 0.0)),
                'holder_count': int(market_data.get('holder_count', 0)),
                'age_hours': float(market_data.get('age_hours', 0.0)),
                'processed_at': datetime.now().isoformat()
            }
            
            # Calculate additional metrics
            processed_data['volatility'] = abs(processed_data['price_change_24h'])
            processed_data['volume_price_ratio'] = processed_data['volume_24h'] / max(processed_data['price'], 0.0001)
            processed_data['liquidity_ratio'] = processed_data['liquidity'] / max(processed_data['volume_24h'], 0.0001)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return market_data
    
    async def generate_trading_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from market data"""
        try:
            signals = {
                'buy_signal': False,
                'sell_signal': False,
                'hold_signal': True,
                'confidence': 0.0,
                'risk_level': 'medium',
                'expected_profit': 0.0,
                'stop_loss': 0.0,
                'take_profit': 0.0
            }
            
            # Simple signal generation logic
            price_change_24h = market_data.get('price_change_24h', 0.0)
            volume_24h = market_data.get('volume_24h', 0.0)
            liquidity = market_data.get('liquidity', 0.0)
            
            # Buy signal conditions
            if (price_change_24h > 5.0 and  # 5% price increase
                volume_24h > 100.0 and      # $100+ volume
                liquidity > 10.0):          # $10+ liquidity
                signals['buy_signal'] = True
                signals['hold_signal'] = False
                signals['confidence'] = 0.7
                signals['expected_profit'] = 0.1  # 10% expected profit
                signals['stop_loss'] = -0.05      # 5% stop loss
                signals['take_profit'] = 0.15     # 15% take profit
            
            # Sell signal conditions
            elif price_change_24h < -10.0:  # 10% price decrease
                signals['sell_signal'] = True
                signals['hold_signal'] = False
                signals['confidence'] = 0.8
                signals['risk_level'] = 'high'
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            return {'buy_signal': False, 'sell_signal': False, 'hold_signal': True, 'confidence': 0.0}

    async def _validate_trade_params(self, token_address: str, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade parameters"""
        try:
            # Check required parameters
            required_params = ['amount', 'wallet_id', 'order_type']
            for param in required_params:
                if param not in trade_params:
                    return {'valid': False, 'error': f"Missing required parameter: {param}"}
            
            # Validate amount
            amount = trade_params['amount']
            if amount <= 0:
                return {'valid': False, 'error': "Amount must be positive"}
            
            # Validate wallet exists
            wallet_id = trade_params['wallet_id']
            if not await self.wallet_manager.wallet_exists(wallet_id):
                return {'valid': False, 'error': f"Wallet {wallet_id} not found"}
            
            # Check wallet balance
            wallet_balance = await self.wallet_manager.get_wallet_balance(wallet_id)
            if wallet_balance < amount:
                return {'valid': False, 'error': f"Insufficient balance: {wallet_balance} < {amount}"}
            
            # Validate token address
            if not token_address or len(token_address) != 44:
                return {'valid': False, 'error': "Invalid token address"}
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    async def _create_trade_order(self, token_address: str, trade_params: Dict[str, Any]) -> TradeOrder:
        """Create a new trade order"""
        order_id = f"order_{int(time.time() * 1000)}_{token_address[:8]}"
        
        order = TradeOrder(
            order_id=order_id,
            token_address=token_address,
            order_type=OrderType(trade_params['order_type']),
            amount=trade_params['amount'],
            wallet_id=trade_params['wallet_id'],
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            metadata=trade_params.get('metadata', {})
        )
        
        await self.order_queue.put(order)
        return order

    async def _wait_for_order_completion(self, order_id: str, timeout_seconds: int = 30) -> TradeResult:
        """Wait for order completion"""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                if order.status in [OrderStatus.COMPLETED, OrderStatus.FAILED]:
                    # Move to history
                    self.order_history.append(order)
                    del self.active_orders[order_id]
                    
                    if order.status == OrderStatus.COMPLETED and order.result:
                        return TradeResult(
                            success=True,
                            order_id=order_id,
                            token_address=order.token_address,
                            amount_executed=order.result.get('amount_executed', 0),
                            profit_sol=order.result.get('profit_sol', 0),
                            gas_fee=order.result.get('gas_fee', 0),
                            slippage=order.result.get('slippage', 0),
                            execution_time_ms=order.result.get('execution_time_ms', 0),
                            transaction_signature=order.result.get('transaction_signature')
                        )
                    else:
                        return TradeResult(
                            success=False,
                            order_id=order_id,
                            token_address=order.token_address,
                            amount_executed=0,
                            profit_sol=0,
                            gas_fee=0,
                            slippage=0,
                            execution_time_ms=0,
                            error_message=order.result.get('error_message', 'Order failed')
                        )
            
            await asyncio.sleep(0.1)
        
        # Timeout
        return TradeResult(
            success=False,
            order_id=order_id,
            token_address="",
            amount_executed=0,
            profit_sol=0,
            gas_fee=0,
            slippage=0,
            execution_time_ms=0,
            error_message="Order execution timeout"
        )

    def _assess_personal_trade(self, wallet: Dict[str, Any], trade_params: Dict[str, Any]) -> bool:
        """Assess trade using wallet's personal risk profile and genetics"""
        try:
            genetics = wallet.get('genetics', {})
            personal_profile = wallet.get('personal_risk_profile', {})
            
            risk_level = trade_params.get('risk_level', 'medium')
            position_size = trade_params.get('position_size_sol', 0.0)
            token_age_hours = trade_params.get('token_age_hours', 24)
            
            # Check aggression level against risk
            aggression = genetics.get('aggression', 0.5)
            if risk_level == 'high' and aggression < 0.6:
                self.logger.info(f"Wallet vetoed high-risk trade (aggression: {aggression})")
                return False
            
            # Check patience level against token age
            patience = genetics.get('patience', 0.5)
            if token_age_hours < 1 and patience > 0.7:
                self.logger.info(f"Wallet vetoed new token trade (patience: {patience})")
                return False
            
            # Check position size against personal limits
            max_position = personal_profile.get('max_position_size_sol', 10.0)
            if position_size > max_position:
                self.logger.info(f"Wallet vetoed large position (size: {position_size}, max: {max_position})")
                return False
            
            # Check herd immunity against trending tokens
            herd_immunity = genetics.get('herd_immunity', 0.5)
            if trade_params.get('is_trending', False) and herd_immunity > 0.8:
                self.logger.info(f"Wallet vetoed trending token (herd_immunity: {herd_immunity})")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in personal trade assessment: {e}")
            return True  # Default to allowing trade if assessment fails
    
    def _assess_squad_trade(self, wallet: Dict[str, Any], trade_params: Dict[str, Any]) -> bool:
        """Assess trade using squad ruleset"""
        try:
            squad_rules = wallet.get('active_squad_ruleset', {})
            if not squad_rules:
                return True
            
            position_size = trade_params.get('position_size_sol', 0.0)
            squad_max_position = squad_rules.get('max_position_size_sol', 50.0)
            
            # Only check squad-specific limits
            if position_size > squad_max_position:
                self.logger.info(f"Wallet vetoed squad trade (size: {position_size}, squad_max: {squad_max_position})")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in squad trade assessment: {e}")
            return True  # Default to allowing trade if assessment fails
    
    async def set_blitzscaling_mode(self, active: bool):
        """Set blitzscaling mode for the trading engine"""
        self.logger.info(f"🚀 Blitzscaling mode {'ACTIVATED' if active else 'DEACTIVATED'} for trading engine")
        # Additional blitzscaling logic can be added here

    def _create_trade_result(self, order: Optional[TradeOrder], success: bool, **kwargs) -> TradeResult:
        """Create trade result"""
        return TradeResult(
            success=success,
            order_id=order.order_id if order else "",
            token_address=order.token_address if order else "",
            amount_executed=kwargs.get('amount_executed', 0),
            profit_sol=kwargs.get('profit_sol', 0),
            gas_fee=kwargs.get('gas_fee', 0),
            slippage=kwargs.get('slippage', 0),
            execution_time_ms=kwargs.get('execution_time_ms', 0),
            transaction_signature=kwargs.get('transaction_signature'),
            error_message=kwargs.get('error_message')
        )

    async def _order_processing_loop(self):
        """Process orders from queue"""
        while self.trading_active:
            try:
                order = await self.order_queue.get()
                await self._process_order(order)
            except Exception as e:
                self.logger.error(f"Order processing error: {e}")
                await asyncio.sleep(1)

    async def _process_order(self, order: TradeOrder):
        """Process a single order"""
        try:
            order.status = OrderStatus.EXECUTING
            self.logger.info(f"🔄 Processing order {order.order_id}: {order.order_type.value} {order.amount} SOL")
            
            # Execute the order
            result = await self._execute_order(order)
            
            # Update order
            order.status = OrderStatus.COMPLETED if result['success'] else OrderStatus.FAILED
            order.result = result
            order.executed_at = datetime.now()
            
            # Update metrics
            self._update_trading_metrics(order, result)
            
            # CRITICAL: Process profits for successful sell orders
            if result['success'] and order.order_type == OrderType.SELL:
                profit_sol = result.get('profit_sol', 0.0)
                if profit_sol > 0:
                    # MANDATORY: Send profits to vault system with transaction safety
                    await self._process_profit_to_vault(profit_sol, order.order_id)
            
            if result['success']:
                self.logger.info(f"✅ Order {order.order_id} completed successfully")
            else:
                self.logger.error(f"❌ Order {order.order_id} failed: {result.get('error_message')}")
                
        except Exception as e:
            self.logger.error(f"Error processing order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
            order.result = {'success': False, 'error_message': str(e)}
            order.executed_at = datetime.now()

    async def _execute_order(self, order: TradeOrder) -> Dict[str, Any]:
        """Execute a real order via Jupiter DEX"""
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
            
            # Get wallet keypair
            wallet_keypair = await self.wallet_manager.get_wallet_keypair(order.wallet_id)
            if not wallet_keypair:
                error_msg = f"CRITICAL FAILURE: Could not retrieve keypair for wallet {order.wallet_id}. Aborting trade."
                self.logger.critical(error_msg)
                return {
                    'success': False,
                    'error_message': error_msg
                }
            
            # Execute real trade via Jupiter
            if order.order_type == OrderType.BUY:
                result = await self._execute_buy_order(order, wallet_keypair)
            elif order.order_type == OrderType.SELL:
                result = await self._execute_sell_order(order, wallet_keypair)
            else:
                return {
                    'success': False,
                    'error_message': f'Unsupported order type: {order.order_type.value}'
                }
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if result['success']:
                result['execution_time_ms'] = int(execution_time)
                result['token_analysis'] = {
                    'confidence_score': token_analysis.confidence_score,
                    'risk_score': token_analysis.risk_score,
                    'recommendation': token_analysis.recommendation
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return {
                'success': False,
                'error_message': str(e)
            }

    async def _execute_buy_order(self, order: TradeOrder, wallet_keypair: Keypair) -> Dict[str, Any]:
        """Execute a real buy order via Jupiter"""
        try:
            # Convert SOL amount to lamports
            sol_lamports = int(order.amount * 1_000_000_000)
            
            # Get swap quote from Jupiter
            quote = await self._get_jupiter_quote(
                input_mint=self.token_mints['SOL'],
                output_mint=order.token_address,
                amount=sol_lamports,
                slippage_bps=100  # 1% slippage
            )
            
            if not quote:
                return {
                    'success': False,
                    'error_message': 'Failed to get Jupiter quote'
                }
            
            # Execute swap
            swap_result = await self._execute_jupiter_swap(quote, wallet_keypair)
            
            if swap_result['success']:
                # Calculate real profit/loss
                token_amount = swap_result['output_amount'] / 1_000_000_000  # Convert from lamports
                gas_fee = swap_result.get('gas_fee', 0.001)  # Estimate gas fee
                
                return {
                    'success': True,
                    'amount_executed': order.amount,
                    'profit_sol': 0.0,  # Will be calculated when selling
                    'gas_fee': gas_fee,
                    'slippage': quote.get('price_impact_pct', 0),
                    'transaction_signature': swap_result['transaction_signature'],
                    'token_amount': token_amount
                }
            else:
                return {
                    'success': False,
                    'error_message': swap_result.get('error_message', 'Swap execution failed')
                }
                
        except Exception as e:
            return {
                'success': False,
                'error_message': str(e)
            }

    async def _execute_sell_order(self, order: TradeOrder, wallet_keypair: Keypair) -> Dict[str, Any]:
        """Execute a real sell order via Jupiter"""
        try:
            # Get token balance
            token_balance = await self.wallet_manager.get_token_balance(order.wallet_id, order.token_address)
            if token_balance < order.amount:
                return {
                    'success': False,
                    'error_message': f'Insufficient token balance: {token_balance} < {order.amount}'
                }
            
            # Convert token amount to lamports (assuming 9 decimals)
            token_lamports = int(order.amount * 1_000_000_000)
            
            # Get swap quote from Jupiter
            quote = await self._get_jupiter_quote(
                input_mint=order.token_address,
                output_mint=self.token_mints['SOL'],
                amount=token_lamports,
                slippage_bps=100  # 1% slippage
            )
            
            if not quote:
                return {
                    'success': False,
                    'error_message': 'Failed to get Jupiter quote'
                }
            
            # Execute swap
            swap_result = await self._execute_jupiter_swap(quote, wallet_keypair)
            
            if swap_result['success']:
                # Calculate real profit/loss
                sol_received = swap_result['output_amount'] / 1_000_000_000  # Convert from lamports
                gas_fee = swap_result.get('gas_fee', 0.001)  # Estimate gas fee
                profit_sol = sol_received - order.amount - gas_fee
                
                return {
                    'success': True,
                    'amount_executed': order.amount,
                    'profit_sol': profit_sol,
                    'gas_fee': gas_fee,
                    'slippage': quote.get('price_impact_pct', 0),
                    'transaction_signature': swap_result['transaction_signature'],
                    'sol_received': sol_received
                }
            else:
                return {
                    'success': False,
                    'error_message': swap_result.get('error_message', 'Swap execution failed')
                }
                
        except Exception as e:
            return {
                'success': False,
                'error_message': str(e)
            }

    async def _get_jupiter_quote(self, input_mint: str, output_mint: str, amount: int, slippage_bps: int) -> Optional[Dict]:
        """Get swap quote from Jupiter"""
        try:
            await self._rate_limit()
            
            params = {
                'inputMint': input_mint,
                'outputMint': output_mint,
                'amount': str(amount),
                'slippageBps': str(slippage_bps),
                'onlyDirectRoutes': 'false',
                'asLegacyTransaction': 'false'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.jupiter_quote_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'data' in data and len(data['data']) > 0:
                            route = data['data'][0]  # Best route
                            
                            return {
                                'input_mint': input_mint,
                                'output_mint': output_mint,
                                'input_amount': int(route['inAmount']),
                                'output_amount': int(route['outAmount']),
                                'price_impact_pct': float(route.get('priceImpactPct', 0)),
                                'market_infos': route.get('marketInfos', []),
                                'route_plan': route.get('routePlan', []),
                                'platform_fee': route.get('platformFee'),
                                'slippage_bps': slippage_bps
                            }
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Jupiter quote request failed: {response.status} - {error_text}")
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting Jupiter quote: {e}")
            return None

    async def _execute_jupiter_swap(self, quote: Dict, wallet_keypair: Keypair) -> Dict[str, Any]:
        """Execute swap transaction via Jupiter"""
        try:
            # Prepare swap request
            swap_request = {
                'quoteResponse': {
                    'inputMint': quote['input_mint'],
                    'inAmount': str(quote['input_amount']),
                    'outputMint': quote['output_mint'],
                    'outAmount': str(quote['output_amount']),
                    'otherAmountThreshold': str(int(quote['output_amount'] * 0.99)),  # 1% slippage
                    'swapMode': 'ExactIn',
                    'slippageBps': quote['slippage_bps'],
                    'platformFee': quote.get('platform_fee'),
                    'priceImpactPct': str(quote['price_impact_pct']),
                    'routePlan': quote['route_plan'],
                    'contextSlot': None,
                    'timeTaken': None
                },
                'userPublicKey': str(wallet_keypair.public_key),
                'wrapAndUnwrapSol': True,
                'prioritizationFeeLamports': 1000  # 0.000001 SOL priority fee
            }
            
            # Get swap transaction from Jupiter
            async with aiohttp.ClientSession() as session:
                async with session.post(self.jupiter_swap_url, json=swap_request) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'swapTransaction' in data:
                            # Deserialize transaction
                            tx_data = base64.b64decode(data['swapTransaction'])
                            transaction = Transaction.deserialize(tx_data)
                            
                            # Sign transaction
                            transaction.sign(wallet_keypair)
                            
                            # Send transaction
                            signature = await self.rpc_client.send_transaction(
                                transaction,
                                opts={"skip_confirmation": False, "preflight_commitment": Commitment("confirmed")}
                            )
                            
                            # Wait for confirmation
                            await self.rpc_client.confirm_transaction(
                                signature.value,
                                commitment=Commitment("confirmed")
                            )
                            
                            return {
                                'success': True,
                                'transaction_signature': signature.value,
                                'input_amount': quote['input_amount'],
                                'output_amount': quote['output_amount'],
                                'gas_fee': 0.001  # Estimate
                            }
                        else:
                            error_msg = data.get('error', 'Unknown swap error')
                            return {
                                'success': False,
                                'error_message': error_msg
                            }
                    else:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'error_message': f"HTTP {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            return {
                'success': False,
                'error_message': str(e)
            }

    async def _rate_limit(self):
        """Rate limiting for API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()

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
                'avg_execution_time_ms': self.trading_metrics['avg_execution_time_ms'],
                'performance': {
                    'total_trades': self.trading_metrics['total_orders'],
                    'win_rate': success_rate,
                    'total_profit_sol': self.trading_metrics['total_profit'],
                    'vault_deposits_sol': 0.0  # Will be updated by vault system
                },
                'stealth_status': {
                    'fake_trades_generated': 0,  # Will be updated by stealth system
                    'stealth_mode_active': True
                }
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
            
            # Close RPC connection
            await self.rpc_client.close()
            
            self.logger.info("✅ Trading engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def _trigger_emergency_shutdown(self):
        """Trigger emergency shutdown"""
        try:
            self.logger.critical("🚨 EMERGENCY SHUTDOWN TRIGGERED")
            
            # Stop trading immediately
            self.trading_active = False
            
            # Cancel all orders
            for order_id in list(self.active_orders.keys()):
                await self.cancel_order(order_id)
            
            # Trigger kill switch
            if self.kill_switch:
                await self.kill_switch.trigger("Emergency shutdown triggered")
            
            # Close RPC connection
            await self.rpc_client.close()
            
            self.logger.critical("✅ Emergency shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown: {e}")

    @property
    def trading_enabled(self):
        """Check if trading is enabled"""
        return self.trading_active and self.initialized and not (self.kill_switch and self.kill_switch.is_triggered)

    @trading_enabled.setter
    def trading_enabled(self, value: bool):
        """Enable/disable trading"""
        if value:
            if not self.initialized:
                self.logger.warning("Cannot enable trading: engine not initialized")
                return
            if self.kill_switch and self.kill_switch.is_triggered:
                self.logger.warning("Cannot enable trading: kill switch triggered")
                return
            self.trading_active = True
            self.logger.info("✅ Trading enabled")
        else:
            self.trading_active = False
            self.logger.info("❌ Trading disabled")

    async def _process_profit_to_vault(self, profit_sol: float, order_id: str):
        """Process profit to vault system with robust error handling and fallback storage"""
        try:
            from decimal import Decimal
            import json
            import time
            from pathlib import Path
            
            # Use Decimal for precise currency calculations
            profit_decimal = Decimal(str(profit_sol))
            
            # Ensure we have vault system reference
            if not self.vault_system:
                self.logger.critical("CRITICAL: Vault system not available for profit processing")
                await self._store_profit_fallback(profit_sol, order_id, "vault_system_unavailable")
                return
            
            # Process profit with enhanced retry mechanism
            max_retries = 5
            base_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    # Check vault system health before attempting deposit
                    if hasattr(self.vault_system, 'health_check'):
                        health_ok = await self.vault_system.health_check()
                        if not health_ok:
                            self.logger.warning(f"Vault system health check failed, attempt {attempt + 1}")
                            if attempt == max_retries - 1:
                                await self._store_profit_fallback(profit_sol, order_id, "vault_health_check_failed")
                                return
                            continue
                    
                    # Attempt to deposit profit
                    success = await asyncio.wait_for(
                        self.vault_system.deposit_profits(float(profit_decimal)), 
                        timeout=30.0  # 30 second timeout
                    )
                    
                    if success:
                        self.logger.info(f"💰 PROFIT SECURED: {profit_decimal} SOL deposited to vault for order {order_id}")
                        # Log successful profit processing for audit trail
                        await self._log_profit_success(profit_sol, order_id)
                        return
                    else:
                        self.logger.warning(f"Vault deposit failed for order {order_id}, attempt {attempt + 1}")
                        
                except asyncio.TimeoutError:
                    self.logger.error(f"Vault deposit timeout for order {order_id}, attempt {attempt + 1}")
                except Exception as e:
                    self.logger.error(f"Vault deposit error for order {order_id}, attempt {attempt + 1}: {e}")
                
                # Wait before retry with exponential backoff
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(min(delay, 10.0))  # Cap at 10 seconds
            
            # If all retries failed, store in fallback system
            self.logger.critical(f"CRITICAL FAILURE: Could not secure profit {profit_decimal} SOL for order {order_id}")
            await self._store_profit_fallback(profit_sol, order_id, "all_retries_failed")
            
        except Exception as e:
            self.logger.critical(f"CRITICAL ERROR in profit processing for order {order_id}: {e}")
            await self._store_profit_fallback(profit_sol, order_id, f"processing_error: {str(e)}")
    
    async def _store_profit_fallback(self, profit_sol: float, order_id: str, reason: str):
        """Store profit in fallback system when vault fails"""
        try:
            import json
            import time
            from pathlib import Path
            
            # Create fallback storage directory
            fallback_dir = Path("logs/profit_fallback")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare fallback record
            fallback_record = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'order_id': order_id,
                'profit_sol': float(profit_sol),
                'reason': reason,
                'status': 'pending_recovery',
                'recovery_attempts': 0
            }
            
            # Store in timestamped file
            filename = f"profit_fallback_{int(time.time())}_{order_id[:8]}.json"
            fallback_file = fallback_dir / filename
            
            with open(fallback_file, 'w') as f:
                json.dump(fallback_record, f, indent=2)
            
            self.logger.critical(f"🆘 PROFIT FALLBACK: {profit_sol} SOL stored in {fallback_file}")
            
            # Also attempt to log to database if available
            try:
                if hasattr(self, 'db_manager') and self.db_manager:
                    from worker_ant_v1.core.schemas import SystemEvent
                    event = SystemEvent(
                        timestamp=datetime.now(),
                        event_id=str(uuid.uuid4()),
                        event_type="profit_fallback",
                        component="trading_engine",
                        severity="CRITICAL",
                        message=f"Profit fallback storage: {profit_sol} SOL for order {order_id}",
                        event_data=fallback_record
                    )
                    await self.db_manager.insert_system_event(event)
            except Exception as db_error:
                self.logger.error(f"Failed to log fallback to database: {db_error}")
            
            # Start background recovery task
            asyncio.create_task(self._schedule_profit_recovery(fallback_file, fallback_record))
            
        except Exception as e:
            self.logger.critical(f"CATASTROPHIC: Failed to store profit fallback for order {order_id}: {e}")
            # Last resort: write to emergency log
            emergency_log = Path("logs/EMERGENCY_PROFIT_LOG.txt")
            emergency_log.parent.mkdir(parents=True, exist_ok=True)
            with open(emergency_log, 'a') as f:
                f.write(f"{datetime.now().isoformat()} | ORDER: {order_id} | PROFIT: {profit_sol} SOL | REASON: {reason} | ERROR: {str(e)}\n")
    
    async def _log_profit_success(self, profit_sol: float, order_id: str):
        """Log successful profit processing for audit trail"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                from worker_ant_v1.core.schemas import SystemEvent
                event = SystemEvent(
                    timestamp=datetime.now(),
                    event_id=str(uuid.uuid4()),
                    event_type="profit_secured",
                    component="trading_engine",
                    severity="INFO",
                    message=f"Profit successfully secured: {profit_sol} SOL for order {order_id}",
                    event_data={
                        'order_id': order_id,
                        'profit_sol': profit_sol,
                        'vault_deposit': True
                    }
                )
                await self.db_manager.insert_system_event(event)
        except Exception as e:
            self.logger.error(f"Failed to log profit success: {e}")
    
    async def _schedule_profit_recovery(self, fallback_file: Path, record: dict):
        """Schedule automatic profit recovery attempts"""
        try:
            # Wait before attempting recovery
            await asyncio.sleep(300)  # Wait 5 minutes before first recovery attempt
            
            max_recovery_attempts = 10
            for attempt in range(max_recovery_attempts):
                try:
                    # Check if vault system is available again
                    if self.vault_system and hasattr(self.vault_system, 'health_check'):
                        health_ok = await self.vault_system.health_check()
                        if health_ok:
                            # Attempt to process the fallback profit
                            success = await self.vault_system.deposit_profits(record['profit_sol'])
                            if success:
                                # Update fallback record
                                record['status'] = 'recovered'
                                record['recovered_at'] = datetime.now().isoformat()
                                record['recovery_attempts'] = attempt + 1
                                
                                with open(fallback_file, 'w') as f:
                                    json.dump(record, f, indent=2)
                                
                                self.logger.info(f"✅ PROFIT RECOVERED: {record['profit_sol']} SOL for order {record['order_id']}")
                                return
                
                except Exception as e:
                    self.logger.error(f"Profit recovery attempt {attempt + 1} failed: {e}")
                
                # Update attempt count
                record['recovery_attempts'] = attempt + 1
                with open(fallback_file, 'w') as f:
                    json.dump(record, f, indent=2)
                
                # Wait before next attempt (exponential backoff)
                await asyncio.sleep(min(600 * (2 ** attempt), 3600))  # Cap at 1 hour
            
            # Mark as failed recovery
            record['status'] = 'recovery_failed'
            with open(fallback_file, 'w') as f:
                json.dump(record, f, indent=2)
            
            self.logger.critical(f"💀 PROFIT RECOVERY FAILED: {record['profit_sol']} SOL for order {record['order_id']} - manual intervention required")
            
        except Exception as e:
            self.logger.critical(f"Profit recovery scheduler failed: {e}")

# Global instance
_trading_engine = None

async def get_trading_engine() -> UnifiedTradingEngine:
    """Get global trading engine instance"""
    global _trading_engine
    if _trading_engine is None:
        _trading_engine = UnifiedTradingEngine()
        await _trading_engine.initialize()
    return _trading_engine 