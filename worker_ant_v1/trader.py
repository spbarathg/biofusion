#!/usr/bin/env python3
"""
High-Performance Trader Module
=============================

Unified trading interface that combines buyer and seller functionality
for the Enhanced Crypto Trading Bot swarm system.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .config import trading_config, deployment_config, TradingMode
from .buyer import enhanced_buyer, BuySignal
from .seller import smart_seller, SellSignal
from .logger import trading_logger, TradeResult
from .stealth_mechanics import stealth_manager
from .swarm_kill_switch import kill_switch


@dataclass
class TradeOpportunity:
    """Trading opportunity data structure"""
    token_symbol: str
    token_address: str
    price_usd: float
    liquidity_usd: float
    volume_24h: float
    price_change_1h: float
    price_change_24h: float
    confidence_score: float = 0.0
    strategy: str = "default"
    entry_signal_strength: float = 0.0
    
    def __post_init__(self):
        # Calculate confidence score based on multiple factors
        self.confidence_score = self._calculate_confidence()
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence score for this opportunity"""
        score = 0.0
        
        # Liquidity factor (higher is better)
        if self.liquidity_usd > 50000:
            score += 30
        elif self.liquidity_usd > 20000:
            score += 20
        elif self.liquidity_usd > 10000:
            score += 10
        
        # Volume factor
        if self.volume_24h > 100000:
            score += 25
        elif self.volume_24h > 50000:
            score += 15
        elif self.volume_24h > 20000:
            score += 10
        
        # Price momentum (positive change is good)
        if self.price_change_1h > 5:
            score += 20
        elif self.price_change_1h > 2:
            score += 15
        elif self.price_change_1h > 0:
            score += 10
        
        # Entry signal strength
        score += self.entry_signal_strength * 25
        
        return min(100, max(0, score))


class HighPerformanceTrader:
    """High-performance trading engine for the swarm system"""
    
    def __init__(self):
        self.setup_complete = False
        self.active_positions: Dict[str, Dict] = {}
        self.trade_history: List[TradeResult] = []
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit_sol': 0.0,
            'total_fees_sol': 0.0,
            'win_rate': 0.0,
            'average_profit_percent': 0.0,
            'last_trade_time': None
        }
        self.last_trade_time = 0
        self.logger = trading_logger
        
    async def setup(self):
        """Initialize the high-performance trader"""
        try:
            self.logger.info("🚀 Initializing High-Performance Trader...")
            
            # Initialize buyer and seller components
            await enhanced_buyer.initialize()
            await smart_seller.initialize()
            
            # Initialize stealth mechanics
            await stealth_manager.initialize()
            
            # Verify kill switch is active
            if not kill_switch.is_initialized:
                await kill_switch.initialize()
            
            self.setup_complete = True
            self.logger.info("✅ High-Performance Trader ready")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize trader: {e}")
            raise
    
    async def execute_profitable_trade(self, opportunity: TradeOpportunity) -> TradeResult:
        """Execute a profitable trade on the given opportunity"""
        
        if not self.setup_complete:
            return TradeResult(
                success=False,
                error_message="Trader not initialized",
                signature="",
                amount_sol=0,
                token_amount=0,
                price_per_token=0
            )
        
        # Check if we can trade (rate limiting, kill switch, etc.)
        if not await self._can_trade():
            return TradeResult(
                success=False,
                error_message="Trading not allowed at this time",
                signature="",
                amount_sol=0,
                token_amount=0,
                price_per_token=0
            )
        
        try:
            # Apply stealth mechanics
            stealth_params = await stealth_manager.get_stealth_parameters()
            
            # Calculate position size
            position_size = self._calculate_position_size(opportunity)
            
            # Create buy signal
            buy_signal = BuySignal(
                token_address=opportunity.token_address,
                confidence=opportunity.confidence_score / 100,
                urgency=min(1.0, opportunity.entry_signal_strength),
                max_slippage=stealth_params.get('slippage', trading_config.max_slippage_percent),
                amount_sol=position_size
            )
            
            # Execute buy order
            self.logger.info(f"🎯 Executing trade: {opportunity.token_symbol} (${opportunity.price_usd})")
            buy_result = await enhanced_buyer.execute_buy(buy_signal)
            
            if buy_result.success:
                # Track the position
                position_id = buy_result.signature
                self.active_positions[position_id] = {
                    'token_address': opportunity.token_address,
                    'token_symbol': opportunity.token_symbol,
                    'entry_price': opportunity.price_usd,
                    'amount_sol': buy_result.amount_sol,
                    'token_amount': buy_result.token_amount,
                    'entry_time': datetime.now(),
                    'strategy': opportunity.strategy,
                    'target_profit': trading_config.base_profit_target_percent,
                    'stop_loss': trading_config.base_stop_loss_percent
                }
                
                # Start monitoring for exit
                asyncio.create_task(self._monitor_position(position_id))
                
                # Update metrics
                self._update_metrics(buy_result, True)
                self.last_trade_time = time.time()
                
                self.logger.info(f"✅ Trade executed: {buy_result.signature}")
                return buy_result
            else:
                self.logger.warning(f"❌ Trade failed: {buy_result.error_message}")
                self._update_metrics(buy_result, False)
                return buy_result
                
        except Exception as e:
            self.logger.error(f"❌ Trade execution error: {e}")
            return TradeResult(
                success=False,
                error_message=str(e),
                signature="",
                amount_sol=0,
                token_amount=0,
                price_per_token=0
            )
    
    async def _monitor_position(self, position_id: str):
        """Monitor a position for exit conditions"""
        
        if position_id not in self.active_positions:
            return
        
        position = self.active_positions[position_id]
        
        try:
            # Monitor for up to the configured timeout
            start_time = position['entry_time']
            timeout = timedelta(seconds=trading_config.timeout_exit_seconds)
            
            while position_id in self.active_positions:
                # Check if timeout reached
                if datetime.now() - start_time > timeout:
                    await self._exit_position(position_id, "timeout")
                    break
                
                # Check profit/loss conditions
                # (This would normally check current price from DEX)
                # For simulation, we'll use a simplified approach
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except Exception as e:
            self.logger.error(f"Error monitoring position {position_id}: {e}")
            # Force exit on error
            await self._exit_position(position_id, "error")
    
    async def _exit_position(self, position_id: str, reason: str):
        """Exit a position"""
        
        if position_id not in self.active_positions:
            return
        
        position = self.active_positions[position_id]
        
        try:
            # Create sell signal
            sell_signal = SellSignal(
                token_address=position['token_address'],
                amount_tokens=position['token_amount'],
                urgency=0.8 if reason == "timeout" else 1.0,
                reason=reason
            )
            
            # Execute sell
            sell_result = await smart_seller.execute_sell(sell_signal)
            
            if sell_result.success:
                # Calculate profit/loss
                profit_sol = sell_result.amount_sol - position['amount_sol']
                profit_percent = (profit_sol / position['amount_sol']) * 100
                
                self.logger.info(f"📈 Position closed: {position['token_symbol']} "
                               f"P&L: {profit_sol:.4f} SOL ({profit_percent:.2f}%)")
                
                # Update performance metrics
                self.performance_metrics['total_profit_sol'] += profit_sol
                self._update_metrics(sell_result, True)
            else:
                self.logger.warning(f"❌ Failed to exit position: {sell_result.error_message}")
            
            # Remove from active positions
            del self.active_positions[position_id]
            
        except Exception as e:
            self.logger.error(f"Error exiting position {position_id}: {e}")
    
    async def _can_trade(self) -> bool:
        """Check if trading is allowed"""
        
        # Check kill switch
        if kill_switch.is_triggered:
            return False
        
        # Check rate limiting
        current_time = time.time()
        if current_time - self.last_trade_time < trading_config.min_time_between_trades_seconds:
            return False
        
        # Check concurrent positions limit
        if len(self.active_positions) >= trading_config.max_concurrent_positions:
            return False
        
        # Check trading mode
        if deployment_config.trading_mode == TradingMode.SIMULATION:
            return True  # Always allow in simulation
        
        return True
    
    def _calculate_position_size(self, opportunity: TradeOpportunity) -> float:
        """Calculate position size based on opportunity and risk management"""
        
        base_size = trading_config.trade_amount_sol
        
        # Adjust based on confidence
        confidence_multiplier = opportunity.confidence_score / 100
        
        # Adjust based on recent performance
        if self.performance_metrics['win_rate'] > 70:
            multiplier = 1.2
        elif self.performance_metrics['win_rate'] < 30:
            multiplier = 0.8
        else:
            multiplier = 1.0
        
        size = base_size * confidence_multiplier * multiplier
        
        # Apply limits
        size = max(trading_config.min_trade_amount_sol, size)
        size = min(trading_config.max_trade_amount_sol, size)
        
        return size
    
    def _update_metrics(self, trade_result: TradeResult, is_entry: bool):
        """Update performance metrics"""
        
        if is_entry:
            self.performance_metrics['total_trades'] += 1
            if trade_result.success:
                self.performance_metrics['successful_trades'] += 1
            else:
                self.performance_metrics['failed_trades'] += 1
        
        # Update win rate
        total = self.performance_metrics['total_trades']
        if total > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['successful_trades'] / total * 100
            )
        
        self.performance_metrics['last_trade_time'] = datetime.now()
        
        # Track fees
        if hasattr(trade_result, 'fee_sol'):
            self.performance_metrics['total_fees_sol'] += trade_result.fee_sol
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        return {
            'total_trades': self.performance_metrics['total_trades'],
            'successful_trades': self.performance_metrics['successful_trades'],
            'failed_trades': self.performance_metrics['failed_trades'],
            'win_rate': self.performance_metrics['win_rate'],
            'total_profit_sol': self.performance_metrics['total_profit_sol'],
            'total_fees_sol': self.performance_metrics['total_fees_sol'],
            'active_positions': len(self.active_positions),
            'last_trade_time': self.performance_metrics['last_trade_time'],
            'setup_complete': self.setup_complete
        }
    
    async def emergency_exit_all(self):
        """Emergency exit all positions"""
        self.logger.warning("🚨 Emergency exit triggered - closing all positions")
        
        exit_tasks = []
        for position_id in list(self.active_positions.keys()):
            exit_tasks.append(self._exit_position(position_id, "emergency"))
        
        if exit_tasks:
            await asyncio.gather(*exit_tasks, return_exceptions=True)
        
        self.logger.info("🚨 Emergency exit completed")
    
    async def shutdown(self):
        """Shutdown the trader gracefully"""
        self.logger.info("🔄 Shutting down High-Performance Trader...")
        
        # Exit all positions
        await self.emergency_exit_all()
        
        # Shutdown components
        if hasattr(enhanced_buyer, 'shutdown'):
            await enhanced_buyer.shutdown()
        if hasattr(smart_seller, 'shutdown'):
            await smart_seller.shutdown()
        
        self.setup_complete = False
        self.logger.info("✅ High-Performance Trader shutdown complete")


# Global instance
high_performance_trader = HighPerformanceTrader()


# Export for compatibility
TradeResult = TradeResult 