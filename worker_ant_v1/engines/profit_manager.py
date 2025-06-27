"""
PROFIT DISCIPLINE ENGINE - COLD SURVIVOR MODE
=============================================

Advanced profit-taking and loss-cutting system designed for survival.
Implements dynamic thresholds, fast exits, and emotionless trading discipline
to maximize capital preservation and compound growth.
"""

import asyncio
import time
import json
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np

# Internal imports
from worker_ant_v1.core.simple_config import get_trading_config, get_security_config
from worker_ant_v1.utils.simple_logger import setup_logger as trading_logger
from worker_ant_v1.trading.order_seller import ProductionSeller, Position

# Temporary placeholder for missing EvolutionGenetics
class EvolutionGenetics:
    def __init__(self):
        pass

class ExitReason(Enum):
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIME_LIMIT = "time_limit"
    TREND_REVERSAL = "trend_reversal"
    VOLUME_DROP = "volume_drop"
    EMERGENCY_EXIT = "emergency_exit"
    RUG_DETECTION = "rug_detection"
    MANUAL_EXIT = "manual_exit"

class ExitUrgency(Enum):
    LOW = "low"           # Normal exit
    MEDIUM = "medium"     # Quick exit
    HIGH = "high"         # Immediate exit
    CRITICAL = "critical" # Emergency exit

@dataclass
class ProfitTarget:
    """Dynamic profit target configuration"""
    
    target_percent: float
    confidence: float = 1.0
    min_hold_seconds: int = 30
    max_hold_seconds: int = 1800  # 30 minutes
    
    # Dynamic adjustments
    volume_factor: float = 1.0
    volatility_factor: float = 1.0
    market_factor: float = 1.0
    
    def get_adjusted_target(self) -> float:
        """Get adjusted profit target based on market conditions"""
        base_target = self.target_percent
        
        # Apply factors
        adjusted_target = base_target * self.volume_factor * self.volatility_factor * self.market_factor
        
        # Clamp to reasonable bounds
        return max(1.0, min(50.0, adjusted_target))  # 1% to 50%

@dataclass
class StopLoss:
    """Dynamic stop loss configuration"""
    
    loss_percent: float
    trailing_enabled: bool = True
    break_even_threshold: float = 2.0  # Move to break-even after 2% profit
    
    # Dynamic adjustments
    volatility_multiplier: float = 1.0
    liquidity_factor: float = 1.0
    time_decay_factor: float = 1.0
    
    def get_adjusted_stop_loss(self, current_profit: float) -> float:
        """Get adjusted stop loss based on current conditions"""
        
        # If in profit, use trailing stop
        if self.trailing_enabled and current_profit > self.break_even_threshold:
            # Protect profits with trailing stop
            return max(-1.0, current_profit - 3.0)  # Trail by 3%
        
        # Otherwise use dynamic stop loss
        base_loss = self.loss_percent
        adjusted_loss = base_loss * self.volatility_multiplier * self.liquidity_factor * self.time_decay_factor
        
        return max(-10.0, min(-0.5, -abs(adjusted_loss)))  # -10% to -0.5%

@dataclass
class PositionManager:
    """Individual position management"""
    
    position: Position
    entry_price: float
    entry_time: datetime
    
    # Profit targets (multiple levels)
    profit_targets: List[ProfitTarget] = field(default_factory=list)
    current_stop_loss: StopLoss = field(default_factory=lambda: StopLoss(loss_percent=-3.0))
    
    # Performance tracking
    peak_profit: float = 0.0
    max_drawdown: float = 0.0
    last_price_check: datetime = field(default_factory=datetime.now)
    
    # Risk management
    position_size_reduction: float = 0.0  # How much we've sold
    emergency_exit_triggered: bool = False
    
    def update_performance(self, current_price: float):
        """Update position performance metrics"""
        current_profit = ((current_price - self.entry_price) / self.entry_price) * 100
        
        # Update peak profit
        if current_profit > self.peak_profit:
            self.peak_profit = current_profit
        
        # Update max drawdown from peak
        drawdown_from_peak = self.peak_profit - current_profit
        if drawdown_from_peak > self.max_drawdown:
            self.max_drawdown = drawdown_from_peak
        
        self.last_price_check = datetime.now()
    
    def get_current_profit_percent(self, current_price: float) -> float:
        """Get current profit percentage"""
        return ((current_price - self.entry_price) / self.entry_price) * 100
    
    def should_exit(self, current_price: float, market_conditions: Dict[str, Any]) -> Tuple[bool, ExitReason, ExitUrgency]:
        """Determine if position should be exited"""
        
        current_profit = self.get_current_profit_percent(current_price)
        self.update_performance(current_price)
        
        # Check emergency conditions first
        if self._check_emergency_conditions(market_conditions):
            return True, ExitReason.EMERGENCY_EXIT, ExitUrgency.CRITICAL
        
        # Check for rug pull indicators
        if self._check_rug_indicators(market_conditions):
            return True, ExitReason.RUG_DETECTION, ExitUrgency.CRITICAL
        
        # Check profit targets
        for target in self.profit_targets:
            if current_profit >= target.get_adjusted_target():
                return True, ExitReason.PROFIT_TARGET, ExitUrgency.MEDIUM
        
        # Check stop loss
        stop_loss_level = self.current_stop_loss.get_adjusted_stop_loss(current_profit)
        if current_profit <= stop_loss_level:
            return True, ExitReason.STOP_LOSS, ExitUrgency.HIGH
        
        # Check time limits
        position_age = (datetime.now() - self.entry_time).total_seconds()
        max_hold_time = max(target.max_hold_seconds for target in self.profit_targets) if self.profit_targets else 1800
        
        if position_age > max_hold_time:
            return True, ExitReason.TIME_LIMIT, ExitUrgency.LOW
        
        # Check trend reversal
        if self._check_trend_reversal(current_price, market_conditions):
            return True, ExitReason.TREND_REVERSAL, ExitUrgency.MEDIUM
        
        # Check volume drop
        if self._check_volume_drop(market_conditions):
            return True, ExitReason.VOLUME_DROP, ExitUrgency.MEDIUM
        
        return False, None, None
    
    def _check_emergency_conditions(self, market_conditions: Dict[str, Any]) -> bool:
        """Check for emergency exit conditions"""
        
        # Sudden liquidity drain
        liquidity_drop = market_conditions.get('liquidity_drop_percent', 0)
        if liquidity_drop > 50:  # 50% liquidity drop
            return True
        
        # Massive sell pressure
        sell_pressure = market_conditions.get('sell_pressure', 0)
        if sell_pressure > 0.8:  # 80% sell pressure
            return True
        
        # Network issues
        network_issues = market_conditions.get('network_congestion', 0)
        if network_issues > 0.9:  # 90% network congestion
            return True
        
        return False
    
    def _check_rug_indicators(self, market_conditions: Dict[str, Any]) -> bool:
        """Check for rug pull indicators"""
        
        # Liquidity removal
        liquidity_removed = market_conditions.get('liquidity_removed', False)
        if liquidity_removed:
            return True
        
        # Creator wallet dumping
        creator_dumping = market_conditions.get('creator_dumping', False)
        if creator_dumping:
            return True
        
        # Whale wallet concentrating
        whale_concentration = market_conditions.get('whale_concentration', 0)
        if whale_concentration > 0.7:  # 70% concentration
            return True
        
        return False
    
    def _check_trend_reversal(self, current_price: float, market_conditions: Dict[str, Any]) -> bool:
        """Check for trend reversal signals"""
        
        # Price momentum reversal
        price_momentum = market_conditions.get('price_momentum', 0)
        if price_momentum < -0.5:  # Strong negative momentum
            return True
        
        # Volume-price divergence
        volume_divergence = market_conditions.get('volume_price_divergence', 0)
        if volume_divergence > 0.6:  # Strong divergence
            return True
        
        # Technical indicator reversal
        rsi = market_conditions.get('rsi', 50)
        if rsi > 80:  # Overbought
            return True
        
        return False
    
    def _check_volume_drop(self, market_conditions: Dict[str, Any]) -> bool:
        """Check for significant volume drop"""
        
        volume_drop = market_conditions.get('volume_drop_percent', 0)
        return volume_drop > 70  # 70% volume drop

class ProfitDisciplineEngine:
    """Main profit discipline engine"""
    
    def __init__(self, genetics: Optional[EvolutionGenetics] = None):
        self.logger = logging.getLogger("ProfitDisciplineEngine")
        self.genetics = genetics or EvolutionGenetics()
        
        # Active positions
        self.active_positions: Dict[str, PositionManager] = {}
        
        # Discipline settings
        self.max_positions = 5
        self.emergency_exit_enabled = True
        self.profit_taking_speed = "fast"  # "slow", "medium", "fast", "instant"
        
        # Performance tracking
        self.discipline_metrics = {
            'total_exits': 0,
            'profit_exits': 0,
            'loss_exits': 0,
            'emergency_exits': 0,
            'avg_hold_time_seconds': 0.0,
            'avg_profit_percent': 0.0,
            'max_consecutive_losses': 0,
            'current_streak': 0
        }
        
        # Dynamic thresholds
        self.base_profit_targets = [3.0, 6.0, 12.0]  # 3%, 6%, 12%
        self.base_stop_loss = -2.5  # -2.5%
        
    async def add_position(self, position: Position, entry_price: float) -> str:
        """Add new position for management"""
        
        position_id = f"pos_{position.token_address}_{int(time.time())}"
        
        # Create profit targets based on genetics
        profit_targets = self._create_profit_targets()
        
        # Create stop loss based on genetics
        stop_loss = self._create_stop_loss()
        
        # Create position manager
        position_manager = PositionManager(
            position=position,
            entry_price=entry_price,
            entry_time=datetime.now(),
            profit_targets=profit_targets,
            current_stop_loss=stop_loss
        )
        
        self.active_positions[position_id] = position_manager
        
        self.logger.info(f"📊 Added position {position_id}: {position.token_symbol} @ ${entry_price}")
        
        return position_id
    
    def _create_profit_targets(self) -> List[ProfitTarget]:
        """Create profit targets based on genetics"""
        
        targets = []
        
        # Adjust base targets based on genetics
        aggression_factor = self.genetics.aggression
        patience_factor = self.genetics.patience
        
        for i, base_target in enumerate(self.base_profit_targets):
            # More aggressive = higher targets, less patient = faster exits
            adjusted_target = base_target * (1 + aggression_factor * 0.5)
            
            # Calculate hold times based on patience
            min_hold = int(30 * (1 + patience_factor))  # 30-60 seconds min
            max_hold = int(300 * (1 + patience_factor * 2))  # 5-15 minutes max
            
            target = ProfitTarget(
                target_percent=adjusted_target,
                confidence=1.0 - (i * 0.1),  # Decreasing confidence for higher targets
                min_hold_seconds=min_hold,
                max_hold_seconds=max_hold
            )
            
            targets.append(target)
        
        return targets
    
    def _create_stop_loss(self) -> StopLoss:
        """Create stop loss based on genetics"""
        
        # Adjust stop loss based on risk tolerance
        risk_factor = self.genetics.risk_tolerance
        base_loss = self.base_stop_loss
        
        # Higher risk tolerance = wider stop loss
        adjusted_loss = base_loss * (1 + risk_factor * 0.5)
        
        return StopLoss(
            loss_percent=adjusted_loss,
            trailing_enabled=True,
            break_even_threshold=2.0 + (self.genetics.aggression * 2.0)  # 2-4%
        )
    
    async def monitor_positions(self, market_data: Dict[str, Any]):
        """Monitor all positions for exit conditions"""
        
        exit_actions = []
        
        for position_id, position_manager in self.active_positions.items():
            token_address = position_manager.position.token_address
            current_price = market_data.get(token_address, {}).get('price', position_manager.entry_price)
            market_conditions = market_data.get(token_address, {})
            
            # Check exit conditions
            should_exit, exit_reason, exit_urgency = position_manager.should_exit(current_price, market_conditions)
            
            if should_exit:
                exit_actions.append({
                    'position_id': position_id,
                    'position_manager': position_manager,
                    'exit_reason': exit_reason,
                    'exit_urgency': exit_urgency,
                    'current_price': current_price
                })
        
        # Execute exits based on urgency
        exit_actions.sort(key=lambda x: self._get_urgency_priority(x['exit_urgency']))
        
        for exit_action in exit_actions:
            await self._execute_exit(exit_action)
    
    def _get_urgency_priority(self, urgency: ExitUrgency) -> int:
        """Get priority score for exit urgency"""
        priorities = {
            ExitUrgency.CRITICAL: 4,
            ExitUrgency.HIGH: 3,
            ExitUrgency.MEDIUM: 2,
            ExitUrgency.LOW: 1
        }
        return priorities.get(urgency, 0)
    
    async def _execute_exit(self, exit_action: Dict[str, Any]):
        """Execute position exit"""
        
        position_id = exit_action['position_id']
        position_manager = exit_action['position_manager']
        exit_reason = exit_action['exit_reason']
        exit_urgency = exit_action['exit_urgency']
        current_price = exit_action['current_price']
        
        # Calculate final profit
        final_profit = position_manager.get_current_profit_percent(current_price)
        
        # Determine exit strategy based on urgency
        exit_strategy = self._determine_exit_strategy(exit_urgency, position_manager)
        
        self.logger.info(f"🎯 Exiting {position_id}: {exit_reason.value} ({final_profit:.2f}%) - {exit_urgency.value}")
        
        # Execute the exit
        success = await self._perform_exit(position_manager, exit_strategy)
        
        if success:
            # Update metrics
            self._update_discipline_metrics(position_manager, final_profit, exit_reason)
            
            # Remove from active positions
            del self.active_positions[position_id]
            
            self.logger.info(f"✅ Position {position_id} closed successfully")
        else:
            self.logger.error(f"❌ Failed to exit position {position_id}")
    
    def _determine_exit_strategy(self, urgency: ExitUrgency, position_manager: PositionManager) -> Dict[str, Any]:
        """Determine optimal exit strategy based on urgency"""
        
        strategy = {
            'sell_percentage': 100,  # Default: sell all
            'max_slippage': 2.0,     # Default: 2% slippage
            'timeout_seconds': 30,   # Default: 30 second timeout
            'use_market_order': False
        }
        
        if urgency == ExitUrgency.CRITICAL:
            # Emergency exit - accept high slippage
            strategy.update({
                'max_slippage': 5.0,
                'timeout_seconds': 10,
                'use_market_order': True
            })
        elif urgency == ExitUrgency.HIGH:
            # Quick exit - moderate slippage
            strategy.update({
                'max_slippage': 3.0,
                'timeout_seconds': 15,
                'use_market_order': True
            })
        elif urgency == ExitUrgency.MEDIUM:
            # Standard exit - normal slippage
            strategy.update({
                'max_slippage': 2.0,
                'timeout_seconds': 30,
                'use_market_order': False
            })
        else:  # LOW urgency
            # Slow exit - minimize slippage
            strategy.update({
                'max_slippage': 1.0,
                'timeout_seconds': 60,
                'use_market_order': False
            })
        
        return strategy
    
    async def _perform_exit(self, position_manager: PositionManager, exit_strategy: Dict[str, Any]) -> bool:
        """Perform the actual exit trade"""
        
        try:
            # This would integrate with the actual seller
            # For now, simulate the exit
            
            # Simulate execution time based on urgency
            execution_time = exit_strategy['timeout_seconds'] * 0.1
            await asyncio.sleep(execution_time)
            
            # Simulate success rate based on market conditions
            success_probability = 0.95 - (exit_strategy['max_slippage'] * 0.05)
            success = np.random.random() < success_probability
            
            return success
            
        except Exception as e:
            self.logger.error(f"Exit execution error: {e}")
            return False
    
    def _update_discipline_metrics(self, position_manager: PositionManager, final_profit: float, exit_reason: ExitReason):
        """Update discipline performance metrics"""
        
        self.discipline_metrics['total_exits'] += 1
        
        # Update profit/loss counts
        if final_profit > 0:
            self.discipline_metrics['profit_exits'] += 1
            self.discipline_metrics['current_streak'] = max(0, self.discipline_metrics['current_streak'] + 1)
        else:
            self.discipline_metrics['loss_exits'] += 1
            consecutive_losses = -min(0, self.discipline_metrics['current_streak'] - 1)
            self.discipline_metrics['max_consecutive_losses'] = max(
                self.discipline_metrics['max_consecutive_losses'], 
                consecutive_losses
            )
            self.discipline_metrics['current_streak'] = min(0, self.discipline_metrics['current_streak'] - 1)
        
        # Update emergency exits
        if exit_reason == ExitReason.EMERGENCY_EXIT:
            self.discipline_metrics['emergency_exits'] += 1
        
        # Update averages
        total_exits = self.discipline_metrics['total_exits']
        
        # Average hold time
        hold_time = (datetime.now() - position_manager.entry_time).total_seconds()
        old_avg_hold = self.discipline_metrics['avg_hold_time_seconds']
        self.discipline_metrics['avg_hold_time_seconds'] = (
            (old_avg_hold * (total_exits - 1) + hold_time) / total_exits
        )
        
        # Average profit
        old_avg_profit = self.discipline_metrics['avg_profit_percent']
        self.discipline_metrics['avg_profit_percent'] = (
            (old_avg_profit * (total_exits - 1) + final_profit) / total_exits
        )
    
    async def emergency_exit_all(self, reason: str = "Emergency stop"):
        """Emergency exit all positions"""
        
        self.logger.critical(f"🚨 EMERGENCY EXIT ALL POSITIONS: {reason}")
        
        exit_tasks = []
        for position_id, position_manager in self.active_positions.items():
            exit_action = {
                'position_id': position_id,
                'position_manager': position_manager,
                'exit_reason': ExitReason.EMERGENCY_EXIT,
                'exit_urgency': ExitUrgency.CRITICAL,
                'current_price': position_manager.entry_price  # Use entry price as fallback
            }
            
            # Execute exits in parallel for speed
            task = asyncio.create_task(self._execute_exit(exit_action))
            exit_tasks.append(task)
        
        # Wait for all exits to complete
        await asyncio.gather(*exit_tasks, return_exceptions=True)
        
        self.logger.info("✅ Emergency exit of all positions completed")
    
    def get_discipline_summary(self) -> Dict[str, Any]:
        """Get discipline performance summary"""
        
        total_exits = self.discipline_metrics['total_exits']
        if total_exits == 0:
            return {'message': 'No exits yet'}
        
        win_rate = (self.discipline_metrics['profit_exits'] / total_exits) * 100
        avg_hold_minutes = self.discipline_metrics['avg_hold_time_seconds'] / 60
        
        return {
            'total_positions_closed': total_exits,
            'win_rate_percent': win_rate,
            'avg_profit_percent': self.discipline_metrics['avg_profit_percent'],
            'avg_hold_time_minutes': avg_hold_minutes,
            'emergency_exits': self.discipline_metrics['emergency_exits'],
            'max_consecutive_losses': self.discipline_metrics['max_consecutive_losses'],
            'current_streak': self.discipline_metrics['current_streak'],
            'active_positions': len(self.active_positions)
        }
    
    def adjust_discipline_parameters(self, performance_data: Dict[str, Any]):
        """Dynamically adjust discipline parameters based on performance"""
        
        win_rate = performance_data.get('win_rate', 0.5)
        avg_profit = performance_data.get('avg_profit', 0.0)
        
        # Adjust profit targets based on performance
        if win_rate > 0.7 and avg_profit > 5.0:
            # Performing well, can be more aggressive
            self.base_profit_targets = [4.0, 8.0, 15.0]
        elif win_rate < 0.4 or avg_profit < 0.0:
            # Underperforming, be more conservative
            self.base_profit_targets = [2.0, 4.0, 8.0]
            self.base_stop_loss = -2.0  # Tighter stop loss
        
        self.logger.info(f"🔧 Adjusted discipline parameters based on performance")

# Export main class
__all__ = ['ProfitDisciplineEngine', 'PositionManager', 'ProfitTarget', 'StopLoss'] 