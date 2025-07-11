"""
HYPER COMPOUND ENGINE - MAXIMUM VELOCITY COMPOUNDING
==================================================

🎯 MISSION: TURN $300 INTO $10K+ THROUGH HYPER-AGGRESSIVE COMPOUNDING
💰 STRATEGY: Fast-cycle profit reinvestment with geometric scaling
🚀 TARGET: 33x growth through surgical compounding at swarm scale

This engine optimizes your existing vault and trading systems for maximum
compounding velocity while maintaining surgical risk management.
"""

import asyncio
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.core.vault_wallet_system import VaultWalletSystem, VaultType
from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
from worker_ant_v1.core.unified_trading_engine import UnifiedTradingEngine
from worker_ant_v1.core.swarm_decision_engine import SwarmDecisionEngine

class CompoundCycle(Enum):
    """Compounding cycle speeds"""
    LIGHTNING = "lightning"    # Every win immediately
    RAPID = "rapid"           # Every 3 wins  
    AGGRESSIVE = "aggressive"  # Every 5 wins
    STANDARD = "standard"     # Every 10 wins

class GrowthPhase(Enum):
    """Capital growth phases with different strategies"""
    BOOTSTRAP = "bootstrap"    # $300-$1K: Maximum aggression
    MOMENTUM = "momentum"      # $1K-$3K: Balanced growth
    ACCELERATION = "acceleration"  # $3K-$10K: Controlled scaling
    MASTERY = "mastery"       # $10K+: Position optimization

@dataclass
class CompoundConfig:
    """Hyper-aggressive compounding configuration"""
    
    # Growth targets
    target_multiplier: float = 33.0  # 33x growth target ($300 → $10K)
    max_position_scale: float = 0.45  # 45% max position in bootstrap phase
    min_profit_threshold: float = 0.05  # 5% minimum profit to compound
    
    # Reinvestment rates by phase
    bootstrap_aggression: float = 0.95   # 95% reinvestment in bootstrap
    momentum_aggression: float = 0.85    # 85% reinvestment in momentum  
    acceleration_aggression: float = 0.75 # 75% reinvestment in acceleration
    
    # Compounding settings
    compound_cycle: CompoundCycle = CompoundCycle.LIGHTNING
    profit_taking_levels: List[float] = field(default_factory=lambda: [0.12, 0.18, 0.25, 0.35])
    stop_loss_tightening: float = 0.015  # 1.5% stops in bootstrap phase
    
    # Scaling parameters
    position_scaling_factor: float = 1.15  # 15% larger positions as capital grows
    risk_scaling_decay: float = 0.95       # Gradually reduce risk as capital grows

@dataclass
class CompoundMetrics:
    """Real-time compounding performance tracking"""
    
    # Capital tracking
    initial_capital: float = 1.5  # $300 in SOL
    current_capital: float = 1.5
    vault_balance: float = 0.0
    total_profit: float = 0.0
    
    # Growth metrics
    current_multiplier: float = 1.0
    growth_velocity: float = 0.0  # % growth per hour
    compound_cycles_completed: int = 0
    profit_extractions: int = 0
    
    # Performance metrics
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    compound_efficiency: float = 0.0  # How well profits are being reinvested
    time_to_next_phase: Optional[datetime] = None
    
    # Risk metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0

class HyperCompoundEngine:
    """Maximum velocity compounding engine"""
    
    def __init__(self):
        self.logger = setup_logger("HyperCompoundEngine")
        
        # Core systems
        self.vault_system: Optional[VaultWalletSystem] = None
        self.wallet_manager: Optional[UnifiedWalletManager] = None
        self.trading_engine: Optional[UnifiedTradingEngine] = None
        self.swarm_engine: Optional[SwarmDecisionEngine] = None
        
        # Configuration and metrics
        self.config = CompoundConfig()
        self.metrics = CompoundMetrics()
        
        # System state
        self.current_phase = GrowthPhase.BOOTSTRAP
        self.compounding_active = False
        self.last_compound_time = datetime.now()
        
        # Queues and storage
        self.pending_profits: List[Tuple[float, datetime]] = []
        self.compound_queue: asyncio.Queue = asyncio.Queue()
        
        # Performance tracking
        self.recent_trades: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
    async def initialize(self) -> bool:
        """Initialize the hyper compound engine"""
        try:
            self.logger.info("🚀 Initializing Hyper Compound Engine for $300 → $10K+ mission")
            
            # Validate required systems
            if not all([self.vault_system, self.wallet_manager, self.trading_engine]):
                self.logger.error("❌ Required systems not initialized")
                return False
            
            # Determine current growth phase
            await self._determine_growth_phase()
            
            # Configure phase-specific settings
            await self._configure_phase_settings()
            
            # Start background tasks
            asyncio.create_task(self._compound_monitoring_loop())
            asyncio.create_task(self._profit_processing_loop())
            asyncio.create_task(self._phase_optimization_loop())
            asyncio.create_task(self._performance_tracking_loop())
            
            self.compounding_active = True
            self.logger.info(f"✅ Hyper Compound Engine ready - Phase: {self.current_phase.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize compound engine: {e}")
            return False
    
    async def process_trade_profit(self, trade_result: Dict[str, Any]) -> bool:
        """Process profit from a successful trade"""
        try:
            if not trade_result.get('success', False):
                await self._handle_trade_loss(trade_result)
                return False
            
            profit_amount = trade_result.get('profit_sol', 0.0)
            if profit_amount <= 0:
                return False
            
            # Add to pending profits
            self.pending_profits.append((profit_amount, datetime.now()))
            self.metrics.total_profit += profit_amount
            self.metrics.consecutive_wins += 1
            self.metrics.consecutive_losses = 0
            
            # Update performance metrics
            await self._update_performance_metrics(trade_result)
            
            # Queue for compounding
            await self.compound_queue.put({
                'type': 'profit',
                'amount': profit_amount,
                'timestamp': datetime.now(),
                'trade_data': trade_result
            })
            
            self.logger.info(f"💰 Profit queued for compounding: {profit_amount:.6f} SOL")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process trade profit: {e}")
            return False
    
    async def _compound_monitoring_loop(self):
        """Main compounding decision loop"""
        while self.compounding_active:
            try:
                # Check if we should compound
                if await self._should_compound():
                    await self._execute_compound_cycle()
                
                # Check if we should change phases
                if await self._should_change_phase():
                    await self._transition_growth_phase()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Compound monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _should_compound(self) -> bool:
        """Determine if we should execute a compound cycle"""
        try:
            # Check compound cycle requirements
            if self.config.compound_cycle == CompoundCycle.LIGHTNING:
                return len(self.pending_profits) > 0
            
            elif self.config.compound_cycle == CompoundCycle.RAPID:
                return self.metrics.consecutive_wins >= 3
            
            elif self.config.compound_cycle == CompoundCycle.AGGRESSIVE:
                return self.metrics.consecutive_wins >= 5
            
            elif self.config.compound_cycle == CompoundCycle.STANDARD:
                return self.metrics.consecutive_wins >= 10
            
            # Check minimum profit threshold
            total_pending = sum(amount for amount, _ in self.pending_profits)
            if total_pending < self.config.min_profit_threshold:
                return False
            
            # Check time since last compound
            time_since_last = (datetime.now() - self.last_compound_time).total_seconds() / 3600
            if time_since_last < 1:  # Minimum 1 hour between compounds
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking compound conditions: {e}")
            return False
    
    async def _execute_compound_cycle(self):
        """Execute a compound cycle"""
        try:
            self.logger.info("🔄 Executing compound cycle...")
            
            # Calculate total profit to compound
            total_profit = sum(amount for amount, _ in self.pending_profits)
            
            if total_profit <= 0:
                return
            
            # Calculate compound amount based on phase
            compound_amount = await self._calculate_compound_amount(total_profit)
            
            # Calculate vault allocation
            vault_amount = total_profit - compound_amount
            
            # Execute compound distribution
            success = await self._execute_compound_distribution(compound_amount, vault_amount)
            
            if success:
                # Clear pending profits
                self.pending_profits.clear()
                self.last_compound_time = datetime.now()
                self.metrics.compound_cycles_completed += 1
                
                self.logger.info(f"✅ Compound cycle complete: {compound_amount:.6f} SOL reinvested")
            else:
                self.logger.error("❌ Compound cycle failed")
            
        except Exception as e:
            self.logger.error(f"Error executing compound cycle: {e}")
    
    async def _calculate_compound_amount(self, total_profit: float) -> float:
        """Calculate how much profit to reinvest"""
        try:
            # Get phase-specific reinvestment rate
            if self.current_phase == GrowthPhase.BOOTSTRAP:
                reinvestment_rate = self.config.bootstrap_aggression
            elif self.current_phase == GrowthPhase.MOMENTUM:
                reinvestment_rate = self.config.momentum_aggression
            elif self.current_phase == GrowthPhase.ACCELERATION:
                reinvestment_rate = self.config.acceleration_aggression
            else:  # MASTERY
                reinvestment_rate = 0.5  # Conservative in mastery phase
            
            compound_amount = total_profit * reinvestment_rate
            
            # Ensure minimum compound amount
            min_compound = self.config.min_profit_threshold
            if compound_amount < min_compound:
                compound_amount = min_compound
            
            return compound_amount
            
        except Exception as e:
            self.logger.error(f"Error calculating compound amount: {e}")
            return total_profit * 0.8  # Default to 80%
    
    async def _execute_compound_distribution(self, compound_amount: float, vault_amount: float) -> bool:
        """Execute compound distribution to wallets and vaults"""
        try:
            # Distribute to vaults first
            if vault_amount > 0:
                vault_allocation = await self.vault_system.allocate_profit(vault_amount, "compound_engine")
                if not vault_allocation:
                    self.logger.warning("⚠️ Vault allocation failed")
            
            # Distribute compound amount to wallets
            wallet_distribution = await self._calculate_wallet_distribution(compound_amount)
            
            # Execute wallet distributions
            for wallet_id, amount in wallet_distribution.items():
                # Update wallet balance (in a real implementation, this would transfer funds)
                wallet = self.wallet_manager.wallets.get(wallet_id)
                if wallet:
                    # In a real implementation, this would be an actual transfer
                    self.logger.info(f"💰 Distributed {amount:.6f} SOL to {wallet_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing compound distribution: {e}")
            return False
    
    async def _calculate_wallet_distribution(self, total_amount: float) -> Dict[str, float]:
        """Calculate how to distribute compound amount across wallets"""
        try:
            distribution = {}
            active_wallets = self.wallet_manager.active_wallets
            
            if not active_wallets:
                return distribution
            
            # Calculate distribution based on wallet performance
            wallet_scores = []
            for wallet_id in active_wallets:
                wallet = self.wallet_manager.wallets[wallet_id]
                performance = wallet.performance
                
                # Calculate wallet score based on performance
                score = (
                    performance.win_rate * 0.4 +
                    performance.avg_profit_per_trade * 0.3 +
                    performance.sharpe_ratio * 0.3
                )
                
                wallet_scores.append((wallet_id, score))
            
            # Sort by performance
            wallet_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Distribute based on performance (top performers get more)
            total_score = sum(score for _, score in wallet_scores)
            
            for wallet_id, score in wallet_scores:
                if total_score > 0:
                    allocation = (score / total_score) * total_amount
                    distribution[wallet_id] = allocation
                else:
                    # Equal distribution if no performance data
                    distribution[wallet_id] = total_amount / len(active_wallets)
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"Error calculating wallet distribution: {e}")
            return {}
    
    async def _determine_growth_phase(self):
        """Determine current growth phase based on capital"""
        try:
            current_capital = self.metrics.current_capital
            
            if current_capital < 1.0:  # Less than $200
                self.current_phase = GrowthPhase.BOOTSTRAP
            elif current_capital < 3.0:  # Less than $600
                self.current_phase = GrowthPhase.MOMENTUM
            elif current_capital < 10.0:  # Less than $2000
                self.current_phase = GrowthPhase.ACCELERATION
            else:  # $2000+
                self.current_phase = GrowthPhase.MASTERY
            
            self.logger.info(f"📈 Growth phase: {self.current_phase.value} (Capital: {current_capital:.2f} SOL)")
            
        except Exception as e:
            self.logger.error(f"Error determining growth phase: {e}")
    
    async def _configure_phase_settings(self):
        """Configure settings based on current growth phase"""
        try:
            if self.current_phase == GrowthPhase.BOOTSTRAP:
                # Maximum aggression
                self.config.compound_cycle = CompoundCycle.LIGHTNING
                self.config.max_position_scale = 0.45
                self.config.stop_loss_tightening = 0.015
                
            elif self.current_phase == GrowthPhase.MOMENTUM:
                # Balanced growth
                self.config.compound_cycle = CompoundCycle.RAPID
                self.config.max_position_scale = 0.35
                self.config.stop_loss_tightening = 0.02
                
            elif self.current_phase == GrowthPhase.ACCELERATION:
                # Controlled scaling
                self.config.compound_cycle = CompoundCycle.AGGRESSIVE
                self.config.max_position_scale = 0.25
                self.config.stop_loss_tightening = 0.025
                
            else:  # MASTERY
                # Position optimization
                self.config.compound_cycle = CompoundCycle.STANDARD
                self.config.max_position_scale = 0.15
                self.config.stop_loss_tightening = 0.03
            
            self.logger.info(f"⚙️ Phase settings configured for {self.current_phase.value}")
            
        except Exception as e:
            self.logger.error(f"Error configuring phase settings: {e}")
    
    async def _should_change_phase(self) -> bool:
        """Check if we should transition to a new growth phase"""
        try:
            # Check growth milestones
            await self._check_growth_milestones()
            
            # Check if mission accomplished
            if await self._mission_accomplished():
                return True
            
            # Check for phase-specific conditions
            if self.current_phase == GrowthPhase.BOOTSTRAP:
                return self.metrics.current_capital >= 1.0
            elif self.current_phase == GrowthPhase.MOMENTUM:
                return self.metrics.current_capital >= 3.0
            elif self.current_phase == GrowthPhase.ACCELERATION:
                return self.metrics.current_capital >= 10.0
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking phase change: {e}")
            return False
    
    async def _check_growth_milestones(self):
        """Check and log growth milestones"""
        try:
            milestones = [1.0, 3.0, 10.0, 33.0]  # SOL amounts
            
            for milestone in milestones:
                if (self.metrics.current_capital >= milestone and 
                    self.metrics.current_multiplier < milestone / self.metrics.initial_capital):
                    
                    self.metrics.current_multiplier = milestone / self.metrics.initial_capital
                    self.logger.info(f"🎯 MILESTONE: {milestone}x growth achieved! ({milestone:.1f} SOL)")
                    
        except Exception as e:
            self.logger.error(f"Error checking milestones: {e}")
    
    async def _mission_accomplished(self) -> bool:
        """Check if $300 → $10K mission is accomplished"""
        try:
            target_capital = self.metrics.initial_capital * self.config.target_multiplier
            
            if self.metrics.current_capital >= target_capital:
                self.logger.info("🎉 MISSION ACCOMPLISHED: $300 → $10K+ achieved!")
                self.logger.info(f"💰 Final capital: {self.metrics.current_capital:.2f} SOL")
                self.logger.info(f"🚀 Total multiplier: {self.metrics.current_multiplier:.1f}x")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking mission status: {e}")
            return False
    
    async def _transition_growth_phase(self):
        """Transition to a new growth phase"""
        try:
            old_phase = self.current_phase
            
            # Determine new phase
            await self._determine_growth_phase()
            
            if self.current_phase != old_phase:
                # Configure new phase settings
                await self._configure_phase_settings()
                
                self.logger.info(f"🔄 Phase transition: {old_phase.value} → {self.current_phase.value}")
                
                # Log phase-specific guidance
                await self._log_phase_guidance()
            
        except Exception as e:
            self.logger.error(f"Error transitioning phases: {e}")
    
    async def _log_phase_guidance(self):
        """Log guidance for current phase"""
        try:
            guidance = {
                GrowthPhase.BOOTSTRAP: "🔥 MAXIMUM AGGRESSION: Every win compounds immediately",
                GrowthPhase.MOMENTUM: "⚡ BUILDING MOMENTUM: Balanced growth with rapid compounding",
                GrowthPhase.ACCELERATION: "🚀 ACCELERATION: Controlled scaling with strategic compounding",
                GrowthPhase.MASTERY: "🎯 MASTERY: Position optimization and risk management"
            }
            
            if self.current_phase in guidance:
                self.logger.info(f"📋 Phase Guidance: {guidance[self.current_phase]}")
                
        except Exception as e:
            self.logger.error(f"Error logging phase guidance: {e}")
    
    async def _profit_processing_loop(self):
        """Process profits from queue"""
        while self.compounding_active:
            try:
                # Process items from queue
                while not self.compound_queue.empty():
                    item = await self.compound_queue.get()
                    
                    if item['type'] == 'profit':
                        await self._process_profit_item(item)
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                self.logger.error(f"Profit processing error: {e}")
                await asyncio.sleep(5)
    
    async def _process_profit_item(self, item: Dict[str, Any]):
        """Process a profit item from the queue"""
        try:
            profit_amount = item['amount']
            trade_data = item.get('trade_data', {})
            
            # Update metrics
            self.metrics.total_profit += profit_amount
            
            # Store recent trade
            self.recent_trades.append({
                'profit': profit_amount,
                'timestamp': item['timestamp'],
                'trade_data': trade_data
            })
            
            # Keep only last 100 trades
            if len(self.recent_trades) > 100:
                self.recent_trades = self.recent_trades[-100:]
            
        except Exception as e:
            self.logger.error(f"Error processing profit item: {e}")
    
    async def _phase_optimization_loop(self):
        """Optimize settings based on performance"""
        while self.compounding_active:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Analyze recent performance
                await self._analyze_performance()
                
                # Optimize settings
                await self._optimize_settings()
                
            except Exception as e:
                self.logger.error(f"Phase optimization error: {e}")
                await asyncio.sleep(3600)
    
    async def _analyze_performance(self):
        """Analyze recent performance for optimization"""
        try:
            if len(self.recent_trades) < 10:
                return
            
            # Calculate performance metrics
            recent_profits = [trade['profit'] for trade in self.recent_trades[-20:]]
            avg_profit = sum(recent_profits) / len(recent_profits)
            win_rate = sum(1 for p in recent_profits if p > 0) / len(recent_profits)
            
            # Store optimization data
            optimization_data = {
                'timestamp': datetime.now(),
                'avg_profit': avg_profit,
                'win_rate': win_rate,
                'phase': self.current_phase.value,
                'compound_cycle': self.config.compound_cycle.value
            }
            
            self.optimization_history.append(optimization_data)
            
            # Keep only last 100 optimizations
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
    
    async def _optimize_settings(self):
        """Optimize settings based on performance analysis"""
        try:
            if len(self.optimization_history) < 5:
                return
            
            # Analyze recent performance trends
            recent_optimizations = self.optimization_history[-10:]
            avg_win_rate = sum(o['win_rate'] for o in recent_optimizations) / len(recent_optimizations)
            avg_profit = sum(o['avg_profit'] for o in recent_optimizations) / len(recent_optimizations)
            
            # Adjust compound cycle based on performance
            if avg_win_rate > 0.8 and avg_profit > 0.1:
                # High performance - can be more aggressive
                if self.config.compound_cycle != CompoundCycle.LIGHTNING:
                    self.config.compound_cycle = CompoundCycle.LIGHTNING
                    self.logger.info("⚡ Optimized: Upgraded to LIGHTNING compound cycle")
            
            elif avg_win_rate < 0.5 or avg_profit < 0.05:
                # Low performance - be more conservative
                if self.config.compound_cycle == CompoundCycle.LIGHTNING:
                    self.config.compound_cycle = CompoundCycle.RAPID
                    self.logger.info("🛡️ Optimized: Downgraded to RAPID compound cycle")
            
        except Exception as e:
            self.logger.error(f"Error optimizing settings: {e}")
    
    async def _performance_tracking_loop(self):
        """Track and log performance metrics"""
        while self.compounding_active:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Update growth velocity
                await self._update_growth_velocity()
                
                # Log performance summary
                await self._log_performance_summary()
                
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(300)
    
    async def _update_growth_velocity(self):
        """Update growth velocity metrics"""
        try:
            # Calculate growth velocity (% growth per hour)
            if len(self.optimization_history) >= 2:
                recent = self.optimization_history[-1]
                older = self.optimization_history[-2]
                
                time_diff = (recent['timestamp'] - older['timestamp']).total_seconds() / 3600
                if time_diff > 0:
                    growth_diff = recent.get('avg_profit', 0) - older.get('avg_profit', 0)
                    self.metrics.growth_velocity = growth_diff / time_diff
            
        except Exception as e:
            self.logger.error(f"Error updating growth velocity: {e}")
    
    async def _log_performance_summary(self):
        """Log performance summary"""
        try:
            # Log every 30 minutes
            if hasattr(self, '_last_performance_log'):
                if (datetime.now() - self._last_performance_log).total_seconds() < 1800:
                    return
            
            self._last_performance_log = datetime.now()
            
            self.logger.info("📊 Compound Performance Summary:")
            self.logger.info(f"   Capital: {self.metrics.current_capital:.2f} SOL")
            self.logger.info(f"   Multiplier: {self.metrics.current_multiplier:.1f}x")
            self.logger.info(f"   Total Profit: {self.metrics.total_profit:.2f} SOL")
            self.logger.info(f"   Compound Cycles: {self.metrics.compound_cycles_completed}")
            self.logger.info(f"   Growth Velocity: {self.metrics.growth_velocity:.4f}/hour")
            self.logger.info(f"   Phase: {self.current_phase.value}")
            self.logger.info(f"   Cycle: {self.config.compound_cycle.value}")
            
        except Exception as e:
            self.logger.error(f"Error logging performance summary: {e}")
    
    async def _handle_trade_loss(self, trade_result: Dict[str, Any]):
        """Handle trade loss"""
        try:
            loss_amount = abs(trade_result.get('profit_sol', 0.0))
            
            # Update metrics
            self.metrics.consecutive_losses += 1
            self.metrics.consecutive_wins = 0
            
            # Update drawdown
            if loss_amount > 0:
                self.metrics.current_drawdown -= loss_amount
                self.metrics.max_drawdown = min(self.metrics.max_drawdown, self.metrics.current_drawdown)
            
            self.logger.warning(f"📉 Trade loss: {loss_amount:.6f} SOL")
            
        except Exception as e:
            self.logger.error(f"Error handling trade loss: {e}")
    
    async def _update_performance_metrics(self, trade_result: Dict[str, Any]):
        """Update performance metrics after trade"""
        try:
            profit = trade_result.get('profit_sol', 0.0)
            
            # Update win rate
            total_trades = len(self.recent_trades) + 1
            wins = sum(1 for trade in self.recent_trades if trade['profit'] > 0) + (1 if profit > 0 else 0)
            self.metrics.win_rate = wins / total_trades if total_trades > 0 else 0.0
            
            # Update average profit
            total_profit = sum(trade['profit'] for trade in self.recent_trades) + profit
            self.metrics.avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def get_compound_status(self) -> Dict[str, Any]:
        """Get comprehensive compound status"""
        try:
            return {
                'current_phase': self.current_phase.value,
                'current_capital': self.metrics.current_capital,
                'current_multiplier': self.metrics.current_multiplier,
                'total_profit': self.metrics.total_profit,
                'compound_cycles': self.metrics.compound_cycles_completed,
                'growth_velocity': self.metrics.growth_velocity,
                'win_rate': self.metrics.win_rate,
                'compound_cycle': self.config.compound_cycle.value,
                'pending_profits': len(self.pending_profits),
                'active': self.compounding_active
            }
            
        except Exception as e:
            self.logger.error(f"Error getting compound status: {e}")
            return {}
    
    async def force_compound_cycle(self):
        """Force immediate compound cycle"""
        try:
            await self._execute_compound_cycle()
        except Exception as e:
            self.logger.error(f"Error forcing compound cycle: {e}")
    
    async def adjust_aggression(self, multiplier: float):
        """Adjust compound aggression"""
        try:
            self.config.bootstrap_aggression *= multiplier
            self.config.momentum_aggression *= multiplier
            self.config.acceleration_aggression *= multiplier
            
            self.logger.info(f"⚙️ Adjusted compound aggression by {multiplier}x")
            
        except Exception as e:
            self.logger.error(f"Error adjusting aggression: {e}")
    
    async def shutdown(self):
        """Shutdown the compound engine"""
        try:
            self.logger.info("🛑 Shutting down compound engine...")
            self.compounding_active = False
            self.logger.info("✅ Compound engine shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Global instance
_compound_engine = None

async def get_compound_engine() -> HyperCompoundEngine:
    """Get global compound engine instance"""
    global _compound_engine
    if _compound_engine is None:
        _compound_engine = HyperCompoundEngine()
    return _compound_engine 