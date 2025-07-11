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
    
    
    target_multiplier: float = 33.0  # 33x growth target ($300 → $10K)
    max_position_scale: float = 0.45  # 45% max position in bootstrap phase
    min_profit_threshold: float = 0.05  # 5% minimum profit to compound
    
    
    bootstrap_aggression: float = 0.95   # 95% reinvestment in bootstrap
    momentum_aggression: float = 0.85    # 85% reinvestment in momentum  
    acceleration_aggression: float = 0.75 # 75% reinvestment in acceleration
    
    
    compound_cycle: CompoundCycle = CompoundCycle.LIGHTNING
    profit_taking_levels: List[float] = field(default_factory=lambda: [0.12, 0.18, 0.25, 0.35])
    stop_loss_tightening: float = 0.015  # 1.5% stops in bootstrap phase
    
    
    position_scaling_factor: float = 1.15  # 15% larger positions as capital grows
    risk_scaling_decay: float = 0.95       # Gradually reduce risk as capital grows

@dataclass
class CompoundMetrics:
    """Real-time compounding performance tracking"""
    
    
    initial_capital: float = 1.5  # $300 in SOL
    current_capital: float = 1.5
    vault_balance: float = 0.0
    total_profit: float = 0.0
    
    
    current_multiplier: float = 1.0
    growth_velocity: float = 0.0  # % growth per hour
    compound_cycles_completed: int = 0
    profit_extractions: int = 0
    
    
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    compound_efficiency: float = 0.0  # How well profits are being reinvested
    time_to_next_phase: Optional[datetime] = None
    
    
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0

class HyperCompoundEngine:
    """Maximum velocity compounding engine"""
    
    def __init__(self):
        self.logger = setup_logger("HyperCompoundEngine")
        
        
        self.vault_system: Optional[VaultWalletSystem] = None
        self.wallet_manager: Optional[UnifiedWalletManager] = None
        self.trading_engine: Optional[UnifiedTradingEngine] = None
        self.swarm_engine: Optional[SwarmDecisionEngine] = None
        
        
        self.config = CompoundConfig()
        self.metrics = CompoundMetrics()
        
        
        self.current_phase = GrowthPhase.BOOTSTRAP
        self.compounding_active = False
        self.last_compound_time = datetime.now()
        
        
        self.pending_profits: List[Tuple[float, datetime]] = []
        self.compound_queue: asyncio.Queue = asyncio.Queue()
        
        
        self.recent_trades: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
    async def initialize(self) -> bool:
        """Initialize the hyper compound engine"""
        try:
            self.logger.info("🚀 Initializing Hyper Compound Engine for $300 → $10K+ mission")
            
            
            if not all([self.vault_system, self.wallet_manager, self.trading_engine]):
                self.logger.error("❌ Required systems not initialized")
                return False
            
            
            await self._determine_growth_phase()
            
            
            await self._configure_phase_settings()
            
            
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
            
            
            self.pending_profits.append((profit_amount, datetime.now()))
            self.metrics.total_profit += profit_amount
            self.metrics.consecutive_wins += 1
            self.metrics.consecutive_losses = 0
            
            
            await self._update_performance_metrics(trade_result)
            
            
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
                if await self._should_compound():
                    await self._execute_compound_cycle()
                
                
                if await self._should_change_phase():
                    await self._transition_growth_phase()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Compound monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _profit_processing_loop(self):
        """Process profits from the queue"""
        while self.compounding_active:
            try:
                profit_data = await asyncio.wait_for(
                    self.compound_queue.get(), 
                    timeout=10
                )
                
                await self._process_profit_compound(profit_data)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"❌ Profit processing error: {e}")
                await asyncio.sleep(5)
    
    async def _should_compound(self) -> bool:
        """Determine if compounding should trigger"""
        
        
        if self.config.compound_cycle == CompoundCycle.LIGHTNING:
            return len(self.pending_profits) > 0
        
        
        elif self.config.compound_cycle == CompoundCycle.RAPID:
            return len(self.pending_profits) >= 3
        
        
        elif self.config.compound_cycle == CompoundCycle.AGGRESSIVE:
            return len(self.pending_profits) >= 5
        
        
        else:
            return len(self.pending_profits) >= 10
    
    async def _execute_compound_cycle(self):
        """Execute a complete compound cycle"""
        try:
            if not self.pending_profits:
                return
            
            
            total_profit = sum(profit for profit, _ in self.pending_profits)
            
            if total_profit < self.config.min_profit_threshold:
                return
            
            
            compound_amount = await self._calculate_compound_amount(total_profit)
            vault_amount = total_profit - compound_amount
            
            
            success = await self._execute_compound_distribution(
                compound_amount, 
                vault_amount
            )
            
            if success:
            if success:
                self.pending_profits.clear()
                self.metrics.compound_cycles_completed += 1
                self.last_compound_time = datetime.now()
                
                
                self.metrics.current_capital += compound_amount
                self.metrics.vault_balance += vault_amount
                self.metrics.current_multiplier = self.metrics.current_capital / self.metrics.initial_capital
                
                self.logger.info(
                    f"🔄 Compound cycle completed: {compound_amount:.6f} SOL reinvested, "
                    f"{vault_amount:.6f} SOL vaulted. "
                    f"Current multiplier: {self.metrics.current_multiplier:.2f}x"
                )
                
                
                await self._check_growth_milestones()
                
        except Exception as e:
            self.logger.error(f"❌ Compound cycle failed: {e}")
    
    async def _calculate_compound_amount(self, total_profit: float) -> float:
        """Calculate how much to reinvest vs vault"""
        
        
        if self.current_phase == GrowthPhase.BOOTSTRAP:
            reinvest_rate = self.config.bootstrap_aggression
        elif self.current_phase == GrowthPhase.MOMENTUM:
            reinvest_rate = self.config.momentum_aggression
        elif self.current_phase == GrowthPhase.ACCELERATION:
            reinvest_rate = self.config.acceleration_aggression
        else:
            reinvest_rate = 0.6  # Conservative in mastery phase
        
        
        if self.metrics.consecutive_wins >= 5:
            reinvest_rate = min(0.98, reinvest_rate + 0.05)  # Boost on hot streak
        elif self.metrics.consecutive_losses >= 2:
            reinvest_rate = max(0.5, reinvest_rate - 0.1)   # Reduce on cold streak
        
        return total_profit * reinvest_rate
    
    async def _execute_compound_distribution(self, compound_amount: float, vault_amount: float) -> bool:
        """Execute the actual compound distribution"""
        try:
            if vault_amount > 0:
                vault_success = await self.vault_system.deposit_profit(
                    vault_amount, 
                    VaultType.DAILY
                )
                if not vault_success:
                    self.logger.warning(f"⚠️ Vault deposit failed: {vault_amount:.6f} SOL")
            
            
            if compound_amount > 0:
                distribution = await self._calculate_wallet_distribution(compound_amount)
                
                for wallet_id, amount in distribution.items():
                    if amount > 0.01:  # Minimum threshold
                        success = await self.wallet_manager.add_trading_capital(
                            wallet_id, 
                            amount
                        )
                        if success:
                            self.logger.info(f"💼 Added {amount:.6f} SOL to wallet {wallet_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Compound distribution failed: {e}")
            return False
    
    async def _calculate_wallet_distribution(self, total_amount: float) -> Dict[str, float]:
        """Calculate how to distribute compound capital across wallets"""
        try:
            active_wallets = await self.wallet_manager.get_active_wallets()
            
            if not active_wallets:
                return {}
            
            
            # Performance-based distribution
            wallet_scores = {}
            
            for wallet_id, wallet in active_wallets.items():
            for wallet_id, wallet in active_wallets.items():
                win_rate = wallet.performance_metrics.get('win_rate', 0.5)
                profit_ratio = wallet.performance_metrics.get('profit_ratio', 1.0)
                consistency = wallet.performance_metrics.get('consistency', 0.5)
                
                
                genetic_bonus = (
                    wallet.genetics.aggression * 0.3 +
                    wallet.genetics.adaptability * 0.2 +
                    wallet.genetics.signal_trust * 0.1
                )
                
                score = (win_rate * 0.4 + profit_ratio * 0.3 + 
                        consistency * 0.2 + genetic_bonus * 0.1)
                
                wallet_scores[wallet_id] = max(0.1, score)  # Minimum 10% allocation
            
            
            total_score = sum(wallet_scores.values())
            if total_score > 0:
                for wallet_id in wallet_scores:
                    wallet_scores[wallet_id] /= total_score
            
            
            distribution = {}
            for wallet_id, ratio in wallet_scores.items():
                distribution[wallet_id] = total_amount * ratio
            
            return distribution
            
        except Exception as e:
            self.logger.error(f"❌ Wallet distribution calculation failed: {e}")
            return {}
    
    async def _determine_growth_phase(self):
        """Determine current growth phase based on capital"""
        current_multiplier = self.metrics.current_multiplier
        
        if current_multiplier < 3.33:  # Less than $1K equivalent
            self.current_phase = GrowthPhase.BOOTSTRAP
        elif current_multiplier < 10.0:  # Less than $3K equivalent
            self.current_phase = GrowthPhase.MOMENTUM
        elif current_multiplier < 33.3:  # Less than $10K equivalent
            self.current_phase = GrowthPhase.ACCELERATION
        else:
            self.current_phase = GrowthPhase.MASTERY
    
    async def _configure_phase_settings(self):
        """Configure settings based on current growth phase"""
        
        if self.current_phase == GrowthPhase.BOOTSTRAP:
        if self.current_phase == GrowthPhase.BOOTSTRAP:
            self.config.compound_cycle = CompoundCycle.LIGHTNING
            self.config.max_position_scale = 0.45
            self.config.stop_loss_tightening = 0.015  # 1.5% stops
            
        elif self.current_phase == GrowthPhase.MOMENTUM:
        elif self.current_phase == GrowthPhase.MOMENTUM:
            self.config.compound_cycle = CompoundCycle.RAPID
            self.config.max_position_scale = 0.35
            self.config.stop_loss_tightening = 0.02   # 2% stops
            
        elif self.current_phase == GrowthPhase.ACCELERATION:
        elif self.current_phase == GrowthPhase.ACCELERATION:
            self.config.compound_cycle = CompoundCycle.AGGRESSIVE
            self.config.max_position_scale = 0.25
            self.config.stop_loss_tightening = 0.025  # 2.5% stops
            
        else:  # MASTERY
        else:  # MASTERY
            self.config.compound_cycle = CompoundCycle.STANDARD
            self.config.max_position_scale = 0.15
            self.config.stop_loss_tightening = 0.03   # 3% stops
        
        self.logger.info(f"📊 Phase configured: {self.current_phase.value} - Cycle: {self.config.compound_cycle.value}")
    
    async def _check_growth_milestones(self):
        """Check and celebrate growth milestones"""
        multiplier = self.metrics.current_multiplier
        
        milestones = [2, 3, 5, 10, 15, 20, 25, 30, 33]
        
        for milestone in milestones:
            if (multiplier >= milestone and 
                not hasattr(self, f'milestone_{milestone}_reached')):
                
                equivalent_usd = int(300 * milestone)
                self.logger.info(
                    f"🎯 MILESTONE REACHED: {milestone}x growth! "
                    f"${equivalent_usd} equivalent achieved!"
                )
                setattr(self, f'milestone_{milestone}_reached', True)
                
                
                if milestone == 33:  # Mission accomplished!
                    self.logger.info("🏆 MISSION ACCOMPLISHED: $10K+ TARGET REACHED!")
                    await self._mission_accomplished()
    
    async def _mission_accomplished(self):
        """Handle mission completion"""
        """Handle mission completion"""
        self.current_phase = GrowthPhase.MASTERY
        await self._configure_phase_settings()
        
        
        extraction_amount = self.metrics.current_capital * 0.3  # Extract 30%
        await self.vault_system.deposit_profit(extraction_amount, VaultType.LEGENDARY)
        
        self.logger.info(f"👑 Switched to mastery mode. {extraction_amount:.6f} SOL secured in legendary vault.")
    
    async def _performance_tracking_loop(self):
        """Track and log performance metrics"""
        while self.compounding_active:
            try:
                time_elapsed = (datetime.now() - self.last_compound_time).total_seconds() / 3600
                if time_elapsed > 0:
                    self.metrics.growth_velocity = (
                        (self.metrics.current_multiplier - 1.0) / time_elapsed * 100
                    )
                
                
                await asyncio.sleep(600)
                await self._log_performance_summary()
                
            except Exception as e:
                self.logger.error(f"❌ Performance tracking error: {e}")
                await asyncio.sleep(300)
    
    async def _log_performance_summary(self):
        """Log comprehensive performance summary"""
        
        self.logger.info(
            f"📈 COMPOUND ENGINE STATUS:\n"
            f"   💰 Capital: {self.metrics.current_capital:.4f} SOL ({self.metrics.current_multiplier:.2f}x)\n"
            f"   🏦 Vault: {self.metrics.vault_balance:.4f} SOL\n"
            f"   📊 Total Profit: {self.metrics.total_profit:.4f} SOL\n"
            f"   🔄 Cycles: {self.metrics.compound_cycles_completed}\n"
            f"   📈 Growth Rate: {self.metrics.growth_velocity:.2f}%/hour\n"
            f"   🎯 Phase: {self.current_phase.value}\n"
            f"   ⚡ Win Streak: {self.metrics.consecutive_wins}\n"
            f"   🎪 Mission Progress: {(self.metrics.current_multiplier/33.0)*100:.1f}%"
        )
    
    async def _handle_trade_loss(self, trade_result: Dict[str, Any]):
        """Handle trade losses and adjust strategy"""
        loss_amount = abs(trade_result.get('profit_sol', 0.0))
        
        self.metrics.consecutive_losses += 1
        self.metrics.consecutive_wins = 0
        
        
        if self.metrics.consecutive_losses >= 3:
        if self.metrics.consecutive_losses >= 3:
            self.config.max_position_scale *= 0.9
            self.logger.warning(f"⚠️ Reducing position scale due to {self.metrics.consecutive_losses} consecutive losses")
        
        
        current_drawdown = loss_amount / self.metrics.current_capital
        self.metrics.current_drawdown = current_drawdown
        if current_drawdown > self.metrics.max_drawdown:
            self.metrics.max_drawdown = current_drawdown
    
    
    async def get_compound_status(self) -> Dict[str, Any]:
        """Get current compound engine status"""
        return {
            'phase': self.current_phase.value,
            'multiplier': self.metrics.current_multiplier,
            'mission_progress': (self.metrics.current_multiplier / 33.0) * 100,
            'cycles_completed': self.metrics.compound_cycles_completed,
            'growth_velocity': self.metrics.growth_velocity,
            'win_streak': self.metrics.consecutive_wins,
            'vault_balance': self.metrics.vault_balance,
            'compound_cycle': self.config.compound_cycle.value,
            'active': self.compounding_active
        }
    
    async def force_compound_cycle(self):
        """Manually trigger a compound cycle"""
        if self.pending_profits:
            await self._execute_compound_cycle()
    
    async def adjust_aggression(self, multiplier: float):
        """Manually adjust aggression level"""
        self.config.bootstrap_aggression *= multiplier
        self.config.momentum_aggression *= multiplier
        self.config.acceleration_aggression *= multiplier
        
        self.logger.info(f"🎛️ Aggression adjusted by {multiplier}x")
    
    async def shutdown(self):
        """Shutdown the compound engine"""
        self.compounding_active = False
        self.logger.info("🛑 Hyper Compound Engine shutdown")


_compound_engine = None

async def get_compound_engine() -> HyperCompoundEngine:
    """Get global compound engine instance"""
    global _compound_engine
    if _compound_engine is None:
        _compound_engine = HyperCompoundEngine()
    return _compound_engine 