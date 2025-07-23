"""
HYPER COMPOUND ENGINE - ANTI-FRAGILE CAPITAL COMPOUNDING
======================================================

Enhanced compounding engine with Calibration phase and vault-based profit protection.
Implements anti-fragile principles: validate edge before scaling, secure gains before risks.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

from worker_ant_v1.utils.logger import setup_logger


class GrowthPhase(Enum):
    """Capital growth phases with anti-fragile progression"""
    CALIBRATION = "calibration"    # NEW: Validate statistical edge with fixed sizing
    CONSERVATIVE = "conservative"  # Safe growth with proven edge
    MODERATE = "moderate"         # Balanced growth
    AGGRESSIVE = "aggressive"     # Maximum growth
    DEFENSIVE = "defensive"       # Protect capital during drawdowns


@dataclass
class CompoundingMetrics:
    """Metrics for compounding performance tracking"""
    phase: GrowthPhase
    initial_capital: float
    current_capital: float
    vault_balance: float
    total_trades: int
    winning_trades: int
    total_profit: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    edge_validated: bool
    days_in_phase: int


@dataclass
class CalibrationMetrics:
    """Specific metrics for calibration phase tracking"""
    trades_completed: int
    trades_required: int
    current_win_rate: float
    required_win_rate: float
    edge_validated: bool
    avg_profit_per_trade: float
    avg_trade_size: float
    calibration_start: datetime


class HyperCompoundEngine:
    """
    Anti-fragile compounding engine that validates statistical edge before scaling
    and protects profits through vault-based batching.
    """
    
    def __init__(self, initial_capital: float = 1000.0):
        self.logger = setup_logger("HyperCompoundEngine")
        
        # Capital allocation
        self.initial_capital = initial_capital
        self.active_capital = initial_capital
        self.profit_vault = 0.0
        
        # Current state
        self.current_phase = GrowthPhase.CALIBRATION
        self.phase_start_time = datetime.now()
        
        # Calibration requirements (ANTI-FRAGILE: Prove edge before scaling)
        self.calibration_config = {
            'required_trades': 150,           # Must complete 150 trades
            'required_win_rate': 0.55,        # Must achieve >55% win rate
            'fixed_trade_size': 2.0,          # Fixed $2 per trade during calibration
            'min_profit_per_trade': 0.01,     # Minimum $0.01 profit per trade average
            'max_consecutive_losses': 10       # Reset calibration if >10 consecutive losses
        }
        
        # Phase transition thresholds
        self.phase_thresholds = {
            GrowthPhase.CONSERVATIVE: {'win_rate': 0.60, 'profit_factor': 1.3},
            GrowthPhase.MODERATE: {'win_rate': 0.65, 'profit_factor': 1.5},
            GrowthPhase.AGGRESSIVE: {'win_rate': 0.70, 'profit_factor': 2.0},
            GrowthPhase.DEFENSIVE: {'drawdown': 0.15}  # Trigger at 15% drawdown
        }
        
        # Vault configuration (ANTI-FRAGILE: Secure gains before reinvestment)
        self.vault_config = {
            'profit_deposit_rate': 1.0,       # 100% of profits go to vault initially
            'compound_threshold': 0.20,       # Compound when vault reaches 20% of active capital
            'vault_compound_rate': 0.80,      # 80% of vault balance used for compounding
            'emergency_reserve': 0.05          # 5% always kept in vault as emergency fund
        }
        
        # Position sizing by phase
        self.position_sizing = {
            GrowthPhase.CALIBRATION: {'fixed_size': 2.0},  # Fixed $2
            GrowthPhase.CONSERVATIVE: {'percentage': 0.01},  # 1% of capital
            GrowthPhase.MODERATE: {'percentage': 0.02},      # 2% of capital
            GrowthPhase.AGGRESSIVE: {'percentage': 0.03},    # 3% of capital
            GrowthPhase.DEFENSIVE: {'percentage': 0.005}     # 0.5% of capital
        }
        
        # Trading history for analysis
        self.trade_history: List[Dict[str, Any]] = []
        self.calibration_metrics = CalibrationMetrics(
            trades_completed=0,
            trades_required=self.calibration_config['required_trades'],
            current_win_rate=0.0,
            required_win_rate=self.calibration_config['required_win_rate'],
            edge_validated=False,
            avg_profit_per_trade=0.0,
            avg_trade_size=self.calibration_config['fixed_trade_size'],
            calibration_start=datetime.now()
        )
        
        self.consecutive_losses = 0
        self.last_compound_time = datetime.now()
        
        self.logger.info(f"🧬 HyperCompoundEngine initialized with anti-fragile Calibration phase")
        self.logger.info(f"💰 Starting capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"🎯 Calibration target: {self.calibration_config['required_trades']} trades @ >{self.calibration_config['required_win_rate']:.0%} win rate")
    
    async def calculate_position_size(self, confidence: float = 1.0) -> float:
        """Calculate position size based on current phase and anti-fragile principles"""
        try:
            if self.current_phase == GrowthPhase.CALIBRATION:
                # ANTI-FRAGILE: Fixed sizing during validation
                return self.calibration_config['fixed_trade_size']
            
            # Get base position size for current phase
            sizing_config = self.position_sizing[self.current_phase]
            
            if 'percentage' in sizing_config:
                base_size = self.active_capital * sizing_config['percentage']
            else:
                base_size = sizing_config.get('fixed_size', 10.0)
            
            # Adjust based on confidence
            adjusted_size = base_size * min(1.5, max(0.5, confidence))
            
            # Apply anti-fragile caps
            max_position = self.active_capital * 0.05  # Never risk more than 5%
            final_size = min(adjusted_size, max_position)
            
            self.logger.debug(f"💰 Position size: ${final_size:.2f} (phase: {self.current_phase.value}, confidence: {confidence:.2f})")
            return final_size
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return self.calibration_config['fixed_trade_size']  # Safe fallback
    
    async def process_trade_result(self, profit_loss: float, trade_details: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process trade result with anti-fragile profit handling"""
        try:
            trade_details = trade_details or {}
            timestamp = datetime.now()
            
            # Record trade
            trade_record = {
                'timestamp': timestamp,
                'profit_loss': profit_loss,
                'phase': self.current_phase.value,
                'capital_before': self.active_capital,
                'vault_before': self.profit_vault,
                **trade_details
            }
            self.trade_history.append(trade_record)
            
            # Update consecutive loss tracking
            if profit_loss <= 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Handle profit/loss based on current phase
            if self.current_phase == GrowthPhase.CALIBRATION:
                result = await self._process_calibration_trade(profit_loss, trade_record)
            else:
                result = await self._process_compound_trade(profit_loss, trade_record)
            
            # Check for phase transitions
            await self._check_phase_transition()
            
            # Process vault compounding if threshold reached
            await self._check_vault_compounding()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Trade result processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _process_calibration_trade(self, profit_loss: float, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Process trade during calibration phase - edge validation"""
        try:
            # ANTI-FRAGILE: All profits go to vault during calibration
            if profit_loss > 0:
                self.profit_vault += profit_loss
                self.logger.debug(f"🏦 Calibration profit ${profit_loss:.2f} deposited to vault (balance: ${self.profit_vault:.2f})")
            else:
                # Losses come from active capital
                self.active_capital += profit_loss  # profit_loss is negative
                self.logger.debug(f"📉 Calibration loss ${abs(profit_loss):.2f} from active capital (balance: ${self.active_capital:.2f})")
            
            # Update calibration metrics
            self.calibration_metrics.trades_completed += 1
            
            # Calculate current performance
            recent_trades = self.trade_history[-50:] if len(self.trade_history) >= 50 else self.trade_history
            winning_trades = len([t for t in recent_trades if t['profit_loss'] > 0])
            self.calibration_metrics.current_win_rate = winning_trades / len(recent_trades) if recent_trades else 0.0
            
            total_profit = sum(t['profit_loss'] for t in self.trade_history)
            self.calibration_metrics.avg_profit_per_trade = total_profit / len(self.trade_history) if self.trade_history else 0.0
            
            # Check calibration completion
            trades_completed = self.calibration_metrics.trades_completed
            required_trades = self.calibration_metrics.trades_required
            win_rate_met = self.calibration_metrics.current_win_rate >= self.calibration_metrics.required_win_rate
            profit_met = self.calibration_metrics.avg_profit_per_trade >= self.calibration_config['min_profit_per_trade']
            
            # Check for calibration failure (too many consecutive losses)
            if self.consecutive_losses >= self.calibration_config['max_consecutive_losses']:
                await self._reset_calibration("Excessive consecutive losses")
                return {'success': True, 'phase': 'calibration_reset', 'reason': 'consecutive_losses'}
            
            # Check calibration completion
            if trades_completed >= required_trades:
                if win_rate_met and profit_met:
                    self.calibration_metrics.edge_validated = True
                    self.logger.info(f"✅ CALIBRATION COMPLETED! Edge validated: {self.calibration_metrics.current_win_rate:.1%} win rate, "
                                   f"${self.calibration_metrics.avg_profit_per_trade:.3f} avg profit/trade")
                    await self._transition_to_conservative()
                else:
                    await self._reset_calibration(f"Failed validation: {self.calibration_metrics.current_win_rate:.1%} win rate")
                    return {'success': True, 'phase': 'calibration_reset', 'reason': 'validation_failed'}
            
            return {
                'success': True,
                'phase': 'calibration',
                'trades_completed': trades_completed,
                'trades_remaining': required_trades - trades_completed,
                'current_win_rate': self.calibration_metrics.current_win_rate,
                'edge_validated': self.calibration_metrics.edge_validated,
                'vault_balance': self.profit_vault,
                'active_capital': self.active_capital
            }
            
        except Exception as e:
            self.logger.error(f"Calibration trade processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _process_compound_trade(self, profit_loss: float, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Process trade during compounding phases"""
        try:
            if profit_loss > 0:
                # ANTI-FRAGILE: Profits go to vault for batching
                vault_deposit = profit_loss * self.vault_config['profit_deposit_rate']
                self.profit_vault += vault_deposit
                
                # Any remainder goes to active capital
                remaining_profit = profit_loss - vault_deposit
                self.active_capital += remaining_profit
                
                self.logger.debug(f"💰 Profit ${profit_loss:.2f}: ${vault_deposit:.2f} to vault, ${remaining_profit:.2f} to active capital")
            else:
                # Losses come from active capital
                self.active_capital += profit_loss  # profit_loss is negative
                self.logger.debug(f"📉 Loss ${abs(profit_loss):.2f} from active capital")
            
            return {
                'success': True,
                'phase': self.current_phase.value,
                'profit_loss': profit_loss,
                'vault_balance': self.profit_vault,
                'active_capital': self.active_capital,
                'total_capital': self.active_capital + self.profit_vault
            }
            
        except Exception as e:
            self.logger.error(f"Compound trade processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _reset_calibration(self, reason: str):
        """Reset calibration phase"""
        self.logger.warning(f"🔄 CALIBRATION RESET: {reason}")
        
        # Reset calibration metrics
        self.calibration_metrics = CalibrationMetrics(
            trades_completed=0,
            trades_required=self.calibration_config['required_trades'],
            current_win_rate=0.0,
            required_win_rate=self.calibration_config['required_win_rate'],
            edge_validated=False,
            avg_profit_per_trade=0.0,
            avg_trade_size=self.calibration_config['fixed_trade_size'],
            calibration_start=datetime.now()
        )
        
        self.consecutive_losses = 0
        self.current_phase = GrowthPhase.CALIBRATION
        self.phase_start_time = datetime.now()
    
    async def _transition_to_conservative(self):
        """Transition from calibration to conservative phase"""
        self.logger.info(f"📈 PHASE TRANSITION: CALIBRATION → CONSERVATIVE")
        
        self.current_phase = GrowthPhase.CONSERVATIVE
        self.phase_start_time = datetime.now()
        
        # Compound initial vault balance to active capital
        compound_amount = self.profit_vault * 0.8  # Keep 20% in vault as buffer
        self.active_capital += compound_amount
        self.profit_vault -= compound_amount
        
        self.logger.info(f"💰 Compounded ${compound_amount:.2f} to active capital (new balance: ${self.active_capital:.2f})")
    
    async def _check_phase_transition(self):
        """Check if phase transition is needed based on performance"""
        try:
            if self.current_phase == GrowthPhase.CALIBRATION:
                return  # Handled separately
            
            # Calculate recent performance metrics
            recent_trades = self.trade_history[-100:] if len(self.trade_history) >= 100 else self.trade_history
            if len(recent_trades) < 20:
                return  # Need more data
            
            winning_trades = len([t for t in recent_trades if t['profit_loss'] > 0])
            win_rate = winning_trades / len(recent_trades)
            
            # Calculate profit factor
            profits = sum(t['profit_loss'] for t in recent_trades if t['profit_loss'] > 0)
            losses = abs(sum(t['profit_loss'] for t in recent_trades if t['profit_loss'] < 0))
            profit_factor = profits / losses if losses > 0 else float('inf')
            
            # Calculate drawdown
            capital_history = [self.initial_capital]
            running_capital = self.initial_capital
            for trade in self.trade_history:
                running_capital += trade['profit_loss']
                capital_history.append(running_capital)
            
            peak = max(capital_history)
            current = capital_history[-1]
            drawdown = (peak - current) / peak if peak > 0 else 0
            
            # Check for defensive transition (drawdown-based)
            if drawdown >= self.phase_thresholds[GrowthPhase.DEFENSIVE]['drawdown']:
                if self.current_phase != GrowthPhase.DEFENSIVE:
                    self.logger.warning(f"📉 PHASE TRANSITION: {self.current_phase.value} → DEFENSIVE (drawdown: {drawdown:.1%})")
                    self.current_phase = GrowthPhase.DEFENSIVE
                    self.phase_start_time = datetime.now()
                return
            
            # Check for upward transitions
            target_phase = None
            if (win_rate >= self.phase_thresholds[GrowthPhase.AGGRESSIVE]['win_rate'] and 
                profit_factor >= self.phase_thresholds[GrowthPhase.AGGRESSIVE]['profit_factor']):
                target_phase = GrowthPhase.AGGRESSIVE
            elif (win_rate >= self.phase_thresholds[GrowthPhase.MODERATE]['win_rate'] and 
                  profit_factor >= self.phase_thresholds[GrowthPhase.MODERATE]['profit_factor']):
                target_phase = GrowthPhase.MODERATE
            elif (win_rate >= self.phase_thresholds[GrowthPhase.CONSERVATIVE]['win_rate'] and 
                  profit_factor >= self.phase_thresholds[GrowthPhase.CONSERVATIVE]['profit_factor']):
                target_phase = GrowthPhase.CONSERVATIVE
            
            # Execute transition if appropriate
            if target_phase and target_phase != self.current_phase:
                # Only allow upward transitions (or from defensive)
                phase_order = [GrowthPhase.DEFENSIVE, GrowthPhase.CONSERVATIVE, GrowthPhase.MODERATE, GrowthPhase.AGGRESSIVE]
                current_idx = phase_order.index(self.current_phase)
                target_idx = phase_order.index(target_phase)
                
                if target_idx > current_idx or self.current_phase == GrowthPhase.DEFENSIVE:
                    self.logger.info(f"📈 PHASE TRANSITION: {self.current_phase.value} → {target_phase.value}")
                    self.logger.info(f"📊 Metrics: {win_rate:.1%} win rate, {profit_factor:.1f} profit factor")
                    self.current_phase = target_phase
                    self.phase_start_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Phase transition check failed: {e}")
    
    async def _check_vault_compounding(self):
        """Check if vault balance should be compounded to active capital"""
        try:
            # ANTI-FRAGILE: Only compound when vault reaches threshold
            vault_threshold = self.active_capital * self.vault_config['compound_threshold']
            
            if self.profit_vault >= vault_threshold:
                # Calculate compound amount (keep emergency reserve)
                emergency_reserve = self.profit_vault * self.vault_config['emergency_reserve']
                available_for_compound = self.profit_vault - emergency_reserve
                compound_amount = available_for_compound * self.vault_config['vault_compound_rate']
                
                # Execute compounding
                if compound_amount > 0:
                    self.active_capital += compound_amount
                    self.profit_vault -= compound_amount
                    self.last_compound_time = datetime.now()
                    
                    self.logger.info(f"🏦 VAULT COMPOUNDING: ${compound_amount:.2f} moved to active capital")
                    self.logger.info(f"💰 New balances - Active: ${self.active_capital:.2f}, Vault: ${self.profit_vault:.2f}")
            
        except Exception as e:
            self.logger.error(f"Vault compounding check failed: {e}")
    
    def get_compounding_metrics(self) -> CompoundingMetrics:
        """Get current compounding performance metrics"""
        try:
            # Calculate performance metrics
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t['profit_loss'] > 0])
            total_profit = sum(t['profit_loss'] for t in self.trade_history)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Calculate profit factor
            profits = sum(t['profit_loss'] for t in self.trade_history if t['profit_loss'] > 0)
            losses = abs(sum(t['profit_loss'] for t in self.trade_history if t['profit_loss'] < 0))
            profit_factor = profits / losses if losses > 0 else 0.0
            
            # Calculate max drawdown
            capital_history = [self.initial_capital]
            running_capital = self.initial_capital
            peak = self.initial_capital
            max_drawdown = 0.0
            
            for trade in self.trade_history:
                running_capital += trade['profit_loss']
                capital_history.append(running_capital)
                if running_capital > peak:
                    peak = running_capital
                drawdown = (peak - running_capital) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate Sharpe ratio (simplified)
            if total_trades > 1:
                returns = [t['profit_loss'] for t in self.trade_history]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            days_in_phase = (datetime.now() - self.phase_start_time).days
            
            return CompoundingMetrics(
                phase=self.current_phase,
                initial_capital=self.initial_capital,
                current_capital=self.active_capital,
                vault_balance=self.profit_vault,
                total_trades=total_trades,
                winning_trades=winning_trades,
                total_profit=total_profit,
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                edge_validated=self.calibration_metrics.edge_validated if self.current_phase == GrowthPhase.CALIBRATION else True,
                days_in_phase=days_in_phase
            )
            
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            return CompoundingMetrics(
                phase=self.current_phase,
                initial_capital=self.initial_capital,
                current_capital=self.active_capital,
                vault_balance=self.profit_vault,
                total_trades=0,
                winning_trades=0,
                total_profit=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                edge_validated=False,
                days_in_phase=0
            )
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get detailed calibration status"""
        if self.current_phase != GrowthPhase.CALIBRATION:
            return {'phase': 'not_in_calibration', 'edge_validated': True}
        
        progress = self.calibration_metrics.trades_completed / self.calibration_metrics.trades_required
        
        return {
            'phase': 'calibration',
            'trades_completed': self.calibration_metrics.trades_completed,
            'trades_required': self.calibration_metrics.trades_required,
            'progress_percent': progress * 100,
            'current_win_rate': self.calibration_metrics.current_win_rate,
            'required_win_rate': self.calibration_metrics.required_win_rate,
            'win_rate_met': self.calibration_metrics.current_win_rate >= self.calibration_metrics.required_win_rate,
            'avg_profit_per_trade': self.calibration_metrics.avg_profit_per_trade,
            'min_profit_required': self.calibration_config['min_profit_per_trade'],
            'profit_met': self.calibration_metrics.avg_profit_per_trade >= self.calibration_config['min_profit_per_trade'],
            'consecutive_losses': self.consecutive_losses,
            'max_allowed_losses': self.calibration_config['max_consecutive_losses'],
            'edge_validated': self.calibration_metrics.edge_validated,
            'days_in_calibration': (datetime.now() - self.calibration_metrics.calibration_start).days
        }
    
    def get_vault_status(self) -> Dict[str, Any]:
        """Get vault status and compounding information"""
        vault_threshold = self.active_capital * self.vault_config['compound_threshold']
        
        return {
            'vault_balance': self.profit_vault,
            'active_capital': self.active_capital,
            'total_capital': self.active_capital + self.profit_vault,
            'vault_threshold': vault_threshold,
            'ready_to_compound': self.profit_vault >= vault_threshold,
            'compound_threshold_percent': self.vault_config['compound_threshold'] * 100,
            'last_compound_time': self.last_compound_time.isoformat(),
            'emergency_reserve_percent': self.vault_config['emergency_reserve'] * 100
        } 