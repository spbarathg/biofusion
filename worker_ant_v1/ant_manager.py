"""
Ant Manager - Individual Worker Ant Lifecycle Management
=======================================================

Manages individual Worker Ant instances for the hyper-compounding swarm.
Handles ant creation, monitoring, performance tracking, and lifecycle events.
"""

import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from worker_ant_v1.swarm_config import (
    worker_ant_config, 
    compounding_config,
    trading_optimization_config
)

class AntStatus(Enum):
    GENESIS = "genesis"           # Initial ant
    ACTIVE = "active"             # Actively trading
    SCALING = "scaling"           # Performing well, scaling up
    UNDERPERFORMING = "underperforming"  # Below targets
    PAUSED = "paused"             # Temporarily stopped
    SPLITTING = "splitting"       # In process of splitting
    MERGING = "merging"           # Being merged with another
    DEAD = "dead"                 # Terminated/failed

class AntStrategy(Enum):
    SNIPER = "sniper"             # Ultra-fast entries
    CONFIRMATION = "confirmation" # Wait for confirmation
    DIP_BUYER = "dip_buyer"       # Buy dips/retracements
    MOMENTUM = "momentum"         # Momentum trading

@dataclass
class AntPerformanceMetrics:
    """Performance tracking for individual ants"""
    
    # === TRADING METRICS ===
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # === FINANCIAL METRICS ===
    starting_capital_usd: float = 0.0
    current_capital_usd: float = 0.0
    total_profit_usd: float = 0.0
    total_fees_usd: float = 0.0
    net_profit_usd: float = 0.0
    roi_percent: float = 0.0
    
    # === PERFORMANCE METRICS ===
    avg_trade_profit_percent: float = 0.0
    avg_trade_duration_seconds: float = 0.0
    avg_entry_latency_ms: float = 0.0
    avg_slippage_percent: float = 0.0
    
    # === RECENT PERFORMANCE ===
    recent_win_rate: float = 0.0        # Last 20 trades
    recent_profit_rate: float = 0.0     # Last hour profit rate
    trades_last_hour: int = 0
    
    # === LIFECYCLE METRICS ===
    birth_time: datetime = None
    last_trade_time: datetime = None
    active_time_hours: float = 0.0
    split_count: int = 0
    generation: int = 1                  # Which generation ant

@dataclass
class WorkerAnt:
    """Individual Worker Ant instance"""
    
    # === IDENTITY ===
    ant_id: str
    name: str
    strategy: AntStrategy
    status: AntStatus
    
    # === CONFIGURATION ===
    wallet_address: str
    private_key: str               # Encrypted in production
    starting_capital_usd: float
    current_trade_size_usd: float
    
    # === PERFORMANCE ===
    metrics: AntPerformanceMetrics
    
    # === OPERATIONAL ===
    is_active: bool = True
    last_health_check: datetime = None
    error_count: int = 0
    pause_reason: Optional[str] = None
    
    # === TRADING STATE ===
    active_positions: List[Dict] = None
    pending_trades: List[Dict] = None
    recent_trades: List[Dict] = None
    
    def __post_init__(self):
        if self.active_positions is None:
            self.active_positions = []
        if self.pending_trades is None:
            self.pending_trades = []
        if self.recent_trades is None:
            self.recent_trades = []
        if self.metrics.birth_time is None:
            self.metrics.birth_time = datetime.now()


class AntManager:
    """Manages the lifecycle and performance of individual Worker Ants"""
    
    def __init__(self):
        self.active_ants: Dict[str, WorkerAnt] = {}
        self.ant_history: List[WorkerAnt] = []
        self.performance_tracker = {}
        self.logger = logging.getLogger("AntManager")
        
        # Performance tracking
        self.total_ants_created = 0
        self.total_ants_split = 0
        self.total_ants_died = 0
        
    async def create_genesis_ant(self, wallet_address: str, private_key: str) -> WorkerAnt:
        """Create the initial genesis ant to start the swarm"""
        
        ant_id = f"genesis_{uuid.uuid4().hex[:8]}"
        
        genesis_ant = WorkerAnt(
            ant_id=ant_id,
            name=f"Genesis Ant",
            strategy=AntStrategy.SNIPER,  # Start with sniper strategy
            status=AntStatus.GENESIS,
            wallet_address=wallet_address,
            private_key=private_key,
            starting_capital_usd=worker_ant_config.starting_capital_usd,
            current_trade_size_usd=worker_ant_config.trade_size_usd_min,
            metrics=AntPerformanceMetrics(
                starting_capital_usd=worker_ant_config.starting_capital_usd,
                current_capital_usd=worker_ant_config.starting_capital_usd,
                birth_time=datetime.now(),
                generation=1
            )
        )
        
        self.active_ants[ant_id] = genesis_ant
        self.total_ants_created += 1
        
        self.logger.info(f"🐜 Genesis Ant created: {ant_id} with ${worker_ant_config.starting_capital_usd}")
        
        return genesis_ant
    
    async def create_offspring_ant(self, parent_ant: WorkerAnt, capital_usd: float, 
                                  strategy: AntStrategy = None) -> WorkerAnt:
        """Create a new ant from splitting a successful parent"""
        
        ant_id = f"ant_{uuid.uuid4().hex[:8]}"
        if strategy is None:
            strategy = parent_ant.strategy  # Inherit parent strategy
            
        # Generate new wallet for offspring (in production, create new wallet)
        offspring_wallet = f"offspring_wallet_{ant_id}"
        offspring_key = f"offspring_key_{ant_id}"  # Generate new keypair
        
        offspring_ant = WorkerAnt(
            ant_id=ant_id,
            name=f"Ant Gen{parent_ant.metrics.generation + 1}",
            strategy=strategy,
            status=AntStatus.ACTIVE,
            wallet_address=offspring_wallet,
            private_key=offspring_key,
            starting_capital_usd=capital_usd,
            current_trade_size_usd=min(worker_ant_config.trade_size_usd_min, capital_usd * 0.1),
            metrics=AntPerformanceMetrics(
                starting_capital_usd=capital_usd,
                current_capital_usd=capital_usd,
                birth_time=datetime.now(),
                generation=parent_ant.metrics.generation + 1
            )
        )
        
        self.active_ants[ant_id] = offspring_ant
        self.total_ants_created += 1
        parent_ant.metrics.split_count += 1
        
        self.logger.info(f"🐜 New Ant born: {ant_id} from {parent_ant.ant_id} with ${capital_usd}")
        
        return offspring_ant
    
    async def update_ant_performance(self, ant_id: str, trade_result: Dict):
        """Update ant performance metrics after a trade"""
        
        if ant_id not in self.active_ants:
            return
            
        ant = self.active_ants[ant_id]
        metrics = ant.metrics
        
        # Update trade counts
        metrics.total_trades += 1
        if trade_result['profit_usd'] > 0:
            metrics.winning_trades += 1
        else:
            metrics.losing_trades += 1
            
        # Update financial metrics
        metrics.total_profit_usd += trade_result['profit_usd']
        metrics.total_fees_usd += trade_result.get('fees_usd', 0)
        metrics.net_profit_usd = metrics.total_profit_usd - metrics.total_fees_usd
        metrics.current_capital_usd += trade_result['profit_usd']
        
        # Calculate derived metrics
        metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100
        metrics.roi_percent = ((metrics.current_capital_usd - metrics.starting_capital_usd) / 
                              metrics.starting_capital_usd) * 100
        
        # Update performance metrics
        if 'duration_seconds' in trade_result:
            metrics.avg_trade_duration_seconds = self._update_average(
                metrics.avg_trade_duration_seconds, 
                trade_result['duration_seconds'], 
                metrics.total_trades
            )
            
        if 'entry_latency_ms' in trade_result:
            metrics.avg_entry_latency_ms = self._update_average(
                metrics.avg_entry_latency_ms,
                trade_result['entry_latency_ms'],
                metrics.total_trades
            )
            
        if 'slippage_percent' in trade_result:
            metrics.avg_slippage_percent = self._update_average(
                metrics.avg_slippage_percent,
                trade_result['slippage_percent'],
                metrics.total_trades
            )
        
        # Update recent performance (last 20 trades)
        ant.recent_trades.append(trade_result)
        if len(ant.recent_trades) > 20:
            ant.recent_trades.pop(0)
            
        metrics.recent_win_rate = self._calculate_recent_win_rate(ant.recent_trades)
        metrics.last_trade_time = datetime.now()
        
        # Update active time
        if metrics.birth_time:
            metrics.active_time_hours = (datetime.now() - metrics.birth_time).total_seconds() / 3600
        
        # Check for status updates
        await self._evaluate_ant_status(ant)
        
        self.logger.debug(f"Updated metrics for {ant_id}: Win Rate {metrics.win_rate:.1f}%, "
                         f"ROI {metrics.roi_percent:.1f}%, Capital ${metrics.current_capital_usd:.0f}")
    
    async def evaluate_splitting_candidates(self) -> List[WorkerAnt]:
        """Identify ants ready for splitting based on performance"""
        
        candidates = []
        
        for ant in self.active_ants.values():
            if ant.status not in [AntStatus.ACTIVE, AntStatus.SCALING]:
                continue
                
            # Check if ant has doubled its capital
            if (ant.metrics.current_capital_usd >= 
                ant.metrics.starting_capital_usd * compounding_config.split_threshold_multiplier):
                
                # Additional performance checks
                if (ant.metrics.win_rate >= worker_ant_config.target_win_rate and
                    ant.metrics.total_trades >= 10 and  # Minimum trade history
                    ant.metrics.recent_win_rate >= 50.0):  # Recent performance check
                    
                    candidates.append(ant)
                    ant.status = AntStatus.SPLITTING
        
        return candidates
    
    async def evaluate_underperforming_ants(self) -> List[WorkerAnt]:
        """Identify ants that need to be paused or merged"""
        
        underperformers = []
        
        for ant in self.active_ants.values():
            if ant.status == AntStatus.DEAD:
                continue
                
            # Check for death conditions
            if (ant.metrics.current_capital_usd <= 
                ant.metrics.starting_capital_usd * compounding_config.ant_death_threshold):
                
                ant.status = AntStatus.DEAD
                ant.pause_reason = "Capital below death threshold"
                underperformers.append(ant)
                continue
            
            # Check for underperformance
            if (ant.metrics.total_trades >= 20 and
                ant.metrics.recent_win_rate < worker_ant_config.min_win_rate_pause):
                
                ant.status = AntStatus.UNDERPERFORMING
                ant.pause_reason = "Low win rate"
                underperformers.append(ant)
                continue
                
            # Check for excessive losses
            if (ant.metrics.current_capital_usd <= 
                ant.metrics.starting_capital_usd * compounding_config.merge_threshold_multiplier):
                
                ant.status = AntStatus.UNDERPERFORMING
                ant.pause_reason = "Capital below merge threshold"
                underperformers.append(ant)
        
        return underperformers
    
    async def pause_ant(self, ant_id: str, reason: str):
        """Pause an ant's trading activities"""
        
        if ant_id in self.active_ants:
            ant = self.active_ants[ant_id]
            ant.status = AntStatus.PAUSED
            ant.pause_reason = reason
            ant.is_active = False
            
            self.logger.warning(f"🛑 Ant {ant_id} paused: {reason}")
    
    async def kill_ant(self, ant_id: str, reason: str):
        """Terminate an ant and move to history"""
        
        if ant_id in self.active_ants:
            ant = self.active_ants[ant_id]
            ant.status = AntStatus.DEAD
            ant.pause_reason = reason
            ant.is_active = False
            
            # Move to history
            self.ant_history.append(ant)
            del self.active_ants[ant_id]
            self.total_ants_died += 1
            
            self.logger.warning(f"💀 Ant {ant_id} terminated: {reason}")
    
    async def get_swarm_summary(self) -> Dict:
        """Get comprehensive swarm performance summary"""
        
        active_count = len([ant for ant in self.active_ants.values() if ant.is_active])
        total_capital = sum(ant.metrics.current_capital_usd for ant in self.active_ants.values())
        total_profit = sum(ant.metrics.net_profit_usd for ant in self.active_ants.values())
        avg_win_rate = sum(ant.metrics.win_rate for ant in self.active_ants.values()) / max(len(self.active_ants), 1)
        
        return {
            "active_ants": active_count,
            "total_ants_created": self.total_ants_created,
            "total_ants_died": self.total_ants_died,
            "total_capital_usd": total_capital,
            "total_profit_usd": total_profit,
            "avg_win_rate": avg_win_rate,
            "ants_ready_to_split": len(await self.evaluate_splitting_candidates()),
            "underperforming_ants": len(await self.evaluate_underperforming_ants())
        }
    
    async def get_ant_details(self, ant_id: str) -> Optional[Dict]:
        """Get detailed information about a specific ant"""
        
        if ant_id not in self.active_ants:
            return None
            
        ant = self.active_ants[ant_id]
        return {
            "ant_id": ant.ant_id,
            "name": ant.name,
            "strategy": ant.strategy.value,
            "status": ant.status.value,
            "metrics": asdict(ant.metrics),
            "is_active": ant.is_active,
            "pause_reason": ant.pause_reason,
            "active_positions": len(ant.active_positions),
            "recent_trades": len(ant.recent_trades)
        }
    
    def _update_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update running average with new value"""
        if count == 1:
            return new_value
        return ((current_avg * (count - 1)) + new_value) / count
    
    def _calculate_recent_win_rate(self, recent_trades: List[Dict]) -> float:
        """Calculate win rate for recent trades"""
        if not recent_trades:
            return 0.0
            
        wins = sum(1 for trade in recent_trades if trade.get('profit_usd', 0) > 0)
        return (wins / len(recent_trades)) * 100
    
    async def _evaluate_ant_status(self, ant: WorkerAnt):
        """Evaluate and update ant status based on performance"""
        
        metrics = ant.metrics
        
        # Scaling status
        if (metrics.recent_win_rate > 75 and 
            metrics.win_rate > worker_ant_config.target_win_rate):
            ant.status = AntStatus.SCALING
            
        # Active status
        elif (metrics.recent_win_rate >= 50 and 
              metrics.win_rate >= 40):
            ant.status = AntStatus.ACTIVE
            
        # Underperforming status
        elif (metrics.recent_win_rate < worker_ant_config.min_win_rate_pause or
              metrics.win_rate < 30):
            ant.status = AntStatus.UNDERPERFORMING


# Global ant manager instance
ant_manager = AntManager() 