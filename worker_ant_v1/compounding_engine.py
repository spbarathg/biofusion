"""
Compounding Engine - 3-Hour Flywheel System
==========================================

Autonomous compounding system that evaluates and executes ant splitting/merging
every 3 hours to achieve exponential capital growth through swarm replication.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

from worker_ant_v1.swarm_config import (
    worker_ant_config,
    compounding_config, 
    swarm_controller_config,
    alert_config
)
from worker_ant_v1.ant_manager import ant_manager, WorkerAnt, AntStatus, AntStrategy

class CompoundingEvent:
    """Represents a compounding event (split, merge, etc.)"""
    
    def __init__(self, event_type: str, ant_id: str, details: Dict):
        self.event_type = event_type  # 'split', 'merge', 'death', 'birth'
        self.ant_id = ant_id
        self.details = details
        self.timestamp = datetime.now()
        self.success = False
        self.error_message = None

class CompoundingEngine:
    """3-Hour flywheel compounding engine"""
    
    def __init__(self):
        self.logger = logging.getLogger("CompoundingEngine")
        self.last_compounding_time = datetime.now()
        self.compounding_history: List[CompoundingEvent] = []
        self.current_cycle = 0
        self.is_running = False
        
        # Performance tracking
        self.total_splits_executed = 0
        self.total_merges_executed = 0
        self.total_capital_at_start = 0.0
        self.peak_ant_count = 0
        
    async def start_compounding_cycle(self):
        """Start the autonomous 3-hour compounding cycle"""
        
        self.is_running = True
        self.logger.info("🔄 Starting 3-hour compounding flywheel")
        
        while self.is_running:
            try:
                # Wait for next compounding interval
                await self._wait_for_next_cycle()
                
                # Execute compounding logic
                await self._execute_compounding_cycle()
                
                # Update cycle tracking
                self.current_cycle += 1
                self.last_compounding_time = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Compounding cycle error: {e}")
                await asyncio.sleep(300)  # 5 minute recovery delay
    
    async def _wait_for_next_cycle(self):
        """Wait for the next 3-hour compounding interval"""
        
        next_cycle_time = (self.last_compounding_time + 
                          timedelta(hours=compounding_config.compounding_interval_hours))
        
        wait_seconds = (next_cycle_time - datetime.now()).total_seconds()
        
        if wait_seconds > 0:
            self.logger.info(f"⏰ Next compounding cycle in {wait_seconds/3600:.1f} hours")
            await asyncio.sleep(wait_seconds)
    
    async def _execute_compounding_cycle(self):
        """Execute a complete compounding cycle"""
        
        self.logger.info(f"🔄 Executing compounding cycle #{self.current_cycle + 1}")
        
        # Get current swarm state
        swarm_summary = await ant_manager.get_swarm_summary()
        self.logger.info(f"Current swarm: {swarm_summary['active_ants']} ants, "
                        f"${swarm_summary['total_capital_usd']:.0f} capital")
        
        # Step 1: Evaluate splitting candidates
        splitting_candidates = await ant_manager.evaluate_splitting_candidates()
        self.logger.info(f"Found {len(splitting_candidates)} ants ready for splitting")
        
        # Step 2: Execute splits
        split_results = await self._execute_splits(splitting_candidates)
        
        # Step 3: Evaluate underperforming ants
        underperformers = await ant_manager.evaluate_underperforming_ants()
        self.logger.info(f"Found {len(underperformers)} underperforming ants")
        
        # Step 4: Handle underperformers
        merge_results = await self._handle_underperformers(underperformers)
        
        # Step 5: Capital rebalancing
        await self._rebalance_capital()
        
        # Step 6: Strategy optimization
        await self._optimize_strategies()
        
        # Step 7: Send alerts and reports
        await self._send_compounding_report(split_results, merge_results)
        
        # Update tracking
        current_ant_count = len(ant_manager.active_ants)
        if current_ant_count > self.peak_ant_count:
            self.peak_ant_count = current_ant_count
        
        self.logger.info(f"✅ Compounding cycle #{self.current_cycle + 1} completed")
    
    async def _execute_splits(self, splitting_candidates: List[WorkerAnt]) -> List[CompoundingEvent]:
        """Execute ant splitting for successful performers"""
        
        split_results = []
        
        for parent_ant in splitting_candidates:
            try:
                # Check if we're at max ant limit
                if len(ant_manager.active_ants) >= compounding_config.max_ants_total:
                    self.logger.warning(f"Max ant limit reached ({compounding_config.max_ants_total})")
                    break
                
                # Calculate split capital
                parent_capital = parent_ant.metrics.current_capital_usd
                split_capital = parent_capital * compounding_config.split_capital_ratio
                
                # Determine offspring strategy (diversify)
                offspring_strategy = self._select_offspring_strategy(parent_ant)
                
                # Create offspring ant
                offspring_ant = await ant_manager.create_offspring_ant(
                    parent_ant=parent_ant,
                    capital_usd=split_capital,
                    strategy=offspring_strategy
                )
                
                # Update parent capital
                parent_ant.metrics.current_capital_usd = split_capital
                parent_ant.status = AntStatus.ACTIVE  # Reset to active after split
                
                # Record split event
                split_event = CompoundingEvent(
                    event_type="split",
                    ant_id=parent_ant.ant_id,
                    details={
                        "parent_id": parent_ant.ant_id,
                        "offspring_id": offspring_ant.ant_id,
                        "parent_capital": split_capital,
                        "offspring_capital": split_capital,
                        "offspring_strategy": offspring_strategy.value,
                        "parent_generation": parent_ant.metrics.generation,
                        "offspring_generation": offspring_ant.metrics.generation
                    }
                )
                split_event.success = True
                split_results.append(split_event)
                
                self.total_splits_executed += 1
                
                self.logger.info(f"🐜 Ant split successful: {parent_ant.ant_id} -> {offspring_ant.ant_id}")
                
                # Send birth alert
                if alert_config.enable_ant_birth_alerts:
                    await self._send_ant_birth_alert(offspring_ant, parent_ant)
                
            except Exception as e:
                split_event = CompoundingEvent(
                    event_type="split",
                    ant_id=parent_ant.ant_id,
                    details={"error": str(e)}
                )
                split_event.success = False
                split_event.error_message = str(e)
                split_results.append(split_event)
                
                self.logger.error(f"Failed to split ant {parent_ant.ant_id}: {e}")
        
        return split_results
    
    async def _handle_underperformers(self, underperformers: List[WorkerAnt]) -> List[CompoundingEvent]:
        """Handle underperforming ants through merging or termination"""
        
        merge_results = []
        
        for ant in underperformers:
            try:
                if ant.status == AntStatus.DEAD:
                    # Terminate dead ants
                    await ant_manager.kill_ant(ant.ant_id, ant.pause_reason)
                    
                    merge_event = CompoundingEvent(
                        event_type="death",
                        ant_id=ant.ant_id,
                        details={
                            "reason": ant.pause_reason,
                            "final_capital": ant.metrics.current_capital_usd,
                            "roi_percent": ant.metrics.roi_percent,
                            "total_trades": ant.metrics.total_trades
                        }
                    )
                    merge_event.success = True
                    merge_results.append(merge_event)
                    
                    # Send death alert
                    if alert_config.enable_ant_death_alerts:
                        await self._send_ant_death_alert(ant)
                
                elif ant.status == AntStatus.UNDERPERFORMING:
                    # Try to merge with better performing ant or pause
                    merge_candidate = await self._find_merge_candidate(ant)
                    
                    if merge_candidate:
                        await self._merge_ants(ant, merge_candidate)
                        merge_event = CompoundingEvent(
                            event_type="merge",
                            ant_id=ant.ant_id,
                            details={
                                "merged_with": merge_candidate.ant_id,
                                "combined_capital": ant.metrics.current_capital_usd + merge_candidate.metrics.current_capital_usd
                            }
                        )
                        merge_event.success = True
                        merge_results.append(merge_event)
                    else:
                        # Just pause the ant
                        await ant_manager.pause_ant(ant.ant_id, "Underperforming - no merge candidate")
                        merge_event = CompoundingEvent(
                            event_type="pause",
                            ant_id=ant.ant_id,
                            details={"reason": "No merge candidate available"}
                        )
                        merge_event.success = True
                        merge_results.append(merge_event)
                
            except Exception as e:
                merge_event = CompoundingEvent(
                    event_type="merge_error",
                    ant_id=ant.ant_id,
                    details={"error": str(e)}
                )
                merge_event.success = False
                merge_event.error_message = str(e)
                merge_results.append(merge_event)
                
                self.logger.error(f"Failed to handle underperformer {ant.ant_id}: {e}")
        
        return merge_results
    
    async def _find_merge_candidate(self, underperforming_ant: WorkerAnt) -> Optional[WorkerAnt]:
        """Find a suitable ant to merge with the underperformer"""
        
        # Look for ants with similar strategy and good performance
        for ant in ant_manager.active_ants.values():
            if (ant.ant_id != underperforming_ant.ant_id and
                ant.status == AntStatus.ACTIVE and
                ant.strategy == underperforming_ant.strategy and
                ant.metrics.win_rate > 60 and
                ant.metrics.current_capital_usd < worker_ant_config.starting_capital_usd * 1.5):
                return ant
        
        return None
    
    async def _merge_ants(self, weak_ant: WorkerAnt, strong_ant: WorkerAnt):
        """Merge two ants by combining their capital"""
        
        # Combine capital
        combined_capital = weak_ant.metrics.current_capital_usd + strong_ant.metrics.current_capital_usd
        strong_ant.metrics.current_capital_usd = combined_capital
        
        # Update trade size for stronger ant
        strong_ant.current_trade_size_usd = min(
            worker_ant_config.trade_size_usd_max,
            combined_capital * 0.1
        )
        
        # Kill the weak ant
        await ant_manager.kill_ant(weak_ant.ant_id, "Merged with stronger ant")
        
        self.total_merges_executed += 1
        self.logger.info(f"🔗 Merged {weak_ant.ant_id} into {strong_ant.ant_id}, "
                        f"new capital: ${combined_capital:.0f}")
    
    async def _rebalance_capital(self):
        """Rebalance capital across the swarm for optimal performance"""
        
        # Calculate total swarm capital
        total_capital = sum(ant.metrics.current_capital_usd for ant in ant_manager.active_ants.values())
        active_ant_count = len([ant for ant in ant_manager.active_ants.values() if ant.is_active])
        
        if active_ant_count == 0:
            return
        
        # Optimal capital per ant
        optimal_capital_per_ant = total_capital / active_ant_count
        
        # Rebalance if needed
        for ant in ant_manager.active_ants.values():
            if not ant.is_active:
                continue
                
            current_capital = ant.metrics.current_capital_usd
            
            # Adjust trade size based on current capital
            if current_capital > optimal_capital_per_ant * 1.5:
                # Large ant - increase trade size
                ant.current_trade_size_usd = min(
                    worker_ant_config.trade_size_usd_max,
                    current_capital * 0.12
                )
            elif current_capital < optimal_capital_per_ant * 0.7:
                # Small ant - decrease trade size
                ant.current_trade_size_usd = max(
                    worker_ant_config.trade_size_usd_min,
                    current_capital * 0.08
                )
    
    async def _optimize_strategies(self):
        """Optimize ant strategies based on market conditions and performance"""
        
        # Analyze strategy performance
        strategy_performance = {}
        for ant in ant_manager.active_ants.values():
            strategy = ant.strategy.value
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {"total_roi": 0, "count": 0}
            
            strategy_performance[strategy]["total_roi"] += ant.metrics.roi_percent
            strategy_performance[strategy]["count"] += 1
        
        # Calculate average ROI per strategy
        for strategy_data in strategy_performance.values():
            if strategy_data["count"] > 0:
                strategy_data["avg_roi"] = strategy_data["total_roi"] / strategy_data["count"]
        
        # Find best performing strategy
        if strategy_performance:
            best_strategy = max(strategy_performance.keys(), 
                               key=lambda s: strategy_performance[s].get("avg_roi", 0))
            
            self.logger.info(f"Best performing strategy: {best_strategy}")
            
            # Consider strategy adjustments for underperforming ants
            for ant in ant_manager.active_ants.values():
                if (ant.metrics.roi_percent < 0 and 
                    ant.metrics.total_trades > 10 and 
                    ant.strategy.value != best_strategy):
                    
                    # Consider switching strategy (implement strategy switching logic)
                    pass
    
    def _select_offspring_strategy(self, parent_ant: WorkerAnt) -> AntStrategy:
        """Select strategy for offspring ant to diversify the swarm"""
        
        # Count current strategy distribution
        strategy_counts = {}
        for ant in ant_manager.active_ants.values():
            strategy = ant.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # If parent is highly successful, 70% chance to inherit strategy
        if parent_ant.metrics.roi_percent > 50 and parent_ant.metrics.win_rate > 70:
            if hash(parent_ant.ant_id) % 10 < 7:  # 70% chance
                return parent_ant.strategy
        
        # Otherwise, diversify by choosing least common strategy
        least_common_strategy = min(AntStrategy, 
                                   key=lambda s: strategy_counts.get(s.value, 0))
        
        return least_common_strategy
    
    async def _send_compounding_report(self, split_results: List[CompoundingEvent], 
                                     merge_results: List[CompoundingEvent]):
        """Send comprehensive compounding cycle report"""
        
        # Calculate cycle statistics
        successful_splits = len([r for r in split_results if r.success])
        successful_merges = len([r for r in merge_results if r.success])
        
        swarm_summary = await ant_manager.get_swarm_summary()
        
        report = {
            "cycle": self.current_cycle + 1,
            "timestamp": datetime.now().isoformat(),
            "splits_executed": successful_splits,
            "merges_executed": successful_merges,
            "current_ants": swarm_summary["active_ants"],
            "total_capital": swarm_summary["total_capital_usd"],
            "avg_win_rate": swarm_summary["avg_win_rate"],
            "target_progress": (swarm_summary["total_capital_usd"] / compounding_config.target_capital_72h) * 100
        }
        
        self.logger.info(f"📊 Compounding Report: {json.dumps(report, indent=2)}")
        
        # Send milestone alerts
        if swarm_summary["total_capital_usd"] >= 5000 and alert_config.enable_milestone_alerts:
            await self._send_milestone_alert(swarm_summary["total_capital_usd"])
    
    async def _send_ant_birth_alert(self, offspring_ant: WorkerAnt, parent_ant: WorkerAnt):
        """Send alert for new ant birth"""
        if not alert_config.discord_webhook_url:
            return
            
        # Implementation would send Discord webhook
        self.logger.info(f"🎉 ANT BIRTH: {offspring_ant.name} born from {parent_ant.name}")
    
    async def _send_ant_death_alert(self, dead_ant: WorkerAnt):
        """Send alert for ant death"""
        if not alert_config.discord_webhook_url:
            return
            
        # Implementation would send Discord webhook
        self.logger.warning(f"💀 ANT DEATH: {dead_ant.name} terminated - {dead_ant.pause_reason}")
    
    async def _send_milestone_alert(self, current_capital: float):
        """Send alert for capital milestones"""
        if not alert_config.discord_webhook_url:
            return
            
        # Implementation would send Discord webhook
        self.logger.info(f"🎯 MILESTONE: ${current_capital:.0f} reached!")
    
    async def stop_compounding(self):
        """Stop the compounding engine"""
        self.is_running = False
        self.logger.info("🛑 Compounding engine stopped")
    
    def get_compounding_stats(self) -> Dict:
        """Get comprehensive compounding statistics"""
        
        return {
            "current_cycle": self.current_cycle,
            "total_splits": self.total_splits_executed,
            "total_merges": self.total_merges_executed,
            "peak_ant_count": self.peak_ant_count,
            "current_ants": len(ant_manager.active_ants),
            "compounding_events": len(self.compounding_history),
            "last_cycle_time": self.last_compounding_time.isoformat(),
            "is_running": self.is_running
        }


# Global compounding engine instance
compounding_engine = CompoundingEngine() 