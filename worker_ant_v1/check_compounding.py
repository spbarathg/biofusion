#!/usr/bin/env python3
"""
Compounding Status Checker
==========================

Utility to check compounding engine status and manually trigger cycles.
"""

import asyncio
import sys
import argparse
import logging
from datetime import datetime, timedelta

from worker_ant_v1.compounding_engine import compounding_engine
from worker_ant_v1.ant_manager import ant_manager
from worker_ant_v1.swarm_config import compounding_config

class CompoundingChecker:
    """Compounding engine status and control utility"""
    
    def __init__(self):
        self.logger = logging.getLogger("CompoundingChecker")
    
    async def show_compounding_status(self):
        """Show detailed compounding engine status"""
        
        print("🔄 COMPOUNDING ENGINE STATUS")
        print("=" * 50)
        
        stats = compounding_engine.get_compounding_stats()
        
        print(f"Engine Running: {'✅ Yes' if stats['is_running'] else '❌ No'}")
        print(f"Current Cycle: {stats['current_cycle']}")
        print(f"Total Splits: {stats['total_splits']}")
        print(f"Total Merges: {stats['total_merges']}")
        print(f"Peak Ant Count: {stats['peak_ant_count']}")
        print(f"Current Ants: {stats['current_ants']}")
        print(f"Total Events: {stats['compounding_events']}")
        print(f"Last Cycle: {stats['last_cycle_time']}")
        
        # Calculate next cycle time
        if stats['is_running']:
            last_cycle = datetime.fromisoformat(stats['last_cycle_time'])
            next_cycle = last_cycle + timedelta(hours=compounding_config.compounding_interval_hours)
            time_to_next = next_cycle - datetime.now()
            
            if time_to_next.total_seconds() > 0:
                hours = time_to_next.total_seconds() / 3600
                print(f"Next Cycle In: {hours:.1f} hours ({next_cycle.strftime('%H:%M:%S')})")
            else:
                print("Next Cycle: ⚠️  Overdue!")
        
        print()
        
        # Show swarm summary
        await self._show_swarm_summary()
    
    async def _show_swarm_summary(self):
        """Show current swarm summary"""
        
        print("🐜 CURRENT SWARM STATUS")
        print("-" * 30)
        
        swarm_summary = await ant_manager.get_swarm_summary()
        
        print(f"Active Ants: {swarm_summary['active_ants']}")
        print(f"Total Capital: ${swarm_summary['total_capital_usd']:,.2f}")
        print(f"Total Profit: ${swarm_summary['total_profit_usd']:,.2f}")
        print(f"Avg Win Rate: {swarm_summary['avg_win_rate']:.1f}%")
        print(f"Ants Ready to Split: {swarm_summary['ants_ready_to_split']}")
        print(f"Underperforming Ants: {swarm_summary['underperforming_ants']}")
        
        # Target progress
        target_progress = (swarm_summary['total_capital_usd'] / compounding_config.target_capital_72h) * 100
        print(f"Target Progress: {target_progress:.1f}% (${compounding_config.target_capital_72h:,.0f} goal)")
        print()
    
    async def show_splitting_candidates(self):
        """Show ants eligible for splitting"""
        
        print("🔍 SPLITTING CANDIDATES")
        print("-" * 30)
        
        candidates = await ant_manager.evaluate_splitting_candidates()
        
        if not candidates:
            print("❌ No ants currently eligible for splitting")
            return
        
        print(f"✅ Found {len(candidates)} eligible ants:\n")
        
        for ant in candidates:
            capital_ratio = ant.metrics.current_capital_usd / ant.metrics.starting_capital_usd
            print(f"🐜 {ant.ant_id} ({ant.name})")
            print(f"   Capital: ${ant.metrics.current_capital_usd:,.2f} ({capital_ratio:.1f}x)")
            print(f"   ROI: {ant.metrics.roi_percent:.1f}%")
            print(f"   Win Rate: {ant.metrics.win_rate:.1f}% (Recent: {ant.metrics.recent_win_rate:.1f}%)")
            print(f"   Trades: {ant.metrics.total_trades}")
            print()
    
    async def show_underperformers(self):
        """Show underperforming ants"""
        
        print("⚠️  UNDERPERFORMING ANTS")
        print("-" * 30)
        
        underperformers = await ant_manager.evaluate_underperforming_ants()
        
        if not underperformers:
            print("✅ No underperforming ants found")
            return
        
        print(f"⚠️  Found {len(underperformers)} underperforming ants:\n")
        
        for ant in underperformers:
            print(f"🐜 {ant.ant_id} ({ant.name})")
            print(f"   Status: {ant.status.value}")
            print(f"   Capital: ${ant.metrics.current_capital_usd:,.2f}")
            print(f"   ROI: {ant.metrics.roi_percent:.1f}%")
            print(f"   Win Rate: {ant.metrics.win_rate:.1f}% (Recent: {ant.metrics.recent_win_rate:.1f}%)")
            print(f"   Reason: {ant.pause_reason or 'Performance issues'}")
            print()
    
    async def force_compounding_cycle(self):
        """Manually trigger a compounding cycle"""
        
        print("🔄 Forcing manual compounding cycle...")
        
        if not compounding_engine.is_running:
            print("❌ Compounding engine is not running")
            print("   Start the swarm first to enable compounding")
            return
        
        try:
            # Execute a single compounding cycle
            await compounding_engine._execute_compounding_cycle()
            print("✅ Manual compounding cycle completed successfully")
            
            # Show updated status
            await self.show_compounding_status()
            
        except Exception as e:
            print(f"❌ Manual compounding cycle failed: {e}")
    
    async def show_cycle_history(self, limit: int = 10):
        """Show recent compounding cycle history"""
        
        print(f"📊 RECENT COMPOUNDING HISTORY (Last {limit})")
        print("-" * 50)
        
        # Get recent events from compounding engine
        recent_events = compounding_engine.compounding_history[-limit:]
        
        if not recent_events:
            print("❌ No compounding history available")
            return
        
        for event in recent_events:
            status_icon = "✅" if event.success else "❌"
            print(f"{status_icon} {event.timestamp.strftime('%H:%M:%S')} - {event.event_type.upper()}")
            print(f"   Ant: {event.ant_id}")
            
            if event.details:
                for key, value in event.details.items():
                    if key != 'error':
                        print(f"   {key}: {value}")
            
            if event.error_message:
                print(f"   Error: {event.error_message}")
            
            print()
    
    async def show_target_projection(self):
        """Show projection towards 72-hour target"""
        
        print("🎯 TARGET PROJECTION ANALYSIS")
        print("-" * 40)
        
        swarm_summary = await ant_manager.get_swarm_summary()
        stats = compounding_engine.get_compounding_stats()
        
        current_capital = swarm_summary['total_capital_usd']
        target_capital = compounding_config.target_capital_72h
        current_progress = (current_capital / target_capital) * 100
        
        print(f"Current Capital: ${current_capital:,.2f}")
        print(f"Target Capital: ${target_capital:,.2f}")
        print(f"Progress: {current_progress:.1f}%")
        print(f"Remaining: ${target_capital - current_capital:,.2f}")
        print()
        
        # Calculate required growth rate
        cycles_in_72h = 72 / compounding_config.compounding_interval_hours
        cycles_remaining = cycles_in_72h - stats['current_cycle']
        
        if cycles_remaining > 0:
            required_multiplier = target_capital / current_capital
            required_growth_per_cycle = (required_multiplier ** (1/cycles_remaining) - 1) * 100
            
            print(f"Cycles in 72h: {cycles_in_72h:.0f}")
            print(f"Cycles Completed: {stats['current_cycle']}")
            print(f"Cycles Remaining: {cycles_remaining:.0f}")
            print(f"Required Growth/Cycle: {required_growth_per_cycle:.1f}%")
            print()
            
            # Feasibility assessment
            if required_growth_per_cycle <= 50:  # 50% per cycle is aggressive but possible
                print("✅ Target is FEASIBLE with current performance")
            elif required_growth_per_cycle <= 100:
                print("⚠️  Target is CHALLENGING but possible with optimization")
            else:
                print("❌ Target is UNLIKELY without major performance improvements")
        else:
            print("⚠️  72-hour window has elapsed")

def main():
    """Main function for compounding checker"""
    
    parser = argparse.ArgumentParser(description="Compounding Status Checker")
    parser.add_argument(
        "--action",
        choices=["status", "candidates", "underperformers", "force-cycle", "history", "projection"],
        default="status",
        help="Action to perform (default: status)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit for history display (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    checker = CompoundingChecker()
    
    try:
        if args.action == "status":
            asyncio.run(checker.show_compounding_status())
            
        elif args.action == "candidates":
            asyncio.run(checker.show_splitting_candidates())
            
        elif args.action == "underperformers":
            asyncio.run(checker.show_underperformers())
            
        elif args.action == "force-cycle":
            asyncio.run(checker.force_compounding_cycle())
            
        elif args.action == "history":
            asyncio.run(checker.show_cycle_history(args.limit))
            
        elif args.action == "projection":
            asyncio.run(checker.show_target_projection())
    
    except KeyboardInterrupt:
        print("\n👋 Compounding checker stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 