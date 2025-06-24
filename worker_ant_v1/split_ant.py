#!/usr/bin/env python3
"""
Manual Ant Splitting Tool
=========================

Utility for manually triggering ant splits for testing or emergency control.
"""

import asyncio
import sys
import argparse
import logging

from worker_ant_v1.ant_manager import ant_manager, AntStrategy
from worker_ant_v1.compounding_engine import compounding_engine

class AntSplitter:
    """Manual ant splitting utility"""
    
    def __init__(self):
        self.logger = logging.getLogger("AntSplitter")
    
    async def list_splitting_candidates(self):
        """List all ants that are eligible for splitting"""
        
        print("🔍 Scanning for splitting candidates...")
        
        candidates = await ant_manager.evaluate_splitting_candidates()
        
        if not candidates:
            print("❌ No ants currently eligible for splitting")
            return
        
        print(f"\n✅ Found {len(candidates)} splitting candidates:\n")
        
        for ant in candidates:
            print(f"🐜 Ant ID: {ant.ant_id}")
            print(f"   Name: {ant.name}")
            print(f"   Strategy: {ant.strategy.value}")
            print(f"   Current Capital: ${ant.metrics.current_capital_usd:,.2f}")
            print(f"   Starting Capital: ${ant.metrics.starting_capital_usd:,.2f}")
            print(f"   ROI: {ant.metrics.roi_percent:.1f}%")
            print(f"   Win Rate: {ant.metrics.win_rate:.1f}%")
            print(f"   Recent Win Rate: {ant.metrics.recent_win_rate:.1f}%")
            print(f"   Total Trades: {ant.metrics.total_trades}")
            print(f"   Generation: {ant.metrics.generation}")
            print()
    
    async def split_ant_by_id(self, ant_id: str, strategy: str = None):
        """Split a specific ant by ID"""
        
        if ant_id not in ant_manager.active_ants:
            print(f"❌ Ant {ant_id} not found")
            return False
        
        parent_ant = ant_manager.active_ants[ant_id]
        
        # Validate ant is eligible
        if parent_ant.metrics.current_capital_usd < parent_ant.metrics.starting_capital_usd * 2:
            print(f"❌ Ant {ant_id} not eligible - insufficient capital")
            print(f"   Current: ${parent_ant.metrics.current_capital_usd:,.2f}")
            print(f"   Required: ${parent_ant.metrics.starting_capital_usd * 2:,.2f}")
            return False
        
        # Parse strategy
        offspring_strategy = None
        if strategy:
            try:
                offspring_strategy = AntStrategy(strategy.lower())
            except ValueError:
                print(f"❌ Invalid strategy: {strategy}")
                print(f"   Valid strategies: {[s.value for s in AntStrategy]}")
                return False
        
        print(f"🔄 Splitting ant {ant_id}...")
        
        try:
            # Calculate split capital
            split_capital = parent_ant.metrics.current_capital_usd * 0.5
            
            # Create offspring
            offspring_ant = await ant_manager.create_offspring_ant(
                parent_ant=parent_ant,
                capital_usd=split_capital,
                strategy=offspring_strategy
            )
            
            # Update parent capital
            parent_ant.metrics.current_capital_usd = split_capital
            
            print(f"✅ Split successful!")
            print(f"   Parent: {parent_ant.ant_id} (${split_capital:,.2f})")
            print(f"   Offspring: {offspring_ant.ant_id} (${split_capital:,.2f})")
            print(f"   Offspring Strategy: {offspring_ant.strategy.value}")
            print(f"   Offspring Generation: {offspring_ant.metrics.generation}")
            
            return True
            
        except Exception as e:
            print(f"❌ Split failed: {e}")
            return False
    
    async def split_all_eligible(self):
        """Split all eligible ants automatically"""
        
        print("🔄 Auto-splitting all eligible ants...")
        
        candidates = await ant_manager.evaluate_splitting_candidates()
        
        if not candidates:
            print("❌ No eligible ants found")
            return
        
        success_count = 0
        
        for ant in candidates:
            try:
                success = await self.split_ant_by_id(ant.ant_id)
                if success:
                    success_count += 1
                    
            except Exception as e:
                print(f"❌ Failed to split {ant.ant_id}: {e}")
        
        print(f"\n📊 Split Summary:")
        print(f"   Attempted: {len(candidates)}")
        print(f"   Successful: {success_count}")
        print(f"   Failed: {len(candidates) - success_count}")
    
    async def show_ant_details(self, ant_id: str):
        """Show detailed information about a specific ant"""
        
        details = await ant_manager.get_ant_details(ant_id)
        
        if not details:
            print(f"❌ Ant {ant_id} not found")
            return
        
        print(f"\n🐜 ANT DETAILS: {ant_id}")
        print("=" * 50)
        print(f"Name: {details['name']}")
        print(f"Strategy: {details['strategy']}")
        print(f"Status: {details['status']}")
        print(f"Active: {details['is_active']}")
        
        if details['pause_reason']:
            print(f"Pause Reason: {details['pause_reason']}")
        
        print("\n📊 PERFORMANCE METRICS:")
        metrics = details['metrics']
        print(f"Starting Capital: ${metrics['starting_capital_usd']:,.2f}")
        print(f"Current Capital: ${metrics['current_capital_usd']:,.2f}")
        print(f"Total Profit: ${metrics['total_profit_usd']:,.2f}")
        print(f"Net Profit: ${metrics['net_profit_usd']:,.2f}")
        print(f"ROI: {metrics['roi_percent']:.1f}%")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"Recent Win Rate: {metrics['recent_win_rate']:.1f}%")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Active Time: {metrics['active_time_hours']:.1f} hours")
        print(f"Generation: {metrics['generation']}")
        print(f"Split Count: {metrics['split_count']}")
        
        print(f"\n🎯 TRADING STATE:")
        print(f"Active Positions: {details['active_positions']}")
        print(f"Recent Trades: {details['recent_trades']}")

def main():
    """Main function for ant splitting utility"""
    
    parser = argparse.ArgumentParser(description="Manual Ant Splitting Tool")
    parser.add_argument(
        "--action",
        choices=["list", "split", "split-all", "details"],
        required=True,
        help="Action to perform"
    )
    parser.add_argument(
        "--ant-id",
        help="Ant ID for split or details commands"
    )
    parser.add_argument(
        "--strategy",
        choices=["sniper", "confirmation", "dip_buyer", "momentum"],
        help="Strategy for offspring ant (optional)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    splitter = AntSplitter()
    
    try:
        if args.action == "list":
            asyncio.run(splitter.list_splitting_candidates())
            
        elif args.action == "split":
            if not args.ant_id:
                print("❌ --ant-id required for split action")
                sys.exit(1)
            asyncio.run(splitter.split_ant_by_id(args.ant_id, args.strategy))
            
        elif args.action == "split-all":
            asyncio.run(splitter.split_all_eligible())
            
        elif args.action == "details":
            if not args.ant_id:
                print("❌ --ant-id required for details action")
                sys.exit(1)
            asyncio.run(splitter.show_ant_details(args.ant_id))
    
    except KeyboardInterrupt:
        print("\n👋 Ant splitter stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 