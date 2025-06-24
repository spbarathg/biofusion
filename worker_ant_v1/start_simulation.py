#!/usr/bin/env python3
"""
Start Simulation Mode
====================

Simple script to start the trading swarm in simulation mode.
Handles Windows compatibility and provides clean output.
"""

import sys
import os
import asyncio
import platform

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def setup_windows_compatibility():
    """Setup Windows event loop compatibility"""
    if platform.system() == "Windows":
        try:
            # Set Windows event loop policy
            if hasattr(asyncio, 'WindowsProactorEventLoopPolicy'):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                print("Windows event loop compatibility enabled")
        except Exception:
            pass

async def start_simulation():
    """Start the trading simulation"""
    
    print("=" * 50)
    print("HYPER-COMPOUNDING TRADING SWARM")
    print("SIMULATION MODE")
    print("=" * 50)
    
    try:
        # Import required modules with absolute imports
        from worker_ant_v1.config import deployment_config, TradingMode
        from worker_ant_v1.scanner import profit_scanner
        from worker_ant_v1.trader import high_performance_trader, TradeOpportunity
        
        print("✅ Modules loaded successfully")
        
        # Set simulation mode
        deployment_config.trading_mode = TradingMode.SIMULATION
        print("✅ Trading Mode: SIMULATION (Safe)")
        
        # Initialize components
        print("🔍 Initializing scanner...")
        await profit_scanner.start()
        
        print("⚡ Initializing trader...")
        await high_performance_trader.setup()
        
        print("🚀 System ready! Starting simulation...")
        print("Press Ctrl+C to stop")
        
        # Run simulation loop
        cycle = 1
        while True:
            print(f"\n--- 📊 Simulation Cycle {cycle} ---")
            
            # Scan for opportunities (simplified for simulation)
            print("🔍 Scanning for opportunities...")
            
            # Create mock opportunities for simulation
            mock_opportunities = [
                TradeOpportunity(
                    token_symbol="MOCK1",
                    token_address="MockAddress1",
                    price_usd=0.0001,
                    liquidity_usd=25000,
                    volume_24h=50000,
                    price_change_1h=3.5,
                    price_change_24h=15.2,
                    strategy="simulation",
                    entry_signal_strength=0.8
                ),
                TradeOpportunity(
                    token_symbol="MOCK2", 
                    token_address="MockAddress2",
                    price_usd=0.0005,
                    liquidity_usd=45000,
                    volume_24h=125000,
                    price_change_1h=7.2,
                    price_change_24h=25.8,
                    strategy="simulation",
                    entry_signal_strength=0.9
                )
            ]
            
            print(f"📈 Found {len(mock_opportunities)} mock opportunities")
            
            # Execute trades on best opportunities
            if mock_opportunities:
                # Sort by confidence score
                mock_opportunities.sort(key=lambda x: x.confidence_score, reverse=True)
                best_opportunity = mock_opportunities[0]
                
                print(f"🎯 Best opportunity: {best_opportunity.token_symbol} "
                      f"(Confidence: {best_opportunity.confidence_score:.1f}%)")
                
                # Execute simulated trade
                trade_result = await high_performance_trader.execute_profitable_trade(best_opportunity)
                
                if trade_result.success:
                    print(f"✅ Simulated trade successful: {trade_result.signature}")
                    print(f"   Amount: {trade_result.amount_sol:.4f} SOL")
                    print(f"   Tokens: {trade_result.token_amount:.0f}")
                else:
                    print(f"❌ Trade failed: {trade_result.error_message}")
            else:
                print("💤 No trading opportunities found")
            
            # Performance summary
            performance = high_performance_trader.get_performance_summary()
            print(f"\n📊 Performance Summary:")
            print(f"   Total Trades: {performance['total_trades']}")
            print(f"   Win Rate: {performance['win_rate']:.1f}%")
            print(f"   Active Positions: {performance['active_positions']}")
            print(f"   Profit: {performance['total_profit_sol']:.4f} SOL")
            
            cycle += 1
            
            # Wait before next cycle (30 seconds)
            print("⏳ Waiting 30 seconds for next cycle...")
            await asyncio.sleep(30)
    
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")
        
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        try:
            print("🧹 Cleaning up...")
            await high_performance_trader.shutdown()
            await profit_scanner.stop()
            print("✅ Cleanup completed")
        except:
            pass
    
    return True

def main():
    """Main function"""
    
    # Setup Windows compatibility
    setup_windows_compatibility()
    
    # Run simulation
    try:
        asyncio.run(start_simulation())
    except Exception as e:
        print(f"❌ Failed to start simulation: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 