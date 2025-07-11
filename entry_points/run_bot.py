"""
SMART APE TRADING BOT - UNIFIED LAUNCHER
======================================

Single entry point for all bot operations.
Choose your deployment mode and let the bot handle the rest.

Usage:
  python run_bot.py --mode production
  python run_bot.py --mode simulation  
  python run_bot.py --mode test
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from worker_ant_v1.trading.main import HyperIntelligentTradingSwarm
from worker_ant_v1.trading.bulletproof_testing_suite import BulletproofTestingSuite
from worker_ant_v1.trading.config_validator import validate_production_config

class BotLauncher:
    """Unified bot launcher with mode selection"""
    
    def __init__(self):
        self.mode = None
        self.capital = 10.0
        
    async def launch(self, mode: str, capital: float = 10.0):
        """Launch bot in specified mode"""
        self.mode = mode
        self.capital = capital
        
        print(f"\n🚀 LAUNCHING SMART APE BOT - {mode.upper()} MODE")
        print("=" * 60)
        
        
        if mode in ['production', 'live']:
            if not self._validate_production_setup():
                return 1
        
        
        if mode in ['production', 'live']:
            return await self._launch_production()
        elif mode == 'simulation':
            return await self._launch_simulation()
        elif mode == 'test':
            return await self._launch_test()
        else:
            print(f"❌ Unknown mode: {mode}")
            return 1
    
    def _validate_production_setup(self) -> bool:
        """Validate production configuration"""
        print("🔍 Validating production configuration...")
        
        
        if not os.path.exists('.env.production'):
            print("❌ Missing .env.production file!")
            print("   Run: cp config/env.template .env.production")
            return False
        
        
        try:
            is_valid = validate_production_config()
            if not is_valid:
                print("❌ Configuration validation failed!")
                print("   Check your .env.production file and ensure all required values are set")
                return False
        except Exception as e:
            print(f"❌ Configuration validation error: {e}")
            return False
        
        print("✅ Production configuration valid")
        return True
    
    async def _launch_production(self):
        """Launch full production system"""
        print("🏭 Starting production trading system...")
        
        swarm = HyperIntelligentTradingSwarm()
        
        try:
            if not await swarm.initialize_all_systems():
                print("❌ System initialization failed")
                return 1
                
            print("✅ All systems operational - starting trading")
            await swarm.run()
            return 0
            
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested")
            await swarm.shutdown()
            return 0
        except Exception as e:
            print(f"❌ Critical error: {e}")
            await swarm.emergency_shutdown()
            return 1
    
    async def _launch_simulation(self):
        """Launch simulation mode"""
        print("🧪 Starting simulation mode...")
        
        
        os.environ['TRADING_MODE'] = 'simulation'
        
        swarm = HyperIntelligentTradingSwarm()
        
        try:
            if not await swarm.initialize_all_systems():
                print("❌ Simulation initialization failed")
                return 1
                
            print("✅ Simulation ready - running paper trading")
            await swarm.run()
            return 0
            
        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped")
            await swarm.shutdown()
            return 0
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            await swarm.emergency_shutdown()
            return 1
    
    async def _launch_test(self):
        """Launch test mode"""
        print("🧪 Starting test mode...")
        
        
        try:
            suite = BulletproofTestingSuite()
            await suite.run_comprehensive_tests()
            return 0
        except Exception as e:
            print(f"❌ Test suite error: {e}")
            return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Smart Ape Trading Bot Launcher')
    
    parser.add_argument('--mode', 
                       choices=['production', 'live', 'simulation', 'test'],
                       default='simulation',
                       help='Bot operation mode (default: simulation)')
    
    parser.add_argument('--capital', 
                       type=float, 
                       default=10.0,
                       help='Initial capital in SOL (default: 10.0)')
    
    parser.add_argument('--config',
                       type=str,
                       default='.env.production',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    
    print("\n🦍 SMART APE TRADING BOT")
    print("=" * 40)
    print(f"Mode: {args.mode.upper()}")
    print(f"Capital: {args.capital} SOL")
    print(f"Config: {args.config}")
    print("=" * 40)
    
    
    launcher = BotLauncher()
    
    try:
        exit_code = asyncio.run(launcher.launch(args.mode, args.capital))
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 