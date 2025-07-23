"""
SMART APE TRADING BOT - UNIFIED LAUNCHER
======================================

Single entry point for all bot operations.
Choose your deployment mode, strategy, and let the bot handle the rest.

Usage:
  python run_bot.py --mode production --strategy hyper_intelligent
  python run_bot.py --mode production --strategy hyper_compound
  python run_bot.py --mode simulation --strategy hyper_compound
  python run_bot.py --mode test
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tests.bulletproof_testing_suite import BulletproofTestingSuite
from worker_ant_v1.core.system_validator import validate_production_config_sync
from worker_ant_v1.trading.main import HyperIntelligentTradingSwarm
from worker_ant_v1.trading.hyper_compound_squad import HyperCompoundSwarm
from entry_points.colony_commander import ColonyCommander


class BotLauncher:
    """Unified bot launcher with mode and strategy selection."""

    def __init__(self):
        """Initialize the bot launcher."""
        self.mode = None
        self.strategy = None
        self.capital = 10.0

    async def launch(self, mode: str, strategy: str = "hyper_intelligent", capital: float = 10.0) -> int:
        """Launch bot in specified mode with selected strategy.

        Args:
            mode: The operation mode (production, simulation, test)
            strategy: The trading strategy (hyper_intelligent, hyper_compound)
            capital: Initial capital in SOL

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        self.mode = mode
        self.strategy = strategy
        self.capital = capital

        print(f"\n🚀 LAUNCHING SMART APE BOT - {mode.upper()} MODE")
        print(f"🎯 STRATEGY: {strategy.upper().replace('_', ' ')}")
        print("=" * 60)

        # Validate production setup
        if mode in ["production", "live"]:
            if not self._validate_production_setup():
                return 1

        # Launch appropriate mode
        if mode in ["production", "live"]:
            return await self._launch_production()
        elif mode == "simulation":
            return await self._launch_simulation()
        elif mode == "test":
            return await self._launch_test()
        else:
            print(f"❌ Unknown mode: {mode}")
            return 1

    def _validate_production_setup(self) -> bool:
        """Validate production configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        print("🔍 Validating production configuration...")

        # Check for environment file
        if not os.path.exists(".env.production"):
            print("❌ Missing .env.production file!")
            print("   Run: cp config/env.template .env.production")
            return False

        # Validate configuration
        try:
            is_valid = validate_production_config_sync()
            if not is_valid:
                print("❌ Configuration validation failed!")
                print(
                    "   Check your .env.production file and ensure all required values are set"
                )
                return False
        except Exception as e:
            print(f"❌ Configuration validation error: {e}")
            return False

        print("✅ Production configuration valid")
        return True

    def _create_trading_system(self):
        """Create the appropriate trading system based on strategy.

        Returns:
            Trading system instance
        """
        if self.strategy == "hyper_intelligent":
            return HyperIntelligentTradingSwarm(initial_capital=self.capital)
        elif self.strategy == "hyper_compound":
            return HyperCompoundSwarm()
        elif self.strategy == "colony":
            # Determine HA mode based on mode and environment
            enable_ha = self.mode in ["production", "live"] and not os.getenv("DISABLE_HA", "").lower() == "true"
            return ColonyCommander(enable_ha=enable_ha)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    async def _launch_production(self) -> int:
        """Launch full production system.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        print("🏭 Starting production trading system...")

        # CRITICAL: Run pre-flight validation before any system initialization
        print("🔍 Running pre-flight system validation...")
        try:
            from worker_ant_v1.core.system_validator import get_system_validator
            validator = await get_system_validator()
            validation_result = await validator.run_full_validation()
            
            if not validation_result.passed:
                print("❌ PRE-FLIGHT VALIDATION FAILED")
                print("   System cannot launch with critical issues")
                print("   Review the validation output above and fix all issues")
                print("   Then run the validation again before attempting launch")
                return 1
            
            print("✅ Pre-flight validation passed - proceeding with launch")
            
        except Exception as e:
            print(f"❌ Pre-flight validation error: {e}")
            print("   Cannot launch without successful validation")
            return 1

        system = self._create_trading_system()

        try:
            if hasattr(system, 'initialize_all_systems'):
                if not await system.initialize_all_systems():
                    print("❌ System initialization failed")
                    return 1
            elif hasattr(system, 'initialize'):
                await system.initialize()
            
            print("✅ All systems operational - starting trading")
            
            if hasattr(system, 'run'):
                await system.run()
            elif hasattr(system, 'start_compound_mission'):
                await system.start_compound_mission()
            
            return 0

        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested")
            await system.shutdown()
            return 0
        except Exception as e:
            print(f"❌ Critical error: {e}")
            if hasattr(system, 'emergency_shutdown'):
                await system.emergency_shutdown()
            else:
                await system.shutdown()
            return 1

    async def _launch_simulation(self) -> int:
        """Launch simulation mode.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        print("🧪 Starting simulation mode...")

        # Set simulation environment
        os.environ["TRADING_MODE"] = "simulation"

        system = self._create_trading_system()

        try:
            if hasattr(system, 'initialize_all_systems'):
                if not await system.initialize_all_systems():
                    print("❌ Simulation initialization failed")
                    return 1
            elif hasattr(system, 'initialize'):
                await system.initialize()

            print("✅ Simulation ready - running paper trading")
            
            if hasattr(system, 'run'):
                await system.run()
            elif hasattr(system, 'start_compound_mission'):
                await system.start_compound_mission()
            
            return 0

        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped")
            await system.shutdown()
            return 0
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            if hasattr(system, 'emergency_shutdown'):
                await system.emergency_shutdown()
            else:
                await system.shutdown()
            return 1

    async def _launch_test(self) -> int:
        """Launch test mode.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        print("🧪 Starting test mode...")

        try:
            # Import here to avoid dependency issues at startup
            from tests.bulletproof_testing_suite import BulletproofTestingSuite
            suite = BulletproofTestingSuite()
            await suite.run_comprehensive_tests()
            return 0
        except ImportError as e:
            print(f"❌ Test suite not available: {e}")
            print("   Run: pip install -r requirements.txt")
            return 1
        except Exception as e:
            print(f"❌ Test suite error: {e}")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Smart Ape Trading Bot Launcher")

    parser.add_argument(
        "--mode",
        choices=["production", "live", "simulation", "test"],
        default="simulation",
        help="Bot operation mode (default: simulation)",
    )

    parser.add_argument(
        "--strategy",
        choices=["hyper_intelligent", "hyper_compound", "colony"],
        default="hyper_intelligent",
        help="Trading strategy (default: hyper_intelligent)",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10.0,
        help="Initial capital in SOL (default: 10.0)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=".env.production",
        help="Configuration file path",
    )

    args = parser.parse_args()

    # Display startup information
    print("\n🦍 SMART APE TRADING BOT")
    print("=" * 40)
    print(f"Mode: {args.mode.upper()}")
    print(f"Strategy: {args.strategy.upper().replace('_', ' ')}")
    print(f"Capital: {args.capital} SOL")
    print(f"Config: {args.config}")
    print("=" * 40)

    # Launch the bot
    launcher = BotLauncher()

    try:
        exit_code = asyncio.run(launcher.launch(args.mode, args.strategy, args.capital))
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 