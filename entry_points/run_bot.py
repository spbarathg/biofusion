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

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from tests.bulletproof_testing_suite import BulletproofTestingSuite
from worker_ant_v1.core.system_validator import validate_production_config_sync
from worker_ant_v1.trading.main import HyperIntelligentTradingSwarm


class BotLauncher:
    """Unified bot launcher with mode selection."""

    def __init__(self):
        """Initialize the bot launcher."""
        self.mode = None
        self.capital = 10.0

    async def launch(self, mode: str, capital: float = 10.0) -> int:
        """Launch bot in specified mode.

        Args:
            mode: The operation mode (production, simulation, test)
            capital: Initial capital in SOL

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        self.mode = mode
        self.capital = capital

        print(f"\n🚀 LAUNCHING SMART APE BOT - {mode.upper()} MODE")
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

    async def _launch_production(self) -> int:
        """Launch full production system.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
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

    async def _launch_simulation(self) -> int:
        """Launch simulation mode.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        print("🧪 Starting simulation mode...")

        # Set simulation environment
        os.environ["TRADING_MODE"] = "simulation"

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

    async def _launch_test(self) -> int:
        """Launch test mode.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        print("🧪 Starting test mode...")

        try:
            suite = BulletproofTestingSuite()
            await suite.run_comprehensive_tests()
            return 0
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
    print(f"Capital: {args.capital} SOL")
    print(f"Config: {args.config}")
    print("=" * 40)

    # Launch the bot
    launcher = BotLauncher()

    try:
        exit_code = asyncio.run(launcher.launch(args.mode, args.capital))
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 