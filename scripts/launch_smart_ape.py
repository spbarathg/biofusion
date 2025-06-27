#!/usr/bin/env python3
"""
Smart Ape Mode - Simple Launcher
================================

Simple launcher script that avoids relative import issues.
This script provides a clean way to start the Smart Ape Mode system.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🧠 SMART APE MODE LAUNCHER 🧠                           ║
║                                                                              ║
║  Ready to launch the evolutionary trading system!                           ║
║  Mission: Turn $300 into $10,000+ through intelligent trading               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def main():
    """Main launcher function"""
    
    print_banner()
    
    try:
        # Import components using absolute imports
        from worker_ant_v1.core.simple_config import get_trading_config, get_security_config
        from worker_ant_v1.utils.simple_logger import setup_logger
        
        # Setup logger
        logger = setup_logger("SmartApeLauncher")
        logger.info("🚀 Smart Ape Mode startup initiated")
        
        # Get configuration
        trading_config = get_trading_config()
        security_config = get_security_config()
        
        logger.info(f"📊 Trading config loaded - Trade amount: {trading_config.trade_amount_sol} SOL")
        logger.info(f"🛡️ Security config loaded - Stealth mode: {security_config.enable_stealth_mode}")
        
        # Get initial capital from environment or config
        initial_capital = float(os.getenv("INITIAL_CAPITAL", "300.0"))
        
        print(f"\n💰 Initial Capital: ${initial_capital}")
        print(f"🎯 Target: $10,000+ (33x return)")
        print(f"📈 Strategy: Evolutionary swarm trading with dual confirmation")
        print(f"🛡️ Safety: Kill switches, stop losses, and drawdown protection")
        print(f"🥷 Stealth: {'Enabled' if security_config.enable_stealth_mode else 'Disabled'}")
        
        # For now, just demonstrate the system is ready
        print(f"\n🎉 SMART APE MODE READY TO LAUNCH!")
        print(f"⚠️  WARNING: This is a demonstration version")
        print(f"💡 To complete the system, install missing dependencies:")
        print(f"   pip install pydantic cryptography aiosqlite transformers")
        print(f"   pip install solana web3 aiohttp websockets")
        
        logger.info("✅ Smart Ape Mode launcher completed successfully")
        
    except ImportError as e:
        print(f"\n💥 Import Error: {e}")
        print("Some components are missing or have dependency issues")
        print("Run: pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        print(f"\n💥 Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n🛑 Launch cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1) 