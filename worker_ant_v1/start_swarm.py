#!/usr/bin/env python3
"""
Hyper-Compounding Swarm Launcher
===============================

Main launcher for the autonomous 3-hour flywheel compounding trading swarm.
Transforms $1,000 into $20,000+ in 72 hours through self-replicating ants.
"""

import asyncio
import sys
import argparse
import logging
from pathlib import Path
import json

from worker_ant_v1.swarm_config import (
    worker_ant_config,
    compounding_config,
    swarm_controller_config,
    load_swarm_env_config,
    validate_swarm_config,
    get_swarm_status_summary
)
from worker_ant_v1.queen_bot import queen_bot
from worker_ant_v1.ant_manager import ant_manager
from worker_ant_v1.compounding_engine import compounding_engine

class SwarmLauncher:
    """Main launcher for the hyper-compounding swarm"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging for the swarm"""
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('swarm.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger("SwarmLauncher")
    
    def display_swarm_banner(self):
        """Display the swarm system banner"""
        
        banner = """
        ╔══════════════════════════════════════════════════════════════╗
        ║               HYPER-COMPOUNDING TRADING SWARM                ║
        ║              3-Hour Flywheel Capital Growth                   ║
        ╠══════════════════════════════════════════════════════════════╣
        ║                                                              ║
        ║  🎯 Target: $1,000 → $20,000+ in 72 hours                   ║
        ║  🐜 Strategy: Autonomous ant replication every 3 hours       ║
        ║  ⚡ Speed: 25 trades/hour per ant, <500ms execution          ║
        ║  📊 Profit: 3% per trade, 65-70% win rate target            ║
        ║  🔄 Compounding: Split ants when capital doubles             ║
        ║                                                              ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        
        print(banner)
        
        # Display current configuration
        config_summary = get_swarm_status_summary()
        print("\n📋 SWARM CONFIGURATION:")
        print(f"  • Genesis Capital: ${config_summary['genesis_capital']:,.0f}")
        print(f"  • Target Capital: ${config_summary['target_72h']:,.0f}")
        print(f"  • Trades per Hour: {config_summary['trades_per_hour']}/ant")
        print(f"  • Profit Target: {config_summary['profit_target']}%")
        print(f"  • Compounding: Every {config_summary['compounding_interval']} hours")
        print(f"  • Max Ants: {config_summary['max_ants']}")
        print(f"  • Safety Limit: {config_summary['safety_drawdown']}% drawdown")
        print()
    
    def validate_requirements(self) -> bool:
        """Validate all requirements before starting"""
        
        self.logger.info("🔍 Validating swarm requirements...")
        
        try:
            # Load environment configuration
            load_swarm_env_config()
            
            # Validate configuration
            validate_swarm_config()
            
            # Check wallet configuration
            if not self._validate_wallet_config():
                return False
            
            # Check API keys (optional but recommended)
            self._check_api_keys()
            
            self.logger.info("✅ All requirements validated")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return False
    
    def _validate_wallet_config(self) -> bool:
        """Validate wallet configuration"""
        
        import os
        
        wallet_key = os.getenv("WALLET_PRIVATE_KEY")
        if not wallet_key:
            self.logger.error("❌ WALLET_PRIVATE_KEY not found in environment")
            self.logger.error("   Please set your Solana wallet private key in .env file")
            return False
        
        # Basic validation (in production, add more comprehensive checks)
        if len(wallet_key) < 32:
            self.logger.error("❌ WALLET_PRIVATE_KEY appears invalid (too short)")
            return False
        
        self.logger.info("✅ Wallet configuration validated")
        return True
    
    def _check_api_keys(self):
        """Check for optional API keys"""
        
        import os
        
        api_keys = {
            "BIRDEYE_API_KEY": "Enhanced token scanning",
            "DISCORD_WEBHOOK_URL": "Swarm alerts and milestones",
            "TELEGRAM_BOT_TOKEN": "Alternative alert system"
        }
        
        for key, description in api_keys.items():
            if os.getenv(key):
                self.logger.info(f"✅ {key} configured - {description}")
            else:
                self.logger.warning(f"⚠️  {key} not set - {description} disabled")
    
    async def start_genesis_mode(self, wallet_private_key: str):
        """Start the swarm in genesis mode with a single ant"""
        
        self.logger.info("🐜 Starting GENESIS mode - Single ant deployment")
        
        try:
            # Generate genesis wallet address (in production, derive from private key)
            genesis_wallet = f"genesis_wallet_{int(asyncio.get_event_loop().time())}"
            
            # Initialize Queen Bot
            await queen_bot.initialize_swarm(genesis_wallet, wallet_private_key)
            
            # Start the swarm
            self.logger.info("🚀 Launching hyper-compounding swarm...")
            await queen_bot.start_swarm()
            
        except KeyboardInterrupt:
            self.logger.info("👑 Graceful shutdown initiated by user")
            await queen_bot.graceful_shutdown()
        except Exception as e:
            self.logger.error(f"❌ Genesis mode failed: {e}")
            await queen_bot.emergency_shutdown(f"Genesis mode error: {e}")
    
    async def start_simulation_mode(self):
        """Start the swarm in simulation mode (no real trades)"""
        
        self.logger.info("🛡️  Starting SIMULATION mode - No real trades")
        
        # Use dummy wallet for simulation
        dummy_wallet = "simulation_wallet_dummy"
        dummy_key = "simulation_key_dummy"
        
        try:
            # Initialize Queen Bot in simulation
            await queen_bot.initialize_swarm(dummy_wallet, dummy_key)
            
            # Start simulation
            self.logger.info("🎮 Launching simulation swarm...")
            await queen_bot.start_swarm()
            
        except KeyboardInterrupt:
            self.logger.info("👑 Simulation stopped by user")
            await queen_bot.graceful_shutdown()
        except Exception as e:
            self.logger.error(f"❌ Simulation failed: {e}")
            await queen_bot.emergency_shutdown(f"Simulation error: {e}")
    
    async def check_swarm_status(self):
        """Check current swarm status"""
        
        self.logger.info("📊 Checking swarm status...")
        
        try:
            # Get status from Queen Bot
            status = queen_bot.get_real_time_status()
            
            print("\n🐜 SWARM STATUS REPORT")
            print("=" * 50)
            print(f"Mode: {status['mode'].upper()}")
            print(f"Uptime: {status['uptime_hours']:.1f} hours")
            print(f"Total Capital: ${status['total_capital']:,.2f}")
            print(f"Total Profit: ${status['total_profit']:,.2f}")
            print(f"Active Ants: {status['active_ants']}")
            print(f"Avg Win Rate: {status['avg_win_rate']:.1f}%")
            print(f"Target Progress: {status['target_progress']:.1f}%")
            print(f"Drawdown: {status['drawdown_percent']:.1f}%")
            print(f"Emergency Stop: {status['emergency_stop']}")
            print()
            
            # Component health
            print("🏥 COMPONENT HEALTH")
            print("-" * 20)
            for component, healthy in status['component_health'].items():
                status_icon = "✅" if healthy else "❌"
                print(f"{status_icon} {component}")
            print()
            
            # Ant details
            swarm_summary = await ant_manager.get_swarm_summary()
            print(f"📈 DETAILED METRICS")
            print("-" * 20)
            print(f"Total Ants Created: {swarm_summary['total_ants_created']}")
            print(f"Total Ants Died: {swarm_summary['total_ants_died']}")
            print(f"Ready to Split: {swarm_summary['ants_ready_to_split']}")
            print(f"Underperforming: {swarm_summary['underperforming_ants']}")
            print()
            
            # Compounding stats
            compounding_stats = compounding_engine.get_compounding_stats()
            print(f"🔄 COMPOUNDING STATS")
            print("-" * 20)
            print(f"Current Cycle: {compounding_stats['current_cycle']}")
            print(f"Total Splits: {compounding_stats['total_splits']}")
            print(f"Total Merges: {compounding_stats['total_merges']}")
            print(f"Peak Ant Count: {compounding_stats['peak_ant_count']}")
            print()
            
        except Exception as e:
            self.logger.error(f"❌ Status check failed: {e}")
    
    def create_env_template(self):
        """Create environment configuration template"""
        
        template = """# ================================================================
# HYPER-COMPOUNDING SWARM CONFIGURATION
# ================================================================
# Copy this to .env and configure for your deployment

# ================================================================
# WALLET CONFIGURATION (REQUIRED)
# ================================================================
# Your Solana wallet private key (Base58 encoded)
# WARNING: Keep this secure! Never commit to version control!
WALLET_PRIVATE_KEY=your_base58_private_key_here

# ================================================================
# CAPITAL & TRADING SETTINGS
# ================================================================
# Starting capital for genesis ant
STARTING_CAPITAL_USD=1000

# Trade size range ($100-150 per trade)
TRADE_SIZE_USD_MIN=100
TRADE_SIZE_USD_MAX=150

# Performance targets
TRADES_PER_HOUR=25
PROFIT_TARGET_PERCENT=3.0

# ================================================================
# COMPOUNDING CONFIGURATION
# ================================================================
# Compounding interval (3 hours for aggressive growth)
COMPOUNDING_INTERVAL_HOURS=3

# Maximum ants in swarm (16 target for 72h goal)
MAX_ANTS_TOTAL=16

# ================================================================
# RPC CONFIGURATION
# ================================================================
# Private RPC URL for best performance (Helius, QuickNode, etc.)
PRIVATE_RPC_URL=https://api.mainnet-beta.solana.com

# Backup RPC URLs (comma-separated)
BACKUP_RPC_URLS=https://solana-api.projectserum.com

# ================================================================
# API KEYS (OPTIONAL - ENHANCED PERFORMANCE)
# ================================================================
# Birdeye API for premium token data
BIRDEYE_API_KEY=your_birdeye_api_key

# DexScreener Pro
DEXSCREENER_PRO=false

# ================================================================
# ALERTS & MONITORING
# ================================================================
# Discord webhook for swarm alerts
DISCORD_WEBHOOK_URL=your_discord_webhook_url

# Telegram notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# ================================================================
# SAFETY SETTINGS
# ================================================================
# Global kill switch (30% max drawdown)
GLOBAL_KILL_SWITCH_DRAWDOWN=30.0

# Rate limiting (2 trades/second max)
MAX_TRADES_PER_SECOND=2.0

# ================================================================
# MODE SELECTION
# ================================================================
# Options: simulation, live
# Start with simulation for testing!
TRADING_MODE=simulation
"""
        
        with open('.env.swarm', 'w') as f:
            f.write(template)
        
        self.logger.info("✅ Environment template created: .env.swarm")
        self.logger.info("📝 Copy to .env and configure your settings")

def main():
    """Main launcher function"""
    
    launcher = SwarmLauncher()
    
    parser = argparse.ArgumentParser(description="Hyper-Compounding Trading Swarm")
    parser.add_argument(
        "--mode",
        choices=["genesis", "simulation", "status", "create-env"],
        default="simulation",
        help="Swarm operation mode (default: simulation)"
    )
    parser.add_argument(
        "--wallet-key",
        help="Wallet private key (or set WALLET_PRIVATE_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Display banner
    launcher.display_swarm_banner()
    
    if args.mode == "create-env":
        launcher.create_env_template()
        return
    
    # Validate requirements (except for status check)
    if args.mode != "status":
        if not launcher.validate_requirements():
            sys.exit(1)
    
    # Get wallet key
    wallet_private_key = args.wallet_key or __import__('os').getenv("WALLET_PRIVATE_KEY")
    
    # Execute based on mode
    try:
        if args.mode == "genesis":
            if not wallet_private_key or wallet_private_key == "your_base58_private_key_here":
                launcher.logger.error("❌ Valid wallet private key required for genesis mode")
                sys.exit(1)
            asyncio.run(launcher.start_genesis_mode(wallet_private_key))
            
        elif args.mode == "simulation":
            asyncio.run(launcher.start_simulation_mode())
            
        elif args.mode == "status":
            asyncio.run(launcher.check_swarm_status())
        
    except KeyboardInterrupt:
        print("\n👋 Swarm launcher stopped")
    except Exception as e:
        launcher.logger.error(f"❌ Launcher error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 