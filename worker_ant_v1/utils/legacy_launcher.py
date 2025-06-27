#!/usr/bin/env python3
"""
SMART APE MODE LAUNCHER
======================

Launch script for the complete evolutionary swarm AI trading system.
Initializes all components and starts autonomous operation to turn $300 into $10,000+
"""

import asyncio
import sys
import os
import signal
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Internal imports - Updated for new structure
try:
    from worker_ant_v1.core.swarm_coordinator import SmartApeCoordinator
    from worker_ant_v1.utils.simple_logger import setup_logger as initialize_logger
except ImportError:
    # Fallback for legacy structure
    from worker_ant_v1.core.swarm_coordinator import SmartApeCoordinator
    from worker_ant_v1.utils.simple_logger import setup_logger as initialize_logger

def setup_logging():
    """Setup comprehensive logging for Smart Ape Mode"""
    
    # Initialize main logger
    logger = initialize_logger()
    
    # Set log levels
    logging.getLogger("EvolutionarySwarmAI").setLevel(logging.INFO)
    logging.getLogger("SmartEntryEngine").setLevel(logging.INFO)
    logging.getLogger("StealthSwarmMechanics").setLevel(logging.INFO)
    logging.getLogger("ProfitDisciplineEngine").setLevel(logging.INFO)
    logging.getLogger("SmartApeCoordinator").setLevel(logging.INFO)
    
    return logger

def print_banner():
    """Print Smart Ape Mode banner"""
    
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🧠 SMART APE MODE - EVOLUTIONARY SWARM AI ACTIVATED 🧠                    ║
║                                                                              ║
║   🎯 MISSION: Turn $300 → $10,000+ through intelligent evolution            ║
║   🧬 STRATEGY: Survival of the fittest + compound growth                    ║
║   ⚡ FEATURES: AI Signals + On-chain Confirmation + Stealth Trading         ║
║                                                                              ║
║   💡 KEY CAPABILITIES:                                                       ║
║   • 10-ant evolutionary swarm with genetic algorithms                       ║
║   • Dual confirmation entry (AI + on-chain signals)                         ║
║   • Dynamic signal trust learning from wins/losses                          ║
║   • Anti-hype filter to avoid manipulation                                  ║
║   • Stealth mechanics with wallet rotation & fake trades                    ║
║   • Fast profit-taking + dynamic stop losses                                ║
║   • Nightly auto-tuning for continuous improvement                          ║
║   • Emergency kill switches for capital protection                          ║
║                                                                              ║
║   ⚠️  DISCLAIMER: This is experimental trading software.                     ║
║      Only use funds you can afford to lose completely.                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_startup_checklist():
    """Print pre-flight checklist"""
    
    checklist = """
🔍 PRE-FLIGHT CHECKLIST:
═══════════════════════

✅ Configuration Files:
   • worker_ant_v1/.env (API keys, RPC URLs)
   • worker_ant_v1/config.py (trading parameters)
   
✅ Capital Requirements:
   • Starting capital: $300 (minimum recommended)
   • Emergency reserve: $50-100 (separate)
   
✅ Network Requirements:
   • Solana RPC access (mainnet-beta)
   • Stable internet connection
   • Low latency preferred (<100ms to RPC)
   
✅ Safety Measures:
   • Kill switch mechanisms enabled
   • Emergency stop protocols active
   • Position size limits enforced
   
✅ Monitoring:
   • Log files: worker_ant_v1/logs/
   • Trade records: automated
   • Performance metrics: real-time
   
🚨 RISK WARNINGS:
   • Cryptocurrency trading is extremely risky
   • Past performance does not guarantee future results
   • Total loss of capital is possible
   • Regulatory risks may apply in your jurisdiction
   
Ready to proceed? Press Ctrl+C to abort, or wait 10 seconds to continue...
    """
    print(checklist)

async def countdown(seconds: int):
    """Countdown timer before launch"""
    
    for i in range(seconds, 0, -1):
        print(f"🚀 Launching in {i} seconds...", end="\r")
        await asyncio.sleep(1)
    print("🚀 LAUNCHING SMART APE MODE!      ")

async def run_smart_ape_mode():
    """Main execution function"""
    
    logger = logging.getLogger("SmartApeLauncher")
    coordinator = None
    
    try:
        logger.info("🚀 Initializing Smart Ape Mode...")
        
        # Create and initialize coordinator
        coordinator = SmartApeCoordinator()
        await coordinator.initialize()
        
        logger.info("✅ Smart Ape Mode fully operational!")
        logger.info("🎯 Beginning autonomous evolution toward $10,000 target...")
        
        # Print operational status
        print_operational_status()
        
        # Keep running until shutdown
        while not coordinator.shutdown_requested:
            # Check system status every minute
            await asyncio.sleep(60)
            
            # Optional: Print periodic status updates
            if hasattr(coordinator, 'swarm_state'):
                capital = coordinator.swarm_state.total_capital
                target_progress = (capital / 10000.0) * 100
                logger.info(f"💰 Capital: ${capital:.2f} | Progress: {target_progress:.1f}%")
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
        print("\n🛑 Graceful shutdown initiated...")
        
    except Exception as e:
        logger.error(f"💥 Fatal error occurred: {e}")
        print(f"\n💥 Fatal error: {e}")
        
    finally:
        if coordinator:
            logger.info("🔄 Shutting down all systems...")
            await coordinator.shutdown()
            logger.info("✅ Shutdown complete")

def print_operational_status():
    """Print operational status information"""
    
    status = """
┌────────────────────────────────────────────────────────────────────────────┐
│                        🎯 SMART APE MODE OPERATIONAL                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  📊 SYSTEM STATUS:                                                         │
│  • Evolutionary AI: ✅ Online                                              │
│  • Smart Trading Engine: ✅ Online                                         │
│  • Stealth Mechanics: ✅ Online                                            │
│  • Profit Discipline: ✅ Online                                            │
│  • Emergency Systems: ✅ Armed                                             │
│                                                                            │
│  🎯 CURRENT MISSION:                                                       │
│  • Phase: Genesis → Growth → Maturation → Selection → Dominance            │
│  • Strategy: Cold survivor, zero emotion, full logic                       │
│  • Approach: Prioritize survival over quick gains                          │
│                                                                            │
│  📈 MONITORING:                                                            │
│  • Real-time performance tracking                                          │
│  • Continuous evolution cycles (every 3 hours)                             │
│  • Dynamic risk adjustment                                                  │
│  • Automatic position management                                           │
│                                                                            │
│  🛡️ SAFETY PROTOCOLS:                                                      │
│  • Multi-layered kill switches active                                      │
│  • Position size limits enforced                                           │
│  • Emergency exit procedures ready                                         │
│  • Capital preservation prioritized                                        │
│                                                                            │
│  ⚡ REAL-TIME CONTROLS:                                                     │
│  • Ctrl+C: Graceful shutdown                                               │
│  • Emergency stop: Automatic on critical conditions                        │
│  • Logs: worker_ant_v1/logs/ directory                                     │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

🧠 The swarm is now evolving... Monitor logs for detailed activity.
    """
    print(status)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
        # The asyncio event loop will handle the KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

async def verify_system_requirements():
    """Verify system requirements before launch"""
    
    logger = logging.getLogger("SystemCheck")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False
    
    # Check required directories
    required_dirs = ["logs", "data", "data/wallets", "data/backups"]
    for dir_name in required_dirs:
        dir_path = Path(f"worker_ant_v1/{dir_name}")
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check for critical configuration
    config_file = Path("worker_ant_v1/.env")
    if not config_file.exists():
        logger.warning("⚠️ .env file not found - using default configuration")
    
    logger.info("✅ System requirements verified")
    return True

async def main():
    """Main launcher function"""
    
    # Setup logging first
    logger = setup_logging()
    
    try:
        # Print banner and checklist
        print_banner()
        print_startup_checklist()
        
        # Wait for user confirmation or timeout
        try:
            await asyncio.wait_for(countdown(10), timeout=10.0)
        except asyncio.TimeoutError:
            pass
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Verify system requirements
        if not await verify_system_requirements():
            logger.error("❌ System requirements not met")
            return 1
        
        # Launch Smart Ape Mode
        await run_smart_ape_mode()
        
        return 0
        
    except Exception as e:
        logger.error(f"💥 Launcher error: {e}")
        print(f"\n💥 Launcher failed: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Launch aborted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Critical launcher error: {e}")
        sys.exit(1) 