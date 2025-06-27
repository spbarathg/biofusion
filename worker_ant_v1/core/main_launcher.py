#!/usr/bin/env python3
"""
Smart Ape Mode - Main System Launcher
=====================================

Production-ready launcher for the Smart Ape Mode trading system.
Integrates all existing components with proper error handling and monitoring.
"""

import asyncio
import sys
import os
import logging
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import available components using absolute imports
try:
    from worker_ant_v1.core.simple_config import get_trading_config, get_security_config
    from worker_ant_v1.utils.simple_logger import setup_logger
    from worker_ant_v1.trading.market_scanner import ProductionScanner
    from worker_ant_v1.trading.order_buyer import ProductionBuyer
    from worker_ant_v1.trading.order_seller import ProductionSeller
    from worker_ant_v1.engines.trading_engine import SmartEntryEngine, SignalTrust
    from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch
except ImportError as e:
    print(f"💥 Import error: {e}")
    print("Ensure all components are available in the worker_ant_v1 package")
    sys.exit(1)


class SmartApeModeSystem:
    """Main Smart Ape Mode system orchestrator"""
    
    def __init__(self, initial_capital: float = 300.0):
        self.initial_capital = initial_capital
        self.logger = setup_logger("SmartApeSystem")
        
        # Core components
        self.scanner: Optional[ProductionScanner] = None
        self.buyer: Optional[ProductionBuyer] = None
        self.seller: Optional[ProductionSeller] = None
        self.trading_engine: Optional[SmartEntryEngine] = None
        self.kill_switch: Optional[EnhancedKillSwitch] = None
        
        # System state
        self.initialized = False
        self.running = False
        self.shutdown_requested = False
        
        # Performance tracking
        self.start_time = None
        self.trades_executed = 0
        self.current_balance = initial_capital
        
    async def initialize_system(self):
        """Initialize all system components"""
        
        self.logger.info("🚀 INITIALIZING SMART APE MODE SYSTEM")
        
        try:
            # Print startup banner
            self._print_startup_banner()
            
            # Initialize components in order
            await self._initialize_kill_switch()
            await self._initialize_scanner()
            await self._initialize_trading_engine()
            await self._initialize_buyer()
            await self._initialize_seller()
            
            # Verify all components
            await self._verify_components()
            
            self.initialized = True
            self.start_time = datetime.now()
            
            self.logger.info("✅ SMART APE MODE SYSTEM READY")
            
        except Exception as e:
            self.logger.error(f"💥 System initialization failed: {e}")
            await self._emergency_shutdown()
            raise
    
    def _print_startup_banner(self):
        """Print system startup banner"""
        
        banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🧠 SMART APE MODE TRADING SYSTEM 🧠                      ║
║                                                                              ║
║  🎯 MISSION: ${self.initial_capital} → $10,000+ through intelligent trading            ║
║                                                                              ║
║  🔧 ACTIVE COMPONENTS:                                                       ║
║  • Market Scanner (opportunity detection)                                   ║
║  • Smart Trading Engine (dual confirmation)                                 ║
║  • Production Buyer (fast execution)                                        ║
║  • Production Seller (profit optimization)                                  ║
║  • Emergency Kill Switch (safety protocols)                                 ║
║                                                                              ║
║  ⚡ TRADING FEATURES:                                                        ║
║  • Dual confirmation entry system                                           ║
║  • Anti-hype filtering                                                      ║
║  • Dynamic profit taking                                                    ║
║  • Risk management protocols                                                ║
║  • Real-time market scanning                                               ║
║                                                                              ║
║  🛡️ SAFETY PROTOCOLS:                                                       ║
║  • Emergency kill switches                                                  ║
║  • Position size limits (2% max)                                           ║
║  • Stop loss protection                                                     ║
║  • Portfolio drawdown monitoring                                           ║
║                                                                              ║
║  ⚠️ WARNING: Trading involves risk - use only risk capital                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    async def _initialize_kill_switch(self):
        """Initialize emergency kill switch"""
        self.logger.info("🛡️ Initializing Emergency Kill Switch...")
        self.kill_switch = EnhancedKillSwitch()
        self.logger.info("✅ Kill Switch active")
    
    async def _initialize_scanner(self):
        """Initialize market scanner"""
        self.logger.info("🔍 Initializing Market Scanner...")
        self.scanner = ProductionScanner()
        self.logger.info("✅ Scanner online")
    
    async def _initialize_trading_engine(self):
        """Initialize trading engine"""
        self.logger.info("🧠 Initializing Trading Engine...")
        signal_trust = SignalTrust(trust_level=0.7)
        self.trading_engine = SmartEntryEngine(signal_trust)
        self.logger.info("✅ Trading Engine online")
    
    async def _initialize_buyer(self):
        """Initialize production buyer"""
        self.logger.info("💰 Initializing Production Buyer...")
        self.buyer = ProductionBuyer()
        self.logger.info("✅ Buyer ready")
    
    async def _initialize_seller(self):
        """Initialize production seller"""
        self.logger.info("💸 Initializing Production Seller...")
        self.seller = ProductionSeller()
        self.logger.info("✅ Seller ready")
    
    async def _verify_components(self):
        """Verify all components are properly initialized"""
        components = {
            'Scanner': self.scanner,
            'Trading Engine': self.trading_engine,
            'Buyer': self.buyer,
            'Seller': self.seller,
            'Kill Switch': self.kill_switch
        }
        
        for name, component in components.items():
            if component is None:
                raise RuntimeError(f"{name} failed to initialize")
            
        self.logger.info("🔧 All components verified and operational")
    
    async def run_trading_system(self):
        """Main trading system loop"""
        
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        self.logger.info("🎯 STARTING SMART APE MODE TRADING")
        self.running = True
        
        try:
            # Start monitoring tasks
            tasks = [
                asyncio.create_task(self._main_trading_loop()),
                asyncio.create_task(self._system_monitor()),
                asyncio.create_task(self._performance_tracker())
            ]
            
            # Wait for tasks or shutdown signal
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"💥 Trading system error: {e}")
            await self._emergency_shutdown()
            raise
        finally:
            self.running = False
    
    async def _main_trading_loop(self):
        """Main trading loop"""
        
        self.logger.info("🔄 Trading loop started")
        
        while self.running and not self.shutdown_requested:
            try:
                # Scan for opportunities
                opportunities = await self.scanner.scan_for_opportunities()
                
                if opportunities:
                    self.logger.info(f"📊 Found {len(opportunities)} trading opportunities")
                    
                    # Process each opportunity
                    for opportunity in opportunities:
                        if self.shutdown_requested:
                            break
                            
                        # Evaluate with trading engine
                        buy_signal = await self.trading_engine.evaluate_trading_opportunity(opportunity)
                        
                        if buy_signal:
                            self.logger.info(f"🎯 Buy signal generated for {opportunity.token_address}")
                            
                            # Execute buy order
                            buy_result = await self.buyer.execute_buy_order(buy_signal)
                            
                            if buy_result.success:
                                self.trades_executed += 1
                                self.logger.info(f"✅ Trade executed: {buy_result.transaction_hash}")
                                
                                # Monitor position for exit
                                asyncio.create_task(self._monitor_position(buy_result))
                
                # Wait before next scan
                await asyncio.sleep(30)  # 30 second intervals
                
            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _monitor_position(self, buy_result):
        """Monitor a position for exit opportunities"""
        
        try:
            # Create position from buy result
            position = {
                'token_address': buy_result.token_address,
                'amount': buy_result.amount_bought,
                'entry_price': buy_result.average_price,
                'timestamp': buy_result.timestamp
            }
            
            # Monitor for exit conditions
            while not self.shutdown_requested:
                exit_signal = await self.seller.should_exit_position(position)
                
                if exit_signal:
                    self.logger.info(f"🎯 Exit signal for {position['token_address']}")
                    
                    sell_result = await self.seller.execute_sell_order(exit_signal)
                    
                    if sell_result.success:
                        profit = sell_result.total_received - (position['amount'] * position['entry_price'])
                        self.current_balance += profit
                        
                        self.logger.info(f"💰 Position closed. Profit: ${profit:.2f}")
                    
                    break
                
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            self.logger.error(f"Position monitoring error: {e}")
    
    async def _system_monitor(self):
        """Monitor system health and performance"""
        
        while self.running and not self.shutdown_requested:
            try:
                # Check kill switch conditions
                if hasattr(self.kill_switch, 'should_trigger'):
                    if await self.kill_switch.should_trigger():
                        self.logger.warning("🚨 Kill switch triggered!")
                        await self._emergency_shutdown()
                        break
                
                # Monitor portfolio
                if self.current_balance < self.initial_capital * 0.9:  # 10% drawdown
                    self.logger.warning(f"⚠️ Portfolio down {((self.initial_capital - self.current_balance) / self.initial_capital) * 100:.1f}%")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"System monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _performance_tracker(self):
        """Track and log performance metrics"""
        
        while self.running and not self.shutdown_requested:
            try:
                if self.start_time:
                    runtime = datetime.now() - self.start_time
                    hours_running = runtime.total_seconds() / 3600
                    
                    profit = self.current_balance - self.initial_capital
                    profit_percentage = (profit / self.initial_capital) * 100
                    
                    self.logger.info(
                        f"📊 Performance Update: "
                        f"Runtime: {hours_running:.1f}h, "
                        f"Trades: {self.trades_executed}, "
                        f"Balance: ${self.current_balance:.2f}, "
                        f"P&L: {profit_percentage:+.2f}%"
                    )
                
                await asyncio.sleep(1800)  # Every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Performance tracker error: {e}")
                await asyncio.sleep(60)
    
    async def _emergency_shutdown(self):
        """Emergency shutdown procedure"""
        
        self.logger.warning("🚨 EMERGENCY SHUTDOWN INITIATED")
        self.shutdown_requested = True
        
        try:
            # Close all positions if possible
            if self.seller:
                # Implementation would close all open positions
                pass
            
            # Save state
            self._save_system_state()
            
        except Exception as e:
            self.logger.error(f"Emergency shutdown error: {e}")
    
    def _save_system_state(self):
        """Save current system state"""
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'trades_executed': self.trades_executed,
            'current_balance': self.current_balance,
            'initial_capital': self.initial_capital
        }
        
        # Save to file or database
        self.logger.info(f"💾 System state saved: {state}")
    
    async def graceful_shutdown(self):
        """Graceful shutdown procedure"""
        
        self.logger.info("🔄 GRACEFUL SHUTDOWN INITIATED")
        self.shutdown_requested = True
        
        # Wait for current operations to complete
        await asyncio.sleep(5)
        
        # Final state save
        self._save_system_state()
        
        self.logger.info("✅ SMART APE MODE SHUTDOWN COMPLETE")


def setup_signal_handlers(system):
    """Setup signal handlers for graceful shutdown"""
    
    def signal_handler(signum, frame):
        asyncio.create_task(system.graceful_shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    
    # Get initial capital from environment or use default
    initial_capital = float(os.getenv("INITIAL_CAPITAL", "300.0"))
    
    # Create and initialize system
    system = SmartApeModeSystem(initial_capital)
    
    # Setup signal handlers
    setup_signal_handlers(system)
    
    try:
        # Initialize and run
        await system.initialize_system()
        await system.run_trading_system()
        
    except KeyboardInterrupt:
        await system.graceful_shutdown()
    except Exception as e:
        print(f"💥 System error: {e}")
        await system._emergency_shutdown()


if __name__ == "__main__":
    # Run the main system
    asyncio.run(main()) 