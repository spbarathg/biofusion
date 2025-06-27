"""
SMART APE MODE COORDINATOR
========================

Master coordinator that orchestrates the entire evolutionary swarm AI system.
Brings together all components: evolution, smart trading, stealth mechanics,
profit discipline, and safety protocols for autonomous operation.
"""

import asyncio
import time
import json
import uuid
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

# Internal imports
from worker_ant_v1.core.simple_config import get_trading_config, get_security_config
from worker_ant_v1.utils.simple_logger import setup_logger as trading_logger
from worker_ant_v1.engines.evolution_engine import EvolutionarySwarmAI, EvolutionPhase
from worker_ant_v1.engines.trading_engine import SmartEntryEngine, DualConfirmation
from worker_ant_v1.engines.stealth_engine import StealthSwarmMechanics
from worker_ant_v1.engines.profit_manager import ProfitDisciplineEngine
from worker_ant_v1.trading.market_scanner import ProductionScanner, TradingOpportunity
from worker_ant_v1.trading.order_buyer import ProductionBuyer, BuySignal, BuyResult
from worker_ant_v1.trading.order_seller import ProductionSeller, Position
from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch

class SystemMode(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    EVOLUTION = "evolution"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"

class OperationalStatus(Enum):
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    EVOLVING = "evolving"
    RESTING = "resting"

@dataclass
class SwarmState:
    """Current state of the swarm system"""
    
    mode: SystemMode = SystemMode.INITIALIZING
    status: OperationalStatus = OperationalStatus.RESTING
    
    # Evolution metrics
    evolution_phase: EvolutionPhase = EvolutionPhase.GENESIS
    generation: int = 1
    total_capital: float = 300.0
    active_ants: int = 0
    
    # Trading metrics
    total_trades: int = 0
    successful_trades: int = 0
    active_positions: int = 0
    win_rate: float = 0.0
    
    # Performance metrics
    hourly_profit: float = 0.0
    daily_profit: float = 0.0
    peak_capital: float = 300.0
    current_drawdown: float = 0.0
    
    # System health
    uptime_hours: float = 0.0
    last_evolution: Optional[datetime] = None
    emergency_stops: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

class SmartApeCoordinator:
    """Master coordinator for Smart Ape Mode"""
    
    def __init__(self):
        self.logger = logging.getLogger("SmartApeCoordinator")
        
        # System state
        self.swarm_state = SwarmState()
        self.start_time = datetime.now()
        self.shutdown_requested = False
        
        # Core components
        self.evolutionary_ai: Optional[EvolutionarySwarmAI] = None
        self.smart_trading: Optional[SmartEntryEngine] = None
        self.stealth_mechanics: Optional[StealthSwarmMechanics] = None
        self.profit_discipline: Optional[ProfitDisciplineEngine] = None
        
        # Trading components
        self.scanner: Optional[ProductionScanner] = None
        self.buyer: Optional[ProductionBuyer] = None
        self.seller: Optional[ProductionSeller] = None
        
        # Safety components
        self.kill_switch: Optional[EnhancedKillSwitch] = None
        
        # Operational parameters
        self.scan_interval_seconds = 30
        self.evolution_check_interval = 3600  # 1 hour
        self.performance_update_interval = 300  # 5 minutes
        
        # Background tasks
        self.running_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.trade_log: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize the complete Smart Ape Mode system"""
        
        self.logger.info("🧠 INITIALIZING SMART APE MODE")
        self.logger.info("🎯 Target: Turn $300 into $10,000+ through evolution")
        
        self.swarm_state.mode = SystemMode.INITIALIZING
        
        # Initialize core components
        await self._initialize_safety_systems()
        await self._initialize_evolutionary_ai()
        await self._initialize_trading_components()
        await self._initialize_stealth_mechanics()
        await self._initialize_profit_discipline()
        
        # Start operational loops
        await self._start_operational_loops()
        
        # System ready
        self.swarm_state.mode = SystemMode.ACTIVE
        self.swarm_state.status = OperationalStatus.SCANNING
        
        self.logger.info("🚀 SMART APE MODE ACTIVATED")
        self.logger.info("🧬 Evolution begins now...")
        
    async def _initialize_safety_systems(self):
        """Initialize emergency safety systems first"""
        
        self.logger.info("🛡️ Initializing safety systems...")
        
        # Initialize kill switch
        self.kill_switch = EnhancedKillSwitch()
        await self.kill_switch.initialize()
        
        # Register emergency callbacks
        self.kill_switch.register_emergency_callback(self._handle_emergency_stop)
        
        self.logger.info("✅ Safety systems online")
        
    async def _initialize_evolutionary_ai(self):
        """Initialize evolutionary AI system"""
        
        self.logger.info("🧬 Initializing evolutionary AI...")
        
        self.evolutionary_ai = EvolutionarySwarmAI()
        await self.evolutionary_ai.initialize()
        
        # Sync state with evolutionary AI
        self.swarm_state.evolution_phase = self.evolutionary_ai.phase
        self.swarm_state.total_capital = self.evolutionary_ai.total_capital
        self.swarm_state.active_ants = len(self.evolutionary_ai.ant_manager.active_ants)
        
        self.logger.info("✅ Evolutionary AI online")
        
    async def _initialize_trading_components(self):
        """Initialize trading components"""
        
        self.logger.info("⚡ Initializing trading engines...")
        
        # Initialize scanner
        self.scanner = ProductionScanner()
        await self.scanner.initialize()
        
        # Initialize buyer
        self.buyer = ProductionBuyer()
        await self.buyer.initialize()
        
        # Initialize seller
        self.seller = ProductionSeller()
        await self.seller.initialize()
        
        # Initialize smart trading engine
        self.smart_trading = SmartEntryEngine(self.evolutionary_ai.signal_trust)
        
        self.logger.info("✅ Trading engines online")
        
    async def _initialize_stealth_mechanics(self):
        """Initialize stealth mechanics"""
        
        self.logger.info("🕵️ Initializing stealth mechanics...")
        
        # Generate initial wallets (in production, create real wallets)
        initial_wallets = []
        for i in range(10):
            wallet_address = f"stealth_wallet_{i}"
            private_key = f"stealth_key_{i}"
            initial_wallets.append((wallet_address, private_key))
        
        self.stealth_mechanics = StealthSwarmMechanics()
        await self.stealth_mechanics.initialize(initial_wallets)
        
        self.logger.info("✅ Stealth mechanics online")
        
    async def _initialize_profit_discipline(self):
        """Initialize profit discipline engine"""
        
        self.logger.info("💰 Initializing profit discipline...")
        
        # Get genetics from genesis ant
        genesis_genetics = None
        if self.evolutionary_ai.ant_manager.active_ants:
            first_ant = next(iter(self.evolutionary_ai.ant_manager.active_ants.values()))
            genesis_genetics = getattr(first_ant.metrics, 'genetics', None)
        
        self.profit_discipline = ProfitDisciplineEngine(genesis_genetics)
        
        self.logger.info("✅ Profit discipline online")
        
    async def _start_operational_loops(self):
        """Start all operational background loops"""
        
        self.logger.info("🔄 Starting operational loops...")
        
        # Main trading loop
        trading_task = asyncio.create_task(self._main_trading_loop())
        self.running_tasks.append(trading_task)
        
        # Position monitoring loop
        monitoring_task = asyncio.create_task(self._position_monitoring_loop())
        self.running_tasks.append(monitoring_task)
        
        # Performance tracking loop
        performance_task = asyncio.create_task(self._performance_tracking_loop())
        self.running_tasks.append(performance_task)
        
        # System health monitoring
        health_task = asyncio.create_task(self._system_health_loop())
        self.running_tasks.append(health_task)
        
        # Status reporting loop
        status_task = asyncio.create_task(self._status_reporting_loop())
        self.running_tasks.append(status_task)
        
        self.logger.info("✅ All operational loops started")
        
    async def _main_trading_loop(self):
        """Main trading decision and execution loop"""
        
        while not self.shutdown_requested:
            try:
                if self.swarm_state.mode != SystemMode.ACTIVE:
                    await asyncio.sleep(30)
                    continue
                
                self.swarm_state.status = OperationalStatus.SCANNING
                
                # Scan for opportunities
                opportunities = await self.scanner.scan_for_opportunities()
                
                if opportunities:
                    self.logger.info(f"🔍 Found {len(opportunities)} opportunities")
                    
                    # Process each opportunity
                    for opportunity in opportunities[:3]:  # Limit to top 3
                        await self._process_trading_opportunity(opportunity)
                
                # Wait before next scan
                await asyncio.sleep(self.scan_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Main trading loop error: {e}")
                await asyncio.sleep(60)
    
    async def _process_trading_opportunity(self, opportunity: TradingOpportunity):
        """Process a single trading opportunity"""
        
        try:
            self.swarm_state.status = OperationalStatus.ANALYZING
            
            # Smart entry analysis with dual confirmation
            buy_signal = await self.smart_trading.evaluate_trading_opportunity(opportunity)
            
            if not buy_signal:
                self.logger.info(f"❌ Opportunity rejected: {opportunity.token_symbol}")
                return
            
            self.swarm_state.status = OperationalStatus.CONFIRMING
            
            # Apply stealth mechanics
            stealth_wallet, delay_ms = await self.stealth_mechanics.execute_stealth_trade(buy_signal)
            
            if not stealth_wallet:
                self.logger.warning(f"⚠️ No stealth wallet available for {opportunity.token_symbol}")
                return
            
            self.swarm_state.status = OperationalStatus.EXECUTING
            
            # Execute the trade
            buy_result = await self._execute_buy_order(buy_signal, stealth_wallet)
            
            if buy_result.success:
                self.logger.info(f"✅ Trade executed: {opportunity.token_symbol}")
                
                # Add to profit discipline monitoring
                position = Position(
                    token_address=opportunity.token_address,
                    token_symbol=opportunity.token_symbol,
                    amount_tokens=buy_result.amount_tokens,
                    entry_price=buy_result.actual_price,
                    timestamp=datetime.now()
                )
                
                await self.profit_discipline.add_position(position, buy_result.actual_price)
                
                # Update swarm state
                self.swarm_state.total_trades += 1
                self.swarm_state.active_positions += 1
                
                # Log trade
                self._log_trade(opportunity, buy_signal, buy_result, True)
                
            else:
                self.logger.warning(f"❌ Trade failed: {opportunity.token_symbol}")
                self._log_trade(opportunity, buy_signal, buy_result, False)
                
        except Exception as e:
            self.logger.error(f"Error processing opportunity {opportunity.token_symbol}: {e}")
    
    async def _execute_buy_order(self, buy_signal: BuySignal, stealth_wallet) -> BuyResult:
        """Execute buy order using the buyer component"""
        
        # In production, this would use the stealth wallet
        # For now, use the main buyer
        return await self.buyer.execute_buy(buy_signal)
    
    async def _position_monitoring_loop(self):
        """Monitor active positions for exit conditions"""
        
        while not self.shutdown_requested:
            try:
                if self.swarm_state.mode != SystemMode.ACTIVE:
                    await asyncio.sleep(30)
                    continue
                
                self.swarm_state.status = OperationalStatus.MONITORING
                
                # Get current market data for all positions
                market_data = await self._get_market_data()
                
                # Monitor positions
                await self.profit_discipline.monitor_positions(market_data)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _performance_tracking_loop(self):
        """Track and update performance metrics"""
        
        while not self.shutdown_requested:
            try:
                # Update swarm state from components
                await self._update_swarm_state()
                
                # Record performance snapshot
                self._record_performance_snapshot()
                
                # Check for evolution triggers
                await self._check_evolution_triggers()
                
                await asyncio.sleep(self.performance_update_interval)
                
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(60)
    
    async def _system_health_loop(self):
        """Monitor system health and trigger safety measures"""
        
        while not self.shutdown_requested:
            try:
                # Check system health
                health_status = await self._check_system_health()
                
                if health_status['critical_issues']:
                    self.logger.critical("🚨 Critical system issues detected")
                    await self._handle_critical_issues(health_status['critical_issues'])
                
                # Update kill switch metrics
                await self._update_kill_switch_metrics()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"System health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _status_reporting_loop(self):
        """Regular status reporting"""
        
        while not self.shutdown_requested:
            try:
                # Log status every 5 minutes
                await asyncio.sleep(300)
                self._log_status_report()
                
            except Exception as e:
                self.logger.error(f"Status reporting error: {e}")
                await asyncio.sleep(300)
    
    def _log_status_report(self):
        """Log comprehensive status report"""
        
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        self.logger.info("📊 SMART APE STATUS REPORT")
        self.logger.info(f"💰 Capital: ${self.swarm_state.total_capital:.2f} (Target: $10,000)")
        self.logger.info(f"🐜 Active Ants: {self.swarm_state.active_ants}")
        self.logger.info(f"📈 Win Rate: {self.swarm_state.win_rate:.1f}%")
        self.logger.info(f"🎯 Active Positions: {self.swarm_state.active_positions}")
        self.logger.info(f"⏱️ Uptime: {uptime_hours:.1f} hours")
        self.logger.info(f"🧬 Evolution Phase: {self.swarm_state.evolution_phase.value}")
        
        # Check progress toward goal
        progress = (self.swarm_state.total_capital / 10000.0) * 100
        self.logger.info(f"🎯 Progress to $10K: {progress:.1f}%")
    
    async def _update_swarm_state(self):
        """Update swarm state from all components"""
        
        if self.evolutionary_ai:
            self.swarm_state.evolution_phase = self.evolutionary_ai.phase
            self.swarm_state.total_capital = self.evolutionary_ai.total_capital
            self.swarm_state.active_ants = len(self.evolutionary_ai.ant_manager.active_ants)
            self.swarm_state.total_trades = self.evolutionary_ai.swarm_performance['total_trades']
            self.swarm_state.successful_trades = self.evolutionary_ai.swarm_performance['successful_trades']
        
        # Calculate win rate
        if self.swarm_state.total_trades > 0:
            self.swarm_state.win_rate = (self.swarm_state.successful_trades / self.swarm_state.total_trades) * 100
        
        # Update position count
        if self.profit_discipline:
            self.swarm_state.active_positions = len(self.profit_discipline.active_positions)
        
        # Update uptime
        self.swarm_state.uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
    
    async def _check_evolution_triggers(self):
        """Check if evolution should be triggered"""
        
        if not self.evolutionary_ai:
            return
        
        # Check if it's time for evolution
        if self.swarm_state.last_evolution:
            hours_since_evolution = (datetime.now() - self.swarm_state.last_evolution).total_seconds() / 3600
            if hours_since_evolution < 3:  # Evolution every 3 hours
                return
        
        # Trigger evolution
        self.swarm_state.mode = SystemMode.EVOLUTION
        self.swarm_state.status = OperationalStatus.EVOLVING
        
        self.logger.info("🧬 Triggering evolution cycle...")
        
        # Evolution is handled by the evolutionary AI automatically
        # Just record the time
        self.swarm_state.last_evolution = datetime.now()
        
        # Return to active mode
        self.swarm_state.mode = SystemMode.ACTIVE
        self.swarm_state.status = OperationalStatus.SCANNING
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        
        health_status = {
            'critical_issues': [],
            'warnings': [],
            'overall_status': 'healthy'
        }
        
        # Check capital preservation
        if self.swarm_state.total_capital < 150:  # Lost 50% of starting capital
            health_status['critical_issues'].append('Major capital loss')
        
        # Check win rate
        if self.swarm_state.total_trades > 20 and self.swarm_state.win_rate < 30:
            health_status['critical_issues'].append('Low win rate')
        
        # Check system responsiveness
        # (Would check actual component health in production)
        
        if health_status['critical_issues']:
            health_status['overall_status'] = 'critical'
        elif health_status['warnings']:
            health_status['overall_status'] = 'warning'
        
        return health_status
    
    async def _handle_critical_issues(self, issues: List[str]):
        """Handle critical system issues"""
        
        self.logger.critical(f"🚨 Handling critical issues: {issues}")
        
        # Switch to emergency mode
        self.swarm_state.mode = SystemMode.EMERGENCY
        
        # Emergency exit all positions
        if self.profit_discipline:
            await self.profit_discipline.emergency_exit_all("Critical system issues")
        
        # Activate kill switch
        if self.kill_switch:
            await self.kill_switch.activate_kill_switch("Critical system issues detected")
        
        self.swarm_state.emergency_stops += 1
    
    async def _update_kill_switch_metrics(self):
        """Update kill switch with current system metrics"""
        
        if not self.kill_switch:
            return
        
        # Update financial metrics
        drawdown = ((self.swarm_state.peak_capital - self.swarm_state.total_capital) / 
                   self.swarm_state.peak_capital * 100) if self.swarm_state.peak_capital > 0 else 0
        
        self.kill_switch.update_financial_metrics(
            profit_loss=self.swarm_state.total_capital - 300.0,
            drawdown_percent=drawdown,
            wallet_balance=self.swarm_state.total_capital,
            daily_loss=self.swarm_state.daily_profit if self.swarm_state.daily_profit < 0 else 0
        )
        
        # Update trading metrics
        failed_trades = self.swarm_state.total_trades - self.swarm_state.successful_trades
        self.kill_switch.update_trading_metrics(
            total_trades=self.swarm_state.total_trades,
            failed_trades=failed_trades,
            consecutive_failures=0  # Would track this properly in production
        )
    
    def _record_performance_snapshot(self):
        """Record performance snapshot for analysis"""
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'total_capital': self.swarm_state.total_capital,
            'win_rate': self.swarm_state.win_rate,
            'active_ants': self.swarm_state.active_ants,
            'active_positions': self.swarm_state.active_positions,
            'evolution_phase': self.swarm_state.evolution_phase.value
        }
        
        self.performance_history.append(snapshot)
        
        # Keep only last 24 hours of data
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.performance_history = [
            s for s in self.performance_history 
            if datetime.fromisoformat(s['timestamp']) > cutoff_time
        ]
    
    def _log_trade(self, opportunity: TradingOpportunity, buy_signal: BuySignal, 
                   buy_result: BuyResult, success: bool):
        """Log trade for analysis"""
        
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'token_symbol': opportunity.token_symbol,
            'token_address': opportunity.token_address,
            'confidence': opportunity.confidence_score,
            'urgency': opportunity.urgency_score,
            'amount_sol': buy_signal.amount_sol,
            'success': success,
            'actual_price': buy_result.actual_price if success else None,
            'slippage': buy_result.actual_slippage_percent if success else None
        }
        
        self.trade_log.append(trade_log)
        
        # Keep only last 1000 trades
        if len(self.trade_log) > 1000:
            self.trade_log = self.trade_log[-1000:]
    
    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data for all tracked tokens"""
        
        # Placeholder - would integrate with real market data
        return {}
    
    async def _handle_emergency_stop(self):
        """Handle emergency stop triggered by kill switch"""
        
        self.logger.critical("🚨 EMERGENCY STOP TRIGGERED BY KILL SWITCH")
        
        self.swarm_state.mode = SystemMode.EMERGENCY
        
        # Emergency shutdown of all components
        await self.emergency_shutdown()
    
    async def emergency_shutdown(self):
        """Emergency shutdown of the entire system"""
        
        self.logger.critical("🚨 EMERGENCY SHUTDOWN INITIATED")
        
        self.shutdown_requested = True
        
        # Emergency exit all positions
        if self.profit_discipline:
            await self.profit_discipline.emergency_exit_all("Emergency shutdown")
        
        # Shutdown all components
        await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown of the Smart Ape system"""
        
        self.logger.info("🛑 Shutting down Smart Ape Mode...")
        
        self.swarm_state.mode = SystemMode.SHUTDOWN
        self.shutdown_requested = True
        
        # Cancel all running tasks
        for task in self.running_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.running_tasks, return_exceptions=True)
        
        # Shutdown components
        if self.evolutionary_ai:
            await self.evolutionary_ai.shutdown()
        
        if self.scanner:
            await self.scanner.shutdown()
        
        if self.buyer:
            await self.buyer.shutdown()
        
        if self.seller:
            await self.seller.shutdown()
        
        if self.kill_switch:
            await self.kill_switch.shutdown()
        
        self.logger.info("✅ Smart Ape Mode shutdown complete")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'swarm_state': self.swarm_state.to_dict(),
            'components_status': {
                'evolutionary_ai': 'online' if self.evolutionary_ai else 'offline',
                'smart_trading': 'online' if self.smart_trading else 'offline',
                'stealth_mechanics': 'online' if self.stealth_mechanics else 'offline',
                'profit_discipline': 'online' if self.profit_discipline else 'offline',
                'scanner': 'online' if self.scanner else 'offline',
                'buyer': 'online' if self.buyer else 'offline',
                'seller': 'online' if self.seller else 'offline',
                'kill_switch': 'online' if self.kill_switch else 'offline'
            },
            'performance_summary': {
                'total_trades': len(self.trade_log),
                'recent_performance': self.performance_history[-10:] if self.performance_history else [],
                'uptime_hours': self.swarm_state.uptime_hours
            }
        }

# Main execution function
async def main():
    """Main execution function for Smart Ape Mode"""
    
    coordinator = SmartApeCoordinator()
    
    try:
        # Initialize and run
        await coordinator.initialize()
        
        # Keep running until shutdown
        while not coordinator.shutdown_requested:
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        coordinator.logger.info("Shutdown requested by user")
    except Exception as e:
        coordinator.logger.error(f"Fatal error: {e}")
    finally:
        await coordinator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

# Export main class
__all__ = ['SmartApeCoordinator', 'SwarmState'] 