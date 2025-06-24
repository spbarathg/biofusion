"""
Queen Bot - Swarm Controller & Global Orchestrator
==================================================

The central command system that orchestrates the entire hyper-compounding swarm.
Manages global state, coordinates between components, and handles safety systems.
"""

import asyncio
import time
import json
import signal
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from worker_ant_v1.swarm_config import (
    worker_ant_config,
    compounding_config,
    swarm_controller_config,
    alert_config,
    SwarmMode,
    get_swarm_status_summary,
    validate_swarm_config
)
from worker_ant_v1.ant_manager import ant_manager, AntStatus
from worker_ant_v1.compounding_engine import compounding_engine
from worker_ant_v1.scanner import profit_scanner
from worker_ant_v1.trader import high_performance_trader

class SwarmState:
    """Global swarm state tracking"""
    
    def __init__(self):
        self.mode: SwarmMode = SwarmMode.GENESIS
        self.total_capital_usd: float = 0.0
        self.total_profit_usd: float = 0.0
        self.active_ants: int = 0
        self.total_trades: int = 0
        self.avg_win_rate: float = 0.0
        self.uptime_hours: float = 0.0
        self.emergency_stop: bool = False
        self.last_update: datetime = datetime.now()

class QueenBot:
    """Central swarm controller and orchestrator"""
    
    def __init__(self):
        self.logger = logging.getLogger("QueenBot")
        self.state = SwarmState()
        self.start_time = datetime.now()
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Safety systems
        self.safety_monitor_active = True
        self.rate_limiter = {"last_trade": 0, "trades_this_second": 0}
        self.drawdown_tracker = {"peak_capital": 0, "current_drawdown": 0}
        
        # Global performance tracking
        self.hourly_reports = []
        self.milestone_alerts_sent = set()
        
        # Component health tracking
        self.component_health = {
            "ant_manager": True,
            "compounding_engine": True,
            "scanner": True,
            "trader": True
        }
    
    async def initialize_swarm(self, genesis_wallet: str, genesis_private_key: str):
        """Initialize the hyper-compounding swarm system"""
        
        self.logger.info("👑 Queen Bot initializing hyper-compounding swarm...")
        
        try:
            # Validate configuration
            validate_swarm_config()
            self.logger.info("✅ Swarm configuration validated")
            
            # Initialize components
            await self._initialize_components()
            
            # Create genesis ant
            genesis_ant = await ant_manager.create_genesis_ant(genesis_wallet, genesis_private_key)
            self.logger.info(f"✅ Genesis ant created: {genesis_ant.ant_id}")
            
            # Set initial state
            self.state.mode = SwarmMode.GENESIS
            self.state.total_capital_usd = worker_ant_config.starting_capital_usd
            self.drawdown_tracker["peak_capital"] = worker_ant_config.starting_capital_usd
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self.logger.info("🚀 Queen Bot initialization complete!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Swarm initialization failed: {e}")
            raise
    
    async def _initialize_components(self):
        """Initialize all swarm components"""
        
        # Initialize scanner
        await profit_scanner.start()
        self.component_health["scanner"] = True
        self.logger.info("✅ Profit scanner initialized")
        
        # Initialize trader
        await high_performance_trader.setup()
        self.component_health["trader"] = True
        self.logger.info("✅ High-performance trader initialized")
        
        # Component health is already tracked by ant_manager and compounding_engine
        self.component_health["ant_manager"] = True
        self.component_health["compounding_engine"] = True
    
    async def start_swarm(self):
        """Start the complete hyper-compounding swarm operation"""
        
        if self.is_running:
            self.logger.warning("Swarm already running")
            return
            
        self.is_running = True
        self.logger.info("🚀 Starting hyper-compounding swarm operation...")
        
        try:
            # Start all concurrent operations
            await asyncio.gather(
                self._swarm_orchestrator(),
                self._safety_monitor(),
                self._performance_tracker(),
                compounding_engine.start_compounding_cycle(),
                return_exceptions=True
            )
            
        except Exception as e:
            self.logger.error(f"Swarm operation error: {e}")
            await self.emergency_shutdown("Critical error in swarm operation")
        finally:
            self.is_running = False
    
    async def _swarm_orchestrator(self):
        """Main orchestration loop that coordinates all swarm activities"""
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Update global state
                await self._update_swarm_state()
                
                # Coordinate ant activities
                await self._coordinate_ant_activities()
                
                # Check mode transitions
                await self._check_mode_transitions()
                
                # Resource management
                await self._manage_resources()
                
                # Health checks
                await self._perform_health_checks()
                
                # Brief pause
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Orchestrator error: {e}")
                await asyncio.sleep(30)  # Recovery delay
    
    async def _coordinate_ant_activities(self):
        """Coordinate activities across all active ants"""
        
        active_ants = [ant for ant in ant_manager.active_ants.values() if ant.is_active]
        
        if not active_ants:
            self.logger.warning("No active ants found")
            return
        
        # Rate limiting check
        if not self._check_rate_limits():
            await asyncio.sleep(1)
            return
        
        # Distribute trading opportunities across ants
        try:
            # Get profitable opportunities
            opportunities = await profit_scanner.scan_for_profitable_opportunities()
            
            if opportunities:
                # Distribute opportunities to available ants
                await self._distribute_opportunities(opportunities, active_ants)
            
        except Exception as e:
            self.logger.error(f"Error coordinating ant activities: {e}")
    
    async def _distribute_opportunities(self, opportunities: List, active_ants: List):
        """Distribute trading opportunities across active ants"""
        
        if not opportunities or not active_ants:
            return
        
        # Sort ants by performance (best performers get first pick)
        sorted_ants = sorted(active_ants, 
                           key=lambda ant: ant.metrics.recent_win_rate, 
                           reverse=True)
        
        # Distribute opportunities
        tasks = []
        for i, opportunity in enumerate(opportunities[:len(sorted_ants)]):
            ant = sorted_ants[i % len(sorted_ants)]
            
            # Check if ant is ready for trading
            if self._is_ant_ready_for_trade(ant):
                task = self._execute_ant_trade(ant, opportunity)
                tasks.append(task)
        
        if tasks:
            # Execute trades concurrently with limits
            await asyncio.gather(*tasks[:swarm_controller_config.max_concurrent_operations], 
                                return_exceptions=True)
    
    def _is_ant_ready_for_trade(self, ant) -> bool:
        """Check if ant is ready for a new trade"""
        
        # Check ant status
        if ant.status not in [AntStatus.ACTIVE, AntStatus.SCALING]:
            return False
        
        # Check if ant has active positions
        if len(ant.active_positions) >= 3:  # Max 3 concurrent positions per ant
            return False
        
        # Check recent trade frequency
        if ant.metrics.last_trade_time:
            time_since_last = (datetime.now() - ant.metrics.last_trade_time).total_seconds()
            min_interval = 3600 / worker_ant_config.trades_per_hour_target  # Seconds per trade
            if time_since_last < min_interval:
                return False
        
        return True
    
    async def _execute_ant_trade(self, ant, opportunity):
        """Execute a trade for a specific ant"""
        
        try:
            self.logger.debug(f"Executing trade for ant {ant.ant_id}")
            
            # Execute the trade
            trade_result = await high_performance_trader.execute_profitable_trade(opportunity)
            
            # Update ant performance
            if trade_result:
                await ant_manager.update_ant_performance(ant.ant_id, {
                    'profit_usd': trade_result.profit_usd if hasattr(trade_result, 'profit_usd') else 0,
                    'fees_usd': trade_result.fees_usd if hasattr(trade_result, 'fees_usd') else 0,
                    'duration_seconds': trade_result.duration_seconds if hasattr(trade_result, 'duration_seconds') else 0,
                    'entry_latency_ms': trade_result.entry_latency_ms if hasattr(trade_result, 'entry_latency_ms') else 0,
                    'slippage_percent': trade_result.slippage_percent if hasattr(trade_result, 'slippage_percent') else 0
                })
                
                self.logger.info(f"Trade completed for ant {ant.ant_id}: "
                               f"Profit ${trade_result.profit_usd if hasattr(trade_result, 'profit_usd') else 0:.2f}")
        
        except Exception as e:
            self.logger.error(f"Trade execution failed for ant {ant.ant_id}: {e}")
    
    def _check_rate_limits(self) -> bool:
        """Check if rate limits allow new trades"""
        
        current_time = time.time()
        
        # Reset counter if new second
        if current_time - self.rate_limiter["last_trade"] >= 1.0:
            self.rate_limiter["trades_this_second"] = 0
            self.rate_limiter["last_trade"] = current_time
        
        # Check rate limit
        if self.rate_limiter["trades_this_second"] >= swarm_controller_config.max_trades_per_second:
            return False
        
        self.rate_limiter["trades_this_second"] += 1
        return True
    
    async def _update_swarm_state(self):
        """Update global swarm state"""
        
        swarm_summary = await ant_manager.get_swarm_summary()
        
        self.state.total_capital_usd = swarm_summary["total_capital_usd"]
        self.state.total_profit_usd = swarm_summary["total_profit_usd"]
        self.state.active_ants = swarm_summary["active_ants"]
        self.state.avg_win_rate = swarm_summary["avg_win_rate"]
        self.state.uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        self.state.last_update = datetime.now()
        
        # Update drawdown tracking
        if self.state.total_capital_usd > self.drawdown_tracker["peak_capital"]:
            self.drawdown_tracker["peak_capital"] = self.state.total_capital_usd
        
        self.drawdown_tracker["current_drawdown"] = (
            (self.drawdown_tracker["peak_capital"] - self.state.total_capital_usd) / 
            self.drawdown_tracker["peak_capital"] * 100
        )
    
    async def _check_mode_transitions(self):
        """Check and handle swarm mode transitions"""
        
        current_capital = self.state.total_capital_usd
        
        # Genesis -> Compounding
        if (self.state.mode == SwarmMode.GENESIS and 
            current_capital >= worker_ant_config.starting_capital_usd * 1.5):
            self.state.mode = SwarmMode.COMPOUNDING
            self.logger.info("🔄 Swarm mode: GENESIS -> COMPOUNDING")
        
        # Compounding -> Aggressive
        elif (self.state.mode == SwarmMode.COMPOUNDING and 
              current_capital >= swarm_controller_config.scale_mode_aggressive_threshold):
            self.state.mode = SwarmMode.AGGRESSIVE
            self.logger.info("🔄 Swarm mode: COMPOUNDING -> AGGRESSIVE")
        
        # Aggressive -> Defensive
        elif (self.state.mode == SwarmMode.AGGRESSIVE and 
              current_capital >= swarm_controller_config.scale_mode_defensive_threshold):
            self.state.mode = SwarmMode.DEFENSIVE
            self.logger.info("🔄 Swarm mode: AGGRESSIVE -> DEFENSIVE")
        
        # Check for milestone alerts
        await self._check_milestone_alerts(current_capital)
    
    async def _check_milestone_alerts(self, current_capital: float):
        """Check and send milestone alerts"""
        
        for milestone in alert_config.milestone_thresholds_usd:
            if (current_capital >= milestone and 
                milestone not in self.milestone_alerts_sent):
                
                self.milestone_alerts_sent.add(milestone)
                await self._send_milestone_alert(milestone)
    
    async def _send_milestone_alert(self, milestone: float):
        """Send milestone achievement alert"""
        
        message = f"🎯 MILESTONE ACHIEVED: ${milestone:,.0f}!"
        progress = (self.state.total_capital_usd / compounding_config.target_capital_72h) * 100
        
        self.logger.info(f"{message} Progress: {progress:.1f}% to 72h target")
    
    async def _safety_monitor(self):
        """Continuous safety monitoring system"""
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Check global drawdown
                if (self.drawdown_tracker["current_drawdown"] >= 
                    swarm_controller_config.global_kill_switch_drawdown):
                    
                    await self.emergency_shutdown(
                        f"Global drawdown exceeded {swarm_controller_config.global_kill_switch_drawdown}%"
                    )
                    break
                
                # Check component health
                await self._monitor_component_health()
                
                # Check resource usage
                await self._monitor_resources()
                
                await asyncio.sleep(swarm_controller_config.health_check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Safety monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_component_health(self):
        """Monitor health of all components"""
        
        # Check if components are responsive
        try:
            # Test ant manager
            summary = await ant_manager.get_swarm_summary()
            self.component_health["ant_manager"] = True
        except:
            self.component_health["ant_manager"] = False
            
        # Test compounding engine
        self.component_health["compounding_engine"] = compounding_engine.is_running
        
        # Count unhealthy components
        unhealthy_count = sum(1 for health in self.component_health.values() if not health)
        
        if unhealthy_count >= 2:
            self.logger.error("Multiple components unhealthy, considering emergency shutdown")
    
    async def _monitor_resources(self):
        """Monitor system resource usage"""
        
        try:
            import psutil
            
            # Memory usage
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                self.logger.warning(f"High memory usage: {memory_percent}%")
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > swarm_controller_config.max_concurrent_operations * 20:
                self.logger.warning(f"High CPU usage: {cpu_percent}%")
                
        except ImportError:
            # psutil not available, skip resource monitoring
            pass
    
    async def _manage_resources(self):
        """Manage swarm resources and optimization"""
        pass  # Placeholder for resource management logic
    
    async def _perform_health_checks(self):
        """Perform comprehensive health checks"""
        pass  # Placeholder for health check logic
    
    async def _performance_tracker(self):
        """Track and log performance metrics"""
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Generate hourly report
                await self._generate_hourly_report()
                
                # Wait for next hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Performance tracker error: {e}")
                await asyncio.sleep(300)
    
    async def _generate_hourly_report(self):
        """Generate comprehensive hourly performance report"""
        
        swarm_summary = await ant_manager.get_swarm_summary()
        compounding_stats = compounding_engine.get_compounding_stats()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": self.state.uptime_hours,
            "swarm_mode": self.state.mode.value,
            "total_capital": self.state.total_capital_usd,
            "total_profit": self.state.total_profit_usd,
            "roi_percent": ((self.state.total_capital_usd - worker_ant_config.starting_capital_usd) / 
                           worker_ant_config.starting_capital_usd) * 100,
            "active_ants": self.state.active_ants,
            "avg_win_rate": self.state.avg_win_rate,
            "compounding_cycles": compounding_stats["current_cycle"],
            "target_progress": (self.state.total_capital_usd / compounding_config.target_capital_72h) * 100,
            "drawdown_percent": self.drawdown_tracker["current_drawdown"],
            "component_health": self.component_health
        }
        
        self.hourly_reports.append(report)
        self.logger.info(f"📊 Hourly Report: {json.dumps(report, indent=2)}")
    
    async def emergency_shutdown(self, reason: str):
        """Emergency shutdown of the entire swarm"""
        
        self.logger.critical(f"🚨 EMERGENCY SHUTDOWN: {reason}")
        
        self.state.emergency_stop = True
        self.is_running = False
        
        # Stop all components
        await compounding_engine.stop_compounding()
        
        # Pause all ants
        for ant in ant_manager.active_ants.values():
            await ant_manager.pause_ant(ant.ant_id, f"Emergency stop: {reason}")
        
        self.shutdown_event.set()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.graceful_shutdown())
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except ValueError:
            # Signals not available (e.g., Windows)
            pass
    
    async def graceful_shutdown(self):
        """Graceful shutdown of the swarm"""
        
        self.logger.info("👑 Queen Bot initiating graceful shutdown...")
        
        self.is_running = False
        
        # Stop compounding engine
        await compounding_engine.stop_compounding()
        
        # Generate final report
        await self._generate_final_report()
        
        self.shutdown_event.set()
        self.logger.info("👑 Queen Bot shutdown complete")
    
    async def _generate_final_report(self):
        """Generate final comprehensive report"""
        
        swarm_summary = await ant_manager.get_swarm_summary()
        compounding_stats = compounding_engine.get_compounding_stats()
        
        final_roi = ((self.state.total_capital_usd - worker_ant_config.starting_capital_usd) / 
                     worker_ant_config.starting_capital_usd) * 100
        
        final_report = {
            "session_start": self.start_time.isoformat(),
            "session_end": datetime.now().isoformat(),
            "total_runtime_hours": self.state.uptime_hours,
            "starting_capital": worker_ant_config.starting_capital_usd,
            "final_capital": self.state.total_capital_usd,
            "total_profit": self.state.total_profit_usd,
            "final_roi_percent": final_roi,
            "target_achievement": (self.state.total_capital_usd / compounding_config.target_capital_72h) * 100,
            "peak_ant_count": compounding_stats["peak_ant_count"],
            "total_splits": compounding_stats["total_splits"],
            "total_merges": compounding_stats["total_merges"],
            "final_mode": self.state.mode.value,
            "emergency_stop": self.state.emergency_stop
        }
        
        self.logger.info(f"📋 FINAL REPORT: {json.dumps(final_report, indent=2)}")
    
    def get_real_time_status(self) -> Dict:
        """Get real-time swarm status"""
        
        return {
            "mode": self.state.mode.value,
            "uptime_hours": self.state.uptime_hours,
            "total_capital": self.state.total_capital_usd,
            "total_profit": self.state.total_profit_usd,
            "active_ants": self.state.active_ants,
            "avg_win_rate": self.state.avg_win_rate,
            "drawdown_percent": self.drawdown_tracker["current_drawdown"],
            "target_progress": (self.state.total_capital_usd / compounding_config.target_capital_72h) * 100,
            "is_running": self.is_running,
            "emergency_stop": self.state.emergency_stop,
            "component_health": self.component_health
        }


# Global Queen Bot instance
queen_bot = QueenBot() 