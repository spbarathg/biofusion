"""
HYPER-INTELLIGENT TRADING SWARM - SWARM COORDINATOR
==================================================

Advanced swarm coordinator that manages multiple SimplifiedTradingBot instances
with collective intelligence, cross-instance learning, and adaptive strategies.
"""

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import statistics

from worker_ant_v1.trading.simplified_trading_bot import SimplifiedTradingBot, SimplifiedConfig
from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch, KillSwitchTrigger


class SwarmMode(Enum):
    """Swarm operation modes"""
    LEGACY = "legacy"              # Single bot for backward compatibility
    COORDINATED = "coordinated"    # Multiple bots with coordination
    COMPETITIVE = "competitive"    # Multiple bots competing
    COLLECTIVE = "collective"      # Full swarm intelligence


@dataclass
class SwarmMetrics:
    """Aggregate swarm performance metrics"""
    total_bots: int = 0
    active_bots: int = 0
    total_capital_sol: float = 0.0
    total_profit_sol: float = 0.0
    aggregate_win_rate: float = 0.0
    best_performer_id: Optional[str] = None
    worst_performer_id: Optional[str] = None
    collective_confidence: float = 0.0
    swarm_cohesion: float = 0.0
    adaptation_cycles: int = 0


@dataclass
class SwarmBotInstance:
    """Individual bot instance in the swarm"""
    bot_id: str
    bot: SimplifiedTradingBot
    config: SimplifiedConfig
    specialization: str = "generalist"
    performance_weight: float = 1.0
    last_adaptation: datetime = None
    contribution_score: float = 0.0
    is_leader: bool = False


class HyperIntelligentTradingSwarm:
    """
    Advanced Trading Swarm with Collective Intelligence
    
    Features:
    - Multiple coordinated trading bots
    - Cross-instance learning and adaptation
    - Dynamic strategy evolution
    - Collective decision making
    - Performance-based specialization
    - Backward compatibility with legacy systems
    """
    
    def __init__(self, config_file: str = None, initial_capital: float = 300.0, 
                 swarm_mode: SwarmMode = SwarmMode.LEGACY, swarm_size: int = 1):
        """Initialize the trading swarm."""
        self.logger = get_logger("HyperIntelligentTradingSwarm")
        
        # Swarm configuration
        self.swarm_mode = swarm_mode
        self.swarm_size = max(1, swarm_size)  # At least 1 bot
        self.initial_capital = initial_capital
        
        # Swarm instances
        self.bot_instances: Dict[str, SwarmBotInstance] = {}
        self.swarm_metrics = SwarmMetrics()
        
        # Swarm intelligence
        self.collective_memory: Dict[str, Any] = {}
        self.strategy_evolution_history: List[Dict[str, Any]] = []
        self.cross_bot_signals: Dict[str, Any] = {}
        
        # System state
        self.initialized = False
        self.running = False
        self.swarm_kill_switch = EnhancedKillSwitch()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        self.logger.info(f"🤖 Swarm initialized: Mode={swarm_mode.value}, Size={self.swarm_size}")
    
    async def initialize_all_systems(self) -> bool:
        """Initialize all swarm systems"""
        return await self.initialize()
    
    async def initialize(self) -> bool:
        """Initialize the trading swarm"""
        try:
            self.logger.info("🚀 Initializing Hyper-Intelligent Trading Swarm...")
            
            # Initialize kill switch
            if not self.swarm_kill_switch.is_triggered:
                self.logger.info("✅ Swarm kill switch initialized")
            
            # Create bot instances based on swarm mode
            await self._create_swarm_instances()
            
            # Initialize all bot instances
            initialization_results = []
            for bot_id, instance in self.bot_instances.items():
                try:
                    result = await instance.bot.initialize()
                    initialization_results.append(result)
                    
                    if result:
                        self.logger.info(f"✅ Bot {bot_id} initialized successfully")
                    else:
                        self.logger.error(f"❌ Bot {bot_id} initialization failed")
                        
                except Exception as e:
                    self.logger.error(f"❌ Bot {bot_id} initialization error: {e}")
                    initialization_results.append(False)
            
            # Check if at least one bot initialized successfully
            successful_bots = sum(initialization_results)
            if successful_bots == 0:
                self.logger.error("❌ No bots initialized successfully")
                return False
            
            # Update swarm metrics
            self.swarm_metrics.total_bots = len(self.bot_instances)
            self.swarm_metrics.active_bots = successful_bots
            
            # Start swarm coordination systems
            if self.swarm_mode != SwarmMode.LEGACY:
                await self._start_swarm_coordination()
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.initialized = True
            self.logger.info(f"✅ Swarm initialized: {successful_bots}/{len(self.bot_instances)} bots active")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Swarm initialization failed: {e}")
            return False
    
    async def _create_swarm_instances(self):
        """Create bot instances based on swarm configuration"""
        capital_per_bot = self.initial_capital / self.swarm_size
        capital_per_bot_sol = capital_per_bot / 200.0  # Assuming $200/SOL
        
        for i in range(self.swarm_size):
            bot_id = f"swarm_bot_{i:02d}"
            
            # Create specialized configurations for different bot roles
            if self.swarm_mode == SwarmMode.LEGACY:
                specialization = "generalist"
                config = SimplifiedConfig(initial_capital_sol=capital_per_bot_sol)
            else:
                specialization, config = self._create_specialized_config(i, capital_per_bot_sol)
            
            # Create the trading bot
            bot = SimplifiedTradingBot(config)
            
            # Create swarm instance
            instance = SwarmBotInstance(
                bot_id=bot_id,
                bot=bot,
                config=config,
                specialization=specialization,
                last_adaptation=datetime.now()
            )
            
            self.bot_instances[bot_id] = instance
            self.logger.info(f"🤖 Created bot {bot_id} with specialization: {specialization}")
    
    def _create_specialized_config(self, bot_index: int, capital_sol: float) -> tuple[str, SimplifiedConfig]:
        """Create specialized configuration for different bot roles"""
        # Define specializations based on bot index
        specializations = [
            ("sniper", "aggressive_fast"),     # Quick entries, high frequency
            ("accumulator", "patient_steady"), # Build positions over time  
            ("scalper", "volume_focused"),     # Small profits, high volume
            ("trend_rider", "momentum_based"), # Follow strong trends
            ("contrarian", "reversal_focused") # Counter-trend strategies
        ]
        
        spec_name, strategy_type = specializations[bot_index % len(specializations)]
        
        # Create specialized config
        config = SimplifiedConfig(initial_capital_sol=capital_sol)
        
        # Adjust parameters based on specialization
        if strategy_type == "aggressive_fast":
            config.hunt_threshold = 0.55  # Lower threshold for more trades
            config.max_hold_time_hours = 2.0  # Shorter holds
            config.kelly_fraction = 0.3  # More aggressive sizing
        elif strategy_type == "patient_steady":
            config.hunt_threshold = 0.7  # Higher threshold for quality
            config.max_hold_time_hours = 8.0  # Longer holds
            config.kelly_fraction = 0.2  # Conservative sizing
        elif strategy_type == "volume_focused":
            config.hunt_threshold = 0.6
            config.max_hold_time_hours = 1.0  # Very short holds
            config.kelly_fraction = 0.15  # Smaller positions, more trades
        elif strategy_type == "momentum_based":
            config.hunt_threshold = 0.65
            config.max_hold_time_hours = 6.0
            config.kelly_fraction = 0.25
        elif strategy_type == "reversal_focused":
            config.hunt_threshold = 0.75  # Very selective
            config.max_hold_time_hours = 12.0
            config.kelly_fraction = 0.2
        
        return spec_name, config
    
    async def _start_swarm_coordination(self):
        """Start swarm coordination background tasks"""
        try:
            # Cross-bot learning task
            task1 = asyncio.create_task(self._cross_bot_learning_loop())
            self.background_tasks.append(task1)
            
            # Performance monitoring task
            task2 = asyncio.create_task(self._swarm_performance_monitoring())
            self.background_tasks.append(task2)
            
            # Strategy evolution task
            task3 = asyncio.create_task(self._strategy_evolution_loop())
            self.background_tasks.append(task3)
            
            # Collective decision making task
            task4 = asyncio.create_task(self._collective_decision_loop())
            self.background_tasks.append(task4)
            
            self.logger.info("✅ Swarm coordination systems started")
            
        except Exception as e:
            self.logger.error(f"❌ Error starting swarm coordination: {e}")
    
    async def run(self):
        """Run the trading swarm"""
        try:
            if not self.initialized:
                self.logger.error("❌ Swarm not initialized")
                return
            
            self.running = True
            self.logger.info("🔥 Starting Hyper-Intelligent Trading Swarm...")
            
            # Start all bot instances
            bot_tasks = []
            for bot_id, instance in self.bot_instances.items():
                if instance.bot.initialized:
                    task = asyncio.create_task(instance.bot.run())
                    bot_tasks.append(task)
                    self.logger.info(f"🚀 Started bot {bot_id}")
            
            # Combine bot tasks with background tasks
            all_tasks = bot_tasks + self.background_tasks
            
            # Run until shutdown
            await asyncio.gather(*all_tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in trading swarm: {e}")
            await self.emergency_shutdown()
    
    async def _cross_bot_learning_loop(self):
        """Cross-bot learning and signal sharing"""
        while self.running:
            try:
                # Collect performance data from all bots
                performance_data = {}
                for bot_id, instance in self.bot_instances.items():
                    try:
                        status = instance.bot.get_status()
                        performance_data[bot_id] = {
                            'win_rate': status['metrics']['win_rate'],
                            'total_profit': status['metrics']['total_profit_sol'],
                            'trades_executed': status['metrics']['trades_executed'],
                            'specialization': instance.specialization
                        }
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not get status for bot {bot_id}: {e}")
                
                # Identify best performing strategies
                if performance_data:
                    best_performer = max(performance_data.items(), 
                                       key=lambda x: x[1]['win_rate'] if x[1]['trades_executed'] > 5 else 0)
                    
                    self.swarm_metrics.best_performer_id = best_performer[0]
                    
                    # Share successful strategies with underperforming bots
                    await self._share_successful_strategies(performance_data)
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Error in cross-bot learning: {e}")
                await asyncio.sleep(300)
    
    async def _swarm_performance_monitoring(self):
        """Monitor and aggregate swarm performance"""
        while self.running:
            try:
                # Aggregate metrics
                total_profit = 0.0
                total_capital = 0.0
                win_rates = []
                active_count = 0
                
                for bot_id, instance in self.bot_instances.items():
                    try:
                        status = instance.bot.get_status()
                        if status['running']:
                            active_count += 1
                            total_profit += status['metrics']['total_profit_sol']
                            total_capital += status['metrics']['current_capital_sol']
                            
                            if status['metrics']['trades_executed'] > 0:
                                win_rates.append(status['metrics']['win_rate'])
                                
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not get metrics for bot {bot_id}: {e}")
                
                # Update swarm metrics
                self.swarm_metrics.active_bots = active_count
                self.swarm_metrics.total_capital_sol = total_capital
                self.swarm_metrics.total_profit_sol = total_profit
                self.swarm_metrics.aggregate_win_rate = statistics.mean(win_rates) if win_rates else 0.0
                
                # Check for kill switch conditions
                await self._check_swarm_kill_conditions()
                
                # Log swarm status
                self.logger.info(f"📊 Swarm Status: {active_count} active bots, "
                               f"Total Profit: {total_profit:.4f} SOL, "
                               f"Aggregate Win Rate: {self.swarm_metrics.aggregate_win_rate:.3f}")
                
                await asyncio.sleep(120)  # Every 2 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Error in swarm monitoring: {e}")
                await asyncio.sleep(120)
    
    async def _strategy_evolution_loop(self):
        """Evolve strategies based on collective performance"""
        while self.running:
            try:
                # Strategy evolution happens less frequently
                await asyncio.sleep(3600)  # Every hour
                
                # Analyze performance patterns
                await self._analyze_performance_patterns()
                
                # Evolve underperforming bots
                await self._evolve_underperformers()
                
                self.swarm_metrics.adaptation_cycles += 1
                
            except Exception as e:
                self.logger.error(f"❌ Error in strategy evolution: {e}")
                await asyncio.sleep(3600)
    
    async def _collective_decision_loop(self):
        """Collective decision making for market opportunities"""
        while self.running:
            try:
                # Collective decisions happen frequently
                await asyncio.sleep(60)  # Every minute
                
                # Gather collective intelligence signals
                await self._gather_collective_signals()
                
                # Make collective decisions
                await self._make_collective_decisions()
                
            except Exception as e:
                self.logger.error(f"❌ Error in collective decision making: {e}")
                await asyncio.sleep(60)
    
    async def _check_swarm_kill_conditions(self):
        """Check if swarm-level kill switch should be triggered"""
        try:
            # Check for massive losses across the swarm
            total_loss = -self.swarm_metrics.total_profit_sol
            if total_loss > self.initial_capital * 0.5:  # 50% loss
                self.swarm_kill_switch.trigger(
                    KillSwitchTrigger.LOSS_THRESHOLD,
                    f"Swarm total loss exceeds 50%: {total_loss:.4f}"
                )
            
            # Check for widespread bot failures
            failure_rate = 1.0 - (self.swarm_metrics.active_bots / self.swarm_metrics.total_bots)
            if failure_rate > 0.7:  # 70% failure rate
                self.swarm_kill_switch.trigger(
                    KillSwitchTrigger.SYSTEM_ERROR,
                    f"Swarm failure rate exceeds 70%: {failure_rate:.3f}"
                )
            
        except Exception as e:
            self.logger.error(f"❌ Error checking swarm kill conditions: {e}")
    
    # Placeholder methods for advanced swarm intelligence features
    async def _share_successful_strategies(self, performance_data: Dict[str, Any]):
        """Share successful strategies between bots"""
        # Implementation would analyze best performing strategies and adapt others
        pass
    
    async def _analyze_performance_patterns(self):
        """Analyze swarm performance patterns"""
        # Implementation would identify patterns in collective performance
        pass
    
    async def _evolve_underperformers(self):
        """Evolve strategies of underperforming bots"""
        # Implementation would modify configs of poor performers
        pass
    
    async def _gather_collective_signals(self):
        """Gather signals from all bots for collective intelligence"""
        # Implementation would aggregate market signals across bots
        pass
    
    async def _make_collective_decisions(self):
        """Make collective decisions based on swarm intelligence"""
        # Implementation would make group decisions on market opportunities
        pass
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"🛑 Received signal {signum}, shutting down swarm...")
        asyncio.create_task(self.shutdown())
    
    async def shutdown(self):
        """Graceful swarm shutdown"""
        try:
            self.logger.info("🛑 Shutting down trading swarm...")
            self.running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Shutdown all bot instances
            shutdown_tasks = []
            for bot_id, instance in self.bot_instances.items():
                if instance.bot.initialized:
                    task = asyncio.create_task(instance.bot.shutdown())
                    shutdown_tasks.append(task)
            
            # Wait for all shutdowns
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            self.logger.info("✅ Swarm shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during swarm shutdown: {e}")
    
    async def emergency_shutdown(self):
        """Emergency swarm shutdown"""
        try:
            self.logger.critical("🚨 EMERGENCY SHUTDOWN - Trading Swarm")
            
            # Trigger swarm kill switch
            if not self.swarm_kill_switch.is_triggered:
                self.swarm_kill_switch.trigger(
                    KillSwitchTrigger.MANUAL,
                    "Emergency shutdown initiated"
                )
            
            self.running = False
            
            # Emergency shutdown all bots
            emergency_tasks = []
            for bot_id, instance in self.bot_instances.items():
                if instance.bot.initialized:
                    task = asyncio.create_task(instance.bot.emergency_shutdown())
                    emergency_tasks.append(task)
            
            # Wait with timeout
            if emergency_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*emergency_tasks, return_exceptions=True),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    self.logger.critical("⚠️ Emergency shutdown timeout - some bots may not have shut down properly")
            
            # Cancel all background tasks immediately
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            self.logger.critical("🚨 Emergency shutdown complete")
            
        except Exception as e:
            self.logger.critical(f"❌ Critical error during emergency shutdown: {e}")
            sys.exit(1)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status"""
        try:
            bot_statuses = {}
            for bot_id, instance in self.bot_instances.items():
                try:
                    bot_statuses[bot_id] = {
                        'specialization': instance.specialization,
                        'performance_weight': instance.performance_weight,
                        'is_leader': instance.is_leader,
                        'bot_status': instance.bot.get_status() if instance.bot.initialized else {}
                    }
                except Exception as e:
                    bot_statuses[bot_id] = {'error': str(e)}
            
            return {
                'swarm_mode': self.swarm_mode.value,
                'swarm_size': self.swarm_size,
                'initialized': self.initialized,
                'running': self.running,
                'kill_switch_status': self.swarm_kill_switch.get_status(),
                'swarm_metrics': {
                    'total_bots': self.swarm_metrics.total_bots,
                    'active_bots': self.swarm_metrics.active_bots,
                    'total_capital_sol': self.swarm_metrics.total_capital_sol,
                    'total_profit_sol': self.swarm_metrics.total_profit_sol,
                    'aggregate_win_rate': self.swarm_metrics.aggregate_win_rate,
                    'best_performer_id': self.swarm_metrics.best_performer_id,
                    'adaptation_cycles': self.swarm_metrics.adaptation_cycles
                },
                'bot_instances': bot_statuses,
                'background_tasks_count': len(self.background_tasks)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error getting swarm status: {e}")
            return {'error': str(e)}


# Legacy compatibility class alias
MemecoinTradingBot = HyperIntelligentTradingSwarm

