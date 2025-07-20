"""
COLONY COMMANDER - HOLARCHIC TRADING COLONY ORCHESTRATOR
=======================================================

The Colony Commander is the top-level holon that manages multiple
HyperIntelligentTradingSwarm instances, each representing a different
trading strategy or market focus.

This implements the Holarchy structure:
- Colony Commander (Top-level holon)
- HyperIntelligentTradingSwarm (Swarm holon) 
- TradingWallet (Worker ant holon)

Each level has autonomy while contributing to the colony's overall success.
"""

import asyncio
import os
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from worker_ant_v1.trading.main import HyperIntelligentTradingSwarm
from worker_ant_v1.core.vault_wallet_system import get_vault_system
from worker_ant_v1.core.unified_config import UnifiedConfigManager
from worker_ant_v1.utils.logger import get_logger


class ColonyState(Enum):
    """Colony operational states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    REBALANCING = "rebalancing"
    BLITZSCALING = "blitzscaling"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


@dataclass
class SwarmConfig:
    """Configuration for a trading swarm"""
    swarm_id: str
    config_file: str
    initial_capital: float
    strategy_type: str
    risk_profile: str
    target_markets: List[str]
    max_positions: int
    enabled: bool = True


@dataclass
class ColonyMetrics:
    """Colony-wide performance metrics"""
    total_capital: float = 0.0
    total_profit: float = 0.0
    overall_win_rate: float = 0.0
    active_swarms: int = 0
    total_trades: int = 0
    successful_trades: int = 0
    colony_efficiency: float = 0.0
    last_rebalance: Optional[datetime] = None
    blitzscaling_active: bool = False
    blitzscaling_trigger_time: Optional[datetime] = None


class ColonyCommander:
    """Top-level holon managing the entire trading colony"""
    
    def __init__(self):
        self.logger = get_logger("ColonyCommander")
        
        # Colony state
        self.state = ColonyState.INITIALIZING
        self.metrics = ColonyMetrics()
        self.blitzscaling_active = False
        
        # Swarm management
        self.swarms: Dict[str, HyperIntelligentTradingSwarm] = {}
        self.swarm_configs: Dict[str, SwarmConfig] = {}
        self.swarm_performance: Dict[str, Dict[str, Any]] = {}
        
        # Master vault system
        self.master_vault = None
        
        # Configuration
        self.config_manager = UnifiedConfigManager()
        self.colony_config = {
            'rebalance_interval_hours': 6,
            'performance_threshold': 0.7,  # 70% win rate for blitzscaling
            'safety_drawdown_threshold': 0.15,  # 15% drawdown to disable blitzscaling
            'tax_rate': 0.1,  # 10% tax on high-performing swarms
            'capital_reallocation_rate': 0.3,  # 30% of taxed capital reallocated
        }
        
        # System state
        self.initialized = False
        self.running = False
        self.start_time = None
        
        self.logger.info("🏛️ Colony Commander initialized")
    
    async def initialize_colony(self) -> bool:
        """Initialize the entire trading colony"""
        try:
            self.logger.info("🏛️ Initializing Trading Colony...")
            
            # Initialize master vault
            self.master_vault = await get_vault_system()
            
            # Load swarm configurations
            await self._load_swarm_configurations()
            
            # Initialize swarms
            await self._initialize_swarms()
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start background tasks
            asyncio.create_task(self._colony_monitoring_loop())
            asyncio.create_task(self._performance_analysis_loop())
            asyncio.create_task(self._operational_mode_loop())
            
            self.state = ColonyState.ACTIVE
            self.initialized = True
            self.running = True
            self.start_time = datetime.now()
            
            self.logger.info("✅ Trading Colony initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Colony initialization failed: {e}")
            return False
    
    async def _load_swarm_configurations(self):
        """Load configurations for all trading swarms"""
        config_dir = Path(__file__).parent.parent / "config"
        
        # Define swarm configurations
        self.swarm_configs = {
            "memecoin_swarm": SwarmConfig(
                swarm_id="memecoin_swarm",
                config_file=str(config_dir / "memecoin_swarm.env"),
                initial_capital=100.0,
                strategy_type="aggressive_memecoin",
                risk_profile="high_risk_high_reward",
                target_markets=["memecoins", "new_launches"],
                max_positions=8
            ),
            "bluechip_swarm": SwarmConfig(
                swarm_id="bluechip_swarm", 
                config_file=str(config_dir / "bluechip_swarm.env"),
                initial_capital=150.0,
                strategy_type="conservative_bluechip",
                risk_profile="moderate_risk",
                target_markets=["bluechips", "established_tokens"],
                max_positions=5
            ),
            "scalping_swarm": SwarmConfig(
                swarm_id="scalping_swarm",
                config_file=str(config_dir / "scalping_swarm.env"), 
                initial_capital=50.0,
                strategy_type="high_frequency_scalping",
                risk_profile="low_risk_high_frequency",
                target_markets=["high_volatility", "liquid_tokens"],
                max_positions=12
            )
        }
        
        # Create config files if they don't exist
        await self._create_swarm_config_files()
        
        self.logger.info(f"📋 Loaded {len(self.swarm_configs)} swarm configurations")
    
    async def _create_swarm_config_files(self):
        """Create configuration files for each swarm"""
        template_file = Path(__file__).parent.parent / "config" / "env.template"
        
        if not template_file.exists():
            self.logger.warning("⚠️ No template file found, using default configurations")
            return
        
        for swarm_id, config in self.swarm_configs.items():
            config_path = Path(config.config_file)
            if not config_path.exists():
                # Copy template and customize for swarm
                with open(template_file, 'r') as f:
                    template_content = f.read()
                
                # Customize template for specific swarm
                swarm_content = template_content.replace(
                    "TRADING_MODE=SIMULATION",
                    f"TRADING_MODE=LIVE\nSWARM_ID={swarm_id}\nSTRATEGY_TYPE={config.strategy_type}\nRISK_PROFILE={config.risk_profile}\nMAX_POSITIONS={config.max_positions}"
                )
                
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    f.write(swarm_content)
                
                self.logger.info(f"📄 Created config file: {config_path}")
    
    async def _initialize_swarms(self):
        """Initialize all trading swarms"""
        for swarm_id, config in self.swarm_configs.items():
            if not config.enabled:
                continue
                
            try:
                self.logger.info(f"🚀 Initializing swarm: {swarm_id}")
                
                # Create swarm instance with specific configuration
                swarm = HyperIntelligentTradingSwarm()
                
                # Set swarm-specific configuration
                os.environ['SWARM_CONFIG_FILE'] = config.config_file
                os.environ['SWARM_ID'] = swarm_id
                os.environ['INITIAL_CAPITAL'] = str(config.initial_capital)
                
                # Initialize the swarm
                if await swarm.initialize_all_systems():
                    self.swarms[swarm_id] = swarm
                    self.swarm_performance[swarm_id] = {
                        'total_trades': 0,
                        'successful_trades': 0,
                        'total_profit': 0.0,
                        'win_rate': 0.0,
                        'current_capital': config.initial_capital,
                        'last_update': datetime.now()
                    }
                    self.logger.info(f"✅ Swarm {swarm_id} initialized successfully")
                else:
                    self.logger.error(f"❌ Failed to initialize swarm {swarm_id}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error initializing swarm {swarm_id}: {e}")
    
    async def run_colony(self):
        """Run the entire trading colony"""
        try:
            if not self.initialized:
                self.logger.error("❌ Colony not initialized")
                return
            
            self.logger.info("🏛️ Starting Trading Colony...")
            
            # Start all swarms
            swarm_tasks = []
            for swarm_id, swarm in self.swarms.items():
                task = asyncio.create_task(self._run_swarm(swarm_id, swarm))
                swarm_tasks.append(task)
            
            # Wait for all swarms to complete or colony shutdown
            while self.running:
                await asyncio.sleep(1)
                
                # Check if any swarm has failed
                for task in swarm_tasks:
                    if task.done() and task.exception():
                        self.logger.error(f"❌ Swarm task failed: {task.exception()}")
            
            # Cancel remaining tasks
            for task in swarm_tasks:
                if not task.done():
                    task.cancel()
            
        except Exception as e:
            self.logger.error(f"❌ Colony error: {e}")
            await self.emergency_shutdown()
    
    async def _run_swarm(self, swarm_id: str, swarm: HyperIntelligentTradingSwarm):
        """Run a single swarm"""
        try:
            self.logger.info(f"🔥 Starting swarm: {swarm_id}")
            await swarm.run()
        except Exception as e:
            self.logger.error(f"❌ Swarm {swarm_id} error: {e}")
    
    async def _colony_monitoring_loop(self):
        """Monitor colony-wide performance and health"""
        while self.running:
            try:
                # Update colony metrics
                await self._update_colony_metrics()
                
                # Check for rebalancing
                if await self._should_rebalance():
                    await self._rebalance_colony_capital()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"❌ Colony monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _update_colony_metrics(self):
        """Update colony-wide performance metrics"""
        total_capital = 0.0
        total_profit = 0.0
        total_trades = 0
        successful_trades = 0
        
        for swarm_id, performance in self.swarm_performance.items():
            total_capital += performance.get('current_capital', 0.0)
            total_profit += performance.get('total_profit', 0.0)
            total_trades += performance.get('total_trades', 0)
            successful_trades += performance.get('successful_trades', 0)
        
        self.metrics.total_capital = total_capital
        self.metrics.total_profit = total_profit
        self.metrics.total_trades = total_trades
        self.metrics.successful_trades = successful_trades
        self.metrics.active_swarms = len(self.swarms)
        
        if total_trades > 0:
            self.metrics.overall_win_rate = successful_trades / total_trades
        
        # Calculate colony efficiency (profit per trade)
        if total_trades > 0:
            self.metrics.colony_efficiency = total_profit / total_trades
    
    async def _should_rebalance(self) -> bool:
        """Check if colony should rebalance capital"""
        if not self.metrics.last_rebalance:
            return True
        
        time_since_rebalance = datetime.now() - self.metrics.last_rebalance
        return time_since_rebalance.total_seconds() >= (self.colony_config['rebalance_interval_hours'] * 3600)
    
    async def _rebalance_colony_capital(self):
        """Rebalance capital across swarms based on performance"""
        try:
            self.logger.info("⚖️ Starting colony capital rebalancing...")
            self.state = ColonyState.REBALANCING
            
            # Calculate performance rankings
            performance_rankings = []
            for swarm_id, performance in self.swarm_performance.items():
                win_rate = performance.get('win_rate', 0.0)
                profit = performance.get('total_profit', 0.0)
                efficiency = profit / max(performance.get('total_trades', 1), 1)
                
                performance_rankings.append({
                    'swarm_id': swarm_id,
                    'win_rate': win_rate,
                    'profit': profit,
                    'efficiency': efficiency,
                    'score': (win_rate * 0.4) + (efficiency * 0.6)
                })
            
            # Sort by performance score
            performance_rankings.sort(key=lambda x: x['score'], reverse=True)
            
            # Tax high-performing swarms
            total_taxed_capital = 0.0
            for ranking in performance_rankings[:2]:  # Top 2 performers
                swarm_id = ranking['swarm_id']
                current_capital = self.swarm_performance[swarm_id]['current_capital']
                tax_amount = current_capital * self.colony_config['tax_rate']
                
                # Transfer to master vault
                if swarm_id in self.swarms:
                    swarm = self.swarms[swarm_id]
                    # Note: In a real implementation, this would transfer actual capital
                    self.swarm_performance[swarm_id]['current_capital'] -= tax_amount
                    total_taxed_capital += tax_amount
                    
                    self.logger.info(f"💰 Taxed {tax_amount:.6f} SOL from {swarm_id}")
            
            # Reallocate capital to underperforming swarms
            reallocation_amount = total_taxed_capital * self.colony_config['capital_reallocation_rate']
            remaining_for_vault = total_taxed_capital - reallocation_amount
            
            # Distribute to bottom performers
            for ranking in performance_rankings[-2:]:  # Bottom 2 performers
                swarm_id = ranking['swarm_id']
                allocation = reallocation_amount / 2
                
                if swarm_id in self.swarms:
                    self.swarm_performance[swarm_id]['current_capital'] += allocation
                    self.logger.info(f"💸 Allocated {allocation:.6f} SOL to {swarm_id}")
            
            # Update master vault
            if self.master_vault:
                # Note: In a real implementation, this would add to actual vault balance
                self.logger.info(f"🏦 Added {remaining_for_vault:.6f} SOL to master vault")
            
            self.metrics.last_rebalance = datetime.now()
            self.state = ColonyState.ACTIVE
            
            self.logger.info("✅ Colony capital rebalancing completed")
            
        except Exception as e:
            self.logger.error(f"❌ Capital rebalancing error: {e}")
            self.state = ColonyState.ACTIVE
    
    async def _performance_analysis_loop(self):
        """Analyze performance and update swarm metrics"""
        while self.running:
            try:
                for swarm_id, swarm in self.swarms.items():
                    # Get swarm status (this would need to be implemented in the swarm)
                    if hasattr(swarm, 'get_status'):
                        status = swarm.get_status()
                        
                        # Update performance metrics
                        self.swarm_performance[swarm_id].update({
                            'total_trades': status.get('total_trades', 0),
                            'successful_trades': status.get('successful_trades', 0),
                            'total_profit': status.get('total_profit', 0.0),
                            'current_capital': status.get('current_capital', 0.0),
                            'last_update': datetime.now()
                        })
                        
                        # Calculate win rate
                        total_trades = self.swarm_performance[swarm_id]['total_trades']
                        successful_trades = self.swarm_performance[swarm_id]['successful_trades']
                        
                        if total_trades > 0:
                            self.swarm_performance[swarm_id]['win_rate'] = successful_trades / total_trades
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Performance analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _operational_mode_loop(self):
        """Monitor and update operational mode (including blitzscaling)"""
        while self.running:
            try:
                await self._update_operational_mode()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"❌ Operational mode error: {e}")
                await asyncio.sleep(60)
    
    async def _update_operational_mode(self):
        """Update operational mode based on colony performance"""
        # Check for blitzscaling trigger
        if not self.blitzscaling_active:
            if (self.metrics.overall_win_rate >= self.colony_config['performance_threshold'] and
                self.metrics.total_trades >= 50):  # Minimum trades for confidence
                
                self.blitzscaling_active = True
                self.metrics.blitzscaling_active = True
                self.metrics.blitzscaling_trigger_time = datetime.now()
                self.state = ColonyState.BLITZSCALING
                
                self.logger.info("🚀 BLITZSCALING MODE ACTIVATED!")
                self.logger.info(f"   Win Rate: {self.metrics.overall_win_rate:.2%}")
                self.logger.info(f"   Total Trades: {self.metrics.total_trades}")
        
        # Check for blitzscaling disable
        elif self.blitzscaling_active:
            # Calculate current drawdown
            if self.metrics.total_capital > 0:
                initial_capital = sum(config.initial_capital for config in self.swarm_configs.values())
                current_drawdown = (initial_capital - self.metrics.total_capital) / initial_capital
                
                if current_drawdown >= self.colony_config['safety_drawdown_threshold']:
                    self.blitzscaling_active = False
                    self.metrics.blitzscaling_active = False
                    self.state = ColonyState.ACTIVE
                    
                    self.logger.warning("⚠️ BLITZSCALING MODE DISABLED - Safety threshold reached")
                    self.logger.warning(f"   Current Drawdown: {current_drawdown:.2%}")
        
        # Notify swarms of mode change
        for swarm_id, swarm in self.swarms.items():
            if hasattr(swarm, 'set_blitzscaling_mode'):
                await swarm.set_blitzscaling_mode(self.blitzscaling_active)
    
    def get_colony_status(self) -> Dict[str, Any]:
        """Get comprehensive colony status"""
        return {
            'state': self.state.value,
            'initialized': self.initialized,
            'running': self.running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'metrics': {
                'total_capital': self.metrics.total_capital,
                'total_profit': self.metrics.total_profit,
                'overall_win_rate': self.metrics.overall_win_rate,
                'active_swarms': self.metrics.active_swarms,
                'total_trades': self.metrics.total_trades,
                'successful_trades': self.metrics.successful_trades,
                'colony_efficiency': self.metrics.colony_efficiency,
                'blitzscaling_active': self.metrics.blitzscaling_active,
                'last_rebalance': self.metrics.last_rebalance.isoformat() if self.metrics.last_rebalance else None
            },
            'swarms': {
                swarm_id: {
                    'config': {
                        'strategy_type': config.strategy_type,
                        'risk_profile': config.risk_profile,
                        'target_markets': config.target_markets,
                        'max_positions': config.max_positions
                    },
                    'performance': self.swarm_performance.get(swarm_id, {})
                }
                for swarm_id, config in self.swarm_configs.items()
            }
        }
    
    async def shutdown(self):
        """Shutdown the entire colony"""
        try:
            self.logger.info("🛑 Shutting down Trading Colony...")
            
            self.running = False
            self.state = ColonyState.SHUTDOWN
            
            # Shutdown all swarms
            for swarm_id, swarm in self.swarms.items():
                try:
                    await swarm.shutdown()
                    self.logger.info(f"✅ Swarm {swarm_id} shutdown complete")
                except Exception as e:
                    self.logger.error(f"❌ Error shutting down swarm {swarm_id}: {e}")
            
            # Shutdown master vault
            if self.master_vault:
                await self.master_vault.shutdown()
            
            self.logger.info("✅ Trading Colony shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during colony shutdown: {e}")
    
    async def emergency_shutdown(self):
        """Emergency shutdown of the colony"""
        try:
            self.logger.error("🚨 EMERGENCY COLONY SHUTDOWN TRIGGERED")
            await self.shutdown()
        except Exception as e:
            self.logger.error(f"Emergency shutdown error: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"🛑 Received signal {signum}, shutting down colony...")
        asyncio.create_task(self.shutdown())


async def main():
    """Main entry point for the colony commander"""
    parser = argparse.ArgumentParser(description="Trading Colony Commander")
    
    parser.add_argument(
        "--mode",
        choices=["production", "simulation", "test"],
        default="simulation",
        help="Colony operation mode (default: simulation)"
    )
    
    args = parser.parse_args()
    
    # Display startup information
    print("\n🏛️ TRADING COLONY COMMANDER")
    print("=" * 50)
    print(f"Mode: {args.mode.upper()}")
    print("=" * 50)
    
    # Create and run colony
    colony = ColonyCommander()
    
    try:
        if await colony.initialize_colony():
            await colony.run_colony()
        else:
            print("❌ Colony initialization failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Colony shutdown requested")
        await colony.shutdown()
    except Exception as e:
        print(f"❌ Colony error: {e}")
        await colony.emergency_shutdown()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    asyncio.run(main()) 