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
import uuid
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import redis.asyncio as redis
import json

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from worker_ant_v1.trading.main import HyperIntelligentTradingSwarm
from worker_ant_v1.safety.vault_wallet_system import get_vault_system
from worker_ant_v1.core.unified_config import UnifiedConfigManager
# NarrativeAnt removed - using simplified approach
from worker_ant_v1.utils.logger import get_logger


class ColonyState(Enum):
    """Colony operational states"""
    INITIALIZING = "initializing"
    STANDBY = "standby"
    ACTIVE = "active"
    REBALANCING = "rebalancing"
    BLITZSCALING = "blitzscaling"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


class LeadershipState(Enum):
    """HA leadership states"""
    CANDIDATE = "candidate"
    FOLLOWER = "follower" 
    LEADER = "leader"
    FAILED = "failed"


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
    """Top-level holon managing the entire trading colony with HA support"""
    
    def __init__(self, enable_ha: bool = True):
        self.logger = get_logger("ColonyCommander")
        
        # HA Configuration
        self.enable_ha = enable_ha
        self.instance_id = str(uuid.uuid4())[:8]
        self.hostname = socket.gethostname()
        self.leadership_state = LeadershipState.CANDIDATE
        self.redis_client: Optional[redis.Redis] = None
        
        # HA Constants
        self.LEADER_LOCK_KEY = "antbot:colony:leader"
        self.HEARTBEAT_KEY = "antbot:colony:heartbeat"
        self.LEADER_TTL = 30  # Leader lock TTL in seconds
        self.HEARTBEAT_INTERVAL = 10  # Heartbeat interval in seconds
        self.ELECTION_TIMEOUT = 15  # Election timeout in seconds
        
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
        
        # Strategic Narrative Intelligence - DISABLED (NarrativeAnt not implemented)
        # self.narrative_ant: Optional[NarrativeAnt] = None
        self.dominant_narratives: Dict[str, float] = {}
        
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
        
        # HA background tasks
        self._ha_tasks: List[asyncio.Task] = []
        
        self.logger.info(f"🏛️ Colony Commander initialized (HA: {enable_ha}, ID: {self.instance_id})")
    
    async def initialize_colony(self) -> bool:
        """Initialize the entire trading colony with HA support"""
        try:
            self.logger.info("🏛️ Initializing Trading Colony...")
            
            # Initialize HA system if enabled
            if self.enable_ha:
                if not await self._initialize_ha():
                    self.logger.error("❌ HA initialization failed")
                    return False
            
            # Initialize master vault
            self.master_vault = await get_vault_system()
            
            # Initialize strategic narrative intelligence - DISABLED (NarrativeAnt not implemented)
            # self.narrative_ant = NarrativeAnt()
            # await self.narrative_ant.initialize()
            self.logger.info("🧠 Narrative intelligence disabled - Using simplified approach")
            
            # Load swarm configurations
            await self._load_swarm_configurations()
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.initialized = True
            self.running = True
            self.start_time = datetime.now()
            
            if self.enable_ha:
                # Start HA processes
                await self._start_ha_processes()
            else:
                # Single instance mode - become leader immediately
                self.leadership_state = LeadershipState.LEADER
                await self._become_active_leader()
            
            self.logger.info("✅ Trading Colony initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Colony initialization failed: {e}")
            return False
    
    async def _initialize_ha(self) -> bool:
        """Initialize High Availability Redis connection"""
        try:
            # Redis configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG
            from worker_ant_v1.core.unified_config import get_redis_config
            redis_config = get_redis_config()
            redis_host = redis_config['host']
            redis_port = redis_config['port']
            redis_password = redis_config['password']
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_client.ping()
            self.logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Redis: {e}")
            return False
    
    async def _start_ha_processes(self):
        """Start HA background processes"""
        if not self.redis_client:
            return
        
        # Start leader election task
        election_task = asyncio.create_task(self._leader_election_loop())
        self._ha_tasks.append(election_task)
        
        self.logger.info("🔄 HA processes started")
    
    async def _leader_election_loop(self):
        """Main leader election loop"""
        while self.running:
            try:
                if self.leadership_state == LeadershipState.CANDIDATE:
                    await self._attempt_leader_election()
                elif self.leadership_state == LeadershipState.LEADER:
                    await self._maintain_leadership()
                elif self.leadership_state == LeadershipState.FOLLOWER:
                    await self._follow_leader()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Leader election error: {e}")
                self.leadership_state = LeadershipState.FAILED
                await asyncio.sleep(10)  # Back off on error
    
    async def _attempt_leader_election(self):
        """Attempt to become the leader"""
        try:
            # Try to acquire the leader lock
            leader_info = {
                'instance_id': self.instance_id,
                'hostname': self.hostname,
                'elected_at': datetime.now().isoformat(),
                'heartbeat': datetime.now().isoformat()
            }
            
            # Use SET with NX (only set if key doesn't exist) and EX (expiration)
            acquired = await self.redis_client.set(
                self.LEADER_LOCK_KEY, 
                json.dumps(leader_info),
                nx=True,  # Only set if key doesn't exist
                ex=self.LEADER_TTL  # Expiration time
            )
            
            if acquired:
                self.leadership_state = LeadershipState.LEADER
                self.logger.info(f"🏆 Became colony leader (ID: {self.instance_id})")
                await self._become_active_leader()
            else:
                # Check if current leader is alive
                await self._check_current_leader()
                
        except Exception as e:
            self.logger.error(f"❌ Leader election failed: {e}")
            self.leadership_state = LeadershipState.FAILED
    
    async def _maintain_leadership(self):
        """Maintain leadership by updating heartbeat"""
        try:
            # Update heartbeat
            leader_info = {
                'instance_id': self.instance_id,
                'hostname': self.hostname,
                'elected_at': self.start_time.isoformat() if self.start_time else datetime.now().isoformat(),
                'heartbeat': datetime.now().isoformat()
            }
            
            # Extend the lock
            updated = await self.redis_client.set(
                self.LEADER_LOCK_KEY,
                json.dumps(leader_info),
                ex=self.LEADER_TTL
            )
            
            if not updated:
                self.logger.warning("⚠️ Lost leadership - failed to update lock")
                self.leadership_state = LeadershipState.CANDIDATE
                await self._become_standby()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to maintain leadership: {e}")
            self.leadership_state = LeadershipState.CANDIDATE
            await self._become_standby()
    
    async def _follow_leader(self):
        """Follow the current leader"""
        try:
            # Check if leader is still active
            await self._check_current_leader()
            
            # Ensure we're in standby mode
            if self.state != ColonyState.STANDBY:
                await self._become_standby()
                
        except Exception as e:
            self.logger.error(f"❌ Error following leader: {e}")
            self.leadership_state = LeadershipState.CANDIDATE
    
    async def _check_current_leader(self):
        """Check if current leader is alive"""
        try:
            leader_data = await self.redis_client.get(self.LEADER_LOCK_KEY)
            
            if not leader_data:
                # No leader - become candidate
                self.leadership_state = LeadershipState.CANDIDATE
                return
            
            leader_info = json.loads(leader_data)
            leader_heartbeat = datetime.fromisoformat(leader_info['heartbeat'])
            current_time = datetime.now()
            
            # Check if heartbeat is stale
            if (current_time - leader_heartbeat).total_seconds() > self.LEADER_TTL * 2:
                self.logger.warning("⚠️ Leader heartbeat stale - starting new election")
                # Try to delete stale lock
                await self.redis_client.delete(self.LEADER_LOCK_KEY)
                self.leadership_state = LeadershipState.CANDIDATE
            else:
                # Leader is alive
                if leader_info['instance_id'] != self.instance_id:
                    self.leadership_state = LeadershipState.FOLLOWER
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to check leader: {e}")
            self.leadership_state = LeadershipState.CANDIDATE
    
    async def _become_active_leader(self):
        """Transition to active leader state"""
        try:
            self.logger.info("🚀 Transitioning to active leader")
            
            # Initialize swarms
            await self._initialize_swarms()
            
            # Start colony management tasks
            management_tasks = [
                asyncio.create_task(self._colony_monitoring_loop()),
                asyncio.create_task(self._performance_analysis_loop()),
                asyncio.create_task(self._operational_mode_loop())
            ]
            
            self._ha_tasks.extend(management_tasks)
            self.state = ColonyState.ACTIVE
            
            self.logger.info("✅ Colony Commander is now active leader")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to become active leader: {e}")
            self.leadership_state = LeadershipState.FAILED
    
    async def _become_standby(self):
        """Transition to standby state"""
        try:
            if self.state == ColonyState.STANDBY:
                return
            
            self.logger.info("⏸️ Transitioning to standby mode")
            
            # Shutdown swarms
            for swarm in self.swarms.values():
                try:
                    if hasattr(swarm, 'pause'):
                        await swarm.pause()
                    elif hasattr(swarm, 'shutdown'):
                        await swarm.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down swarm: {e}")
            
            # Cancel management tasks (keep HA tasks)
            for task in self._ha_tasks[1:]:  # Skip the election task
                if not task.done():
                    task.cancel()
            
            self.swarms.clear()
            self.state = ColonyState.STANDBY
            
            self.logger.info("✅ Colony Commander is now in standby")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to become standby: {e}")
    
    def is_leader(self) -> bool:
        """Check if this instance is the current leader"""
        return self.leadership_state == LeadershipState.LEADER and self.state == ColonyState.ACTIVE
    
    def get_ha_status(self) -> Dict[str, Any]:
        """Get HA status information"""
        return {
            'instance_id': self.instance_id,
            'hostname': self.hostname,
            'leadership_state': self.leadership_state.value,
            'colony_state': self.state.value,
            'is_leader': self.is_leader(),
            'ha_enabled': self.enable_ha,
            'running': self.running,
            'active_swarms': len(self.swarms),
            'start_time': self.start_time.isoformat() if self.start_time else None
        }
    
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
        """Monitor and update operational mode (including blitzscaling) with narrative intelligence"""
        while self.running:
            try:
                # Update narrative intelligence first
                await self._update_narrative_direction()
                
                # Then update operational mode
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
    
    async def _update_narrative_direction(self):
        """Query NarrativeAnt for dominant narratives and direct swarms accordingly - DISABLED"""
        try:
            # NarrativeAnt not implemented - using simplified approach
            # if not self.narrative_ant:
            #     return
            
            # Simplified fallback: clear dominant narratives
            self.dominant_narratives = {}
            
            self.logger.debug("🧠 Narrative direction update skipped - NarrativeAnt not implemented")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating narrative direction: {e}")
    
    async def _direct_swarms_to_narratives(self, dominant_narratives: List[Tuple[str, float]]):
        """Direct HyperIntelligentTradingSwarm instances to prioritize dominant narratives"""
        try:
            # Create narrative priority mapping for swarms
            narrative_weights = {}
            total_weight = sum(weight for _, weight in dominant_narratives)
            
            # Normalize weights and ensure they don't exceed 1.0
            for category, weight in dominant_narratives:
                normalized_weight = min(1.0, weight / max(total_weight, 1.0))
                narrative_weights[category] = normalized_weight
            
            # Direct each swarm to focus on these narratives
            for swarm_id, swarm in self.swarms.items():
                if hasattr(swarm, 'set_narrative_priorities'):
                    await swarm.set_narrative_priorities(narrative_weights)
                    self.logger.debug(f"📡 Directed {swarm_id} to prioritize narratives")
                elif hasattr(swarm, 'update_narrative_focus'):
                    await swarm.update_narrative_focus(narrative_weights)
                    self.logger.debug(f"📡 Updated {swarm_id} narrative focus")
            
        except Exception as e:
            self.logger.error(f"❌ Error directing swarms to narratives: {e}")
    
    def get_colony_status(self) -> Dict[str, Any]:
        """Get comprehensive colony status"""
        return {
            'state': self.state.value,
            'initialized': self.initialized,
            'running': self.running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'narrative_intelligence': {
                'narrative_ant_active': False,  # NarrativeAnt not implemented
                'dominant_narratives': {},  # Simplified approach
                'narrative_count': 0,
                'cultural_awareness_enabled': False
            },
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
        """Shutdown the entire colony with HA cleanup"""
        try:
            self.logger.info("🛑 Shutting down Trading Colony...")
            
            self.running = False
            self.state = ColonyState.SHUTDOWN
            
            # HA cleanup
            if self.enable_ha and self.redis_client:
                try:
                    # Release leader lock if we hold it
                    if self.leadership_state == LeadershipState.LEADER:
                        leader_data = await self.redis_client.get(self.LEADER_LOCK_KEY)
                        if leader_data:
                            leader_info = json.loads(leader_data)
                            if leader_info.get('instance_id') == self.instance_id:
                                await self.redis_client.delete(self.LEADER_LOCK_KEY)
                                self.logger.info("🔓 Released leader lock")
                    
                    # Cancel HA tasks
                    for task in self._ha_tasks:
                        if not task.done():
                            task.cancel()
                    
                    # Close Redis connection
                    await self.redis_client.close()
                    self.logger.info("✅ HA cleanup complete")
                    
                except Exception as e:
                    self.logger.error(f"❌ HA cleanup error: {e}")
            
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


    # Strategy interface methods for integration with run_bot.py
    async def initialize_all_systems(self) -> bool:
        """Initialize all systems (Strategy interface)"""
        return await self.initialize_colony()
    
    async def run(self):
        """Run the strategy (Strategy interface)"""
        await self.run_colony()
    
    def get_status(self) -> Dict[str, Any]:
        """Get strategy status (Strategy interface)"""
        return self.get_colony_status()


# NOTE: Entry point execution removed to comply with Entry Point Doctrine
# All execution must go through entry_points/run_bot.py --strategy colony
# This maintains the Single Entry Point Rule as defined in CONTRIBUTING.md 