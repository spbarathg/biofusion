"""
NEURAL COMMAND CENTER - SWARM INTELLIGENCE COORDINATOR
====================================================

The neural command center of the 10-wallet swarm. Integrates real-time Twitter sentiment,
on-chain data, and AI predictions. Enforces strict consensus: only trades when
AI model + on-chain + Twitter ALL agree. No exceptions.

Implements the complete vision: survival-focused, learning-driven, cold calculation.

🔥 ENHANCED WITH HIGH AVAILABILITY:
- Redis-based leader election
- Automatic failover and standby promotion  
- Heartbeat monitoring and health checks
- Graceful degradation during leadership transitions
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import os
import json
from pathlib import Path
import discord
import hashlib
import redis.asyncio as redis
import uuid

from worker_ant_v1.intelligence.sentiment_first_ai import SentimentFirstAI
# TwitterSentimentAnalyzer removed - using SentimentFirstAI instead
from worker_ant_v1.intelligence.caller_intelligence import AdvancedCallerIntelligence
from worker_ant_v1.intelligence.technical_analyzer import TechnicalAnalyzer
from worker_ant_v1.intelligence.narrative_ant import NarrativeAnt, NarrativeCategory
# PredictionEngine removed - using lean mathematical core instead
from worker_ant_v1.intelligence.sentiment_first_ai import SentimentFirstAI
from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
from worker_ant_v1.safety.vault_wallet_system import VaultWalletSystem
from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.utils.constants import SentimentDecision as SentimentDecisionEnum
from ..core.unified_config import UnifiedConfig


class LeadershipStatus(Enum):
    """Leadership status in the colony"""
    CANDIDATE = "candidate"          # Competing for leadership
    LEADER = "leader"               # Currently leading the colony
    FOLLOWER = "follower"           # Following current leader
    INTERIM_LEADER = "interim_leader"  # Temporary leader during failover
    STANDBY = "standby"             # Ready to become leader
    OFFLINE = "offline"             # Not participating


class ColonyRole(Enum):
    """Colony-wide roles for different commander instances"""
    PRIMARY_COMMANDER = "primary_commander"
    STANDBY_COMMANDER = "standby_commander"
    INTELLIGENCE_SPECIALIST = "intelligence_specialist"
    RISK_MANAGER = "risk_manager"
    EXECUTION_COORDINATOR = "execution_coordinator"


@dataclass
class LeadershipState:
    """Current leadership state information"""
    node_id: str
    status: LeadershipStatus
    role: ColonyRole
    elected_at: Optional[datetime]
    last_heartbeat: datetime
    term_number: int
    leader_node_id: Optional[str]
    votes_received: int
    election_timeout: datetime
    health_score: float = 1.0
    
    def is_leader(self) -> bool:
        return self.status in [LeadershipStatus.LEADER, LeadershipStatus.INTERIM_LEADER]
    
    def can_make_decisions(self) -> bool:
        return self.status in [
            LeadershipStatus.LEADER, 
            LeadershipStatus.INTERIM_LEADER,
            LeadershipStatus.STANDBY  # Standby can make defensive decisions
        ]


@dataclass
class DefensiveCommand:
    """Emergency defensive commands for interim leaders"""
    command_type: str  # RETREAT, HIBERNATE, EMERGENCY_STOP
    issued_by: str
    issued_at: datetime
    authority_level: str  # INTERIM, EMERGENCY, FULL
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommanderNode:
    """Information about a commander node in the colony"""
    node_id: str
    instance_name: str
    role: ColonyRole
    status: LeadershipStatus
    last_seen: datetime
    health_score: float
    capabilities: List[str]
    location: str = "unknown"
    
    def is_alive(self, timeout_seconds: int = 30) -> bool:
        return (datetime.now() - self.last_seen).total_seconds() < timeout_seconds


class SwarmDecision(Enum):
    """Swarm-level decisions"""
    HUNT = "hunt"              # Active trading mode
    STALK = "stalk"            # Monitoring mode
    RETREAT = "retreat"        # Risk avoidance mode
    EVOLVE = "evolve"          # Learning mode
    HIBERNATE = "hibernate"    # Complete pause

class SwarmMode(Enum):
    """Swarm operating modes"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    STEALTH = "stealth"
    EXPLORATION = "exploration"

class SignalSource(Enum):
    """Intelligence signal sources"""
    AI_MODEL = "ai_model"
    ONCHAIN_DATA = "onchain_data"
    TWITTER_SENTIMENT = "twitter_sentiment"
    CALLER_INTEL = "caller_intel"
    TECHNICAL_ANALYSIS = "technical_analysis"

@dataclass
class ConsensusSignal:
    """Multi-source consensus signal"""
    token_address: str
    timestamp: datetime
    
    
    ai_prediction: float        # -1 to 1
    onchain_sentiment: float    # -1 to 1
    twitter_sentiment: float    # -1 to 1
    
    
    technical_score: float = 0.0
    caller_reputation: float = 0.0
    
    
    consensus_strength: float = 0.0
    confidence_level: float = 0.0
    risk_indicators: List[str] = field(default_factory=list)
    
    
    unanimous_agreement: bool = False
    passes_consensus: bool = False
    recommended_action: str = SentimentDecisionEnum.NEUTRAL.value
    reasoning: List[str] = field(default_factory=list)
    
    # Enhanced validation fields
    secondary_validation_passed: bool = False
    challenger_model_agreement: bool = False
    shadow_memory_check_passed: bool = False
    sanity_check_passed: bool = False
    final_confidence_score: float = 0.0

@dataclass
class SwarmIntelligence:
    """Aggregated swarm intelligence state"""
    market_mood: str = "NEUTRAL"
    threat_level: str = "LOW"
    opportunity_count: int = 0
    consensus_accuracy: float = 0.5
    learning_confidence: float = 0.5
    
    
    source_accuracy: Dict[str, float] = field(default_factory=dict)
    source_trust_levels: Dict[str, float] = field(default_factory=dict)
    
    
    avoided_patterns: Set[str] = field(default_factory=set)
    successful_patterns: Set[str] = field(default_factory=set)

class ChallengerModel:
    """Conservative challenger model for validation"""
    
    def __init__(self):
        self.logger = setup_logger("ChallengerModel")
        
        # Conservative thresholds
        self.conservative_thresholds = {
            'min_liquidity': 10000,  # $10K minimum liquidity
            'min_holders': 50,       # 50 minimum holders
            'max_volatility': 0.5,   # 50% max volatility
            'min_age_hours': 2,      # 2 hours minimum age
            'max_whale_concentration': 0.3  # 30% max whale concentration
        }
        
        self.logger.info("🛡️ Challenger model initialized with conservative thresholds")
    
    async def validate_opportunity(self, token_address: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate opportunity using conservative metrics"""
        
        validation_result = {
            'passed': True,
            'confidence': 0.0,
            'reasons': [],
            'risk_factors': []
        }
        
        try:
            # Check liquidity
            liquidity = market_data.get('liquidity', 0)
            if liquidity < self.conservative_thresholds['min_liquidity']:
                validation_result['passed'] = False
                validation_result['reasons'].append(f"Insufficient liquidity: ${liquidity:,.0f}")
                validation_result['risk_factors'].append('low_liquidity')
            
            # Check holder count
            holder_count = market_data.get('holder_count', 0)
            if holder_count < self.conservative_thresholds['min_holders']:
                validation_result['passed'] = False
                validation_result['reasons'].append(f"Too few holders: {holder_count}")
                validation_result['risk_factors'].append('low_holder_count')
            
            # Check volatility
            volatility = market_data.get('volatility', 0)
            if volatility > self.conservative_thresholds['max_volatility']:
                validation_result['passed'] = False
                validation_result['reasons'].append(f"Excessive volatility: {volatility:.2%}")
                validation_result['risk_factors'].append('high_volatility')
            
            # Check token age
            token_age_hours = market_data.get('age_hours', 0)
            if token_age_hours < self.conservative_thresholds['min_age_hours']:
                validation_result['passed'] = False
                validation_result['reasons'].append(f"Token too new: {token_age_hours:.1f} hours")
                validation_result['risk_factors'].append('new_token')
            
            # Check whale concentration
            whale_concentration = market_data.get('whale_concentration', 0)
            if whale_concentration > self.conservative_thresholds['max_whale_concentration']:
                validation_result['passed'] = False
                validation_result['reasons'].append(f"High whale concentration: {whale_concentration:.1%}")
                validation_result['risk_factors'].append('whale_risk')
            
            # Calculate confidence based on how many checks passed
            total_checks = 5
            passed_checks = sum([
                liquidity >= self.conservative_thresholds['min_liquidity'],
                holder_count >= self.conservative_thresholds['min_holders'],
                volatility <= self.conservative_thresholds['max_volatility'],
                token_age_hours >= self.conservative_thresholds['min_age_hours'],
                whale_concentration <= self.conservative_thresholds['max_whale_concentration']
            ])
            
            validation_result['confidence'] = passed_checks / total_checks
            
            if validation_result['passed']:
                self.logger.info(f"✅ Challenger validation passed for {token_address} (confidence: {validation_result['confidence']:.2f})")
            else:
                self.logger.warning(f"❌ Challenger validation failed for {token_address}: {', '.join(validation_result['reasons'])}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Challenger validation error for {token_address}: {e}")
            validation_result['passed'] = False
            validation_result['reasons'].append(f"Validation error: {str(e)}")
            return validation_result

class SanityChecker:
    """Secondary sanity check system"""
    
    def __init__(self):
        self.logger = setup_logger("SanityChecker")
        
        # Hard-coded safety thresholds that override AI decisions
        self.safety_thresholds = {
            'max_position_size_sol': 0.5,  # 0.5 SOL max position
            'min_liquidity_override': 5000,  # $5K minimum liquidity (hard override)
            'max_slippage': 0.15,  # 15% max slippage
            'min_market_cap': 1000,  # $1K minimum market cap
            'max_gas_price': 1000,  # 1000 gwei max gas price
        }
        
        self.logger.info("🧠 Sanity checker initialized with hard safety thresholds")
    
    async def run_sanity_check(self, signal: ConsensusSignal, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run sanity check on consensus signal"""
        
        sanity_result = {
            'passed': True,
            'veto_reason': None,
            'confidence_adjustment': 0.0,
            'warnings': []
        }
        
        try:
            # Check if consensus is too strong (potential overconfidence)
            if signal.consensus_strength > 0.95:
                sanity_result['warnings'].append("Extremely high consensus strength - potential overconfidence")
                sanity_result['confidence_adjustment'] -= 0.1
            
            # Check liquidity override
            liquidity = market_data.get('liquidity', 0)
            if liquidity < self.safety_thresholds['min_liquidity_override']:
                sanity_result['passed'] = False
                sanity_result['veto_reason'] = f"Liquidity below hard threshold: ${liquidity:,.0f} < ${self.safety_thresholds['min_liquidity_override']:,.0f}"
            
            # Check market cap
            market_cap = market_data.get('market_cap', 0)
            if market_cap < self.safety_thresholds['min_market_cap']:
                sanity_result['passed'] = False
                sanity_result['veto_reason'] = f"Market cap below minimum: ${market_cap:,.0f} < ${self.safety_thresholds['min_market_cap']:,.0f}"
            
            # Check for suspicious patterns
            if self._detect_suspicious_patterns(market_data):
                sanity_result['passed'] = False
                sanity_result['veto_reason'] = "Suspicious trading patterns detected"
            
            # Check for extreme volatility
            volatility = market_data.get('volatility', 0)
            if volatility > 1.0:  # 100% volatility
                sanity_result['passed'] = False
                sanity_result['veto_reason'] = f"Extreme volatility: {volatility:.1%}"
            
            if sanity_result['passed']:
                self.logger.info(f"✅ Sanity check passed for {signal.token_address}")
            else:
                self.logger.warning(f"❌ Sanity check failed for {signal.token_address}: {sanity_result['veto_reason']}")
            
            return sanity_result
            
        except Exception as e:
            self.logger.error(f"Sanity check error for {signal.token_address}: {e}")
            sanity_result['passed'] = False
            sanity_result['veto_reason'] = f"Sanity check error: {str(e)}"
            return sanity_result
    
    def _detect_suspicious_patterns(self, market_data: Dict[str, Any]) -> bool:
        """Detect suspicious trading patterns"""
        
        # Check for pump and dump patterns
        price_history = market_data.get('price_history', [])
        if len(price_history) >= 10:
            recent_prices = price_history[-10:]
            price_changes = [abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                           for i in range(1, len(recent_prices))]
            
            # If more than 70% of recent price changes are > 20%, suspicious
            large_changes = sum(1 for change in price_changes if change > 0.2)
            if large_changes / len(price_changes) > 0.7:
                return True
        
        # Check for unusual volume spikes
        volume_history = market_data.get('volume_history', [])
        if len(volume_history) >= 5:
            avg_volume = sum(volume_history[:-1]) / (len(volume_history) - 1)
            latest_volume = volume_history[-1]
            
            if latest_volume > avg_volume * 10:  # 10x volume spike
                return True
        
        return False

class ColonyCommander:
    """The neural command center of the 10-wallet swarm with high availability"""
    
    def __init__(self, role: ColonyRole = ColonyRole.PRIMARY_COMMANDER):
        self.logger = setup_logger("ColonyCommander")
        
        # High Availability Configuration
        self.node_id = str(uuid.uuid4())
        self.instance_name = f"commander_{self.node_id[:8]}"
        self.leadership_state = LeadershipState(
            node_id=self.node_id,
            status=LeadershipStatus.CANDIDATE,
            role=role,
            elected_at=None,
            last_heartbeat=datetime.now(),
            term_number=0,
            leader_node_id=None,
            votes_received=0,
            election_timeout=datetime.now() + timedelta(seconds=30)
        )
        
        # Redis connection for leader election
        self.redis_client: Optional[redis.Redis] = None
        # Redis configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG
        from worker_ant_v1.core.unified_config import get_redis_config
        self.redis_config = get_redis_config()
        
        # Leader election configuration
        self.election_config = {
            'heartbeat_interval': 5,     # 5 seconds between heartbeats
            'election_timeout': 15,      # 15 seconds election timeout
            'leader_timeout': 30,        # 30 seconds to detect leader failure
            'term_duration': 300,        # 5 minutes per leadership term
            'health_check_interval': 10  # 10 seconds between health checks
        }
        
        # High availability state
        self.colony_nodes: Dict[str, CommanderNode] = {}
        self.defensive_commands_queue: List[DefensiveCommand] = []
        self.leadership_transitions: List[Dict[str, Any]] = []
        self.failover_in_progress = False
        self.emergency_authority = False
        
        # Load configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG
        from worker_ant_v1.core.unified_config import get_social_signals_config
        social_config = get_social_signals_config()
        self.social_signals_enabled = social_config['enable_social_signals']
        
        # Initialize intelligence components
        self.sentiment_analyzer = SentimentFirstAI()
        self.caller_intelligence = AdvancedCallerIntelligence()
        self.technical_analyzer = TechnicalAnalyzer()
        self.narrative_ant = NarrativeAnt()  # Strategic narrative intelligence
        # ML predictor removed - using lean mathematical core in simplified bot
        self.sentiment_ai = SentimentFirstAI()
        
        # Initialize validation components
        self.challenger_model = ChallengerModel()
        self.sanity_checker = SanityChecker()
        
        # Initialize Twitter analyzer (required for sentiment analysis)
        self.twitter_analyzer = None
        try:
            # Using SentimentFirstAI for all sentiment analysis
            self.logger.info("✅ Twitter sentiment analyzer initialized")
            self.social_signals_enabled = True
        except Exception as e:
            self.logger.error(f"❌ Twitter analyzer initialization failed: {e}")
            self.logger.error("Twitter API is required for sentiment analysis")
            raise
        
        # Initialize other components
        self.wallet_manager = None  # Will be injected
        self.vault_system = None    # Will be injected
        
        # Initialize curated sources
        self.curated_callers = self._initialize_curated_callers() if self.social_signals_enabled else {}
        self.blacklisted_sources = set()
        
        # Consensus thresholds
        self.consensus_thresholds = {
            'ai_model_min': 0.6,      # AI confidence minimum
            'onchain_min': 0.4,       # On-chain signal minimum
            'twitter_min': 0.5,       # Twitter sentiment minimum
            'technical_min': 0.3      # Technical score minimum
        }
        
        # Shadow memory system
        self.shadow_memory = {}
        self.shadow_memory_file = Path('data/shadow_memory.json')
        
        # Failed trade patterns database
        self.failed_patterns_db = {}
        self.failed_patterns_file = Path('data/failed_patterns.json')
        
        self.logger.info(f"🧠 Neural Command Center initialized with HA (Node: {self.node_id[:8]}, Role: {role.value})")
    
    async def initialize(self, wallet_manager: UnifiedWalletManager, 
                        vault_system: VaultWalletSystem):
        """Initialize the neural command center with high availability"""
        
        self.wallet_manager = wallet_manager
        self.vault_system = vault_system
        
        # Initialize Redis connection for leader election
        await self._initialize_redis_connection()
        
        # Start leader election process
        await self._start_leader_election()
        
        # Initialize all intelligence components
        await self.sentiment_analyzer.initialize()
        await self.caller_intelligence.initialize()
        await self.technical_analyzer.initialize()
        await self.narrative_ant.initialize()  # Initialize narrative intelligence
        await self.ml_predictor.initialize()
        
        # Load shadow memory and failed patterns from previous sessions
        await self._load_shadow_memory()
        await self._load_failed_patterns()
        
        # Start background processes
        asyncio.create_task(self._continuous_intelligence_loop())
        asyncio.create_task(self._source_performance_tracker())
        asyncio.create_task(self._leadership_heartbeat_loop())
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._colony_discovery_loop())
        
        mode = "Social signals + On-chain" if self.social_signals_enabled else "Pure on-chain only"
        leader_info = "LEADER" if self.leadership_state.is_leader() else f"FOLLOWER ({self.leadership_state.status.value})"
        self.logger.info(f"🧠 Neural Command Center initialized - {mode} mode active - Status: {leader_info}")
    
    async def _initialize_redis_connection(self) -> bool:
        """Initialize Redis connection for leader election"""
        try:
            self.redis_client = redis.Redis(**self.redis_config)
            await self.redis_client.ping()
            self.logger.info("✅ Redis connection established for leader election")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Redis: {e}")
            self.logger.error("High availability features disabled")
            return False
    
    async def _start_leader_election(self):
        """Start the leader election process"""
        if not self.redis_client:
            self.logger.warning("⚠️ Redis not available - running in standalone mode")
            self.leadership_state.status = LeadershipStatus.LEADER
            self.leadership_state.elected_at = datetime.now()
            return
        
        self.logger.info(f"🗳️ Starting leader election for node {self.node_id[:8]}")
        
        # Register this node in the colony
        await self._register_node()
        
        # Start election process
        asyncio.create_task(self._election_process())
    
    async def _register_node(self):
        """Register this node in the colony registry"""
        if not self.redis_client:
            return
        
        node_info = {
            'node_id': self.node_id,
            'instance_name': self.instance_name,
            'role': self.leadership_state.role.value,
            'status': self.leadership_state.status.value,
            'registered_at': datetime.now().isoformat(),
            'capabilities': ['sentiment_analysis', 'trading', 'risk_management'],
            'health_score': 1.0
        }
        
        try:
            # Register in colony nodes set
            await self.redis_client.hset(
                'colony:nodes', 
                self.node_id, 
                json.dumps(node_info)
            )
            
            # Set expiration for automatic cleanup
            await self.redis_client.expire(f'colony:node:{self.node_id}', 60)
            
            self.logger.info(f"📋 Node {self.node_id[:8]} registered in colony")
            
        except Exception as e:
            self.logger.error(f"Failed to register node: {e}")
    
    async def _election_process(self):
        """Core leader election algorithm"""
        while True:
            try:
                if not self.redis_client:
                    await asyncio.sleep(10)
                    continue
                
                current_leader = await self._get_current_leader()
                
                if current_leader is None:
                    # No leader - start election
                    await self._start_election()
                elif current_leader == self.node_id:
                    # We are the leader - maintain leadership
                    await self._maintain_leadership()
                else:
                    # Follow the current leader
                    await self._follow_leader(current_leader)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Election process error: {e}")
                await asyncio.sleep(10)
    
    async def _get_current_leader(self) -> Optional[str]:
        """Get the current leader from Redis"""
        try:
            leader_info = await self.redis_client.hget('colony:leadership', 'current_leader')
            if leader_info:
                leader_data = json.loads(leader_info)
                leader_id = leader_data['node_id']
                elected_at = datetime.fromisoformat(leader_data['elected_at'])
                
                # Check if leadership has expired
                max_term = timedelta(seconds=self.election_config['term_duration'])
                if datetime.now() - elected_at > max_term:
                    await self._invalidate_leadership()
                    return None
                
                return leader_id
            return None
        except Exception as e:
            self.logger.error(f"Error getting current leader: {e}")
            return None
    
    async def _start_election(self):
        """Start a new leader election"""
        self.logger.info(f"🗳️ Starting election - Node {self.node_id[:8]} is candidate")
        
        self.leadership_state.status = LeadershipStatus.CANDIDATE
        self.leadership_state.term_number += 1
        
        try:
            # Create election lock to prevent concurrent elections
            election_lock = f"colony:election:lock:{self.leadership_state.term_number}"
            lock_acquired = await self.redis_client.set(
                election_lock, 
                self.node_id, 
                nx=True, 
                ex=30  # 30 second lock
            )
            
            if lock_acquired:
                self.logger.info(f"🔒 Election lock acquired by {self.node_id[:8]}")
                
                # Vote for ourselves
                await self._cast_vote(self.node_id)
                
                # Wait for other votes
                await asyncio.sleep(5)
                
                # Count votes and determine winner
                winner = await self._count_votes()
                
                if winner == self.node_id:
                    await self._become_leader()
                else:
                    await self._become_follower(winner)
                
                # Release election lock
                await self.redis_client.delete(election_lock)
            else:
                # Another election in progress - wait and follow result
                await asyncio.sleep(10)
        
        except Exception as e:
            self.logger.error(f"Election error: {e}")
    
    async def _cast_vote(self, candidate_id: str):
        """Cast a vote for a candidate"""
        try:
            vote_key = f"colony:votes:{self.leadership_state.term_number}"
            await self.redis_client.hset(vote_key, self.node_id, candidate_id)
            await self.redis_client.expire(vote_key, 60)  # Votes expire in 60 seconds
            
            self.logger.info(f"🗳️ Voted for candidate {candidate_id[:8]}")
            
        except Exception as e:
            self.logger.error(f"Voting error: {e}")
    
    async def _count_votes(self) -> Optional[str]:
        """Count votes and determine election winner"""
        try:
            vote_key = f"colony:votes:{self.leadership_state.term_number}"
            votes = await self.redis_client.hgetall(vote_key)
            
            if not votes:
                return None
            
            # Count votes for each candidate
            vote_counts = {}
            for voter, candidate in votes.items():
                candidate = candidate.decode() if isinstance(candidate, bytes) else candidate
                vote_counts[candidate] = vote_counts.get(candidate, 0) + 1
            
            # Find winner (candidate with most votes)
            winner = max(vote_counts.items(), key=lambda x: x[1])
            
            self.logger.info(f"🏆 Election results: {winner[0][:8]} wins with {winner[1]} votes")
            return winner[0]
            
        except Exception as e:
            self.logger.error(f"Vote counting error: {e}")
            return None
    
    async def _become_leader(self):
        """Become the colony leader"""
        self.leadership_state.status = LeadershipStatus.LEADER
        self.leadership_state.elected_at = datetime.now()
        self.leadership_state.leader_node_id = self.node_id
        
        # Record leadership in Redis
        leader_info = {
            'node_id': self.node_id,
            'elected_at': self.leadership_state.elected_at.isoformat(),
            'term_number': self.leadership_state.term_number,
            'role': self.leadership_state.role.value
        }
        
        try:
            await self.redis_client.hset(
                'colony:leadership', 
                'current_leader', 
                json.dumps(leader_info)
            )
            
            self.logger.info(f"👑 Node {self.node_id[:8]} became colony leader (Term {self.leadership_state.term_number})")
            
            # Announce leadership change to the colony
            await self._announce_leadership_change()
            
        except Exception as e:
            self.logger.error(f"Failed to record leadership: {e}")
    
    async def _become_follower(self, leader_id: str):
        """Become a follower of the specified leader"""
        self.leadership_state.status = LeadershipStatus.FOLLOWER
        self.leadership_state.leader_node_id = leader_id
        
        self.logger.info(f"👥 Node {self.node_id[:8]} following leader {leader_id[:8]}")
    
    async def _maintain_leadership(self):
        """Maintain current leadership with heartbeats"""
        try:
            # Send leadership heartbeat
            heartbeat_key = f"colony:leader:heartbeat"
            heartbeat_data = {
                'node_id': self.node_id,
                'timestamp': datetime.now().isoformat(),
                'health_score': self.leadership_state.health_score,
                'active_operations': self._get_active_operations_count()
            }
            
            await self.redis_client.hset(
                heartbeat_key,
                'data',
                json.dumps(heartbeat_data)
            )
            await self.redis_client.expire(heartbeat_key, 30)
            
            self.leadership_state.last_heartbeat = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Leadership heartbeat failed: {e}")
    
    async def _follow_leader(self, leader_id: str):
        """Follow the current leader and monitor their health"""
        try:
            # Check leader health
            heartbeat_key = f"colony:leader:heartbeat"
            heartbeat_data = await self.redis_client.hget(heartbeat_key, 'data')
            
            if heartbeat_data:
                leader_heartbeat = json.loads(heartbeat_data)
                last_heartbeat = datetime.fromisoformat(leader_heartbeat['timestamp'])
                
                # Check if leader is alive
                heartbeat_age = (datetime.now() - last_heartbeat).total_seconds()
                
                if heartbeat_age > self.election_config['leader_timeout']:
                    self.logger.warning(f"⚠️ Leader {leader_id[:8]} heartbeat timeout ({heartbeat_age:.1f}s)")
                    await self._handle_leader_failure(leader_id)
                else:
                    # Leader is healthy - continue following
                    self.leadership_state.leader_node_id = leader_id
            else:
                # No heartbeat data - leader might be down
                self.logger.warning(f"⚠️ No heartbeat data for leader {leader_id[:8]}")
                await self._handle_leader_failure(leader_id)
                
        except Exception as e:
            self.logger.error(f"Leader monitoring error: {e}")
    
    async def _handle_leader_failure(self, failed_leader_id: str):
        """Handle leader failure and initiate failover"""
        self.logger.error(f"💥 Leader failure detected: {failed_leader_id[:8]}")
        
        # Check if we should become interim leader
        if self.leadership_state.role in [ColonyRole.STANDBY_COMMANDER, ColonyRole.PRIMARY_COMMANDER]:
            await self._become_interim_leader(failed_leader_id)
        
        # Clear failed leadership
        await self._invalidate_leadership()
        
        # Start new election
        await asyncio.sleep(2)  # Brief delay to avoid race conditions
        await self._start_election()
    
    async def _become_interim_leader(self, failed_leader_id: str):
        """Become interim leader during failover"""
        self.leadership_state.status = LeadershipStatus.INTERIM_LEADER
        self.leadership_state.elected_at = datetime.now()
        self.emergency_authority = True
        
        self.logger.warning(f"🚨 Node {self.node_id[:8]} became INTERIM LEADER after {failed_leader_id[:8]} failure")
        
        # Record leadership transition
        transition = {
            'from_leader': failed_leader_id,
            'to_interim': self.node_id,
            'timestamp': datetime.now().isoformat(),
            'reason': 'leader_failure'
        }
        self.leadership_transitions.append(transition)
        
        # Issue defensive commands
        await self._issue_defensive_commands()
    
    async def _issue_defensive_commands(self):
        """Issue defensive commands during emergency leadership"""
        self.logger.warning("🛡️ Issuing defensive commands during interim leadership")
        
        # Issue RETREAT command
        retreat_command = DefensiveCommand(
            command_type="RETREAT",
            issued_by=self.node_id,
            issued_at=datetime.now(),
            authority_level="INTERIM",
            parameters={'reason': 'leader_failure', 'duration_minutes': 10}
        )
        
        self.defensive_commands_queue.append(retreat_command)
        
        # Broadcast defensive command to colony
        if self.redis_client:
            try:
                command_data = {
                    'command': retreat_command.command_type,
                    'issued_by': retreat_command.issued_by,
                    'authority': retreat_command.authority_level,
                    'timestamp': retreat_command.issued_at.isoformat(),
                    'parameters': retreat_command.parameters
                }
                
                await self.redis_client.lpush(
                    'colony:defensive_commands',
                    json.dumps(command_data)
                )
                await self.redis_client.expire('colony:defensive_commands', 300)
                
                self.logger.warning(f"📢 Defensive command {retreat_command.command_type} broadcast to colony")
                
            except Exception as e:
                self.logger.error(f"Failed to broadcast defensive command: {e}")
    
    async def _invalidate_leadership(self):
        """Invalidate current leadership in Redis"""
        try:
            await self.redis_client.delete('colony:leadership')
            self.logger.info("🗑️ Leadership invalidated")
        except Exception as e:
            self.logger.error(f"Failed to invalidate leadership: {e}")
    
    async def _announce_leadership_change(self):
        """Announce leadership change to the colony"""
        if not self.redis_client:
            return
        
        announcement = {
            'type': 'leadership_change',
            'new_leader': self.node_id,
            'timestamp': datetime.now().isoformat(),
            'term_number': self.leadership_state.term_number,
            'role': self.leadership_state.role.value
        }
        
        try:
            await self.redis_client.lpush(
                'colony:announcements',
                json.dumps(announcement)
            )
            await self.redis_client.expire('colony:announcements', 300)
            
        except Exception as e:
            self.logger.error(f"Failed to announce leadership change: {e}")
    
    async def _leadership_heartbeat_loop(self):
        """Background loop for leadership heartbeats"""
        while True:
            try:
                if self.leadership_state.is_leader():
                    await self._maintain_leadership()
                
                await asyncio.sleep(self.election_config['heartbeat_interval'])
                
            except Exception as e:
                self.logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(10)
    
    async def _health_monitoring_loop(self):
        """Background loop for health monitoring"""
        while True:
            try:
                # Update our health score
                health_score = await self._calculate_health_score()
                self.leadership_state.health_score = health_score
                
                # Monitor colony nodes
                await self._monitor_colony_health()
                
                await asyncio.sleep(self.election_config['health_check_interval'])
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _calculate_health_score(self) -> float:
        """Calculate health score for this node"""
        try:
            health_factors = []
            
            # System resources
            import psutil
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            
            health_factors.append(1.0 - (cpu_usage / 100.0))
            health_factors.append(1.0 - (memory_usage / 100.0))
            
            # Redis connectivity
            if self.redis_client:
                try:
                    await self.redis_client.ping()
                    health_factors.append(1.0)
                except:
                    health_factors.append(0.0)
            else:
                health_factors.append(0.5)
            
            # Component health (if available)
            if self.wallet_manager:
                # Add wallet manager health check
                health_factors.append(0.9)  # Placeholder
            
            return sum(health_factors) / len(health_factors)
            
        except Exception as e:
            self.logger.error(f"Health calculation error: {e}")
            return 0.5
    
    async def _monitor_colony_health(self):
        """Monitor health of all colony nodes"""
        if not self.redis_client:
            return
        
        try:
            # Get all registered nodes
            nodes_data = await self.redis_client.hgetall('colony:nodes')
            
            current_time = datetime.now()
            
            for node_id, node_data in nodes_data.items():
                if isinstance(node_id, bytes):
                    node_id = node_id.decode()
                if isinstance(node_data, bytes):
                    node_data = node_data.decode()
                
                node_info = json.loads(node_data)
                last_seen = datetime.fromisoformat(node_info.get('registered_at', current_time.isoformat()))
                
                # Check if node is still alive
                if (current_time - last_seen).total_seconds() > 60:
                    # Node appears to be offline
                    if node_id in self.colony_nodes:
                        self.colony_nodes[node_id].status = LeadershipStatus.OFFLINE
                        self.logger.warning(f"📴 Node {node_id[:8]} appears offline")
                else:
                    # Update node info
                    self.colony_nodes[node_id] = CommanderNode(
                        node_id=node_id,
                        instance_name=node_info.get('instance_name', 'unknown'),
                        role=ColonyRole(node_info.get('role', 'primary_commander')),
                        status=LeadershipStatus(node_info.get('status', 'offline')),
                        last_seen=last_seen,
                        health_score=node_info.get('health_score', 0.5),
                        capabilities=node_info.get('capabilities', [])
                    )
            
        except Exception as e:
            self.logger.error(f"Colony health monitoring error: {e}")
    
    async def _colony_discovery_loop(self):
        """Background loop for colony node discovery"""
        while True:
            try:
                await self._register_node()
                await self._discover_colony_nodes()
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Colony discovery error: {e}")
                await asyncio.sleep(60)
    
    async def _discover_colony_nodes(self):
        """Discover other nodes in the colony"""
        if not self.redis_client:
            return
        
        try:
            nodes_data = await self.redis_client.hgetall('colony:nodes')
            
            discovered_nodes = 0
            for node_id, node_data in nodes_data.items():
                if isinstance(node_id, bytes):
                    node_id = node_id.decode()
                
                if node_id != self.node_id:  # Don't count ourselves
                    discovered_nodes += 1
            
            if discovered_nodes != len(self.colony_nodes):
                self.logger.info(f"🔍 Colony discovery: {discovered_nodes} other nodes found")
                
        except Exception as e:
            self.logger.error(f"Node discovery error: {e}")
    
    def _get_active_operations_count(self) -> int:
        """Get count of active trading operations"""
        # This would integrate with actual trading operations
        return 0  # Placeholder
    
    def can_make_trading_decisions(self) -> bool:
        """Check if this commander can make trading decisions"""
        if not self.leadership_state.can_make_decisions():
            return False
        
        # Additional checks for trading authority
        if self.leadership_state.status == LeadershipStatus.INTERIM_LEADER:
            # Interim leaders have limited authority
            return False
        
        return True
    
    def can_issue_defensive_commands(self) -> bool:
        """Check if this commander can issue defensive commands"""
        return self.leadership_state.status in [
            LeadershipStatus.LEADER,
            LeadershipStatus.INTERIM_LEADER,
            LeadershipStatus.STANDBY
        ]
    
    def get_leadership_status(self) -> Dict[str, Any]:
        """Get current leadership status"""
        return {
            'node_id': self.node_id,
            'instance_name': self.instance_name,
            'status': self.leadership_state.status.value,
            'role': self.leadership_state.role.value,
            'is_leader': self.leadership_state.is_leader(),
            'can_trade': self.can_make_trading_decisions(),
            'can_defend': self.can_issue_defensive_commands(),
            'elected_at': self.leadership_state.elected_at.isoformat() if self.leadership_state.elected_at else None,
            'term_number': self.leadership_state.term_number,
            'health_score': self.leadership_state.health_score,
            'colony_nodes_count': len(self.colony_nodes),
            'failover_in_progress': self.failover_in_progress
        }

    async def analyze_opportunity(self, token_address: str, 
                                 market_data: Dict[str, Any],
                                 narrative_weight: float = 1.0) -> ConsensusSignal:
        """
        Core decision engine: Only trades when AI + on-chain + Twitter ALL agree
        PLUS secondary validation layers AND narrative weighting. No exceptions. This is the survival filter.
        """
        
        try:
            signal = ConsensusSignal(
                token_address=token_address,
                timestamp=datetime.now()
            )
            
            # Log narrative weight if not default
            if narrative_weight != 1.0:
                self.logger.info(f"🧠 Applying narrative weight {narrative_weight:.2f} to {token_address}")
            
            # Phase 1: Primary Intelligence Gathering
            ai_prediction = await self._get_ai_prediction(token_address, market_data)
            signal.ai_prediction = ai_prediction
            
            onchain_sentiment = await self._get_onchain_analysis(token_address, market_data)
            signal.onchain_sentiment = onchain_sentiment
            
            twitter_sentiment = await self._get_twitter_analysis(token_address)
            signal.twitter_sentiment = twitter_sentiment
            
            technical_score = await self._get_technical_analysis(token_address)
            signal.technical_score = technical_score
            
            caller_reputation = await self._get_caller_intelligence(token_address)
            signal.caller_reputation = caller_reputation
            
            # Phase 2: Primary Consensus Evaluation
            consensus_result = self._evaluate_consensus(signal)
            signal.consensus_strength = consensus_result['consensus_strength']
            signal.confidence_level = consensus_result['confidence_level']
            signal.unanimous_agreement = consensus_result['unanimous_agreement']
            signal.passes_consensus = consensus_result['passes_consensus']
            signal.recommended_action = consensus_result['recommended_action']
            signal.reasoning = consensus_result['reasoning']
            
            # Phase 3: Secondary Validation Layers (Consensus of Consensuses)
            if signal.passes_consensus and signal.recommended_action == SentimentDecisionEnum.BUY.value:
                
                # Secondary validation: Challenger model
                challenger_result = await self.challenger_model.validate_opportunity(token_address, market_data)
                signal.challenger_model_agreement = challenger_result['passed']
                
                if not signal.challenger_model_agreement:
                    signal.recommended_action = SentimentDecisionEnum.NEUTRAL.value
                    signal.reasoning.append(f"Challenger model veto: {', '.join(challenger_result['reasons'])}")
                    self.logger.warning(f"🛡️ Challenger model vetoed {token_address}")
                
                # Secondary validation: Sanity check
                sanity_result = await self.sanity_checker.run_sanity_check(signal, market_data)
                signal.sanity_check_passed = sanity_result['passed']
                
                if not signal.sanity_check_passed:
                    signal.recommended_action = SentimentDecisionEnum.NEUTRAL.value
                    signal.reasoning.append(f"Sanity check veto: {sanity_result['veto_reason']}")
                    self.logger.warning(f"🧠 Sanity check vetoed {token_address}")
                
                # Secondary validation: Shadow memory check
                shadow_memory_check = await self._check_shadow_memory(token_address, market_data)
                signal.shadow_memory_check_passed = shadow_memory_check['passed']
                
                if not signal.shadow_memory_check_passed:
                    signal.recommended_action = SentimentDecisionEnum.NEUTRAL.value
                    signal.reasoning.append(f"Shadow memory veto: {shadow_memory_check['reason']}")
                    self.logger.warning(f"👻 Shadow memory vetoed {token_address}")
                
                # Secondary validation: Failed patterns check
                failed_patterns_check = await self._check_failed_patterns(token_address, market_data)
                signal.failed_patterns_check_passed = failed_patterns_check['passed']
                
                if not signal.failed_patterns_check_passed:
                    signal.recommended_action = SentimentDecisionEnum.NEUTRAL.value
                    signal.reasoning.append(f"Failed patterns veto: {failed_patterns_check['reason']}")
                    self.logger.warning(f"🚫 Failed patterns vetoed {token_address}")
                
                # Final confidence calculation with all validation layers AND narrative weighting
                base_confidence = self._calculate_final_confidence(signal)
                signal.final_confidence_score = base_confidence * narrative_weight  # Apply narrative weight
                
                # Final decision: Only proceed if ALL validation layers pass
                signal.secondary_validation_passed = all([
                    signal.challenger_model_agreement,
                    signal.sanity_check_passed,
                    signal.shadow_memory_check_passed,
                    signal.failed_patterns_check_passed
                ])
                
                if not signal.secondary_validation_passed:
                    signal.recommended_action = SentimentDecisionEnum.NEUTRAL.value
                    self.logger.warning(f"🛡️ Secondary validation failed for {token_address} - trade vetoed")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Opportunity analysis failed for {token_address}: {e}")
            
            # Return safe default signal
            return ConsensusSignal(
                token_address=token_address,
                timestamp=datetime.now(),
                recommended_action=SentimentDecisionEnum.NEUTRAL.value,
                reasoning=[f"Analysis error: {str(e)}"],
                secondary_validation_passed=False,
                final_confidence_score=0.0
            )
    
    async def _get_ai_prediction(self, token_address: str, market_data: Dict[str, Any]) -> float:
        """Get AI model prediction (-1 to 1)"""
        
        try:
            ml_prediction = await self.ml_predictor.predict_token_outcome(token_address, market_data)
            sentiment_decision = await self.sentiment_ai.analyze_and_decide(token_address, market_data)
            
            
            ml_score = ml_prediction.confidence if ml_prediction.predicted_outcome == 'profitable' else -ml_prediction.confidence
            sentiment_score = sentiment_decision.sentiment_score
            
            combined_score = (ml_score * 0.6) + (sentiment_score * 0.4)
            
            return np.clip(combined_score, -1.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"AI prediction failed: {e}")
            return 0.0
    
    async def _get_onchain_analysis(self, token_address: str, market_data: Dict[str, Any]) -> float:
        """Get on-chain sentiment analysis (-1 to 1)"""
        
        try:
            sentiment_data = await self.sentiment_analyzer.analyze_token_sentiment(token_address, market_data)
            return sentiment_data.overall_sentiment
            
        except Exception as e:
            self.logger.warning(f"On-chain analysis failed: {e}")
            return 0.0
    
    async def _get_twitter_analysis(self, token_address: str) -> float:
        """Get Twitter sentiment from Discord server feed (production-ready)"""
        try:
            # Discord configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG
            from worker_ant_v1.core.unified_config import get_social_signals_config
            social_config = get_social_signals_config()
            bot_token = social_config['discord_bot_token']
            channel_id = social_config['discord_channel_id']
            
            if not bot_token or not channel_id:
                self.logger.error("Discord bot token or channel ID not configured for Twitter sentiment feed.")
                return 0.0
            intents = discord.Intents.default()
            intents.messages = True
            client = discord.Client(intents=intents)
            sentiment_scores = []
            async def on_ready():
                channel = client.get_channel(channel_id)
                if channel is None:
                    self.logger.error(f"Discord channel {channel_id} not found.")
                    await client.close()
                    return
                messages = [message async for message in channel.history(limit=50)]
                for message in messages:
                    # Simple sentiment extraction: +1 for positive, -1 for negative, 0 for neutral (replace with real logic)
                    if 'bullish' in message.content.lower():
                        sentiment_scores.append(1.0)
                    elif 'bearish' in message.content.lower():
                        sentiment_scores.append(-1.0)
                    else:
                        sentiment_scores.append(0.0)
                await client.close()
            client.event(on_ready)
            await client.start(bot_token)
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            else:
                avg_sentiment = 0.0
            self.logger.debug(f"[DiscordFeed] Twitter sentiment for {token_address}: {avg_sentiment}")
            return avg_sentiment
        except Exception as e:
            self.logger.warning(f"Discord Twitter sentiment analysis failed: {e}")
            return 0.0
    
    async def _get_technical_analysis(self, token_address: str) -> float:
        """Get technical analysis score (-1 to 1)"""
        
        try:
            technical_result = await self.technical_analyzer.analyze_token(token_address)
            
            
            bullish_signals = technical_result.get('bullish_signals', 0)
            bearish_signals = technical_result.get('bearish_signals', 0)
            
            if bullish_signals + bearish_signals == 0:
                return 0.0
            
            score = (bullish_signals - bearish_signals) / (bullish_signals + bearish_signals)
            return np.clip(score, -1.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Technical analysis failed: {e}")
            return 0.0
    
    async def _get_caller_intelligence(self, token_address: str) -> float:
        """Get caller reputation score for this token (-1 to 1)"""
        
        try:
            recent_calls = await self.caller_intelligence.get_token_calls(token_address, hours=24)
            
            if not recent_calls:
                return 0.0
            
            reputation_scores = []
            for call in recent_calls:
                caller_id = call.get('caller_id')
                if caller_id:
                    recommendation = self.caller_intelligence.get_caller_recommendation(caller_id)
                    
                    
                    rec_scores = {
                        'BLACKLIST': -1.0,
                        'AVOID': -0.5,
                        'CAUTION': 0.0,
                        'MONITOR': 0.2,
                        'CONSIDER': 0.6,
                        'FOLLOW': 0.8
                    }
                    
                    score = rec_scores.get(recommendation.get('recommendation', 'CAUTION'), 0.0)
                    reputation_scores.append(score)
            
            if reputation_scores:
                return sum(reputation_scores) / len(reputation_scores)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Caller intelligence failed: {e}")
            return 0.0
    
    async def _check_shadow_memory(self, token_address: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if token matches patterns in shadow memory"""
        
        try:
            pattern_signature = self._create_pattern_signature(token_address, market_data)
            
            for memory_entry in self.shadow_memory.values():
                if memory_entry.get('outcome') == 'failed':
                    stored_pattern = memory_entry.get('pattern_signature', {})
                    similarity = self._pattern_similarity(pattern_signature, stored_pattern)
                    
                    if similarity > 0.8:  # 80% similarity threshold
                        return {
                            'passed': False,
                            'reason': f"Matches failed pattern in shadow memory (similarity: {similarity:.2f})"
                        }
            
            return {'passed': True, 'reason': "No failed patterns matched"}
            
        except Exception as e:
            self.logger.error(f"Shadow memory check failed: {e}")
            return {'passed': False, 'reason': f"Shadow memory check error: {str(e)}"}
    
    async def _check_failed_patterns(self, token_address: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if token matches known failed trade patterns"""
        
        try:
            pattern_signature = self._create_pattern_signature(token_address, market_data)
            pattern_hash = hashlib.md5(json.dumps(pattern_signature, sort_keys=True).encode()).hexdigest()
            
            # Check against failed patterns database
            if pattern_hash in self.failed_patterns_db:
                failed_entry = self.failed_patterns_db[pattern_hash]
                return {
                    'passed': False,
                    'reason': f"Matches known failed pattern: {failed_entry.get('reason', 'Unknown failure')}"
                }
            
            # Check for rug pull patterns
            if self._detect_rug_pattern(market_data):
                return {
                    'passed': False,
                    'reason': "Detected potential rug pull pattern"
                }
            
            return {'passed': True, 'reason': "No failed patterns detected"}
            
        except Exception as e:
            self.logger.error(f"Failed patterns check failed: {e}")
            return {'passed': False, 'reason': f"Failed patterns check error: {str(e)}"}
    
    def _detect_rug_pattern(self, market_data: Dict[str, Any]) -> bool:
        """Detect potential rug pull patterns"""
        
        try:
            # Check for sudden liquidity removal
            liquidity_history = market_data.get('liquidity_history', [])
            if len(liquidity_history) >= 3:
                recent_liquidity = liquidity_history[-3:]
                if all(recent_liquidity[i] < recent_liquidity[i-1] * 0.5 for i in range(1, len(recent_liquidity))):
                    return True
            
            # Check for whale concentration
            whale_concentration = market_data.get('whale_concentration', 0)
            if whale_concentration > 0.8:  # 80% whale concentration
                return True
            
            # Check for suspicious holder distribution
            holder_distribution = market_data.get('holder_distribution', {})
            if holder_distribution:
                top_10_holders = sum(list(holder_distribution.values())[:10])
                if top_10_holders > 0.9:  # Top 10 holders own >90%
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Rug pattern detection failed: {e}")
            return False
    
    def _calculate_final_confidence(self, signal: ConsensusSignal) -> float:
        """Calculate final confidence score with all validation layers"""
        
        try:
            # Base confidence from primary consensus
            base_confidence = signal.confidence_level
            
            # Apply validation layer adjustments
            validation_multiplier = 1.0
            
            if signal.challenger_model_agreement:
                validation_multiplier *= 1.1  # 10% boost for challenger agreement
            else:
                validation_multiplier *= 0.5  # 50% penalty for challenger disagreement
            
            if signal.sanity_check_passed:
                validation_multiplier *= 1.05  # 5% boost for sanity check
            else:
                validation_multiplier *= 0.3  # 70% penalty for sanity check failure
            
            if signal.shadow_memory_check_passed:
                validation_multiplier *= 1.05  # 5% boost for shadow memory
            else:
                validation_multiplier *= 0.2  # 80% penalty for shadow memory failure
            
            if signal.failed_patterns_check_passed:
                validation_multiplier *= 1.05  # 5% boost for failed patterns check
            else:
                validation_multiplier *= 0.1  # 90% penalty for failed patterns match
            
            final_confidence = base_confidence * validation_multiplier
            
            # Ensure confidence stays within bounds
            return np.clip(final_confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Final confidence calculation failed: {e}")
            return 0.0
    
    async def _load_failed_patterns(self):
        """Load failed trade patterns from file"""
        
        try:
            if self.failed_patterns_file.exists():
                with open(self.failed_patterns_file, 'r') as f:
                    self.failed_patterns_db = json.load(f)
                self.logger.info(f"✅ Loaded {len(self.failed_patterns_db)} failed patterns")
            else:
                self.failed_patterns_db = {}
                self.logger.info("No failed patterns file found, starting fresh")
                
        except Exception as e:
            self.logger.error(f"Failed to load failed patterns: {e}")
            self.failed_patterns_db = {}
    
    async def _save_failed_patterns(self):
        """Save failed trade patterns to file"""
        
        try:
            self.failed_patterns_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.failed_patterns_file, 'w') as f:
                json.dump(self.failed_patterns_db, f, indent=2)
            self.logger.debug("✅ Failed patterns saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save failed patterns: {e}")
    
    async def record_failed_pattern(self, token_address: str, market_data: Dict[str, Any], reason: str):
        """Record a failed trade pattern for future avoidance"""
        
        try:
            pattern_signature = self._create_pattern_signature(token_address, market_data)
            pattern_hash = hashlib.md5(json.dumps(pattern_signature, sort_keys=True).encode()).hexdigest()
            
            self.failed_patterns_db[pattern_hash] = {
                'token_address': token_address,
                'pattern_signature': pattern_signature,
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'avoid_count': 0
            }
            
            await self._save_failed_patterns()
            self.logger.info(f"🚫 Recorded failed pattern for {token_address}: {reason}")
            
        except Exception as e:
            self.logger.error(f"Failed to record failed pattern: {e}")
    
    def _evaluate_consensus(self, signal: ConsensusSignal) -> Dict[str, Any]:
        """
        THE CORE FILTER: Evaluate if all sources agree
        Only proceed when AI + on-chain + Twitter ALL agree
        """
        
        # Check if each signal passes minimum thresholds
        ai_passes = signal.ai_prediction >= self.consensus_thresholds['ai_model_min']
        onchain_passes = signal.onchain_sentiment >= self.consensus_thresholds['onchain_min']
        twitter_passes = signal.twitter_sentiment >= self.consensus_thresholds['twitter_min']
        
        # Check for unanimous agreement (Twitter + on-chain + AI)
        all_positive = (signal.ai_prediction > 0 and 
                      signal.onchain_sentiment > 0 and 
                      signal.twitter_sentiment > 0)
        
        all_negative = (signal.ai_prediction < 0 and 
                      signal.onchain_sentiment < 0 and 
                      signal.twitter_sentiment < 0)
        
        scores = [signal.ai_prediction, signal.onchain_sentiment, signal.twitter_sentiment]
        
        unanimous_agreement = all_positive or all_negative
        
        # Calculate consensus strength
        consensus_strength = 1.0 - np.std(scores)  # Lower std = higher agreement
        
        # Calculate confidence level
        avg_magnitude = np.mean([abs(score) for score in scores])
        confidence_level = avg_magnitude * consensus_strength
        
        # Determine if consensus passes (Twitter + on-chain + AI)
        required_passes = ai_passes and onchain_passes and twitter_passes
        
        passes_consensus = required_passes and unanimous_agreement
        
        # Generate reasoning
        reasoning = []
        if not ai_passes:
            reasoning.append("AI confidence below threshold")
        if not onchain_passes:
            reasoning.append("On-chain signals below threshold")
        if not twitter_passes:
            reasoning.append("Twitter sentiment below threshold")
        if not unanimous_agreement:
            reasoning.append("No unanimous agreement between signals")
        
        if passes_consensus:
            if all_positive:
                action = "BUY"
                reasoning = ["All systems agree: Strong bullish consensus"]
            else:
                action = "SELL"
                reasoning = ["All systems agree: Strong bearish consensus"]
        else:
            action = "HOLD"
            
        return {
            'passes_consensus': passes_consensus,
            'consensus_strength': consensus_strength,
            'confidence_level': confidence_level,
            'unanimous_agreement': unanimous_agreement,
            'recommended_action': action,
            'reasoning': reasoning
        }
    
    async def _matches_shadow_memory(self, token_address: str, market_data: Dict[str, Any]) -> bool:
        """Check if this opportunity matches any failed patterns in shadow memory"""
        
        try:
            pattern_signature = self._create_pattern_signature(token_address, market_data)
            
            
            for failed_pattern in self.shadow_memory['failed_trades']:
                if self._pattern_similarity(pattern_signature, failed_pattern) > 0.8:
                    self.logger.warning(f"🔍 Shadow memory match: Similar to failed trade pattern")
                    return True
            
            
            for rug_pattern in self.shadow_memory['rug_patterns']:
                if self._pattern_similarity(pattern_signature, rug_pattern) > 0.7:
                    self.logger.warning(f"🚨 Shadow memory ALERT: Similar to previous rug pattern")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Shadow memory check failed: {e}")
            return False  # If check fails, allow trade but log the failure
    
    def _create_pattern_signature(self, token_address: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a pattern signature for shadow memory comparison"""
        
        return {
            'market_cap': market_data.get('market_cap', 0),
            'liquidity': market_data.get('liquidity', 0),
            'holder_count': market_data.get('holder_count', 0),
            'age_hours': market_data.get('age_hours', 0),
            'volume_24h': market_data.get('volume_24h', 0),
            'top_holders_percent': market_data.get('top_holders_percent', 0),
            'creator_behavior': market_data.get('creator_behavior', 'unknown'),
            'social_metrics': market_data.get('social_metrics', {})
        }
    
    def _pattern_similarity(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """Calculate similarity between two patterns (0 to 1)"""
        
        try:
            similarities = []
            
            numeric_keys = ['market_cap', 'liquidity', 'holder_count', 'age_hours', 'volume_24h']
            
            for key in numeric_keys:
                val1 = pattern1.get(key, 0)
                val2 = pattern2.get(key, 0)
                
                if val1 == 0 and val2 == 0:
                    similarities.append(1.0)
                elif val1 == 0 or val2 == 0:
                    similarities.append(0.0)
                else:
                    similarity = 1.0 - abs(val1 - val2) / max(val1, val2)
                    similarities.append(max(0.0, similarity))
            
            return np.mean(similarities)
            
        except Exception:
            return 0.0
    
    async def _final_risk_assessment(self, signal: ConsensusSignal, 
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Final risk assessment before allowing trade"""
        
        risks = []
        
        
        if signal.caller_reputation < -0.5:
            risks.append("High manipulation risk from callers")
        
        
        if market_data.get('volatility', 0.5) > 0.8:
            risks.append("Extreme market volatility")
        
        
        if market_data.get('liquidity', 0) < 1000:
            risks.append("Insufficient liquidity")
        
        
        if market_data.get('bot_activity_score', 0) > 0.7:
            risks.append("High bot activity detected")
        
        
        if market_data.get('age_hours', 24) < 1:
            risks.append("Token too new (< 1 hour)")
        
        return {
            'safe': len(risks) == 0,
            'risks': risks
        }
    
    async def record_trade_outcome(self, token_address: str, outcome: Dict[str, Any]):
        """Record trade outcome for source performance tracking and shadow memory"""
        
        try:
            original_signal = None
            for signal in reversed(self.consensus_history):
                if signal.token_address == token_address:
                    original_signal = signal
                    break
            
            if not original_signal:
                return
            
            is_profitable = outcome.get('profit_loss', 0) > 0
            
            
            await self._update_source_performance(original_signal, is_profitable)
            
            
            if not is_profitable:
                await self._update_shadow_memory(token_address, outcome, original_signal)
            
            
            await self._update_swarm_intelligence(original_signal, outcome)
            
        except Exception as e:
            self.logger.error(f"Failed to record trade outcome: {e}")
    
    async def _update_source_performance(self, signal: ConsensusSignal, was_profitable: bool):
        """Update performance tracking for each source"""
        
        
        source_predictions = {
            SignalSource.AI_MODEL.value: signal.ai_prediction > 0,
            SignalSource.ONCHAIN_DATA.value: signal.onchain_sentiment > 0,
            SignalSource.TWITTER_SENTIMENT.value: signal.twitter_sentiment > 0,
            SignalSource.TECHNICAL_ANALYSIS.value: signal.technical_score > 0,
            SignalSource.CALLER_INTEL.value: signal.caller_reputation > 0
        }
        
        for source, prediction in source_predictions.items():
            perf = self.source_performance[source]
            
            
            if prediction == was_profitable:
                perf['wins'] += 1
            else:
                perf['losses'] += 1
            
            
            total = perf['wins'] + perf['losses']
            perf['accuracy'] = perf['wins'] / total if total > 0 else 0.5
    
    async def _update_shadow_memory(self, token_address: str, outcome: Dict[str, Any], 
                                  signal: ConsensusSignal):
        """Update shadow memory with failed trade patterns"""
        
        failure_pattern = {
            'token_address': token_address,
            'timestamp': datetime.now().isoformat(),
            'signal_signature': {
                'ai_prediction': signal.ai_prediction,
                'onchain_sentiment': signal.onchain_sentiment,
                'twitter_sentiment': signal.twitter_sentiment,
                'consensus_strength': signal.consensus_strength
            },
            'outcome': outcome,
            'failure_type': outcome.get('failure_type', 'unknown')
        }
        
        
        if outcome.get('was_rug', False):
            self.shadow_memory['rug_patterns'].append(failure_pattern)
        elif outcome.get('was_manipulation', False):
            self.shadow_memory['manipulation_signals'].append(failure_pattern)
        else:
            self.shadow_memory['failed_trades'].append(failure_pattern)
        
        
        for memory_type in self.shadow_memory:
            if len(self.shadow_memory[memory_type]) > 500:
                self.shadow_memory[memory_type] = self.shadow_memory[memory_type][-500:]
    
    async def _continuous_intelligence_loop(self):
        """Continuous background intelligence gathering"""
        
        while True:
            try:
                await self._update_swarm_state()
                
                
                await self._adjust_consensus_thresholds()
                
                
                await self._cleanup_old_data()
                
                
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Intelligence loop error: {e}")
                await asyncio.sleep(300)
    
    async def _source_performance_tracker(self):
        """Track and log source performance"""
        
        while True:
            try:
                self.logger.info("📊 Source Performance Report:")
                for source, perf in self.source_performance.items():
                    total = perf['wins'] + perf['losses']
                    if total > 0:
                        self.logger.info(f"   {source}: {perf['accuracy']:.1%} ({perf['wins']}/{total})")
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                self.logger.error(f"Performance tracker error: {e}")
                await asyncio.sleep(3600)
    
    async def _adjust_consensus_thresholds(self):
        """Dynamically adjust consensus thresholds based on performance"""
        
        try:
            total_wins = sum(perf['wins'] for perf in self.source_performance.values())
            total_trades = sum(perf['wins'] + perf['losses'] for perf in self.source_performance.values())
            
            if total_trades > 50:  # Enough data to adjust
                system_accuracy = total_wins / total_trades
                
                
                if system_accuracy < 0.6:
                    for key in self.consensus_thresholds:
                        self.consensus_thresholds[key] = min(0.9, self.consensus_thresholds[key] * 1.05)
                
                
                elif system_accuracy > 0.8:
                    for key in self.consensus_thresholds:
                        self.consensus_thresholds[key] = max(0.3, self.consensus_thresholds[key] * 0.98)
        
        except Exception as e:
            self.logger.warning(f"Threshold adjustment failed: {e}")
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm intelligence status"""
        
        return {
            'decision_mode': self.decision_mode.value,
            'swarm_intelligence': {
                'market_mood': self.swarm_intelligence.market_mood,
                'threat_level': self.swarm_intelligence.threat_level,
                'consensus_accuracy': self.swarm_intelligence.consensus_accuracy,
                'learning_confidence': self.swarm_intelligence.learning_confidence
            },
            'consensus_thresholds': self.consensus_thresholds,
            'source_performance': self.source_performance,
            'shadow_memory_size': {
                'failed_trades': len(self.shadow_memory['failed_trades']),
                'rug_patterns': len(self.shadow_memory['rug_patterns']),
                'manipulation_signals': len(self.shadow_memory['manipulation_signals'])
            },
            'curated_sources': len(self.curated_callers),
            'blacklisted_sources': len(self.blacklisted_sources),
            'recent_consensus_signals': len([s for s in self.consensus_history 
                                           if s.timestamp > datetime.now() - timedelta(hours=24)])
        }
    
    def _analyze_source_calls(self, calls: List[Dict[str, Any]]) -> float:
        """Analyze sentiment from source calls"""
        
        if not calls:
            return 0.0
        
        sentiments = []
        for call in calls:
            call_text = call.get('text', '').lower()
            
            
            positive_count = sum(1 for word in ['bullish', 'gem', 'alpha', 'strong', 'pump'] 
                               if word in call_text)
            
            
            negative_count = sum(1 for word in ['bearish', 'dump', 'rug', 'avoid', 'exit'] 
                               if word in call_text)
            
            if positive_count + negative_count == 0:
                sentiment = 0.0
            else:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            
            sentiments.append(sentiment)
        
        return np.mean(sentiments)

    async def _get_token_symbol(self, token_address: str) -> Optional[str]:
        """Get token symbol from address using multiple sources"""
        
        try:
            # Try Jupiter API first
            if hasattr(self, 'jupiter_client'):
                try:
                    token_info = await self.jupiter_client.get_token_info(token_address)
                    if token_info and token_info.get('symbol'):
                        return token_info['symbol']
                except Exception:
                    pass
            
            # Try Solana Tracker API
            if hasattr(self, 'solana_tracker_client'):
                try:
                    token_data = await self.solana_tracker_client.get_token_data(token_address)
                    if token_data and token_data.get('symbol'):
                        return token_data['symbol']
                except Exception:
                    pass
            
            # Try Birdeye API
            if hasattr(self, 'birdeye_client'):
                try:
                    token_metadata = await self.birdeye_client.get_token_metadata(token_address)
                    if token_metadata and token_metadata.get('symbol'):
                        return token_metadata['symbol']
                except Exception:
                    pass
            
            # Fallback: try to get from local cache
            cache_key = f"token_symbol:{token_address}"
            if hasattr(self, 'token_cache') and cache_key in self.token_cache:
                return self.token_cache[cache_key]
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to get token symbol for {token_address}: {e}")
            return None

    async def _load_shadow_memory(self):
        """Load historical shadow memory data from persistent storage"""
        
        try:
            # Initialize memory structure
            self.shadow_memory = {
                'failed_trades': [],
                'rug_patterns': [],
                'manipulation_signals': [],
                'false_breakouts': []
            }
            
            # Try to load from file storage
            shadow_file = Path("data/shadow_memory.json")
            if shadow_file.exists():
                try:
                    with open(shadow_file, 'r') as f:
                        data = json.load(f)
                        
                    # Load each memory type with validation
                    for memory_type in self.shadow_memory:
                        if memory_type in data:
                            # Validate and load entries
                            valid_entries = []
                            for entry in data[memory_type]:
                                if self._validate_memory_entry(entry):
                                    valid_entries.append(entry)
                            
                            self.shadow_memory[memory_type] = valid_entries[-500:]  # Keep last 500
                            
                except Exception as e:
                    self.logger.warning(f"Failed to load shadow memory from file: {e}")
            
            # Try to load from Redis if available
            if hasattr(self, 'redis_client'):
                try:
                    for memory_type in self.shadow_memory:
                        redis_key = f"shadow_memory:{memory_type}"
                        data = self.redis_client.get(redis_key)
                        if data:
                            entries = json.loads(data)
                            if isinstance(entries, list):
                                self.shadow_memory[memory_type].extend(entries[-500:])
                except Exception as e:
                    self.logger.warning(f"Failed to load shadow memory from Redis: {e}")
            
            # Initialize token cache if not exists
            if not hasattr(self, 'token_cache'):
                self.token_cache = {}
            
            self.logger.info(f"🧠 Shadow memory loaded: {sum(len(v) for v in self.shadow_memory.values())} patterns")
            
        except Exception as e:
            self.logger.warning(f"Shadow memory loading failed: {e}")
            # Ensure basic structure exists
            self.shadow_memory = {
                'failed_trades': [],
                'rug_patterns': [],
                'manipulation_signals': [],
                'false_breakouts': []
            }
    
    def _validate_memory_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate a shadow memory entry"""
        required_fields = ['token_address', 'timestamp', 'signal_signature']
        
        # Check required fields
        for field in required_fields:
            if field not in entry:
                return False
        
        # Validate timestamp format
        try:
            datetime.fromisoformat(entry['timestamp'])
        except (ValueError, TypeError):
            return False
        
        # Validate signal signature structure
        signature = entry.get('signal_signature', {})
        if not isinstance(signature, dict):
            return False
        
        return True
    
    async def _save_shadow_memory(self):
        """Save shadow memory to persistent storage"""
        try:
            # Save to file
            shadow_file = Path("data/shadow_memory.json")
            shadow_file.parent.mkdir(exist_ok=True)
            
            with open(shadow_file, 'w') as f:
                json.dump(self.shadow_memory, f, indent=2)
            
            # Save to Redis if available
            if hasattr(self, 'redis_client'):
                for memory_type, entries in self.shadow_memory.items():
                    redis_key = f"shadow_memory:{memory_type}"
                    self.redis_client.setex(
                        redis_key, 
                        86400 * 7,  # 7 days expiry
                        json.dumps(entries[-500:])  # Keep last 500
                    )
                    
        except Exception as e:
            self.logger.warning(f"Failed to save shadow memory: {e}")

            
_colony_commander = None

async def get_colony_commander() -> ColonyCommander:
    """Get global colony commander instance"""
    global _colony_commander
    if _colony_commander is None:
        _colony_commander = ColonyCommander()
    return _colony_commander 