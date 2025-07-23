"""
PHEROMONE ANT - THE UNIFIED INFLUENCE GRAPH ENGINE
=================================================

🎯 MISSION: Model the causal chain of events from idea to influence to capital flow
⚡ STRATEGY: Track pheromone trails through the unified influence graph
🧬 EVOLUTION: Predict where information will become action
🔥 OBJECTIVE: Gain structural timing advantage by trading the flow of influence itself

The PheromoneAnt represents the pinnacle of market intelligence - it doesn't trade
charts or sentiment, it trades the FLOW OF INFLUENCE. By modeling how information
propagates through interconnected networks of wallets, social accounts, and contracts,
it can predict market movements before they appear in traditional signals.

🚀 CORE CAPABILITIES:
- Unified Influence Graph: Real-time modeling of all market participants
- Pheromone Trail Analysis: Tracking information and capital flows
- Convergence Event Detection: Identifying when information becomes action
- Causal Chain Prediction: Forecasting influence propagation patterns
- Priority One HUNT signals: Highest-conviction trading opportunities
"""

import asyncio
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import pickle
from pathlib import Path
from collections import defaultdict, deque
import random

from worker_ant_v1.utils.logger import setup_logger


class NodeType(Enum):
    """Types of nodes in the Unified Influence Graph"""
    WALLET = "wallet"
    SOCIAL_ACCOUNT = "social_account"  
    TELEGRAM_CHANNEL = "telegram_channel"
    SMART_CONTRACT = "smart_contract"
    TOKEN = "token"
    EXCHANGE = "exchange"
    NEWS_SOURCE = "news_source"


class EdgeType(Enum):
    """Types of edges (relationships) in the Unified Influence Graph"""
    FOLLOWS = "follows"
    MENTIONS = "mentions"
    RETWEETS = "retweets"
    TRANSACTION = "transaction"
    DEPLOYED = "deployed"
    LIQUIDITY_PROVIDED = "liquidity_provided"
    COPIED_TRADE = "copied_trade"
    INFLUENCED_BY = "influenced_by"


class PheromoneType(Enum):
    """Types of pheromones flowing through the graph"""
    MEMETIC = "memetic"      # Information/narrative flow
    CAPITAL = "capital"      # Money/transaction flow
    ATTENTION = "attention"  # Focus/interest flow
    SENTIMENT = "sentiment"  # Emotional/sentiment flow


class ConvergenceType(Enum):
    """Types of convergence events"""
    INFORMATION_TO_ACTION = "info_to_action"      # Information becomes trading action
    WHALE_FOLLOW = "whale_follow"                 # Retail follows whale activity
    NARRATIVE_EXPLOSION = "narrative_explosion"   # Narrative reaches critical mass
    SOCIAL_TO_FINANCIAL = "social_to_financial"   # Social activity drives financial activity
    CROSS_PLATFORM_SYNC = "cross_platform_sync"  # Coordination across platforms


@dataclass
class InfluenceNode:
    """Node in the Unified Influence Graph"""
    node_id: str
    node_type: NodeType
    metadata: Dict[str, Any]
    
    # Influence metrics
    influence_score: float = 0.0
    centrality_score: float = 0.0
    connection_count: int = 0
    
    # Activity tracking
    last_activity: Optional[datetime] = None
    activity_frequency: float = 0.0
    
    # Pheromone concentrations
    memetic_concentration: float = 0.0
    capital_concentration: float = 0.0
    attention_concentration: float = 0.0
    sentiment_concentration: float = 0.0
    
    # Historical tracking
    pheromone_history: Dict[PheromoneType, List[Tuple[datetime, float]]] = field(default_factory=dict)


@dataclass
class InfluenceEdge:
    """Edge in the Unified Influence Graph"""
    edge_id: str
    source_node: str
    target_node: str
    edge_type: EdgeType
    
    # Edge strength and flow
    connection_strength: float = 1.0
    flow_rate: float = 0.0
    
    # Pheromone flow tracking
    memetic_flow: float = 0.0
    capital_flow: float = 0.0
    attention_flow: float = 0.0
    sentiment_flow: float = 0.0
    
    # Temporal data
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: Optional[datetime] = None
    flow_history: List[Tuple[datetime, Dict[PheromoneType, float]]] = field(default_factory=list)


@dataclass 
class PheromoneTrail:
    """A trail of pheromones through the influence graph"""
    trail_id: str
    pheromone_type: PheromoneType
    source_node: str
    current_nodes: Set[str]
    
    # Trail characteristics
    strength: float
    propagation_speed: float
    decay_rate: float
    
    # Path tracking
    visited_nodes: List[str] = field(default_factory=list)
    propagation_path: List[Tuple[datetime, str, float]] = field(default_factory=list)
    
    # Trail metadata
    origin_event: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    expected_lifetime: timedelta = field(default_factory=lambda: timedelta(hours=6))


@dataclass
class ConvergenceEvent:
    """A detected convergence of pheromone trails"""
    event_id: str
    convergence_type: ConvergenceType
    convergence_node: str
    
    # Converging trails
    memetic_trail: Optional[PheromoneTrail] = None
    capital_trail: Optional[PheromoneTrail] = None
    
    # Convergence metrics
    convergence_strength: float = 0.0
    confidence_score: float = 0.0
    urgency_level: int = 1  # 1-10, 10 being immediate action required
    
    # Predicted impact
    predicted_price_impact: float = 0.0
    predicted_volume_impact: float = 0.0
    predicted_timeline: timedelta = field(default_factory=lambda: timedelta(hours=2))
    
    # Detection details
    detected_at: datetime = field(default_factory=datetime.now)
    contributing_signals: List[str] = field(default_factory=list)
    

class PheromoneAnt:
    """The Master Intelligence Agent - Trading the Flow of Influence"""
    
    def __init__(self):
        self.logger = setup_logger("PheromoneAnt")
        
        # The Unified Influence Graph - the core data structure
        self.influence_graph = nx.DiGraph()
        self.nodes: Dict[str, InfluenceNode] = {}
        self.edges: Dict[str, InfluenceEdge] = {}
        
        # Active pheromone trails
        self.active_trails: Dict[str, PheromoneTrail] = {}
        self.trail_history: List[PheromoneTrail] = []
        
        # Convergence event tracking
        self.active_convergences: Dict[str, ConvergenceEvent] = {}
        self.convergence_history: List[ConvergenceEvent] = []
        
        # Data streams and processors
        self.data_streams = {
            'onchain_monitor': None,
            'social_monitor': None,
            'contract_monitor': None,
            'news_monitor': None
        }
        
        # Pheromone propagation configuration
        self.propagation_config = {
            'memetic_speed_multiplier': 1.2,     # Information travels fastest
            'capital_speed_multiplier': 0.8,     # Capital follows slower
            'attention_speed_multiplier': 1.5,   # Attention is fastest
            'sentiment_speed_multiplier': 1.0,   # Sentiment at baseline
            'decay_rate_per_hour': 0.1,         # 10% decay per hour
            'min_trail_strength': 0.01,         # Minimum viable trail strength
            'convergence_threshold': 0.7,       # Threshold for convergence detection
        }
        
        # Analysis and prediction
        self.influence_models = {}
        self.prediction_cache = {}
        self.pattern_database = defaultdict(list)
        
        # Performance tracking
        self.metrics = {
            'convergences_detected': 0,
            'priority_one_hunts_issued': 0,
            'successful_predictions': 0,
            'false_positives': 0,
            'average_prediction_accuracy': 0.0,
            'graph_update_frequency': 0.0,
        }
        
        # System state
        self.initialized = False
        self.processing_active = False
        
        # Data persistence
        self.data_dir = Path('data/pheromone_ant')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("👁️ PheromoneAnt - Unified Influence Graph Engine initialized")
    
    async def initialize(self, data_sources: Dict[str, Any]) -> bool:
        """Initialize the PheromoneAnt with data sources"""
        try:
            self.logger.info("👁️ Initializing PheromoneAnt Unified Influence Graph...")
            
            # Initialize data streams
            await self._initialize_data_streams(data_sources)
            
            # Load existing graph data if available
            await self._load_graph_state()
            
            # Bootstrap initial graph structure
            await self._bootstrap_influence_graph()
            
            # Start real-time processing loops
            asyncio.create_task(self._graph_update_loop())
            asyncio.create_task(self._pheromone_propagation_loop())
            asyncio.create_task(self._convergence_detection_loop())
            asyncio.create_task(self._pattern_learning_loop())
            
            self.initialized = True
            self.processing_active = True
            
            self.logger.info(f"✅ PheromoneAnt initialized: {len(self.nodes)} nodes, {len(self.edges)} edges")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ PheromoneAnt initialization failed: {e}")
            return False
    
    async def detect_convergence_events(self) -> List[ConvergenceEvent]:
        """
        PRIMARY METHOD: Detect convergence events where information becomes action
        
        Returns list of detected convergence events, with Priority One HUNT signals
        marked for immediate action by ColonyCommander
        """
        try:
            detected_events = []
            
            # Analyze all active pheromone trails for convergence patterns
            memetic_trails = [t for t in self.active_trails.values() if t.pheromone_type == PheromoneType.MEMETIC]
            capital_trails = [t for t in self.active_trails.values() if t.pheromone_type == PheromoneType.CAPITAL]
            
            # Check for convergence between memetic and capital trails
            for memetic_trail in memetic_trails:
                for capital_trail in capital_trails:
                    convergence = await self._analyze_trail_convergence(memetic_trail, capital_trail)
                    
                    if convergence:
                        detected_events.append(convergence)
                        
                        # Check if this qualifies for Priority One HUNT signal
                        if self._is_priority_one_hunt(convergence):
                            await self._issue_priority_one_hunt(convergence)
            
            # Check for other convergence patterns
            cross_platform_convergences = await self._detect_cross_platform_convergences()
            whale_follow_convergences = await self._detect_whale_follow_convergences()
            narrative_explosion_convergences = await self._detect_narrative_explosions()
            
            detected_events.extend(cross_platform_convergences)
            detected_events.extend(whale_follow_convergences) 
            detected_events.extend(narrative_explosion_convergences)
            
            # Update metrics
            self.metrics['convergences_detected'] += len(detected_events)
            
            return detected_events
            
        except Exception as e:
            self.logger.error(f"❌ Convergence detection failed: {e}")
            return []
    
    async def _analyze_trail_convergence(self, memetic_trail: PheromoneTrail, 
                                       capital_trail: PheromoneTrail) -> Optional[ConvergenceEvent]:
        """Analyze two trails for convergence patterns"""
        try:
            # Check for spatial convergence (same nodes)
            common_nodes = set(memetic_trail.current_nodes) & set(capital_trail.current_nodes)
            
            if not common_nodes:
                return None
            
            # Check for temporal convergence (timing alignment)
            temporal_alignment = self._calculate_temporal_alignment(memetic_trail, capital_trail)
            
            if temporal_alignment < 0.5:  # Minimum alignment threshold
                return None
            
            # Calculate convergence strength
            convergence_strength = self._calculate_convergence_strength(
                memetic_trail, capital_trail, common_nodes, temporal_alignment
            )
            
            if convergence_strength < self.propagation_config['convergence_threshold']:
                return None
            
            # Select primary convergence node (highest influence)
            primary_node = max(common_nodes, key=lambda n: self.nodes[n].influence_score)
            
            # Create convergence event
            event_id = f"conv_{int(time.time() * 1000)}_{random.randint(100, 999)}"
            
            convergence = ConvergenceEvent(
                event_id=event_id,
                convergence_type=ConvergenceType.INFORMATION_TO_ACTION,
                convergence_node=primary_node,
                memetic_trail=memetic_trail,
                capital_trail=capital_trail,
                convergence_strength=convergence_strength,
                confidence_score=min(0.95, convergence_strength * temporal_alignment),
                urgency_level=self._calculate_urgency_level(convergence_strength, temporal_alignment),
                contributing_signals=[memetic_trail.trail_id, capital_trail.trail_id]
            )
            
            # Predict impact
            convergence.predicted_price_impact = self._predict_price_impact(convergence)
            convergence.predicted_volume_impact = self._predict_volume_impact(convergence)
            convergence.predicted_timeline = self._predict_impact_timeline(convergence)
            
            self.logger.info(f"🔗 Convergence detected: {convergence_strength:.2f} strength at node {primary_node[:8]}")
            
            return convergence
            
        except Exception as e:
            self.logger.error(f"Trail convergence analysis error: {e}")
            return None
    
    def _calculate_temporal_alignment(self, trail1: PheromoneTrail, trail2: PheromoneTrail) -> float:
        """Calculate temporal alignment between two trails"""
        try:
            # Analyze timing patterns in trail propagation
            if not trail1.propagation_path or not trail2.propagation_path:
                return 0.0
            
            # Get recent propagation events from both trails
            recent_duration = timedelta(hours=2)
            cutoff_time = datetime.now() - recent_duration
            
            recent_events_1 = [(t, node) for t, node, _ in trail1.propagation_path if t >= cutoff_time]
            recent_events_2 = [(t, node) for t, node, _ in trail2.propagation_path if t >= cutoff_time]
            
            if not recent_events_1 or not recent_events_2:
                return 0.0
            
            # Calculate temporal overlap
            times_1 = [t for t, _ in recent_events_1]
            times_2 = [t for t, _ in recent_events_2]
            
            # Find overlapping time windows
            overlap_score = 0.0
            window_size = timedelta(minutes=30)
            
            for t1 in times_1:
                for t2 in times_2:
                    time_diff = abs((t1 - t2).total_seconds())
                    if time_diff <= window_size.total_seconds():
                        overlap_score += 1.0 - (time_diff / window_size.total_seconds())
            
            # Normalize by trail lengths
            max_possible_overlap = min(len(times_1), len(times_2))
            alignment = overlap_score / max_possible_overlap if max_possible_overlap > 0 else 0.0
            
            return min(1.0, alignment)
            
        except Exception as e:
            self.logger.error(f"Temporal alignment calculation error: {e}")
            return 0.0
    
    def _calculate_convergence_strength(self, memetic_trail: PheromoneTrail, capital_trail: PheromoneTrail,
                                      common_nodes: Set[str], temporal_alignment: float) -> float:
        """Calculate overall convergence strength"""
        try:
            # Base strength from trail strengths
            base_strength = (memetic_trail.strength + capital_trail.strength) / 2
            
            # Boost from number of convergence points
            node_count_boost = min(1.0, len(common_nodes) / 5)  # Up to 100% boost for 5+ nodes
            
            # Boost from influence of convergence nodes
            influence_boost = 0.0
            for node_id in common_nodes:
                node = self.nodes.get(node_id)
                if node:
                    influence_boost += node.influence_score
            influence_boost = min(1.0, influence_boost / len(common_nodes))
            
            # Temporal alignment multiplier
            temporal_multiplier = 0.5 + (temporal_alignment * 0.5)  # 0.5x to 1.0x
            
            # Calculate final strength
            convergence_strength = base_strength * (1 + node_count_boost + influence_boost) * temporal_multiplier
            
            return min(1.0, convergence_strength)
            
        except Exception as e:
            self.logger.error(f"Convergence strength calculation error: {e}")
            return 0.0
    
    def _calculate_urgency_level(self, convergence_strength: float, temporal_alignment: float) -> int:
        """Calculate urgency level (1-10) for convergence event"""
        try:
            # Base urgency from convergence strength
            base_urgency = convergence_strength * 10
            
            # Boost from temporal alignment (aligned trails are more urgent)
            alignment_boost = temporal_alignment * 3
            
            # Cap at maximum urgency
            urgency = min(10, int(base_urgency + alignment_boost))
            
            return max(1, urgency)
            
        except Exception as e:
            self.logger.error(f"Urgency calculation error: {e}")
            return 1
    
    def _is_priority_one_hunt(self, convergence: ConvergenceEvent) -> bool:
        """Determine if convergence qualifies for Priority One HUNT signal"""
        try:
            # Priority One criteria (ultra-high conviction):
            criteria = [
                convergence.convergence_strength >= 0.85,    # Very strong convergence
                convergence.confidence_score >= 0.8,        # High confidence
                convergence.urgency_level >= 8,             # High urgency
                len(convergence.contributing_signals) >= 2   # Multiple signals
            ]
            
            # Additional checks for node type and influence
            primary_node = self.nodes.get(convergence.convergence_node)
            if primary_node:
                criteria.extend([
                    primary_node.influence_score >= 0.7,    # High-influence node
                    primary_node.connection_count >= 10     # Well-connected node
                ])
            
            # Must meet majority of criteria
            return sum(criteria) >= 4
            
        except Exception as e:
            self.logger.error(f"Priority One assessment error: {e}")
            return False
    
    async def _issue_priority_one_hunt(self, convergence: ConvergenceEvent):
        """Issue Priority One HUNT signal to ColonyCommander"""
        try:
            self.logger.warning(f"🚨 PRIORITY ONE HUNT SIGNAL ISSUED! 🚨")
            self.logger.warning(f"   Convergence: {convergence.convergence_strength:.2f}")
            self.logger.warning(f"   Confidence: {convergence.confidence_score:.2f}")
            self.logger.warning(f"   Urgency: {convergence.urgency_level}/10")
            self.logger.warning(f"   Node: {convergence.convergence_node[:12]}")
            
            # Mark as Priority One
            convergence.urgency_level = 10
            
            # Add to active convergences for immediate action
            self.active_convergences[convergence.event_id] = convergence
            
            # Update metrics
            self.metrics['priority_one_hunts_issued'] += 1
            
            # In production, this would directly notify ColonyCommander
            # For now, log the critical information needed for immediate action
            hunt_signal = {
                'signal_type': 'PRIORITY_ONE_HUNT',
                'convergence_id': convergence.event_id,
                'target_node': convergence.convergence_node,
                'convergence_strength': convergence.convergence_strength,
                'confidence_score': convergence.confidence_score,
                'predicted_price_impact': convergence.predicted_price_impact,
                'predicted_timeline_minutes': convergence.predicted_timeline.total_seconds() / 60,
                'memetic_trail_strength': convergence.memetic_trail.strength if convergence.memetic_trail else 0,
                'capital_trail_strength': convergence.capital_trail.strength if convergence.capital_trail else 0,
                'immediate_action_required': True
            }
            
            self.logger.warning(f"📡 HUNT SIGNAL DATA: {json.dumps(hunt_signal, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Priority One HUNT signal issuance error: {e}")
    
    # Prediction methods
    
    def _predict_price_impact(self, convergence: ConvergenceEvent) -> float:
        """Predict price impact from convergence event"""
        try:
            # Base impact from convergence strength
            base_impact = convergence.convergence_strength * 0.15  # Up to 15% impact
            
            # Adjust based on node influence
            node = self.nodes.get(convergence.convergence_node)
            if node:
                influence_multiplier = 1 + node.influence_score
                base_impact *= influence_multiplier
            
            # Adjust based on trail strengths
            if convergence.capital_trail:
                capital_multiplier = 1 + convergence.capital_trail.strength
                base_impact *= capital_multiplier
            
            return min(0.5, base_impact)  # Cap at 50% impact
            
        except Exception as e:
            self.logger.error(f"Price impact prediction error: {e}")
            return 0.0
    
    def _predict_volume_impact(self, convergence: ConvergenceEvent) -> float:
        """Predict volume impact from convergence event"""
        try:
            # Volume impact tends to be higher than price impact
            price_impact = self._predict_price_impact(convergence)
            volume_impact = price_impact * 3.0  # 3x volume multiplier
            
            return min(2.0, volume_impact)  # Cap at 200% volume increase
            
        except Exception as e:
            self.logger.error(f"Volume impact prediction error: {e}")
            return 0.0
    
    def _predict_impact_timeline(self, convergence: ConvergenceEvent) -> timedelta:
        """Predict timeline for convergence impact"""
        try:
            # Base timeline from urgency
            base_hours = max(0.5, 10 - convergence.urgency_level)  # Higher urgency = faster impact
            
            # Adjust based on trail propagation speeds
            if convergence.memetic_trail and convergence.capital_trail:
                avg_speed = (convergence.memetic_trail.propagation_speed + 
                           convergence.capital_trail.propagation_speed) / 2
                speed_multiplier = 2.0 - avg_speed  # Faster propagation = sooner impact
                base_hours *= speed_multiplier
            
            return timedelta(hours=base_hours)
            
        except Exception as e:
            self.logger.error(f"Impact timeline prediction error: {e}")
            return timedelta(hours=2)
    
    # Additional convergence detection methods
    
    async def _detect_cross_platform_convergences(self) -> List[ConvergenceEvent]:
        """Detect convergences across different platforms"""
        try:
            convergences = []
            
            # Group nodes by platform
            platform_nodes = defaultdict(list)
            for node_id, node in self.nodes.items():
                platform = node.metadata.get('platform', 'unknown')
                platform_nodes[platform].append(node_id)
            
            # Look for synchronized activity across platforms
            platforms = list(platform_nodes.keys())
            for i, platform1 in enumerate(platforms):
                for platform2 in platforms[i+1:]:
                    convergence = await self._analyze_cross_platform_activity(
                        platform1, platform_nodes[platform1],
                        platform2, platform_nodes[platform2]
                    )
                    if convergence:
                        convergences.append(convergence)
            
            return convergences
            
        except Exception as e:
            self.logger.error(f"Cross-platform convergence detection error: {e}")
            return []
    
    async def _detect_whale_follow_convergences(self) -> List[ConvergenceEvent]:
        """Detect whale-follow convergence patterns"""
        try:
            convergences = []
            
            # Identify whale nodes (high influence, large transactions)
            whale_nodes = [
                node_id for node_id, node in self.nodes.items()
                if (node.node_type == NodeType.WALLET and 
                    node.influence_score > 0.8 and
                    node.capital_concentration > 0.7)
            ]
            
            # Look for retail following whale activity
            for whale_id in whale_nodes:
                followers = self._get_influenced_nodes(whale_id)
                if len(followers) >= 5:  # Significant following
                    convergence = await self._analyze_whale_follow_pattern(whale_id, followers)
                    if convergence:
                        convergences.append(convergence)
            
            return convergences
            
        except Exception as e:
            self.logger.error(f"Whale follow convergence detection error: {e}")
            return []
    
    async def _detect_narrative_explosions(self) -> List[ConvergenceEvent]:
        """Detect narrative explosion convergence patterns"""
        try:
            convergences = []
            
            # Look for rapid increases in memetic concentration across multiple nodes
            high_memetic_nodes = [
                node_id for node_id, node in self.nodes.items()
                if node.memetic_concentration > 0.6
            ]
            
            if len(high_memetic_nodes) >= 10:  # Threshold for potential explosion
                convergence = await self._analyze_narrative_explosion_pattern(high_memetic_nodes)
                if convergence:
                    convergences.append(convergence)
            
            return convergences
            
        except Exception as e:
            self.logger.error(f"Narrative explosion detection error: {e}")
            return []
    
    # Graph management and processing loops
    
    async def _graph_update_loop(self):
        """Continuously update the influence graph with new data"""
        while self.processing_active:
            try:
                # Process incoming data from all streams
                await self._process_onchain_data()
                await self._process_social_data()
                await self._process_contract_data()
                await self._process_news_data()
                
                # Update graph metrics and centralities
                await self._update_graph_metrics()
                
                # Cleanup old data
                await self._cleanup_expired_data()
                
                # Update processing frequency metric
                self.metrics['graph_update_frequency'] += 1
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Graph update loop error: {e}")
                await asyncio.sleep(60)
    
    async def _pheromone_propagation_loop(self):
        """Simulate pheromone propagation through the graph"""
        while self.processing_active:
            try:
                # Propagate all active trails
                for trail in list(self.active_trails.values()):
                    await self._propagate_trail(trail)
                
                # Clean up expired trails
                await self._cleanup_expired_trails()
                
                await asyncio.sleep(10)  # Propagate every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Pheromone propagation error: {e}")
                await asyncio.sleep(30)
    
    async def _convergence_detection_loop(self):
        """Continuously monitor for convergence events"""
        while self.processing_active:
            try:
                # Detect new convergence events
                new_convergences = await self.detect_convergence_events()
                
                # Process and store convergences
                for convergence in new_convergences:
                    if convergence.event_id not in self.active_convergences:
                        self.active_convergences[convergence.event_id] = convergence
                        self.logger.info(f"🔗 New convergence detected: {convergence.event_id}")
                
                # Clean up resolved convergences
                await self._cleanup_resolved_convergences()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Convergence detection loop error: {e}")
                await asyncio.sleep(120)
    
    async def _pattern_learning_loop(self):
        """Learn patterns from convergence outcomes"""
        while self.processing_active:
            try:
                # Analyze successful convergence predictions
                await self._analyze_convergence_outcomes()
                
                # Update prediction models
                await self._update_prediction_models()
                
                # Save learned patterns
                await self._save_pattern_database()
                
                await asyncio.sleep(3600)  # Learn every hour
                
            except Exception as e:
                self.logger.error(f"Pattern learning error: {e}")
                await asyncio.sleep(3600)
    
    # Placeholder implementations for data processing
    
    async def _process_onchain_data(self):
        """Process on-chain transaction data"""
        # Placeholder for on-chain data processing
        pass
    
    async def _process_social_data(self):
        """Process social media data"""
        # Placeholder for social data processing  
        pass
    
    async def _process_contract_data(self):
        """Process smart contract deployment and interaction data"""
        # Placeholder for contract data processing
        pass
    
    async def _process_news_data(self):
        """Process news and media data"""
        # Placeholder for news data processing
        pass
    
    # Helper methods and utilities
    
    def _get_influenced_nodes(self, influencer_id: str) -> List[str]:
        """Get nodes influenced by a given node"""
        try:
            influenced = []
            
            # Find nodes with incoming edges from influencer
            for edge_id, edge in self.edges.items():
                if (edge.source_node == influencer_id and 
                    edge.edge_type in [EdgeType.INFLUENCED_BY, EdgeType.COPIED_TRADE]):
                    influenced.append(edge.target_node)
            
            return influenced
            
        except Exception as e:
            self.logger.error(f"Influenced nodes lookup error: {e}")
            return []
    
    async def _initialize_data_streams(self, data_sources: Dict[str, Any]):
        """Initialize data streams"""
        try:
            # Placeholder for data stream initialization
            self.data_streams = {
                'onchain_monitor': data_sources.get('onchain_monitor'),
                'social_monitor': data_sources.get('social_monitor'),
                'contract_monitor': data_sources.get('contract_monitor'),
                'news_monitor': data_sources.get('news_monitor')
            }
            
            self.logger.info("📡 Data streams initialized")
            
        except Exception as e:
            self.logger.error(f"Data stream initialization error: {e}")
    
    async def _load_graph_state(self):
        """Load existing graph state from disk"""
        try:
            graph_file = self.data_dir / 'influence_graph.pkl'
            if graph_file.exists():
                with open(graph_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.influence_graph = saved_data.get('graph', nx.DiGraph())
                    self.nodes = saved_data.get('nodes', {})
                    self.edges = saved_data.get('edges', {})
                
                self.logger.info(f"📁 Loaded graph state: {len(self.nodes)} nodes, {len(self.edges)} edges")
            
        except Exception as e:
            self.logger.error(f"Graph state loading error: {e}")
    
    async def _bootstrap_influence_graph(self):
        """Bootstrap initial graph structure"""
        try:
            # Add some initial nodes for development/testing
            initial_nodes = [
                ('wallet_001', NodeType.WALLET, {'platform': 'solana', 'balance': 1000}),
                ('wallet_002', NodeType.WALLET, {'platform': 'solana', 'balance': 5000}),
                ('social_001', NodeType.SOCIAL_ACCOUNT, {'platform': 'twitter', 'followers': 10000}),
                ('contract_001', NodeType.SMART_CONTRACT, {'platform': 'solana', 'type': 'token'}),
            ]
            
            for node_id, node_type, metadata in initial_nodes:
                if node_id not in self.nodes:
                    node = InfluenceNode(
                        node_id=node_id,
                        node_type=node_type,
                        metadata=metadata,
                        influence_score=random.uniform(0.3, 0.8)
                    )
                    self.nodes[node_id] = node
                    self.influence_graph.add_node(node_id)
            
            self.logger.info(f"🌱 Bootstrapped graph with {len(self.nodes)} initial nodes")
            
        except Exception as e:
            self.logger.error(f"Graph bootstrapping error: {e}")
    
    # Status and reporting
    
    def get_pheromone_status(self) -> Dict[str, Any]:
        """Get current PheromoneAnt status"""
        try:
            return {
                'graph_nodes': len(self.nodes),
                'graph_edges': len(self.edges),
                'active_trails': len(self.active_trails),
                'active_convergences': len(self.active_convergences),
                'priority_one_hunts_issued': self.metrics['priority_one_hunts_issued'],
                'convergences_detected': self.metrics['convergences_detected'],
                'processing_active': self.processing_active,
                'recent_convergences': [
                    {
                        'event_id': conv.event_id,
                        'strength': conv.convergence_strength,
                        'urgency': conv.urgency_level,
                        'node': conv.convergence_node[:12]
                    }
                    for conv in list(self.active_convergences.values())[-5:]
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Status reporting error: {e}")
            return {}
    
    def get_convergence_details(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a convergence event"""
        try:
            convergence = self.active_convergences.get(event_id)
            if not convergence:
                return None
            
            return {
                'event_id': convergence.event_id,
                'convergence_type': convergence.convergence_type.value,
                'convergence_node': convergence.convergence_node,
                'convergence_strength': convergence.convergence_strength,
                'confidence_score': convergence.confidence_score,
                'urgency_level': convergence.urgency_level,
                'predicted_price_impact': convergence.predicted_price_impact,
                'predicted_volume_impact': convergence.predicted_volume_impact,
                'predicted_timeline_hours': convergence.predicted_timeline.total_seconds() / 3600,
                'contributing_signals': convergence.contributing_signals,
                'detected_at': convergence.detected_at.isoformat(),
                'is_priority_one': convergence.urgency_level >= 8
            }
            
        except Exception as e:
            self.logger.error(f"Convergence details error: {e}")
            return None


# Integration point for the swarm systems
async def get_pheromone_ant() -> PheromoneAnt:
    """Get global PheromoneAnt instance"""
    pheromone_ant = PheromoneAnt()
    
    # Initialize with mock data sources
    mock_data_sources = {
        'onchain_monitor': None,
        'social_monitor': None,
        'contract_monitor': None,
        'news_monitor': None
    }
    
    await pheromone_ant.initialize(mock_data_sources)
    return pheromone_ant 