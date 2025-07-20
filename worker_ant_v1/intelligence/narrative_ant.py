"""
NARRATIVE ANT - MEMETIC DRIFT SENSOR
===================================

The NarrativeAnt is an AI agent that analyzes memetic drift and narrative strength
across the crypto ecosystem. It tracks trending narrative categories and provides
strategic capital allocation guidance based on narrative momentum.

This system embodies the Colony's cultural intelligence, detecting shifts in:
- Social media velocity and sentiment patterns
- On-chain capital flows between token categories  
- Contract deployment patterns and themes
- Cross-platform narrative synchronization
"""

import asyncio
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import logging

from worker_ant_v1.utils.logger import setup_logger

class NarrativeCategory(Enum):
    """Memetic narrative categories tracked by the NarrativeAnt"""
    POLITICAL_FINANCE = "political_finance"
    CAT_COINS = "cat_coins"
    DOG_COINS = "dog_coins"
    RETRO_MEMES = "retro_memes"
    AI_THEMES = "ai_themes"
    GAMING_METAVERSE = "gaming_metaverse"
    DEFI_INNOVATION = "defi_innovation"
    NFT_CULTURE = "nft_culture"
    ENVIRONMENTAL = "environmental"
    CELEBRITY_COINS = "celebrity_coins"
    COMMUNITY_DRIVEN = "community_driven"
    UTILITY_TOKENS = "utility_tokens"
    SOCIAL_EXPERIMENTS = "social_experiments"
    CROSS_CHAIN = "cross_chain"
    UNKNOWN = "unknown"


class NarrativeStrength(Enum):
    """Narrative strength levels"""
    DEAD = "dead"           # < 10% momentum
    WEAK = "weak"           # 10-30% momentum
    MODERATE = "moderate"   # 30-50% momentum  
    STRONG = "strong"       # 50-75% momentum
    DOMINANT = "dominant"   # 75-90% momentum
    VIRAL = "viral"         # > 90% momentum


class DataSource(Enum):
    """Data sources for narrative analysis"""
    FARCASTER = "farcaster"
    TWITTER = "twitter"
    REDDIT = "reddit"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    ONCHAIN_FLOWS = "onchain_flows"
    CONTRACT_DEPLOYMENTS = "contract_deployments"
    DEX_ACTIVITY = "dex_activity"


@dataclass
class NarrativeSignal:
    """Individual narrative signal from a data source"""
    source: DataSource
    category: NarrativeCategory
    signal_strength: float  # 0.0 to 1.0
    velocity: float  # Rate of change
    sentiment_score: float  # -1.0 to 1.0
    volume_score: float  # Relative volume/activity
    timestamp: datetime
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeTrend:
    """Aggregated narrative trend analysis"""
    category: NarrativeCategory
    strength: NarrativeStrength
    momentum_score: float  # 0.0 to 1.0
    velocity: float  # Rate of momentum change
    duration_hours: float  # How long this trend has been active
    peak_strength: float  # Peak momentum reached
    capital_flow_score: float  # On-chain capital flow intensity
    social_velocity: float  # Social media discussion velocity
    deployment_rate: float  # New token deployment rate in category
    cross_platform_sync: float  # How synchronized across platforms
    sustainability_score: float  # Predicted sustainability
    recommended_allocation: float  # Suggested capital allocation (0.0-1.0)
    last_updated: datetime


@dataclass
class MemecoinClassification:
    """Classification of a memecoin into narrative categories"""
    token_address: str
    primary_category: NarrativeCategory
    secondary_categories: List[NarrativeCategory]
    confidence_score: float
    narrative_strength: NarrativeStrength
    trend_alignment: float  # How well aligned with current trends
    predicted_momentum: float  # Predicted momentum potential
    classification_time: datetime


class NarrativeAnt:
    """The Colony's Memetic Drift Sensor and Narrative Intelligence Agent"""
    
    def __init__(self):
        self.logger = setup_logger("NarrativeAnt")
        
        # Narrative tracking
        self.active_narratives: Dict[NarrativeCategory, NarrativeTrend] = {}
        self.narrative_history: Dict[NarrativeCategory, deque] = {
            category: deque(maxlen=1000) for category in NarrativeCategory
        }
        
        # Signal processing
        self.signal_buffer: Dict[DataSource, deque] = {
            source: deque(maxlen=500) for source in DataSource
        }
        self.signal_processors: Dict[DataSource, Any] = {}
        
        # Token classification
        self.token_classifications: Dict[str, MemecoinClassification] = {}
        self.classification_cache_ttl = 3600  # 1 hour cache
        
        # Colony intelligence
        self.capital_allocation_recommendations: Dict[NarrativeCategory, float] = {}
        self.swarm_deployment_signals: List[Dict[str, Any]] = []
        
        # Configuration
        self.config = {
            'signal_processing_interval': 30,  # seconds
            'trend_analysis_interval': 300,    # 5 minutes
            'capital_allocation_interval': 600, # 10 minutes
            'min_signal_confidence': 0.6,
            'trend_detection_threshold': 0.3,
            'viral_threshold': 0.9,
            'sustainability_weight': 0.4,
            'momentum_weight': 0.6
        }
        
        # Data source integrations (to be injected)
        self.social_feeds = {}
        self.onchain_analyzers = {}
        self.contract_scanners = {}
        
        # Performance tracking
        self.total_signals_processed = 0
        self.trends_detected = 0
        self.successful_predictions = 0
        self.capital_allocation_accuracy = 0.0
    
    async def initialize(self, data_sources: Dict[str, Any] = None):
        """Initialize the NarrativeAnt memetic intelligence system"""
        self.logger.info("🧠 Initializing NarrativeAnt - Memetic Drift Sensor...")
        
        # Initialize data source connections
        if data_sources:
            self.social_feeds = data_sources.get('social_feeds', {})
            self.onchain_analyzers = data_sources.get('onchain_analyzers', {})
            self.contract_scanners = data_sources.get('contract_scanners', {})
        
        # Initialize narrative baselines
        await self._initialize_narrative_baselines()
        
        # Start monitoring loops
        asyncio.create_task(self._signal_processing_loop())
        asyncio.create_task(self._trend_analysis_loop())
        asyncio.create_task(self._capital_allocation_loop())
        asyncio.create_task(self._memetic_learning_loop())
        
        self.logger.info("✅ NarrativeAnt active - Memetic intelligence online")
    
    async def analyze_token_narrative(self, token_address: str, token_metadata: Dict[str, Any] = None) -> MemecoinClassification:
        """Analyze and classify a token's narrative positioning"""
        try:
            # Check cache first
            if token_address in self.token_classifications:
                cached = self.token_classifications[token_address]
                if (datetime.now() - cached.classification_time).seconds < self.classification_cache_ttl:
                    return cached
            
            # Gather token intelligence
            token_data = token_metadata or {}
            token_name = token_data.get('name', '').lower()
            token_symbol = token_data.get('symbol', '').lower()
            token_description = token_data.get('description', '').lower()
            
            # Analyze narrative categories
            category_scores = await self._calculate_category_scores(token_name, token_symbol, token_description, token_data)
            
            # Determine primary and secondary categories
            sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
            primary_category = sorted_categories[0][0]
            primary_score = sorted_categories[0][1]
            
            secondary_categories = [cat for cat, score in sorted_categories[1:4] if score > 0.3]
            
            # Get current narrative strength for primary category
            current_trend = self.active_narratives.get(primary_category)
            narrative_strength = current_trend.strength if current_trend else NarrativeStrength.WEAK
            
            # Calculate trend alignment
            trend_alignment = await self._calculate_trend_alignment(primary_category, secondary_categories)
            
            # Predict momentum potential
            predicted_momentum = await self._predict_token_momentum(primary_category, category_scores, token_data)
            
            # Create classification
            classification = MemecoinClassification(
                token_address=token_address,
                primary_category=primary_category,
                secondary_categories=secondary_categories,
                confidence_score=primary_score,
                narrative_strength=narrative_strength,
                trend_alignment=trend_alignment,
                predicted_momentum=predicted_momentum,
                classification_time=datetime.now()
            )
            
            # Cache classification
            self.token_classifications[token_address] = classification
            
            self.logger.info(f"🏷️ Token classified: {token_address[:8]} -> {primary_category.value} | "
                           f"Strength: {narrative_strength.value} | Momentum: {predicted_momentum:.2f}")
            
            return classification
            
        except Exception as e:
            self.logger.error(f"Error analyzing token narrative: {e}")
            return MemecoinClassification(
                token_address=token_address,
                primary_category=NarrativeCategory.UNKNOWN,
                secondary_categories=[],
                confidence_score=0.0,
                narrative_strength=NarrativeStrength.DEAD,
                trend_alignment=0.0,
                predicted_momentum=0.0,
                classification_time=datetime.now()
            )
    
    async def get_capital_allocation_recommendations(self) -> Dict[NarrativeCategory, Dict[str, Any]]:
        """Get strategic capital allocation recommendations based on narrative analysis"""
        try:
            recommendations = {}
            
            total_momentum = sum(trend.momentum_score for trend in self.active_narratives.values())
            
            for category, trend in self.active_narratives.items():
                # Calculate recommended allocation based on multiple factors
                momentum_weight = trend.momentum_score / max(total_momentum, 1) if total_momentum > 0 else 0
                sustainability_factor = trend.sustainability_score
                velocity_factor = min(1.0, trend.velocity / 0.5)  # Normalize velocity
                
                # Combine factors with configured weights
                allocation_score = (
                    momentum_weight * self.config['momentum_weight'] +
                    sustainability_factor * self.config['sustainability_weight'] +
                    velocity_factor * 0.2
                )
                
                # Apply risk adjustment based on trend duration
                if trend.duration_hours < 6:  # Very new trends are riskier
                    allocation_score *= 0.7
                elif trend.duration_hours > 48:  # Very old trends are also riskier
                    allocation_score *= 0.8
                
                recommendations[category] = {
                    'allocation_percentage': min(0.4, allocation_score),  # Cap at 40%
                    'confidence': trend.cross_platform_sync,
                    'momentum_score': trend.momentum_score,
                    'sustainability_score': trend.sustainability_score,
                    'velocity': trend.velocity,
                    'strength': trend.strength.value,
                    'duration_hours': trend.duration_hours,
                    'risk_level': self._assess_narrative_risk(trend),
                    'action_recommendation': self._generate_action_recommendation(trend)
                }
            
            # Store recommendations
            self.capital_allocation_recommendations = {
                cat: rec['allocation_percentage'] for cat, rec in recommendations.items()
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating capital allocation recommendations: {e}")
            return {}
    
    async def detect_emerging_narratives(self) -> List[Dict[str, Any]]:
        """Detect emerging narrative trends before they become mainstream"""
        try:
            emerging_narratives = []
            
            # Look for rapid velocity increases in weak narratives
            for category, trend in self.active_narratives.items():
                if trend.strength in [NarrativeStrength.WEAK, NarrativeStrength.MODERATE]:
                    # Check if velocity is accelerating
                    recent_signals = list(self.narrative_history[category])[-10:]  # Last 10 signals
                    if len(recent_signals) >= 5:
                        velocities = [signal.velocity for signal in recent_signals if hasattr(signal, 'velocity')]
                        if velocities:
                            velocity_trend = np.polyfit(range(len(velocities)), velocities, 1)[0]
                            
                            if velocity_trend > 0.1:  # Significant acceleration
                                emerging_narratives.append({
                                    'category': category.value,
                                    'current_strength': trend.strength.value,
                                    'momentum_score': trend.momentum_score,
                                    'velocity_acceleration': velocity_trend,
                                    'deployment_rate': trend.deployment_rate,
                                    'social_velocity': trend.social_velocity,
                                    'emergence_confidence': min(1.0, velocity_trend * 5),
                                    'predicted_peak_time_hours': self._predict_narrative_peak_time(trend, velocity_trend),
                                    'early_entry_opportunity': True
                                })
            
            # Sort by emergence confidence
            emerging_narratives.sort(key=lambda x: x['emergence_confidence'], reverse=True)
            
            if emerging_narratives:
                self.logger.info(f"🚀 Detected {len(emerging_narratives)} emerging narratives")
                for narrative in emerging_narratives[:3]:  # Log top 3
                    self.logger.info(f"   📈 {narrative['category']}: {narrative['emergence_confidence']:.2f} confidence")
            
            return emerging_narratives
            
        except Exception as e:
            self.logger.error(f"Error detecting emerging narratives: {e}")
            return []
    
    async def _signal_processing_loop(self):
        """Continuously process incoming narrative signals"""
        while True:
            try:
                await self._process_social_signals()
                await self._process_onchain_signals()
                await self._process_deployment_signals()
                
                await asyncio.sleep(self.config['signal_processing_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in signal processing loop: {e}")
                await asyncio.sleep(self.config['signal_processing_interval'])
    
    async def _trend_analysis_loop(self):
        """Continuously analyze narrative trends"""
        while True:
            try:
                await self._analyze_narrative_trends()
                await self._update_trend_sustainability()
                
                self.trends_detected += 1
                await asyncio.sleep(self.config['trend_analysis_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in trend analysis loop: {e}")
                await asyncio.sleep(self.config['trend_analysis_interval'])
    
    async def _capital_allocation_loop(self):
        """Continuously update capital allocation recommendations"""
        while True:
            try:
                recommendations = await self.get_capital_allocation_recommendations()
                await self._generate_swarm_deployment_signals(recommendations)
                
                await asyncio.sleep(self.config['capital_allocation_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in capital allocation loop: {e}")
                await asyncio.sleep(self.config['capital_allocation_interval'])
    
    async def _memetic_learning_loop(self):
        """Continuously learn from memetic patterns and outcomes"""
        while True:
            try:
                await self._update_narrative_models()
                await self._validate_predictions()
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                self.logger.error(f"Error in memetic learning loop: {e}")
                await asyncio.sleep(3600)
    
    # === IMPLEMENTATION METHODS ===
    
    async def _initialize_narrative_baselines(self):
        """Initialize baseline narrative trends"""
        for category in NarrativeCategory:
            if category != NarrativeCategory.UNKNOWN:
                self.active_narratives[category] = NarrativeTrend(
                    category=category,
                    strength=NarrativeStrength.WEAK,
                    momentum_score=0.1,
                    velocity=0.0,
                    duration_hours=0.0,
                    peak_strength=0.1,
                    capital_flow_score=0.0,
                    social_velocity=0.0,
                    deployment_rate=0.0,
                    cross_platform_sync=0.5,
                    sustainability_score=0.5,
                    recommended_allocation=0.0,
                    last_updated=datetime.now()
                )
    
    async def _calculate_category_scores(self, name: str, symbol: str, description: str, metadata: Dict[str, Any]) -> Dict[NarrativeCategory, float]:
        """Calculate narrative category scores for a token"""
        scores = {category: 0.0 for category in NarrativeCategory}
        
        # Keyword-based classification (simplified)
        keywords = {
            NarrativeCategory.CAT_COINS: ['cat', 'kitty', 'feline', 'meow', 'purr'],
            NarrativeCategory.DOG_COINS: ['dog', 'doge', 'shiba', 'puppy', 'woof'],
            NarrativeCategory.AI_THEMES: ['ai', 'artificial', 'intelligence', 'neural', 'machine'],
            NarrativeCategory.POLITICAL_FINANCE: ['trump', 'biden', 'politics', 'election', 'vote'],
            NarrativeCategory.GAMING_METAVERSE: ['game', 'gaming', 'metaverse', 'nft', 'play'],
            NarrativeCategory.ENVIRONMENTAL: ['green', 'eco', 'climate', 'carbon', 'sustainable']
        }
        
        text = f"{name} {symbol} {description}".lower()
        
        for category, category_keywords in keywords.items():
            for keyword in category_keywords:
                if keyword in text:
                    scores[category] += 0.3
        
        # Boost scores based on current narrative momentum
        for category in scores:
            if category in self.active_narratives:
                momentum_boost = self.active_narratives[category].momentum_score * 0.2
                scores[category] += momentum_boost
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {cat: score / max_score for cat, score in scores.items()}
        
        return scores
    
    async def _calculate_trend_alignment(self, primary_category: NarrativeCategory, secondary_categories: List[NarrativeCategory]) -> float:
        """Calculate how well aligned a token is with current trends"""
        if primary_category not in self.active_narratives:
            return 0.0
        
        primary_trend = self.active_narratives[primary_category]
        alignment = primary_trend.momentum_score * 0.7
        
        # Add alignment from secondary categories
        for secondary in secondary_categories:
            if secondary in self.active_narratives:
                alignment += self.active_narratives[secondary].momentum_score * 0.1
        
        return min(1.0, alignment)
    
    async def _predict_token_momentum(self, primary_category: NarrativeCategory, category_scores: Dict[NarrativeCategory, float], token_data: Dict[str, Any]) -> float:
        """Predict momentum potential for a token"""
        if primary_category not in self.active_narratives:
            return 0.0
        
        trend = self.active_narratives[primary_category]
        
        # Base momentum from category trend
        momentum = trend.momentum_score * category_scores[primary_category]
        
        # Adjust based on trend velocity
        momentum += trend.velocity * 0.3
        
        # Adjust based on sustainability
        momentum *= trend.sustainability_score
        
        return min(1.0, momentum)
    
    def _assess_narrative_risk(self, trend: NarrativeTrend) -> str:
        """Assess risk level of investing in a narrative"""
        if trend.duration_hours < 6:
            return "HIGH"  # Very new trends are high risk
        elif trend.sustainability_score < 0.3:
            return "HIGH"  # Unsustainable trends are high risk
        elif trend.momentum_score > 0.8 and trend.duration_hours > 24:
            return "MEDIUM"  # Mature trends with high momentum are medium risk
        else:
            return "LOW"
    
    def _generate_action_recommendation(self, trend: NarrativeTrend) -> str:
        """Generate action recommendation for a narrative"""
        if trend.strength == NarrativeStrength.VIRAL:
            return "SCALE_DOWN"  # Peak reached, consider scaling down
        elif trend.strength == NarrativeStrength.STRONG and trend.velocity > 0.3:
            return "SCALE_UP"  # Strong with good velocity, scale up
        elif trend.strength in [NarrativeStrength.MODERATE, NarrativeStrength.WEAK] and trend.velocity > 0.5:
            return "EARLY_ENTRY"  # Emerging trend, early entry opportunity
        else:
            return "MONITOR"  # Monitor for changes
    
    def _predict_narrative_peak_time(self, trend: NarrativeTrend, velocity_acceleration: float) -> float:
        """Predict when a narrative will reach its peak"""
        # Simple prediction based on current momentum and acceleration
        if velocity_acceleration <= 0:
            return float('inf')  # No predicted peak
        
        time_to_peak = (0.9 - trend.momentum_score) / velocity_acceleration
        return max(1.0, min(72.0, time_to_peak))  # Between 1-72 hours
    
    # Placeholder methods for data processing (to be implemented with real data sources)
    async def _process_social_signals(self):
        """Process social media signals"""
        self.total_signals_processed += 1
        pass
    
    async def _process_onchain_signals(self):
        """Process on-chain flow signals"""
        pass
    
    async def _process_deployment_signals(self):
        """Process contract deployment signals"""
        pass
    
    async def _analyze_narrative_trends(self):
        """Analyze and update narrative trends"""
        pass
    
    async def _update_trend_sustainability(self):
        """Update trend sustainability scores"""
        pass
    
    async def _generate_swarm_deployment_signals(self, recommendations: Dict[NarrativeCategory, Dict[str, Any]]):
        """Generate deployment signals for the swarm"""
        pass
    
    async def _update_narrative_models(self):
        """Update narrative prediction models"""
        pass
    
    async def _validate_predictions(self):
        """Validate previous predictions and update accuracy"""
        pass
    
    def get_narrative_status(self) -> Dict[str, Any]:
        """Get comprehensive narrative intelligence status"""
        return {
            'active_narratives': len(self.active_narratives),
            'total_signals_processed': self.total_signals_processed,
            'trends_detected': self.trends_detected,
            'successful_predictions': self.successful_predictions,
            'prediction_accuracy': self.successful_predictions / max(self.trends_detected, 1),
            'capital_allocation_accuracy': self.capital_allocation_accuracy,
            'dominant_narratives': [
                {'category': cat.value, 'strength': trend.strength.value, 'momentum': trend.momentum_score}
                for cat, trend in self.active_narratives.items()
                if trend.strength in [NarrativeStrength.STRONG, NarrativeStrength.DOMINANT, NarrativeStrength.VIRAL]
            ],
            'memetic_intelligence_active': True,
            'cultural_awareness_level': 'ENHANCED'
        } 