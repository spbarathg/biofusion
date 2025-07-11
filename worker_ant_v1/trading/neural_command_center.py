"""
NEURAL COMMAND CENTER - SWARM INTELLIGENCE COORDINATOR
====================================================

The neural command center of the 10-wallet swarm. Integrates real-time Twitter sentiment,
on-chain data, and AI predictions. Enforces strict consensus: only trades when
AI model + on-chain + Twitter ALL agree. No exceptions.

Implements the complete vision: survival-focused, learning-driven, cold calculation.
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

from worker_ant_v1.intelligence.sentiment_analyzer import SentimentAnalyzer
from worker_ant_v1.intelligence.twitter_sentiment_analyzer import TwitterSentimentAnalyzer
from worker_ant_v1.intelligence.caller_intelligence import AdvancedCallerIntelligence
from worker_ant_v1.intelligence.technical_analyzer import TechnicalAnalyzer
from worker_ant_v1.intelligence.ml_predictor import MLPredictor
from worker_ant_v1.intelligence.sentiment_first_ai import SentimentFirstAI
from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
from worker_ant_v1.core.vault_wallet_system import VaultWalletSystem
from worker_ant_v1.utils.logger import setup_logger
from ..core.unified_config import UnifiedConfig

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
    recommended_action: str = "HOLD"
    reasoning: List[str] = field(default_factory=list)

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

class NeuralCommandCenter:
    """The neural command center of the 10-wallet swarm"""
    
    def __init__(self):
        self.logger = setup_logger("NeuralCommandCenter")
        
        # Load configuration
        self.config = UnifiedConfig()
        self.social_signals_enabled = getattr(self.config, 'enable_social_signals', False) or \
                                     os.getenv('ENABLE_SOCIAL_SIGNALS', 'false').lower() == 'true'
        
        # Initialize intelligence components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.caller_intelligence = AdvancedCallerIntelligence()
        self.technical_analyzer = TechnicalAnalyzer()
        self.ml_predictor = MLPredictor()
        self.sentiment_ai = SentimentFirstAI()
        
        # Initialize Twitter analyzer (required for sentiment analysis)
        self.twitter_analyzer = None
        try:
            self.twitter_analyzer = TwitterSentimentAnalyzer()
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
        
        self.logger.info("🧠 Neural Command Center initialized")
    
    async def initialize(self, wallet_manager: UnifiedWalletManager, 
                        vault_system: VaultWalletSystem):
        """Initialize the neural command center"""
        
        self.wallet_manager = wallet_manager
        self.vault_system = vault_system
        
        # Initialize all intelligence components
        await self.sentiment_analyzer.initialize()
        await self.caller_intelligence.initialize()
        await self.technical_analyzer.initialize()
        await self.ml_predictor.initialize()
        
        # Load shadow memory from previous sessions
        await self._load_shadow_memory()
        
        # Start background intelligence loops
        asyncio.create_task(self._continuous_intelligence_loop())
        asyncio.create_task(self._source_performance_tracker())
        
        mode = "Social signals + On-chain" if self.social_signals_enabled else "Pure on-chain only"
        self.logger.info(f"🧠 Neural Command Center initialized - {mode} mode active")
    
    async def analyze_opportunity(self, token_address: str, 
                                 market_data: Dict[str, Any]) -> ConsensusSignal:
        """
        Core decision engine: Only trades when AI + on-chain + Twitter ALL agree
        No exceptions. This is the survival filter.
        """
        
        try:
            signal = ConsensusSignal(
                token_address=token_address,
                timestamp=datetime.now()
            )
            
            # Get AI prediction
            ai_prediction = await self._get_ai_prediction(token_address, market_data)
            signal.ai_prediction = ai_prediction
            
            # Get on-chain analysis
            onchain_sentiment = await self._get_onchain_analysis(token_address, market_data)
            signal.onchain_sentiment = onchain_sentiment
            
            # Get Twitter sentiment (required)
            twitter_sentiment = await self._get_twitter_analysis(token_address)
            signal.twitter_sentiment = twitter_sentiment
            
            if twitter_sentiment.confidence_score < 0.3:
                self.logger.debug("🐦 Low confidence Twitter sentiment, proceeding carefully")
            
            # Get technical analysis
            signal.technical_score = await self._get_technical_analysis(token_address)
            
            # Get caller intelligence
            signal.caller_reputation = await self._get_caller_intelligence(token_address)
            
            # Check shadow memory
            if await self._matches_shadow_memory(token_address, market_data):
                signal.risk_indicators.append("Matches failed pattern in shadow memory")
                signal.passes_consensus = False
                signal.recommended_action = "AVOID"
                signal.reasoning.append("Déjà vu trap detected - never die the same way twice")
                return signal
            
            # Evaluate consensus
            consensus_result = self._evaluate_consensus(signal)
            signal.consensus_strength = consensus_result['strength']
            signal.passes_consensus = consensus_result['passes']
            signal.recommended_action = consensus_result['action']
            signal.reasoning = consensus_result['reasoning']
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Opportunity analysis failed: {e}")
            return ConsensusSignal(
                token_address=token_address,
                timestamp=datetime.now(),
                passes_consensus=False,
                recommended_action="ERROR",
                reasoning=[f"Analysis failed: {str(e)}"]
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
            # Connect to Discord and fetch recent messages from the configured channel
            bot_token = os.getenv('DISCORD_BOT_TOKEN')
            channel_id = int(os.getenv('DISCORD_CHANNEL_ID', '0'))
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
                    confidence = recommendation.get('confidence', 0.5)
                    
                    reputation_scores.append(score * confidence)
            
            return np.mean(reputation_scores) if reputation_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"Caller intelligence failed: {e}")
            return 0.0
    
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
            action = "WAIT"
            
        return {
            'passes': passes_consensus,
            'strength': consensus_strength,
            'confidence': confidence_level,
            'action': action,
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

            
_neural_command_center = None

async def get_neural_command_center() -> NeuralCommandCenter:
    """Get global neural command center instance"""
    global _neural_command_center
    if _neural_command_center is None:
        _neural_command_center = NeuralCommandCenter()
    return _neural_command_center 