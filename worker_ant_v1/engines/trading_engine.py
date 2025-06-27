"""
SMART TRADING ENGINE - DUAL CONFIRMATION SYSTEM
===============================================

Advanced trading engine that only enters trades when BOTH AI logic and 
on-chain signals confirm. Implements smart entry conditions to avoid 
hype-driven noise and bot manipulation.
"""

import asyncio
import time
import json
import uuid
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import numpy as np

# Internal imports
from worker_ant_v1.core.simple_config import get_trading_config, get_security_config
from worker_ant_v1.utils.simple_logger import setup_logger as trading_logger
from worker_ant_v1.trading.order_buyer import ProductionBuyer, BuySignal, BuyResult
from worker_ant_v1.trading.order_seller import ProductionSeller
from worker_ant_v1.trading.market_scanner import ProductionScanner, TradingOpportunity
from worker_ant_v1.intelligence.ml_predictor import MLPredictor, MLPrediction

# Create signal trust and memory pattern classes with full implementation
class SignalTrust:
    """Advanced signal trust management system"""
    
    def __init__(self, trust_level: float = 0.5):
        self.trust_level = max(0.0, min(1.0, trust_level))  # Clamp between 0 and 1
        self.signal_history = []
        self.accuracy_tracker = {}
        self.confidence_multiplier = 1.0
        
    def update_trust(self, signal_type: str, predicted_outcome: bool, actual_outcome: bool):
        """Update trust level based on signal accuracy"""
        if signal_type not in self.accuracy_tracker:
            self.accuracy_tracker[signal_type] = {'correct': 0, 'total': 0}
        
        self.accuracy_tracker[signal_type]['total'] += 1
        if predicted_outcome == actual_outcome:
            self.accuracy_tracker[signal_type]['correct'] += 1
            
        # Calculate new trust level
        accuracy = self.accuracy_tracker[signal_type]['correct'] / self.accuracy_tracker[signal_type]['total']
        self.trust_level = (self.trust_level * 0.7) + (accuracy * 0.3)  # Weighted average
        
    def get_adjusted_confidence(self, base_confidence: float, signal_type: str) -> float:
        """Get confidence adjusted by trust level"""
        type_accuracy = 1.0
        if signal_type in self.accuracy_tracker and self.accuracy_tracker[signal_type]['total'] > 0:
            type_accuracy = self.accuracy_tracker[signal_type]['correct'] / self.accuracy_tracker[signal_type]['total']
            
        return base_confidence * self.trust_level * type_accuracy * self.confidence_multiplier
        
    def is_signal_trustworthy(self, confidence: float, signal_type: str, threshold: float = 0.6) -> bool:
        """Check if a signal meets trustworthiness threshold"""
        adjusted_confidence = self.get_adjusted_confidence(confidence, signal_type)
        return adjusted_confidence >= threshold


class MemoryPattern:
    """Pattern recognition and memory management for trading signals"""
    
    def __init__(self, pattern_type: str = "default", max_patterns: int = 1000):
        self.pattern_type = pattern_type
        self.max_patterns = max_patterns
        self.patterns = {}
        self.pattern_success_rates = {}
        self.recent_patterns = []
        
    def add_pattern(self, pattern_key: str, context: dict, outcome: bool):
        """Add a new pattern to memory"""
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = []
            self.pattern_success_rates[pattern_key] = {'successes': 0, 'total': 0}
            
        self.patterns[pattern_key].append({
            'context': context,
            'outcome': outcome,
            'timestamp': datetime.now()
        })
        
        # Update success rate
        self.pattern_success_rates[pattern_key]['total'] += 1
        if outcome:
            self.pattern_success_rates[pattern_key]['successes'] += 1
            
        # Add to recent patterns
        self.recent_patterns.append({
            'key': pattern_key,
            'context': context,
            'outcome': outcome,
            'timestamp': datetime.now()
        })
        
        # Cleanup old patterns if we exceed max
        if len(self.recent_patterns) > self.max_patterns:
            self.recent_patterns = self.recent_patterns[-self.max_patterns:]
            
    def get_pattern_success_rate(self, pattern_key: str) -> float:
        """Get success rate for a specific pattern"""
        if pattern_key not in self.pattern_success_rates:
            return 0.5  # Default neutral rate
            
        stats = self.pattern_success_rates[pattern_key]
        if stats['total'] == 0:
            return 0.5
            
        return stats['successes'] / stats['total']
        
    def find_similar_patterns(self, context: dict, similarity_threshold: float = 0.8) -> list:
        """Find patterns similar to the given context"""
        similar_patterns = []
        
        for pattern_key, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                similarity = self._calculate_similarity(context, pattern['context'])
                if similarity >= similarity_threshold:
                    similar_patterns.append({
                        'key': pattern_key,
                        'similarity': similarity,
                        'success_rate': self.get_pattern_success_rate(pattern_key),
                        'pattern': pattern
                    })
                    
        return sorted(similar_patterns, key=lambda x: x['similarity'], reverse=True)
        
    def _calculate_similarity(self, context1: dict, context2: dict) -> float:
        """Calculate similarity between two contexts"""
        if not context1 or not context2:
            return 0.0
            
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
            
        matches = 0
        total = len(common_keys)
        
        for key in common_keys:
            if context1[key] == context2[key]:
                matches += 1
            elif isinstance(context1[key], (int, float)) and isinstance(context2[key], (int, float)):
                # For numeric values, consider them similar if within 10%
                diff = abs(context1[key] - context2[key])
                avg = (context1[key] + context2[key]) / 2
                if avg > 0 and diff / avg <= 0.1:
                    matches += 0.8
                    
        return matches / total if total > 0 else 0.0
        
    def get_pattern_confidence(self, context: dict) -> float:
        """Get confidence based on historical patterns"""
        similar_patterns = self.find_similar_patterns(context)
        
        if not similar_patterns:
            return 0.5  # Neutral confidence if no patterns found
            
        # Weight by similarity and success rate
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for pattern in similar_patterns[:5]:  # Top 5 most similar
            weight = pattern['similarity']
            confidence = pattern['success_rate']
            weighted_confidence += confidence * weight
            total_weight += weight
            
        return weighted_confidence / total_weight if total_weight > 0 else 0.5

class SignalType(Enum):
    AI_PREDICTION = "ai_prediction"
    VOLUME_SPIKE = "volume_spike"
    LIQUIDITY_FLOW = "liquidity_flow"
    SOCIAL_SENTIMENT = "social_sentiment"
    TECHNICAL_BREAKOUT = "technical_breakout"
    WHALE_ACTIVITY = "whale_activity"
    CREATOR_VERIFICATION = "creator_verification"
    CONTRACT_SAFETY = "contract_safety"

class SignalStrength(Enum):
    WEAK = "weak"           # 0.0 - 0.3
    MODERATE = "moderate"   # 0.3 - 0.6
    STRONG = "strong"       # 0.6 - 0.8
    VERY_STRONG = "very_strong"  # 0.8 - 1.0

class EntryCondition(Enum):
    WAITING = "waiting"
    AI_CONFIRMED = "ai_confirmed"
    ONCHAIN_CONFIRMED = "onchain_confirmed"
    DUAL_CONFIRMED = "dual_confirmed"
    REJECTED = "rejected"

@dataclass
class TradingSignal:
    """Individual trading signal with metadata"""
    
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    token_address: str
    
    # Signal-specific data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    source: str = ""
    expires_at: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if signal is still valid"""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return self.strength > 0.1 and self.confidence > 0.1
    
    def get_signal_strength_category(self) -> SignalStrength:
        """Get signal strength category"""
        if self.strength >= 0.8:
            return SignalStrength.VERY_STRONG
        elif self.strength >= 0.6:
            return SignalStrength.STRONG
        elif self.strength >= 0.3:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

@dataclass
class DualConfirmation:
    """Dual confirmation entry system"""
    
    token_address: str
    ai_signals: List[TradingSignal] = field(default_factory=list)
    onchain_signals: List[TradingSignal] = field(default_factory=list)
    
    ai_score: float = 0.0
    onchain_score: float = 0.0
    combined_score: float = 0.0
    
    status: EntryCondition = EntryCondition.WAITING
    confirmation_time: Optional[datetime] = None
    
    # Thresholds
    ai_threshold: float = 0.6
    onchain_threshold: float = 0.6
    combined_threshold: float = 0.7
    
    def update_scores(self):
        """Update AI and on-chain scores"""
        # Calculate AI score
        if self.ai_signals:
            ai_weights = [s.strength * s.confidence for s in self.ai_signals if s.is_valid()]
            self.ai_score = sum(ai_weights) / len(ai_weights) if ai_weights else 0.0
        
        # Calculate on-chain score
        if self.onchain_signals:
            onchain_weights = [s.strength * s.confidence for s in self.onchain_signals if s.is_valid()]
            self.onchain_score = sum(onchain_weights) / len(onchain_weights) if onchain_weights else 0.0
        
        # Calculate combined score (weighted average)
        self.combined_score = (self.ai_score * 0.4 + self.onchain_score * 0.6)
        
        # Update status
        self._update_status()
    
    def _update_status(self):
        """Update confirmation status"""
        ai_confirmed = self.ai_score >= self.ai_threshold
        onchain_confirmed = self.onchain_score >= self.onchain_threshold
        
        if ai_confirmed and onchain_confirmed and self.combined_score >= self.combined_threshold:
            self.status = EntryCondition.DUAL_CONFIRMED
            if not self.confirmation_time:
                self.confirmation_time = datetime.now()
        elif ai_confirmed and not onchain_confirmed:
            self.status = EntryCondition.AI_CONFIRMED
        elif onchain_confirmed and not ai_confirmed:
            self.status = EntryCondition.ONCHAIN_CONFIRMED
        else:
            self.status = EntryCondition.WAITING
    
    def should_execute_trade(self) -> bool:
        """Check if dual confirmation criteria are met"""
        return self.status == EntryCondition.DUAL_CONFIRMED

class AntiHypeFilter:
    """Filter to avoid hype-driven noise and manipulation"""
    
    def __init__(self):
        self.logger = logging.getLogger("AntiHypeFilter")
        
        # Hype keywords to filter out
        self.hype_keywords = {
            'immediate_reject': [
                'lp live now', 'just launched', 'pump it', '100x gem',
                'to the moon', 'diamond hands', 'ape in now', 'fomo',
                'last chance', 'going viral', 'next shib', 'next doge'
            ],
            'suspicious_patterns': [
                'airdrop', 'free tokens', 'giveaway', 'presale',
                'whitelist', 'exclusive', 'vip access', 'insider'
            ]
        }
        
        # Social media burst patterns
        self.burst_thresholds = {
            'mentions_per_minute': 50,  # Too many mentions too fast
            'account_age_days': 30,     # Accounts too new
            'follower_ratio': 0.1       # Follower to following ratio
        }
        
    async def evaluate_token(self, token_address: str, token_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate if token passes anti-hype filter"""
        
        # Check for hype keywords in name/symbol
        token_name = token_data.get('name', '').lower()
        token_symbol = token_data.get('symbol', '').lower()
        
        for keyword in self.hype_keywords['immediate_reject']:
            if keyword in token_name or keyword in token_symbol:
                return False, f"Hype keyword detected: {keyword}"
        
        # Check social media patterns
        social_data = token_data.get('social_media', {})
        
        # Sudden social media burst
        mentions_growth = social_data.get('mentions_24h_growth', 0)
        if mentions_growth > 1000:  # 1000% growth in mentions
            return False, "Suspicious social media burst"
        
        # Check for bot-like social activity
        mention_velocity = social_data.get('mentions_per_minute', 0)
        if mention_velocity > self.burst_thresholds['mentions_per_minute']:
            return False, "Bot-like mention velocity"
        
        # Check creator wallet patterns
        creator_data = token_data.get('creator_info', {})
        wallet_age_days = creator_data.get('wallet_age_days', 0)
        
        if wallet_age_days < 7:  # Very new wallet
            return False, "Creator wallet too new"
        
        # Check for coordinated promotion
        if self._detect_coordinated_promotion(social_data):
            return False, "Coordinated promotion detected"
        
        return True, "Passed anti-hype filter"
    
    def _detect_coordinated_promotion(self, social_data: Dict[str, Any]) -> bool:
        """Detect coordinated social media promotion"""
        
        # Check for synchronized posting times
        post_times = social_data.get('recent_post_times', [])
        if len(post_times) > 10:
            # Check if posts are too synchronized (within 5-minute windows)
            time_clusters = []
            for post_time in post_times:
                clustered = False
                for cluster in time_clusters:
                    if abs(post_time - cluster[0]) < 300:  # 5 minutes
                        cluster.append(post_time)
                        clustered = True
                        break
                if not clustered:
                    time_clusters.append([post_time])
            
            # If more than 20% of posts are in tight clusters
            clustered_posts = sum(len(cluster) for cluster in time_clusters if len(cluster) > 1)
            if clustered_posts / len(post_times) > 0.2:
                return True
        
        # Check for similar posting patterns
        post_similarities = social_data.get('post_similarity_score', 0)
        if post_similarities > 0.8:  # Very similar posts
            return True
        
        return False

class SmartEntryEngine:
    """Smart entry engine with dual confirmation"""
    
    def __init__(self, signal_trust: SignalTrust):
        self.logger = logging.getLogger("SmartEntryEngine")
        self.signal_trust = signal_trust
        
        # Components
        self.ml_predictor = MLPredictor()
        self.anti_hype_filter = AntiHypeFilter()
        
        # Confirmation tracking
        self.pending_confirmations: Dict[str, DualConfirmation] = {}
        self.signal_history: List[TradingSignal] = []
        
        # Timing controls
        self.min_confirmation_delay_ms = 500   # Minimum delay for confirmation
        self.max_confirmation_wait_ms = 30000  # Maximum wait for dual confirmation
        
        # Quality thresholds
        self.min_liquidity_sol = 50.0
        self.min_volume_24h_sol = 500.0
        self.max_manipulation_score = 0.3
        
    async def evaluate_trading_opportunity(self, opportunity: TradingOpportunity) -> Optional[BuySignal]:
        """Evaluate opportunity using dual confirmation system"""
        
        token_address = opportunity.token_address
        
        self.logger.info(f"🧠 Evaluating opportunity: {opportunity.token_symbol}")
        
        # Step 1: Anti-hype filter
        token_data = await self._get_token_data(token_address)
        hype_passed, hype_reason = await self.anti_hype_filter.evaluate_token(token_address, token_data)
        
        if not hype_passed:
            self.logger.info(f"❌ Hype filter rejected: {hype_reason}")
            return None
        
        # Step 2: Basic quality checks
        if not self._passes_quality_checks(opportunity):
            self.logger.info(f"❌ Quality checks failed")
            return None
        
        # Step 3: Generate AI signals
        ai_signals = await self._generate_ai_signals(opportunity, token_data)
        
        # Step 4: Generate on-chain signals
        onchain_signals = await self._generate_onchain_signals(opportunity, token_data)
        
        # Step 5: Create dual confirmation
        confirmation = DualConfirmation(
            token_address=token_address,
            ai_signals=ai_signals,
            onchain_signals=onchain_signals
        )
        
        confirmation.update_scores()
        
        # Step 6: Check for dual confirmation
        if confirmation.should_execute_trade():
            self.logger.info(f"✅ DUAL CONFIRMATION ACHIEVED for {opportunity.token_symbol}")
            return await self._create_buy_signal(opportunity, confirmation)
        else:
            self.logger.info(f"⏳ Waiting for confirmation: AI={confirmation.ai_score:.2f}, OnChain={confirmation.onchain_score:.2f}")
            
            # Store for potential later confirmation
            self.pending_confirmations[token_address] = confirmation
            return None
    
    async def _generate_ai_signals(self, opportunity: TradingOpportunity, token_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate AI-based trading signals"""
        signals = []
        
        # ML prediction signal
        try:
            ml_prediction = await self.ml_predictor.predict_token_price(
                opportunity.token_symbol,
                opportunity.current_price_sol,
                token_data.get('sentiment_data', {}),
                token_data.get('technical_data', {})
            )
            
            ml_signal = TradingSignal(
                signal_type=SignalType.AI_PREDICTION,
                strength=abs(ml_prediction.ml_signal),
                confidence=ml_prediction.confidence,
                timestamp=datetime.now(),
                token_address=opportunity.token_address,
                data={'prediction': ml_prediction},
                source='ml_predictor'
            )
            signals.append(ml_signal)
            
        except Exception as e:
            self.logger.warning(f"ML prediction failed: {e}")
        
        # Technical analysis signal
        technical_score = self._calculate_technical_score(token_data.get('technical_data', {}))
        if technical_score > 0.1:
            tech_signal = TradingSignal(
                signal_type=SignalType.TECHNICAL_BREAKOUT,
                strength=technical_score,
                confidence=0.8,
                timestamp=datetime.now(),
                token_address=opportunity.token_address,
                source='technical_analysis'
            )
            signals.append(tech_signal)
        
        # Social sentiment signal
        sentiment_score = self._calculate_sentiment_score(token_data.get('sentiment_data', {}))
        if sentiment_score > 0.1:
            sentiment_signal = TradingSignal(
                signal_type=SignalType.SOCIAL_SENTIMENT,
                strength=sentiment_score,
                confidence=0.6,
                timestamp=datetime.now(),
                token_address=opportunity.token_address,
                source='sentiment_analysis'
            )
            signals.append(sentiment_signal)
        
        return signals
    
    async def _generate_onchain_signals(self, opportunity: TradingOpportunity, token_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate on-chain trading signals"""
        signals = []
        
        # Volume spike signal
        volume_score = self._calculate_volume_spike_score(token_data)
        if volume_score > 0.1:
            volume_signal = TradingSignal(
                signal_type=SignalType.VOLUME_SPIKE,
                strength=volume_score,
                confidence=0.9,
                timestamp=datetime.now(),
                token_address=opportunity.token_address,
                source='volume_analysis'
            )
            signals.append(volume_signal)
        
        # Liquidity flow signal
        liquidity_score = self._calculate_liquidity_flow_score(token_data)
        if liquidity_score > 0.1:
            liquidity_signal = TradingSignal(
                signal_type=SignalType.LIQUIDITY_FLOW,
                strength=liquidity_score,
                confidence=0.85,
                timestamp=datetime.now(),
                token_address=opportunity.token_address,
                source='liquidity_analysis'
            )
            signals.append(liquidity_signal)
        
        # Whale activity signal
        whale_score = self._calculate_whale_activity_score(token_data)
        if whale_score > 0.1:
            whale_signal = TradingSignal(
                signal_type=SignalType.WHALE_ACTIVITY,
                strength=whale_score,
                confidence=0.7,
                timestamp=datetime.now(),
                token_address=opportunity.token_address,
                source='whale_tracking'
            )
            signals.append(whale_signal)
        
        # Contract safety signal
        safety_score = self._calculate_contract_safety_score(token_data)
        if safety_score > 0.5:  # Must pass safety threshold
            safety_signal = TradingSignal(
                signal_type=SignalType.CONTRACT_SAFETY,
                strength=safety_score,
                confidence=0.95,
                timestamp=datetime.now(),
                token_address=opportunity.token_address,
                source='contract_analysis'
            )
            signals.append(safety_signal)
        
        return signals
    
    def _passes_quality_checks(self, opportunity: TradingOpportunity) -> bool:
        """Basic quality checks for trading opportunity"""
        
        # Liquidity check
        if opportunity.liquidity_sol < self.min_liquidity_sol:
            return False
        
        # Volume check
        if opportunity.volume_24h_sol < self.min_volume_24h_sol:
            return False
        
        # Security check
        if opportunity.rug_probability > 0.7:
            return False
        
        # Confidence check
        if opportunity.confidence_score < 0.4:
            return False
        
        return True
    
    def _calculate_technical_score(self, technical_data: Dict[str, Any]) -> float:
        """Calculate technical analysis score"""
        score = 0.0
        
        # RSI analysis
        rsi = technical_data.get('rsi', 50)
        if 30 <= rsi <= 40:  # Oversold but not extreme
            score += 0.3
        elif 60 <= rsi <= 70:  # Momentum building
            score += 0.2
        
        # MACD analysis
        macd_signal = technical_data.get('macd_signal', 0)
        if macd_signal > 0:  # Bullish signal
            score += 0.3
        
        # Volume trend
        volume_trend = technical_data.get('volume_trend', 0)
        if volume_trend > 0.2:  # Increasing volume
            score += 0.4
        
        return min(1.0, score)
    
    def _calculate_sentiment_score(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate social sentiment score"""
        score = 0.0
        
        # Overall sentiment
        sentiment = sentiment_data.get('overall_sentiment', 0)
        if sentiment > 0.6:
            score += 0.4
        
        # Mention velocity (organic growth)
        mention_growth = sentiment_data.get('organic_mention_growth', 0)
        if 0.2 < mention_growth < 2.0:  # Healthy growth, not explosion
            score += 0.3
        
        # Influencer mentions
        influencer_mentions = sentiment_data.get('influencer_mentions', 0)
        if influencer_mentions > 0:
            score += 0.3
        
        return min(1.0, score)
    
    def _calculate_volume_spike_score(self, token_data: Dict[str, Any]) -> float:
        """Calculate volume spike score"""
        volume_data = token_data.get('volume_data', {})
        
        # 1-hour volume spike
        hour_spike = volume_data.get('1h_volume_spike', 0)
        if hour_spike > 5.0:  # 5x volume increase
            return min(1.0, hour_spike / 10.0)
        
        return 0.0
    
    def _calculate_liquidity_flow_score(self, token_data: Dict[str, Any]) -> float:
        """Calculate liquidity flow score"""
        liquidity_data = token_data.get('liquidity_data', {})
        
        # Net liquidity flow
        net_flow = liquidity_data.get('net_flow_1h', 0)
        if net_flow > 0:  # Positive flow
            return min(1.0, net_flow / 100.0)  # Normalize by 100 SOL
        
        return 0.0
    
    def _calculate_whale_activity_score(self, token_data: Dict[str, Any]) -> float:
        """Calculate whale activity score"""
        whale_data = token_data.get('whale_data', {})
        
        # Recent whale buys
        whale_buys = whale_data.get('large_buys_1h', 0)
        whale_sells = whale_data.get('large_sells_1h', 0)
        
        # Positive whale activity
        if whale_buys > whale_sells:
            return min(1.0, (whale_buys - whale_sells) / 5.0)
        
        return 0.0
    
    def _calculate_contract_safety_score(self, token_data: Dict[str, Any]) -> float:
        """Calculate contract safety score"""
        safety_data = token_data.get('safety_data', {})
        
        score = 0.0
        
        # Mint authority renounced
        if safety_data.get('mint_authority_renounced', False):
            score += 0.3
        
        # Liquidity locked
        if safety_data.get('liquidity_locked', False):
            score += 0.3
        
        # No suspicious functions
        if not safety_data.get('has_suspicious_functions', True):
            score += 0.2
        
        # Verified contract
        if safety_data.get('contract_verified', False):
            score += 0.2
        
        return score
    
    async def _create_buy_signal(self, opportunity: TradingOpportunity, confirmation: DualConfirmation) -> BuySignal:
        """Create optimized buy signal from confirmed opportunity"""
        
        # Calculate optimal trade size based on confirmation strength
        base_amount = 0.1  # Base trade size in SOL
        confidence_multiplier = confirmation.combined_score
        
        # Apply signal trust weights
        trust_multiplier = (
            self.signal_trust.technical_analysis * 0.3 +
            self.signal_trust.ml_predictions * 0.3 +
            self.signal_trust.volume_analysis * 0.4
        )
        
        # Calculate final trade amount
        trade_amount = base_amount * confidence_multiplier * trust_multiplier
        trade_amount = max(0.05, min(0.25, trade_amount))  # Clamp between 0.05 and 0.25 SOL
        
        # Calculate optimal slippage based on urgency
        base_slippage = 1.0  # 1%
        urgency_multiplier = opportunity.urgency_score
        max_slippage = base_slippage * (1 + urgency_multiplier)
        max_slippage = min(3.0, max_slippage)  # Cap at 3%
        
        return BuySignal(
            token_address=opportunity.token_address,
            confidence=confirmation.combined_score,
            urgency=opportunity.urgency_score,
            max_slippage=max_slippage,
            amount_sol=trade_amount,
            token_symbol=opportunity.token_symbol,
            priority_fee_multiplier=1.5 if confirmation.combined_score > 0.8 else 1.0
        )
    
    async def _get_token_data(self, token_address: str) -> Dict[str, Any]:
        """Get comprehensive token data for analysis"""
        # This would integrate with real data sources
        return {
            'sentiment_data': {},
            'technical_data': {},
            'volume_data': {},
            'liquidity_data': {},
            'whale_data': {},
            'safety_data': {}
        }

# Export main classes
__all__ = ['SmartEntryEngine', 'DualConfirmation', 'AntiHypeFilter', 'TradingSignal'] 