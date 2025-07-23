"""
SWARM DECISION ENGINE - NARRATIVE-WEIGHTED OPPORTUNITY ANALYSIS
=============================================================

Central decision-making engine that analyzes trading opportunities
with narrative intelligence weighting and anti-fragile principles.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from worker_ant_v1.utils.logger import setup_logger


@dataclass
class OpportunitySignal:
    """Trading opportunity signal with confidence metrics"""
    token_address: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    price_target: Optional[float]
    risk_level: float
    narrative_alignment: float
    time_horizon: str  # 'short', 'medium', 'long'
    reasoning: str
    metadata: Dict[str, Any]


@dataclass
class SwarmConsensus:
    """Consensus result from swarm analysis"""
    recommended_action: str
    consensus_confidence: float
    narrative_weight: float
    risk_assessment: float
    position_size_recommendation: float
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    reasoning: str


class SwarmDecisionEngine:
    """
    Central decision engine that processes multiple signals and applies
    narrative weighting to determine optimal trading decisions.
    """
    
    def __init__(self):
        self.logger = setup_logger("SwarmDecisionEngine")
        
        # Signal processing configuration
        self.confidence_threshold = 0.6
        self.narrative_weight_multiplier = 1.5
        self.max_risk_per_trade = 0.02
        
        # Signal sources and weights
        self.signal_weights = {
            'technical_analysis': 0.25,
            'sentiment_analysis': 0.25, 
            'narrative_analysis': 0.30,
            'market_structure': 0.20
        }
        
        # Anti-fragile decision factors
        self.anti_fragile_factors = {
            'market_volatility_adjustment': True,
            'position_sizing_discipline': True,
            'narrative_momentum_bias': True,
            'risk_cascading_prevention': True
        }
        
        self.logger.info("🧠 Swarm Decision Engine initialized with narrative weighting")
    
    async def analyze_opportunity(
        self, 
        token_address: str, 
        market_data: Dict[str, Any],
        narrative_weight: float = 1.0
    ) -> SwarmConsensus:
        """
        Analyze trading opportunity with narrative intelligence weighting
        
        Args:
            token_address: Token contract address
            market_data: Current market data for the token
            narrative_weight: Narrative strength multiplier (0.0 to 2.0)
        """
        try:
            self.logger.debug(f"🔍 Analyzing opportunity for {token_address} with narrative weight {narrative_weight:.2f}")
            
            # Gather signals from multiple sources
            signals = await self._gather_signals(token_address, market_data)
            
            # Apply narrative weighting to signal confidence
            weighted_signals = self._apply_narrative_weighting(signals, narrative_weight)
            
            # Calculate consensus
            consensus = await self._calculate_swarm_consensus(weighted_signals, narrative_weight)
            
            # Apply anti-fragile decision filters
            final_consensus = await self._apply_anti_fragile_filters(consensus, market_data)
            
            self.logger.debug(f"📊 Analysis complete: {final_consensus.recommended_action} "
                            f"(confidence: {final_consensus.consensus_confidence:.2f})")
            
            return final_consensus
            
        except Exception as e:
            self.logger.error(f"❌ Opportunity analysis failed for {token_address}: {e}")
            return self._create_safe_consensus("hold", 0.0, narrative_weight)
    
    async def _gather_signals(self, token_address: str, market_data: Dict[str, Any]) -> List[OpportunitySignal]:
        """Gather signals from multiple analysis sources"""
        signals = []
        
        try:
            # Technical analysis signal
            technical_signal = await self._generate_technical_signal(token_address, market_data)
            if technical_signal:
                signals.append(technical_signal)
            
            # Sentiment analysis signal
            sentiment_signal = await self._generate_sentiment_signal(token_address, market_data)
            if sentiment_signal:
                signals.append(sentiment_signal)
            
            # Market structure signal
            structure_signal = await self._generate_structure_signal(token_address, market_data)
            if structure_signal:
                signals.append(structure_signal)
            
            # Narrative momentum signal
            narrative_signal = await self._generate_narrative_signal(token_address, market_data)
            if narrative_signal:
                signals.append(narrative_signal)
            
            return signals
            
        except Exception as e:
            self.logger.warning(f"Signal gathering failed for {token_address}: {e}")
            return []
    
    async def _generate_technical_signal(self, token_address: str, market_data: Dict[str, Any]) -> Optional[OpportunitySignal]:
        """Generate technical analysis signal"""
        try:
            # Extract technical indicators from market data
            price = market_data.get('current_price', 0)
            volume = market_data.get('volume_24h', 0)
            price_change_24h = market_data.get('price_change_24h', 0)
            
            # Simple momentum-based signal
            if price_change_24h > 0.05 and volume > 10000:  # 5% gain with volume
                signal_type = 'buy'
                confidence = min(0.8, abs(price_change_24h) * 10)
            elif price_change_24h < -0.05:  # 5% loss
                signal_type = 'sell'
                confidence = min(0.7, abs(price_change_24h) * 8)
            else:
                signal_type = 'hold'
                confidence = 0.5
            
            return OpportunitySignal(
                token_address=token_address,
                signal_type=signal_type,
                confidence=confidence,
                price_target=price * (1 + price_change_24h * 0.5),
                risk_level=abs(price_change_24h),
                narrative_alignment=0.5,  # Neutral for technical
                time_horizon='short',
                reasoning=f"Technical: {price_change_24h:.1%} change, volume {volume:,.0f}",
                metadata={'source': 'technical_analysis', 'price': price, 'volume': volume}
            )
            
        except Exception as e:
            self.logger.warning(f"Technical signal generation failed: {e}")
            return None
    
    async def _generate_sentiment_signal(self, token_address: str, market_data: Dict[str, Any]) -> Optional[OpportunitySignal]:
        """Generate sentiment analysis signal"""
        try:
            # Mock sentiment analysis - would integrate with real sentiment engine
            sentiment_score = market_data.get('sentiment_score', 0.5)
            social_volume = market_data.get('social_volume', 0)
            
            if sentiment_score > 0.7:
                signal_type = 'buy'
                confidence = sentiment_score * 0.9
            elif sentiment_score < 0.3:
                signal_type = 'sell'
                confidence = (1 - sentiment_score) * 0.8
            else:
                signal_type = 'hold'
                confidence = 0.4
            
            return OpportunitySignal(
                token_address=token_address,
                signal_type=signal_type,
                confidence=confidence,
                price_target=None,
                risk_level=1 - sentiment_score if sentiment_score < 0.5 else sentiment_score - 0.5,
                narrative_alignment=sentiment_score,
                time_horizon='medium',
                reasoning=f"Sentiment: {sentiment_score:.2f} score, social volume {social_volume}",
                metadata={'source': 'sentiment_analysis', 'sentiment': sentiment_score}
            )
            
        except Exception as e:
            self.logger.warning(f"Sentiment signal generation failed: {e}")
            return None
    
    async def _generate_structure_signal(self, token_address: str, market_data: Dict[str, Any]) -> Optional[OpportunitySignal]:
        """Generate market structure signal"""
        try:
            # Market structure analysis
            liquidity = market_data.get('liquidity_usd', 0)
            holder_count = market_data.get('holder_count', 0)
            mcap = market_data.get('market_cap', 0)
            
            # Structure score based on liquidity and distribution
            structure_score = 0.0
            if liquidity > 50000:  # Good liquidity
                structure_score += 0.3
            if holder_count > 1000:  # Good distribution
                structure_score += 0.3
            if 1000000 < mcap < 100000000:  # Sweet spot market cap
                structure_score += 0.4
            
            signal_type = 'buy' if structure_score > 0.6 else 'hold' if structure_score > 0.3 else 'sell'
            confidence = structure_score
            
            return OpportunitySignal(
                token_address=token_address,
                signal_type=signal_type,
                confidence=confidence,
                price_target=None,
                risk_level=1 - structure_score,
                narrative_alignment=0.5,
                time_horizon='long',
                reasoning=f"Structure: liquidity ${liquidity:,.0f}, {holder_count} holders",
                metadata={'source': 'market_structure', 'liquidity': liquidity}
            )
            
        except Exception as e:
            self.logger.warning(f"Structure signal generation failed: {e}")
            return None
    
    async def _generate_narrative_signal(self, token_address: str, market_data: Dict[str, Any]) -> Optional[OpportunitySignal]:
        """Generate narrative momentum signal"""
        try:
            # Narrative indicators
            trending_keywords = market_data.get('trending_keywords', [])
            narrative_mentions = market_data.get('narrative_mentions', 0)
            influencer_sentiment = market_data.get('influencer_sentiment', 0.5)
            
            # Calculate narrative momentum
            narrative_score = 0.0
            if len(trending_keywords) > 3:
                narrative_score += 0.3
            if narrative_mentions > 100:
                narrative_score += 0.3
            narrative_score += influencer_sentiment * 0.4
            
            signal_type = 'buy' if narrative_score > 0.7 else 'hold' if narrative_score > 0.4 else 'sell'
            confidence = narrative_score
            
            return OpportunitySignal(
                token_address=token_address,
                signal_type=signal_type,
                confidence=confidence,
                price_target=None,
                risk_level=0.3,  # Narrative trades are medium risk
                narrative_alignment=narrative_score,
                time_horizon='short',
                reasoning=f"Narrative: {len(trending_keywords)} keywords, {narrative_mentions} mentions",
                metadata={'source': 'narrative_analysis', 'keywords': trending_keywords}
            )
            
        except Exception as e:
            self.logger.warning(f"Narrative signal generation failed: {e}")
            return None
    
    def _apply_narrative_weighting(self, signals: List[OpportunitySignal], narrative_weight: float) -> List[OpportunitySignal]:
        """Apply narrative weighting to boost signal confidence"""
        weighted_signals = []
        
        for signal in signals:
            # Clone the signal
            weighted_signal = OpportunitySignal(
                token_address=signal.token_address,
                signal_type=signal.signal_type,
                confidence=signal.confidence,
                price_target=signal.price_target,
                risk_level=signal.risk_level,
                narrative_alignment=signal.narrative_alignment,
                time_horizon=signal.time_horizon,
                reasoning=signal.reasoning,
                metadata=signal.metadata.copy()
            )
            
            # Apply narrative weighting to confidence
            if narrative_weight > 1.0:  # Strong narrative
                boost_factor = min(1.5, 1.0 + (narrative_weight - 1.0) * 0.5)
                weighted_signal.confidence = min(1.0, signal.confidence * boost_factor)
                weighted_signal.reasoning += f" [Narrative boost: {boost_factor:.2f}x]"
            
            weighted_signals.append(weighted_signal)
        
        return weighted_signals
    
    async def _calculate_swarm_consensus(self, signals: List[OpportunitySignal], narrative_weight: float) -> SwarmConsensus:
        """Calculate consensus from weighted signals"""
        if not signals:
            return self._create_safe_consensus("hold", 0.0, narrative_weight)
        
        try:
            # Aggregate signals by type
            signal_votes = {'buy': 0, 'sell': 0, 'hold': 0}
            weighted_confidence = {'buy': 0, 'sell': 0, 'hold': 0}
            
            for signal in signals:
                weight = self.signal_weights.get(signal.metadata.get('source', ''), 0.25)
                signal_votes[signal.signal_type] += weight
                weighted_confidence[signal.signal_type] += signal.confidence * weight
            
            # Determine consensus action
            consensus_action = max(signal_votes, key=signal_votes.get)
            
            # Calculate consensus confidence
            total_votes = sum(signal_votes.values())
            action_confidence = weighted_confidence[consensus_action] / max(signal_votes[consensus_action], 0.1)
            consensus_confidence = action_confidence * (signal_votes[consensus_action] / total_votes)
            
            # Calculate risk assessment
            risk_scores = [s.risk_level for s in signals]
            avg_risk = np.mean(risk_scores) if risk_scores else 0.5
            
            # Position sizing based on confidence and risk
            position_size = self._calculate_position_size(consensus_confidence, avg_risk, narrative_weight)
            
            return SwarmConsensus(
                recommended_action=consensus_action,
                consensus_confidence=consensus_confidence,
                narrative_weight=narrative_weight,
                risk_assessment=avg_risk,
                position_size_recommendation=position_size,
                entry_conditions={'min_confidence': 0.6, 'max_risk': 0.02},
                exit_conditions={'stop_loss': 0.05, 'take_profit': 0.15},
                reasoning=f"Swarm consensus: {len(signals)} signals, narrative weight {narrative_weight:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Consensus calculation failed: {e}")
            return self._create_safe_consensus("hold", 0.0, narrative_weight)
    
    def _calculate_position_size(self, confidence: float, risk: float, narrative_weight: float) -> float:
        """Calculate recommended position size based on confidence and risk"""
        base_size = min(self.max_risk_per_trade, confidence * 0.03)
        
        # Adjust for risk
        risk_adjusted_size = base_size * (1 - risk)
        
        # Narrative weighting adjustment
        if narrative_weight > 1.2:  # Strong narrative
            risk_adjusted_size *= min(1.3, narrative_weight)
        
        return max(0.005, min(0.05, risk_adjusted_size))  # Cap between 0.5% and 5%
    
    async def _apply_anti_fragile_filters(self, consensus: SwarmConsensus, market_data: Dict[str, Any]) -> SwarmConsensus:
        """Apply anti-fragile decision filters"""
        filtered_consensus = consensus
        
        try:
            # Market volatility adjustment
            if self.anti_fragile_factors['market_volatility_adjustment']:
                market_volatility = market_data.get('volatility_score', 0.5)
                if market_volatility > 0.8:  # High volatility
                    filtered_consensus.position_size_recommendation *= 0.5
                    filtered_consensus.reasoning += " [Volatility reduction applied]"
            
            # Position sizing discipline
            if self.anti_fragile_factors['position_sizing_discipline']:
                if filtered_consensus.position_size_recommendation > self.max_risk_per_trade:
                    filtered_consensus.position_size_recommendation = self.max_risk_per_trade
                    filtered_consensus.reasoning += " [Position size capped]"
            
            # Confidence threshold enforcement
            if filtered_consensus.consensus_confidence < self.confidence_threshold:
                filtered_consensus.recommended_action = "hold"
                filtered_consensus.reasoning += " [Confidence too low, defaulting to hold]"
            
            return filtered_consensus
            
        except Exception as e:
            self.logger.warning(f"Anti-fragile filter application failed: {e}")
            return consensus
    
    def _create_safe_consensus(self, action: str, confidence: float, narrative_weight: float) -> SwarmConsensus:
        """Create a safe consensus for error cases"""
        return SwarmConsensus(
            recommended_action=action,
            consensus_confidence=confidence,
            narrative_weight=narrative_weight,
            risk_assessment=0.5,
            position_size_recommendation=0.0,
            entry_conditions={},
            exit_conditions={},
            reasoning="Safe consensus due to insufficient data or error"
        )
    
    def get_narrative_weight_for_token(self, token_address: str, narrative_data: Dict[str, Any]) -> float:
        """Calculate narrative weight for a specific token"""
        try:
            # Extract narrative metrics
            narrative_strength = narrative_data.get('narrative_strength', 0.5)
            cultural_relevance = narrative_data.get('cultural_relevance', 0.5)
            momentum_score = narrative_data.get('momentum_score', 0.5)
            
            # Calculate composite narrative weight
            narrative_weight = (
                narrative_strength * 0.4 +
                cultural_relevance * 0.3 +
                momentum_score * 0.3
            )
            
            # Scale to 0.5 - 2.0 range
            scaled_weight = 0.5 + (narrative_weight * 1.5)
            
            return max(0.5, min(2.0, scaled_weight))
            
        except Exception as e:
            self.logger.warning(f"Narrative weight calculation failed: {e}")
            return 1.0
    
    def update_signal_weights(self, new_weights: Dict[str, float]):
        """Update signal source weights"""
        self.signal_weights.update(new_weights)
        self.logger.info(f"📊 Updated signal weights: {self.signal_weights}")
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold for decisions"""
        self.confidence_threshold = max(0.1, min(0.9, new_threshold))
        self.logger.info(f"🎯 Updated confidence threshold: {self.confidence_threshold}") 