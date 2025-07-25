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
import json # Added for Naive Bayes analysis

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
    
    def __init__(self, kill_switch=None):
        self.logger = setup_logger("SwarmDecisionEngine")
        
        # Safety systems
        self.kill_switch = kill_switch
        
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
    ) -> float:
        """
        Win-Rate Engine: Naive Bayes Probability Calculator
        
        Calculates the precise probability of trade success using historical signal patterns.
        Uses Naive Bayes formula: P(Win | Signals) ∝ P(Win) * Π P(Signal_i | Win)
        
        Args:
            token_address: Token contract address
            market_data: Current market data for the token
            narrative_weight: Narrative strength multiplier (legacy parameter)
            
        Returns:
            float: Win probability (0.0 to 1.0)
        """
        # CRITICAL SAFETY CHECK: Kill switch verification
        if self.kill_switch and self.kill_switch.is_triggered:
            self.logger.critical("Kill switch is active. Aborting call to analyze_opportunity.")
            return 0.0
            
        try:
            self.logger.debug(f"🧠 Naive Bayes analysis for {token_address}")
            
            # Load cached signal probabilities
            signal_probabilities = await self._load_signal_probabilities()
            if not signal_probabilities:
                self.logger.warning("⚠️ No signal probabilities available, using default")
                return 0.5  # Default 50% probability
            
            # Gather current signals from all AI ants
            current_signals = await self._gather_current_signals(token_address, market_data)
            if not current_signals:
                self.logger.warning("⚠️ No current signals available")
                return 0.5
                
            # Get base probabilities
            base_probs = signal_probabilities.get('base_probabilities', {'p_win': 0.5, 'p_loss': 0.5})
            p_win_base = base_probs['p_win']
            p_loss_base = base_probs['p_loss']
            
            # Calculate Naive Bayes likelihoods
            likelihood_win = p_win_base
            likelihood_loss = p_loss_base
            
            signal_conditionals = signal_probabilities.get('signal_conditionals', {})
            
            # Apply Naive Bayes for each active signal
            active_signals_count = 0
            for signal_name, signal_value in current_signals.items():
                if signal_name in signal_conditionals and self._signal_is_positive(signal_value):
                    signal_data = signal_conditionals[signal_name]
                    
                    # Get conditional probabilities
                    p_signal_given_win = signal_data.get('p_signal_given_win', 0.5)
                    p_signal_given_loss = signal_data.get('p_signal_given_loss', 0.5)
                    confidence = signal_data.get('confidence', 0.5)
                    
                    # Only use signals with reasonable confidence
                    if confidence > 0.3:
                        likelihood_win *= p_signal_given_win
                        likelihood_loss *= p_signal_given_loss
                        active_signals_count += 1
                        
                        self.logger.debug(f"📊 {signal_name}: P(S|W)={p_signal_given_win:.3f}, P(S|L)={p_signal_given_loss:.3f}")
            
            # Normalize to get final probability
            total_likelihood = likelihood_win + likelihood_loss
            if total_likelihood > 0:
                final_win_probability = likelihood_win / total_likelihood
            else:
                final_win_probability = 0.5  # Default if no valid signals
            
            # Apply smoothing to prevent extreme probabilities
            final_win_probability = max(0.05, min(0.95, final_win_probability))
            
            self.logger.info(f"✅ Naive Bayes result for {token_address}: {final_win_probability:.3f} "
                           f"({active_signals_count} signals analyzed)")
            
            return final_win_probability
            
        except Exception as e:
            self.logger.error(f"❌ Naive Bayes analysis failed for {token_address}: {e}")
            return 0.5  # Safe default
    
    async def _load_signal_probabilities(self) -> Optional[Dict[str, Any]]:
        """
        Load cached signal probabilities from Redis or JSON file
        
        Returns:
            Dict containing signal probabilities or None if unavailable
        """
        try:
            # Try Redis first (fastest)
            try:
                import redis.asyncio as redis
                redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
                cached_data = await redis_client.get("signal_probabilities")
                await redis_client.close()
                
                if cached_data:
                    return json.loads(cached_data)
            except Exception:
                pass  # Fall back to file
            
            # Fall back to JSON file
            import os
            if os.path.exists("data/signal_probabilities.json"):
                with open("data/signal_probabilities.json", "r") as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading signal probabilities: {e}")
            return None
    
    async def _gather_current_signals(self, token_address: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather current signals from all AI ants for Naive Bayes analysis
        
        Args:
            token_address: Token address being analyzed
            market_data: Current market data
            
        Returns:
            Dict of current signal values
        """
        try:
            signals = {}
            
            # Extract signals from market_data (populated by various AI ants)
            signals['sentiment_score'] = market_data.get('sentiment_score', 0.5)
            signals['rug_risk_score'] = market_data.get('rug_risk_score', 0.5)
            signals['narrative_strength'] = market_data.get('narrative_strength', 0.5)
            signals['volume_momentum'] = market_data.get('volume_momentum', 0.5)
            signals['price_momentum'] = market_data.get('price_momentum', 0.5)
            signals['social_buzz'] = market_data.get('social_buzz', 0.5)
            signals['whale_activity'] = market_data.get('whale_activity', 0.5)
            signals['liquidity_health'] = market_data.get('liquidity_health', 0.5)
            
            # Technical indicators
            signals['rsi_signal'] = 1.0 if market_data.get('rsi', 50) < 30 else 0.0  # Oversold
            signals['volume_spike'] = 1.0 if market_data.get('volume_change_24h', 0) > 2.0 else 0.0
            signals['price_breakout'] = market_data.get('price_breakout_signal', 0.0)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error gathering current signals: {e}")
            return {}
    
    def _signal_is_positive(self, signal_value: Any) -> bool:
        """
        Determine if a signal value should be considered 'positive'
        Same logic as in NightlyEvolutionSystem for consistency
        """
        if isinstance(signal_value, (int, float)):
            return signal_value > 0.5
        elif isinstance(signal_value, bool):
            return signal_value
        elif isinstance(signal_value, str):
            return signal_value.lower() in ['true', 'positive', 'bullish', 'buy']
        else:
            return False
    
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
            # CRITICAL: TRUE CONSENSUS LOGIC - ALL SOURCES MUST AGREE FOR BUY
            # Categorize signals by source
            technical_signals = [s for s in signals if s.metadata.get('source') == 'technical_analysis']
            sentiment_signals = [s for s in signals if s.metadata.get('source') == 'sentiment_analysis']
            narrative_signals = [s for s in signals if s.metadata.get('source') == 'narrative_analysis']
            security_signals = [s for s in signals if s.metadata.get('source') == 'security_analysis']
            
            # Check for unanimous BUY consensus across ALL critical sources
            consensus_action = "hold"  # Default to safe hold
            consensus_confidence = 0.0
            
            # Require ALL sources to have BUY signals for a BUY recommendation
            technical_buy = any(s.signal_type == 'buy' and s.confidence >= self.confidence_threshold for s in technical_signals)
            sentiment_buy = any(s.signal_type == 'buy' and s.confidence >= self.confidence_threshold for s in sentiment_signals)
            narrative_buy = any(s.signal_type == 'buy' and s.confidence >= self.confidence_threshold for s in narrative_signals)
            security_buy = any(s.signal_type == 'buy' and s.confidence >= self.confidence_threshold for s in security_signals)
            
            # TRUE AND CONDITION: ALL must be true for BUY
            if technical_buy and sentiment_buy and narrative_buy and security_buy:
                consensus_action = "buy"
                # Calculate minimum confidence across sources (weakest link determines overall confidence)
                all_confidences = [s.confidence for s in signals if s.signal_type == 'buy']
                consensus_confidence = min(all_confidences) if all_confidences else 0.0
            
            # Check for ANY sell signal (any source can trigger sell)
            elif any(s.signal_type == 'sell' and s.confidence >= self.confidence_threshold for s in signals):
                consensus_action = "sell"
                sell_confidences = [s.confidence for s in signals if s.signal_type == 'sell']
                consensus_confidence = max(sell_confidences) if sell_confidences else 0.0
            
            # Default: hold with average confidence
            else:
                consensus_action = "hold"
                all_confidences = [s.confidence for s in signals]
                consensus_confidence = np.mean(all_confidences) if all_confidences else 0.0
            
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