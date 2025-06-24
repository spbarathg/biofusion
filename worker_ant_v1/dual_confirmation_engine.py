"""
Dual Confirmation Entry System
=============================

Ensures trades only execute when BOTH on-chain signals AND internal AI models agree.
Implements sophisticated signal coordination and validation.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from worker_ant_v1.scanner import TokenOpportunity
from worker_ant_v1.sentiment_analyzer import SentimentAnalyzer
from worker_ant_v1.technical_analyzer import TechnicalAnalyzer
from worker_ant_v1.ml_predictor import MLPredictor


class SignalType(Enum):
    ON_CHAIN = "on_chain"
    AI_MODEL = "ai_model"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    ML_PREDICTION = "ml_prediction"


@dataclass
class SignalConfirmation:
    """Individual signal confirmation data"""
    signal_type: SignalType
    strength: float  # 0-1
    confidence: float  # 0-1
    timestamp: datetime
    details: Dict
    source: str


@dataclass
class DualConfirmationResult:
    """Result of dual confirmation analysis"""
    token_address: str
    confirmed: bool
    confidence_score: float
    on_chain_signals: List[SignalConfirmation]
    ai_signals: List[SignalConfirmation]
    combined_strength: float
    risk_factors: List[str]
    recommendation: str  # "ENTER", "WAIT", "REJECT"


class DualConfirmationEngine:
    """Coordinates between on-chain and AI signals for trade confirmation"""
    
    def __init__(self):
        self.logger = logging.getLogger("DualConfirmation")
        
        # Initialize analyzers
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.ml_predictor = MLPredictor()
        
        # Confirmation requirements
        self.min_on_chain_signals = 2
        self.min_ai_signals = 2
        self.min_combined_confidence = 0.7
        self.min_individual_strength = 0.6
        
        # Signal weights (learned over time)
        self.signal_weights = {
            SignalType.ON_CHAIN: 0.4,
            SignalType.SENTIMENT: 0.2,
            SignalType.TECHNICAL: 0.2,
            SignalType.ML_PREDICTION: 0.2
        }
        
        # Learning system
        self.signal_performance_history = {}
        
    async def evaluate_dual_confirmation(self, opportunity: TokenOpportunity) -> DualConfirmationResult:
        """Evaluate both on-chain and AI signals for confirmation"""
        
        start_time = time.time()
        
        # Gather all signals in parallel
        on_chain_signals_task = self._gather_on_chain_signals(opportunity)
        ai_signals_task = self._gather_ai_signals(opportunity)
        
        on_chain_signals, ai_signals = await asyncio.gather(
            on_chain_signals_task, ai_signals_task
        )
        
        # Analyze signal quality
        on_chain_strength = self._calculate_signal_strength(on_chain_signals)
        ai_strength = self._calculate_signal_strength(ai_signals)
        
        # Check confirmation requirements
        confirmed = self._check_confirmation_requirements(
            on_chain_signals, ai_signals, on_chain_strength, ai_strength
        )
        
        # Calculate combined confidence
        combined_confidence = self._calculate_combined_confidence(
            on_chain_signals + ai_signals
        )
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(opportunity, on_chain_signals, ai_signals)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            confirmed, combined_confidence, risk_factors
        )
        
        result = DualConfirmationResult(
            token_address=opportunity.token_address,
            confirmed=confirmed,
            confidence_score=combined_confidence,
            on_chain_signals=on_chain_signals,
            ai_signals=ai_signals,
            combined_strength=(on_chain_strength + ai_strength) / 2,
            risk_factors=risk_factors,
            recommendation=recommendation
        )
        
        # Log analysis
        analysis_time = (time.time() - start_time) * 1000
        self.logger.info(
            f"Dual confirmation for {opportunity.token_symbol}: "
            f"{recommendation} (confidence: {combined_confidence:.2f}, "
            f"analysis_time: {analysis_time:.0f}ms)"
        )
        
        return result
    
    async def _gather_on_chain_signals(self, opportunity: TokenOpportunity) -> List[SignalConfirmation]:
        """Gather on-chain signals for confirmation"""
        
        signals = []
        
        # LP liquidity signals
        if opportunity.liquidity_usd > 50000:  # Strong liquidity
            signals.append(SignalConfirmation(
                signal_type=SignalType.ON_CHAIN,
                strength=min(opportunity.liquidity_usd / 100000, 1.0),
                confidence=0.8,
                timestamp=datetime.now(),
                details={"liquidity_usd": opportunity.liquidity_usd},
                source="liquidity_check"
            ))
        
        # Volume surge detection
        if opportunity.volume_1h_usd > opportunity.volume_24h_usd * 0.2:  # 20% of daily in 1h
            volume_strength = min(opportunity.volume_1h_usd / (opportunity.volume_24h_usd * 0.2), 1.0)
            signals.append(SignalConfirmation(
                signal_type=SignalType.ON_CHAIN,
                strength=volume_strength,
                confidence=0.7,
                timestamp=datetime.now(),
                details={"volume_surge": True, "volume_1h": opportunity.volume_1h_usd},
                source="volume_analysis"
            ))
        
        # Price momentum
        if opportunity.price_change_5m > 5.0:  # 5% price increase in 5 minutes
            momentum_strength = min(opportunity.price_change_5m / 20.0, 1.0)  # Cap at 20%
            signals.append(SignalConfirmation(
                signal_type=SignalType.ON_CHAIN,
                strength=momentum_strength,
                confidence=0.6,
                timestamp=datetime.now(),
                details={"price_change_5m": opportunity.price_change_5m},
                source="price_momentum"
            ))
        
        # Holder concentration check (anti-rug signal)
        if opportunity.top_holder_percent < 15.0 and opportunity.holder_count > 100:
            signals.append(SignalConfirmation(
                signal_type=SignalType.ON_CHAIN,
                strength=0.8,
                confidence=0.9,
                timestamp=datetime.now(),
                details={
                    "top_holder_percent": opportunity.top_holder_percent,
                    "holder_count": opportunity.holder_count
                },
                source="holder_analysis"
            ))
        
        return signals
    
    async def _gather_ai_signals(self, opportunity: TokenOpportunity) -> List[SignalConfirmation]:
        """Gather AI model signals for confirmation"""
        
        signals = []
        
        try:
            # Sentiment analysis
            sentiment_data = await self.sentiment_analyzer.analyze_token_sentiment(
                opportunity.token_symbol
            )
            
            if sentiment_data.overall_sentiment > 0.3:  # Positive sentiment
                signals.append(SignalConfirmation(
                    signal_type=SignalType.SENTIMENT,
                    strength=abs(sentiment_data.overall_sentiment),
                    confidence=sentiment_data.confidence,
                    timestamp=datetime.now(),
                    details={"sentiment_score": sentiment_data.overall_sentiment},
                    source="sentiment_analyzer"
                ))
            
            # Technical analysis (mock price data for now)
            price_data = []  # Would get real price data in production
            if price_data:
                technical_signals = await self.technical_analyzer.analyze_token_technicals(
                    opportunity.token_symbol, price_data
                )
                
                if technical_signals.overall_signal in ["buy", "strong_buy"]:
                    signals.append(SignalConfirmation(
                        signal_type=SignalType.TECHNICAL,
                        strength=technical_signals.confidence,
                        confidence=technical_signals.confidence,
                        timestamp=datetime.now(),
                        details={"technical_signal": technical_signals.overall_signal},
                        source="technical_analyzer"
                    ))
            
            # ML prediction
            ml_prediction = await self.ml_predictor.predict_token_price(
                opportunity.token_symbol,
                opportunity.price_usd,
                sentiment_data.__dict__,
                {}  # technical_data placeholder
            )
            
            if ml_prediction.ml_signal > 0.3:  # Positive ML signal
                signals.append(SignalConfirmation(
                    signal_type=SignalType.ML_PREDICTION,
                    strength=abs(ml_prediction.ml_signal),
                    confidence=ml_prediction.confidence,
                    timestamp=datetime.now(),
                    details={"ml_signal": ml_prediction.ml_signal},
                    source="ml_predictor"
                ))
                
        except Exception as e:
            self.logger.error(f"Error gathering AI signals: {e}")
        
        return signals
    
    def _calculate_signal_strength(self, signals: List[SignalConfirmation]) -> float:
        """Calculate overall strength of a signal group"""
        
        if not signals:
            return 0.0
        
        # Weighted average of signal strengths
        total_weight = 0.0
        weighted_strength = 0.0
        
        for signal in signals:
            weight = self.signal_weights.get(signal.signal_type, 0.25) * signal.confidence
            weighted_strength += signal.strength * weight
            total_weight += weight
        
        return weighted_strength / total_weight if total_weight > 0 else 0.0
    
    def _check_confirmation_requirements(self, on_chain_signals: List[SignalConfirmation], 
                                       ai_signals: List[SignalConfirmation],
                                       on_chain_strength: float, ai_strength: float) -> bool:
        """Check if dual confirmation requirements are met"""
        
        # Must have minimum number of signals
        if len(on_chain_signals) < self.min_on_chain_signals:
            return False
        
        if len(ai_signals) < self.min_ai_signals:
            return False
        
        # Both signal types must meet minimum strength
        if on_chain_strength < self.min_individual_strength:
            return False
        
        if ai_strength < self.min_individual_strength:
            return False
        
        return True
    
    def _calculate_combined_confidence(self, all_signals: List[SignalConfirmation]) -> float:
        """Calculate combined confidence from all signals"""
        
        if not all_signals:
            return 0.0
        
        # Weighted confidence calculation
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for signal in all_signals:
            weight = self.signal_weights.get(signal.signal_type, 0.25)
            weighted_confidence += signal.confidence * signal.strength * weight
            total_weight += weight
        
        return min(weighted_confidence / total_weight if total_weight > 0 else 0.0, 1.0)
    
    def _identify_risk_factors(self, opportunity: TokenOpportunity,
                             on_chain_signals: List[SignalConfirmation],
                             ai_signals: List[SignalConfirmation]) -> List[str]:
        """Identify potential risk factors"""
        
        risk_factors = []
        
        # Low liquidity risk
        if opportunity.liquidity_usd < 25000:
            risk_factors.append("Low liquidity")
        
        # High concentration risk
        if opportunity.top_holder_percent > 20.0:
            risk_factors.append("High whale concentration")
        
        # New token risk
        if opportunity.pool_age_seconds < 300:  # Less than 5 minutes
            risk_factors.append("Very new token")
        
        # Conflicting signals
        on_chain_count = len(on_chain_signals)
        ai_count = len(ai_signals)
        if abs(on_chain_count - ai_count) > 2:
            risk_factors.append("Signal imbalance")
        
        # High volatility risk
        if opportunity.volatility_score > 0.8:
            risk_factors.append("Extreme volatility")
        
        return risk_factors
    
    def _generate_recommendation(self, confirmed: bool, confidence: float, 
                               risk_factors: List[str]) -> str:
        """Generate final trading recommendation"""
        
        if not confirmed:
            return "REJECT"
        
        if confidence < self.min_combined_confidence:
            return "WAIT"
        
        # High risk conditions
        if len(risk_factors) > 2:
            return "WAIT"
        
        if confidence > 0.85 and len(risk_factors) <= 1:
            return "ENTER"
        
        if confidence > 0.75 and len(risk_factors) == 0:
            return "ENTER"
        
        return "WAIT"
    
    def update_signal_performance(self, signal_confirmations: List[SignalConfirmation], 
                                trade_result: Dict):
        """Update signal performance for learning"""
        
        trade_success = trade_result.get("success", False)
        profit_percent = trade_result.get("profit_percent", 0.0)
        
        for signal in signal_confirmations:
            key = f"{signal.signal_type.value}_{signal.source}"
            
            if key not in self.signal_performance_history:
                self.signal_performance_history[key] = {
                    "total_trades": 0,
                    "successful_trades": 0,
                    "total_profit": 0.0,
                    "avg_strength": 0.0
                }
            
            perf = self.signal_performance_history[key]
            perf["total_trades"] += 1
            
            if trade_success:
                perf["successful_trades"] += 1
            
            perf["total_profit"] += profit_percent
            perf["avg_strength"] = (
                perf["avg_strength"] * (perf["total_trades"] - 1) + signal.strength
            ) / perf["total_trades"]
            
            # Adjust signal weight based on performance
            win_rate = perf["successful_trades"] / perf["total_trades"]
            if perf["total_trades"] >= 10:  # Minimum trades for adjustment
                if win_rate > 0.7:
                    self.signal_weights[signal.signal_type] *= 1.05  # Increase weight
                elif win_rate < 0.4:
                    self.signal_weights[signal.signal_type] *= 0.95  # Decrease weight
                
                # Normalize weights
                total_weight = sum(self.signal_weights.values())
                for signal_type in self.signal_weights:
                    self.signal_weights[signal_type] /= total_weight
        
        self.logger.info(f"Updated signal weights: {self.signal_weights}")


# Global instance
dual_confirmation_engine = DualConfirmationEngine() 