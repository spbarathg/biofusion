"""
SENTIMENT FIRST AI - MEMECOIN SENTIMENT MASTER
============================================

AI system that prioritizes sentiment analysis above all other factors
for aggressive memecoin trading and rapid compounding.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import aiohttp
import json
import os

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.intelligence.sentiment_analyzer import SentimentAnalyzer
from enum import Enum

@dataclass
class SentimentDecision:
    """Sentiment-based trading decision"""
    token_address: str
    decision: str  # "BUY", "SELL", "HOLD", "AVOID"
    sentiment_score: float
    confidence: float
    reasoning: List[str]
    priority: int  # 1-5, 5 being highest priority
    timestamp: datetime
    expected_profit: float = 0.0
    risk_level: str = "medium"

class SentimentFirstAI:
    """AI system that prioritizes sentiment above all other factors for memecoin trading"""
    
    def __init__(self):
        self.logger = get_logger("SentimentFirstAI")
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Aggressive memecoin trading thresholds
        self.decision_thresholds = {
            'strong_buy': 0.6,      # Lower threshold for aggressive buying
            'buy': 0.3,             # Moderate sentiment is enough to buy
            'neutral_high': 0.1,    # Slight positive sentiment
            'neutral_low': -0.1,    # Slight negative sentiment
            'sell': -0.3,           # Moderate negative sentiment
            'strong_sell': -0.6     # Strong negative sentiment
        }
        
        # Sentiment weights for memecoin trading
        self.sentiment_weights = {
            'immediate': 0.5,     # Current sentiment (highest weight)
            'trend': 0.3,         # Sentiment trend over time
            'stability': 0.1,     # Sentiment stability
            'strength': 0.1       # Sentiment strength/confidence
        }
        
        # Decision history for pattern recognition
        self.decision_history: Dict[str, List[SentimentDecision]] = {}
        
        # Sentiment blacklist for tokens with consistently bad sentiment
        self.sentiment_blacklist: Dict[str, datetime] = {}
        
        # Memecoin-specific patterns
        self.memecoin_patterns = {
            'hype_cycle': 'rapid_sentiment_spike',
            'fomo_pattern': 'accelerating_positive_sentiment',
            'dump_pattern': 'sentiment_reversal',
            'accumulation': 'steady_positive_sentiment'
        }
        
        # Aggressive compounding settings
        self.compounding_settings = {
            'min_sentiment_for_buy': 0.3,
            'max_position_size_multiplier': 3.0,  # Can use up to 3x normal position size
            'profit_taking_sentiment_threshold': -0.2,  # Sell if sentiment drops below -0.2
            'aggressive_reentry_threshold': 0.4,  # Re-enter if sentiment recovers to 0.4
            'max_hold_time_hours': 4,  # Maximum hold time for memecoins
            'min_profit_target': 0.05  # 5% minimum profit target
        }
        
        self.logger.info("✅ Sentiment First AI initialized for aggressive memecoin trading")
    
    async def analyze_and_decide(self, token_address: str, 
                               market_data: Dict[str, Any],
                               additional_signals: Dict[str, float] = None) -> SentimentDecision:
        """Make a trading decision based primarily on sentiment analysis for memecoin trading"""
        
        try:
            # Check if token is blacklisted
            if await self._is_sentiment_blacklisted(token_address):
                return SentimentDecision(
                    token_address=token_address,
                    decision="AVOID",
                    sentiment_score=-1.0,
                    confidence=1.0,
                    reasoning=["Token on sentiment blacklist"],
                    priority=5,
                    timestamp=datetime.now(),
                    expected_profit=0.0,
                    risk_level="high"
                )
            
            # Get current sentiment
            current_sentiment = await self.sentiment_analyzer.analyze_token_sentiment(
                token_address, market_data
            )
            
            # Analyze sentiment trend
            sentiment_trend = await self._analyze_sentiment_trend(token_address)
            
            # Calculate composite sentiment score
            composite_score = await self._calculate_composite_sentiment(
                current_sentiment, sentiment_trend
            )
            
            # Detect memecoin patterns
            memecoin_pattern = await self._detect_memecoin_pattern(token_address, composite_score, sentiment_trend)
            
            # Make aggressive trading decision
            decision, confidence, reasoning, priority, expected_profit, risk_level = self._make_aggressive_decision(
                composite_score, current_sentiment, sentiment_trend, memecoin_pattern, additional_signals
            )
            
            sentiment_decision = SentimentDecision(
                token_address=token_address,
                decision=decision,
                sentiment_score=composite_score,
                confidence=confidence,
                reasoning=reasoning,
                priority=priority,
                timestamp=datetime.now(),
                expected_profit=expected_profit,
                risk_level=risk_level
            )
            
            # Store decision history
            if token_address not in self.decision_history:
                self.decision_history[token_address] = []
            
            self.decision_history[token_address].append(sentiment_decision)
            
            # Keep only last 50 decisions
            if len(self.decision_history[token_address]) > 50:
                self.decision_history[token_address] = self.decision_history[token_address][-50:]
            
            # Update sentiment blacklist
            await self._update_sentiment_blacklist(token_address, sentiment_decision)
            
            return sentiment_decision
            
        except Exception as e:
            self.logger.error(f"Sentiment decision analysis failed for {token_address}: {e}")
            
            return SentimentDecision(
                token_address=token_address,
                decision="HOLD",
                sentiment_score=0.0,
                confidence=0.0,
                reasoning=[f"Analysis failed: {str(e)}"],
                priority=1,
                timestamp=datetime.now(),
                expected_profit=0.0,
                risk_level="high"
            )
    
    async def _analyze_sentiment_trend(self, token_address: str) -> Dict[str, float]:
        """Analyze sentiment trend over time for memecoin patterns"""
        
        try:
            if token_address not in self.sentiment_analyzer.sentiment_history:
                return {
                    'trend_direction': 0.0,
                    'trend_strength': 0.0,
                    'stability': 0.5,
                    'acceleration': 0.0,
                    'volatility': 0.5
                }
            
            sentiments = self.sentiment_analyzer.sentiment_history[token_address]
            
            if len(sentiments) < 3:
                return {
                    'trend_direction': 0.0,
                    'trend_strength': 0.0,
                    'stability': 0.5,
                    'acceleration': 0.0,
                    'volatility': 0.5
                }
            
            # Calculate trend direction
            recent_sentiment = np.mean([s.overall_sentiment for s in sentiments[-5:]])
            older_sentiment = np.mean([s.overall_sentiment for s in sentiments[:-5]]) if len(sentiments) > 5 else recent_sentiment
            trend_direction = recent_sentiment - older_sentiment
            
            # Calculate trend strength
            sentiment_values = [s.overall_sentiment for s in sentiments]
            trend_strength = abs(np.corrcoef(range(len(sentiment_values)), sentiment_values)[0, 1]) if len(sentiment_values) > 1 else 0
            
            # Calculate stability
            sentiment_std = np.std(sentiment_values)
            stability = max(0.0, 1.0 - sentiment_std)
            
            # Calculate acceleration (rate of change)
            if len(sentiment_values) >= 3:
                recent_change = sentiment_values[-1] - sentiment_values[-2]
                previous_change = sentiment_values[-2] - sentiment_values[-3]
                acceleration = recent_change - previous_change
            else:
                acceleration = 0.0
            
            # Calculate volatility
            volatility = sentiment_std
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'stability': stability,
                'acceleration': acceleration,
                'volatility': volatility
            }
            
        except Exception as e:
            self.logger.warning(f"Sentiment trend analysis failed: {e}")
            return {
                'trend_direction': 0.0,
                'trend_strength': 0.0,
                'stability': 0.5,
                'acceleration': 0.0,
                'volatility': 0.5
            }
    
    async def _calculate_composite_sentiment(self, current_sentiment: Any,
                                           sentiment_trend: Dict[str, float]) -> float:
        """Calculate composite sentiment score from multiple factors for memecoin trading"""
        
        try:
            # Immediate sentiment (highest weight for memecoins)
            immediate_score = current_sentiment.overall_sentiment * self.sentiment_weights['immediate']
            
            # Trend score (important for momentum)
            trend_score = sentiment_trend['trend_direction'] * self.sentiment_weights['trend']
            
            # Stability score (lower weight for memecoins - volatility is expected)
            stability_score = (sentiment_trend['stability'] - 0.5) * 2 * self.sentiment_weights['stability']
            
            # Strength score
            strength_score = (current_sentiment.confidence - 0.5) * 2 * self.sentiment_weights['strength']
            
            # Acceleration bonus for memecoins (rapid sentiment changes are good)
            acceleration_bonus = sentiment_trend['acceleration'] * 0.2 if sentiment_trend['acceleration'] > 0 else 0
            
            # Composite score
            composite_score = immediate_score + trend_score + stability_score + strength_score + acceleration_bonus
            
            # Clamp to [-1, 1] range
            return max(-1.0, min(1.0, composite_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating composite sentiment: {e}")
            return 0.0
    
    async def _detect_memecoin_pattern(self, token_address: str, composite_score: float, 
                                     sentiment_trend: Dict[str, float]) -> str:
        """Detect memecoin-specific sentiment patterns"""
        
        try:
            # Hype cycle pattern
            if sentiment_trend['acceleration'] > 0.3 and composite_score > 0.5:
                return 'hype_cycle'
            
            # FOMO pattern
            if sentiment_trend['trend_direction'] > 0.2 and sentiment_trend['trend_strength'] > 0.7:
                return 'fomo_pattern'
            
            # Dump pattern
            if sentiment_trend['trend_direction'] < -0.2 and sentiment_trend['acceleration'] < -0.2:
                return 'dump_pattern'
            
            # Accumulation pattern
            if 0.1 < composite_score < 0.4 and sentiment_trend['stability'] > 0.6:
                return 'accumulation'
            
            # No clear pattern
            return 'no_pattern'
            
        except Exception as e:
            self.logger.error(f"Error detecting memecoin pattern: {e}")
            return 'no_pattern'
    
    def _make_aggressive_decision(self, composite_score: float, 
                                current_sentiment: Any,
                                sentiment_trend: Dict[str, float],
                                memecoin_pattern: str,
                                additional_signals: Dict[str, float] = None) -> Tuple[str, float, List[str], int, float, str]:
        """Make aggressive trading decision based on sentiment"""
        
        try:
            reasoning = []
            confidence = 0.0
            priority = 1
            expected_profit = 0.0
            risk_level = "medium"
            
            # Base decision on sentiment score
            if composite_score >= self.decision_thresholds['strong_buy']:
                decision = "BUY"
                confidence = 0.9
                priority = 5
                expected_profit = 0.15  # 15% expected profit
                risk_level = "medium"
                reasoning.append(f"Strong positive sentiment: {composite_score:.2f}")
                
            elif composite_score >= self.decision_thresholds['buy']:
                decision = "BUY"
                confidence = 0.7
                priority = 4
                expected_profit = 0.10  # 10% expected profit
                risk_level = "medium"
                reasoning.append(f"Positive sentiment: {composite_score:.2f}")
                
            elif composite_score >= self.decision_thresholds['neutral_high']:
                decision = "HOLD"
                confidence = 0.5
                priority = 2
                expected_profit = 0.02  # 2% expected profit
                risk_level = "low"
                reasoning.append(f"Slightly positive sentiment: {composite_score:.2f}")
                
            elif composite_score >= self.decision_thresholds['neutral_low']:
                decision = "HOLD"
                confidence = 0.3
                priority = 1
                expected_profit = 0.0
                risk_level = "low"
                reasoning.append(f"Neutral sentiment: {composite_score:.2f}")
                
            elif composite_score >= self.decision_thresholds['sell']:
                decision = "SELL"
                confidence = 0.7
                priority = 3
                expected_profit = 0.0
                risk_level = "medium"
                reasoning.append(f"Negative sentiment: {composite_score:.2f}")
                
            else:  # composite_score < self.decision_thresholds['strong_sell']
                decision = "SELL"
                confidence = 0.9
                priority = 4
                expected_profit = 0.0
                risk_level = "high"
                reasoning.append(f"Strong negative sentiment: {composite_score:.2f}")
            
            # Adjust based on memecoin pattern
            if memecoin_pattern == 'hype_cycle':
                if decision == "BUY":
                    priority = 5
                    expected_profit = 0.25  # 25% expected profit for hype cycle
                    reasoning.append("Hype cycle detected - high profit potential")
                else:
                    decision = "BUY"
                    priority = 4
                    expected_profit = 0.20
                    reasoning.append("Hype cycle detected - converting to BUY")
                    
            elif memecoin_pattern == 'fomo_pattern':
                if decision == "BUY":
                    priority = 5
                    expected_profit = 0.20  # 20% expected profit for FOMO
                    reasoning.append("FOMO pattern detected - momentum building")
                    
            elif memecoin_pattern == 'dump_pattern':
                if decision != "SELL":
                    decision = "SELL"
                    priority = 5
                    reasoning.append("Dump pattern detected - immediate sell signal")
                    
            elif memecoin_pattern == 'accumulation':
                if decision == "BUY":
                    expected_profit = 0.08  # 8% expected profit for accumulation
                    reasoning.append("Accumulation pattern - steady growth expected")
            
            # Adjust confidence based on trend strength
            if sentiment_trend['trend_strength'] > 0.8:
                confidence = min(1.0, confidence + 0.1)
                reasoning.append("Strong sentiment trend")
            elif sentiment_trend['trend_strength'] < 0.3:
                confidence = max(0.1, confidence - 0.1)
                reasoning.append("Weak sentiment trend")
            
            # Adjust for volatility (memecoins are volatile)
            if sentiment_trend['volatility'] > 0.3:
                risk_level = "high"
                reasoning.append("High sentiment volatility")
            
            # Additional signals adjustment
            if additional_signals:
                if 'volume_spike' in additional_signals and additional_signals['volume_spike'] > 2.0:
                    if decision == "BUY":
                        priority = 5
                        expected_profit += 0.05
                        reasoning.append("Volume spike detected")
                
                if 'price_momentum' in additional_signals and additional_signals['price_momentum'] > 0.1:
                    if decision == "BUY":
                        expected_profit += 0.03
                        reasoning.append("Price momentum detected")
            
            return decision, confidence, reasoning, priority, expected_profit, risk_level
            
        except Exception as e:
            self.logger.error(f"Error making aggressive decision: {e}")
            return "HOLD", 0.0, [f"Decision error: {str(e)}"], 1, 0.0, "high"
    
    async def _is_sentiment_blacklisted(self, token_address: str) -> bool:
        """Check if token is on sentiment blacklist"""
        try:
            if token_address in self.sentiment_blacklist:
                blacklist_time = self.sentiment_blacklist[token_address]
                # Remove from blacklist after 24 hours
                if datetime.now() - blacklist_time > timedelta(hours=24):
                    del self.sentiment_blacklist[token_address]
                    return False
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking sentiment blacklist: {e}")
            return False
    
    async def _update_sentiment_blacklist(self, token_address: str, decision: SentimentDecision):
        """Update sentiment blacklist based on decisions"""
        try:
            # Add to blacklist if consistently negative sentiment
            if token_address in self.decision_history:
                recent_decisions = self.decision_history[token_address][-5:]  # Last 5 decisions
                
                if len(recent_decisions) >= 3:
                    negative_count = sum(1 for d in recent_decisions if d.sentiment_score < -0.5)
                    
                    if negative_count >= 3:  # 3 out of 5 decisions were strongly negative
                        self.sentiment_blacklist[token_address] = datetime.now()
                        self.logger.warning(f"Added {token_address} to sentiment blacklist")
            
        except Exception as e:
            self.logger.error(f"Error updating sentiment blacklist: {e}")
    
    async def get_sentiment_priority_tokens(self, tokens: List[str], 
                                          market_data: Dict[str, Dict]) -> List[Tuple[str, float, str]]:
        """Get tokens prioritized by sentiment for aggressive trading"""
        
        try:
            token_priorities = []
            
            for token_address in tokens:
                if token_address in market_data:
                    decision = await self.analyze_and_decide(token_address, market_data[token_address])
                    
                    if decision.decision == "BUY":
                        token_priorities.append((
                            token_address,
                            decision.priority,
                            decision.decision
                        ))
            
            # Sort by priority (highest first)
            token_priorities.sort(key=lambda x: x[1], reverse=True)
            
            return token_priorities
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment priority tokens: {e}")
            return []
    
    async def validate_buy_signal(self, token_address: str, market_data: Dict[str, Any]) -> bool:
        """Validate if a buy signal is strong enough for memecoin trading"""
        try:
            decision = await self.analyze_and_decide(token_address, market_data)
            
            # Aggressive validation for memecoins
            return (decision.decision == "BUY" and 
                   decision.confidence >= 0.6 and 
                   decision.sentiment_score >= self.compounding_settings['min_sentiment_for_buy'])
            
        except Exception as e:
            self.logger.error(f"Error validating buy signal: {e}")
            return False
    
    async def validate_sell_signal(self, token_address: str, market_data: Dict[str, Any]) -> bool:
        """Validate if a sell signal is strong enough"""
        try:
            decision = await self.analyze_and_decide(token_address, market_data)
            
            return (decision.decision == "SELL" and 
                   decision.confidence >= 0.7 and 
                   decision.sentiment_score <= self.compounding_settings['profit_taking_sentiment_threshold'])
            
        except Exception as e:
            self.logger.error(f"Error validating sell signal: {e}")
            return False
    
    def get_sentiment_summary(self, token_address: str) -> Dict[str, Any]:
        """Get sentiment summary for a token"""
        try:
            if token_address not in self.decision_history:
                return {
                    'token_address': token_address,
                    'decision_count': 0,
                    'avg_sentiment': 0.0,
                    'trend': 'neutral',
                    'confidence': 0.0
                }
            
            decisions = self.decision_history[token_address]
            
            if not decisions:
                return {
                    'token_address': token_address,
                    'decision_count': 0,
                    'avg_sentiment': 0.0,
                    'trend': 'neutral',
                    'confidence': 0.0
                }
            
            # Calculate summary statistics
            sentiment_scores = [d.sentiment_score for d in decisions]
            avg_sentiment = np.mean(sentiment_scores)
            
            # Determine trend
            if len(decisions) >= 2:
                recent_avg = np.mean([d.sentiment_score for d in decisions[-3:]])
                older_avg = np.mean([d.sentiment_score for d in decisions[:-3]]) if len(decisions) > 3 else recent_avg
                
                if recent_avg > older_avg + 0.1:
                    trend = 'improving'
                elif recent_avg < older_avg - 0.1:
                    trend = 'declining'
                else:
                    trend = 'stable'
            else:
                trend = 'neutral'
            
            # Average confidence
            avg_confidence = np.mean([d.confidence for d in decisions])
            
            return {
                'token_address': token_address,
                'decision_count': len(decisions),
                'avg_sentiment': avg_sentiment,
                'trend': trend,
                'confidence': avg_confidence,
                'last_decision': decisions[-1].decision if decisions else 'none',
                'last_sentiment': decisions[-1].sentiment_score if decisions else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment summary: {e}")
            return {
                'token_address': token_address,
                'decision_count': 0,
                'avg_sentiment': 0.0,
                'trend': 'error',
                'confidence': 0.0
            }
    
    def get_compounding_settings(self) -> Dict[str, Any]:
        """Get current compounding settings"""
        return self.compounding_settings.copy()
    
    def update_compounding_settings(self, new_settings: Dict[str, Any]):
        """Update compounding settings"""
        try:
            for key, value in new_settings.items():
                if key in self.compounding_settings:
                    self.compounding_settings[key] = value
            
            self.logger.info(f"Updated compounding settings: {new_settings}")
            
        except Exception as e:
            self.logger.error(f"Error updating compounding settings: {e}")
    
    async def get_aggressive_trading_recommendations(self, tokens: List[str], 
                                                   market_data: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Get aggressive trading recommendations for memecoin compounding"""
        
        try:
            recommendations = []
            
            for token_address in tokens:
                if token_address in market_data:
                    decision = await self.analyze_and_decide(token_address, market_data[token_address])
                    
                    if decision.decision == "BUY" and decision.priority >= 4:
                        # Calculate position size multiplier based on sentiment
                        position_multiplier = min(
                            self.compounding_settings['max_position_size_multiplier'],
                            1.0 + (decision.sentiment_score - 0.3) * 2.0
                        )
                        
                        recommendation = {
                            'token_address': token_address,
                            'action': 'BUY',
                            'priority': decision.priority,
                            'sentiment_score': decision.sentiment_score,
                            'confidence': decision.confidence,
                            'expected_profit': decision.expected_profit,
                            'risk_level': decision.risk_level,
                            'position_multiplier': position_multiplier,
                            'reasoning': decision.reasoning,
                            'max_hold_time_hours': self.compounding_settings['max_hold_time_hours'],
                            'profit_target': decision.expected_profit,
                            'stop_loss_sentiment': self.compounding_settings['profit_taking_sentiment_threshold']
                        }
                        
                        recommendations.append(recommendation)
            
            # Sort by priority and expected profit
            recommendations.sort(key=lambda x: (x['priority'], x['expected_profit']), reverse=True)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting aggressive trading recommendations: {e}")
            return []

# Global instance
_sentiment_first_ai = None

async def get_sentiment_first_ai() -> SentimentFirstAI:
    """Get global sentiment first AI instance"""
    global _sentiment_first_ai
    if _sentiment_first_ai is None:
        _sentiment_first_ai = SentimentFirstAI()
    return _sentiment_first_ai 