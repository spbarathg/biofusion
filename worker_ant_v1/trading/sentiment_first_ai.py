"""
SENTIMENT FIRST AI - PRIORITY SENTIMENT DECISION ENGINE
=====================================================

AI system that prioritizes sentiment analysis over all other factors.
Uses sentiment as the primary filter before any other analysis.
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
from worker_ant_v1.intelligence.sentiment_analyzer import SentimentAnalyzer, SentimentData
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

class SentimentFirstAI:
    """AI system that prioritizes sentiment above all other factors"""
    
    def __init__(self):
        self.logger = setup_logger("SentimentFirstAI")
        self.sentiment_analyzer = SentimentAnalyzer()
        
        
        self.decision_thresholds = {
            'strong_buy': 0.7,
            'buy': 0.4,
            'neutral_high': 0.2,
            'neutral_low': -0.2,
            'sell': -0.4,
            'strong_sell': -0.7
        }
        
        
        self.sentiment_weights = {
            'immediate': 0.4,     # Current sentiment
            'trend': 0.3,         # Sentiment trend over time
            'stability': 0.2,     # Sentiment stability
            'strength': 0.1       # Sentiment strength/confidence
        }
        
        
        self.decision_history: Dict[str, List[SentimentDecision]] = {}
        
        
        self.sentiment_blacklist: Dict[str, datetime] = {}
        
        self.logger.info("✅ Sentiment First AI initialized")
    
    async def analyze_and_decide(self, token_address: str, 
                               market_data: Dict[str, Any],
                               additional_signals: Dict[str, float] = None) -> SentimentDecision:
        """Make a trading decision based primarily on sentiment analysis"""
        
        try:
            if await self._is_sentiment_blacklisted(token_address):
                return SentimentDecision(
                    token_address=token_address,
                    decision="AVOID",
                    sentiment_score=-1.0,
                    confidence=1.0,
                    reasoning=["Token on sentiment blacklist"],
                    priority=5,
                    timestamp=datetime.now()
                )
            
            
            current_sentiment = await self.sentiment_analyzer.analyze_token_sentiment(
                token_address, market_data
            )
            
            
            sentiment_trend = await self._analyze_sentiment_trend(token_address)
            
            
            composite_score = await self._calculate_composite_sentiment(
                current_sentiment, sentiment_trend
            )
            
            
            decision, confidence, reasoning, priority = self._make_sentiment_decision(
                composite_score, current_sentiment, sentiment_trend, additional_signals
            )
            
            sentiment_decision = SentimentDecision(
                token_address=token_address,
                decision=decision,
                sentiment_score=composite_score,
                confidence=confidence,
                reasoning=reasoning,
                priority=priority,
                timestamp=datetime.now()
            )
            
            
            if token_address not in self.decision_history:
                self.decision_history[token_address] = []
            
            self.decision_history[token_address].append(sentiment_decision)
            
            
            if len(self.decision_history[token_address]) > 50:
                self.decision_history[token_address] = self.decision_history[token_address][-50:]
            
            
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
                timestamp=datetime.now()
            )
    
    async def _analyze_sentiment_trend(self, token_address: str) -> Dict[str, float]:
        """Analyze sentiment trend over time"""
        
        try:
            if token_address not in self.sentiment_analyzer.sentiment_history:
                return {
                    'trend_direction': 0.0,
                    'trend_strength': 0.0,
                    'stability': 0.5
                }
            
            sentiments = self.sentiment_analyzer.sentiment_history[token_address]
            
            if len(sentiments) < 3:
                return {
                    'trend_direction': 0.0,
                    'trend_strength': 0.0,
                    'stability': 0.5
                }
            
            
            recent_sentiment = np.mean([s.overall_sentiment for s in sentiments[-5:]])
            older_sentiment = np.mean([s.overall_sentiment for s in sentiments[:-5]]) if len(sentiments) > 5 else recent_sentiment
            
            trend_direction = recent_sentiment - older_sentiment
            
            
            sentiment_values = [s.overall_sentiment for s in sentiments]
            trend_strength = abs(np.corrcoef(range(len(sentiment_values)), sentiment_values)[0, 1]) if len(sentiment_values) > 1 else 0
            
            
            sentiment_std = np.std(sentiment_values)
            stability = max(0.0, 1.0 - sentiment_std)
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'stability': stability
            }
            
        except Exception as e:
            self.logger.warning(f"Sentiment trend analysis failed: {e}")
            return {
                'trend_direction': 0.0,
                'trend_strength': 0.0,
                'stability': 0.5
            }
    
    async def _calculate_composite_sentiment(self, current_sentiment: SentimentData,
                                           sentiment_trend: Dict[str, float]) -> float:
        """Calculate composite sentiment score from multiple factors"""
        
        try:
            immediate_score = current_sentiment.overall_sentiment * self.sentiment_weights['immediate']
            
            
            trend_score = sentiment_trend['trend_direction'] * self.sentiment_weights['trend']
            
            
            stability_score = (sentiment_trend['stability'] - 0.5) * 2 * self.sentiment_weights['stability']
            
            
            strength_score = (current_sentiment.confidence - 0.5) * 2 * self.sentiment_weights['strength']
            
            
            composite_score = immediate_score + trend_score + stability_score + strength_score
            
            
            return np.clip(composite_score, -1.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Composite sentiment calculation failed: {e}")
            return current_sentiment.overall_sentiment
    
    def _make_sentiment_decision(self, composite_score: float, 
                               current_sentiment: SentimentData,
                               sentiment_trend: Dict[str, float],
                               additional_signals: Dict[str, float] = None) -> Tuple[str, float, List[str], int]:
        """Make trading decision based on sentiment analysis"""
        
        reasoning = []
        
        
        if composite_score >= self.decision_thresholds['strong_buy']:
            decision = "BUY"
            confidence = 0.9
            priority = 5
            reasoning.append(f"Strong bullish sentiment: {composite_score:.3f}")
            
        elif composite_score >= self.decision_thresholds['buy']:
            decision = "BUY"
            confidence = 0.7
            priority = 4
            reasoning.append(f"Bullish sentiment: {composite_score:.3f}")
            
        elif composite_score <= self.decision_thresholds['strong_sell']:
            decision = "SELL"
            confidence = 0.9
            priority = 5
            reasoning.append(f"Strong bearish sentiment: {composite_score:.3f}")
            
        elif composite_score <= self.decision_thresholds['sell']:
            decision = "SELL"
            confidence = 0.7
            priority = 4
            reasoning.append(f"Bearish sentiment: {composite_score:.3f}")
            
        else:
            decision = "HOLD"
            confidence = 0.5
            priority = 2
            reasoning.append(f"Neutral sentiment: {composite_score:.3f}")
        
        
        if sentiment_trend['trend_direction'] > 0.2:
            reasoning.append("Positive sentiment trend")
            if decision == "HOLD":
                decision = "BUY"
                confidence += 0.1
                priority += 1
        elif sentiment_trend['trend_direction'] < -0.2:
            reasoning.append("Negative sentiment trend")
            if decision == "HOLD":
                decision = "SELL"
                confidence += 0.1
                priority += 1
        
        
        if sentiment_trend['stability'] < 0.3:
            reasoning.append("Unstable sentiment - reducing confidence")
            confidence = max(0.1, confidence - 0.2)
            priority = max(1, priority - 1)
        elif sentiment_trend['stability'] > 0.8:
            reasoning.append("Stable sentiment - increasing confidence")
            confidence = min(1.0, confidence + 0.1)
        
        
        if current_sentiment.confidence < 0.3:
            reasoning.append("Low sentiment data quality")
            confidence = max(0.1, confidence - 0.3)
            priority = max(1, priority - 2)
        
        
        if additional_signals:
            signal_influence = 0.1  # 10% weight for non-sentiment signals
            
            
            if additional_signals.get('technical_score', 0) > 0.5:
                reasoning.append("Positive technical signals (minor factor)")
                confidence = min(1.0, confidence + signal_influence)
            elif additional_signals.get('technical_score', 0) < -0.5:
                reasoning.append("Negative technical signals (minor factor)")
                confidence = max(0.1, confidence - signal_influence)
            
            
            if additional_signals.get('volume_score', 0) > 0.5:
                reasoning.append("Strong volume pattern (minor factor)")
                confidence = min(1.0, confidence + signal_influence)
        
        
        if composite_score < 0 and decision == "BUY":
            decision = "HOLD"
            reasoning.append("Override: Negative sentiment blocks buy signal")
            confidence = max(0.1, confidence - 0.2)
        
        
        if composite_score > 0.3 and decision == "SELL":
            if not additional_signals or max(additional_signals.values()) < 0.8:
                decision = "HOLD"
                reasoning.append("Override: Positive sentiment blocks sell signal")
                confidence = max(0.1, confidence - 0.2)
        
        return decision, confidence, reasoning, priority
    
    async def _is_sentiment_blacklisted(self, token_address: str) -> bool:
        """Check if token is on sentiment blacklist"""
        
        if token_address not in self.sentiment_blacklist:
            return False
        
        
        blacklist_time = self.sentiment_blacklist[token_address]
        if datetime.now() - blacklist_time > timedelta(hours=24):
            del self.sentiment_blacklist[token_address]
            return False
        
        return True
    
    async def _update_sentiment_blacklist(self, token_address: str, decision: SentimentDecision):
        """Update sentiment blacklist based on consistently negative sentiment"""
        
        if token_address not in self.decision_history:
            return
        
        recent_decisions = self.decision_history[token_address][-10:]  # Last 10 decisions
        
        if len(recent_decisions) < 5:
            return
        
        
        negative_count = sum(1 for d in recent_decisions if d.sentiment_score < -0.3)
        
        if negative_count >= 4:  # 4 out of last 5 decisions negative
            self.sentiment_blacklist[token_address] = datetime.now()
            self.logger.warning(f"🔴 Token {token_address} blacklisted due to consistent negative sentiment")
    
    async def get_sentiment_priority_tokens(self, tokens: List[str], 
                                          market_data: Dict[str, Dict]) -> List[Tuple[str, float, str]]:
        """Get tokens ranked by sentiment priority"""
        
        token_priorities = []
        
        for token in tokens:
            try:
                token_market_data = market_data.get(token, {})
                decision = await self.analyze_and_decide(token, token_market_data)
                
                
                priority_score = decision.sentiment_score * decision.confidence * decision.priority
                
                token_priorities.append((token, priority_score, decision.decision))
                
            except Exception as e:
                self.logger.error(f"Failed to analyze token {token}: {e}")
                continue
        
        
        token_priorities.sort(key=lambda x: x[1], reverse=True)
        
        return token_priorities
    
    async def validate_buy_signal(self, token_address: str, market_data: Dict[str, Any]) -> bool:
        """Validate if a buy signal should proceed based on sentiment"""
        
        decision = await self.analyze_and_decide(token_address, market_data)
        
        
        return (decision.decision == "BUY" and 
                decision.sentiment_score > 0.2 and 
                decision.confidence > 0.4)
    
    async def validate_sell_signal(self, token_address: str, market_data: Dict[str, Any]) -> bool:
        """Validate if a sell signal should proceed based on sentiment"""
        
        decision = await self.analyze_and_decide(token_address, market_data)
        
        
        return (decision.decision == "SELL" or 
                decision.sentiment_score < -0.3 or 
                decision.confidence > 0.8)
    
    def get_sentiment_summary(self, token_address: str) -> Dict[str, Any]:
        """Get sentiment analysis summary for a token"""
        
        if token_address not in self.decision_history:
            return {
                'status': 'no_data',
                'recent_decisions': [],
                'avg_sentiment': 0.0,
                'trend': 'unknown'
            }
        
        recent_decisions = self.decision_history[token_address][-5:]
        
        avg_sentiment = np.mean([d.sentiment_score for d in recent_decisions])
        
        
        if len(recent_decisions) >= 2:
            recent_sentiment = np.mean([d.sentiment_score for d in recent_decisions[-2:]])
            older_sentiment = np.mean([d.sentiment_score for d in recent_decisions[:-2]]) if len(recent_decisions) > 2 else recent_sentiment
            
            if recent_sentiment > older_sentiment + 0.1:
                trend = 'improving'
            elif recent_sentiment < older_sentiment - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'status': 'active',
            'recent_decisions': [
                {
                    'decision': d.decision,
                    'sentiment_score': d.sentiment_score,
                    'confidence': d.confidence,
                    'timestamp': d.timestamp.isoformat()
                } for d in recent_decisions
            ],
            'avg_sentiment': avg_sentiment,
            'trend': trend,
            'blacklisted': token_address in self.sentiment_blacklist
        } 