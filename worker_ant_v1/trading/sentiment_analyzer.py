"""
Advanced Sentiment Analysis System
================================

Multi-dimensional sentiment analysis combining:
- Wallet behavior patterns
- Volume analysis
- Price action sentiment
- Holder distribution metrics

Provides real-time sentiment scoring for token intelligence.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from worker_ant_v1.utils.logger import setup_logger

@dataclass
class SentimentData:
    """On-chain sentiment data point"""
    token_address: str
    timestamp: datetime
    wallet_sentiment: float  # Based on wallet behavior patterns
    volume_sentiment: float  # Based on volume patterns
    price_sentiment: float   # Based on price action patterns
    holder_sentiment: float  # Based on holder distribution
    overall_sentiment: float # Weighted average
    confidence: float       # Confidence in sentiment reading

@dataclass
class AggregatedSentiment:
    """Aggregated sentiment analysis"""
    token_address: str
    period_minutes: int
    sentiment_trend: str    # "BULLISH", "BEARISH", "NEUTRAL"
    sentiment_score: float  # -1 to 1
    sentiment_strength: float # 0 to 1
    key_factors: List[str]
    sentiment_history: List[SentimentData] = field(default_factory=list)

class SentimentAnalyzer:
    """Advanced multi-dimensional sentiment analyzer"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        
        
        self.sentiment_weights = {
            'wallet': 0.3,
            'volume': 0.25,
            'price': 0.25,
            'holder': 0.2
        }
        
        
        self.sentiment_thresholds = {
            'bullish': 0.3,
            'bearish': -0.3,
            'strong_bullish': 0.6,
            'strong_bearish': -0.6
        }
        
        
        self.sentiment_history: Dict[str, List[SentimentData]] = {}
        self.max_history_length = 1000
        
        
        self._cache = {}
        self._cache_timeout = 60  # seconds
        
    async def analyze_token_sentiment(self, token_address: str, 
                                    market_data: Dict[str, Any]) -> SentimentData:
        """Perform comprehensive sentiment analysis for a token"""
        
        try:
            cache_key = f"{token_address}_{hash(str(market_data))}"
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if (datetime.now() - timestamp).total_seconds() < self._cache_timeout:
                    return cached_data
            
            
            wallet_sentiment = await self._analyze_wallet_sentiment(token_address, market_data)
            volume_sentiment = await self._analyze_volume_sentiment(token_address, market_data)
            price_sentiment = await self._analyze_price_sentiment(token_address, market_data)
            holder_sentiment = await self._analyze_holder_sentiment(token_address, market_data)
            
            
            overall_sentiment = (
                wallet_sentiment * self.sentiment_weights['wallet'] +
                volume_sentiment * self.sentiment_weights['volume'] +
                price_sentiment * self.sentiment_weights['price'] +
                holder_sentiment * self.sentiment_weights['holder']
            )
            
            
            confidence = self._calculate_sentiment_confidence(market_data)
            
            
            sentiment_data = SentimentData(
                token_address=token_address,
                timestamp=datetime.now(),
                wallet_sentiment=wallet_sentiment,
                volume_sentiment=volume_sentiment,
                price_sentiment=price_sentiment,
                holder_sentiment=holder_sentiment,
                overall_sentiment=overall_sentiment,
                confidence=confidence
            )
            
            
            if token_address not in self.sentiment_history:
                self.sentiment_history[token_address] = []
            
            self.sentiment_history[token_address].append(sentiment_data)
            
            
            if len(self.sentiment_history[token_address]) > self.max_history_length:
                self.sentiment_history[token_address] = self.sentiment_history[token_address][-self.max_history_length:]
            
            
            self._cache[cache_key] = (sentiment_data, datetime.now())
            
            self.logger.debug(f"Sentiment analysis for {token_address}: {overall_sentiment:.3f} (confidence: {confidence:.3f})")
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {token_address}: {e}")
            return SentimentData(
                token_address=token_address,
                timestamp=datetime.now(),
                wallet_sentiment=0.0,
                volume_sentiment=0.0,
                price_sentiment=0.0,
                holder_sentiment=0.0,
                overall_sentiment=0.0,
                confidence=0.0
            )
    
    async def _analyze_wallet_sentiment(self, token_address: str, market_data: Dict) -> float:
        """Analyze wallet behavior patterns for sentiment"""
        
        try:
            buy_volume = market_data.get('buy_volume_24h', 0)
            sell_volume = market_data.get('sell_volume_24h', 0)
            
            if buy_volume + sell_volume == 0:
                return 0.0
            
            buy_sell_ratio = buy_volume / (buy_volume + sell_volume)
            
            
            whale_buy_count = market_data.get('whale_buys_24h', 0)
            whale_sell_count = market_data.get('whale_sells_24h', 0)
            
            whale_sentiment = 0.0
            if whale_buy_count + whale_sell_count > 0:
                whale_sentiment = (whale_buy_count - whale_sell_count) / (whale_buy_count + whale_sell_count)
            
            
            holder_growth = market_data.get('holder_growth_24h', 0)
            holder_sentiment = np.tanh(holder_growth / 100.0)  # Normalize
            
            
            wallet_sentiment = (
                (buy_sell_ratio - 0.5) * 2.0 * 0.5 +  # Buy/sell ratio
                whale_sentiment * 0.3 +                # Whale activity
                holder_sentiment * 0.2                 # Holder growth
            )
            
            return np.clip(wallet_sentiment, -1.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Wallet sentiment analysis failed: {e}")
            return 0.0
    
    async def _analyze_volume_sentiment(self, token_address: str, market_data: Dict) -> float:
        """Analyze volume patterns for sentiment"""
        
        try:
            current_volume = market_data.get('volume_24h', 0)
            avg_volume = market_data.get('avg_volume_7d', current_volume)
            
            if avg_volume == 0:
                return 0.0
            
            volume_ratio = current_volume / avg_volume
            volume_sentiment = np.tanh((volume_ratio - 1.0) * 2.0)  # Normalize around 1.0
            
            
            volume_trend = market_data.get('volume_trend_24h', 0)  # Percentage change
            trend_sentiment = np.tanh(volume_trend / 100.0)
            
            
            combined_sentiment = volume_sentiment * 0.7 + trend_sentiment * 0.3
            
            return np.clip(combined_sentiment, -1.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Volume sentiment analysis failed: {e}")
            return 0.0
    
    async def _analyze_price_sentiment(self, token_address: str, market_data: Dict) -> float:
        """Analyze price action patterns for sentiment"""
        
        try:
            price_change_1h = market_data.get('price_change_1h', 0)
            price_change_24h = market_data.get('price_change_24h', 0)
            
            
            momentum_sentiment = (
                np.tanh(price_change_1h / 20.0) * 0.6 +  # 1h change
                np.tanh(price_change_24h / 50.0) * 0.4   # 24h change
            )
            
            
            volatility = market_data.get('price_volatility_24h', 0)
            if price_change_24h > 0:
                volatility_factor = min(volatility / 100.0, 0.5)  # Positive contribution
            else:
                volatility_factor = -min(volatility / 100.0, 0.5)  # Negative contribution
            
            price_sentiment = momentum_sentiment + volatility_factor
            
            return np.clip(price_sentiment, -1.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Price sentiment analysis failed: {e}")
            return 0.0
    
    async def _analyze_holder_sentiment(self, token_address: str, market_data: Dict) -> float:
        """Analyze holder distribution for sentiment"""
        
        try:
            top_10_holders_percent = market_data.get('top_10_holders_percent', 0)
            
            
            concentration_sentiment = np.tanh((40 - top_10_holders_percent) / 20.0)
            
            
            new_holders_24h = market_data.get('new_holders_24h', 0)
            new_holder_sentiment = np.tanh(new_holders_24h / 100.0)
            
            
            avg_hold_time_hours = market_data.get('avg_hold_time_hours', 24)
            hold_time_sentiment = np.tanh((avg_hold_time_hours - 24) / 48.0)
            
            
            holder_sentiment = (
                concentration_sentiment * 0.4 +
                new_holder_sentiment * 0.4 +
                hold_time_sentiment * 0.2
            )
            
            return np.clip(holder_sentiment, -1.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Holder sentiment analysis failed: {e}")
            return 0.0
    
    def _calculate_sentiment_confidence(self, market_data: Dict) -> float:
        """Calculate confidence in sentiment analysis based on data quality"""
        
        confidence_factors = []
        
        
        required_fields = ['volume_24h', 'price_change_24h', 'holder_count']
        available_fields = sum(1 for field in required_fields if market_data.get(field) is not None)
        data_completeness = available_fields / len(required_fields)
        confidence_factors.append(data_completeness)
        
        
        volume_24h = market_data.get('volume_24h', 0)
        volume_confidence = min(volume_24h / 10000.0, 1.0)  # Normalize to 10k volume
        confidence_factors.append(volume_confidence)
        
        
        holder_count = market_data.get('holder_count', 0)
        holder_confidence = min(holder_count / 1000.0, 1.0)  # Normalize to 1k holders
        confidence_factors.append(holder_confidence)
        
        
        token_age_hours = market_data.get('age_hours', 0)
        age_confidence = min(token_age_hours / 168.0, 1.0)  # Normalize to 1 week
        confidence_factors.append(age_confidence)
        
        
        overall_confidence = np.mean(confidence_factors)
        
        return overall_confidence
    
    async def get_aggregated_sentiment(self, token_address: str, 
                                     period_minutes: int = 60) -> AggregatedSentiment:
        """Get aggregated sentiment analysis for a token over a time period"""
        
        if token_address not in self.sentiment_history:
            return AggregatedSentiment(
                token_address=token_address,
                period_minutes=period_minutes,
                sentiment_trend="NEUTRAL",
                sentiment_score=0.0,
                sentiment_strength=0.0,
                key_factors=["Insufficient data"]
            )
        
        
        cutoff_time = datetime.now() - timedelta(minutes=period_minutes)
        recent_sentiments = [
            s for s in self.sentiment_history[token_address] 
            if s.timestamp >= cutoff_time
        ]
        
        if not recent_sentiments:
            return AggregatedSentiment(
                token_address=token_address,
                period_minutes=period_minutes,
                sentiment_trend="NEUTRAL",
                sentiment_score=0.0,
                sentiment_strength=0.0,
                key_factors=["No recent data"]
            )
        
        
        avg_sentiment = np.mean([s.overall_sentiment for s in recent_sentiments])
        sentiment_strength = np.mean([abs(s.overall_sentiment) for s in recent_sentiments])
        
        
        if avg_sentiment >= self.sentiment_thresholds['bullish']:
            trend = "BULLISH"
        elif avg_sentiment <= self.sentiment_thresholds['bearish']:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        
        key_factors = self._identify_key_sentiment_factors(recent_sentiments)
        
        return AggregatedSentiment(
            token_address=token_address,
            period_minutes=period_minutes,
            sentiment_trend=trend,
            sentiment_score=avg_sentiment,
            sentiment_strength=sentiment_strength,
            key_factors=key_factors,
            sentiment_history=recent_sentiments
        )
    
    def _identify_key_sentiment_factors(self, sentiments: List[SentimentData]) -> List[str]:
        """Identify key factors driving sentiment"""
        
        factors = []
        
        if not sentiments:
            return ["Insufficient data"]
        
        
        avg_wallet = np.mean([s.wallet_sentiment for s in sentiments])
        avg_volume = np.mean([s.volume_sentiment for s in sentiments])
        avg_price = np.mean([s.price_sentiment for s in sentiments])
        avg_holder = np.mean([s.holder_sentiment for s in sentiments])
        
        sentiment_components = [
            ('wallet', abs(avg_wallet)),
            ('volume', abs(avg_volume)),
            ('price', abs(avg_price)),
            ('holder', abs(avg_holder))
        ]
        
        
        sentiment_components.sort(key=lambda x: x[1], reverse=True)
        
        for component, strength in sentiment_components[:2]:
            if strength > 0.2:
                factors.append(f"Strong {component} signals")
        
        return factors if factors else ["Mixed signals"]
    
    async def analyze_sentiment(self, text: str) -> float:
        """Basic text sentiment analysis (fallback method)"""
        try:
            positive_keywords = ['moon', 'pump', 'bullish', 'buy', 'hold', 'gem', 'rocket']
            negative_keywords = ['dump', 'bearish', 'sell', 'rug', 'scam', 'exit']
            
            text_lower = text.lower()
            
            positive_score = sum(1 for word in positive_keywords if word in text_lower)
            negative_score = sum(1 for word in negative_keywords if word in text_lower)
            
            if positive_score + negative_score == 0:
                return 0.0
            
            sentiment = (positive_score - negative_score) / (positive_score + negative_score)
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {text[:50]}...: {e}")
            return 0.0 