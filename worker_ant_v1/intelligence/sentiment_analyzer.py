"""
Aggressive Memecoin Sentiment Analysis Engine
==========================================

Real-time sentiment analysis from multiple sources for aggressive memecoin trading.
Integrates Twitter, Reddit, Telegram, and Discord sentiment data.
"""

import asyncio
import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

# External dependencies - conditional imports
try:
    import praw
    import aiohttp
    import pandas as pd
    import numpy as np
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import tweepy
    EXTERNAL_DEPS_AVAILABLE = True
except ImportError:
    # Create mock classes for development/testing
    EXTERNAL_DEPS_AVAILABLE = False
    
    class praw:
        class Reddit:
            def __init__(self, *args, **kwargs): pass
    
    class aiohttp:
        @staticmethod
        def ClientSession(*args, **kwargs):
            class MockSession:
                async def get(self, *args, **kwargs):
                    class MockResponse:
                        async def json(self): return {}
                        async def text(self): return ""
                    return MockResponse()
                async def __aenter__(self): return self
                async def __aexit__(self, *args): pass
            return MockSession()
    
    class pd:
        @staticmethod
        def DataFrame(*args, **kwargs): return []
    
    class np:
        @staticmethod
        def mean(data): return 0.5
        @staticmethod
        def std(data): return 0.1
    
    class TextBlob:
        def __init__(self, text): 
            self.sentiment = type('obj', (object,), {'polarity': 0.0, 'subjectivity': 0.5})()
    
    class SentimentIntensityAnalyzer:
        def polarity_scores(self, text):
            return {'compound': 0.0, 'pos': 0.33, 'neu': 0.34, 'neg': 0.33}
    
    class torch:
        class cuda:
            @staticmethod
            def is_available():
                return False
    
    def pipeline(*args, **kwargs):
        def mock_pipeline(text):
            return [{'label': 'POSITIVE', 'score': 0.5}]
        return mock_pipeline
    
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs): return None
    
    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(*args, **kwargs): return None
    
    class tweepy:
        class API: 
            def __init__(self, *args, **kwargs): pass
        class Client:
            def __init__(self, *args, **kwargs): pass

# Internal imports
from worker_ant_v1.config.swarm_config import aggressive_meme_strategy, ml_model_config
from worker_ant_v1.utils.simple_logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class SentimentData:
    """Sentiment analysis result data structure"""
    token_symbol: str
    timestamp: datetime
    source: str  # twitter, reddit, telegram, discord
    sentiment_score: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0 to 1
    mention_count: int
    engagement_score: float  # likes, shares, comments weighted
    trending_score: float  # velocity of mentions
    raw_data: List[Dict] = field(default_factory=list)


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment across all sources"""
    token_symbol: str
    timestamp: datetime
    overall_sentiment: float  # -1 to +1
    confidence: float
    total_mentions: int
    social_dominance: float  # relative mention volume
    momentum_score: float  # sentiment velocity
    source_breakdown: Dict[str, float] = field(default_factory=dict)


class SentimentAnalyzer:
    """Advanced sentiment analysis engine for memecoin trading"""
    
    def __init__(self):
        self.twitter_api = None
        self.reddit_api = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.crypto_sentiment_model = None
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Initialize ML sentiment model
        self._initialize_ml_models()
        
        # Initialize APIs
        self._initialize_apis()
        
        # Tracking data
        self.sentiment_history: Dict[str, List[SentimentData]] = {}
        self.aggregated_history: Dict[str, List[AggregatedSentiment]] = {}
        
    def _initialize_ml_models(self):
        """Initialize pre-trained sentiment models"""
        try:
            # Use crypto-specific sentiment model if available
            model_name = "ElKulako/cryptobert"
            self.crypto_sentiment_model = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Loaded crypto-specific sentiment model")
        except Exception as e:
            logger.warning(f"Failed to load crypto model, using default: {e}")
            self.crypto_sentiment_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
    
    def _initialize_apis(self):
        """Initialize social media APIs"""
        try:
            # Twitter API v2
            twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
            if twitter_bearer_token:
                self.twitter_api = tweepy.Client(bearer_token=twitter_bearer_token)
                logger.info("Twitter API initialized")
            
            # Reddit API
            reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
            reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
            reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "MemecoinTrader/1.0")
            
            if reddit_client_id and reddit_client_secret:
                self.reddit_api = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent
                )
                logger.info("Reddit API initialized")
                
        except Exception as e:
            logger.error(f"API initialization error: {e}")
    
    async def analyze_token_sentiment(self, token_symbol: str, lookback_hours: int = 1) -> AggregatedSentiment:
        """Analyze sentiment for a specific token across all sources"""
        
        tasks = []
        
        # Twitter sentiment
        if aggressive_meme_strategy.enable_twitter_sentiment and self.twitter_api:
            tasks.append(self._analyze_twitter_sentiment(token_symbol, lookback_hours))
        
        # Reddit sentiment
        if aggressive_meme_strategy.enable_reddit_sentiment and self.reddit_api:
            tasks.append(self._analyze_reddit_sentiment(token_symbol, lookback_hours))
        
        # Telegram sentiment
        if aggressive_meme_strategy.enable_telegram_monitoring:
            tasks.append(self._analyze_telegram_sentiment(token_symbol, lookback_hours))
        
        # Execute all sentiment analysis in parallel
        sentiment_results = []
        if tasks:
            sentiment_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in sentiment_results if not isinstance(r, Exception)]
        
        # Aggregate sentiment data
        aggregated = self._aggregate_sentiment_data(token_symbol, valid_results)
        
        # Store in history
        if token_symbol not in self.aggregated_history:
            self.aggregated_history[token_symbol] = []
        self.aggregated_history[token_symbol].append(aggregated)
        
        # Keep only last 24 hours of data
        cutoff = datetime.now() - timedelta(hours=24)
        self.aggregated_history[token_symbol] = [
            s for s in self.aggregated_history[token_symbol] 
            if s.timestamp > cutoff
        ]
        
        return aggregated
    
    async def _analyze_twitter_sentiment(self, token_symbol: str, hours: int) -> SentimentData:
        """Analyze Twitter sentiment for token"""
        
        search_terms = [
            f"${token_symbol}",
            f"#{token_symbol}",
            token_symbol,
            f"{token_symbol} crypto",
            f"{token_symbol} moon",
            f"{token_symbol} pump"
        ]
        
        all_tweets = []
        total_engagement = 0
        
        try:
            for term in search_terms[:3]:  # Limit to avoid rate limits
                tweets = tweepy.Paginator(
                    self.twitter_api.search_recent_tweets,
                    query=f"{term} -is:retweet lang:en",
                    tweet_fields=['created_at', 'public_metrics', 'context_annotations'],
                    max_results=100
                ).flatten(limit=500)
                
                for tweet in tweets:
                    if tweet.created_at > datetime.now() - timedelta(hours=hours):
                        metrics = tweet.public_metrics
                        engagement = (
                            metrics['like_count'] * 1 +
                            metrics['retweet_count'] * 2 +
                            metrics['reply_count'] * 1.5 +
                            metrics['quote_count'] * 1.5
                        )
                        
                        all_tweets.append({
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'engagement': engagement
                        })
                        total_engagement += engagement
                
                # Rate limit protection
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Twitter API error for {token_symbol}: {e}")
            return SentimentData(
                token_symbol=token_symbol,
                timestamp=datetime.now(),
                source="twitter",
                sentiment_score=0.0,
                confidence=0.0,
                mention_count=0,
                engagement_score=0.0,
                trending_score=0.0
            )
        
        if not all_tweets:
            return SentimentData(
                token_symbol=token_symbol,
                timestamp=datetime.now(),
                source="twitter",
                sentiment_score=0.0,
                confidence=0.0,
                mention_count=0,
                engagement_score=0.0,
                trending_score=0.0
            )
        
        # Analyze sentiment
        sentiment_scores = []
        for tweet in all_tweets:
            score = self._analyze_text_sentiment(tweet['text'])
            # Weight by engagement
            weight = min(tweet['engagement'] / 100, 5.0)  # Cap at 5x weight
            sentiment_scores.extend([score] * int(weight + 1))
        
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        confidence = min(len(all_tweets) / 50, 1.0)  # Higher confidence with more tweets
        
        # Calculate trending score (mentions per hour)
        trending_score = len(all_tweets) / hours
        
        return SentimentData(
            token_symbol=token_symbol,
            timestamp=datetime.now(),
            source="twitter",
            sentiment_score=avg_sentiment,
            confidence=confidence,
            mention_count=len(all_tweets),
            engagement_score=total_engagement,
            trending_score=trending_score,
            raw_data=all_tweets
        )
    
    async def _analyze_reddit_sentiment(self, token_symbol: str, hours: int) -> SentimentData:
        """Analyze Reddit sentiment for token"""
        
        subreddits = [
            "CryptoMoonShots", "SatoshiStreetBets", "CryptoCurrency", 
            "altcoin", "CryptoMarkets", "defi", "solana"
        ]
        
        all_posts = []
        total_engagement = 0
        
        try:
            for subreddit_name in subreddits:
                subreddit = self.reddit_api.subreddit(subreddit_name)
                
                # Search recent posts
                for post in subreddit.search(token_symbol, sort="new", time_filter="day", limit=20):
                    post_time = datetime.fromtimestamp(post.created_utc)
                    if post_time > datetime.now() - timedelta(hours=hours):
                        
                        engagement = post.score + post.num_comments * 2
                        
                        # Include post title and body
                        text = f"{post.title} {post.selftext}"
                        
                        all_posts.append({
                            'text': text,
                            'created_at': post_time,
                            'engagement': engagement,
                            'subreddit': subreddit_name
                        })
                        total_engagement += engagement
                
                # Rate limit protection
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"Reddit API error for {token_symbol}: {e}")
            return SentimentData(
                token_symbol=token_symbol,
                timestamp=datetime.now(),
                source="reddit",
                sentiment_score=0.0,
                confidence=0.0,
                mention_count=0,
                engagement_score=0.0,
                trending_score=0.0
            )
        
        if not all_posts:
            return SentimentData(
                token_symbol=token_symbol,
                timestamp=datetime.now(),
                source="reddit",
                sentiment_score=0.0,
                confidence=0.0,
                mention_count=0,
                engagement_score=0.0,
                trending_score=0.0
            )
        
        # Analyze sentiment
        sentiment_scores = []
        for post in all_posts:
            score = self._analyze_text_sentiment(post['text'])
            # Weight by engagement (Reddit scores can be higher)
            weight = min(post['engagement'] / 10, 3.0)
            sentiment_scores.extend([score] * int(weight + 1))
        
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        confidence = min(len(all_posts) / 10, 1.0)
        trending_score = len(all_posts) / hours
        
        return SentimentData(
            token_symbol=token_symbol,
            timestamp=datetime.now(),
            source="reddit",
            sentiment_score=avg_sentiment,
            confidence=confidence,
            mention_count=len(all_posts),
            engagement_score=total_engagement,
            trending_score=trending_score,
            raw_data=all_posts
        )
    
    async def _analyze_telegram_sentiment(self, token_symbol: str, hours: int) -> SentimentData:
        """Analyze Telegram sentiment (placeholder for Telegram API integration)"""
        
        # Telegram analysis would require specific bot setup and channel access
        # This is a placeholder that can be extended with actual Telegram API
        
        logger.info(f"Telegram sentiment analysis for {token_symbol} (placeholder)")
        
        return SentimentData(
            token_symbol=token_symbol,
            timestamp=datetime.now(),
            source="telegram",
            sentiment_score=0.0,
            confidence=0.0,
            mention_count=0,
            engagement_score=0.0,
            trending_score=0.0
        )
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of individual text using multiple methods"""
        
        # Clean text
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
        
        scores = []
        
        # VADER sentiment
        vader_score = self.vader_analyzer.polarity_scores(text)['compound']
        scores.append(vader_score)
        
        # TextBlob sentiment
        blob_score = TextBlob(text).sentiment.polarity
        scores.append(blob_score)
        
        # Crypto-specific model
        if self.crypto_sentiment_model and len(text.strip()) > 0:
            try:
                result = self.crypto_sentiment_model(text[:512])  # Truncate for model
                if result[0]['label'] == 'POSITIVE':
                    ml_score = result[0]['score']
                elif result[0]['label'] == 'NEGATIVE':
                    ml_score = -result[0]['score']
                else:  # NEUTRAL
                    ml_score = 0.0
                scores.append(ml_score)
            except:
                pass
        
        # Return weighted average
        if scores:
            return np.mean(scores)
        return 0.0
    
    def _aggregate_sentiment_data(self, token_symbol: str, sentiment_data: List[SentimentData]) -> AggregatedSentiment:
        """Aggregate sentiment data from multiple sources"""
        
        if not sentiment_data:
            return AggregatedSentiment(
                token_symbol=token_symbol,
                timestamp=datetime.now(),
                overall_sentiment=0.0,
                confidence=0.0,
                total_mentions=0,
                social_dominance=0.0,
                momentum_score=0.0
            )
        
        # Weight sentiment by confidence and mention count
        weighted_sentiment = 0.0
        total_weight = 0.0
        total_mentions = 0
        source_breakdown = {}
        
        for data in sentiment_data:
            weight = data.confidence * (1 + np.log(1 + data.mention_count))
            weighted_sentiment += data.sentiment_score * weight
            total_weight += weight
            total_mentions += data.mention_count
            source_breakdown[data.source] = data.sentiment_score
        
        overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        # Calculate social dominance (relative to baseline)
        baseline_mentions = 10  # Baseline mention count
        social_dominance = min(total_mentions / baseline_mentions, 5.0)  # Cap at 5x
        
        # Calculate momentum (change in sentiment over time)
        momentum_score = self._calculate_momentum(token_symbol, overall_sentiment)
        
        # Overall confidence based on total mentions and source diversity
        confidence = min(total_mentions / 20, 1.0) * min(len(sentiment_data) / 2, 1.0)
        
        return AggregatedSentiment(
            token_symbol=token_symbol,
            timestamp=datetime.now(),
            overall_sentiment=overall_sentiment,
            confidence=confidence,
            total_mentions=total_mentions,
            social_dominance=social_dominance,
            momentum_score=momentum_score,
            source_breakdown=source_breakdown
        )
    
    def _calculate_momentum(self, token_symbol: str, current_sentiment: float) -> float:
        """Calculate sentiment momentum based on historical data"""
        
        if token_symbol not in self.aggregated_history:
            return 0.0
        
        history = self.aggregated_history[token_symbol]
        if len(history) < 2:
            return 0.0
        
        # Calculate sentiment change over last hour
        recent_sentiments = [h.overall_sentiment for h in history[-6:]]  # Last 6 data points
        if len(recent_sentiments) >= 2:
            momentum = recent_sentiments[-1] - recent_sentiments[0]
            return np.tanh(momentum * 5)  # Scale and bound between -1 and 1
        
        return 0.0
    
    def get_sentiment_signals(self, token_symbol: str) -> Dict[str, float]:
        """Get trading signals based on sentiment analysis"""
        
        if token_symbol not in self.aggregated_history or not self.aggregated_history[token_symbol]:
            return {
                'sentiment_signal': 0.0,
                'confidence': 0.0,
                'trend_strength': 0.0,
                'social_buzz': 0.0
            }
        
        latest = self.aggregated_history[token_symbol][-1]
        
        # Generate signals
        sentiment_signal = latest.overall_sentiment  # -1 to +1
        trend_strength = abs(latest.momentum_score)   # 0 to 1
        social_buzz = min(latest.social_dominance / 2, 1.0)  # 0 to 1
        
        return {
            'sentiment_signal': sentiment_signal,
            'confidence': latest.confidence,
            'trend_strength': trend_strength,
            'social_buzz': social_buzz,
            'total_mentions': latest.total_mentions,
            'momentum': latest.momentum_score
        }


# Remove module-level instantiation to prevent import errors
# sentiment_analyzer = SentimentAnalyzer() 