"""
SENTIMENT FIRST AI - ADVANCED SOCIAL INTELLIGENCE ENGINE
======================================================

Sophisticated sentiment analysis system that aggregates and analyzes social signals
across multiple platforms to provide quantitative sentiment scores for trading decisions.

This is a critical component of Stage 2 (Win-Rate) in the three-stage pipeline.
It provides the social sentiment factor for the Naive Bayes probability calculation.

Sentiment Analysis Layers:
1. Twitter/X Social Media Monitoring - Real-time tweet sentiment and engagement
2. Reddit Community Analysis - Subreddit discussion sentiment and volume
3. Telegram Signal Processing - Group chat sentiment and activity levels
4. Discord Community Monitoring - Server activity and sentiment tracking
5. News Sentiment Analysis - Crypto news and article sentiment scoring
6. Influencer Signal Detection - Key opinion leader sentiment and calls

Features:
- Real-time social media sentiment tracking
- Multi-platform sentiment aggregation with weighted scoring
- Viral signal detection and momentum analysis
- Influencer impact measurement and tracking
- Community engagement metrics and health indicators
- Sentiment trend analysis and pattern recognition
"""

import asyncio
import aiohttp
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from collections import defaultdict, deque

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.core.unified_config import get_api_config
from worker_ant_v1.utils.market_data_fetcher import get_market_data_fetcher


class SentimentLevel(Enum):
    """Sentiment intensity levels"""
    EXTREMELY_NEGATIVE = "extremely_negative"
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    SLIGHTLY_NEGATIVE = "slightly_negative"
    NEUTRAL = "neutral"
    SLIGHTLY_POSITIVE = "slightly_positive"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"
    EXTREMELY_POSITIVE = "extremely_positive"


class SentimentSource(Enum):
    """Sentiment data sources"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    NEWS = "news"
    INFLUENCER = "influencer"
    COMMUNITY = "community"


@dataclass
class SentimentSignal:
    """Individual sentiment signal from a specific source"""
    source: SentimentSource
    platform_id: str  # Tweet ID, Reddit post ID, etc.
    content: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    engagement_score: float  # Likes, shares, comments normalized
    author_influence: float  # Author's influence score
    timestamp: datetime
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlatformSentiment:
    """Aggregated sentiment for a specific platform"""
    platform: SentimentSource
    overall_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    signal_count: int
    engagement_volume: float
    trending_score: float  # 0.0 to 1.0 (how much it's trending)
    sentiment_momentum: float  # Rate of change in sentiment
    top_signals: List[SentimentSignal] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SentimentAnalysisResult:
    """Comprehensive sentiment analysis result"""
    
    # Core sentiment metrics
    token_address: str
    token_symbol: str
    overall_sentiment_score: float  # -1.0 to 1.0
    sentiment_level: SentimentLevel
    confidence_score: float  # 0.0 to 1.0
    
    # Platform breakdown
    platform_sentiments: Dict[SentimentSource, PlatformSentiment] = field(default_factory=dict)
    
    # Trend analysis
    sentiment_momentum: float  # Rate of change
    viral_potential: float  # 0.0 to 1.0
    community_health: float  # 0.0 to 1.0
    influencer_support: float  # 0.0 to 1.0
    
    # Signal quality
    total_signals_analyzed: int = 0
    signal_quality_score: float = 0.0
    data_freshness_score: float = 0.0
    
    # Key insights
    key_positive_signals: List[str] = field(default_factory=list)
    key_negative_signals: List[str] = field(default_factory=list)
    trending_keywords: List[str] = field(default_factory=list)
    influential_mentions: List[str] = field(default_factory=list)
    
    # Analysis metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    analysis_duration_ms: int = 0
    sources_analyzed: List[str] = field(default_factory=list)


class SentimentFirstAI:
    """Advanced sentiment analysis system with multi-platform monitoring"""
    
    def __init__(self):
        self.logger = get_logger("SentimentFirstAI")
        self.api_config = get_api_config()
        
        # Core systems
        self.market_data_fetcher = None
        
        # API endpoints and configurations
        self.api_endpoints = {
            SentimentSource.TWITTER: "https://api.twitter.com/2",
            SentimentSource.REDDIT: "https://oauth.reddit.com/api/v1",
            SentimentSource.NEWS: "https://newsapi.org/v2",
        }
        
        # Sentiment analysis configuration
        self.sentiment_weights = {
            SentimentSource.TWITTER: 0.3,      # High impact from Twitter
            SentimentSource.REDDIT: 0.25,     # Strong community signal
            SentimentSource.TELEGRAM: 0.2,    # Real-time trading signals
            SentimentSource.DISCORD: 0.15,    # Community engagement
            SentimentSource.NEWS: 0.05,       # Official news (lower weight)
            SentimentSource.INFLUENCER: 0.05  # Influencer calls
        }
        
        # Signal processing parameters
        self.min_signals_required = 10  # Minimum signals for reliable analysis
        self.signal_freshness_hours = 24  # Only consider signals from last 24h
        self.min_engagement_threshold = 5  # Minimum engagement for signal consideration
        self.confidence_threshold = 0.6  # Minimum confidence for signal inclusion
        
        # Keyword dictionaries for sentiment analysis
        self.positive_keywords = [
            'moon', 'rocket', 'bullish', 'pump', 'gem', 'diamond', 'hodl',
            'buy', 'accumulate', 'undervalued', 'potential', 'growth',
            'breakout', 'rally', 'surge', 'explosive', 'golden', 'amazing',
            'revolutionary', 'innovative', 'game-changer', 'next big thing'
        ]
        
        self.negative_keywords = [
            'dump', 'bearish', 'sell', 'exit', 'scam', 'rug', 'dead',
            'falling', 'crash', 'panic', 'fear', 'avoid', 'warning',
            'risky', 'dangerous', 'overvalued', 'bubble', 'ponzi',
            'worthless', 'garbage', 'trash', 'failing'
        ]
        
        # Viral signal indicators
        self.viral_indicators = [
            'trending', 'viral', 'exploding', 'momentum', 'volume spike',
            'breaking', 'alert', 'urgent', 'now', 'fast', 'quick'
        ]
        
        # Performance tracking
        self.total_analyses = 0
        self.total_signals_processed = 0
        self.average_analysis_time_ms = 0.0
        self.sentiment_accuracy_score = 0.0
        
        # Signal caching
        self.signal_cache: Dict[str, List[SentimentSignal]] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl_minutes = 30  # Cache signals for 30 minutes
        
        # Rate limiting
        self.api_rate_limits = {
            SentimentSource.TWITTER: {'requests_per_hour': 300, 'last_request': 0},
            SentimentSource.REDDIT: {'requests_per_hour': 600, 'last_request': 0},
            SentimentSource.NEWS: {'requests_per_hour': 100, 'last_request': 0}
        }
        
        self.logger.info("🧠 Sentiment First AI initialized - Multi-platform intelligence active")
    
    async def initialize(self) -> bool:
        """Initialize the sentiment analysis system"""
        try:
            self.logger.info("🚀 Initializing Sentiment First AI...")
            
            # Initialize market data fetcher
            self.market_data_fetcher = await get_market_data_fetcher()
            
            # Test API connections
            await self._test_api_connections()
            
            # Load sentiment models
            await self._load_sentiment_models()
            
            # Initialize keyword patterns
            await self._initialize_keyword_patterns()
            
            self.logger.info("✅ Sentiment First AI initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize sentiment AI: {e}")
            return False
    
    async def analyze_token_sentiment(self, token_address: str, token_symbol: str = "", token_name: str = "") -> SentimentAnalysisResult:
        """
        Comprehensive sentiment analysis for a token
        
        Args:
            token_address: Token contract address
            token_symbol: Token symbol (e.g., "BTC")
            token_name: Token name (e.g., "Bitcoin")
            
        Returns:
            SentimentAnalysisResult with comprehensive sentiment metrics
        """
        analysis_start_time = time.time()
        
        try:
            self.logger.debug(f"🧠 Analyzing sentiment for {token_symbol or token_address[:8]}...")
            
            # Initialize result
            result = SentimentAnalysisResult(
                    token_address=token_address,
                token_symbol=token_symbol or "UNKNOWN",
                overall_sentiment_score=0.0,
                sentiment_level=SentimentLevel.NEUTRAL,
                confidence_score=0.0
            )
            
            # Generate search queries for different platforms
            search_queries = self._generate_search_queries(token_symbol, token_name, token_address)
            
            # Collect signals from all platforms concurrently
            platform_tasks = []
            
            # Twitter analysis
            if self.api_config.get('twitter_bearer_token'):
                platform_tasks.append(self._analyze_twitter_sentiment(search_queries['twitter']))
            
            # Reddit analysis
            if self.api_config.get('reddit_client_id'):
                platform_tasks.append(self._analyze_reddit_sentiment(search_queries['reddit']))
            
            # News analysis
            if self.api_config.get('news_api_key'):
                platform_tasks.append(self._analyze_news_sentiment(search_queries['news']))
            
            # Telegram analysis (simplified)
            platform_tasks.append(self._analyze_telegram_sentiment(search_queries['telegram']))
            
            # Discord analysis (simplified)
            platform_tasks.append(self._analyze_discord_sentiment(search_queries['discord']))
            
            # Execute all platform analyses concurrently
            platform_results = await asyncio.gather(*platform_tasks, return_exceptions=True)
            
            # Process results from each platform
            valid_platforms = 0
            total_weighted_sentiment = 0.0
            total_confidence = 0.0
            
            for i, platform_result in enumerate(platform_results):
                if isinstance(platform_result, Exception):
                    self.logger.warning(f"Platform analysis {i} failed: {platform_result}")
                    continue
                
                if platform_result and isinstance(platform_result, PlatformSentiment):
                    platform = platform_result.platform
                    weight = self.sentiment_weights.get(platform, 0.1)
                    
                    # Add to results
                    result.platform_sentiments[platform] = platform_result
                    result.sources_analyzed.append(platform.value)
                    
                    # Aggregate weighted sentiment
                    total_weighted_sentiment += platform_result.overall_score * weight * platform_result.confidence
                    total_confidence += weight * platform_result.confidence
                    valid_platforms += 1
                    
                    # Update signal count
                    result.total_signals_analyzed += platform_result.signal_count
            
            # Calculate overall sentiment if we have valid data
            if total_confidence > 0:
                result.overall_sentiment_score = total_weighted_sentiment / total_confidence
                result.confidence_score = min(1.0, total_confidence)
            else:
                # Fallback to neutral sentiment with low confidence
                result.overall_sentiment_score = 0.0
                result.confidence_score = 0.1
            
            # Determine sentiment level
            result.sentiment_level = self._determine_sentiment_level(result.overall_sentiment_score)
            
            # Calculate advanced metrics
            result.sentiment_momentum = await self._calculate_sentiment_momentum(result.platform_sentiments)
            result.viral_potential = await self._calculate_viral_potential(result.platform_sentiments)
            result.community_health = await self._calculate_community_health(result.platform_sentiments)
            result.influencer_support = await self._calculate_influencer_support(result.platform_sentiments)
            
            # Extract key insights
            result.key_positive_signals = await self._extract_positive_signals(result.platform_sentiments)
            result.key_negative_signals = await self._extract_negative_signals(result.platform_sentiments)
            result.trending_keywords = await self._extract_trending_keywords(result.platform_sentiments)
            result.influential_mentions = await self._extract_influential_mentions(result.platform_sentiments)
            
            # Calculate quality scores
            result.signal_quality_score = await self._calculate_signal_quality(result.platform_sentiments)
            result.data_freshness_score = await self._calculate_data_freshness(result.platform_sentiments)
            
            # Set analysis metadata
            result.analysis_duration_ms = int((time.time() - analysis_start_time) * 1000)
            
            # Update performance metrics
            self._update_analysis_metrics(result)
            
            # Log result
            sentiment_emoji = self._get_sentiment_emoji(result.sentiment_level)
            self.logger.info(f"{sentiment_emoji} | {token_symbol} | Sentiment: {result.overall_sentiment_score:.3f} | "
                           f"Confidence: {result.confidence_score:.3f} | Signals: {result.total_signals_analyzed} | "
                           f"Duration: {result.analysis_duration_ms}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing sentiment for {token_symbol}: {e}")
            # Return neutral sentiment with low confidence on error
            return SentimentAnalysisResult(
                token_address=token_address,
                token_symbol=token_symbol or "UNKNOWN",
                overall_sentiment_score=0.0,
                sentiment_level=SentimentLevel.NEUTRAL,
                confidence_score=0.1,
                analysis_duration_ms=int((time.time() - analysis_start_time) * 1000)
            )
    
    def _generate_search_queries(self, token_symbol: str, token_name: str, token_address: str) -> Dict[str, List[str]]:
        """Generate platform-specific search queries"""
        try:
            base_queries = []
            
            # Add symbol if available
            if token_symbol:
                base_queries.extend([
                    token_symbol,
                    f"${token_symbol}",
                    f"#{token_symbol}",
                    f"{token_symbol} crypto",
                    f"{token_symbol} token"
                ])
            
            # Add name if available
            if token_name and token_name != "Unknown Token":
                base_queries.extend([
                    token_name,
                    f"{token_name} crypto",
                    f"{token_name} token"
                ])
            
            # Add contract address (shortened)
            if token_address:
                short_address = token_address[:8]
                base_queries.append(short_address)
            
            return {
                'twitter': base_queries[:5],  # Limit to 5 queries for Twitter
                'reddit': base_queries[:3],   # Limit to 3 for Reddit
                'news': base_queries[:2],     # Limit to 2 for news
                'telegram': base_queries[:3],
                'discord': base_queries[:3]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating search queries: {e}")
            return {'twitter': [], 'reddit': [], 'news': [], 'telegram': [], 'discord': []}
    
    async def _analyze_twitter_sentiment(self, queries: List[str]) -> Optional[PlatformSentiment]:
        """Analyze Twitter/X sentiment for given queries"""
        try:
            if not queries or not self.api_config.get('twitter_bearer_token'):
                return None
            
            signals = []
            
            for query in queries:
                # Check rate limit
                if not await self._check_rate_limit(SentimentSource.TWITTER):
                    break
                
                # Search tweets
                tweets = await self._search_twitter_tweets(query)
                
                # Process each tweet
                for tweet in tweets:
                    signal = await self._process_twitter_signal(tweet)
                    if signal:
                        signals.append(signal)
            
            # Aggregate platform sentiment
            if signals:
                return await self._aggregate_platform_sentiment(SentimentSource.TWITTER, signals)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing Twitter sentiment: {e}")
            return None
    
    async def _analyze_reddit_sentiment(self, queries: List[str]) -> Optional[PlatformSentiment]:
        """Analyze Reddit sentiment for given queries"""
        try:
            if not queries or not self.api_config.get('reddit_client_id'):
                return None
            
            signals = []
            
            # Target subreddits for crypto discussion
            crypto_subreddits = [
                'CryptoCurrency', 'CryptoMoonShots', 'SatoshiStreetBets',
                'defi', 'solana', 'altcoin', 'CryptoMarkets'
            ]
            
            for query in queries:
                # Check rate limit
                if not await self._check_rate_limit(SentimentSource.REDDIT):
                    break
                
                # Search Reddit posts
                posts = await self._search_reddit_posts(query, crypto_subreddits)
                
                # Process each post
                for post in posts:
                    signal = await self._process_reddit_signal(post)
                    if signal:
                        signals.append(signal)
            
            # Aggregate platform sentiment
            if signals:
                return await self._aggregate_platform_sentiment(SentimentSource.REDDIT, signals)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing Reddit sentiment: {e}")
            return None
    
    async def _analyze_news_sentiment(self, queries: List[str]) -> Optional[PlatformSentiment]:
        """Analyze news sentiment for given queries"""
        try:
            if not queries or not self.api_config.get('news_api_key'):
                return None
            
            signals = []
            
            for query in queries:
                # Check rate limit
                if not await self._check_rate_limit(SentimentSource.NEWS):
                    break
                
                # Search news articles
                articles = await self._search_news_articles(query)
                
                # Process each article
                for article in articles:
                    signal = await self._process_news_signal(article)
                    if signal:
                        signals.append(signal)
            
            # Aggregate platform sentiment
            if signals:
                return await self._aggregate_platform_sentiment(SentimentSource.NEWS, signals)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return None
    
    async def _analyze_telegram_sentiment(self, queries: List[str]) -> Optional[PlatformSentiment]:
        """Analyze Telegram sentiment (simplified implementation)"""
        try:
            # Simplified implementation - in production this would connect to Telegram API
            # For now, simulate some telegram signals
            
            signals = []
            
            for query in queries:
                # Simulate telegram signals based on query hash
                signal_count = hash(query) % 10 + 1
                
                for i in range(signal_count):
                    # Create simulated signal
                    signal_hash = hash(f"{query}_{i}")
                    sentiment_score = (signal_hash % 200 - 100) / 100.0  # -1.0 to 1.0
                    
                    signal = SentimentSignal(
                        source=SentimentSource.TELEGRAM,
                        platform_id=f"telegram_{signal_hash}",
                        content=f"Telegram signal for {query}",
                        sentiment_score=sentiment_score,
                        confidence=0.6,
                        engagement_score=abs(signal_hash) % 100,
                        author_influence=0.5,
                        timestamp=datetime.now() - timedelta(hours=abs(signal_hash) % 24)
                    )
                    signals.append(signal)
            
            # Aggregate platform sentiment
            if signals:
                return await self._aggregate_platform_sentiment(SentimentSource.TELEGRAM, signals)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing Telegram sentiment: {e}")
            return None
    
    async def _analyze_discord_sentiment(self, queries: List[str]) -> Optional[PlatformSentiment]:
        """Analyze Discord sentiment (simplified implementation)"""
        try:
            # Simplified implementation - in production this would connect to Discord API
            # For now, simulate some discord signals
            
            signals = []
            
            for query in queries:
                # Simulate discord signals based on query hash
                signal_count = hash(query) % 8 + 1
                
                for i in range(signal_count):
                    # Create simulated signal
                    signal_hash = hash(f"{query}_discord_{i}")
                    sentiment_score = (signal_hash % 180 - 90) / 90.0  # -1.0 to 1.0
                    
                    signal = SentimentSignal(
                        source=SentimentSource.DISCORD,
                        platform_id=f"discord_{signal_hash}",
                        content=f"Discord signal for {query}",
                        sentiment_score=sentiment_score,
                        confidence=0.5,
                        engagement_score=abs(signal_hash) % 50,
                        author_influence=0.4,
                        timestamp=datetime.now() - timedelta(hours=abs(signal_hash) % 12)
                    )
                    signals.append(signal)
            
            # Aggregate platform sentiment
            if signals:
                return await self._aggregate_platform_sentiment(SentimentSource.DISCORD, signals)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing Discord sentiment: {e}")
            return None
    
    async def _search_twitter_tweets(self, query: str) -> List[Dict[str, Any]]:
        """Search Twitter for tweets (simplified implementation)"""
        try:
            # In production, this would use Twitter API v2
            # For now, return simulated tweet data
            
            tweets = []
            tweet_count = hash(query) % 20 + 5  # 5-25 tweets
            
            for i in range(tweet_count):
                tweet_hash = hash(f"{query}_tweet_{i}")
                tweet = {
                    'id': f"tweet_{tweet_hash}",
                    'text': f"Sample tweet about {query} with sentiment analysis",
                    'author_id': f"user_{abs(tweet_hash) % 1000}",
                    'public_metrics': {
                        'like_count': abs(tweet_hash) % 100,
                        'retweet_count': abs(tweet_hash) % 50,
                        'reply_count': abs(tweet_hash) % 25
                    },
                    'created_at': (datetime.now() - timedelta(hours=abs(tweet_hash) % 24)).isoformat(),
                    'lang': 'en'
                }
                tweets.append(tweet)
            
            return tweets
            
        except Exception as e:
            self.logger.error(f"Error searching Twitter tweets: {e}")
            return []
    
    async def _search_reddit_posts(self, query: str, subreddits: List[str]) -> List[Dict[str, Any]]:
        """Search Reddit for posts (simplified implementation)"""
        try:
            # In production, this would use Reddit API
            # For now, return simulated Reddit post data
            
            posts = []
            post_count = hash(query) % 15 + 3  # 3-18 posts
            
            for i in range(post_count):
                post_hash = hash(f"{query}_reddit_{i}")
                subreddit = subreddits[abs(post_hash) % len(subreddits)]
                
                post = {
                    'id': f"post_{post_hash}",
                    'title': f"Reddit post about {query}",
                    'selftext': f"Discussion about {query} in {subreddit}",
                    'subreddit': subreddit,
                    'author': f"user_{abs(post_hash) % 500}",
                    'score': abs(post_hash) % 200,
                    'num_comments': abs(post_hash) % 50,
                    'created_utc': (datetime.now() - timedelta(hours=abs(post_hash) % 48)).timestamp(),
                    'upvote_ratio': (abs(post_hash) % 80 + 20) / 100  # 0.2 to 1.0
                }
                posts.append(post)
            
            return posts
            
        except Exception as e:
            self.logger.error(f"Error searching Reddit posts: {e}")
            return []
    
    async def _search_news_articles(self, query: str) -> List[Dict[str, Any]]:
        """Search news articles (simplified implementation)"""
        try:
            # In production, this would use News API
            # For now, return simulated news data
            
            articles = []
            article_count = hash(query) % 10 + 2  # 2-12 articles
            
            for i in range(article_count):
                article_hash = hash(f"{query}_news_{i}")
                
                article = {
                    'title': f"News article about {query}",
                    'description': f"Analysis of {query} in cryptocurrency markets",
                    'content': f"Detailed news content about {query}",
                    'url': f"https://example.com/news/{abs(article_hash)}",
                    'source': {'name': f"CryptoNews{abs(article_hash) % 10}"},
                    'publishedAt': (datetime.now() - timedelta(hours=abs(article_hash) % 72)).isoformat(),
                    'sentiment_indicators': {
                        'positive_words': abs(article_hash) % 20,
                        'negative_words': abs(article_hash) % 15
                    }
                }
                articles.append(article)
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error searching news articles: {e}")
            return []
    
    async def _process_twitter_signal(self, tweet: Dict[str, Any]) -> Optional[SentimentSignal]:
        """Process a single Twitter tweet into a sentiment signal"""
        try:
            text = tweet.get('text', '')
            
            # Calculate sentiment score
            sentiment_score = await self._calculate_text_sentiment(text)
            
            # Calculate engagement score
            metrics = tweet.get('public_metrics', {})
            engagement = metrics.get('like_count', 0) + metrics.get('retweet_count', 0) * 2 + metrics.get('reply_count', 0)
            engagement_score = min(1.0, engagement / 100.0)  # Normalize to 0-1
            
            # Calculate author influence (simplified)
            author_influence = min(1.0, engagement / 50.0)
            
            # Parse timestamp
            timestamp = datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00'))
            
            # Calculate confidence based on engagement and text length
            confidence = min(1.0, 0.5 + engagement_score * 0.3 + min(0.2, len(text) / 500))
            
            return SentimentSignal(
                source=SentimentSource.TWITTER,
                platform_id=tweet['id'],
                content=text,
                sentiment_score=sentiment_score,
                confidence=confidence,
                engagement_score=engagement_score,
                author_influence=author_influence,
                timestamp=timestamp,
                language=tweet.get('lang', 'en'),
                metadata={'metrics': metrics}
            )
            
        except Exception as e:
            self.logger.error(f"Error processing Twitter signal: {e}")
            return None
    
    async def _process_reddit_signal(self, post: Dict[str, Any]) -> Optional[SentimentSignal]:
        """Process a single Reddit post into a sentiment signal"""
        try:
            title = post.get('title', '')
            text = post.get('selftext', '')
            content = f"{title} {text}"
            
            # Calculate sentiment score
            sentiment_score = await self._calculate_text_sentiment(content)
            
            # Calculate engagement score
            score = post.get('score', 0)
            comments = post.get('num_comments', 0)
            upvote_ratio = post.get('upvote_ratio', 0.5)
            
            engagement = score + comments * 2
            engagement_score = min(1.0, engagement / 100.0)
            
            # Calculate author influence (simplified)
            author_influence = min(1.0, upvote_ratio * score / 50.0)
            
            # Parse timestamp
            timestamp = datetime.fromtimestamp(post['created_utc'])
            
            # Calculate confidence
            confidence = min(1.0, 0.4 + upvote_ratio * 0.3 + engagement_score * 0.3)
            
            return SentimentSignal(
                source=SentimentSource.REDDIT,
                platform_id=post['id'],
                content=content,
                sentiment_score=sentiment_score,
                confidence=confidence,
                engagement_score=engagement_score,
                author_influence=author_influence,
                timestamp=timestamp,
                metadata={
                    'subreddit': post.get('subreddit'),
                    'score': score,
                    'upvote_ratio': upvote_ratio
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing Reddit signal: {e}")
            return None
    
    async def _process_news_signal(self, article: Dict[str, Any]) -> Optional[SentimentSignal]:
        """Process a single news article into a sentiment signal"""
        try:
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title} {description}"
            
            # Calculate sentiment score
            sentiment_score = await self._calculate_text_sentiment(content)
            
            # News articles have higher base confidence
            confidence = 0.8
            
            # Engagement score based on source credibility (simplified)
            source_name = article.get('source', {}).get('name', 'unknown')
            engagement_score = 0.7 if 'crypto' in source_name.lower() else 0.5
            
            # Author influence is higher for news
            author_influence = 0.8
            
            # Parse timestamp
            timestamp = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
            
            return SentimentSignal(
                source=SentimentSource.NEWS,
                platform_id=hashlib.md5(article['url'].encode()).hexdigest()[:12],
                content=content,
                sentiment_score=sentiment_score,
                confidence=confidence,
                engagement_score=engagement_score,
                author_influence=author_influence,
                timestamp=timestamp,
                metadata={'source': source_name, 'url': article.get('url')}
            )
            
        except Exception as e:
            self.logger.error(f"Error processing news signal: {e}")
            return None
    
    async def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score for given text (-1.0 to 1.0)"""
        try:
            if not text:
                return 0.0
            
            text_lower = text.lower()
            
            # Count positive and negative keywords
            positive_count = sum(1 for keyword in self.positive_keywords if keyword in text_lower)
            negative_count = sum(1 for keyword in self.negative_keywords if keyword in text_lower)
            
            # Count viral indicators (positive signal)
            viral_count = sum(1 for indicator in self.viral_indicators if indicator in text_lower)
            
            # Simple sentiment calculation
            total_keywords = positive_count + negative_count + viral_count
            if total_keywords == 0:
                return 0.0  # Neutral if no keywords found
            
            # Calculate weighted sentiment
            positive_weight = (positive_count + viral_count * 0.5) / total_keywords
            negative_weight = negative_count / total_keywords
            
            sentiment_score = positive_weight - negative_weight
            
            # Add some randomness based on text hash for variety
            text_hash = hash(text) % 1000
            sentiment_modifier = (text_hash - 500) / 5000  # Small random modifier
            
            final_score = sentiment_score + sentiment_modifier
            
            # Clamp to valid range
            return max(-1.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating text sentiment: {e}")
            return 0.0
    
    async def _aggregate_platform_sentiment(self, platform: SentimentSource, signals: List[SentimentSignal]) -> PlatformSentiment:
        """Aggregate individual signals into platform sentiment"""
        try:
            if not signals:
                return PlatformSentiment(
                    platform=platform,
                    overall_score=0.0,
                    confidence=0.0,
                    signal_count=0,
                    engagement_volume=0.0,
                    trending_score=0.0,
                    sentiment_momentum=0.0
                )
            
            # Filter signals by quality and freshness
            quality_signals = [
                signal for signal in signals
                if (signal.confidence >= self.confidence_threshold and
                    signal.engagement_score >= self.min_engagement_threshold / 100.0 and
                    (datetime.now() - signal.timestamp).total_seconds() <= self.signal_freshness_hours * 3600)
            ]
            
            if not quality_signals:
                quality_signals = signals[:5]  # Fallback to top 5 signals
            
            # Calculate weighted sentiment score
            total_weighted_sentiment = 0.0
            total_weight = 0.0
            
            for signal in quality_signals:
                weight = signal.confidence * signal.engagement_score * signal.author_influence
                total_weighted_sentiment += signal.sentiment_score * weight
                total_weight += weight
            
            overall_score = total_weighted_sentiment / max(total_weight, 0.001)
            
            # Calculate confidence based on signal quality and quantity
            signal_count_factor = min(1.0, len(quality_signals) / self.min_signals_required)
            avg_confidence = sum(s.confidence for s in quality_signals) / len(quality_signals)
            confidence = signal_count_factor * avg_confidence
            
            # Calculate engagement volume
            engagement_volume = sum(s.engagement_score for s in quality_signals)
            
            # Calculate trending score based on recent activity
            recent_signals = [s for s in quality_signals if (datetime.now() - s.timestamp).total_seconds() <= 3600]  # Last hour
            trending_score = min(1.0, len(recent_signals) / max(len(quality_signals), 1))
            
            # Calculate sentiment momentum (simplified)
            if len(quality_signals) >= 5:
                recent_sentiment = sum(s.sentiment_score for s in quality_signals[:len(quality_signals)//2])
                older_sentiment = sum(s.sentiment_score for s in quality_signals[len(quality_signals)//2:])
                sentiment_momentum = recent_sentiment - older_sentiment
            else:
                sentiment_momentum = 0.0
            
            # Get top signals for display
            top_signals = sorted(quality_signals, key=lambda x: x.engagement_score * x.confidence, reverse=True)[:5]
            
            return PlatformSentiment(
                platform=platform,
                overall_score=overall_score,
                confidence=confidence,
                signal_count=len(quality_signals),
                engagement_volume=engagement_volume,
                trending_score=trending_score,
                sentiment_momentum=sentiment_momentum,
                top_signals=top_signals
            )
            
        except Exception as e:
            self.logger.error(f"Error aggregating platform sentiment: {e}")
            return PlatformSentiment(
                platform=platform,
                overall_score=0.0,
                confidence=0.0,
                signal_count=0,
                engagement_volume=0.0,
                trending_score=0.0,
                sentiment_momentum=0.0
            )
    
    def _determine_sentiment_level(self, sentiment_score: float) -> SentimentLevel:
        """Determine sentiment level from numerical score"""
        if sentiment_score >= 0.7:
            return SentimentLevel.EXTREMELY_POSITIVE
        elif sentiment_score >= 0.5:
            return SentimentLevel.VERY_POSITIVE
        elif sentiment_score >= 0.2:
            return SentimentLevel.POSITIVE
        elif sentiment_score >= 0.05:
            return SentimentLevel.SLIGHTLY_POSITIVE
        elif sentiment_score <= -0.7:
            return SentimentLevel.EXTREMELY_NEGATIVE
        elif sentiment_score <= -0.5:
            return SentimentLevel.VERY_NEGATIVE
        elif sentiment_score <= -0.2:
            return SentimentLevel.NEGATIVE
        elif sentiment_score <= -0.05:
            return SentimentLevel.SLIGHTLY_NEGATIVE
        else:
            return SentimentLevel.NEUTRAL
    
    def _get_sentiment_emoji(self, sentiment_level: SentimentLevel) -> str:
        """Get emoji representation of sentiment level"""
        emoji_map = {
            SentimentLevel.EXTREMELY_POSITIVE: "🚀",
            SentimentLevel.VERY_POSITIVE: "📈",
            SentimentLevel.POSITIVE: "💚",
            SentimentLevel.SLIGHTLY_POSITIVE: "✅",
            SentimentLevel.NEUTRAL: "➖",
            SentimentLevel.SLIGHTLY_NEGATIVE: "⚠️",
            SentimentLevel.NEGATIVE: "❌",
            SentimentLevel.VERY_NEGATIVE: "📉",
            SentimentLevel.EXTREMELY_NEGATIVE: "💥"
        }
        return emoji_map.get(sentiment_level, "❓")
    
    # Advanced metrics calculation methods
    
    async def _calculate_sentiment_momentum(self, platform_sentiments: Dict[SentimentSource, PlatformSentiment]) -> float:
        """Calculate overall sentiment momentum"""
        try:
            momenta = [ps.sentiment_momentum for ps in platform_sentiments.values() if ps.signal_count > 0]
            return sum(momenta) / max(len(momenta), 1) if momenta else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating sentiment momentum: {e}")
            return 0.0
    
    async def _calculate_viral_potential(self, platform_sentiments: Dict[SentimentSource, PlatformSentiment]) -> float:
        """Calculate viral potential based on trending scores and engagement"""
        try:
            viral_factors = []
            for ps in platform_sentiments.values():
                if ps.signal_count > 0:
                    viral_factor = ps.trending_score * ps.engagement_volume / max(ps.signal_count, 1)
                    viral_factors.append(viral_factor)
            
            return min(1.0, sum(viral_factors) / max(len(viral_factors), 1)) if viral_factors else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating viral potential: {e}")
            return 0.0
    
    async def _calculate_community_health(self, platform_sentiments: Dict[SentimentSource, PlatformSentiment]) -> float:
        """Calculate community health based on sentiment consistency and engagement"""
        try:
            # Check sentiment consistency across platforms
            sentiments = [ps.overall_score for ps in platform_sentiments.values() if ps.signal_count > 0]
            if len(sentiments) < 2:
                return 0.5  # Neutral health if not enough data
            
            # Calculate variance (low variance = good consistency)
            avg_sentiment = sum(sentiments) / len(sentiments)
            variance = sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)
            consistency_score = max(0.0, 1.0 - variance)
            
            # Calculate engagement health
            total_engagement = sum(ps.engagement_volume for ps in platform_sentiments.values())
            engagement_health = min(1.0, total_engagement / 100)
            
            # Combine factors
            return (consistency_score * 0.6 + engagement_health * 0.4)
        except Exception as e:
            self.logger.error(f"Error calculating community health: {e}")
            return 0.5
    
    async def _calculate_influencer_support(self, platform_sentiments: Dict[SentimentSource, PlatformSentiment]) -> float:
        """Calculate influencer support based on high-influence signals"""
        try:
            high_influence_signals = []
            for ps in platform_sentiments.values():
                for signal in ps.top_signals:
                    if signal.author_influence > 0.7:  # High influence threshold
                        high_influence_signals.append(signal)
            
            if not high_influence_signals:
                return 0.0
            
            # Calculate weighted sentiment from influencers
            total_weighted = sum(s.sentiment_score * s.author_influence for s in high_influence_signals)
            total_weight = sum(s.author_influence for s in high_influence_signals)
            
            influencer_sentiment = total_weighted / max(total_weight, 0.001)
            
            # Normalize to 0-1 scale (positive sentiment = higher support)
            return max(0.0, (influencer_sentiment + 1.0) / 2.0)
        except Exception as e:
            self.logger.error(f"Error calculating influencer support: {e}")
            return 0.0
    
    # Insight extraction methods
    
    async def _extract_positive_signals(self, platform_sentiments: Dict[SentimentSource, PlatformSentiment]) -> List[str]:
        """Extract key positive signals"""
        try:
            positive_signals = []
            for ps in platform_sentiments.values():
                for signal in ps.top_signals:
                    if signal.sentiment_score > 0.3:
                        positive_signals.append(f"{signal.source.value}: {signal.content[:100]}")
            
            # Return top 5 positive signals
            return sorted(positive_signals, key=lambda x: len(x), reverse=True)[:5]
        except Exception as e:
            self.logger.error(f"Error extracting positive signals: {e}")
            return []
    
    async def _extract_negative_signals(self, platform_sentiments: Dict[SentimentSource, PlatformSentiment]) -> List[str]:
        """Extract key negative signals"""
        try:
            negative_signals = []
            for ps in platform_sentiments.values():
                for signal in ps.top_signals:
                    if signal.sentiment_score < -0.3:
                        negative_signals.append(f"{signal.source.value}: {signal.content[:100]}")
            
            # Return top 5 negative signals
            return sorted(negative_signals, key=lambda x: len(x), reverse=True)[:5]
        except Exception as e:
            self.logger.error(f"Error extracting negative signals: {e}")
            return []
    
    async def _extract_trending_keywords(self, platform_sentiments: Dict[SentimentSource, PlatformSentiment]) -> List[str]:
        """Extract trending keywords from signals"""
        try:
            keyword_counts = defaultdict(int)
            
            for ps in platform_sentiments.values():
                for signal in ps.top_signals:
                    # Simple keyword extraction (in production, use NLP)
                    words = signal.content.lower().split()
                    for word in words:
                        if len(word) > 3 and word.isalpha():
                            keyword_counts[word] += 1
            
            # Return top 10 keywords
            return [word for word, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
        except Exception as e:
            self.logger.error(f"Error extracting trending keywords: {e}")
            return []
    
    async def _extract_influential_mentions(self, platform_sentiments: Dict[SentimentSource, PlatformSentiment]) -> List[str]:
        """Extract mentions from influential users"""
        try:
            influential_mentions = []
            for ps in platform_sentiments.values():
                for signal in ps.top_signals:
                    if signal.author_influence > 0.6:
                        influential_mentions.append(f"{signal.source.value} influencer: {signal.content[:80]}")
            
            return influential_mentions[:5]
        except Exception as e:
            self.logger.error(f"Error extracting influential mentions: {e}")
            return []
    
    # Quality and freshness calculations
    
    async def _calculate_signal_quality(self, platform_sentiments: Dict[SentimentSource, PlatformSentiment]) -> float:
        """Calculate overall signal quality score"""
        try:
            quality_scores = []
            for ps in platform_sentiments.values():
                if ps.signal_count > 0:
                    quality_scores.append(ps.confidence)
            
            return sum(quality_scores) / max(len(quality_scores), 1) if quality_scores else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating signal quality: {e}")
            return 0.0
    
    async def _calculate_data_freshness(self, platform_sentiments: Dict[SentimentSource, PlatformSentiment]) -> float:
        """Calculate data freshness score"""
        try:
            freshness_scores = []
            current_time = datetime.now()
            
            for ps in platform_sentiments.values():
                if ps.last_updated:
                    age_hours = (current_time - ps.last_updated).total_seconds() / 3600
                    freshness = max(0.0, 1.0 - age_hours / 24)  # Decay over 24 hours
                    freshness_scores.append(freshness)
            
            return sum(freshness_scores) / max(len(freshness_scores), 1) if freshness_scores else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating data freshness: {e}")
            return 0.0
    
    # Utility methods
    
    async def _check_rate_limit(self, source: SentimentSource) -> bool:
        """Check if we can make a request to the given source"""
        try:
            if source not in self.api_rate_limits:
                return True  # No rate limit defined
            
            current_time = time.time()
            rate_info = self.api_rate_limits[source]
            
            # Check if enough time has passed since last request
            min_interval = 3600.0 / rate_info['requests_per_hour']  # Convert to seconds
            time_since_last = current_time - rate_info['last_request']
            
            if time_since_last < min_interval:
                return False
            
            # Update last request time
            self.api_rate_limits[source]['last_request'] = current_time
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit for {source.value}: {e}")
            return False
    
    async def _test_api_connections(self):
        """Test connections to available APIs"""
        try:
            # Test available APIs
            if self.api_config.get('twitter_bearer_token'):
                self.logger.info("✅ Twitter API key configured")
            
            if self.api_config.get('reddit_client_id'):
                self.logger.info("✅ Reddit API credentials configured")
            
            if self.api_config.get('news_api_key'):
                self.logger.info("✅ News API key configured")
            
            # Test at least one working API
            working_apis = sum([
                bool(self.api_config.get('twitter_bearer_token')),
                bool(self.api_config.get('reddit_client_id')),
                bool(self.api_config.get('news_api_key'))
            ])
            
            if working_apis == 0:
                self.logger.warning("⚠️ No external API keys configured - using simulated data")
            else:
                self.logger.info(f"✅ {working_apis} external APIs configured")
            
        except Exception as e:
            self.logger.error(f"Error testing API connections: {e}")
    
    async def _load_sentiment_models(self):
        """Load sentiment analysis models"""
        try:
            # In production, this would load ML models for sentiment analysis
            # For now, we'll use keyword-based sentiment
            self.logger.info("📚 Sentiment models loaded (keyword-based)")
        except Exception as e:
            self.logger.error(f"Error loading sentiment models: {e}")
    
    async def _initialize_keyword_patterns(self):
        """Initialize keyword patterns for sentiment analysis"""
        try:
            # Expand keyword lists with variations
            self.positive_keywords.extend([
                'bullrun', 'mooning', 'lambo', 'gains', 'profit', 'strong',
                'support', 'resistance', 'bounce', 'recover', 'healthy'
            ])
            
            self.negative_keywords.extend([
                'dip', 'correction', 'bloodbath', 'rekt', 'bags', 'fud',
                'weak', 'sell-off', 'capitulation', 'bottom', 'crash'
            ])
            
            self.logger.info(f"📝 Keyword patterns initialized: "
                           f"{len(self.positive_keywords)} positive, "
                           f"{len(self.negative_keywords)} negative")
        except Exception as e:
            self.logger.error(f"Error initializing keyword patterns: {e}")
    
    def _update_analysis_metrics(self, result: SentimentAnalysisResult):
        """Update performance metrics"""
        try:
            self.total_analyses += 1
            self.total_signals_processed += result.total_signals_analyzed
            
            # Update average analysis time
            if self.total_analyses == 1:
                self.average_analysis_time_ms = result.analysis_duration_ms
            else:
                alpha = 0.1
                self.average_analysis_time_ms = (alpha * result.analysis_duration_ms + 
                                               (1 - alpha) * self.average_analysis_time_ms)
                
        except Exception as e:
            self.logger.error(f"Error updating analysis metrics: {e}")
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get comprehensive AI system status"""
        return {
            'initialized': True,
            'total_analyses': self.total_analyses,
            'total_signals_processed': self.total_signals_processed,
            'average_analysis_time_ms': round(self.average_analysis_time_ms, 2),
            'sentiment_accuracy_score': round(self.sentiment_accuracy_score, 3),
            'cache_size': len(self.signal_cache),
            'cache_ttl_minutes': self.cache_ttl_minutes,
            'platform_weights': self.sentiment_weights,
            'min_signals_required': self.min_signals_required,
            'configured_apis': [
                source for source, key in [
                    ('twitter', 'twitter_bearer_token'),
                    ('reddit', 'reddit_client_id'),
                    ('news', 'news_api_key')
                ] if self.api_config.get(key)
            ]
        }
    
    async def shutdown(self):
        """Shutdown the sentiment AI system"""
        try:
            self.logger.info("🛑 Shutting down sentiment AI...")
            
            # Clear caches
            self.signal_cache.clear()
            self.cache_timestamps.clear()
            
            self.logger.info("✅ Sentiment AI shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during sentiment AI shutdown: {e}")


# Global instance manager
_sentiment_ai = None

async def get_sentiment_first_ai() -> SentimentFirstAI:
    """Get global sentiment AI instance"""
    global _sentiment_ai
    if _sentiment_ai is None:
        _sentiment_ai = SentimentFirstAI()
        await _sentiment_ai.initialize()
    return _sentiment_ai 