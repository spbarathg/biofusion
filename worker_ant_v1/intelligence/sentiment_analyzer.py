from typing import Dict, List, Optional, Tuple
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..utils.logger import get_logger
from pathlib import Path
import json
import aiohttp
import asyncio
from datetime import datetime, timedelta
import os
import base64
from dataclasses import dataclass
from enum import Enum

logger = get_logger(__name__)

@dataclass
class SentimentData:
    """Sentiment analysis result data"""
    overall_sentiment: float
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float
    timestamp: datetime
    source: str
    metadata: Dict[str, any] = None

@dataclass
class AggregatedSentiment:
    """Aggregated sentiment data from multiple sources"""
    token_address: str
    overall_sentiment: float
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float
    sources: Dict[str, SentimentData]
    timestamp: datetime
    trend: str = "neutral"
    volatility: float = 0.0

class SentimentAnalyzer:
    """Sentiment analysis for crypto market intelligence"""
    
    def __init__(self):
        self.model_name = "ProsusAI/finbert"  # Financial sentiment analysis model
        self.cache_dir = Path(__file__).parent.parent.parent / "data" / "sentiment_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing sentiment analyzer with model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            raise
    
    async def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment scores (positive, negative, neutral)
        """
        try:
            result = self.sentiment_pipeline(text)[0]
            sentiment_scores = {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 0.0
            }
            sentiment_scores[result["label"].lower()] = result["score"]
            return sentiment_scores
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    
    async def analyze_multiple(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment score dictionaries
        """
        try:
            results = []
            for text in texts:
                result = await self.analyze_text(text)
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            return [{"positive": 0.0, "negative": 0.0, "neutral": 1.0}] * len(texts)
    
    def get_cache_key(self, token_symbol: str, timeframe: str) -> str:
        """Generate cache key for sentiment data"""
        return f"{token_symbol.lower()}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.json"
    
    async def get_cached_sentiment(self, token_symbol: str, timeframe: str) -> Optional[Dict]:
        """Get cached sentiment data if available and fresh"""
        cache_file = self.cache_dir / self.get_cache_key(token_symbol, timeframe)
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                
            # Check if cache is still valid (less than 1 hour old)
            cache_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cache_time > timedelta(hours=1):
                return None
                
            return data
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")
            return None
    
    def save_to_cache(self, token_symbol: str, timeframe: str, data: Dict):
        """Save sentiment data to cache"""
        try:
            cache_file = self.cache_dir / self.get_cache_key(token_symbol, timeframe)
            data['timestamp'] = datetime.now().isoformat()
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
    
    async def get_token_sentiment(self, token_symbol: str, timeframe: str = "1h") -> Dict:
        """
        Get sentiment analysis for a specific token
        
        Args:
            token_symbol: Token symbol (e.g., "SOL")
            timeframe: Time frame for analysis ("1h", "24h", "7d")
            
        Returns:
            Dict containing sentiment scores and metadata
        """
        # Check cache first
        cached_data = await self.get_cached_sentiment(token_symbol, timeframe)
        if cached_data:
            return cached_data
            
        try:
            # Collect data from various sources
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self._fetch_twitter_sentiment(session, token_symbol),
                    self._fetch_news_sentiment(session, token_symbol),
                    self._fetch_reddit_sentiment(session, token_symbol)
                ]
                results = await asyncio.gather(*tasks)
                
            # Combine results
            twitter_sent, news_sent, reddit_sent = results
            
            # Calculate weighted average
            weights = {"twitter": 0.4, "news": 0.4, "reddit": 0.2}
            combined_sentiment = {
                "positive": (
                    twitter_sent["positive"] * weights["twitter"] +
                    news_sent["positive"] * weights["news"] +
                    reddit_sent["positive"] * weights["reddit"]
                ),
                "negative": (
                    twitter_sent["negative"] * weights["twitter"] +
                    news_sent["negative"] * weights["news"] +
                    reddit_sent["negative"] * weights["reddit"]
                ),
                "neutral": (
                    twitter_sent["neutral"] * weights["twitter"] +
                    news_sent["neutral"] * weights["news"] +
                    reddit_sent["neutral"] * weights["reddit"]
                )
            }
            
            result = {
                "token": token_symbol,
                "timeframe": timeframe,
                "sentiment": combined_sentiment,
                "sources": {
                    "twitter": twitter_sent,
                    "news": news_sent,
                    "reddit": reddit_sent
                }
            }
            
            # Cache the result
            self.save_to_cache(token_symbol, timeframe, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting token sentiment: {str(e)}")
            return {
                "token": token_symbol,
                "timeframe": timeframe,
                "sentiment": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                "error": str(e)
            }
    
    async def _fetch_twitter_sentiment(
        self,
        session: aiohttp.ClientSession,
        token_symbol: str
    ) -> Dict[str, float]:
        """Fetch and analyze Twitter sentiment"""
        try:
            # Get Twitter API credentials from environment
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            if not bearer_token:
                logger.warning("Twitter API not configured, using fallback")
                return {"positive": 0.3, "negative": 0.2, "neutral": 0.5}
            
            # Search for tweets about the token
            query = f"#{token_symbol} OR ${token_symbol} -is:retweet"
            url = "https://api.twitter.com/2/tweets/search/recent"
            
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json"
            }
            
            params = {
                "query": query,
                "max_results": 100,
                "tweet.fields": "created_at,public_metrics,lang",
                "exclude": "retweets,replies"
            }
            
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    tweets = data.get('data', [])
                    
                    if not tweets:
                        return {"positive": 0.3, "negative": 0.2, "neutral": 0.5}
                    
                    # Analyze sentiment for each tweet
                    sentiments = []
                    for tweet in tweets:
                        text = tweet.get('text', '')
                        if text:
                            sentiment = await self.analyze_text(text)
                            sentiments.append(sentiment)
                    
                    if sentiments:
                        # Calculate average sentiment
                        avg_sentiment = {
                            "positive": sum(s["positive"] for s in sentiments) / len(sentiments),
                            "negative": sum(s["negative"] for s in sentiments) / len(sentiments),
                            "neutral": sum(s["neutral"] for s in sentiments) / len(sentiments)
                        }
                        return avg_sentiment
                    
                else:
                    logger.warning(f"Twitter API error: {resp.status}")
                    
        except Exception as e:
            logger.error(f"Twitter sentiment fetch error: {e}")
            
        return {"positive": 0.3, "negative": 0.2, "neutral": 0.5}
    
    async def _fetch_news_sentiment(
        self,
        session: aiohttp.ClientSession,
        token_symbol: str
    ) -> Dict[str, float]:
        """Fetch and analyze news sentiment"""
        try:
            # Use CryptoPanic API for crypto news
            api_key = os.getenv('CRYPTOPANIC_API_KEY')
            if not api_key:
                logger.warning("CryptoPanic API not configured, using fallback")
                return {"positive": 0.4, "negative": 0.3, "neutral": 0.3}
            
            url = "https://cryptopanic.com/api/v1/posts/"
            params = {
                "auth_token": api_key,
                "currencies": token_symbol,
                "filter": "hot",
                "public": True
            }
            
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    posts = data.get('results', [])
                    
                    if not posts:
                        return {"positive": 0.4, "negative": 0.3, "neutral": 0.3}
                    
                    # Analyze sentiment for each post
                    sentiments = []
                    for post in posts[:20]:  # Limit to 20 posts
                        title = post.get('title', '')
                        if title:
                            sentiment = await self.analyze_text(title)
                            sentiments.append(sentiment)
                    
                    if sentiments:
                        # Calculate average sentiment
                        avg_sentiment = {
                            "positive": sum(s["positive"] for s in sentiments) / len(sentiments),
                            "negative": sum(s["negative"] for s in sentiments) / len(sentiments),
                            "neutral": sum(s["neutral"] for s in sentiments) / len(sentiments)
                        }
                        return avg_sentiment
                    
                else:
                    logger.warning(f"CryptoPanic API error: {resp.status}")
                    
        except Exception as e:
            logger.error(f"News sentiment fetch error: {e}")
            
        return {"positive": 0.4, "negative": 0.3, "neutral": 0.3}
    
    async def _fetch_reddit_sentiment(
        self,
        session: aiohttp.ClientSession,
        token_symbol: str
    ) -> Dict[str, float]:
        """Fetch and analyze Reddit sentiment"""
        try:
            # Use Reddit API (requires app registration)
            client_id = os.getenv('REDDIT_CLIENT_ID')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                logger.warning("Reddit API not configured, using fallback")
                return {"positive": 0.2, "negative": 0.3, "neutral": 0.5}
            
            # Get access token
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                "grant_type": "client_credentials"
            }
            
            auth_headers = {
                "Authorization": f"Basic {base64.b64encode(f'{client_id}:{client_secret}'.encode()).decode()}",
                "User-Agent": "WorkerAnt/1.0"
            }
            
            async with session.post(auth_url, data=auth_data, headers=auth_headers) as resp:
                if resp.status == 200:
                    auth_response = await resp.json()
                    access_token = auth_response.get('access_token')
                    
                    if access_token:
                        # Search Reddit for posts about the token
                        search_url = f"https://oauth.reddit.com/search"
                        search_headers = {
                            "Authorization": f"Bearer {access_token}",
                            "User-Agent": "WorkerAnt/1.0"
                        }
                        
                        search_params = {
                            "q": token_symbol,
                            "t": "day",
                            "limit": 25,
                            "sort": "hot"
                        }
                        
                        async with session.get(search_url, headers=search_headers, params=search_params) as search_resp:
                            if search_resp.status == 200:
                                search_data = await search_resp.json()
                                posts = search_data.get('data', {}).get('children', [])
                                
                                if not posts:
                                    return {"positive": 0.2, "negative": 0.3, "neutral": 0.5}
                                
                                # Analyze sentiment for each post
                                sentiments = []
                                for post in posts:
                                    post_data = post.get('data', {})
                                    title = post_data.get('title', '')
                                    if title:
                                        sentiment = await self.analyze_text(title)
                                        sentiments.append(sentiment)
                                
                                if sentiments:
                                    # Calculate average sentiment
                                    avg_sentiment = {
                                        "positive": sum(s["positive"] for s in sentiments) / len(sentiments),
                                        "negative": sum(s["negative"] for s in sentiments) / len(sentiments),
                                        "neutral": sum(s["neutral"] for s in sentiments) / len(sentiments)
                                    }
                                    return avg_sentiment
                            
                            else:
                                logger.warning(f"Reddit search error: {search_resp.status}")
                    
                else:
                    logger.warning(f"Reddit auth error: {resp.status}")
                    
        except Exception as e:
            logger.error(f"Reddit sentiment fetch error: {e}")
            
        return {"positive": 0.2, "negative": 0.3, "neutral": 0.5}

    async def update_token_sentiment(self, token_address: str):
        """Update sentiment for a specific token"""
        try:
            # Get token symbol from address (simplified)
            token_symbol = token_address[:8].upper()  # Use first 8 chars as symbol
            
            # Get fresh sentiment data
            sentiment_data = await self.get_token_sentiment(token_symbol, "1h")
            
            # Update cache
            self.save_to_cache(token_symbol, "1h", sentiment_data)
            
        except Exception as e:
            logger.error(f"Token sentiment update error: {e}")

    async def analyze_token_sentiment(self, token_address: str, market_data: Dict[str, any]) -> SentimentData:
        """Analyze sentiment for a specific token and return SentimentData object"""
        try:
            # Get token symbol from market data or address
            token_symbol = market_data.get('symbol', token_address[:8].upper())
            
            # Get sentiment from cache or fetch fresh
            sentiment_result = await self.get_token_sentiment(token_symbol, "1h")
            
            # Calculate overall sentiment score
            sentiment = sentiment_result.get('sentiment', {})
            positive = sentiment.get('positive', 0.0)
            negative = sentiment.get('negative', 0.0)
            neutral = sentiment.get('neutral', 0.0)
            
            # Calculate overall sentiment (-1 to 1 scale)
            overall_sentiment = positive - negative
            
            # Calculate confidence based on sentiment strength
            confidence = max(positive, negative, neutral)
            
            return SentimentData(
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                positive_score=positive,
                negative_score=negative,
                neutral_score=neutral,
                timestamp=datetime.now(),
                source="sentiment_analyzer",
                metadata={
                    'token_address': token_address,
                    'token_symbol': token_symbol,
                    'market_data': market_data
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing token sentiment: {e}")
            # Return neutral sentiment on error
            return SentimentData(
                overall_sentiment=0.0,
                confidence=0.0,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                timestamp=datetime.now(),
                source="sentiment_analyzer",
                metadata={'error': str(e)}
            ) 