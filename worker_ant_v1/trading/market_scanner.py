"""
REAL MARKET SCANNER - LIVE MEMECOIN HUNTER
=========================================

Connects to real DEX APIs to find live memecoin opportunities
with sentiment-driven filtering and aggressive compounding targets.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
import numpy as np
import os

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.intelligence.sentiment_first_ai import get_sentiment_first_ai

@dataclass
class TokenData:
    address: str
    symbol: str
    name: str
    price: float
    volume_24h: float
    liquidity: float
    market_cap: float
    price_change_24h: float
    price_change_1h: float
    holder_count: int
    age_hours: float
    last_updated: datetime

@dataclass
class TradingOpportunity:
    token_address: str
    token_symbol: str
    sentiment_score: float
    liquidity_sol: float
    volume_24h_sol: float
    price_impact_percent: float
    risk_score: float
    confidence: float
    expected_profit: float
    priority: int
    memecoin_pattern: str
    timestamp: datetime

class RealMarketScanner:
    """Real market scanner with live DEX integration for memecoin hunting"""
    
    def __init__(self):
        self.logger = get_logger("MarketScanner")
        
        # API endpoints
        self.jupiter_api = "https://price.jup.ag/v4"
        self.birdeye_api = "https://public-api.birdeye.so"
        self.dexscreener_api = "https://api.dexscreener.com/latest/dex"
        self.helius_api = "https://api.helius.xyz/v0"
        
        # Data storage
        self.token_data: Dict[str, TokenData] = {}
        self.opportunities: List[TradingOpportunity] = []
        self.last_scan = datetime.now()
        
        # Memecoin-specific configuration
        self.min_liquidity = 5.0  # Lower threshold for memecoins
        self.min_volume = 50.0    # Lower volume threshold
        self.max_price_impact = 5.0  # Higher price impact tolerance
        self.max_age_hours = 168  # 1 week max age for memecoins
        self.min_holder_count = 50  # Minimum holders
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = time.time()
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        # Sentiment AI
        self.sentiment_ai = None
        
        # Performance tracking
        self.scan_count = 0
        self.opportunities_found = 0
        self.successful_trades = 0
        
        self.logger.info("🔍 Real Market Scanner initialized for memecoin hunting")
        
    async def initialize(self):
        """Initialize market scanner"""
        self.logger.info("🔍 Initializing real market scanner...")
        
        # Initialize sentiment AI
        self.sentiment_ai = await get_sentiment_first_ai()
        
        # Test API connections
        await self._test_api_connections()
        
        # Load initial market data
        await self.update_market_data()
        
        self.logger.info("✅ Market scanner initialized")
        
    async def _test_api_connections(self):
        """Test all API connections"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test Jupiter API
                async with session.get(f"{self.jupiter_api}/price?ids=SOL") as resp:
                    if resp.status == 200:
                        self.logger.info("✅ Jupiter API connected")
                    else:
                        self.logger.warning("⚠️ Jupiter API connection failed")
                        
                # Test Birdeye API
                async with session.get(f"{self.birdeye_api}/public/price?address=So11111111111111111111111111111111111111112") as resp:
                    if resp.status == 200:
                        self.logger.info("✅ Birdeye API connected")
                    else:
                        self.logger.warning("⚠️ Birdeye API connection failed")
                        
        except Exception as e:
            self.logger.error(f"❌ API connection test failed: {e}")
            
    async def update_market_data(self):
        """Update market data from all sources"""
        try:
            self.scan_count += 1
            
            # Get trending tokens
            trending_tokens = await self._get_trending_tokens()
            
            # Get detailed data for each token
            for token_address in trending_tokens:
                await self._get_token_data(token_address)
                await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
                
            # Generate opportunities with sentiment analysis
            await self._generate_opportunities()
            
            self.last_scan = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Market data update error: {e}")
            
    async def _get_trending_tokens(self) -> List[str]:
        """Get trending tokens from multiple sources"""
        trending_tokens = set()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get from Birdeye trending
                try:
                    async with session.get(f"{self.birdeye_api}/public/trending") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if 'data' in data:
                                for token in data['data'][:30]:  # Top 30
                                    if 'address' in token:
                                        trending_tokens.add(token['address'])
                except Exception as e:
                    self.logger.warning(f"Birdeye trending fetch failed: {e}")
                    
                # Get from DexScreener trending
                try:
                    async with session.get(f"{self.dexscreener_api}/tokens/trending") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if 'pairs' in data:
                                for pair in data['pairs'][:30]:  # Top 30
                                    if 'tokenAddress' in pair:
                                        trending_tokens.add(pair['tokenAddress'])
                except Exception as e:
                    self.logger.warning(f"DexScreener trending fetch failed: {e}")
                    
                # Get from Jupiter trending
                try:
                    async with session.get(f"{self.jupiter_api}/trending") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if 'data' in data:
                                for token in data['data'][:20]:  # Top 20
                                    if 'address' in token:
                                        trending_tokens.add(token['address'])
                except Exception as e:
                    self.logger.warning(f"Jupiter trending fetch failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Trending tokens fetch error: {e}")
            
        return list(trending_tokens)
        
    async def _get_token_data(self, token_address: str):
        """Get detailed token data"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get price from Jupiter
                price_data = await self._get_jupiter_price(session, token_address)
                
                # Get additional data from Birdeye
                birdeye_data = await self._get_birdeye_data(session, token_address)
                
                # Get token metadata from Helius (if available)
                metadata = await self._get_token_metadata(session, token_address)
                
                # Combine data
                if price_data and birdeye_data:
                    token_data = TokenData(
                        address=token_address,
                        symbol=birdeye_data.get('symbol', metadata.get('symbol', 'UNKNOWN')),
                        name=birdeye_data.get('name', metadata.get('name', 'Unknown Token')),
                        price=price_data.get('price', 0),
                        volume_24h=birdeye_data.get('volume24h', 0),
                        liquidity=birdeye_data.get('liquidity', 0),
                        market_cap=birdeye_data.get('marketCap', 0),
                        price_change_24h=birdeye_data.get('priceChange24h', 0),
                        price_change_1h=birdeye_data.get('priceChange1h', 0),
                        holder_count=birdeye_data.get('holderCount', 0),
                        age_hours=birdeye_data.get('ageHours', 0),
                        last_updated=datetime.now()
                    )
                    
                    self.token_data[token_address] = token_data
                    
        except Exception as e:
            self.logger.error(f"Token data fetch error for {token_address}: {e}")
            
    async def _get_jupiter_price(self, session: aiohttp.ClientSession, token_address: str) -> Optional[Dict]:
        """Get price from Jupiter API"""
        try:
            async with session.get(f"{self.jupiter_api}/price?ids={token_address}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'data' in data and token_address in data['data']:
                        return data['data'][token_address]
                return None
        except Exception as e:
            self.logger.error(f"Jupiter price fetch error: {e}")
            return None
            
    async def _get_birdeye_data(self, session: aiohttp.ClientSession, token_address: str) -> Optional[Dict]:
        """Get token data from Birdeye API"""
        try:
            # Get token info
            async with session.get(f"{self.birdeye_api}/public/token/{token_address}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'data' in data:
                        token_info = data['data']
                        
                        # Get additional metrics
                        metrics_data = await self._get_birdeye_metrics(session, token_address)
                        
                        return {
                            'symbol': token_info.get('symbol', 'UNKNOWN'),
                            'name': token_info.get('name', 'Unknown Token'),
                            'volume24h': token_info.get('volume24h', 0),
                            'liquidity': token_info.get('liquidity', 0),
                            'marketCap': token_info.get('marketCap', 0),
                            'priceChange24h': token_info.get('priceChange24h', 0),
                            'priceChange1h': token_info.get('priceChange1h', 0),
                            'holderCount': metrics_data.get('holderCount', 0),
                            'ageHours': metrics_data.get('ageHours', 0)
                        }
                return None
        except Exception as e:
            self.logger.error(f"Birdeye data fetch error: {e}")
            return None
            
    async def _get_birdeye_metrics(self, session: aiohttp.ClientSession, token_address: str) -> Dict:
        """Get additional metrics from Birdeye"""
        try:
            async with session.get(f"{self.birdeye_api}/public/token/{token_address}/metrics") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'data' in data:
                        return data['data']
                return {}
        except Exception as e:
            self.logger.error(f"Birdeye metrics fetch error: {e}")
            return {}
            
    async def _get_token_metadata(self, session: aiohttp.ClientSession, token_address: str) -> Dict:
        """Get token metadata from Helius"""
        try:
            helius_key = os.getenv('HELIUS_API_KEY')
            if not helius_key:
                return {}
                
            url = f"{self.helius_api}/token-metadata?api-key={helius_key}"
            payload = {"mintAccounts": [token_address]}
            
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and len(data) > 0:
                        return data[0].get('onChainMetadata', {}).get('metadata', {})
                return {}
        except Exception as e:
            self.logger.error(f"Helius metadata fetch error: {e}")
            return {}
            
    async def _generate_opportunities(self):
        """Generate trading opportunities with sentiment analysis"""
        try:
            self.opportunities.clear()
            
            for token_address, token_data in self.token_data.items():
                # Apply memecoin filters
                if not self._passes_memecoin_filters(token_data):
                    continue
                
                # Get sentiment analysis
                market_data = {
                    'symbol': token_data.symbol,
                    'price': token_data.price,
                    'volume': token_data.volume_24h,
                    'liquidity': token_data.liquidity,
                    'price_change_24h': token_data.price_change_24h,
                    'price_change_1h': token_data.price_change_1h,
                    'holder_count': token_data.holder_count,
                    'age_hours': token_data.age_hours
                }
                
                sentiment_decision = await self.sentiment_ai.analyze_and_decide(
                    token_address, market_data
                )
                
                # Only consider BUY opportunities
                if sentiment_decision.decision == "BUY":
                    # Calculate additional metrics
                    risk_score = self._calculate_risk_score(token_data)
                    price_impact = self._calculate_price_impact(token_data)
                    
                    opportunity = TradingOpportunity(
                        token_address=token_address,
                        token_symbol=token_data.symbol,
                        sentiment_score=sentiment_decision.sentiment_score,
                        liquidity_sol=token_data.liquidity,
                        volume_24h_sol=token_data.volume_24h,
                        price_impact_percent=price_impact,
                        risk_score=risk_score,
                        confidence=sentiment_decision.confidence,
                        expected_profit=sentiment_decision.expected_profit,
                        priority=sentiment_decision.priority,
                        memecoin_pattern=sentiment_decision.metadata.get('pattern', 'unknown'),
                        timestamp=datetime.now()
                    )
                    
                    self.opportunities.append(opportunity)
            
            # Sort by priority and expected profit
            self.opportunities.sort(key=lambda x: (x.priority, x.expected_profit), reverse=True)
            
            self.opportunities_found = len(self.opportunities)
            self.logger.info(f"🎯 Found {self.opportunities_found} memecoin opportunities")
            
        except Exception as e:
            self.logger.error(f"Error generating opportunities: {e}")
            
    def _passes_memecoin_filters(self, token_data: TokenData) -> bool:
        """Check if token passes memecoin-specific filters"""
        try:
            # Liquidity filter
            if token_data.liquidity < self.min_liquidity:
                return False
                
            # Volume filter
            if token_data.volume_24h < self.min_volume:
                return False
                
            # Age filter (not too old)
            if token_data.age_hours > self.max_age_hours:
                return False
                
            # Holder count filter
            if token_data.holder_count < self.min_holder_count:
                return False
                
            # Price impact filter
            price_impact = self._calculate_price_impact(token_data)
            if price_impact > self.max_price_impact:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in memecoin filters: {e}")
            return False
            
    def _calculate_risk_score(self, token_data: TokenData) -> float:
        """Calculate risk score for memecoin"""
        try:
            risk_score = 0.0
            
            # Low liquidity = higher risk
            if token_data.liquidity < 10:
                risk_score += 0.3
            elif token_data.liquidity < 20:
                risk_score += 0.2
                
            # Low volume = higher risk
            if token_data.volume_24h < 100:
                risk_score += 0.2
            elif token_data.volume_24h < 500:
                risk_score += 0.1
                
            # Few holders = higher risk
            if token_data.holder_count < 100:
                risk_score += 0.3
            elif token_data.holder_count < 500:
                risk_score += 0.2
                
            # Very new token = higher risk
            if token_data.age_hours < 24:
                risk_score += 0.2
                
            # High price volatility = higher risk
            if abs(token_data.price_change_1h) > 0.5:  # 50% hourly change
                risk_score += 0.2
                
            return min(risk_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5
            
    def _calculate_price_impact(self, token_data: TokenData) -> float:
        """Calculate estimated price impact for a 1 SOL trade"""
        try:
            # Simple estimation based on liquidity
            if token_data.liquidity > 0:
                # Rough estimation: price impact = trade_size / liquidity * 100
                trade_size_sol = 1.0  # 1 SOL trade
                price_impact = (trade_size_sol / token_data.liquidity) * 100
                return min(price_impact, 10.0)  # Cap at 10%
            return 5.0  # Default high impact if no liquidity data
            
        except Exception as e:
            self.logger.error(f"Error calculating price impact: {e}")
            return 5.0
            
    async def scan_markets(self) -> List[Dict]:
        """Scan markets for memecoin opportunities"""
        try:
            # Update market data
            await self.update_market_data()
            
            # Return top opportunities
            top_opportunities = self.opportunities[:10]  # Top 10
            
            return [
                {
                    'token_address': opp.token_address,
                    'token_symbol': opp.token_symbol,
                    'sentiment_score': opp.sentiment_score,
                    'liquidity_sol': opp.liquidity_sol,
                    'volume_24h_sol': opp.volume_24h_sol,
                    'price_impact_percent': opp.price_impact_percent,
                    'risk_score': opp.risk_score,
                    'confidence': opp.confidence,
                    'expected_profit': opp.expected_profit,
                    'priority': opp.priority,
                    'memecoin_pattern': opp.memecoin_pattern,
                    'timestamp': opp.timestamp.isoformat()
                }
                for opp in top_opportunities
            ]
            
        except Exception as e:
            self.logger.error(f"Market scan error: {e}")
            return []
            
    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get current token price"""
        try:
            if token_address in self.token_data:
                return self.token_data[token_address].price
                
            # Fetch fresh price
            async with aiohttp.ClientSession() as session:
                price_data = await self._get_jupiter_price(session, token_address)
                if price_data:
                    return price_data.get('price', 0)
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting token price: {e}")
            return None
            
    def get_system_status(self) -> Dict:
        """Get market scanner status"""
        try:
            return {
                'initialized': True,
                'scan_count': self.scan_count,
                'opportunities_found': self.opportunities_found,
                'successful_trades': self.successful_trades,
                'last_scan': self.last_scan.isoformat(),
                'total_tokens_tracked': len(self.token_data),
                'current_opportunities': len(self.opportunities),
                'api_status': {
                    'jupiter': 'connected',
                    'birdeye': 'connected',
                    'dexscreener': 'connected'
                },
                'filters': {
                    'min_liquidity_sol': self.min_liquidity,
                    'min_volume_sol': self.min_volume,
                    'max_price_impact_percent': self.max_price_impact,
                    'max_age_hours': self.max_age_hours,
                    'min_holder_count': self.min_holder_count
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}

class ScanResult:
    """Result of market scan"""
    
    def __init__(self, success: bool = True, opportunities: list = None, errors: list = None):
        self.success = success
        self.opportunities = opportunities or []
        self.errors = errors or []
        self.timestamp = datetime.now()
    
    def __str__(self):
        return f"ScanResult(success={self.success}, opportunities={len(self.opportunities)}, errors={len(self.errors)})"

# Global instance
_market_scanner = None

async def get_market_scanner() -> RealMarketScanner:
    """Get global market scanner instance"""
    global _market_scanner
    if _market_scanner is None:
        _market_scanner = RealMarketScanner()
        await _market_scanner.initialize()
    return _market_scanner

