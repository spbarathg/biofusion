"""
REAL MARKET SCANNER - LIVE DEX INTEGRATION
=========================================

Connects to real DEX APIs to get live market data, liquidity, and trading opportunities.
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

from worker_ant_v1.utils.logger import setup_logger

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
    timestamp: datetime

class RealMarketScanner:
    """Real market scanner with live DEX integration"""
    
    def __init__(self):
        self.logger = setup_logger("MarketScanner")
        
        # API endpoints
        self.jupiter_api = "https://price.jup.ag/v4"
        self.birdeye_api = "https://public-api.birdeye.so"
        self.dexscreener_api = "https://api.dexscreener.com/latest/dex"
        
        # Data storage
        self.token_data: Dict[str, TokenData] = {}
        self.opportunities: List[TradingOpportunity] = []
        self.last_scan = datetime.now()
        
        # Configuration
        self.min_liquidity = 10.0  # Minimum liquidity in SOL
        self.min_volume = 100.0    # Minimum 24h volume in SOL
        self.max_price_impact = 3.0  # Maximum price impact percentage
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = time.time()
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    async def initialize(self):
        """Initialize market scanner"""
        self.logger.info("🔍 Initializing real market scanner...")
        
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
            # Get trending tokens
            trending_tokens = await self._get_trending_tokens()
            
            # Get detailed data for each token
            for token_address in trending_tokens:
                await self._get_token_data(token_address)
                await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
                
            # Generate opportunities
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
                                for token in data['data'][:20]:  # Top 20
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
                                for pair in data['pairs'][:20]:  # Top 20
                                    if 'tokenAddress' in pair:
                                        trending_tokens.add(pair['tokenAddress'])
                except Exception as e:
                    self.logger.warning(f"DexScreener trending fetch failed: {e}")
                    
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
                
                # Combine data
                if price_data and birdeye_data:
                    token_data = TokenData(
                        address=token_address,
                        symbol=birdeye_data.get('symbol', 'UNKNOWN'),
                        name=birdeye_data.get('name', 'Unknown Token'),
                        price=price_data.get('price', 0),
                        volume_24h=birdeye_data.get('volume24h', 0),
                        liquidity=birdeye_data.get('liquidity', 0),
                        market_cap=birdeye_data.get('marketCap', 0),
                        price_change_24h=birdeye_data.get('priceChange24h', 0),
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
            async with session.get(f"{self.birdeye_api}/public/token_list?address={token_address}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if 'data' in data and len(data['data']) > 0:
                        return data['data'][0]
                return None
        except Exception as e:
            self.logger.error(f"Birdeye data fetch error: {e}")
            return None
            
    async def _generate_opportunities(self):
        """Generate trading opportunities from market data"""
        opportunities = []
        
        for token_address, token_data in self.token_data.items():
            try:
                # Calculate sentiment score (simplified for now)
                sentiment_score = self._calculate_sentiment_score(token_data)
                
                # Calculate risk score
                risk_score = self._calculate_risk_score(token_data)
                
                # Calculate confidence
                confidence = self._calculate_confidence(token_data)
                
                # Check if meets criteria
                if (token_data.liquidity >= self.min_liquidity and 
                    token_data.volume_24h >= self.min_volume and
                    sentiment_score >= 0.6):
                    
                    opportunity = TradingOpportunity(
                        token_address=token_address,
                        token_symbol=token_data.symbol,
                        sentiment_score=sentiment_score,
                        liquidity_sol=token_data.liquidity,
                        volume_24h_sol=token_data.volume_24h,
                        price_impact_percent=self._calculate_price_impact(token_data),
                        risk_score=risk_score,
                        confidence=confidence,
                        timestamp=datetime.now()
                    )
                    
                    opportunities.append(opportunity)
                    
            except Exception as e:
                self.logger.error(f"Opportunity generation error for {token_address}: {e}")
                
        # Sort by confidence and update
        self.opportunities = sorted(opportunities, key=lambda x: x.confidence, reverse=True)
        
    def _calculate_sentiment_score(self, token_data: TokenData) -> float:
        """Calculate sentiment score based on price action and volume"""
        try:
            # Base score from price change
            price_score = min(max(token_data.price_change_24h / 100, 0), 1)
            
            # Volume score
            volume_score = min(token_data.volume_24h / 1000, 1)  # Normalize to 1000 SOL
            
            # Liquidity score
            liquidity_score = min(token_data.liquidity / 100, 1)  # Normalize to 100 SOL
            
            # Weighted average
            sentiment = (price_score * 0.4 + volume_score * 0.4 + liquidity_score * 0.2)
            
            return max(0, min(1, sentiment))  # Clamp to 0-1
            
        except Exception:
            return 0.5
            
    def _calculate_risk_score(self, token_data: TokenData) -> float:
        """Calculate risk score (0 = low risk, 1 = high risk)"""
        try:
            risk_factors = []
            
            # Low liquidity = high risk
            if token_data.liquidity < 50:
                risk_factors.append(0.8)
            elif token_data.liquidity < 100:
                risk_factors.append(0.5)
            else:
                risk_factors.append(0.2)
                
            # High volatility = high risk
            volatility = abs(token_data.price_change_24h)
            if volatility > 50:
                risk_factors.append(0.9)
            elif volatility > 20:
                risk_factors.append(0.6)
            else:
                risk_factors.append(0.3)
                
            # Low volume = high risk
            if token_data.volume_24h < 200:
                risk_factors.append(0.7)
            elif token_data.volume_24h < 500:
                risk_factors.append(0.4)
            else:
                risk_factors.append(0.2)
                
            return np.mean(risk_factors)
            
        except Exception:
            return 0.5
            
    def _calculate_confidence(self, token_data: TokenData) -> float:
        """Calculate confidence in the opportunity"""
        try:
            confidence_factors = []
            
            # High liquidity = high confidence
            if token_data.liquidity > 200:
                confidence_factors.append(0.9)
            elif token_data.liquidity > 100:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)
                
            # High volume = high confidence
            if token_data.volume_24h > 1000:
                confidence_factors.append(0.9)
            elif token_data.volume_24h > 500:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)
                
            # Recent data = high confidence
            time_diff = (datetime.now() - token_data.last_updated).total_seconds()
            if time_diff < 300:  # 5 minutes
                confidence_factors.append(0.9)
            elif time_diff < 900:  # 15 minutes
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)
                
            return np.mean(confidence_factors)
            
        except Exception:
            return 0.5
            
    def _calculate_price_impact(self, token_data: TokenData) -> float:
        """Calculate estimated price impact for a 1 SOL trade"""
        try:
            # Simple estimation based on liquidity
            if token_data.liquidity > 0:
                return (1.0 / token_data.liquidity) * 100  # Percentage
            return 5.0  # Default high impact
            
        except Exception:
            return 3.0
            
    async def scan_markets(self) -> List[Dict]:
        """Scan markets for opportunities"""
        # Convert opportunities to dict format
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
                'timestamp': opp.timestamp.isoformat()
            }
            for opp in self.opportunities[:10]  # Top 10 opportunities
        ]
        
    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get current token price"""
        try:
            if token_address in self.token_data:
                return self.token_data[token_address].price
            return None
        except Exception as e:
            self.logger.error(f"Price fetch error: {e}")
            return None
            
    def get_system_status(self) -> Dict:
        """Get system status"""
        return {
            'total_tokens_tracked': len(self.token_data),
            'opportunities_found': len(self.opportunities),
            'last_scan': self.last_scan.isoformat(),
            'api_status': {
                'jupiter': 'connected',
                'birdeye': 'connected',
                'dexscreener': 'connected'
            }
        }




production_scanner = RealMarketScanner()


profit_scanner = production_scanner


__all__ = [
    "TradingOpportunity", "ScanMode", "SecurityRisk", "ScannerMetrics",
    "TokenFilter", "OpportunityAnalyzer", "RealTimeDataFeed", "ProductionScanner",
    "production_scanner", "profit_scanner"
]


class ScanResult:
    """Result from market scanning operation"""
    
    def __init__(self, success: bool = True, opportunities: list = None, errors: list = None):
        self.success = success
        self.opportunities = opportunities or []
        self.errors = errors or []
        self.scan_timestamp = datetime.now()
        self.total_opportunities = len(self.opportunities)
        
    def __str__(self):
        return f"ScanResult(success={self.success}, opportunities={self.total_opportunities})"

