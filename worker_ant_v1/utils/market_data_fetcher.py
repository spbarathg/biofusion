"""
CENTRALIZED MARKET DATA FETCHER
==============================

Unified utility for fetching market data from multiple sources.
Consolidates all external API calls to reduce redundancy and improve maintainability.
"""

import asyncio
import aiohttp
import os
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.utils.constants import (
    DataSource, APIEndpoints, TokenMints, MarketScannerConstants,
    DefaultValues, ErrorMessages, SuccessMessages
)


# DataSource enum is now imported from constants


@dataclass
class TokenMarketData:
    """Standardized token market data structure"""
    
    # Basic token info
    token_address: str
    symbol: str
    name: str
    
    # Price data
    price: float
    price_change_1h: float = 0.0
    price_change_24h: float = 0.0
    
    # Volume and liquidity
    volume_24h: float = 0.0
    liquidity: float = 0.0
    market_cap: float = 0.0
    
    # Token metrics
    holder_count: int = 0
    age_hours: float = 0.0
    
    # Metadata
    decimals: int = 9
    supply: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Source tracking
    data_sources: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class MarketOpportunity:
    """Trading opportunity with market data"""
    
    token_address: str
    token_symbol: str
    market_data: TokenMarketData
    
    # Opportunity metrics
    opportunity_score: float
    risk_level: float
    expected_profit: float
    
    # Trading parameters
    recommended_position_size: float
    max_slippage_percent: float
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.now)
    source: str = "market_scanner"


class MarketDataFetcher:
    """Centralized market data fetcher with caching and rate limiting"""
    
    def __init__(self):
        self.logger = get_logger("MarketDataFetcher")
        
        # API endpoints
        self.endpoints = {
            DataSource.JUPITER: {
                "base_url": APIEndpoints.JUPITER_BASE_URL,
                "price_url": APIEndpoints.JUPITER_PRICE_URL,
                "quote_url": APIEndpoints.JUPITER_QUOTE_URL,
                "swap_url": APIEndpoints.JUPITER_SWAP_URL,
            },
            DataSource.BIRDEYE: {
                "base_url": APIEndpoints.BIRDEYE_BASE_URL,
                "token_url": APIEndpoints.BIRDEYE_TOKEN_URL,
                "trending_url": APIEndpoints.BIRDEYE_TRENDING_URL,
                "price_history_url": f"{APIEndpoints.BIRDEYE_BASE_URL}/public/token/{{}}/price_history",
            },
            DataSource.DEXSCREENER: {
                "base_url": APIEndpoints.DEXSCREENER_BASE_URL,
                "token_url": APIEndpoints.DEXSCREENER_TOKEN_URL,
                "trending_url": APIEndpoints.DEXSCREENER_TRENDING_URL,
            },
            DataSource.HELIUS: {
                "base_url": APIEndpoints.HELIUS_BASE_URL,
                "token_metadata_url": APIEndpoints.HELIUS_TOKEN_METADATA_URL,
            },
        }
        
        # API keys - CANONICAL ACCESS THROUGH UNIFIED CONFIG
        from worker_ant_v1.core.unified_config import get_trading_config
        config = get_trading_config()
        
        self.api_keys = {
            DataSource.BIRDEYE: config.birdeye_api_key,
            DataSource.HELIUS: config.helius_api_key,
            DataSource.DEXSCREENER: config.dexscreener_api_key,
        }
        
        # Rate limiting
        self.rate_limits = {
            DataSource.JUPITER: {"requests_per_second": 10, "last_request": 0.0},
            DataSource.BIRDEYE: {"requests_per_second": 5, "last_request": 0.0},
            DataSource.DEXSCREENER: {"requests_per_second": 5, "last_request": 0.0},
            DataSource.HELIUS: {"requests_per_second": 3, "last_request": 0.0},
        }
        
        # Caching
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = MonitoringConstants.DEFAULT_CACHE_DURATION_SECONDS
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=10)
        
        # Statistics
        self.stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "errors": 0,
            "last_reset": datetime.now(),
        }
        
        self.logger.info("🔍 Market Data Fetcher initialized")
    
    async def initialize(self) -> bool:
        """Initialize the market data fetcher"""
        try:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            
            # Test connections to available APIs
            await self._test_api_connections()
            
            self.logger.info("✅ Market Data Fetcher initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Market Data Fetcher initialization failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the market data fetcher"""
        if self.session:
            await self.session.close()
        self.logger.info("🛑 Market Data Fetcher shutdown complete")
    
    async def get_token_data(self, token_address: str, sources: Optional[List[DataSource]] = None) -> Optional[TokenMarketData]:
        """Get comprehensive token data from multiple sources"""
        try:
            # Check cache first
            cache_key = f"token_{token_address}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                self.stats["cache_hits"] += 1
                return TokenMarketData(**cached_data)
            
            # Use default sources if none specified
            if sources is None:
                sources = [DataSource.JUPITER, DataSource.BIRDEYE, DataSource.DEXSCREENER]
            
            # Fetch data from all sources
            tasks = []
            for source in sources:
                if source in self.endpoints:
                    tasks.append(self._fetch_from_source(source, token_address))
            
            if not tasks:
                self.logger.warning(f"No valid sources available for {token_address}")
                return None
            
            # Wait for all requests to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            combined_data = await self._combine_token_data(token_address, results, sources)
            
            # Cache the result
            self._cache_data(cache_key, combined_data.__dict__)
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error fetching token data for {token_address}: {e}")
            self.stats["errors"] += 1
            return None
    
    async def get_trending_tokens(self, limit: int = 50) -> List[str]:
        """Get trending tokens from multiple sources"""
        try:
            trending_tokens = set()
            
            # Fetch from Birdeye
            if DataSource.BIRDEYE in self.endpoints:
                try:
                    birdeye_tokens = await self._get_birdeye_trending(limit // 2)
                    trending_tokens.update(birdeye_tokens)
                except Exception as e:
                    self.logger.warning(f"Birdeye trending fetch failed: {e}")
            
            # Fetch from DexScreener
            if DataSource.DEXSCREENER in self.endpoints:
                try:
                    dexscreener_tokens = await self._get_dexscreener_trending(limit // 2)
                    trending_tokens.update(dexscreener_tokens)
                except Exception as e:
                    self.logger.warning(f"DexScreener trending fetch failed: {e}")
            
            return list(trending_tokens)[:limit]
            
        except Exception as e:
            self.logger.error(f"Error fetching trending tokens: {e}")
            return []
    
    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get current token price (fast method)"""
        try:
            # Try Jupiter first (fastest)
            price = await self._get_jupiter_price(token_address)
            if price:
                return price
            
            # Fallback to full token data
            token_data = await self.get_token_data(token_address, [DataSource.JUPITER, DataSource.BIRDEYE])
            return token_data.price if token_data else None
            
        except Exception as e:
            self.logger.error(f"Error getting token price for {token_address}: {e}")
            return None
    
    async def get_market_opportunities(
        self, 
        min_liquidity: float = MarketScannerConstants.MIN_LIQUIDITY_SOL, 
        min_volume: float = MarketScannerConstants.MIN_VOLUME_24H_SOL
    ) -> List[MarketOpportunity]:
        """Get market opportunities based on criteria"""
        try:
            opportunities = []
            trending_tokens = await self.get_trending_tokens(MarketScannerConstants.MAX_TRENDING_TOKENS)
            
            # Get data for trending tokens
            for token_address in trending_tokens[:MarketScannerConstants.DEFAULT_TRENDING_TOKENS]:  # Limit for performance
                token_data = await self.get_token_data(token_address)
                if not token_data:
                    continue
                
                # Apply filters
                if token_data.liquidity < min_liquidity or token_data.volume_24h < min_volume:
                    continue
                
                # Calculate opportunity metrics
                opportunity = await self._calculate_opportunity(token_data)
                if opportunity:
                    opportunities.append(opportunity)
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
            
            return opportunities[:MarketScannerConstants.DEFAULT_TRENDING_TOKENS]  # Return top opportunities
            
        except Exception as e:
            self.logger.error(f"Error getting market opportunities: {e}")
            return []
    
    async def _fetch_from_source(self, source: DataSource, token_address: str) -> Optional[Dict[str, Any]]:
        """Fetch data from a specific source"""
        try:
            await self._rate_limit(source)
            
            if source == DataSource.JUPITER:
                return await self._fetch_jupiter_data(token_address)
            elif source == DataSource.BIRDEYE:
                return await self._fetch_birdeye_data(token_address)
            elif source == DataSource.DEXSCREENER:
                return await self._fetch_dexscreener_data(token_address)
            elif source == DataSource.HELIUS:
                return await self._fetch_helius_data(token_address)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching from {source.value}: {e}")
            return None
    
    async def _fetch_jupiter_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Fetch data from Jupiter API"""
        try:
            url = f"{self.endpoints[DataSource.JUPITER]['price_url']}?ids={token_address}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data and token_address in data["data"]:
                        token_data = data["data"][token_address]
                        return {
                            "price": float(token_data.get("price", 0)),
                            "volume_24h": float(token_data.get("volume24h", 0)),
                            "liquidity": float(token_data.get("liquidity", 0)),
                            "source": DataSource.JUPITER.value,
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Jupiter fetch error: {e}")
            return None
    
    async def _fetch_birdeye_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Fetch data from Birdeye API"""
        try:
            headers = {}
            if self.api_keys[DataSource.BIRDEYE]:
                headers["X-API-KEY"] = self.api_keys[DataSource.BIRDEYE]
            
            url = f"{self.endpoints[DataSource.BIRDEYE]['token_url']}/{token_address}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data:
                        token_info = data["data"]
                        return {
                            "symbol": token_info.get("symbol", "UNKNOWN"),
                            "name": token_info.get("name", "Unknown Token"),
                            "price": float(token_info.get("price", 0)),
                            "volume_24h": float(token_info.get("volume24h", 0)),
                            "liquidity": float(token_info.get("liquidity", 0)),
                            "market_cap": float(token_info.get("marketCap", 0)),
                            "price_change_24h": float(token_info.get("priceChange24h", 0)),
                            "price_change_1h": float(token_info.get("priceChange1h", 0)),
                            "holder_count": int(token_info.get("holder", 0)),
                            "age_hours": float(token_info.get("ageHours", 0)),
                            "source": DataSource.BIRDEYE.value,
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Birdeye fetch error: {e}")
            return None
    
    async def _fetch_dexscreener_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Fetch data from DexScreener API"""
        try:
            url = f"{self.endpoints[DataSource.DEXSCREENER]['token_url']}/{token_address}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get("pairs", [])
                    if pairs:
                        pair = pairs[0]  # Take first pair
                        return {
                            "symbol": pair.get("baseToken", {}).get("symbol", "UNKNOWN"),
                            "name": pair.get("baseToken", {}).get("name", "Unknown Token"),
                            "price": float(pair.get("priceNative", 0)),
                            "volume_24h": float(pair.get("volume", {}).get("h24", 0)),
                            "liquidity": float(pair.get("liquidity", {}).get("base", 0)),
                            "price_change_24h": float(pair.get("priceChange", {}).get("h24", 0)),
                            "source": DataSource.DEXSCREENER.value,
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"DexScreener fetch error: {e}")
            return None
    
    async def _fetch_helius_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata from Helius API"""
        try:
            if not self.api_keys[DataSource.HELIUS]:
                return None
            
            url = f"{self.endpoints[DataSource.HELIUS]['token_metadata_url']}?api-key={self.api_keys[DataSource.HELIUS]}"
            payload = {"mintAccounts": [token_address]}
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        metadata = data[0].get("onChainMetadata", {}).get("metadata", {})
                        return {
                            "symbol": metadata.get("symbol", "UNKNOWN"),
                            "name": metadata.get("name", "Unknown Token"),
                            "decimals": int(metadata.get("decimals", 9)),
                            "source": DataSource.HELIUS.value,
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Helius fetch error: {e}")
            return None
    
    async def _get_jupiter_price(self, token_address: str) -> Optional[float]:
        """Get price from Jupiter (fast method)"""
        try:
            url = f"{self.endpoints[DataSource.JUPITER]['price_url']}?ids={token_address}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data and token_address in data["data"]:
                        return float(data["data"][token_address].get("price", 0))
            
            return None
            
        except Exception as e:
            self.logger.error(f"Jupiter price fetch error: {e}")
            return None
    
    async def _get_birdeye_trending(self, limit: int) -> List[str]:
        """Get trending tokens from Birdeye"""
        try:
            headers = {}
            if self.api_keys[DataSource.BIRDEYE]:
                headers["X-API-KEY"] = self.api_keys[DataSource.BIRDEYE]
            
            url = self.endpoints[DataSource.BIRDEYE]["trending_url"]
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if "data" in data:
                        tokens = []
                        for token in data["data"][:limit]:
                            if "address" in token:
                                tokens.append(token["address"])
                        return tokens
            
            return []
            
        except Exception as e:
            self.logger.error(f"Birdeye trending fetch error: {e}")
            return []
    
    async def _get_dexscreener_trending(self, limit: int) -> List[str]:
        """Get trending tokens from DexScreener"""
        try:
            url = self.endpoints[DataSource.DEXSCREENER]["trending_url"]
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if "pairs" in data:
                        tokens = []
                        for pair in data["pairs"][:limit]:
                            if "tokenAddress" in pair:
                                tokens.append(pair["tokenAddress"])
                        return tokens
            
            return []
            
        except Exception as e:
            self.logger.error(f"DexScreener trending fetch error: {e}")
            return []
    
    async def _combine_token_data(self, token_address: str, results: List[Any], sources: List[DataSource]) -> TokenMarketData:
        """Combine data from multiple sources"""
        try:
            # Filter out exceptions and None results
            valid_results = [r for r in results if isinstance(r, dict) and r is not None]
            
            if not valid_results:
                return TokenMarketData(
                    token_address=token_address,
                    symbol="UNKNOWN",
                    name="Unknown Token",
                    price=0.0,
                )
            
            # Combine data with priority
            combined = {
                "token_address": token_address,
                "symbol": "UNKNOWN",
                "name": "Unknown Token",
                "price": 0.0,
                "price_change_1h": 0.0,
                "price_change_24h": 0.0,
                "volume_24h": 0.0,
                "liquidity": 0.0,
                "market_cap": 0.0,
                "holder_count": 0,
                "age_hours": 0.0,
                "decimals": 9,
                "data_sources": [],
                "confidence_score": 0.0,
            }
            
            # Priority order: Jupiter > Birdeye > DexScreener > Helius
            priority_order = [DataSource.JUPITER, DataSource.BIRDEYE, DataSource.DEXSCREENER, DataSource.HELIUS]
            
            for source in priority_order:
                source_data = next((r for r in valid_results if r.get("source") == source.value), None)
                if source_data:
                    combined["data_sources"].append(source.value)
                    
                    # Update fields if not already set
                    for field, value in source_data.items():
                        if field != "source" and (combined.get(field) == 0 or combined.get(field) == "UNKNOWN"):
                            combined[field] = value
            
            # Calculate confidence score based on number of sources
            combined["confidence_score"] = min(len(combined["data_sources"]) / len(sources), 1.0)
            
            return TokenMarketData(**combined)
            
        except Exception as e:
            self.logger.error(f"Error combining token data: {e}")
            return TokenMarketData(
                token_address=token_address,
                symbol="UNKNOWN",
                name="Unknown Token",
                price=0.0,
            )
    
    async def _calculate_opportunity(self, token_data: TokenMarketData) -> Optional[MarketOpportunity]:
        """Calculate trading opportunity from token data"""
        try:
            # Basic opportunity scoring
            volume_score = min(token_data.volume_24h / 1000, 1.0)  # Normalize to 1000 SOL
            liquidity_score = min(token_data.liquidity / 100, 1.0)  # Normalize to 100 SOL
            price_change_score = abs(token_data.price_change_24h) / 100  # Normalize to 100%
            
            # Calculate opportunity score
            opportunity_score = (volume_score * 0.3 + liquidity_score * 0.3 + price_change_score * 0.4)
            
            # Calculate risk level (inverse of liquidity)
            risk_level = max(0.1, 1.0 - liquidity_score)
            
            # Calculate expected profit (simplified)
            expected_profit = price_change_score * 0.5  # Conservative estimate
            
            # Determine position size based on liquidity
            recommended_position_size = min(token_data.liquidity * 0.1, 50.0)  # Max 10% of liquidity, 50 SOL
            
            # Calculate max slippage based on liquidity
            max_slippage_percent = max(0.5, 5.0 - (liquidity_score * 4.5))  # 0.5% to 5%
            
            return MarketOpportunity(
                token_address=token_data.token_address,
                token_symbol=token_data.symbol,
                market_data=token_data,
                opportunity_score=opportunity_score,
                risk_level=risk_level,
                expected_profit=expected_profit,
                recommended_position_size=recommended_position_size,
                max_slippage_percent=max_slippage_percent,
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating opportunity: {e}")
            return None
    
    async def _rate_limit(self, source: DataSource):
        """Apply rate limiting for API calls"""
        if source not in self.rate_limits:
            return
        
        rate_limit = self.rate_limits[source]
        min_interval = 1.0 / rate_limit["requests_per_second"]
        
        current_time = time.time()
        time_since_last = current_time - rate_limit["last_request"]
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        rate_limit["last_request"] = time.time()
        self.stats["requests_made"] += 1
    
    def _get_cached_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache"""
        if key in self.cache:
            cached_item = self.cache[key]
            if datetime.now() - cached_item["timestamp"] < timedelta(seconds=self.cache_ttl):
                return cached_item["data"]
            else:
                del self.cache[key]
        return None
    
    def _cache_data(self, key: str, data: Dict[str, Any]):
        """Cache data"""
        self.cache[key] = {
            "data": data,
            "timestamp": datetime.now(),
        }
        
        # Clean old cache entries
        current_time = datetime.now()
        keys_to_remove = []
        for k, v in self.cache.items():
            if current_time - v["timestamp"] > timedelta(seconds=self.cache_ttl * 2):
                keys_to_remove.append(k)
        
        for k in keys_to_remove:
            del self.cache[k]
    
    async def _test_api_connections(self):
        """Test connections to available APIs"""
        test_tasks = []
        
        if DataSource.JUPITER in self.endpoints:
            test_tasks.append(self._test_jupiter_connection())
        
        if DataSource.BIRDEYE in self.endpoints:
            test_tasks.append(self._test_birdeye_connection())
        
        if DataSource.DEXSCREENER in self.endpoints:
            test_tasks.append(self._test_dexscreener_connection())
        
        if test_tasks:
            results = await asyncio.gather(*test_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"API connection test failed: {result}")
                else:
                    self.logger.info(f"✅ API connection test passed")
    
    async def _test_jupiter_connection(self) -> bool:
        """Test Jupiter API connection"""
        try:
            url = f"{self.endpoints[DataSource.JUPITER]['price_url']}?ids=SOL"
            async with self.session.get(url) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def _test_birdeye_connection(self) -> bool:
        """Test Birdeye API connection"""
        try:
            headers = {}
            if self.api_keys[DataSource.BIRDEYE]:
                headers["X-API-KEY"] = self.api_keys[DataSource.BIRDEYE]
            
            url = f"{self.endpoints[DataSource.BIRDEYE]['token_url']}/So11111111111111111111111111111111111111112"
            async with self.session.get(url, headers=headers) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def _test_dexscreener_connection(self) -> bool:
        """Test DexScreener API connection"""
        try:
            url = f"{self.endpoints[DataSource.DEXSCREENER]['token_url']}/So11111111111111111111111111111111111111112"
            async with self.session.get(url) as response:
                return response.status == 200
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics"""
        return {
            **self.stats,
            "cache_size": len(self.cache),
            "uptime_hours": (datetime.now() - self.stats["last_reset"]).total_seconds() / 3600,
        }


# Global instance
_market_data_fetcher: Optional[MarketDataFetcher] = None


async def get_market_data_fetcher() -> MarketDataFetcher:
    """Get or create global market data fetcher instance"""
    global _market_data_fetcher
    
    if _market_data_fetcher is None:
        _market_data_fetcher = MarketDataFetcher()
        await _market_data_fetcher.initialize()
    
    return _market_data_fetcher


async def shutdown_market_data_fetcher():
    """Shutdown global market data fetcher"""
    global _market_data_fetcher
    
    if _market_data_fetcher:
        await _market_data_fetcher.shutdown()
        _market_data_fetcher = None 