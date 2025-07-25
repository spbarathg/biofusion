"""
MARKET DATA FETCHER - CENTRALIZED DATA SOURCING ENGINE
====================================================

Production-ready market data fetching system that aggregates real-time data
from multiple sources with intelligent caching, rate limiting, and fallback mechanisms.

This module provides the data foundation for the entire trading system:
- Real-time price feeds
- Volume and liquidity analysis
- Token metadata and contract verification
- Social sentiment indicators
- Historical trend analysis

Features:
- Multi-source data aggregation (Birdeye, DexScreener, Jupiter, Helius)
- Intelligent caching with TTL
- Rate limiting and request management
- Automatic fallback and retry logic
- Data quality validation and sanitization
- Comprehensive error handling and monitoring
"""

import asyncio
import aiohttp
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from urllib.parse import urlencode

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.core.unified_config import get_api_config, get_network_rpc_url


class DataSource(Enum):
    """Market data sources"""
    BIRDEYE = "birdeye"
    DEXSCREENER = "dexscreener"
    JUPITER = "jupiter"
    HELIUS = "helius"
    SOLANA_TRACKER = "solana_tracker"
    RAYDIUM = "raydium"


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"  # <5 min old, multiple sources
    GOOD = "good"           # <15 min old, 2+ sources
    FAIR = "fair"           # <60 min old, 1 source
    POOR = "poor"           # >60 min old or incomplete
    INVALID = "invalid"     # Missing critical data


@dataclass
class DataMetrics:
    """Data quality and freshness metrics"""
    sources_used: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    quality_score: float = 0.0  # 0.0 to 1.0
    quality_level: DataQuality = DataQuality.POOR
    completeness: float = 0.0  # 0.0 to 1.0
    confidence: float = 0.0  # 0.0 to 1.0
    latency_ms: float = 0.0
    cache_hit: bool = False


@dataclass
class TokenData:
    """Comprehensive token data structure"""
    
    # Core identification
    address: str
    symbol: str = "UNKNOWN"
    name: str = "Unknown Token"
    
    # Price data
    price: float = 0.0
    price_usd: float = 0.0
    
    # Volume metrics
    volume_24h_usd: float = 0.0
    volume_24h_sol: float = 0.0
    volume_7d_usd: float = 0.0
    volume_momentum: float = 1.0  # 24h vs 7d average
    
    # Liquidity metrics
    liquidity_sol: float = 0.0
    liquidity_usd: float = 0.0
    liquidity_concentration: float = 0.5  # 0=distributed, 1=concentrated
    
    # Price changes
    price_change_1h_percent: float = 0.0
    price_change_24h_percent: float = 0.0
    price_change_7d_percent: float = 0.0
    
    # Market metrics
    market_cap_usd: Optional[float] = None
    fully_diluted_valuation: Optional[float] = None
    circulating_supply: Optional[float] = None
    total_supply: Optional[float] = None
    
    # Token characteristics
    age_hours: float = 0.0
    holder_count: int = 0
    dev_holdings_percent: float = 0.0
    top_10_holder_percent: float = 0.0
    
    # Contract information
    contract_verified: bool = False
    has_transfer_restrictions: bool = False
    has_blacklist_function: bool = False
    has_mint_function: bool = True
    max_transaction_amount: Optional[float] = None
    
    # Trading metrics
    sell_buy_ratio: float = 1.0
    large_transactions_24h: int = 0
    whale_activity_score: float = 0.0
    
    # Social signals
    social_buzz_score: float = 0.5
    twitter_mentions: int = 0
    telegram_members: int = 0
    discord_members: int = 0
    
    # Technical indicators
    rsi_14: Optional[float] = None
    moving_avg_20: Optional[float] = None
    moving_avg_50: Optional[float] = None
    volatility_24h: float = 0.0
    
    # Risk factors
    honeypot_risk: float = 0.0
    rug_risk_score: float = 0.0
    scam_probability: float = 0.0
    
    # Data quality
    data_metrics: DataMetrics = field(default_factory=DataMetrics)
    
    # Raw data storage
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching/serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if key == 'data_metrics':
                result[key] = {
                    'sources_used': value.sources_used,
                    'last_updated': value.last_updated.isoformat(),
                    'quality_score': value.quality_score,
                    'quality_level': value.quality_level.value,
                    'completeness': value.completeness,
                    'confidence': value.confidence,
                    'latency_ms': value.latency_ms,
                    'cache_hit': value.cache_hit
                }
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result


class MarketDataFetcher:
    """Production-ready market data fetching system"""
    
    def __init__(self):
        self.logger = get_logger("MarketDataFetcher")
        self.api_config = get_api_config()
        self.rpc_url = get_network_rpc_url()
        
        # API endpoints
        self.endpoints = {
            DataSource.BIRDEYE: "https://public-api.birdeye.so",
            DataSource.DEXSCREENER: "https://api.dexscreener.com/latest",
            DataSource.JUPITER: "https://quote-api.jup.ag/v6",
            DataSource.HELIUS: "https://api.helius.xyz/v0",
            DataSource.SOLANA_TRACKER: "https://api.solanatracker.io/tokens",
            DataSource.RAYDIUM: "https://api.raydium.io/v2"
        }
        
        # Caching system
        self.cache: Dict[str, TokenData] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl_seconds = 300  # 5 minutes default TTL
        self.max_cache_size = 1000
        
        # Rate limiting
        self.rate_limits = {
            DataSource.BIRDEYE: {'requests_per_minute': 100, 'last_request': 0},
            DataSource.DEXSCREENER: {'requests_per_minute': 300, 'last_request': 0},
            DataSource.JUPITER: {'requests_per_minute': 200, 'last_request': 0},
            DataSource.HELIUS: {'requests_per_minute': 100, 'last_request': 0},
            DataSource.SOLANA_TRACKER: {'requests_per_minute': 60, 'last_request': 0},
            DataSource.RAYDIUM: {'requests_per_minute': 120, 'last_request': 0}
        }
        
        # Performance metrics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.error_count = 0
        self.average_response_time_ms = 0.0
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.logger.info("📊 Market Data Fetcher initialized - Multi-source aggregation ready")
    
    async def initialize(self) -> bool:
        """Initialize the market data fetcher"""
        try:
            self.logger.info("🚀 Initializing Market Data Fetcher...")
            
            # Create HTTP session with proper configuration
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=20,  # Connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    'User-Agent': 'AntBot/1.0 (Trading Bot)',
                    'Accept': 'application/json',
                    'Accept-Encoding': 'gzip, deflate'
                }
            )
            
            # Test API connections
            await self._test_api_connections()
            
            self.logger.info("✅ Market Data Fetcher initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize market data fetcher: {e}")
            return False
    
    async def get_comprehensive_token_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive token data from multiple sources
        
        Args:
            token_address: Solana token address
            
        Returns:
            Dict with comprehensive token data or None if failed
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cached_data = await self._get_cached_data(token_address)
            if cached_data:
                self.cache_hits += 1
                cached_data.data_metrics.cache_hit = True
                return cached_data.to_dict()

            self.cache_misses += 1
            
            # Fetch from multiple sources
            token_data = await self._fetch_multi_source_data(token_address)
            if not token_data:
                return None
            
            # Calculate data quality metrics
            token_data.data_metrics = await self._calculate_data_quality(token_data)
            
            # Cache the result
            await self._cache_token_data(token_address, token_data)
            
            # Update performance metrics
            response_time_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(response_time_ms)
            
            self.logger.debug(f"📊 Fetched comprehensive data for {token_address[:8]} "
                           f"(Quality: {token_data.data_metrics.quality_level.value}, "
                           f"Sources: {len(token_data.data_metrics.sources_used)})")
            
            return token_data.to_dict()
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching comprehensive token data for {token_address}: {e}")
            self.error_count += 1
            return None
    
    async def _fetch_multi_source_data(self, token_address: str) -> Optional[TokenData]:
        """Fetch data from multiple sources and aggregate"""
        try:
            # Initialize token data structure
            token_data = TokenData(address=token_address)
            sources_used = []
            
            # Fetch from each available source concurrently
            fetch_tasks = []
            
            # Birdeye API (comprehensive data)
            if self.api_config.get('birdeye_api_key'):
                fetch_tasks.append(self._fetch_birdeye_data(token_address))
            
            # DexScreener API (pair and trading data)
            fetch_tasks.append(self._fetch_dexscreener_data(token_address))
            
            # Jupiter API (price data)
            if self.api_config.get('jupiter_api_key'):
                fetch_tasks.append(self._fetch_jupiter_data(token_address))
            
            # Helius API (blockchain data)
            if self.api_config.get('helius_api_key'):
                fetch_tasks.append(self._fetch_helius_data(token_address))
            
            # Solana Tracker API (token metrics)
            if self.api_config.get('solana_tracker_api_key'):
                fetch_tasks.append(self._fetch_solana_tracker_data(token_address))
            
            # Execute all fetch tasks concurrently
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            
            # Aggregate data from all sources
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"❌ Source {i} fetch failed: {result}")
                    continue
                
                if result and isinstance(result, dict):
                    source_name = result.get('source', f'unknown_{i}')
                    sources_used.append(source_name)
                    
                    # Merge data using source-specific logic
                    token_data = await self._merge_source_data(token_data, result, source_name)
            
            # Validate minimum data requirements
            if not self._validate_minimum_data(token_data):
                self.logger.warning(f"⚠️ Insufficient data for token {token_address[:8]}")
                return None
            
            # Store sources used
            token_data.data_metrics.sources_used = sources_used
            
            return token_data
            
        except Exception as e:
            self.logger.error(f"❌ Error in multi-source data fetch: {e}")
            return None
    
    async def _fetch_birdeye_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive data from Birdeye API"""
        try:
            if not await self._check_rate_limit(DataSource.BIRDEYE):
                return None
            
            headers = {'X-API-KEY': self.api_config['birdeye_api_key']}
            
            # Fetch multiple endpoints from Birdeye
            base_url = self.endpoints[DataSource.BIRDEYE]
            
            # Token overview
            overview_url = f"{base_url}/defi/token_overview"
            overview_params = {'address': token_address}
            
            # Price history
            price_url = f"{base_url}/defi/history_price"
            price_params = {
                'address': token_address,
                'address_type': 'token',
                'type': '1D'
            }
            
            # Token security
            security_url = f"{base_url}/defi/token_security"
            security_params = {'address': token_address}
            
            async with self.session.get(overview_url, headers=headers, params=overview_params) as response:
                if response.status == 200:
                    overview_data = await response.json()
                else:
                    self.logger.warning(f"⚠️ Birdeye overview API error: {response.status}")
            
            # Fetch price history
            price_data = {}
            try:
                async with self.session.get(price_url, headers=headers, params=price_params) as response:
                    if response.status == 200:
                        price_data = await response.json()
            except Exception as e:
                self.logger.debug(f"Birdeye price history fetch failed: {e}")
            
            # Fetch security data
            security_data = {}
            try:
                async with self.session.get(security_url, headers=headers, params=security_params) as response:
                    if response.status == 200:
                        security_data = await response.json()
            except Exception as e:
                self.logger.debug(f"Birdeye security data fetch failed: {e}")
            
            # Transform to standard format
            return self._transform_birdeye_data(overview_data, price_data, security_data)
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching Birdeye data: {e}")
            return None
    
    async def _fetch_dexscreener_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Fetch trading data from DexScreener API"""
        try:
            if not await self._check_rate_limit(DataSource.DEXSCREENER):
                return None
            
            url = f"{self.endpoints[DataSource.DEXSCREENER]}/dex/tokens/{token_address}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._transform_dexscreener_data(data)
                else:
                    self.logger.warning(f"⚠️ DexScreener API error: {response.status}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching DexScreener data: {e}")
            return None
    
    async def _fetch_jupiter_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Fetch price data from Jupiter API"""
        try:
            if not await self._check_rate_limit(DataSource.JUPITER):
                return None
            
            url = f"{self.endpoints[DataSource.JUPITER]}/price"
            params = {'ids': token_address}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._transform_jupiter_data(data, token_address)
                else:
                    self.logger.warning(f"⚠️ Jupiter API error: {response.status}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching Jupiter data: {e}")
            return None
    
    async def _fetch_helius_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Fetch blockchain data from Helius API"""
        try:
            if not await self._check_rate_limit(DataSource.HELIUS):
                return None
            
            # Get token metadata
            url = f"{self.endpoints[DataSource.HELIUS]}/token-metadata"
            params = {
                'api-key': self.api_config['helius_api_key'],
                'mint': token_address
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._transform_helius_data(data)
                else:
                    self.logger.warning(f"⚠️ Helius API error: {response.status}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching Helius data: {e}")
            return None
    
    async def _fetch_solana_tracker_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Fetch token metrics from Solana Tracker API"""
        try:
            if not await self._check_rate_limit(DataSource.SOLANA_TRACKER):
                return None
            
            headers = {'Authorization': f'Bearer {self.api_config["solana_tracker_api_key"]}'}
            url = f"{self.endpoints[DataSource.SOLANA_TRACKER]}/{token_address}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._transform_solana_tracker_data(data)
                else:
                    self.logger.warning(f"⚠️ Solana Tracker API error: {response.status}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching Solana Tracker data: {e}")
            return None
    
    def _transform_birdeye_data(self, overview_data: Dict, price_data: Dict, security_data: Dict) -> Dict[str, Any]:
        """Transform Birdeye API response to standard format"""
        try:
            data = overview_data.get('data', {})
            
            return {
                'source': DataSource.BIRDEYE.value,
                'symbol': data.get('symbol', 'UNKNOWN'),
                'name': data.get('name', 'Unknown Token'),
                'price': float(data.get('price', 0)),
                'price_usd': float(data.get('price', 0)),
                'volume_24h_usd': float(data.get('volume24h', 0)),
                'liquidity_usd': float(data.get('liquidity', 0)),
                'price_change_24h_percent': float(data.get('priceChange24h', 0)) * 100,
                'market_cap_usd': data.get('mc'),
                'holder_count': int(data.get('numberMarkets', 0)),
                'contract_verified': security_data.get('data', {}).get('isVerified', False),
                'raw_data': {
                    'overview': overview_data,
                    'price_history': price_data,
                    'security': security_data
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error transforming Birdeye data: {e}")
            return {'source': DataSource.BIRDEYE.value}
    
    def _transform_dexscreener_data(self, data: Dict) -> Dict[str, Any]:
        """Transform DexScreener API response to standard format"""
        try:
            pairs = data.get('pairs', [])
            if not pairs:
                return {'source': DataSource.DEXSCREENER.value}
            
            # Use the first pair (usually the most liquid)
            pair = pairs[0]
            base_token = pair.get('baseToken', {})
            
            # Calculate age
            created_at = pair.get('pairCreatedAt')
            age_hours = 0.0
            if created_at:
                try:
                    created_time = datetime.fromtimestamp(created_at / 1000)
                    age_hours = (datetime.now() - created_time).total_seconds() / 3600
                except:
                    pass
            
            return {
                'source': DataSource.DEXSCREENER.value,
                'symbol': base_token.get('symbol', 'UNKNOWN'),
                'name': base_token.get('name', 'Unknown Token'),
                'price': float(pair.get('priceUsd', 0)),
                'price_usd': float(pair.get('priceUsd', 0)),
                'volume_24h_usd': float(pair.get('volume', {}).get('h24', 0)),
                'liquidity_usd': float(pair.get('liquidity', {}).get('usd', 0)),
                'price_change_1h_percent': float(pair.get('priceChange', {}).get('h1', 0)),
                'price_change_24h_percent': float(pair.get('priceChange', {}).get('h24', 0)),
                'age_hours': age_hours,
                'raw_data': data
            }
            
        except Exception as e:
            self.logger.error(f"Error transforming DexScreener data: {e}")
            return {'source': DataSource.DEXSCREENER.value}
    
    def _transform_jupiter_data(self, data: Dict, token_address: str) -> Dict[str, Any]:
        """Transform Jupiter API response to standard format"""
        try:
            price_data = data.get('data', {}).get(token_address, {})
            
            return {
                'source': DataSource.JUPITER.value,
                'price': float(price_data.get('price', 0)),
                'price_usd': float(price_data.get('price', 0)),
                'raw_data': data
            }
            
        except Exception as e:
            self.logger.error(f"Error transforming Jupiter data: {e}")
            return {'source': DataSource.JUPITER.value}
    
    def _transform_helius_data(self, data: Dict) -> Dict[str, Any]:
        """Transform Helius API response to standard format"""
        try:
            return {
                'source': DataSource.HELIUS.value,
                'symbol': data.get('symbol', 'UNKNOWN'),
                'name': data.get('name', 'Unknown Token'),
                'total_supply': float(data.get('supply', 0)),
                'contract_verified': data.get('verified', False),
                'raw_data': data
            }
            
        except Exception as e:
            self.logger.error(f"Error transforming Helius data: {e}")
            return {'source': DataSource.HELIUS.value}
    
    def _transform_solana_tracker_data(self, data: Dict) -> Dict[str, Any]:
        """Transform Solana Tracker API response to standard format"""
        try:
            return {
                'source': DataSource.SOLANA_TRACKER.value,
                'symbol': data.get('symbol', 'UNKNOWN'),
                'name': data.get('name', 'Unknown Token'),
                'holder_count': int(data.get('holders', 0)),
                'volume_24h_usd': float(data.get('volume24h', 0)),
                'social_buzz_score': min(1.0, float(data.get('socialScore', 0)) / 100),
                'raw_data': data
            }
            
        except Exception as e:
            self.logger.error(f"Error transforming Solana Tracker data: {e}")
            return {'source': DataSource.SOLANA_TRACKER.value}
    
    async def _merge_source_data(self, token_data: TokenData, source_data: Dict[str, Any], source_name: str) -> TokenData:
        """Merge data from a specific source into the main token data"""
        try:
            # Use weighted averaging for numerical fields where we have multiple sources
            weight = self._get_source_weight(source_name)
            
            # Basic fields (take first non-empty value)
            if not token_data.symbol or token_data.symbol == "UNKNOWN":
                token_data.symbol = source_data.get('symbol', token_data.symbol)
            
            if not token_data.name or token_data.name == "Unknown Token":
                token_data.name = source_data.get('name', token_data.name)
            
            # Price data (weighted average)
            if source_data.get('price', 0) > 0:
                if token_data.price == 0:
                    token_data.price = float(source_data['price'])
                else:
                    token_data.price = self._weighted_average(token_data.price, float(source_data['price']), weight)
            
            # Volume data (prefer higher values)
            if source_data.get('volume_24h_usd', 0) > token_data.volume_24h_usd:
                token_data.volume_24h_usd = float(source_data['volume_24h_usd'])
            
            # Liquidity data (prefer higher values)
            if source_data.get('liquidity_usd', 0) > token_data.liquidity_usd:
                token_data.liquidity_usd = float(source_data['liquidity_usd'])
            
            # Price changes (weighted average)
            if source_data.get('price_change_24h_percent') is not None:
                if token_data.price_change_24h_percent == 0:
                    token_data.price_change_24h_percent = float(source_data['price_change_24h_percent'])
                else:
                    token_data.price_change_24h_percent = self._weighted_average(
                        token_data.price_change_24h_percent, 
                        float(source_data['price_change_24h_percent']), 
                        weight
                    )
            
            # Market cap (prefer non-zero values)
            if source_data.get('market_cap_usd') and not token_data.market_cap_usd:
                token_data.market_cap_usd = float(source_data['market_cap_usd'])
            
            # Holder count (prefer higher values)
            if source_data.get('holder_count', 0) > token_data.holder_count:
                token_data.holder_count = int(source_data['holder_count'])
            
            # Age (prefer actual values)
            if source_data.get('age_hours', 0) > 0 and token_data.age_hours == 0:
                token_data.age_hours = float(source_data['age_hours'])
            
            # Contract verification (any true value makes it true)
            if source_data.get('contract_verified', False):
                token_data.contract_verified = True
            
            # Social signals
            if source_data.get('social_buzz_score', 0) > token_data.social_buzz_score:
                token_data.social_buzz_score = float(source_data['social_buzz_score'])
            
            # Store raw data
            token_data.raw_data[source_name] = source_data.get('raw_data', source_data)
            
            return token_data
            
        except Exception as e:
            self.logger.error(f"Error merging source data from {source_name}: {e}")
            return token_data
    
    def _get_source_weight(self, source_name: str) -> float:
        """Get reliability weight for different data sources"""
        weights = {
            DataSource.BIRDEYE.value: 0.9,      # Most comprehensive
            DataSource.DEXSCREENER.value: 0.8,  # Good trading data
            DataSource.JUPITER.value: 0.7,      # Reliable pricing
            DataSource.HELIUS.value: 0.8,       # Good blockchain data
            DataSource.SOLANA_TRACKER.value: 0.6, # Social signals
            DataSource.RAYDIUM.value: 0.7       # DEX data
        }
        return weights.get(source_name, 0.5)
    
    def _weighted_average(self, current_value: float, new_value: float, weight: float) -> float:
        """Calculate weighted average between current and new value"""
        return current_value * (1 - weight) + new_value * weight
    
    def _validate_minimum_data(self, token_data: TokenData) -> bool:
        """Validate that we have minimum required data"""
        return (
            token_data.address and
            token_data.symbol and token_data.symbol != "UNKNOWN" and
            token_data.price > 0
        )
    
    async def _calculate_data_quality(self, token_data: TokenData) -> DataMetrics:
        """Calculate comprehensive data quality metrics"""
        try:
            metrics = DataMetrics()
            metrics.sources_used = token_data.data_metrics.sources_used.copy()
            metrics.last_updated = datetime.now()
            
            # Calculate completeness score
            required_fields = [
                'symbol', 'name', 'price', 'volume_24h_usd', 'liquidity_usd',
                'price_change_24h_percent', 'market_cap_usd', 'holder_count'
            ]
            
            completed_fields = 0
            for field in required_fields:
                value = getattr(token_data, field, None)
                if value and value != "UNKNOWN" and value != "Unknown Token" and value != 0:
                    completed_fields += 1
            
            metrics.completeness = completed_fields / len(required_fields)
            
            # Calculate quality score based on multiple factors
            quality_factors = []
            
            # Source diversity (0-0.3)
            source_score = min(0.3, len(metrics.sources_used) * 0.1)
            quality_factors.append(source_score)
            
            # Data completeness (0-0.4)
            completeness_score = metrics.completeness * 0.4
            quality_factors.append(completeness_score)
            
            # Data freshness (0-0.2)
            freshness_score = 0.2  # Full score for new data
            quality_factors.append(freshness_score)
            
            # Price validation (0-0.1)
            price_valid = 0.1 if token_data.price > 0 else 0.0
            quality_factors.append(price_valid)
            
            metrics.quality_score = sum(quality_factors)
            
            # Determine quality level
            if metrics.quality_score >= 0.8 and len(metrics.sources_used) >= 3:
                metrics.quality_level = DataQuality.EXCELLENT
            elif metrics.quality_score >= 0.6 and len(metrics.sources_used) >= 2:
                metrics.quality_level = DataQuality.GOOD
            elif metrics.quality_score >= 0.4:
                metrics.quality_level = DataQuality.FAIR
            elif metrics.quality_score >= 0.2:
                metrics.quality_level = DataQuality.POOR
            else:
                metrics.quality_level = DataQuality.INVALID
            
            # Calculate confidence
            metrics.confidence = min(1.0, metrics.quality_score + (len(metrics.sources_used) * 0.1))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating data quality: {e}")
            return DataMetrics()
    
    async def _get_cached_data(self, token_address: str) -> Optional[TokenData]:
        """Get data from cache if valid"""
        try:
            if token_address not in self.cache:
                return None
    
            # Check if cache is still valid
            cache_time = self.cache_timestamps.get(token_address)
            if not cache_time:
                return None
            
            age_seconds = (datetime.now() - cache_time).total_seconds()
            if age_seconds > self.cache_ttl_seconds:
                # Remove expired entry
                self.cache.pop(token_address, None)
                self.cache_timestamps.pop(token_address, None)
                return None
            
            return self.cache[token_address]
            
        except Exception as e:
            self.logger.error(f"Error getting cached data: {e}")
            return None
    
    async def _cache_token_data(self, token_address: str, token_data: TokenData):
        """Cache token data with TTL management"""
        try:
            # Clean cache if it's getting too large
            if len(self.cache) >= self.max_cache_size:
                await self._clean_cache()
            
            self.cache[token_address] = token_data
            self.cache_timestamps[token_address] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error caching token data: {e}")
    
    async def _clean_cache(self):
        """Remove oldest cache entries"""
        try:
            # Sort by timestamp and remove oldest 20%
            sorted_entries = sorted(
                self.cache_timestamps.items(),
                key=lambda x: x[1]
            )
            
            entries_to_remove = len(sorted_entries) // 5  # Remove 20%
            
            for address, _ in sorted_entries[:entries_to_remove]:
                self.cache.pop(address, None)
                self.cache_timestamps.pop(address, None)
            
            self.logger.debug(f"🗑️ Cleaned {entries_to_remove} old cache entries")
            
        except Exception as e:
            self.logger.error(f"Error cleaning cache: {e}")
    
    async def _check_rate_limit(self, source: DataSource) -> bool:
        """Check if we can make a request to the given source"""
        try:
            current_time = time.time()
            rate_info = self.rate_limits[source]
            
            # Check if enough time has passed since last request
            min_interval = 60.0 / rate_info['requests_per_minute']
            time_since_last = current_time - rate_info['last_request']
            
            if time_since_last < min_interval:
                self.logger.debug(f"⏳ Rate limit hit for {source.value}, waiting...")
                await asyncio.sleep(min_interval - time_since_last)
            
            # Update last request time
            self.rate_limits[source]['last_request'] = time.time()
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking rate limit for {source.value}: {e}")
            return False
    
    async def _test_api_connections(self):
        """Test connections to available APIs"""
        try:
            connection_tests = []
            
            # Test each API that has keys configured
            if self.api_config.get('birdeye_api_key'):
                connection_tests.append(self._test_birdeye_connection())
            
            connection_tests.append(self._test_dexscreener_connection())
            
            if self.api_config.get('jupiter_api_key'):
                connection_tests.append(self._test_jupiter_connection())
            
            if self.api_config.get('helius_api_key'):
                connection_tests.append(self._test_helius_connection())
            
            # Run all tests concurrently
            results = await asyncio.gather(*connection_tests, return_exceptions=True)
            
            successful_tests = sum(1 for result in results if result is True)
            self.logger.info(f"✅ API Connection Tests: {successful_tests}/{len(results)} passed")
            
        except Exception as e:
            self.logger.error(f"Error testing API connections: {e}")
    
    async def _test_birdeye_connection(self) -> bool:
        """Test Birdeye API connection"""
        try:
            headers = {'X-API-KEY': self.api_config['birdeye_api_key']}
            url = f"{self.endpoints[DataSource.BIRDEYE]}/public/tokenlist"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    self.logger.info("✅ Birdeye API connection verified")
                    return True
                else:
                    self.logger.warning(f"⚠️ Birdeye API test failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Birdeye API connection test failed: {e}")
            return False
    
    async def _test_dexscreener_connection(self) -> bool:
        """Test DexScreener API connection"""
        try:
            url = f"{self.endpoints[DataSource.DEXSCREENER]}/dex/search?q=SOL"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    self.logger.info("✅ DexScreener API connection verified")
                    return True
                else:
                    self.logger.warning(f"⚠️ DexScreener API test failed: {response.status}")
            return False
    
        except Exception as e:
            self.logger.warning(f"⚠️ DexScreener API connection test failed: {e}")
            return False
    
    async def _test_jupiter_connection(self) -> bool:
        """Test Jupiter API connection"""
        try:
            url = f"{self.endpoints[DataSource.JUPITER]}/price?ids=So11111111111111111111111111111111111111112"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    self.logger.info("✅ Jupiter API connection verified")
                    return True
                else:
                    self.logger.warning(f"⚠️ Jupiter API test failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Jupiter API connection test failed: {e}")
            return False
    
    async def _test_helius_connection(self) -> bool:
        """Test Helius API connection"""
        try:
            url = f"{self.endpoints[DataSource.HELIUS]}/addresses/So11111111111111111111111111111111111111112"
            params = {'api-key': self.api_config['helius_api_key']}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    self.logger.info("✅ Helius API connection verified")
                    return True
                else:
                    self.logger.warning(f"⚠️ Helius API test failed: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Helius API connection test failed: {e}")
            return False
    
    def _update_performance_metrics(self, response_time_ms: float):
        """Update performance tracking metrics"""
        try:
            self.total_requests += 1
            
            # Update average response time
            if self.total_requests == 1:
                self.average_response_time_ms = response_time_ms
            else:
                # Exponential moving average
                alpha = 0.1
                self.average_response_time_ms = (alpha * response_time_ms + 
                                               (1 - alpha) * self.average_response_time_ms)
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def get_fetcher_status(self) -> Dict[str, Any]:
        """Get comprehensive fetcher status"""
        cache_hit_rate = 0.0
        if self.total_requests > 0:
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
        
        return {
            'initialized': self.session is not None,
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': round(cache_hit_rate, 3),
            'error_count': self.error_count,
            'average_response_time_ms': round(self.average_response_time_ms, 2),
            'cache_size': len(self.cache),
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'data_sources_available': len([
                source for source, key in [
                    (DataSource.BIRDEYE, 'birdeye_api_key'),
                    (DataSource.JUPITER, 'jupiter_api_key'),
                    (DataSource.HELIUS, 'helius_api_key'),
                    (DataSource.SOLANA_TRACKER, 'solana_tracker_api_key'),
                    (DataSource.RAYDIUM, 'raydium_api_key')
                ] if self.api_config.get(key)
            ]) + 1  # +1 for DexScreener (no key required)
        }
    
    async def shutdown(self):
        """Shutdown the market data fetcher"""
        try:
            self.logger.info("🛑 Shutting down market data fetcher...")
            
            # Clear caches
            self.cache.clear()
            self.cache_timestamps.clear()
            
            # Close HTTP session
            if self.session:
                await self.session.close()
                self.session = None
            
            self.logger.info("✅ Market data fetcher shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during fetcher shutdown: {e}")


# Global instance manager
_market_data_fetcher = None

async def get_market_data_fetcher() -> MarketDataFetcher:
    """Get global market data fetcher instance"""
    global _market_data_fetcher
    if _market_data_fetcher is None:
        _market_data_fetcher = MarketDataFetcher()
        await _market_data_fetcher.initialize()
    return _market_data_fetcher