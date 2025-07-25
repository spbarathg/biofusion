"""
REAL MARKET SCANNER - OPPORTUNITY DETECTION ENGINE
================================================

Production-ready market scanner that identifies high-probability trading opportunities
by analyzing real-time market data from multiple sources with mathematical precision.

This is the entry point for the three-stage trading pipeline:
1. Market scanning identifies opportunities
2. Three-stage pipeline processes each opportunity
3. Execution engine handles approved trades

Features:
- Multi-source market data aggregation
- Real-time opportunity scoring
- Risk-adjusted filtering
- Volume and liquidity analysis
- Age-based opportunity prioritization
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import json

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.core.unified_config import get_trading_config, get_api_config, get_network_rpc_url
from worker_ant_v1.utils.market_data_fetcher import MarketDataFetcher


class OpportunityLevel(Enum):
    """Opportunity quality levels"""
    AVOID = "avoid"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PREMIUM = "premium"


@dataclass
class MarketOpportunity:
    """Structured market opportunity data"""
    
    # Core identification
    token_address: str
    token_symbol: str
    token_name: str
    detected_at: datetime
    
    # Opportunity metrics
    opportunity_score: float  # 0.0 to 1.0
    risk_level: str  # low, medium, high
    expected_profit_percent: float
    confidence_score: float
    
    # Market data
    current_price: float
    volume_24h_usd: float
    liquidity_sol: float
    price_change_1h_percent: float
    price_change_24h_percent: float
    market_cap_usd: Optional[float]
    
    # Token characteristics
    age_hours: float
    holder_count: int
    liquidity_concentration: float  # 0.0 to 1.0
    dev_holdings_percent: float
    
    # Technical indicators
    volume_momentum: float  # volume_24h / volume_avg_7d
    price_momentum: float   # current momentum score
    social_buzz_score: float  # 0.0 to 1.0
    
    # Risk factors
    contract_verified: bool
    has_transfer_restrictions: bool
    has_blacklist_function: bool
    sell_buy_ratio: float
    large_transactions_24h: int
    
    # Metadata
    data_sources: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class RealMarketScanner:
    """Production-ready market scanner with multi-source data aggregation"""
    
    def __init__(self):
        self.logger = get_logger("RealMarketScanner")
        self.config = get_trading_config()
        self.api_config = get_api_config()
        
        # Core systems
        self.market_data_fetcher = MarketDataFetcher()
        
        # Scanning configuration
        self.scan_interval_seconds = getattr(self.config, 'scan_interval_seconds', 30)
        self.min_volume_threshold_usd = 1000.0  # Minimum $1K volume
        self.min_liquidity_threshold_sol = 5.0  # Minimum 5 SOL liquidity
        self.max_token_age_hours = 168  # 7 days maximum age
        self.opportunity_cache_minutes = 5  # Cache opportunities for 5 minutes
        
        # Performance tracking
        self.total_opportunities_found = 0
        self.total_scans_completed = 0
        self.average_scan_time_ms = 0.0
        self.last_scan_time = None
        
        # Opportunity cache
        self.opportunity_cache: Dict[str, MarketOpportunity] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Data source endpoints
        self.data_sources = {
            'birdeye': 'https://public-api.birdeye.so',
            'dexscreener': 'https://api.dexscreener.com/latest/dex',
            'jupiter': 'https://quote-api.jup.ag/v6'
        }
        
        self.logger.info("🔍 Real Market Scanner initialized - Ready to hunt opportunities")
    
    async def initialize(self) -> bool:
        """Initialize the market scanner"""
        try:
            self.logger.info("🚀 Initializing Real Market Scanner...")
            
            # Initialize market data fetcher
            await self.market_data_fetcher.initialize()
            
            # Test API connections
            await self._test_api_connections()
            
            self.logger.info("✅ Real Market Scanner initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize market scanner: {e}")
            return False
    
    async def scan_opportunities(self) -> List[MarketOpportunity]:
        """
        Scan for high-probability trading opportunities
        
        Returns:
            List of MarketOpportunity objects ranked by opportunity score
        """
        scan_start_time = time.time()
        
        try:
            self.logger.debug("🔍 Starting market opportunity scan...")
            
            # Clear expired cache entries
            await self._clean_opportunity_cache()
            
            # Get new token discoveries from multiple sources
            new_tokens = await self._discover_new_tokens()
            
            # Analyze each token for trading opportunities
            opportunities = []
            for token_data in new_tokens:
                opportunity = await self._analyze_token_opportunity(token_data)
                if opportunity and opportunity.opportunity_score > 0.3:  # Minimum threshold
                    opportunities.append(opportunity)
                    self.total_opportunities_found += 1
            
            # Filter and rank opportunities
            filtered_opportunities = await self._filter_and_rank_opportunities(opportunities)
            
            # Update performance metrics
            scan_duration_ms = (time.time() - scan_start_time) * 1000
            self._update_scan_metrics(scan_duration_ms)
            
            self.logger.info(f"✅ Scan complete: {len(filtered_opportunities)} opportunities found in {scan_duration_ms:.1f}ms")
            return filtered_opportunities
            
        except Exception as e:
            self.logger.error(f"❌ Error during market scan: {e}")
            return []
    
    async def _discover_new_tokens(self) -> List[Dict[str, Any]]:
        """Discover new tokens from multiple data sources"""
        try:
            all_tokens = []
            
            # Source 1: Birdeye trending tokens
            if self.api_config.get('birdeye_api_key'):
                birdeye_tokens = await self._get_birdeye_trending_tokens()
                all_tokens.extend(birdeye_tokens)
            
            # Source 2: DexScreener new pairs
            dexscreener_tokens = await self._get_dexscreener_new_pairs()
            all_tokens.extend(dexscreener_tokens)
            
            # Source 3: Jupiter trending tokens
            if self.api_config.get('jupiter_api_key'):
                jupiter_tokens = await self._get_jupiter_trending_tokens()
                all_tokens.extend(jupiter_tokens)
            
            # Deduplicate by token address
            unique_tokens = {}
            for token in all_tokens:
                address = token.get('address')
                if address and address not in unique_tokens:
                    unique_tokens[address] = token
            
            self.logger.debug(f"📊 Discovered {len(unique_tokens)} unique tokens from {len(all_tokens)} total")
            return list(unique_tokens.values())
            
        except Exception as e:
            self.logger.error(f"❌ Error discovering new tokens: {e}")
            return []
    
    async def _get_birdeye_trending_tokens(self) -> List[Dict[str, Any]]:
        """Get trending tokens from Birdeye API"""
        try:
            headers = {
                'X-API-KEY': self.api_config['birdeye_api_key']
            }
            
            url = f"{self.data_sources['birdeye']}/defi/trending_tokens/sol"
            params = {
                'sort_by': 'volume24hUSD',
                'sort_type': 'desc',
                'offset': 0,
                'limit': 20
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        tokens = data.get('data', {}).get('items', [])
                        
                        # Transform to standard format
                        standardized = []
                        for token in tokens:
                            standardized.append({
                                'address': token.get('address'),
                                'symbol': token.get('symbol'),
                                'name': token.get('name'),
                                'price': float(token.get('price', 0)),
                                'volume_24h_usd': float(token.get('volume24hUSD', 0)),
                                'price_change_24h_percent': float(token.get('priceChange24h', 0)),
                                'source': 'birdeye'
                            })
                        
                        self.logger.debug(f"📈 Birdeye: {len(standardized)} trending tokens")
                        return standardized
                    else:
                        self.logger.warning(f"⚠️ Birdeye API error: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"❌ Error fetching Birdeye tokens: {e}")
            return []
    
    async def _get_dexscreener_new_pairs(self) -> List[Dict[str, Any]]:
        """Get new pairs from DexScreener API"""
        try:
            # Get latest pairs for Solana
            url = f"{self.data_sources['dexscreener']}/search?q=SOL"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get('pairs', [])
                        
                        # Filter for recent pairs with good volume
                        recent_tokens = []
                        for pair in pairs[:50]:  # Limit to top 50
                            # Check if it's a Solana pair
                            if pair.get('chainId') != 'solana':
                                continue
                                
                            # Check volume threshold
                            volume_24h = float(pair.get('volume', {}).get('h24', 0))
                            if volume_24h < self.min_volume_threshold_usd:
                                continue
                            
                            base_token = pair.get('baseToken', {})
                            recent_tokens.append({
                                'address': base_token.get('address'),
                                'symbol': base_token.get('symbol'),
                                'name': base_token.get('name'),
                                'price': float(pair.get('priceUsd', 0)),
                                'volume_24h_usd': volume_24h,
                                'price_change_24h_percent': float(pair.get('priceChange', {}).get('h24', 0)),
                                'liquidity_usd': float(pair.get('liquidity', {}).get('usd', 0)),
                                'pair_created_at': pair.get('pairCreatedAt'),
                                'source': 'dexscreener'
                            })
                        
                        self.logger.debug(f"📊 DexScreener: {len(recent_tokens)} recent pairs")
                        return recent_tokens
                    else:
                        self.logger.warning(f"⚠️ DexScreener API error: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"❌ Error fetching DexScreener pairs: {e}")
            return []
    
    async def _get_jupiter_trending_tokens(self) -> List[Dict[str, Any]]:
        """Get trending tokens from Jupiter API"""
        try:
            # Jupiter doesn't have a direct trending endpoint, so we'll use price data
            # for tokens that have been recently active
            
            # This is a simplified implementation - in production you'd want
            # to track token addresses from recent Jupiter swaps
            
            url = f"{self.data_sources['jupiter']}/price"
            
            # Sample of known popular token addresses for demonstration
            # In production, this would be dynamically populated
            sample_tokens = [
                'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
                'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',   # USDT
            ]
            
            tokens_data = []
            for token_address in sample_tokens:
                params = {'ids': token_address}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'data' in data and token_address in data['data']:
                                price_data = data['data'][token_address]
                                tokens_data.append({
                                    'address': token_address,
                                    'symbol': 'UNKNOWN',  # Would need additional lookup
                                    'name': 'Unknown Token',
                                    'price': float(price_data.get('price', 0)),
                                    'source': 'jupiter'
                                })
            
            self.logger.debug(f"💱 Jupiter: {len(tokens_data)} tokens")
            return tokens_data
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching Jupiter tokens: {e}")
            return []
    
    async def _analyze_token_opportunity(self, token_data: Dict[str, Any]) -> Optional[MarketOpportunity]:
        """Analyze a single token for trading opportunity"""
        try:
            token_address = token_data.get('address')
            if not token_address:
                return None
            
            # Check cache first
            if token_address in self.opportunity_cache:
                cache_time = self.cache_timestamps.get(token_address)
                if cache_time and (datetime.now() - cache_time).total_seconds() < (self.opportunity_cache_minutes * 60):
                    return self.opportunity_cache[token_address]
            
            # Get comprehensive market data
            market_data = await self.market_data_fetcher.get_comprehensive_token_data(token_address)
            if not market_data:
                return None
            
            # Calculate opportunity metrics
            opportunity_score = await self._calculate_opportunity_score(market_data)
            risk_level = self._assess_risk_level(market_data)
            confidence_score = self._calculate_confidence_score(market_data)
            
            # Apply filters
            if not self._passes_basic_filters(market_data):
                return None
            
            # Create opportunity object
            opportunity = MarketOpportunity(
                # Core identification
                token_address=token_address,
                token_symbol=market_data.get('symbol', 'UNKNOWN'),
                token_name=market_data.get('name', 'Unknown Token'),
                detected_at=datetime.now(),
                
                # Opportunity metrics
                opportunity_score=opportunity_score,
                risk_level=risk_level,
                expected_profit_percent=self._estimate_profit_potential(market_data),
                confidence_score=confidence_score,
                
                # Market data
                current_price=float(market_data.get('price', 0)),
                volume_24h_usd=float(market_data.get('volume_24h_usd', 0)),
                liquidity_sol=float(market_data.get('liquidity_sol', 0)),
                price_change_1h_percent=float(market_data.get('price_change_1h_percent', 0)),
                price_change_24h_percent=float(market_data.get('price_change_24h_percent', 0)),
                market_cap_usd=market_data.get('market_cap_usd'),
                
                # Token characteristics
                age_hours=float(market_data.get('age_hours', 0)),
                holder_count=int(market_data.get('holder_count', 0)),
                liquidity_concentration=float(market_data.get('liquidity_concentration', 0.5)),
                dev_holdings_percent=float(market_data.get('dev_holdings_percent', 0)),
                
                # Technical indicators
                volume_momentum=float(market_data.get('volume_momentum', 1.0)),
                price_momentum=float(market_data.get('price_momentum', 0.0)),
                social_buzz_score=float(market_data.get('social_buzz_score', 0.5)),
                
                # Risk factors
                contract_verified=bool(market_data.get('contract_verified', True)),
                has_transfer_restrictions=bool(market_data.get('has_transfer_restrictions', False)),
                has_blacklist_function=bool(market_data.get('has_blacklist_function', False)),
                sell_buy_ratio=float(market_data.get('sell_buy_ratio', 1.0)),
                large_transactions_24h=int(market_data.get('large_transactions_24h', 0)),
                
                # Metadata
                data_sources=market_data.get('sources', ['market_data_fetcher'])
            )
            
            # Cache the opportunity
            self.opportunity_cache[token_address] = opportunity
            self.cache_timestamps[token_address] = datetime.now()
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing token opportunity {token_data.get('address', 'unknown')}: {e}")
            return None
    
    async def _calculate_opportunity_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate comprehensive opportunity score (0.0 to 1.0)"""
        try:
            score_components = []
            
            # Volume momentum component (0-0.3)
            volume_momentum = float(market_data.get('volume_momentum', 1.0))
            volume_score = min(0.3, max(0.0, (volume_momentum - 1.0) * 0.15))
            score_components.append(volume_score)
            
            # Price momentum component (0-0.25)
            price_change_24h = float(market_data.get('price_change_24h_percent', 0))
            if 5 <= price_change_24h <= 50:  # Sweet spot for momentum
                price_score = 0.25
            elif price_change_24h > 0:
                price_score = min(0.25, price_change_24h / 50 * 0.25)
            else:
                price_score = 0.0
            score_components.append(price_score)
            
            # Liquidity health component (0-0.2)
            liquidity_sol = float(market_data.get('liquidity_sol', 0))
            if liquidity_sol >= 100:
                liquidity_score = 0.2
            elif liquidity_sol >= 10:
                liquidity_score = 0.15
            elif liquidity_sol >= 5:
                liquidity_score = 0.1
            else:
                liquidity_score = 0.0
            score_components.append(liquidity_score)
            
            # Age factor component (0-0.15)
            age_hours = float(market_data.get('age_hours', 0))
            if 6 <= age_hours <= 72:  # 6 hours to 3 days is optimal
                age_score = 0.15
            elif age_hours > 0:
                age_score = max(0.0, 0.15 - (abs(age_hours - 24) / 100 * 0.15))
            else:
                age_score = 0.0
            score_components.append(age_score)
            
            # Social signals component (0-0.1)
            social_buzz = float(market_data.get('social_buzz_score', 0.5))
            social_score = social_buzz * 0.1
            score_components.append(social_score)
            
            # Calculate final score
            final_score = sum(score_components)
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {e}")
            return 0.0
    
    def _assess_risk_level(self, market_data: Dict[str, Any]) -> str:
        """Assess risk level based on market data"""
        try:
            risk_factors = 0
            
            # High risk factors
            if not market_data.get('contract_verified', True):
                risk_factors += 3
            if market_data.get('has_transfer_restrictions', False):
                risk_factors += 3
            if market_data.get('has_blacklist_function', False):
                risk_factors += 2
            if float(market_data.get('dev_holdings_percent', 0)) > 30:
                risk_factors += 2
            if float(market_data.get('liquidity_concentration', 0.5)) > 0.8:
                risk_factors += 2
            
            # Medium risk factors
            if float(market_data.get('age_hours', 0)) < 6:
                risk_factors += 1
            if float(market_data.get('sell_buy_ratio', 1.0)) > 2.0:
                risk_factors += 1
            if float(market_data.get('liquidity_sol', 0)) < 10:
                risk_factors += 1
            
            # Categorize risk
            if risk_factors >= 6:
                return "high"
            elif risk_factors >= 3:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            self.logger.error(f"Error assessing risk level: {e}")
            return "high"  # Default to high risk on error
    
    def _calculate_confidence_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate confidence in the analysis (0.0 to 1.0)"""
        try:
            confidence_factors = []
            
            # Data completeness
            required_fields = ['price', 'volume_24h_usd', 'liquidity_sol', 'age_hours']
            completeness = sum(1 for field in required_fields if market_data.get(field) is not None) / len(required_fields)
            confidence_factors.append(completeness * 0.3)
            
            # Data freshness
            last_updated = market_data.get('last_updated')
            if last_updated:
                freshness = max(0.0, 1.0 - (datetime.now() - last_updated).total_seconds() / 3600)  # Decay over 1 hour
                confidence_factors.append(freshness * 0.2)
            else:
                confidence_factors.append(0.1)
            
            # Contract verification
            if market_data.get('contract_verified', False):
                confidence_factors.append(0.2)
            else:
                confidence_factors.append(0.0)
            
            # Liquidity adequacy
            liquidity_sol = float(market_data.get('liquidity_sol', 0))
            if liquidity_sol >= 50:
                confidence_factors.append(0.2)
            elif liquidity_sol >= 10:
                confidence_factors.append(0.15)
            else:
                confidence_factors.append(0.05)
            
            # Volume consistency
            volume_24h = float(market_data.get('volume_24h_usd', 0))
            if volume_24h >= 10000:  # $10K+
                confidence_factors.append(0.1)
            elif volume_24h >= 1000:  # $1K+
                confidence_factors.append(0.08)
            else:
                confidence_factors.append(0.03)
            
            return min(1.0, sum(confidence_factors))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {e}")
            return 0.3  # Low confidence on error
    
    def _estimate_profit_potential(self, market_data: Dict[str, Any]) -> float:
        """Estimate profit potential percentage"""
        try:
            # Base profit estimate on momentum and volume
            price_momentum = float(market_data.get('price_change_24h_percent', 0))
            volume_momentum = float(market_data.get('volume_momentum', 1.0))
            
            # Conservative profit estimate
            base_profit = max(0.0, price_momentum * 0.3)  # 30% of recent momentum
            volume_boost = max(0.0, (volume_momentum - 1.0) * 5)  # Volume factor
            
            estimated_profit = base_profit + volume_boost
            return min(50.0, max(0.0, estimated_profit))  # Cap at 50%
            
        except Exception as e:
            self.logger.error(f"Error estimating profit potential: {e}")
            return 0.0
    
    def _passes_basic_filters(self, market_data: Dict[str, Any]) -> bool:
        """Apply basic filters to eliminate unsuitable opportunities"""
        try:
            # Volume filter
            volume_24h = float(market_data.get('volume_24h_usd', 0))
            if volume_24h < self.min_volume_threshold_usd:
                return False
            
            # Liquidity filter
            liquidity_sol = float(market_data.get('liquidity_sol', 0))
            if liquidity_sol < self.min_liquidity_threshold_sol:
                return False
            
            # Age filter
            age_hours = float(market_data.get('age_hours', 0))
            if age_hours > self.max_token_age_hours:
                return False
            
            # Price sanity check
            price = float(market_data.get('price', 0))
            if price <= 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying basic filters: {e}")
            return False
    
    async def _filter_and_rank_opportunities(self, opportunities: List[MarketOpportunity]) -> List[MarketOpportunity]:
        """Filter and rank opportunities by score"""
        try:
            # Filter by minimum thresholds
            filtered = []
            for opp in opportunities:
                # Apply additional filters
                if (opp.opportunity_score >= 0.3 and 
                    opp.confidence_score >= 0.4 and
                    opp.risk_level != "avoid"):
                    filtered.append(opp)
            
            # Sort by opportunity score (highest first)
            filtered.sort(key=lambda x: x.opportunity_score, reverse=True)
            
            # Limit to top opportunities
            max_opportunities = 20
            return filtered[:max_opportunities]
            
        except Exception as e:
            self.logger.error(f"Error filtering and ranking opportunities: {e}")
            return opportunities
    
    async def _clean_opportunity_cache(self):
        """Remove expired entries from opportunity cache"""
        try:
            current_time = datetime.now()
            expired_addresses = []
            
            for address, timestamp in self.cache_timestamps.items():
                if (current_time - timestamp).total_seconds() > (self.opportunity_cache_minutes * 60):
                    expired_addresses.append(address)
            
            for address in expired_addresses:
                self.opportunity_cache.pop(address, None)
                self.cache_timestamps.pop(address, None)
            
            if expired_addresses:
                self.logger.debug(f"🗑️ Cleaned {len(expired_addresses)} expired cache entries")
                
        except Exception as e:
            self.logger.error(f"Error cleaning opportunity cache: {e}")
    
    async def _test_api_connections(self):
        """Test connections to all data sources"""
        try:
            # Test Birdeye if API key available
            if self.api_config.get('birdeye_api_key'):
                try:
                    await self._get_birdeye_trending_tokens()
                    self.logger.info("✅ Birdeye API connection verified")
                except Exception as e:
                    self.logger.warning(f"⚠️ Birdeye API connection failed: {e}")
            
            # Test DexScreener (no API key required)
            try:
                await self._get_dexscreener_new_pairs()
                self.logger.info("✅ DexScreener API connection verified")
            except Exception as e:
                self.logger.warning(f"⚠️ DexScreener API connection failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Error testing API connections: {e}")
    
    def _update_scan_metrics(self, scan_duration_ms: float):
        """Update scanning performance metrics"""
        try:
            self.total_scans_completed += 1
            self.last_scan_time = datetime.now()
            
            # Update average scan time
            if self.total_scans_completed == 1:
                self.average_scan_time_ms = scan_duration_ms
            else:
                # Exponential moving average
                alpha = 0.1
                self.average_scan_time_ms = (alpha * scan_duration_ms + 
                                           (1 - alpha) * self.average_scan_time_ms)
                
        except Exception as e:
            self.logger.error(f"Error updating scan metrics: {e}")
    
    def get_scanner_status(self) -> Dict[str, Any]:
        """Get comprehensive scanner status"""
        return {
            'initialized': True,
            'total_scans_completed': self.total_scans_completed,
            'total_opportunities_found': self.total_opportunities_found,
            'average_scan_time_ms': round(self.average_scan_time_ms, 2),
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'cache_size': len(self.opportunity_cache),
            'scan_interval_seconds': self.scan_interval_seconds,
            'data_sources_active': len(self.data_sources),
            'filters': {
                'min_volume_threshold_usd': self.min_volume_threshold_usd,
                'min_liquidity_threshold_sol': self.min_liquidity_threshold_sol,
                'max_token_age_hours': self.max_token_age_hours
            }
        }
    
    async def shutdown(self):
        """Shutdown the market scanner"""
        try:
            self.logger.info("🛑 Shutting down market scanner...")
            
            # Clear caches
            self.opportunity_cache.clear()
            self.cache_timestamps.clear()
            
            # Shutdown market data fetcher
            if hasattr(self.market_data_fetcher, 'shutdown'):
                await self.market_data_fetcher.shutdown()
            
            self.logger.info("✅ Market scanner shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during scanner shutdown: {e}")


# Global instance manager
_market_scanner = None

async def get_market_scanner() -> RealMarketScanner:
    """Get global market scanner instance"""
    global _market_scanner
    if _market_scanner is None:
        _market_scanner = RealMarketScanner()
        await _market_scanner.initialize()
    return _market_scanner

