"""
High-Performance Profitable Token Scanner
========================================

Optimized for 12+ trades/hour with maximum profitability and minimal API costs.
Advanced filtering, honeypot detection, and cost-efficient scanning.
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from worker_ant_v1.config import (
    scanner_config, trading_config, monitoring_config,
    get_dynamic_slippage, TradingMode, deployment_config
)

@dataclass
class TokenOpportunity:
    """Enhanced trading opportunity with profitability metrics"""
    
    # Basic token info
    token_address: str
    token_symbol: str
    token_name: str
    pool_address: str
    dex: str
    
    # Pool metrics
    liquidity_sol: float
    liquidity_usd: float
    pool_age_seconds: int
    volume_24h_usd: float
    price_usd: float
    
    # Additional metrics with defaults
    volume_1h_usd: float = 0.0
    price_change_5m: float = 0.0
    price_change_1h: float = 0.0
    volatility_score: float = 0.0
    
    # Risk indicators
    is_verified: bool = False
    has_metadata: bool = False
    holder_count: int = 0
    top_holder_percent: float = 0.0
    is_honeypot: bool = False
    buy_tax_percent: float = 0.0
    sell_tax_percent: float = 0.0
    liquidity_locked: bool = False
    
    # Profitability metrics
    confidence_score: float = 0.0
    profit_potential: float = 0.0
    risk_score: float = 0.0
    expected_slippage: float = 0.0
    
    # Discovery tracking
    detected_at: float = field(default_factory=time.time)
    source: str = "unknown"
    scan_latency_ms: int = 0


class ProfitOptimizedScanner:
    """High-performance scanner optimized for profitability"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.seen_tokens: set = set()
        self.blacklisted_tokens: set = set(scanner_config.blacklisted_tokens)
        self.performance_cache: Dict[str, float] = {}
        self.last_scan_time = 0.0
        self.api_costs_tracker = {"total": 0.0, "hourly": 0.0}
        
        # Rate limiting
        self.last_api_calls = {
            "raydium": 0.0,
            "orca": 0.0, 
            "dexscreener": 0.0,
            "birdeye": 0.0
        }
        
        self.logger = logging.getLogger("ProfitScanner")
        
    async def start(self):
        """Initialize scanner with optimized session (Windows compatible)"""
        # Windows-compatible connector without DNS caching
        connector = aiohttp.TCPConnector(
            limit=20,  # Connection pool limit
            limit_per_host=10,
            # Removed DNS caching to avoid aiodns dependency on Windows
        )
        
        timeout = aiohttp.ClientTimeout(
            total=5,
            connect=2
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "WorkerAnt/1.0 (High-Performance Trading Bot)"
            }
        )
        
        self.logger.info("Profit-optimized scanner initialized (Windows compatible)")
        
    async def stop(self):
        """Clean shutdown"""
        if self.session:
            await self.session.close()
            
    async def scan_for_profitable_opportunities(self) -> List[TokenOpportunity]:
        """Main scanning function optimized for profit"""
        
        scan_start = time.time()
        
        # Rate limiting check
        if scan_start - self.last_scan_time < scanner_config.scan_interval_seconds:
            await asyncio.sleep(scanner_config.scan_interval_seconds - (scan_start - self.last_scan_time))
        
        opportunities = []
        
        # Parallel scanning for speed
        scan_tasks = []
        
        if scanner_config.use_raydium_api:
            scan_tasks.append(self._scan_raydium_profitable())
            
        if scanner_config.use_orca_api:
            scan_tasks.append(self._scan_orca_profitable())
            
        if scanner_config.use_dexscreener_api:
            scan_tasks.append(self._scan_dexscreener_profitable())
            
        if scanner_config.use_birdeye_api and scanner_config.birdeye_api_key:
            scan_tasks.append(self._scan_birdeye_profitable())
        
        # Execute all scans in parallel
        results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        # Combine results
        for result in results:
            if isinstance(result, list):
                opportunities.extend(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Scan error: {result}")
        
        # Advanced filtering and scoring
        profitable_opportunities = await self._filter_and_optimize_for_profit(opportunities)
        
        # Update tracking
        self.last_scan_time = time.time()
        scan_duration = self.last_scan_time - scan_start
        
        if profitable_opportunities:
            self.logger.info(
                f"Found {len(profitable_opportunities)} profitable opportunities "
                f"in {scan_duration*1000:.0f}ms"
            )
        
        return profitable_opportunities[:scanner_config.max_tokens_per_scan]
    
    async def _scan_raydium_profitable(self) -> List[TokenOpportunity]:
        """Scan Raydium for profitable new pools"""
        
        opportunities = []
        
        try:
            # Rate limiting
            await self._rate_limit_check("raydium", 0.5)
            
            url = "https://api.raydium.io/v2/sdk/liquidity/mainnet.json"
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    current_time = time.time()
                    
                    # Focus on newest pools first
                    pools = data.get('official', []) + data.get('unOfficial', [])
                    pools = sorted(pools, key=lambda x: x.get('openTime', 0), reverse=True)
                    
                    for pool in pools[:20]:  # Top 20 newest
                        try:
                            opportunity = await self._parse_raydium_pool_profitable(pool)
                            if opportunity and self._meets_profitability_criteria(opportunity):
                                opportunities.append(opportunity)
                        except Exception as e:
                            continue
                            
        except Exception as e:
            self.logger.error(f"Raydium scan error: {e}")
            
        return opportunities
    
    async def _scan_orca_profitable(self) -> List[TokenOpportunity]:
        """Scan Orca for profitable opportunities"""
        
        opportunities = []
        
        try:
            await self._rate_limit_check("orca", 0.5)
            
            url = "https://api.orca.so/v1/whirlpool/list"
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    pools = data.get('whirlpools', [])
                    
                    # Sort by TVL and volume for profitability
                    pools = sorted(pools, 
                                 key=lambda x: float(x.get('tvl', 0)) * float(x.get('volume', {}).get('day', 0)),
                                 reverse=True)
                    
                    for pool in pools[:15]:  # Top 15 by liquidity*volume
                        try:
                            opportunity = await self._parse_orca_pool_profitable(pool)
                            if opportunity and self._meets_profitability_criteria(opportunity):
                                opportunities.append(opportunity)
                        except Exception as e:
                            continue
                            
        except Exception as e:
            self.logger.error(f"Orca scan error: {e}")
            
        return opportunities
    
    async def _scan_dexscreener_profitable(self) -> List[TokenOpportunity]:
        """Scan DexScreener for trending profitable tokens"""
        
        opportunities = []
        
        try:
            await self._rate_limit_check("dexscreener", 1.0)
            
            # Focus on trending tokens with good metrics
            url = "https://api.dexscreener.com/latest/dex/tokens/trending"
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    for pair in data.get('pairs', [])[:10]:  # Top 10 trending
                        if pair.get('chainId') != 'solana':
                            continue
                            
                        try:
                            opportunity = await self._parse_dexscreener_profitable(pair)
                            if opportunity and self._meets_profitability_criteria(opportunity):
                                opportunities.append(opportunity)
                        except Exception as e:
                            continue
                            
        except Exception as e:
            self.logger.error(f"DexScreener scan error: {e}")
            
        return opportunities
    
    async def _scan_birdeye_profitable(self) -> List[TokenOpportunity]:
        """Scan Birdeye for high-quality profitable opportunities"""
        
        if not scanner_config.birdeye_api_key:
            return []
            
        opportunities = []
        
        try:
            await self._rate_limit_check("birdeye", 1.0)
            
            headers = {
                "X-API-KEY": scanner_config.birdeye_api_key
            }
            
            # Get trending tokens with good fundamentals
            url = "https://public-api.birdeye.so/defi/trending_tokens/sol"
            
            async with self.session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    for token in data.get('data', {}).get('items', [])[:8]:
                        try:
                            opportunity = await self._parse_birdeye_profitable(token)
                            if opportunity and self._meets_profitability_criteria(opportunity):
                                opportunities.append(opportunity)
                        except Exception as e:
                            continue
                            
                    # Track API costs
                    self.api_costs_tracker["total"] += 0.001  # Estimated cost
                    
        except Exception as e:
            self.logger.error(f"Birdeye scan error: {e}")
            
        return opportunities
    
    async def _parse_raydium_pool_profitable(self, pool_data: Dict) -> Optional[TokenOpportunity]:
        """Parse Raydium pool with profitability focus"""
        
        try:
            base_mint = pool_data.get('baseMint', '')
            quote_mint = pool_data.get('quoteMint', '')
            
            # Ensure SOL is quote token for proper pricing
            if quote_mint != "So11111111111111111111111111111111111112":
                return None
                
            liquidity_sol = float(pool_data.get('quoteReserve', 0)) / 1e9
            open_time = pool_data.get('openTime', 0) / 1000
            pool_age = time.time() - open_time
            
            # Quick profitability filters
            if liquidity_sol < scanner_config.min_liquidity_sol:
                return None
                
            if pool_age < scanner_config.min_pool_age_seconds or pool_age > scanner_config.max_pool_age_seconds:
                return None
            
            return TokenOpportunity(
                token_address=base_mint,
                token_symbol=pool_data.get('lpMint', 'UNK')[:8],
                token_name=f"Raydium-{base_mint[:8]}",
                pool_address=pool_data.get('id', ''),
                dex='raydium',
                liquidity_sol=liquidity_sol,
                liquidity_usd=liquidity_sol * 200,  # Approximate USD
                pool_age_seconds=int(pool_age),
                volume_24h_usd=0.0,  # Not available directly
                price_usd=0.0,
                source='raydium',
                expected_slippage=get_dynamic_slippage(liquidity_sol * 200)
            )
            
        except Exception as e:
            return None
    
    async def _parse_orca_pool_profitable(self, pool_data: Dict) -> Optional[TokenOpportunity]:
        """Parse Orca pool with profitability metrics"""
        
        try:
            token_a = pool_data.get('tokenA', {})
            token_b = pool_data.get('tokenB', {})
            
            # Determine which is the trading token
            if token_b.get('symbol') == 'SOL':
                trading_token = token_a
            elif token_a.get('symbol') == 'SOL':
                trading_token = token_b
            else:
                return None
                
            tvl = float(pool_data.get('tvl', 0))
            volume_24h = float(pool_data.get('volume', {}).get('day', 0))
            
            if tvl < scanner_config.min_liquidity_usd:
                return None
                
            return TokenOpportunity(
                token_address=trading_token.get('mint', ''),
                token_symbol=trading_token.get('symbol', 'UNK'),
                token_name=trading_token.get('name', ''),
                pool_address=pool_data.get('address', ''),
                dex='orca',
                liquidity_sol=tvl / 200,  # Approximate SOL
                liquidity_usd=tvl,
                pool_age_seconds=60,  # Default
                volume_24h_usd=volume_24h,
                price_usd=float(trading_token.get('price', 0)),
                source='orca',
                expected_slippage=get_dynamic_slippage(tvl)
            )
            
        except Exception as e:
            return None
    
    async def _parse_dexscreener_profitable(self, pair_data: Dict) -> Optional[TokenOpportunity]:
        """Parse DexScreener with profitability analysis"""
        
        try:
            base_token = pair_data.get('baseToken', {})
            
            liquidity_usd = float(pair_data.get('liquidity', {}).get('usd', 0))
            volume_24h = float(pair_data.get('volume', {}).get('h24', 0))
            price_change_5m = float(pair_data.get('priceChange', {}).get('m5', 0))
            price_change_1h = float(pair_data.get('priceChange', {}).get('h1', 0))
            
            if liquidity_usd < scanner_config.min_liquidity_usd:
                return None
                
            if volume_24h < scanner_config.min_volume_24h_usd:
                return None
            
            # Calculate volatility score for profit potential
            volatility = abs(price_change_5m) + abs(price_change_1h) / 2
            
            return TokenOpportunity(
                token_address=base_token.get('address', ''),
                token_symbol=base_token.get('symbol', 'UNK'),
                token_name=base_token.get('name', ''),
                pool_address=pair_data.get('pairAddress', ''),
                dex=pair_data.get('dexId', 'unknown'),
                liquidity_sol=liquidity_usd / 200,
                liquidity_usd=liquidity_usd,
                pool_age_seconds=int(time.time() - pair_data.get('pairCreatedAt', 0) / 1000),
                volume_24h_usd=volume_24h,
                price_usd=float(pair_data.get('priceUsd', 0)),
                price_change_5m=price_change_5m,
                price_change_1h=price_change_1h,
                volatility_score=min(volatility / 10, 1.0),  # Normalize to 0-1
                source='dexscreener',
                expected_slippage=get_dynamic_slippage(liquidity_usd)
            )
            
        except Exception as e:
            return None
    
    async def _parse_birdeye_profitable(self, token_data: Dict) -> Optional[TokenOpportunity]:
        """Parse Birdeye with comprehensive profitability metrics"""
        
        try:
            liquidity = float(token_data.get('liquidity', 0))
            volume_24h = float(token_data.get('volume24h', 0))
            price_change_24h = float(token_data.get('priceChange24h', 0))
            
            if liquidity < scanner_config.min_liquidity_usd:
                return None
                
            return TokenOpportunity(
                token_address=token_data.get('address', ''),
                token_symbol=token_data.get('symbol', 'UNK'),
                token_name=token_data.get('name', ''),
                pool_address='',  # Not provided
                dex='birdeye',
                liquidity_sol=liquidity / 200,
                liquidity_usd=liquidity,
                pool_age_seconds=300,  # Default
                volume_24h_usd=volume_24h,
                price_usd=float(token_data.get('price', 0)),
                price_change_1h=price_change_24h / 24,  # Approximate
                is_verified=token_data.get('verified', False),
                source='birdeye',
                expected_slippage=get_dynamic_slippage(liquidity)
            )
            
        except Exception as e:
            return None
    
    def _meets_profitability_criteria(self, opportunity: TokenOpportunity) -> bool:
        """Quick profitability screening"""
        
        # Skip if already seen
        if opportunity.token_address in self.seen_tokens:
            return False
            
        # Skip if blacklisted
        if opportunity.token_address in self.blacklisted_tokens:
            return False
            
        # Basic profitability checks
        if opportunity.liquidity_usd < scanner_config.min_liquidity_usd:
            return False
            
        if opportunity.liquidity_usd > scanner_config.max_liquidity_usd:
            return False  # Too big, likely established token
            
        # Pool age check
        if (opportunity.pool_age_seconds < scanner_config.min_pool_age_seconds or 
            opportunity.pool_age_seconds > scanner_config.max_pool_age_seconds):
            return False
        
        return True
    
    async def _filter_and_optimize_for_profit(self, opportunities: List[TokenOpportunity]) -> List[TokenOpportunity]:
        """Advanced filtering and profitability optimization"""
        
        if not opportunities:
            return []
        
        # Remove duplicates
        seen_addresses = set()
        unique_opportunities = []
        
        for opp in opportunities:
            if opp.token_address not in seen_addresses:
                seen_addresses.add(opp.token_address)
                unique_opportunities.append(opp)
        
        # Enhanced filtering tasks
        filter_tasks = []
        
        for opp in unique_opportunities:
            filter_tasks.append(self._analyze_profitability(opp))
            
        # Run profitability analysis in parallel
        analyzed_opportunities = await asyncio.gather(*filter_tasks, return_exceptions=True)
        
        # Filter successful analyses
        profitable_opportunities = []
        for result in analyzed_opportunities:
            if isinstance(result, TokenOpportunity) and result.confidence_score > 0.3:
                profitable_opportunities.append(result)
                self.seen_tokens.add(result.token_address)
        
        # Sort by profit potential
        profitable_opportunities.sort(
            key=lambda x: (x.confidence_score * x.profit_potential), 
            reverse=True
        )
        
        return profitable_opportunities
    
    async def _analyze_profitability(self, opportunity: TokenOpportunity) -> TokenOpportunity:
        """Comprehensive profitability analysis"""
        
        # Honeypot check if enabled
        if scanner_config.enable_honeypot_check:
            honeypot_result = await self._check_honeypot(opportunity.token_address)
            if honeypot_result['is_honeypot']:
                opportunity.is_honeypot = True
                opportunity.confidence_score = 0.0
                return opportunity
                
            opportunity.buy_tax_percent = honeypot_result.get('buy_tax', 0)
            opportunity.sell_tax_percent = honeypot_result.get('sell_tax', 0)
        
        # Calculate confidence score
        confidence = self._calculate_profit_confidence(opportunity)
        opportunity.confidence_score = confidence
        
        # Calculate profit potential
        profit_potential = self._calculate_profit_potential(opportunity)
        opportunity.profit_potential = profit_potential
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(opportunity)
        opportunity.risk_score = risk_score
        
        return opportunity
    
    def _calculate_profit_confidence(self, opp: TokenOpportunity) -> float:
        """Calculate confidence score optimized for profitability"""
        
        score = 0.0
        
        # Liquidity score (0-0.4) - Higher weight for profitability
        if opp.liquidity_usd >= 50000:
            score += 0.4
        elif opp.liquidity_usd >= 20000:
            score += 0.3
        elif opp.liquidity_usd >= 10000:
            score += 0.2
        elif opp.liquidity_usd >= 5000:
            score += 0.1
        
        # Volume score (0-0.3) - Volume indicates trading activity
        if opp.volume_24h_usd >= 50000:
            score += 0.3
        elif opp.volume_24h_usd >= 20000:
            score += 0.2
        elif opp.volume_24h_usd >= 10000:
            score += 0.1
        
        # Pool age score (0-0.2) - Sweet spot for memecoins
        if 60 <= opp.pool_age_seconds <= 180:  # 1-3 minutes
            score += 0.2
        elif 30 <= opp.pool_age_seconds <= 300:  # 30s-5min
            score += 0.1
        
        # DEX preference (0-0.1)
        if opp.dex in ['raydium', 'orca']:
            score += 0.1
            
        # Volatility bonus for profit potential
        if 0.3 <= opp.volatility_score <= 0.7:  # Good volatility range
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_profit_potential(self, opp: TokenOpportunity) -> float:
        """Calculate expected profit potential"""
        
        base_potential = 0.5  # 50% base
        
        # Volatility increases profit potential
        volatility_bonus = opp.volatility_score * 0.3
        
        # Volume/liquidity ratio indicates momentum
        if opp.volume_24h_usd > 0 and opp.liquidity_usd > 0:
            volume_ratio = opp.volume_24h_usd / opp.liquidity_usd
            momentum_bonus = min(volume_ratio * 0.2, 0.3)
        else:
            momentum_bonus = 0
        
        # Price trend bonus
        price_trend_bonus = 0
        if opp.price_change_5m > 5:  # Strong upward momentum
            price_trend_bonus = 0.2
        elif opp.price_change_5m > 0:  # Positive momentum
            price_trend_bonus = 0.1
            
        return min(base_potential + volatility_bonus + momentum_bonus + price_trend_bonus, 1.0)
    
    def _calculate_risk_score(self, opp: TokenOpportunity) -> float:
        """Calculate risk score (lower is better)"""
        
        risk = 0.0
        
        # Honeypot risk
        if opp.is_honeypot:
            return 1.0  # Maximum risk
        
        # Tax risk
        if opp.buy_tax_percent > scanner_config.max_buy_tax_percent:
            risk += 0.3
        if opp.sell_tax_percent > scanner_config.max_sell_tax_percent:
            risk += 0.3
        
        # Liquidity risk
        if opp.liquidity_usd < 10000:
            risk += 0.2
        elif opp.liquidity_usd < 5000:
            risk += 0.4
        
        # Age risk
        if opp.pool_age_seconds < 30:  # Too new
            risk += 0.2
        elif opp.pool_age_seconds > 600:  # Too old for memecoin
            risk += 0.1
        
        return min(risk, 1.0)
    
    async def _check_honeypot(self, token_address: str) -> Dict:
        """Check if token is a honeypot"""
        
        try:
            # Simulate small buy/sell to check taxes and honeypot
            # This is a simplified check - in production, use specialized APIs
            
            url = f"https://api.honeypot.is/v1/GetHoneypotStatus?address={token_address}"
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        'is_honeypot': data.get('IsHoneypot', False),
                        'buy_tax': float(data.get('BuyTax', 0)),
                        'sell_tax': float(data.get('SellTax', 0))
                    }
        except Exception as e:
            self.logger.debug(f"Honeypot check failed for {token_address}: {e}")
        
        # Conservative default
        return {'is_honeypot': False, 'buy_tax': 0, 'sell_tax': 0}
    
    async def _rate_limit_check(self, api_name: str, min_interval: float):
        """Enforce rate limiting for API calls"""
        
        current_time = time.time()
        last_call = self.last_api_calls.get(api_name, 0)
        
        if current_time - last_call < min_interval:
            await asyncio.sleep(min_interval - (current_time - last_call))
        
        self.last_api_calls[api_name] = time.time()
    
    def add_to_blacklist(self, token_address: str, reason: str = "performance"):
        """Add token to blacklist"""
        self.blacklisted_tokens.add(token_address)
        if scanner_config.auto_blacklist_rugs:
            scanner_config.blacklisted_tokens.append(token_address)
        
        self.logger.info(f"Blacklisted {token_address}: {reason}")
    
    def get_cost_summary(self) -> Dict:
        """Get API cost summary"""
        return {
            "total_cost": self.api_costs_tracker["total"],
            "hourly_cost": self.api_costs_tracker["hourly"],
            "cost_per_scan": self.api_costs_tracker["total"] / max(1, len(self.seen_tokens))
        }


# Global scanner instance
profit_scanner = ProfitOptimizedScanner() 