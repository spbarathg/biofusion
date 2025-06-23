"""
Token Scanner for Worker Ant V1
===============================

Monitors Raydium/Orca for new token listings and filters for tradeable opportunities.
"""

import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

from .config import scanner_config
from .logger import trading_logger


@dataclass
class TokenOpportunity:
    """A detected trading opportunity"""
    
    token_address: str
    token_symbol: str
    token_name: str
    pool_address: str
    dex: str  # 'raydium' or 'orca'
    
    # Pool metrics
    liquidity_sol: float
    pool_age_seconds: int
    volume_24h_sol: float
    price_usd: float
    
    # Risk indicators
    is_verified: bool
    has_metadata: bool
    holder_count: int
    
    # Discovery
    detected_at: float
    confidence_score: float  # 0-1 scoring for trade worthiness


class TokenScanner:
    """Main token scanning engine"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.seen_tokens: Set[str] = set()
        self.last_scan_time = 0
        self.scanning = False
        
    async def start(self):
        """Start the token scanner"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        self.scanning = True
        trading_logger.logger.info("Token scanner started")
        
    async def stop(self):
        """Stop the token scanner"""
        self.scanning = False
        if self.session:
            await self.session.close()
        trading_logger.logger.info("Token scanner stopped")
        
    async def scan_for_opportunities(self) -> List[TokenOpportunity]:
        """Main scanning function - returns new trading opportunities"""
        
        if not self.scanning or not self.session:
            return []
            
        opportunities = []
        
        try:
            # Scan different sources in parallel
            tasks = []
            
            if scanner_config.use_birdeye_api:
                tasks.append(self._scan_birdeye())
            if scanner_config.use_dexscreener_api:
                tasks.append(self._scan_dexscreener())
                
            # Add Raydium and Orca direct scanning
            tasks.append(self._scan_raydium_new_pools())
            tasks.append(self._scan_orca_new_pools())
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            for result in results:
                if isinstance(result, list):
                    opportunities.extend(result)
                elif isinstance(result, Exception):
                    trading_logger.logger.error(f"Scanner error: {result}")
                    
        except Exception as e:
            trading_logger.logger.error(f"Error in scan_for_opportunities: {e}")
            
        # Filter and score opportunities
        filtered_opportunities = self._filter_and_score(opportunities)
        
        # Update tracking
        self.last_scan_time = time.time()
        
        return filtered_opportunities[:scanner_config.max_tokens_per_scan]
        
    async def _scan_birdeye(self) -> List[TokenOpportunity]:
        """Scan Birdeye API for new tokens"""
        
        if not scanner_config.birdeye_api_key:
            return []
            
        opportunities = []
        
        try:
            headers = {
                'X-API-KEY': scanner_config.birdeye_api_key,
                'Content-Type': 'application/json'
            }
            
            # Get new tokens from Birdeye
            url = 'https://public-api.birdeye.so/public/tokenlist'
            params = {
                'sort_by': 'created_at',
                'sort_type': 'desc',
                'limit': 50,
                'min_liquidity': int(scanner_config.min_pool_age_seconds * 1000)  # Convert to USD
            }
            
            async with self.session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    for token in data.get('data', {}).get('tokens', []):
                        opportunity = self._parse_birdeye_token(token)
                        if opportunity:
                            opportunities.append(opportunity)
                            
        except Exception as e:
            trading_logger.logger.error(f"Birdeye API error: {e}")
            
        return opportunities
        
    async def _scan_dexscreener(self) -> List[TokenOpportunity]:
        """Scan DexScreener API for new tokens"""
        
        opportunities = []
        
        try:
            # Get latest tokens from DexScreener
            url = 'https://api.dexscreener.com/latest/dex/tokens'
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    for pair in data.get('pairs', []):
                        if pair.get('chainId') == 'solana':
                            opportunity = self._parse_dexscreener_pair(pair)
                            if opportunity:
                                opportunities.append(opportunity)
                                
        except Exception as e:
            trading_logger.logger.error(f"DexScreener API error: {e}")
            
        return opportunities
        
    async def _scan_raydium_new_pools(self) -> List[TokenOpportunity]:
        """Scan Raydium for new liquidity pools"""
        
        opportunities = []
        
        try:
            # Raydium API endpoint for new pools
            url = 'https://api.raydium.io/v2/sdk/liquidity/mainnet.json'
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    current_time = time.time()
                    
                    for pool in data.get('official', []):
                        # Check if pool is new enough
                        pool_creation = pool.get('openTime', 0) / 1000  # Convert from ms
                        pool_age = current_time - pool_creation
                        
                        if (scanner_config.min_pool_age_seconds <= pool_age <= 
                            scanner_config.max_pool_age_seconds):
                            
                            opportunity = self._parse_raydium_pool(pool)
                            if opportunity:
                                opportunities.append(opportunity)
                                
        except Exception as e:
            trading_logger.logger.error(f"Raydium API error: {e}")
            
        return opportunities
        
    async def _scan_orca_new_pools(self) -> List[TokenOpportunity]:
        """Scan Orca for new liquidity pools"""
        
        opportunities = []
        
        try:
            # Orca API endpoint
            url = 'https://api.orca.so/v1/whirlpool/list'
            
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    current_time = time.time()
                    
                    for pool in data.get('whirlpools', []):
                        # Parse pool creation time and check age
                        opportunity = self._parse_orca_pool(pool)
                        if opportunity:
                            pool_age = current_time - opportunity.detected_at
                            if (scanner_config.min_pool_age_seconds <= pool_age <= 
                                scanner_config.max_pool_age_seconds):
                                opportunities.append(opportunity)
                                
        except Exception as e:
            trading_logger.logger.error(f"Orca API error: {e}")
            
        return opportunities
        
    def _parse_birdeye_token(self, token_data: Dict) -> Optional[TokenOpportunity]:
        """Parse Birdeye token data into TokenOpportunity"""
        
        try:
            return TokenOpportunity(
                token_address=token_data.get('address', ''),
                token_symbol=token_data.get('symbol', 'UNKNOWN'),
                token_name=token_data.get('name', ''),
                pool_address=token_data.get('pool_address', ''),
                dex='birdeye',
                liquidity_sol=float(token_data.get('liquidity', 0)) / 1e9,  # Convert to SOL
                pool_age_seconds=int(time.time() - token_data.get('created_at', 0)),
                volume_24h_sol=float(token_data.get('volume_24h', 0)) / 1e9,
                price_usd=float(token_data.get('price', 0)),
                is_verified=token_data.get('verified', False),
                has_metadata=bool(token_data.get('metadata')),
                holder_count=token_data.get('holder_count', 0),
                detected_at=time.time(),
                confidence_score=0.0  # Will be calculated later
            )
        except Exception as e:
            trading_logger.logger.debug(f"Error parsing Birdeye token: {e}")
            return None
            
    def _parse_dexscreener_pair(self, pair_data: Dict) -> Optional[TokenOpportunity]:
        """Parse DexScreener pair data into TokenOpportunity"""
        
        try:
            base_token = pair_data.get('baseToken', {})
            
            return TokenOpportunity(
                token_address=base_token.get('address', ''),
                token_symbol=base_token.get('symbol', 'UNKNOWN'),
                token_name=base_token.get('name', ''),
                pool_address=pair_data.get('pairAddress', ''),
                dex=pair_data.get('dexId', 'unknown'),
                liquidity_sol=float(pair_data.get('liquidity', {}).get('usd', 0)) / 100,  # Rough SOL conversion
                pool_age_seconds=int(time.time() - pair_data.get('pairCreatedAt', 0) / 1000),
                volume_24h_sol=float(pair_data.get('volume', {}).get('h24', 0)) / 100,
                price_usd=float(pair_data.get('priceUsd', 0)),
                is_verified=False,  # DexScreener doesn't provide this
                has_metadata=bool(base_token.get('name')),
                holder_count=0,  # Not available
                detected_at=time.time(),
                confidence_score=0.0
            )
        except Exception as e:
            trading_logger.logger.debug(f"Error parsing DexScreener pair: {e}")
            return None
            
    def _parse_raydium_pool(self, pool_data: Dict) -> Optional[TokenOpportunity]:
        """Parse Raydium pool data into TokenOpportunity"""
        
        try:
            base_mint = pool_data.get('baseMint', '')
            base_decimals = pool_data.get('baseDecimals', 9)
            
            return TokenOpportunity(
                token_address=base_mint,
                token_symbol=pool_data.get('lpMint', 'UNKNOWN')[:8],  # Abbreviated
                token_name='',
                pool_address=pool_data.get('id', ''),
                dex='raydium',
                liquidity_sol=float(pool_data.get('quoteReserve', 0)) / 1e9,
                pool_age_seconds=int(time.time() - pool_data.get('openTime', 0) / 1000),
                volume_24h_sol=0.0,  # Not provided directly
                price_usd=0.0,  # Calculate from reserves
                is_verified=False,
                has_metadata=False,
                holder_count=0,
                detected_at=time.time(),
                confidence_score=0.0
            )
        except Exception as e:
            trading_logger.logger.debug(f"Error parsing Raydium pool: {e}")
            return None
            
    def _parse_orca_pool(self, pool_data: Dict) -> Optional[TokenOpportunity]:
        """Parse Orca pool data into TokenOpportunity"""
        
        try:
            return TokenOpportunity(
                token_address=pool_data.get('tokenA', {}).get('mint', ''),
                token_symbol=pool_data.get('tokenA', {}).get('symbol', 'UNKNOWN'),
                token_name=pool_data.get('tokenA', {}).get('name', ''),
                pool_address=pool_data.get('address', ''),
                dex='orca',
                liquidity_sol=float(pool_data.get('tvl', 0)) / 100,  # Rough conversion
                pool_age_seconds=60,  # Default since Orca doesn't provide creation time
                volume_24h_sol=float(pool_data.get('volume', {}).get('day', 0)) / 100,
                price_usd=float(pool_data.get('tokenA', {}).get('price', 0)),
                is_verified=False,
                has_metadata=bool(pool_data.get('tokenA', {}).get('name')),
                holder_count=0,
                detected_at=time.time(),
                confidence_score=0.0
            )
        except Exception as e:
            trading_logger.logger.debug(f"Error parsing Orca pool: {e}")
            return None
            
    def _filter_and_score(self, opportunities: List[TokenOpportunity]) -> List[TokenOpportunity]:
        """Filter opportunities and assign confidence scores"""
        
        filtered = []
        
        for opp in opportunities:
            # Skip if already seen
            if opp.token_address in self.seen_tokens:
                continue
                
            # Skip blacklisted tokens
            if opp.token_address in scanner_config.blacklisted_tokens:
                continue
                
            # Basic filtering
            if opp.liquidity_sol < 5.0:  # Minimum liquidity
                continue
                
            if not opp.token_symbol or opp.token_symbol == 'UNKNOWN':
                continue
                
            # Calculate confidence score
            confidence = self._calculate_confidence_score(opp)
            opp.confidence_score = confidence
            
            # Only include if confidence is above threshold
            if confidence > 0.3:  # 30% minimum confidence
                filtered.append(opp)
                self.seen_tokens.add(opp.token_address)
                
        # Sort by confidence score (highest first)
        filtered.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return filtered
        
    def _calculate_confidence_score(self, opp: TokenOpportunity) -> float:
        """Calculate confidence score for trading opportunity (0-1)"""
        
        score = 0.0
        
        # Liquidity score (0-0.3)
        if opp.liquidity_sol >= 50:
            score += 0.3
        elif opp.liquidity_sol >= 20:
            score += 0.2
        elif opp.liquidity_sol >= 5:
            score += 0.1
            
        # Metadata score (0-0.2)
        if opp.has_metadata:
            score += 0.1
        if opp.is_verified:
            score += 0.1
            
        # Volume score (0-0.2)
        if opp.volume_24h_sol >= 10:
            score += 0.2
        elif opp.volume_24h_sol >= 5:
            score += 0.1
            
        # Pool age score (0-0.2)
        if 30 <= opp.pool_age_seconds <= 300:  # Sweet spot: 30s to 5min
            score += 0.2
        elif 10 <= opp.pool_age_seconds <= 600:  # Acceptable: 10s to 10min
            score += 0.1
            
        # DEX preference (0-0.1)
        if opp.dex in ['raydium', 'orca']:
            score += 0.1
            
        return min(score, 1.0)


# Global scanner instance
token_scanner = TokenScanner() 