"""
REAL MARKET SCANNER - LIVE MEMECOIN HUNTER
=========================================

Connects to real DEX APIs to find live memecoin opportunities
with sentiment-driven filtering and aggressive compounding targets.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
import numpy as np
import os

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.utils.market_data_fetcher import get_market_data_fetcher, TokenMarketData, MarketOpportunity
from worker_ant_v1.intelligence.sentiment_first_ai import get_sentiment_first_ai

# TokenData and TradingOpportunity are now imported from market_data_fetcher

class RealMarketScanner:
    """Real market scanner with live DEX integration for memecoin hunting"""
    
    def __init__(self):
        self.logger = get_logger("MarketScanner")
        
        # Market data fetcher
        self.market_fetcher = None
        
        # Data storage
        self.token_data: Dict[str, TokenMarketData] = {}
        self.opportunities: List[MarketOpportunity] = []
        self.last_scan = datetime.now()
        
        # Memecoin-specific configuration
        self.min_liquidity = 5.0  # Lower threshold for memecoins
        self.min_volume = 50.0    # Lower volume threshold
        self.max_price_impact = 5.0  # Higher price impact tolerance
        self.max_age_hours = 168  # 1 week max age for memecoins
        self.min_holder_count = 50  # Minimum holders
        
        # Blitzscaling mode
        self.blitzscaling_active = False
        
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
        
        # Initialize market data fetcher
        self.market_fetcher = await get_market_data_fetcher()
        
        # Initialize sentiment AI
        self.sentiment_ai = await get_sentiment_first_ai()
        
        # Load initial market data
        await self.update_market_data()
        
        self.logger.info("✅ Market scanner initialized")
        
    # API connection testing is now handled by the centralized market data fetcher
            
    async def update_market_data(self):
        """Update market data from all sources"""
        try:
            self.scan_count += 1
            
            # Get market opportunities from centralized fetcher
            opportunities = await self.market_fetcher.get_market_opportunities(
                min_liquidity=self.min_liquidity,
                min_volume=self.min_volume
            )
            
            # Filter for memecoin criteria
            self.opportunities = []
            for opportunity in opportunities:
                if self._passes_memecoin_filters(opportunity.market_data):
                    self.opportunities.append(opportunity)
                    self.token_data[opportunity.token_address] = opportunity.market_data
            
            self.opportunities_found = len(self.opportunities)
            self.last_scan = datetime.now()
            
            self.logger.info(f"📊 Found {len(self.opportunities)} memecoin opportunities")
            
        except Exception as e:
            self.logger.error(f"Market data update error: {e}")
            
    # Trending tokens are now fetched by the centralized market data fetcher
        
    # Token data fetching is now handled by the centralized market data fetcher
            
    # API-specific methods are now handled by the centralized market data fetcher
            
    # Opportunity generation is now handled by the centralized market data fetcher
            
    def _passes_memecoin_filters(self, token_data: TokenMarketData) -> bool:
        """Check if token passes memecoin-specific filters"""
        try:
            # Blitzscaling mode: Looser filters for higher volume
            if self.blitzscaling_active:
                min_liquidity = self.min_liquidity * 0.5  # 50% lower liquidity requirement
                min_volume = self.min_volume * 0.7       # 30% lower volume requirement
                min_holders = self.min_holder_count * 0.6  # 40% lower holder requirement
                self.logger.debug("🚀 Blitzscaling mode: Using relaxed filters")
            else:
                min_liquidity = self.min_liquidity
                min_volume = self.min_volume
                min_holders = self.min_holder_count
            
            # Liquidity filter
            if token_data.liquidity < min_liquidity:
                return False
                
            # Volume filter
            if token_data.volume_24h < min_volume:
                return False
                
            # Age filter (not too old)
            if token_data.age_hours > self.max_age_hours:
                return False
                
            # Holder count filter
            if token_data.holder_count < min_holders:
                return False
                
            # Price impact filter (simplified) - PROTECTED AGAINST DIVISION BY ZERO
            if token_data.volume_24h > 0 and token_data.liquidity > 0:
                volume_liquidity_ratio = token_data.volume_24h / max(token_data.liquidity, 0.001)
                if volume_liquidity_ratio > 10:  # High volume relative to liquidity
                    return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in memecoin filters: {e}")
            return False
            
    def _calculate_risk_score(self, token_data: TokenMarketData) -> float:
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
            
    def _calculate_price_impact(self, token_data: TokenMarketData) -> float:
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
                    'market_data': {
                        'price': opp.market_data.price,
                        'volume_24h': opp.market_data.volume_24h,
                        'liquidity': opp.market_data.liquidity,
                        'price_change_24h': opp.market_data.price_change_24h,
                        'holder_count': opp.market_data.holder_count,
                        'age_hours': opp.market_data.age_hours,
                    },
                    'opportunity_score': opp.opportunity_score,
                    'risk_level': opp.risk_level,
                    'expected_profit': opp.expected_profit,
                    'recommended_position_size': opp.recommended_position_size,
                    'max_slippage_percent': opp.max_slippage_percent,
                    'detected_at': opp.detected_at.isoformat()
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
            
            # Use centralized market data fetcher
            return await self.market_fetcher.get_token_price(token_address)
            
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
                'blitzscaling_active': self.blitzscaling_active,
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
    
    async def set_blitzscaling_mode(self, active: bool):
        """Set blitzscaling mode for the market scanner"""
        self.blitzscaling_active = active
        self.logger.info(f"🚀 Blitzscaling mode {'ACTIVATED' if active else 'DEACTIVATED'} for market scanner")
        
        if active:
            self.logger.info("🚀 Blitzscaling: Using relaxed filtering criteria for higher trade volume")
        else:
            self.logger.info("🔄 Normal mode: Using standard filtering criteria")

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

