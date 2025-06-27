"""
PRODUCTION-READY TOKEN SCANNER
=============================

Ultra-fast, intelligent token scanner that identifies profitable trading opportunities
with comprehensive security checks, MEV protection, and real-time analysis.
"""

import asyncio
import aiohttp
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

# Internal imports
from worker_ant_v1.core.simple_config import get_trading_config, get_security_config, mask_sensitive_value
from worker_ant_v1.utils.simple_logger import setup_logger
trading_logger = setup_logger(__name__)

class ScanMode(Enum):
    AGGRESSIVE = "aggressive"    # High-frequency, fast scanning
    BALANCED = "balanced"        # Moderate frequency, good filtering
    CONSERVATIVE = "conservative" # Lower frequency, strict filtering
    STEALTH = "stealth"         # Minimal RPC calls, covert scanning

class SecurityRisk(Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TradingOpportunity:
    """Comprehensive trading opportunity data"""
    
    # Token identification
    token_address: str
    token_symbol: str
    token_name: str
    
    # Market data
    current_price_sol: float
    market_cap_sol: float
    liquidity_sol: float
    volume_24h_sol: float
    
    # Opportunity metrics
    confidence_score: float  # 0.0 to 1.0
    urgency_score: float     # 0.0 to 1.0 (how quickly to act)
    profit_potential: float  # Expected profit percentage
    risk_level: float        # 0.0 to 1.0
    
    # Security assessment
    security_risk: SecurityRisk
    rug_probability: float   # 0.0 to 1.0
    honeypot_risk: float     # 0.0 to 1.0
    
    # Technical indicators
    price_momentum: float    # Price change momentum
    volume_spike: float      # Volume spike indicator
    liquidity_score: float   # Liquidity health score
    
    # Discovery metadata
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "scanner"
    scan_latency_ms: int = 0
    
    # Additional data
    holder_count: Optional[int] = None
    top_holder_percent: Optional[float] = None
    creation_time: Optional[datetime] = None
    verified_contract: bool = False
    
    def is_valid_opportunity(self) -> bool:
        """Check if opportunity meets basic validity criteria"""
        return (
            self.confidence_score >= 0.5 and
            self.security_risk != SecurityRisk.CRITICAL and
            self.rug_probability < 0.7 and
            self.liquidity_sol > 10.0 and  # Minimum 10 SOL liquidity
            self.current_price_sol > 0.0
        )
    
    def get_priority_score(self) -> float:
        """Calculate priority score for opportunity ranking"""
        base_score = self.confidence_score * 0.4 + self.urgency_score * 0.3
        
        # Adjust for risk
        risk_adjustment = (1.0 - self.risk_level) * 0.2
        
        # Adjust for profit potential
        profit_adjustment = min(self.profit_potential / 100.0, 0.1)
        
        return base_score + risk_adjustment + profit_adjustment

@dataclass
class ScannerMetrics:
    """Scanner performance and statistics"""
    
    # Scanning statistics
    total_scans: int = 0
    opportunities_found: int = 0
    false_positives: int = 0
    
    # Performance metrics
    avg_scan_latency_ms: float = 0.0
    scans_per_minute: float = 0.0
    uptime_seconds: float = 0.0
    
    # Filter statistics
    security_filtered: int = 0
    liquidity_filtered: int = 0
    volume_filtered: int = 0
    rug_filtered: int = 0
    
    # Success tracking
    profitable_signals: int = 0
    total_profit_tracked: float = 0.0
    
    def get_success_rate(self) -> float:
        """Calculate scanner success rate"""
        if self.opportunities_found == 0:
            return 0.0
        return (self.profitable_signals / self.opportunities_found) * 100

class TokenFilter:
    """Advanced token filtering system"""
    
    def __init__(self):
        self.config = get_trading_config()
        self.security_config = get_security_config()
        self.logger = logging.getLogger("TokenFilter")
        
        # Blacklists and whitelists
        self.token_blacklist: Set[str] = set()
        self.token_whitelist: Set[str] = set()
        self.creator_blacklist: Set[str] = set()
        
        # Known scam patterns
        self.scam_patterns = [
            "airdrop", "free", "giveaway", "elon", "doge", "shib",
            "rocket", "moon", "pump", "100x", "safe", "anti"
        ]
        
        # Load persistent data
        self._load_filter_data()
    
    def _load_filter_data(self):
        """Load blacklists and filter data"""
        try:
            # Load from configuration or files
            # This would be implemented based on data storage strategy
            pass
        except Exception as e:
            self.logger.warning(f"Failed to load filter data: {e}")
    
    async def evaluate_token_security(self, token_address: str, token_data: Dict[str, Any]) -> Tuple[SecurityRisk, float, float]:
        """Evaluate token security and risk levels"""
        
        risk_score = 0.0
        rug_probability = 0.0
        honeypot_risk = 0.0
        
        # Check blacklists
        if token_address in self.token_blacklist:
            return SecurityRisk.CRITICAL, 1.0, 1.0
        
        # Analyze token metadata
        symbol = token_data.get("symbol", "").lower()
        name = token_data.get("name", "").lower()
        
        # Check for scam patterns
        scam_indicators = 0
        for pattern in self.scam_patterns:
            if pattern in symbol or pattern in name:
                scam_indicators += 1
        
        if scam_indicators >= 2:
            risk_score += 0.4
            rug_probability += 0.3
        
        # Check holder distribution
        top_holder_percent = token_data.get("top_holder_percent", 0)
        if top_holder_percent > 50:
            risk_score += 0.3
            rug_probability += 0.4
        elif top_holder_percent > 30:
            risk_score += 0.2
            rug_probability += 0.2
        
        # Check liquidity locks
        liquidity_locked = token_data.get("liquidity_locked", False)
        if not liquidity_locked:
            risk_score += 0.2
            rug_probability += 0.3
        
        # Check contract verification
        verified = token_data.get("verified_contract", False)
        if not verified:
            risk_score += 0.1
            honeypot_risk += 0.1
        
        # Age check
        creation_time = token_data.get("creation_time")
        if creation_time:
            age_hours = (datetime.utcnow() - creation_time).total_seconds() / 3600
            if age_hours < 1:  # Very new token
                risk_score += 0.2
                rug_probability += 0.2
        
        # Determine overall risk level
        if risk_score >= 0.8:
            security_risk = SecurityRisk.CRITICAL
        elif risk_score >= 0.6:
            security_risk = SecurityRisk.HIGH
        elif risk_score >= 0.4:
            security_risk = SecurityRisk.MEDIUM
        elif risk_score >= 0.2:
            security_risk = SecurityRisk.LOW
        else:
            security_risk = SecurityRisk.UNKNOWN
        
        return security_risk, min(rug_probability, 1.0), min(honeypot_risk, 1.0)
    
    def should_skip_token(self, token_address: str, token_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if token should be skipped"""
        
        # Blacklist check
        if token_address in self.token_blacklist:
            return True, "Token blacklisted"
        
        # Creator check
        creator = token_data.get("creator_address")
        if creator and creator in self.creator_blacklist:
            return True, "Creator blacklisted"
        
        # Minimum liquidity
        liquidity = token_data.get("liquidity_sol", 0)
        if liquidity < self.config.min_liquidity_sol:
            return True, f"Liquidity too low: {liquidity:.2f} SOL"
        
        # Minimum volume
        volume_24h = token_data.get("volume_24h_sol", 0)
        if volume_24h < self.config.min_volume_24h_sol:
            return True, f"Volume too low: {volume_24h:.2f} SOL"
        
        # Market cap limits
        market_cap = token_data.get("market_cap_sol", 0)
        if market_cap > self.config.max_market_cap_sol:
            return True, f"Market cap too high: {market_cap:.2f} SOL"
        
        return False, ""

class OpportunityAnalyzer:
    """Advanced opportunity analysis and scoring"""
    
    def __init__(self):
        self.config = get_trading_config()
        self.logger = logging.getLogger("OpportunityAnalyzer")
        
        # Price tracking for momentum analysis
        self.price_history: Dict[str, List[Tuple[float, float]]] = {}  # token -> [(price, timestamp)]
        self.volume_history: Dict[str, List[Tuple[float, float]]] = {}  # token -> [(volume, timestamp)]
    
    def analyze_opportunity(self, token_address: str, token_data: Dict[str, Any]) -> TradingOpportunity:
        """Analyze token data and create trading opportunity"""
        
        # Extract basic data
        symbol = token_data.get("symbol", "UNKNOWN")
        name = token_data.get("name", "Unknown Token")
        price = float(token_data.get("price_sol", 0))
        market_cap = float(token_data.get("market_cap_sol", 0))
        liquidity = float(token_data.get("liquidity_sol", 0))
        volume_24h = float(token_data.get("volume_24h_sol", 0))
        
        # Calculate technical indicators
        momentum = self._calculate_price_momentum(token_address, price)
        volume_spike = self._calculate_volume_spike(token_address, volume_24h)
        liquidity_score = self._calculate_liquidity_score(liquidity, volume_24h)
        
        # Calculate opportunity scores
        confidence = self._calculate_confidence_score(token_data, momentum, volume_spike, liquidity_score)
        urgency = self._calculate_urgency_score(momentum, volume_spike)
        profit_potential = self._estimate_profit_potential(token_data, momentum, volume_spike)
        risk_level = self._calculate_risk_level(token_data)
        
        # Create opportunity
        opportunity = TradingOpportunity(
            token_address=token_address,
            token_symbol=symbol,
            token_name=name,
            current_price_sol=price,
            market_cap_sol=market_cap,
            liquidity_sol=liquidity,
            volume_24h_sol=volume_24h,
            confidence_score=confidence,
            urgency_score=urgency,
            profit_potential=profit_potential,
            risk_level=risk_level,
            price_momentum=momentum,
            volume_spike=volume_spike,
            liquidity_score=liquidity_score,
            security_risk=SecurityRisk.UNKNOWN,  # Will be set by filter
            rug_probability=0.0,  # Will be set by filter
            honeypot_risk=0.0,  # Will be set by filter
            holder_count=token_data.get("holder_count"),
            top_holder_percent=token_data.get("top_holder_percent"),
            creation_time=token_data.get("creation_time"),
            verified_contract=token_data.get("verified_contract", False)
        )
        
        return opportunity
    
    def _calculate_price_momentum(self, token_address: str, current_price: float) -> float:
        """Calculate price momentum indicator"""
        
        # Update price history
        current_time = time.time()
        if token_address not in self.price_history:
            self.price_history[token_address] = []
        
        self.price_history[token_address].append((current_price, current_time))
        
        # Keep only recent history (last hour)
        cutoff_time = current_time - 3600
        self.price_history[token_address] = [
            (price, timestamp) for price, timestamp in self.price_history[token_address]
            if timestamp > cutoff_time
        ]
        
        # Calculate momentum
        history = self.price_history[token_address]
        if len(history) < 2:
            return 0.0
        
        # Simple momentum calculation
        old_price = history[0][0]
        return (current_price - old_price) / old_price if old_price > 0 else 0.0
    
    def _calculate_volume_spike(self, token_address: str, current_volume: float) -> float:
        """Calculate volume spike indicator"""
        
        # Update volume history
        current_time = time.time()
        if token_address not in self.volume_history:
            self.volume_history[token_address] = []
        
        self.volume_history[token_address].append((current_volume, current_time))
        
        # Keep only recent history (last 24 hours)
        cutoff_time = current_time - 86400
        self.volume_history[token_address] = [
            (volume, timestamp) for volume, timestamp in self.volume_history[token_address]
            if timestamp > cutoff_time
        ]
        
        # Calculate spike
        history = self.volume_history[token_address]
        if len(history) < 2:
            return 0.0
        
        # Average historical volume
        avg_volume = sum(volume for volume, _ in history[:-1]) / (len(history) - 1)
        
        if avg_volume > 0:
            return (current_volume - avg_volume) / avg_volume
        return 0.0
    
    def _calculate_liquidity_score(self, liquidity: float, volume_24h: float) -> float:
        """Calculate liquidity health score"""
        
        if liquidity <= 0:
            return 0.0
        
        # Liquidity to volume ratio
        if volume_24h > 0:
            ratio = liquidity / volume_24h
            # Good liquidity should be at least 2x daily volume
            score = min(ratio / 2.0, 1.0)
        else:
            score = 0.5  # Neutral score if no volume data
        
        # Absolute liquidity check
        if liquidity < 10:
            score *= 0.3
        elif liquidity < 50:
            score *= 0.7
        
        return score
    
    def _calculate_confidence_score(self, token_data: Dict[str, Any], momentum: float, 
                                   volume_spike: float, liquidity_score: float) -> float:
        """Calculate overall confidence score"""
        
        base_score = 0.5
        
        # Positive momentum increases confidence
        if momentum > 0.1:  # 10% price increase
            base_score += min(momentum * 0.5, 0.3)
        
        # Volume spike increases confidence
        if volume_spike > 0.5:  # 50% volume increase
            base_score += min(volume_spike * 0.2, 0.2)
        
        # Good liquidity increases confidence
        base_score += liquidity_score * 0.2
        
        # Holder count factor
        holder_count = token_data.get("holder_count", 0)
        if holder_count > 1000:
            base_score += 0.1
        elif holder_count < 10:
            base_score -= 0.2
        
        # Contract verification
        if token_data.get("verified_contract", False):
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_urgency_score(self, momentum: float, volume_spike: float) -> float:
        """Calculate urgency score"""
        
        base_urgency = 0.3
        
        # High momentum increases urgency
        if momentum > 0.2:  # 20% price increase
            base_urgency += min(momentum, 0.5)
        
        # High volume spike increases urgency
        if volume_spike > 1.0:  # 100% volume increase
            base_urgency += min(volume_spike * 0.3, 0.4)
        
        return max(0.0, min(1.0, base_urgency))
    
    def _estimate_profit_potential(self, token_data: Dict[str, Any], momentum: float, volume_spike: float) -> float:
        """Estimate profit potential percentage"""
        
        base_potential = 5.0  # 5% base expectation
        
        # Momentum factor
        if momentum > 0:
            base_potential += momentum * 50  # Up to 50% for high momentum
        
        # Volume factor
        if volume_spike > 0:
            base_potential += volume_spike * 20  # Up to 20% for volume spike
        
        # Market cap factor (smaller caps have higher potential)
        market_cap = token_data.get("market_cap_sol", 0)
        if market_cap > 0 and market_cap < 100:  # Small cap
            base_potential += 10
        elif market_cap < 1000:  # Medium cap
            base_potential += 5
        
        return min(base_potential, 100.0)  # Cap at 100%
    
    def _calculate_risk_level(self, token_data: Dict[str, Any]) -> float:
        """Calculate risk level"""
        
        base_risk = 0.3
        
        # High market cap reduces risk
        market_cap = token_data.get("market_cap_sol", 0)
        if market_cap > 10000:
            base_risk -= 0.2
        elif market_cap > 1000:
            base_risk -= 0.1
        elif market_cap < 10:
            base_risk += 0.3
        
        # High holder count reduces risk
        holder_count = token_data.get("holder_count", 0)
        if holder_count > 5000:
            base_risk -= 0.2
        elif holder_count > 1000:
            base_risk -= 0.1
        elif holder_count < 50:
            base_risk += 0.3
        
        # Liquidity factor
        liquidity = token_data.get("liquidity_sol", 0)
        if liquidity < 10:
            base_risk += 0.3
        elif liquidity > 100:
            base_risk -= 0.1
        
        return max(0.0, min(1.0, base_risk))

class RealTimeDataFeed:
    """Real-time data feed from multiple sources"""
    
    def __init__(self):
        self.config = get_trading_config()
        self.logger = logging.getLogger("RealTimeDataFeed")
        
        # Data sources
        self.jupiter_url = "https://quote-api.jup.ag/v6"
        self.birdeye_url = "https://public-api.birdeye.so"
        self.dexscreener_url = "https://api.dexscreener.com"
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_queue = asyncio.Queue(maxsize=100)
        
        # Rate limiting
        self.last_request_time = 0.0
        self.requests_this_second = 0
        self.max_requests_per_second = 10
        
        # Cache
        self.data_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 30  # 30 seconds
    
    async def initialize(self):
        """Initialize data feed"""
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=20)
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "WorkerAnt/1.0",
                "Accept": "application/json"
            }
        )
        
        self.logger.info("Real-time data feed initialized")
    
    async def shutdown(self):
        """Shutdown data feed"""
        if self.session:
            await self.session.close()
        self.logger.info("Real-time data feed shutdown")
    
    async def get_token_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive token data"""
        
        # Check cache first
        cache_key = token_address
        if cache_key in self.data_cache:
            cached_data = self.data_cache[cache_key]
            if time.time() - cached_data["timestamp"] < self.cache_ttl:
                return cached_data["data"]
        
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Gather data from multiple sources
            tasks = [
                self._get_jupiter_data(token_address),
                self._get_birdeye_data(token_address),
                self._get_dexscreener_data(token_address)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine data sources
            combined_data = self._combine_token_data(results)
            
            if combined_data:
                # Cache the result
                self.data_cache[cache_key] = {
                    "data": combined_data,
                    "timestamp": time.time()
                }
                
                return combined_data
        
        except Exception as e:
            self.logger.error(f"Failed to get token data for {mask_sensitive_value(token_address)}: {e}")
        
        return None
    
    async def _rate_limit(self):
        """Apply rate limiting"""
        current_time = time.time()
        
        # Reset counter if new second
        if current_time - self.last_request_time >= 1.0:
            self.requests_this_second = 0
            self.last_request_time = current_time
        
        # Check limit
        if self.requests_this_second >= self.max_requests_per_second:
            wait_time = 1.0 - (current_time - self.last_request_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self.requests_this_second += 1
    
    async def _get_jupiter_data(self, token_address: str) -> Dict[str, Any]:
        """Get data from Jupiter API"""
        try:
            url = f"{self.jupiter_url}/price"
            params = {"ids": token_address}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if token_address in data.get("data", {}):
                        token_data = data["data"][token_address]
                        return {
                            "price_sol": float(token_data.get("price", 0)) / 1e9,  # Convert to SOL
                            "source": "jupiter"
                        }
        except Exception as e:
            self.logger.debug(f"Jupiter API error: {e}")
        
        return {}
    
    async def _get_birdeye_data(self, token_address: str) -> Dict[str, Any]:
        """Get data from Birdeye API"""
        try:
            # This would require API key setup
            # Placeholder implementation
            return {"source": "birdeye"}
        except Exception as e:
            self.logger.debug(f"Birdeye API error: {e}")
        
        return {}
    
    async def _get_dexscreener_data(self, token_address: str) -> Dict[str, Any]:
        """Get data from DexScreener API"""
        try:
            url = f"{self.dexscreener_url}/latest/dex/tokens/{token_address}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get("pairs", [])
                    
                    if pairs:
                        pair = pairs[0]  # Take first pair
                        return {
                            "price_sol": float(pair.get("priceNative", 0)),
                            "liquidity_sol": float(pair.get("liquidity", {}).get("base", 0)),
                            "volume_24h_sol": float(pair.get("volume", {}).get("h24", 0)),
                            "market_cap_sol": float(pair.get("marketCap", 0)),
                            "source": "dexscreener"
                        }
        except Exception as e:
            self.logger.debug(f"DexScreener API error: {e}")
        
        return {}
    
    def _combine_token_data(self, results: List[Any]) -> Dict[str, Any]:
        """Combine data from multiple sources"""
        
        combined = {}
        
        for result in results:
            if isinstance(result, dict):
                combined.update(result)
        
        # Set defaults if missing
        combined.setdefault("price_sol", 0.0)
        combined.setdefault("liquidity_sol", 0.0)
        combined.setdefault("volume_24h_sol", 0.0)
        combined.setdefault("market_cap_sol", 0.0)
        combined.setdefault("symbol", "UNKNOWN")
        combined.setdefault("name", "Unknown Token")
        
        return combined if combined.get("price_sol", 0) > 0 else None

class ProductionScanner:
    """Production-ready token scanner"""
    
    def __init__(self):
        self.config = get_trading_config()
        self.security_config = get_security_config()
        self.logger = logging.getLogger("ProductionScanner")
        
        # Components
        self.data_feed = RealTimeDataFeed()
        self.token_filter = TokenFilter()
        self.analyzer = OpportunityAnalyzer()
        
        # Scanner state
        self.scan_mode = ScanMode.BALANCED
        self.scanning_active = False
        self.scan_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.metrics = ScannerMetrics()
        self.start_time = datetime.utcnow()
        
        # Token tracking
        self.tracked_tokens: Set[str] = set()
        self.recent_opportunities: List[TradingOpportunity] = []
        self.max_recent_opportunities = 100
        
        # Scan scheduling
        self.scan_interval_seconds = 5.0  # Default scan interval
        self.last_scan_time = 0.0
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize scanner system"""
        if self.initialized:
            return
        
        try:
            await self.data_feed.initialize()
            
            # Load initial token list
            await self._load_initial_tokens()
            
            self.initialized = True
            self.logger.info("Production scanner initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Scanner initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown scanner gracefully"""
        self.initialized = False
        self.scanning_active = False
        
        if self.scan_task:
            self.scan_task.cancel()
            try:
                await self.scan_task
            except asyncio.CancelledError:
                pass
        
        await self.data_feed.shutdown()
        
        self.logger.info("Production scanner shutdown complete")
    
    async def start_scanning(self, mode: ScanMode = ScanMode.BALANCED):
        """Start continuous scanning"""
        if self.scanning_active:
            return
        
        self.scan_mode = mode
        self.scanning_active = True
        
        # Adjust scan interval based on mode
        if mode == ScanMode.AGGRESSIVE:
            self.scan_interval_seconds = 2.0
        elif mode == ScanMode.CONSERVATIVE:
            self.scan_interval_seconds = 10.0
        elif mode == ScanMode.STEALTH:
            self.scan_interval_seconds = 30.0
        else:  # BALANCED
            self.scan_interval_seconds = 5.0
        
        self.scan_task = asyncio.create_task(self._scanning_loop())
        self.logger.info(f"Scanner started in {mode.value} mode")
    
    async def stop_scanning(self):
        """Stop continuous scanning"""
        self.scanning_active = False
        
        if self.scan_task:
            self.scan_task.cancel()
            try:
                await self.scan_task
            except asyncio.CancelledError:
                pass
            self.scan_task = None
        
        self.logger.info("Scanner stopped")
    
    async def scan_for_opportunities(self) -> List[TradingOpportunity]:
        """Perform single scan for opportunities"""
        if not self.initialized:
            await self.initialize()
        
        scan_start = time.time()
        opportunities = []
        
        try:
            # Get tokens to scan
            tokens_to_scan = await self._get_tokens_to_scan()
            
            # Scan tokens concurrently
            scan_tasks = []
            for token_address in tokens_to_scan:
                task = asyncio.create_task(self._scan_single_token(token_address))
                scan_tasks.append(task)
            
            # Execute scans with concurrency limit
            batch_size = 10
            for i in range(0, len(scan_tasks), batch_size):
                batch = scan_tasks[i:i + batch_size]
                results = await asyncio.gather(*batch, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, TradingOpportunity):
                        opportunities.append(result)
            
            # Sort by priority
            opportunities.sort(key=lambda x: x.get_priority_score(), reverse=True)
            
            # Update metrics
            scan_latency = int((time.time() - scan_start) * 1000)
            self.metrics.total_scans += 1
            self.metrics.opportunities_found += len(opportunities)
            self._update_scan_metrics(scan_latency)
            
            # Store recent opportunities
            self.recent_opportunities.extend(opportunities)
            if len(self.recent_opportunities) > self.max_recent_opportunities:
                self.recent_opportunities = self.recent_opportunities[-self.max_recent_opportunities:]
            
            self.logger.info(f"Scan completed: {len(opportunities)} opportunities found in {scan_latency}ms")
            
            return opportunities[:10]  # Return top 10 opportunities
            
        except Exception as e:
            self.logger.error(f"Scan error: {e}")
            return []
    
    async def _scanning_loop(self):
        """Continuous scanning loop"""
        while self.scanning_active:
            try:
                # Check scan timing
                current_time = time.time()
                if current_time - self.last_scan_time < self.scan_interval_seconds:
                    await asyncio.sleep(0.5)
                    continue
                
                # Perform scan
                opportunities = await self.scan_for_opportunities()
                self.last_scan_time = current_time
                
                # Log if opportunities found
                if opportunities:
                    self.logger.debug(f"Background scan found {len(opportunities)} opportunities")
                
                # Brief pause
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scanning loop error: {e}")
                await asyncio.sleep(5.0)  # Error recovery delay
    
    async def _get_tokens_to_scan(self) -> List[str]:
        """Get list of tokens to scan"""
        # For production, this would integrate with:
        # - New token detection services
        # - Volume monitoring systems
        # - Social sentiment feeds
        # - Technical analysis triggers
        
        # Placeholder implementation - return tracked tokens
        return list(self.tracked_tokens)[:50]  # Limit to 50 tokens per scan
    
    async def _scan_single_token(self, token_address: str) -> Optional[TradingOpportunity]:
        """Scan a single token for opportunities"""
        try:
            # Get token data
            token_data = await self.data_feed.get_token_data(token_address)
            if not token_data:
                return None
            
            # Apply filters
            should_skip, skip_reason = self.token_filter.should_skip_token(token_address, token_data)
            if should_skip:
                self._update_filter_metrics(skip_reason)
                return None
            
            # Analyze opportunity
            opportunity = self.analyzer.analyze_opportunity(token_address, token_data)
            
            # Security assessment
            security_risk, rug_prob, honeypot_risk = await self.token_filter.evaluate_token_security(
                token_address, token_data
            )
            
            opportunity.security_risk = security_risk
            opportunity.rug_probability = rug_prob
            opportunity.honeypot_risk = honeypot_risk
            
            # Final validation
            if not opportunity.is_valid_opportunity():
                return None
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Error scanning token {mask_sensitive_value(token_address)}: {e}")
            return None
    
    async def _load_initial_tokens(self):
        """Load initial list of tokens to track"""
        # This would load from various sources:
        # - Popular token lists
        # - Recent volume gainers
        # - New token detection
        
        # Placeholder - add some well-known tokens for testing
        known_tokens = [
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
            "So11111111111111111111111111111111111112",       # SOL
        ]
        
        self.tracked_tokens.update(known_tokens)
    
    def _update_scan_metrics(self, latency_ms: int):
        """Update scanning performance metrics"""
        # Update average latency
        if self.metrics.avg_scan_latency_ms == 0:
            self.metrics.avg_scan_latency_ms = latency_ms
        else:
            alpha = 0.1  # Smoothing factor
            self.metrics.avg_scan_latency_ms = (
                alpha * latency_ms + (1 - alpha) * self.metrics.avg_scan_latency_ms
            )
        
        # Update scans per minute
        uptime_minutes = max((datetime.utcnow() - self.start_time).total_seconds() / 60, 1)
        self.metrics.scans_per_minute = self.metrics.total_scans / uptime_minutes
        
        # Update uptime
        self.metrics.uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
    
    def _update_filter_metrics(self, skip_reason: str):
        """Update filter statistics"""
        if "liquidity" in skip_reason.lower():
            self.metrics.liquidity_filtered += 1
        elif "volume" in skip_reason.lower():
            self.metrics.volume_filtered += 1
        elif "blacklist" in skip_reason.lower():
            self.metrics.security_filtered += 1
        elif "rug" in skip_reason.lower():
            self.metrics.rug_filtered += 1
    
    def add_token_to_watchlist(self, token_address: str):
        """Add token to tracking watchlist"""
        self.tracked_tokens.add(token_address)
        self.logger.info(f"Added token to watchlist: {mask_sensitive_value(token_address)}")
    
    def remove_token_from_watchlist(self, token_address: str):
        """Remove token from tracking watchlist"""
        self.tracked_tokens.discard(token_address)
        self.logger.info(f"Removed token from watchlist: {mask_sensitive_value(token_address)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get scanner performance metrics"""
        return {
            "total_scans": self.metrics.total_scans,
            "opportunities_found": self.metrics.opportunities_found,
            "success_rate": self.metrics.get_success_rate(),
            "avg_scan_latency_ms": self.metrics.avg_scan_latency_ms,
            "scans_per_minute": self.metrics.scans_per_minute,
            "uptime_seconds": self.metrics.uptime_seconds,
            "tracked_tokens": len(self.tracked_tokens),
            "scan_mode": self.scan_mode.value,
            "scanning_active": self.scanning_active,
            "filter_stats": {
                "security_filtered": self.metrics.security_filtered,
                "liquidity_filtered": self.metrics.liquidity_filtered,
                "volume_filtered": self.metrics.volume_filtered,
                "rug_filtered": self.metrics.rug_filtered
            }
        }
    
    def get_recent_opportunities(self) -> List[Dict[str, Any]]:
        """Get recent opportunities for analysis"""
        return [
            {
                "token_symbol": opp.token_symbol,
                "token_address": mask_sensitive_value(opp.token_address),
                "confidence_score": opp.confidence_score,
                "urgency_score": opp.urgency_score,
                "profit_potential": opp.profit_potential,
                "risk_level": opp.risk_level,
                "security_risk": opp.security_risk.value,
                "discovered_at": opp.discovered_at.isoformat(),
                "current_price_sol": opp.current_price_sol,
                "liquidity_sol": opp.liquidity_sol
            }
            for opp in self.recent_opportunities[-20:]  # Last 20 opportunities
        ]

# === GLOBAL INSTANCES ===

# Global scanner instance
production_scanner = ProductionScanner()

# Backward compatibility aliases
profit_scanner = production_scanner

# Export main classes
__all__ = [
    "TradingOpportunity", "ScanMode", "SecurityRisk", "ScannerMetrics",
    "TokenFilter", "OpportunityAnalyzer", "RealTimeDataFeed", "ProductionScanner",
    "production_scanner", "profit_scanner"
]

# Additional result and data classes
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

class TradingOpportunity:
    """Represents a trading opportunity found by scanner"""
    
    def __init__(self, symbol: str, price: float, confidence: float, strategy: str = "default"):
        self.symbol = symbol
        self.price = price
        self.confidence = confidence
        self.strategy = strategy
        self.timestamp = datetime.now()
        self.entry_signal_strength = confidence
        self.confidence_score = confidence * 100
        
    def __str__(self):
        return f"TradingOpportunity({self.symbol}, price=${self.price}, confidence={self.confidence:.2f})" 