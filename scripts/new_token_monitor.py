#!/usr/bin/env python3
"""
NEW TOKEN LAUNCH MONITOR
=======================

Real-time monitoring of all new token launches on Solana with multiple detection methods.
Integrates with your existing trading bot for immediate opportunity identification.
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import os
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@dataclass
class NewTokenLaunch:
    """Data structure for newly detected tokens"""
    
    # Basic token info
    token_address: str
    symbol: str
    name: str
    creator_address: str
    
    # Launch details
    launch_time: datetime
    initial_price_sol: float
    initial_liquidity_sol: float
    dex_platform: str  # Raydium, Orca, etc.
    
    # Market data
    total_supply: int
    current_holders: int
    market_cap_sol: float
    
    # Detection info
    detection_method: str
    detection_latency_ms: int
    confidence_score: float
    
    # Fields with defaults come last
    volume_1h_sol: float = 0.0
    potential_rug: bool = False
    honeypot_risk: bool = False
    suspicious_activity: bool = False
    metadata_uri: Optional[str] = None
    social_links: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.total_supply > 0 and self.initial_price_sol > 0:
            self.market_cap_sol = (self.total_supply * self.initial_price_sol) / 1e6

class DetectionMethod(Enum):
    RAYDIUM_POOLS = "raydium_pools"
    ORCA_POOLS = "orca_pools"
    JUPITER_TOKENS = "jupiter_tokens"
    BLOCKCHAIN_LOGS = "blockchain_logs"
    DEXSCREENER_API = "dexscreener_api"
    BIRDEYE_API = "birdeye_api"
    PUMP_FUN = "pump_fun"

class NewTokenDetector:
    """Multi-source new token detection system"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger("NewTokenDetector")
        
        # Detection state
        self.known_tokens: Set[str] = set()
        self.recent_launches: List[NewTokenLaunch] = []
        self.max_recent_launches = 500
        
        # API endpoints
        self.endpoints = {
            "helius_rpc": f"https://mainnet.helius-rpc.com/?api-key={helius_api_key}",
            "dexscreener": "https://api.dexscreener.com/latest",
            "birdeye": "https://public-api.birdeye.so/public",
            "raydium": "https://api.raydium.io/v2",
            "jupiter": "https://quote-api.jup.ag/v6",
            "pump_fun": "https://frontend-api.pump.fun"
        }
        
        # Rate limiting
        self.last_request_time: Dict[str, float] = {}
        self.min_request_interval = {
            "dexscreener": 1.0,
            "birdeye": 0.5,
            "raydium": 0.2,
            "jupiter": 0.1,
            "pump_fun": 0.5
        }
        
        # Detection callbacks
        self.new_token_callbacks: List[callable] = []
        
    async def __aenter__(self):
        # Create connector that works on Windows
        connector = aiohttp.TCPConnector(
            resolver=aiohttp.AsyncResolver(),
            use_dns_cache=False
        )
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "SolanaTokenMonitor/1.0"},
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def add_callback(self, callback: callable):
        """Add callback for new token detection"""
        self.new_token_callbacks.append(callback)
    
    async def _rate_limited_request(self, service: str, url: str, **kwargs) -> Optional[Dict]:
        """Make rate-limited API request"""
        
        # Check rate limit
        current_time = time.time()
        last_request = self.last_request_time.get(service, 0)
        min_interval = self.min_request_interval.get(service, 1.0)
        
        if current_time - last_request < min_interval:
            await asyncio.sleep(min_interval - (current_time - last_request))
        
        try:
            async with self.session.get(url, **kwargs) as response:
                self.last_request_time[service] = time.time()
                
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.warning(f"{service} API returned {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Error requesting {service}: {e}")
        
        return None
    
    async def detect_via_dexscreener(self) -> List[NewTokenLaunch]:
        """Detect new tokens via DexScreener API"""
        
        launches = []
        
        try:
            # Get tokens with recent high volume on Solana
            url = f"{self.endpoints['dexscreener']}/dex/tokens"
            data = await self._rate_limited_request("dexscreener", url)
            
            if data and "pairs" in data:
                for pair in data["pairs"][:50]:  # Check latest 50 pairs
                    base_token = pair.get("baseToken", {})
                    token_address = base_token.get("address")
                    
                    # Only check Solana pairs
                    if pair.get("chainId") != "solana":
                        continue
                    
                    if token_address and token_address not in self.known_tokens:
                        # Check if pair is very recent
                        pair_created_at = pair.get("pairCreatedAt", 0)
                        if pair_created_at > 0:
                            created_time = datetime.fromtimestamp(pair_created_at / 1000)
                            if datetime.now() - created_time < timedelta(hours=1):
                                launch = await self._create_launch_from_dexscreener(pair)
                                if launch:
                                    launches.append(launch)
                                    self.known_tokens.add(token_address)
        
        except Exception as e:
            self.logger.error(f"DexScreener detection error: {e}")
        
        return launches
    
    async def detect_via_raydium_pools(self) -> List[NewTokenLaunch]:
        """Detect new tokens via Raydium pool creation"""
        
        launches = []
        
        try:
            # Get recent pools
            url = f"{self.endpoints['raydium']}/main/pairs"
            data = await self._rate_limited_request("raydium", url)
            
            if data:
                for pool in data[:20]:  # Check last 20 pools
                    base_mint = pool.get("baseMint")
                    
                    if base_mint and base_mint not in self.known_tokens:
                        # Check if pool is very recent
                        pool_open_time = pool.get("poolOpenTime", 0)
                        if pool_open_time > 0:
                            pool_time = datetime.fromtimestamp(pool_open_time / 1000)
                            if datetime.now() - pool_time < timedelta(hours=1):
                                launch = await self._create_launch_from_raydium(pool)
                                if launch:
                                    launches.append(launch)
                                    self.known_tokens.add(base_mint)
        
        except Exception as e:
            self.logger.error(f"Raydium detection error: {e}")
        
        return launches
    
    async def detect_via_pump_fun(self) -> List[NewTokenLaunch]:
        """Detect new tokens from Pump.fun"""
        
        launches = []
        
        try:
            # Get latest coins from pump.fun
            url = f"{self.endpoints['pump_fun']}/coins?sort=created_timestamp&order=desc&limit=50"
            data = await self._rate_limited_request("pump_fun", url)
            
            if data:
                for coin in data:
                    mint = coin.get("mint")
                    
                    if mint and mint not in self.known_tokens:
                        # Check if very recent
                        created_timestamp = coin.get("created_timestamp", 0)
                        if created_timestamp > 0:
                            created_time = datetime.fromtimestamp(created_timestamp / 1000)
                            if datetime.now() - created_time < timedelta(minutes=30):
                                launch = await self._create_launch_from_pump_fun(coin)
                                if launch:
                                    launches.append(launch)
                                    self.known_tokens.add(mint)
        
        except Exception as e:
            self.logger.error(f"Pump.fun detection error: {e}")
        
        return launches
    
    async def _create_launch_from_dexscreener(self, pair_data: Dict) -> Optional[NewTokenLaunch]:
        """Create launch object from DexScreener data"""
        
        try:
            base_token = pair_data.get("baseToken", {})
            quote_token = pair_data.get("quoteToken", {})
            
            # Only process SOL pairs
            if quote_token.get("symbol") != "SOL":
                return None
            
            token_address = base_token.get("address")
            if not token_address:
                return None
            
            # Calculate detection latency
            pair_created_at = pair_data.get("pairCreatedAt", 0)
            if pair_created_at > 0:
                created_time = datetime.fromtimestamp(pair_created_at / 1000)
                detection_latency = int((datetime.now() - created_time).total_seconds() * 1000)
            else:
                created_time = datetime.now()
                detection_latency = 0
            
            launch = NewTokenLaunch(
                token_address=token_address,
                symbol=base_token.get("symbol", "UNKNOWN"),
                name=base_token.get("name", "Unknown Token"),
                creator_address="unknown",
                launch_time=created_time,
                initial_price_sol=float(pair_data.get("priceNative", 0)),
                initial_liquidity_sol=float(pair_data.get("liquidity", {}).get("usd", 0)) / 200,
                dex_platform=pair_data.get("dexId", "unknown"),
                total_supply=0,
                current_holders=0,
                market_cap_sol=float(pair_data.get("marketCap", 0)) / 200,
                volume_1h_sol=float(pair_data.get("volume", {}).get("h1", 0)) / 200,
                detection_method=DetectionMethod.DEXSCREENER_API.value,
                detection_latency_ms=detection_latency,
                confidence_score=0.8,
                social_links={
                    "website": pair_data.get("info", {}).get("websites", [{}])[0].get("url", ""),
                    "twitter": pair_data.get("info", {}).get("socials", [{}])[0].get("url", "")
                }
            )
            
            return launch
            
        except Exception as e:
            self.logger.error(f"Error creating launch from DexScreener: {e}")
            return None
    
    async def _create_launch_from_raydium(self, pool_data: Dict) -> Optional[NewTokenLaunch]:
        """Create launch object from Raydium data"""
        
        try:
            base_mint = pool_data.get("baseMint")
            if not base_mint:
                return None
            
            open_time = pool_data.get("poolOpenTime", 0)
            if open_time > 0:
                launch_time = datetime.fromtimestamp(open_time / 1000)
                detection_latency = int((datetime.now() - launch_time).total_seconds() * 1000)
            else:
                launch_time = datetime.now()
                detection_latency = 0
            
            launch = NewTokenLaunch(
                token_address=base_mint,
                symbol=pool_data.get("baseSymbol", "UNKNOWN"),
                name=pool_data.get("baseName", "Unknown Token"),
                creator_address="unknown",
                launch_time=launch_time,
                initial_price_sol=float(pool_data.get("price", 0)),
                initial_liquidity_sol=float(pool_data.get("liquidity", 0)),
                dex_platform="Raydium",
                total_supply=0,
                current_holders=0,
                market_cap_sol=0,
                detection_method=DetectionMethod.RAYDIUM_POOLS.value,
                detection_latency_ms=detection_latency,
                confidence_score=0.9,
            )
            
            return launch
            
        except Exception as e:
            self.logger.error(f"Error creating launch from Raydium: {e}")
            return None
    
    async def _create_launch_from_pump_fun(self, coin_data: Dict) -> Optional[NewTokenLaunch]:
        """Create launch object from Pump.fun data"""
        
        try:
            mint = coin_data.get("mint")
            if not mint:
                return None
            
            created_timestamp = coin_data.get("created_timestamp", 0)
            if created_timestamp > 0:
                launch_time = datetime.fromtimestamp(created_timestamp / 1000)
                detection_latency = int((datetime.now() - launch_time).total_seconds() * 1000)
            else:
                launch_time = datetime.now()
                detection_latency = 0
            
            launch = NewTokenLaunch(
                token_address=mint,
                symbol=coin_data.get("symbol", "UNKNOWN"),
                name=coin_data.get("name", "Unknown Token"),
                creator_address=coin_data.get("creator", "unknown"),
                launch_time=launch_time,
                initial_price_sol=float(coin_data.get("usd_market_cap", 0)) / 200,
                initial_liquidity_sol=0,
                dex_platform="Pump.fun",
                total_supply=int(coin_data.get("total_supply", 0)),
                current_holders=0,
                market_cap_sol=float(coin_data.get("usd_market_cap", 0)) / 200,
                detection_method=DetectionMethod.PUMP_FUN.value,
                detection_latency_ms=detection_latency,
                confidence_score=0.7,
                metadata_uri=coin_data.get("uri"),
                social_links={
                    "website": coin_data.get("website", ""),
                    "twitter": coin_data.get("twitter", ""),
                    "telegram": coin_data.get("telegram", "")
                }
            )
            
            return launch
            
        except Exception as e:
            self.logger.error(f"Error creating launch from Pump.fun: {e}")
            return None
    
    async def continuous_monitoring(self, callback: callable = None):
        """Run continuous monitoring for new token launches"""
        
        self.logger.info("🔍 Starting continuous new token monitoring...")
        
        while True:
            try:
                all_launches = []
                
                # Run detection methods concurrently
                detection_tasks = [
                    self.detect_via_dexscreener(),
                    self.detect_via_raydium_pools(),
                    self.detect_via_pump_fun()
                ]
                
                results = await asyncio.gather(*detection_tasks, return_exceptions=True)
                
                # Collect all launches
                for result in results:
                    if isinstance(result, list):
                        all_launches.extend(result)
                
                # Process new launches
                if all_launches:
                    self.logger.info(f"🚀 Detected {len(all_launches)} new token launches")
                    
                    for launch in all_launches:
                        # Add to recent launches
                        self.recent_launches.append(launch)
                        
                        # Maintain max size
                        if len(self.recent_launches) > self.max_recent_launches:
                            self.recent_launches = self.recent_launches[-self.max_recent_launches:]
                        
                        # Call callbacks
                        for cb in self.new_token_callbacks:
                            try:
                                await cb(launch)
                            except Exception as e:
                                self.logger.error(f"Callback error: {e}")
                        
                        # Call additional callback if provided
                        if callback:
                            try:
                                await callback(launch)
                            except Exception as e:
                                self.logger.error(f"Additional callback error: {e}")
                        
                        # Log the launch
                        self.logger.info(
                            f"🔥 NEW TOKEN: {launch.symbol} ({launch.token_address[:8]}...) "
                            f"via {launch.detection_method} - Latency: {launch.detection_latency_ms}ms"
                        )
                
                # Wait before next scan
                await asyncio.sleep(10)  # Scan every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)  # Wait longer on error

class TokenLaunchAnalyzer:
    """Analyze new token launches for trading opportunities"""
    
    def __init__(self):
        self.logger = logging.getLogger("TokenLaunchAnalyzer")
    
    async def analyze_launch(self, launch: NewTokenLaunch) -> Dict[str, Any]:
        """Analyze a new token launch for trading potential"""
        
        analysis = {
            "token_address": launch.token_address,
            "symbol": launch.symbol,
            "trading_score": 0.0,
            "risk_score": 0.0,
            "speed_score": 0.0,
            "recommendation": "SKIP",
            "reasons": []
        }
        
        # Speed scoring (faster detection = higher score)
        if launch.detection_latency_ms < 30000:  # < 30 seconds
            analysis["speed_score"] = 1.0
            analysis["reasons"].append("Very fast detection")
        elif launch.detection_latency_ms < 300000:  # < 5 minutes
            analysis["speed_score"] = 0.7
            analysis["reasons"].append("Fast detection")
        else:
            analysis["speed_score"] = 0.3
            analysis["reasons"].append("Slow detection")
        
        # Platform scoring
        platform_scores = {
            "Raydium": 0.9,
            "Orca": 0.8,
            "Pump.fun": 0.6,
            "unknown": 0.3
        }
        platform_score = platform_scores.get(launch.dex_platform, 0.3)
        
        # Liquidity scoring
        liquidity_score = 0.0
        if launch.initial_liquidity_sol > 100:
            liquidity_score = 0.9
            analysis["reasons"].append("High initial liquidity")
        elif launch.initial_liquidity_sol > 50:
            liquidity_score = 0.7
            analysis["reasons"].append("Good initial liquidity")
        elif launch.initial_liquidity_sol > 10:
            liquidity_score = 0.5
            analysis["reasons"].append("Moderate liquidity")
        else:
            liquidity_score = 0.2
            analysis["reasons"].append("Low liquidity")
        
        # Risk scoring
        risk_factors = 0
        
        if any(word in launch.symbol.lower() for word in ["test", "fake", "scam", "rug"]):
            risk_factors += 0.3
            analysis["reasons"].append("Suspicious symbol")
        
        if launch.initial_liquidity_sol < 5:
            risk_factors += 0.4
            analysis["reasons"].append("Very low liquidity")
        
        analysis["risk_score"] = min(risk_factors, 1.0)
        
        # Overall trading score
        analysis["trading_score"] = (
            (analysis["speed_score"] * 0.4) +
            (platform_score * 0.3) +
            (liquidity_score * 0.3) -
            (analysis["risk_score"] * 0.5)
        )
        
        # Recommendation
        if analysis["trading_score"] > 0.7 and analysis["risk_score"] < 0.3:
            analysis["recommendation"] = "BUY"
            analysis["reasons"].append("High potential, low risk")
        elif analysis["trading_score"] > 0.5 and analysis["risk_score"] < 0.5:
            analysis["recommendation"] = "WATCH"
            analysis["reasons"].append("Monitor for better entry")
        else:
            analysis["recommendation"] = "SKIP"
            analysis["reasons"].append("Low potential or high risk")
        
        return analysis

async def main():
    """Main monitoring function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("NewTokenMonitor")
    
    # Get Helius API key
    helius_api_key = os.getenv("HELIUS_API_KEY")
    if not helius_api_key:
        print("🔑 Please set HELIUS_API_KEY environment variable")
        helius_api_key = input("Enter your Helius API key: ").strip()
        
        if not helius_api_key:
            print("❌ No API key provided. Exiting.")
            return
    
    print("🚀 Starting New Token Launch Monitor")
    print("=" * 50)
    
    analyzer = TokenLaunchAnalyzer()
    
    async def handle_new_launch(launch: NewTokenLaunch):
        """Handle newly detected token launch"""
        
        # Analyze the launch
        analysis = await analyzer.analyze_launch(launch)
        
        print(f"\n🔥 NEW TOKEN DETECTED:")
        print(f"   Symbol: {launch.symbol}")
        print(f"   Address: {launch.token_address[:8]}...")
        print(f"   Platform: {launch.dex_platform}")
        print(f"   Liquidity: {launch.initial_liquidity_sol:.2f} SOL")
        print(f"   Detection: {launch.detection_method} ({launch.detection_latency_ms}ms)")
        print(f"   Score: {analysis['trading_score']:.2f}")
        print(f"   Recommendation: {analysis['recommendation']}")
        
        # Save to file for bot integration
        launch_data = {
            "timestamp": launch.launch_time.isoformat(),
            "token_address": launch.token_address,
            "symbol": launch.symbol,
            "name": launch.name,
            "dex_platform": launch.dex_platform,
            "initial_price_sol": launch.initial_price_sol,
            "initial_liquidity_sol": launch.initial_liquidity_sol,
            "detection_method": launch.detection_method,
            "detection_latency_ms": launch.detection_latency_ms,
            "analysis": analysis
        }
        
        # Append to launches log
        os.makedirs("logs", exist_ok=True)
        with open("logs/new_token_launches.jsonl", "a") as f:
            f.write(json.dumps(launch_data) + "\n")
    
    # Start monitoring
    async with NewTokenDetector(helius_api_key) as detector:
        detector.add_callback(handle_new_launch)
        await detector.continuous_monitoring()

if __name__ == "__main__":
    asyncio.run(main()) 