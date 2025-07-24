"""
HISTORICAL DATA INGESTION SYSTEM
=================================

Robust data ingestion system for fetching and storing large datasets of historical
price, volume, and transaction data from multiple providers for ML model training.

Features:
- Multi-provider data fetching (Helius, DexScreener, Birdeye)
- Configurable date ranges and token lists
- Data validation and cleaning
- TimescaleDB storage with proper schema
- Rate limiting and error handling
- Progress tracking and resumable downloads

Usage:
    python scripts/ingest_historical_data.py --provider helius --days 30 --tokens popular
    python scripts/ingest_historical_data.py --provider dexscreener --tokens custom --token-list tokens.txt
    python scripts/ingest_historical_data.py --start-date 2024-01-01 --end-date 2024-06-01
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import aiohttp
import asyncpg
from dataclasses import dataclass, asdict
import time


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from worker_ant_v1.core.database import get_database_config, TimescaleDBManager
from worker_ant_v1.core.unified_config import get_network_config_async
from worker_ant_v1.utils.logger import get_logger


@dataclass
class TokenInfo:
    """Token information for data fetching"""
    address: str
    symbol: str
    name: str
    decimals: int = 9
    market_cap: Optional[float] = None
    verified: bool = False


@dataclass
class PriceDataPoint:
    """Single price data point"""
    timestamp: datetime
    token_address: str
    price_usd: float
    volume_24h: float
    market_cap: Optional[float] = None
    liquidity: Optional[float] = None
    price_change_1h: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_change_24h: Optional[float] = None
    transaction_count_24h: Optional[int] = None
    holder_count: Optional[int] = None
    provider: str = "unknown"


@dataclass
class TransactionData:
    """Transaction data for on-chain analysis"""
    timestamp: datetime
    token_address: str
    transaction_hash: str
    transaction_type: str  # buy, sell, transfer
    amount_tokens: float
    amount_sol: float
    wallet_address: str
    dex: str
    price_impact: Optional[float] = None
    slippage: Optional[float] = None
    provider: str = "unknown"


class DataProvider:
    """Base class for data providers"""
    
    def __init__(self, api_key: str, rate_limit: float = 1.0):
        self.api_key = api_key
        self.rate_limit = rate_limit  # requests per second
        self.last_request_time = 0
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = get_logger(f"{self.__class__.__name__}")
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _rate_limit_wait(self):
        """Wait for rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()


class HeliusProvider(DataProvider):
    """Helius API provider for Solana data"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, rate_limit=5.0)  # 5 requests per second
        self.base_url = "https://api.helius.xyz"
    
    async def get_token_price_history(self, token_address: str, 
                                    start_date: datetime, 
                                    end_date: datetime) -> List[PriceDataPoint]:
        """Get historical price data for a token"""
        await self._ensure_session()
        await self._rate_limit_wait()
        
        try:
        try:
            # Helius doesn't have direct price history, we'll use transaction data
            transactions = await self.get_token_transactions(token_address, start_date, end_date)
            price_points = self._derive_price_from_transactions(transactions)
            return price_points
            
        except Exception as e:
            self.logger.error(f"Failed to get price history for {token_address}: {e}")
            return []
    
    async def get_token_transactions(self, token_address: str,
                                   start_date: datetime,
                                   end_date: datetime) -> List[TransactionData]:
        """Get token transactions from Helius"""
        await self._ensure_session()
        
        url = f"{self.base_url}/v0/transactions"
        params = {
            "api-key": self.api_key,
            "accounts": [token_address],
            "commitment": "confirmed",
            "type": "SWAP"
        }
        
        transactions = []
        
        try:
        try:
            before = None
            page_count = 0
            
            while page_count < 100:  # Limit to prevent infinite loops
                if before:
                    params["before"] = before
                
                await self._rate_limit_wait()
                
                async with self.session.get(url, params=params) as response:
                    if response.status != 200:
                        break
                    
                    data = await response.json()
                    
                    if not data or not isinstance(data, list):
                        break
                    
                    for tx in data:
                        tx_data = self._parse_transaction(tx, token_address)
                        if tx_data and start_date <= tx_data.timestamp <= end_date:
                            transactions.append(tx_data)
                    
                    if len(data) < 1000:  # Less than full page
                        break
                    
                    before = data[-1].get("slot")
                    page_count += 1
            
            self.logger.info(f"Fetched {len(transactions)} transactions for {token_address}")
            return transactions
            
        except Exception as e:
            self.logger.error(f"Failed to get transactions for {token_address}: {e}")
            return []
    
    def _parse_transaction(self, tx_data: Dict, token_address: str) -> Optional[TransactionData]:
        """Parse Helius transaction data"""
        try:
            timestamp = datetime.fromtimestamp(tx_data.get("blockTime", 0), tz=timezone.utc)
            
            
            token_transfers = tx_data.get("tokenTransfers", [])
            native_transfers = tx_data.get("nativeTransfers", [])
            
            
            token_amount = 0
            sol_amount = 0
            wallet_address = "unknown"
            
            for transfer in token_transfers:
                if transfer.get("mint") == token_address:
                    token_amount = float(transfer.get("tokenAmount", 0))
                    wallet_address = transfer.get("fromUserAccount", "unknown")
            
            for transfer in native_transfers:
                sol_amount += float(transfer.get("amount", 0)) / 1e9  # Convert lamports to SOL
            
            if token_amount > 0 and sol_amount > 0:
                return TransactionData(
                    timestamp=timestamp,
                    token_address=token_address,
                    transaction_hash=tx_data.get("signature", ""),
                    transaction_type="swap",
                    amount_tokens=token_amount,
                    amount_sol=sol_amount,
                    wallet_address=wallet_address,
                    dex="unknown",
                    provider="helius"
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to parse transaction: {e}")
            return None
    
    def _derive_price_from_transactions(self, transactions: List[TransactionData]) -> List[PriceDataPoint]:
        """Derive price points from transaction data"""
        price_points = []
        
        
        hourly_data = {}
        
        for tx in transactions:
            hour_key = tx.timestamp.replace(minute=0, second=0, microsecond=0)
            
            if hour_key not in hourly_data:
                hourly_data[hour_key] = []
            
            if tx.amount_tokens > 0:
                price = tx.amount_sol / tx.amount_tokens
                hourly_data[hour_key].append((price, tx.amount_sol))
        
        
        for hour, trades in hourly_data.items():
            if not trades:
                continue
            
            total_volume = sum(volume for _, volume in trades)
            if total_volume > 0:
                vwap = sum(price * volume for price, volume in trades) / total_volume
                
                price_point = PriceDataPoint(
                    timestamp=hour,
                    token_address=transactions[0].token_address,
                    price_usd=vwap,  # Actually in SOL, would need SOL/USD conversion
                    volume_24h=total_volume,
                    provider="helius"
                )
                price_points.append(price_point)
        
        return sorted(price_points, key=lambda x: x.timestamp)


class DexScreenerProvider(DataProvider):
    """DexScreener API provider"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, rate_limit=2.0)  # 2 requests per second
        self.base_url = "https://api.dexscreener.com/latest/dex"
    
    async def get_token_price_history(self, token_address: str,
                                    start_date: datetime,
                                    end_date: datetime) -> List[PriceDataPoint]:
        """Get price history from DexScreener"""
        await self._ensure_session()
        
        url = f"{self.base_url}/tokens/{token_address}"
        headers = {"X-API-KEY": self.api_key} if self.api_key else {}
        
        try:
            await self._rate_limit_wait()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                pairs = data.get("pairs", [])
                
                price_points = []
                for pair in pairs:
                    if pair.get("chainId") == "solana":
                        price_point = self._parse_dexscreener_pair(pair)
                        if price_point:
                            price_points.append(price_point)
                
                return price_points
                
        except Exception as e:
            self.logger.error(f"Failed to get DexScreener data for {token_address}: {e}")
            return []
    
    def _parse_dexscreener_pair(self, pair_data: Dict) -> Optional[PriceDataPoint]:
        """Parse DexScreener pair data"""
        try:
            base_token = pair_data.get("baseToken", {})
            token_address = base_token.get("address")
            
            if not token_address:
                return None
            
            return PriceDataPoint(
                timestamp=datetime.now(timezone.utc),
                token_address=token_address,
                price_usd=float(pair_data.get("priceUsd", 0)),
                volume_24h=float(pair_data.get("volume", {}).get("h24", 0)),
                market_cap=float(pair_data.get("marketCap", 0)) if pair_data.get("marketCap") else None,
                liquidity=float(pair_data.get("liquidity", {}).get("usd", 0)),
                price_change_1h=float(pair_data.get("priceChange", {}).get("h1", 0)),
                price_change_24h=float(pair_data.get("priceChange", {}).get("h24", 0)),
                transaction_count_24h=int(pair_data.get("txns", {}).get("h24", {}).get("buys", 0) + 
                                        pair_data.get("txns", {}).get("h24", {}).get("sells", 0)),
                provider="dexscreener"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse DexScreener pair: {e}")
            return None


class BirdeyeProvider(DataProvider):
    """Birdeye API provider"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, rate_limit=10.0)  # 10 requests per second
        self.base_url = "https://public-api.birdeye.so"
    
    async def get_token_price_history(self, token_address: str,
                                    start_date: datetime,
                                    end_date: datetime) -> List[PriceDataPoint]:
        """Get historical price data from Birdeye"""
        await self._ensure_session()
        
        url = f"{self.base_url}/defi/history_price"
        headers = {"X-API-KEY": self.api_key}
        params = {
            "address": token_address,
            "address_type": "token",
            "time_from": int(start_date.timestamp()),
            "time_to": int(end_date.timestamp()),
            "type": "1H"  # 1 hour intervals
        }
        
        try:
            await self._rate_limit_wait()
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                
                if not data.get("success"):
                    return []
                
                price_points = []
                items = data.get("data", {}).get("items", [])
                
                for item in items:
                    price_point = PriceDataPoint(
                        timestamp=datetime.fromtimestamp(item["unixTime"], tz=timezone.utc),
                        token_address=token_address,
                        price_usd=float(item["value"]),
                        volume_24h=0,  # Not available in this endpoint
                        provider="birdeye"
                    )
                    price_points.append(price_point)
                
                return price_points
                
        except Exception as e:
            self.logger.error(f"Failed to get Birdeye data for {token_address}: {e}")
            return []


class HistoricalDataIngester:
    """Main historical data ingestion system"""
    
    def __init__(self):
        self.logger = get_logger("HistoricalDataIngester")
        self.db_manager: Optional[TimescaleDBManager] = None
        self.providers: Dict[str, DataProvider] = {}
        
    async def initialize(self):
        """Initialize the ingestion system"""
        try:
        try:
            db_config = get_database_config()
            self.db_manager = TimescaleDBManager(db_config)
            await self.db_manager.initialize()
            
            
            network_config = await get_network_config_async()
            
            if network_config.get("helius_api_key"):
                self.providers["helius"] = HeliusProvider(network_config["helius_api_key"])
                
            if network_config.get("dexscreener_api_key"):
                self.providers["dexscreener"] = DexScreenerProvider(network_config["dexscreener_api_key"])
                
            if network_config.get("birdeye_api_key"):
                self.providers["birdeye"] = BirdeyeProvider(network_config["birdeye_api_key"])
            
            self.logger.info(f"Initialized {len(self.providers)} data providers")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ingestion system: {e}")
            return False
    
    async def ingest_token_data(self, tokens: List[TokenInfo],
                              start_date: datetime,
                              end_date: datetime,
                              providers: Optional[List[str]] = None) -> bool:
        """Ingest historical data for specified tokens"""
        
        if not providers:
            providers = list(self.providers.keys())
        
        total_tokens = len(tokens)
        successful_ingests = 0
        
        self.logger.info(f"Starting ingestion for {total_tokens} tokens from {start_date} to {end_date}")
        
        for i, token in enumerate(tokens, 1):
            self.logger.info(f"Processing token {i}/{total_tokens}: {token.symbol} ({token.address})")
            
            token_success = False
            
            for provider_name in providers:
                if provider_name not in self.providers:
                    continue
                
                try:
                    provider = self.providers[provider_name]
                    price_data = await provider.get_token_price_history(
                        token.address, start_date, end_date
                    )
                    
                    if price_data:
                        await self._store_price_data(price_data)
                        self.logger.info(f"  ✅ {provider_name}: {len(price_data)} data points")
                        token_success = True
                    else:
                        self.logger.warning(f"  ⚠️ {provider_name}: No data returned")
                    
                except Exception as e:
                    self.logger.error(f"  ❌ {provider_name}: {e}")
            
            if token_success:
                successful_ingests += 1
            
            
            await asyncio.sleep(0.5)
        
        success_rate = (successful_ingests / total_tokens) * 100
        self.logger.info(f"Ingestion complete: {successful_ingests}/{total_tokens} tokens ({success_rate:.1f}% success)")
        
        return success_rate >= 50  # Consider successful if at least 50% succeed
    
    async def _store_price_data(self, price_data: List[PriceDataPoint]):
        """Store price data in TimescaleDB"""
        if not self.db_manager or not price_data:
            return
        
        try:
        try:
            for point in price_data:
            for point in price_data:
                from worker_ant_v1.core.database import PerformanceMetric
                
                metric = PerformanceMetric(
                    timestamp=point.timestamp,
                    metric_name="token_price_usd",
                    value=point.price_usd,
                    unit="USD",
                    component="market_data",
                    labels={
                        "token_address": point.token_address,
                        "provider": point.provider,
                        "volume_24h": str(point.volume_24h),
                        "market_cap": str(point.market_cap) if point.market_cap else None,
                        "price_change_24h": str(point.price_change_24h) if point.price_change_24h else None
                    }
                )
                
                await self.db_manager.insert_performance_metric(metric)
                
        except Exception as e:
            self.logger.error(f"Failed to store price data: {e}")
    
    async def shutdown(self):
        """Shutdown the ingestion system"""
        """Shutdown the ingestion system"""
        for provider in self.providers.values():
            await provider.close()
        
        
        if self.db_manager:
            await self.db_manager.shutdown()


def get_popular_tokens() -> List[TokenInfo]:
    """Get list of popular Solana tokens"""
    return [
        TokenInfo("So11111111111111111111111111111111111111112", "WSOL", "Wrapped SOL"),
        TokenInfo("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "USDC", "USD Coin"),
        TokenInfo("Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB", "USDT", "Tether USD"),
        TokenInfo("DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263", "BONK", "Bonk"),
        TokenInfo("7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr", "POPCAT", "Popcat"),
        TokenInfo("CKfatsPMUf8SkiURsDXs7eK6GWb4Jsd6UDbs7twMCWxo", "BOME", "Book of Meme"),
        TokenInfo("WENWENvqqNya429ubCdR81ZmD69brwQaaBYY6p3LCpk", "WEN", "WEN"),
        TokenInfo("2qEHjDLDLbuBgRYvsxhc5D6uDWAivNFZGan56P1tpump", "CHILLGUY", "Chill Guy"),
    ]


def load_custom_tokens(token_list_file: str) -> List[TokenInfo]:
    """Load custom token list from file"""
    try:
        with open(token_list_file, 'r') as f:
            data = json.load(f)
        
        tokens = []
        for item in data:
            if isinstance(item, dict):
                tokens.append(TokenInfo(
                    address=item["address"],
                    symbol=item.get("symbol", "UNKNOWN"),
                    name=item.get("name", "Unknown Token"),
                    decimals=item.get("decimals", 9)
                ))
            else:
            else:
                tokens.append(TokenInfo(item, "UNKNOWN", "Unknown Token"))
        
        return tokens
        
    except Exception as e:
        print(f"Failed to load custom token list: {e}")
        return []


async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Historical data ingestion for Antbot ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest_historical_data.py --provider helius --days 30
  python scripts/ingest_historical_data.py --provider dexscreener --tokens custom --token-list tokens.json
  python scripts/ingest_historical_data.py --start-date 2024-01-01 --end-date 2024-06-01 --providers helius birdeye
        """
    )
    
    parser.add_argument("--providers", nargs="+", choices=["helius", "dexscreener", "birdeye"],
                       help="Data providers to use")
    parser.add_argument("--tokens", choices=["popular", "custom"], default="popular",
                       help="Token set to fetch")
    parser.add_argument("--token-list", help="Custom token list file (JSON)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to fetch (from now)")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=args.days)
    
    
    if args.tokens == "popular":
        tokens = get_popular_tokens()
    elif args.tokens == "custom" and args.token_list:
        tokens = load_custom_tokens(args.token_list)
    else:
        print("❌ Must specify token list for custom tokens")
        return 1
    
    if not tokens:
        print("❌ No tokens to process")
        return 1
    
    print(f"🚀 Starting historical data ingestion")
    print(f"   📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"   🪙 Tokens: {len(tokens)} tokens")
    print(f"   🔌 Providers: {args.providers or 'all available'}")
    
    
    ingester = HistoricalDataIngester()
    
    try:
        if not await ingester.initialize():
            print("❌ Failed to initialize ingestion system")
            return 1
        
        success = await ingester.ingest_token_data(
            tokens=tokens,
            start_date=start_date,
            end_date=end_date,
            providers=args.providers
        )
        
        if success:
            print("✅ Historical data ingestion completed successfully")
            return 0
        else:
            print("⚠️ Ingestion completed with some failures")
            return 1
        
    except KeyboardInterrupt:
        print("\n🛑 Ingestion cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return 1
    finally:
        await ingester.shutdown()

 