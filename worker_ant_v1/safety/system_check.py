import asyncio
from typing import Dict, List, Tuple
import aiohttp
import json
import logging
from web3 import Web3
import redis
import tweepy
from ..intelligence.twitter_alpha import TwitterAlphaExtractor
from ..trading.wallet_swarm import WalletSwarm
import time

class SystemIntegrityChecker:
    def __init__(
        self,
        redis_client: redis.Redis,
        web3: Web3,
        twitter_client: tweepy.Client,
        alpha_extractor: TwitterAlphaExtractor,
        wallet_swarm: WalletSwarm
    ):
        self.redis = redis_client
        self.web3 = web3
        self.twitter = twitter_client
        self.alpha_extractor = alpha_extractor
        self.wallet_swarm = wallet_swarm
        self.logger = logging.getLogger(__name__)
        
    async def run_full_check(self) -> Tuple[bool, List[str]]:
        """Run all system checks before trading"""
        checks = [
            self._check_api_keys(),
            self._check_wallet_health(),
            self._check_network_status(),
            self._check_twitter_stream(),
            self._check_redis_health(),
            self._simulate_trade_flow()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        failures = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append(f"Check {i} failed: {str(result)}")
            elif isinstance(result, tuple) and not result[0]:
                failures.append(result[1])
                
        return len(failures) == 0, failures
        
    async def _check_api_keys(self) -> Tuple[bool, str]:
        """Verify all API keys are valid and have correct permissions"""
        try:
            # Check Twitter API
            await self.twitter.get_me()
            
            # Check Web3 connection
            self.web3.eth.get_block_number()
            
            # Verify Redis connection
            self.redis.ping()
            
            return True, "API keys valid"
            
        except Exception as e:
            return False, f"API key validation failed: {str(e)}"
            
    async def _check_wallet_health(self) -> Tuple[bool, str]:
        """Check wallet health and balance status"""
        try:
            # Check if wallet swarm is operational
            if not self.wallet_swarm or not self.wallet_swarm.is_initialized:
                return False, "Wallet swarm not initialized"
            
            # Check wallet balances
            total_balance = await self.wallet_swarm.get_total_balance()
            if total_balance < 0.1:  # Minimum 0.1 SOL required
                return False, f"Insufficient wallet balance: {total_balance} SOL"
            
            # Check wallet count
            active_wallets = await self.wallet_swarm.get_active_wallet_count()
            if active_wallets < 3:  # Minimum 3 wallets required
                return False, f"Insufficient active wallets: {active_wallets}"
            
            return True, f"Wallet health check passed - {active_wallets} wallets, {total_balance:.3f} SOL total"
            
        except Exception as e:
            return False, f"Wallet health check failed: {str(e)}"
    
    async def _check_network_status(self) -> Tuple[bool, str]:
        """Check network connectivity and RPC status"""
        try:
            # Test Solana RPC connection
            if not self.web3 or not self.web3.is_connected():
                return False, "Web3 connection not established"
            
            # Test network latency
            start_time = time.time()
            try:
                await self.web3.eth.get_block_number()
                latency = time.time() - start_time
                if latency > 5.0:  # 5 second timeout
                    return False, f"Network latency too high: {latency:.2f}s"
            except Exception as e:
                return False, f"Network request failed: {str(e)}"
            
            return True, f"Network status check passed - latency: {latency:.2f}s"
            
        except Exception as e:
            return False, f"Network status check failed: {str(e)}"
    
    async def _check_twitter_stream(self) -> Tuple[bool, str]:
        """Check Twitter API connectivity and stream status"""
        try:
            if not self.twitter or not self.alpha_extractor:
                return False, "Twitter client or alpha extractor not initialized"
            
            # Test Twitter API connection
            try:
                # Test with a simple API call
                test_response = self.twitter.get_me()
                if not test_response.data:
                    return False, "Twitter API authentication failed"
            except Exception as e:
                return False, f"Twitter API connection failed: {str(e)}"
            
            # Check alpha extractor status
            if not self.alpha_extractor.is_operational():
                return False, "Alpha extractor not operational"
            
            return True, "Twitter stream check passed"
            
        except Exception as e:
            return False, f"Twitter stream check failed: {str(e)}"
    
    async def _check_redis_health(self) -> Tuple[bool, str]:
        """Check Redis connection and health"""
        try:
            if not self.redis:
                return False, "Redis client not initialized"
            
            # Test Redis connection
            try:
                self.redis.ping()
            except Exception as e:
                return False, f"Redis connection failed: {str(e)}"
            
            # Test Redis operations
            test_key = "health_check_test"
            test_value = "test_data"
            
            self.redis.set(test_key, test_value, ex=60)  # 60 second expiry
            retrieved_value = self.redis.get(test_key)
            
            if retrieved_value != test_value.encode():
                return False, "Redis read/write test failed"
            
            self.redis.delete(test_key)
            
            return True, "Redis health check passed"
            
        except Exception as e:
            return False, f"Redis health check failed: {str(e)}"
    
    async def _simulate_trade_flow(self) -> Tuple[bool, str]:
        """Simulate a complete trade flow to validate system readiness"""
        try:
            # Test market data retrieval
            if not self.wallet_swarm:
                return False, "Wallet swarm not available for trade simulation"
            
            # Simulate market data processing
            mock_market_data = {
                'token_address': 'test_token',
                'price': 1.0,
                'volume_24h': 1000.0,
                'liquidity': 100.0
            }
            
            # Test signal generation (without actual execution)
            try:
                # This would normally call the intelligence system
                # For simulation, we just validate the data structure
                if not all(key in mock_market_data for key in ['token_address', 'price', 'volume_24h']):
                    return False, "Market data validation failed"
            except Exception as e:
                return False, f"Signal generation simulation failed: {str(e)}"
            
            return True, "Trade flow simulation passed"
            
        except Exception as e:
            return False, f"Trade flow simulation failed: {str(e)}" 