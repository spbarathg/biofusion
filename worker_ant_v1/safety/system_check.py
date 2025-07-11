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
        """Verify wallet swarm health and balances"""
        try:
            wallets = self.wallet_swarm.wallets
            
            if len(wallets) < self.wallet_swarm.SWARM_SIZE:
                return False, f"Insufficient wallets: {len(wallets)}/{self.wallet_swarm.SWARM_SIZE}"
                
            # Check each wallet's balance
            for address in wallets:
                balance = self.web3.eth.get_balance(address)
                if balance < self.web3.to_wei(self.wallet_swarm.MIN_WALLET_BALANCE, 'ether'):
                    return False, f"Wallet {address} below minimum balance"
                    
            return True, "Wallet health check passed"
            
        except Exception as e:
            return False, f"Wallet health check failed: {str(e)}"
            
    async def _check_network_status(self) -> Tuple[bool, str]:
        """Check network conditions and RPC health"""
        try:
            # Check gas prices
            gas_price = self.web3.eth.gas_price
            if gas_price > self.web3.to_wei(500, 'gwei'):  # Arbitrary high gas threshold
                return False, f"Gas price too high: {gas_price}"
                
            # Check block times
            latest = self.web3.eth.get_block('latest')
            if latest.timestamp < (await self._get_current_time() - 120):  # 2 min delay threshold
                return False, "Network might be congested"
                
            return True, "Network status check passed"
            
        except Exception as e:
            return False, f"Network status check failed: {str(e)}"
            
    async def _check_twitter_stream(self) -> Tuple[bool, str]:
        """Verify Twitter stream processing"""
        try:
            # Test tweet processing
            test_tweet = {
                "id": "test",
                "author_id": "test_author",
                "text": "Test tweet for system check",
                "created_at": await self._get_current_time(),
                "public_metrics": {"like_count": 0, "retweet_count": 0}
            }
            
            score = await self.alpha_extractor.process_tweet(test_tweet)
            if score is None:
                return False, "Tweet processing failed"
                
            return True, "Twitter stream check passed"
            
        except Exception as e:
            return False, f"Twitter stream check failed: {str(e)}"
            
    async def _check_redis_health(self) -> Tuple[bool, str]:
        """Verify Redis performance and data integrity"""
        try:
            # Check write performance
            start = await self._get_current_time()
            self.redis.set("health_check", "1", ex=60)
            if await self._get_current_time() - start > 0.1:  # 100ms threshold
                return False, "Redis write latency too high"
                
            # Check data integrity
            if self.redis.get("health_check") != b"1":
                return False, "Redis data integrity check failed"
                
            return True, "Redis health check passed"
            
        except Exception as e:
            return False, f"Redis health check failed: {str(e)}"
            
    async def _simulate_trade_flow(self) -> Tuple[bool, str]:
        """Run a simulated trade flow without actual execution"""
        try:
            # 1. Get best wallet
            wallet = await self.wallet_swarm.get_best_wallet(0.5)
            if not wallet:
                return False, "No suitable wallet found"
                
            # 2. Mark wallet in trade
            await self.wallet_swarm.mark_wallet_in_trade(wallet)
            
            # 3. Verify wallet lock
            if not self.redis.get(f"wallet:in_trade:{wallet}"):
                return False, "Wallet locking failed"
                
            # 4. Clean up
            self.redis.delete(f"wallet:in_trade:{wallet}")
            
            return True, "Trade flow simulation passed"
            
        except Exception as e:
            return False, f"Trade flow simulation failed: {str(e)}"
            
    async def _get_current_time(self) -> int:
        """Get current timestamp"""
        return int(await asyncio.to_thread(lambda: time.time())) 