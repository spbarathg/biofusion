#!/usr/bin/env python3
"""
DEBUG TOKEN MONITOR
==================

Debug version to see what's happening with token detection.
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
import os
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class DebugTokenMonitor:
    """Debug version of token monitor"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.session = None
        self.logger = logging.getLogger("DebugTokenMonitor")
        self.known_tokens = set()
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            resolver=aiohttp.AsyncResolver(),
            use_dns_cache=False
        )
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Content-Type": "application/json"},
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_raydium_api(self):
        """Test and debug Raydium API"""
        
        print("\n🔍 Testing Raydium API...")
        
        try:
            raydium_url = "https://api.raydium.io/v2/main/pairs"
            
            async with self.session.get(raydium_url) as response:
                print(f"Status: {response.status}")
                
                if response.status == 200:
                    pairs = await response.json()
                    print(f"Found {len(pairs)} total pairs")
                    
                    # Show first few pairs for debugging
                    print("\n📊 First 3 pairs:")
                    for i, pair in enumerate(pairs[:3]):
                        print(f"\nPair {i+1}:")
                        print(f"  Base Symbol: {pair.get('baseSymbol', 'Unknown')}")
                        print(f"  Base Mint: {pair.get('baseMint', 'Unknown')}")
                        print(f"  Pool Open Time: {pair.get('poolOpenTime', 'Unknown')}")
                        print(f"  Liquidity: {pair.get('liquidity', 'Unknown')}")
                        print(f"  Price: {pair.get('price', 'Unknown')}")
                        
                        # Check age
                        pool_open_time = pair.get("poolOpenTime", 0)
                        if pool_open_time > 0:
                            pool_time = datetime.fromtimestamp(pool_open_time / 1000)
                            age_hours = (datetime.now() - pool_time).total_seconds() / 3600
                            print(f"  Age: {age_hours:.1f} hours")
                    
                    # Count recent tokens
                    recent_count = 0
                    for pair in pairs:
                        pool_open_time = pair.get("poolOpenTime", 0)
                        if pool_open_time > 0:
                            pool_time = datetime.fromtimestamp(pool_open_time / 1000)
                            age_hours = (datetime.now() - pool_time).total_seconds() / 3600
                            
                            if age_hours < 24:  # Less than 24 hours
                                recent_count += 1
                    
                    print(f"\n📈 Tokens launched in last 24 hours: {recent_count}")
                    
                    return pairs
                
                else:
                    print(f"❌ API Error: {response.status}")
                    text = await response.text()
                    print(f"Response: {text[:200]}...")
        
        except Exception as e:
            print(f"❌ Exception: {e}")
            
        return []
    
    async def check_new_tokens_relaxed(self) -> list:
        """Check for new tokens with relaxed criteria"""
        
        new_tokens = []
        
        try:
            raydium_url = "https://api.raydium.io/v2/main/pairs"
            
            async with self.session.get(raydium_url) as response:
                if response.status == 200:
                    pairs = await response.json()
                    
                    checked_count = 0
                    for pair in pairs[:20]:  # Check first 20 pairs
                        checked_count += 1
                        base_mint = pair.get("baseMint")
                        base_symbol = pair.get("baseSymbol", "UNKNOWN")
                        
                        if base_mint and base_mint not in self.known_tokens:
                            # RELAXED: Check if pool is recent (24 hours instead of 2)
                            pool_open_time = pair.get("poolOpenTime", 0)
                            if pool_open_time > 0:
                                pool_time = datetime.fromtimestamp(pool_open_time / 1000)
                                age_hours = (datetime.now() - pool_time).total_seconds() / 3600
                                
                                if age_hours < 24:  # Less than 24 hours old (was 2)
                                    token_info = {
                                        "address": base_mint,
                                        "symbol": base_symbol,
                                        "name": pair.get("baseName", "Unknown"),
                                        "price": float(pair.get("price", 0)),
                                        "liquidity": float(pair.get("liquidity", 0)),
                                        "platform": "Raydium",
                                        "age_hours": age_hours,
                                        "pool_open_time": pool_time.isoformat()
                                    }
                                    
                                    new_tokens.append(token_info)
                                    self.known_tokens.add(base_mint)
                    
                    print(f"🔍 Checked {checked_count} pairs, found {len(new_tokens)} new tokens")
                
                else:
                    print(f"❌ Raydium API returned {response.status}")
        
        except Exception as e:
            print(f"❌ Error checking tokens: {e}")
        
        return new_tokens
    
    async def run_debug_cycle(self):
        """Run one debug cycle"""
        
        print("\n" + "="*50)
        print(f"🔍 DEBUG CYCLE - {datetime.now().strftime('%H:%M:%S')}")
        print("="*50)
        
        # Test API first
        pairs = await self.test_raydium_api()
        
        if pairs:
            # Check for new tokens
            new_tokens = await self.check_new_tokens_relaxed()
            
            if new_tokens:
                print(f"\n🎉 FOUND {len(new_tokens)} NEW TOKENS:")
                
                for i, token in enumerate(new_tokens, 1):
                    print(f"\n🚀 Token #{i}:")
                    print(f"   Symbol: {token['symbol']}")
                    print(f"   Address: {token['address']}")
                    print(f"   Liquidity: ${token['liquidity']:,.2f}")
                    print(f"   Age: {token['age_hours']:.1f} hours")
                    print(f"   Price: ${token['price']:.8f}")
            else:
                print("\n💤 No new tokens found this cycle")
        
        print(f"\n⏱️ Next check in 20 seconds...")

async def main():
    """Main debug function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get API key
    helius_api_key = os.getenv("HELIUS_API_KEY")
    if not helius_api_key:
        print("Using default API key for testing...")
        helius_api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
    
    print("🐛 DEBUG TOKEN MONITOR")
    print("=" * 40)
    print("This version will show:")
    print("• What Raydium API returns")
    print("• Why tokens are/aren't detected")
    print("• Relaxed filtering (24h instead of 2h)")
    print("")
    
    async with DebugTokenMonitor(helius_api_key) as monitor:
        try:
            # Run a few debug cycles
            for cycle in range(3):
                await monitor.run_debug_cycle()
                if cycle < 2:  # Don't wait after last cycle
                    await asyncio.sleep(20)
            
            print("\n✅ Debug complete!")
            
        except KeyboardInterrupt:
            print("\n⏹️ Debug stopped by user")

if __name__ == "__main__":
    asyncio.run(main()) 