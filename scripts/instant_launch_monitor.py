#!/usr/bin/env python3
"""
INSTANT LAUNCH MONITOR
=====================

Monitor for tokens launched RIGHT NOW (last 5-10 minutes only).
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class InstantLaunchMonitor:
    """Monitor for instant fresh launches"""
    
    def __init__(self):
        self.session = None
        self.seen_tokens = set()
        self.launch_count = 0
        self.start_time = datetime.now()
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            resolver=aiohttp.AsyncResolver(),
            use_dns_cache=False
        )
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_pump_fun_fresh(self):
        """Check pump.fun for tokens launched in last 10 minutes"""
        
        fresh_tokens = []
        
        try:
            # Get latest tokens from pump.fun
            pump_url = "https://frontend-api.pump.fun/coins?offset=0&limit=100&sort=created_timestamp&order=DESC"
            
            print("      🔍 Checking pump.fun...", end=" ")
            
            async with self.session.get(pump_url) as response:
                if response.status == 200:
                    coins = await response.json()
                    
                    now = datetime.now()
                    found_fresh = 0
                    
                    for coin in coins:
                        mint = coin.get("mint", "")
                        created_timestamp = coin.get("created_timestamp", 0)
                        
                        if mint and created_timestamp and mint not in self.seen_tokens:
                            # Convert timestamp
                            created_time = datetime.fromtimestamp(created_timestamp / 1000)
                            age_minutes = (now - created_time).total_seconds() / 60
                            
                            # Only tokens from last 10 minutes
                            if age_minutes <= 10:
                                found_fresh += 1
                                
                                fresh_tokens.append({
                                    "address": mint,
                                    "name": coin.get("name", "Unknown"),
                                    "symbol": coin.get("symbol", "UNKNOWN"),
                                    "description": coin.get("description", "")[:100],
                                    "market_cap": float(coin.get("usd_market_cap", 0)),
                                    "age_minutes": age_minutes,
                                    "platform": "Pump.fun",
                                    "created_time": created_time.strftime('%H:%M:%S'),
                                    "website": coin.get("website", ""),
                                    "twitter": coin.get("twitter", "")
                                })
                                
                                self.seen_tokens.add(mint)
                    
                    print(f"Found {found_fresh} fresh tokens")
                
                else:
                    print(f"Error {response.status}")
        
        except Exception as e:
            print(f"Error: {str(e)[:50]}")
        
        return fresh_tokens
    
    async def check_raydium_new_pools(self):
        """Check Raydium for pools created in last few minutes"""
        
        fresh_tokens = []
        
        try:
            print("      🔍 Checking Raydium new pools...", end=" ")
            
            raydium_url = "https://api.raydium.io/v2/main/pairs"
            
            async with self.session.get(raydium_url) as response:
                if response.status == 200:
                    pairs = await response.json()
                    
                    SOL_MINT = "So11111111111111111111111111111111111111112"
                    now = datetime.now()
                    found_fresh = 0
                    
                    # Check first 1000 pairs for recent ones
                    for pair in pairs[:1000]:
                        quote_mint = pair.get("quoteMint", "")
                        base_mint = pair.get("baseMint", "")
                        
                        if (quote_mint == SOL_MINT and 
                            base_mint and 
                            base_mint not in self.seen_tokens):
                            
                            # Check various time fields for recent creation
                            pool_open_time = pair.get("poolOpenTime", 0)
                            
                            if pool_open_time and pool_open_time > 0:
                                try:
                                    pool_time = datetime.fromtimestamp(pool_open_time / 1000)
                                    age_minutes = (now - pool_time).total_seconds() / 60
                                    
                                    # Only pools from last 15 minutes
                                    if age_minutes <= 15:
                                        found_fresh += 1
                                        liquidity = float(pair.get("liquidity", 0))
                                        
                                        fresh_tokens.append({
                                            "address": base_mint,
                                            "name": pair.get("name", "Unknown"),
                                            "symbol": "NEW",
                                            "liquidity": liquidity,
                                            "price": float(pair.get("price", 0)),
                                            "age_minutes": age_minutes,
                                            "platform": "Raydium",
                                            "created_time": pool_time.strftime('%H:%M:%S')
                                        })
                                        
                                        self.seen_tokens.add(base_mint)
                                
                                except:
                                    continue
                    
                    print(f"Found {found_fresh} new pools")
                
                else:
                    print(f"Error {response.status}")
        
        except Exception as e:
            print(f"Error: {str(e)[:50]}")
        
        return fresh_tokens
    
    async def display_instant_launch(self, token):
        """Display an instant launch"""
        
        self.launch_count += 1
        
        age_minutes = token.get('age_minutes', 0)
        platform = token.get('platform', 'Unknown')
        
        # Ultra fresh indicator
        if age_minutes <= 1:
            freshness = "🔥🔥🔥 JUST LAUNCHED (< 1 min)"
            urgency = "ULTRA HOT"
        elif age_minutes <= 3:
            freshness = "🔥🔥 SUPER FRESH (< 3 min)"
            urgency = "VERY HOT"
        elif age_minutes <= 5:
            freshness = "🔥 FRESH (< 5 min)"
            urgency = "HOT"
        else:
            freshness = "⏰ Recent (< 10 min)"
            urgency = "WARM"
        
        print(f"\n" + "🚀" * 60)
        print(f"⚡ INSTANT LAUNCH #{self.launch_count} - {datetime.now().strftime('%H:%M:%S')} ⚡")
        print(f"🚀" * 60)
        print(f"📍 FULL ADDRESS: {token['address']}")
        print(f"🏷️  TOKEN NAME: {token.get('name', 'Unknown')}")
        print(f"🎯 SYMBOL: {token.get('symbol', 'UNKNOWN')}")
        print(f"⏰ AGE: {age_minutes:.1f} minutes ago")
        print(f"🔥 FRESHNESS: {freshness}")
        print(f"🚨 URGENCY: {urgency}")
        print(f"🏭 PLATFORM: {platform}")
        print(f"🕐 CREATED: {token.get('created_time', 'Unknown')}")
        
        if token.get('liquidity'):
            print(f"💰 LIQUIDITY: ${token['liquidity']:,.2f}")
        
        if token.get('market_cap'):
            print(f"📊 MARKET CAP: ${token['market_cap']:,.2f}")
        
        if token.get('price'):
            print(f"💲 PRICE: {token['price']:.10f} SOL")
        
        if token.get('description'):
            print(f"📝 DESCRIPTION: {token['description']}")
        
        if token.get('website'):
            print(f"🌐 WEBSITE: {token['website']}")
        
        if token.get('twitter'):
            print(f"🐦 TWITTER: {token['twitter']}")
        
        print(f"🔗 VERIFY: https://solscan.io/token/{token['address']}")
        print(f"🔗 DEX: https://dexscreener.com/solana/{token['address']}")
        print(f"🔗 PUMP: https://pump.fun/{token['address']}")
        
        print(f"⏰ RUNTIME: {(datetime.now() - self.start_time).total_seconds():.0f}s")
        print(f"🚀" * 60)
        
        # Save instantly
        self.save_instant_launch(token)
    
    def save_instant_launch(self, token):
        """Save instant launch"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/instant_launches.txt", "a") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            age = token.get('age_minutes', 0)
            f.write(f"{timestamp} | {token['address']} | {token.get('symbol', 'UNKNOWN')} | {age:.1f}min | {token.get('platform', 'Unknown')}\n")
    
    async def monitor_instant_launches(self):
        """Main monitoring loop"""
        
        print("⚡ INSTANT TOKEN LAUNCH MONITOR ⚡")
        print("=" * 70)
        print("🔥 MONITORING TOKENS LAUNCHED RIGHT NOW")
        print("⏰ Focus: Last 10 minutes only")
        print("🎯 Sources: Pump.fun + Raydium new pools")
        print("⚡ Scanning every 20 seconds")
        print("🚨 Maximum freshness detection")
        print("")
        
        while True:
            try:
                scan_time = datetime.now().strftime('%H:%M:%S')
                runtime = (datetime.now() - self.start_time).total_seconds()
                
                print(f"\n⚡ INSTANT SCAN - {scan_time} | Runtime: {runtime:.0f}s | Detected: {self.launch_count}")
                
                # Check pump.fun for ultra fresh launches
                pump_tokens = await self.check_pump_fun_fresh()
                
                # Check Raydium for new pools
                raydium_tokens = await self.check_raydium_new_pools()
                
                # Combine all fresh tokens
                all_fresh = pump_tokens + raydium_tokens
                
                if all_fresh:
                    # Sort by freshness (newest first)
                    all_fresh.sort(key=lambda x: x.get('age_minutes', 999))
                    
                    print(f"      🎉 FOUND {len(all_fresh)} INSTANT LAUNCHES!")
                    
                    for token in all_fresh:
                        await self.display_instant_launch(token)
                        await asyncio.sleep(0.5)  # Brief pause between displays
                
                else:
                    print("      💤 No instant launches this scan")
                
                print(f"      ⏳ Next scan in 20 seconds...")
                await asyncio.sleep(20)
                
            except KeyboardInterrupt:
                print(f"\n⏹️ Instant monitor stopped")
                print(f"📊 Total instant launches: {self.launch_count}")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(30)

async def main():
    """Main function"""
    
    async with InstantLaunchMonitor() as monitor:
        await monitor.monitor_instant_launches()

if __name__ == "__main__":
    asyncio.run(main()) 