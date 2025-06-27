#!/usr/bin/env python3
"""
PUMP.FUN LIVE MONITOR
====================

Multiple endpoint monitor for pump.fun fresh launches.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class PumpFunLiveMonitor:
    """Live monitor for pump.fun launches"""
    
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
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin"
        }
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers=headers,
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_pump_fun_direct(self):
        """Direct check of pump.fun data"""
        
        fresh_tokens = []
        
        try:
            # Try the main pump.fun API
            url = "https://frontend-api.pump.fun/coins?offset=0&limit=100&sort=created_timestamp&order=DESC&includeNsfw=true"
            
            print(f"      🎯 Direct pump.fun check...", end=" ")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    coins = await response.json()
                    print(f"Found {len(coins)} total coins")
                    
                    now = datetime.now()
                    ultra_fresh = 0
                    
                    for coin in coins:
                        mint = coin.get("mint", "")
                        created_timestamp = coin.get("created_timestamp", 0)
                        
                        if mint and created_timestamp and mint not in self.seen_tokens:
                            created_time = datetime.fromtimestamp(created_timestamp / 1000)
                            age_seconds = (now - created_time).total_seconds()
                            age_minutes = age_seconds / 60
                            
                            # Show tokens from last hour but highlight ultra fresh ones
                            if age_minutes <= 60:
                                ultra_fresh += 1
                                
                                # Special marking for ultra fresh
                                if age_seconds <= 60:
                                    freshness_level = "🔥🔥🔥 SECONDS OLD"
                                elif age_minutes <= 5:
                                    freshness_level = "🔥🔥 MINUTES OLD"
                                elif age_minutes <= 30:
                                    freshness_level = "🔥 RECENT"
                                else:
                                    freshness_level = "⏰ HOUR OLD"
                                
                                fresh_tokens.append({
                                    "address": mint,
                                    "name": coin.get("name", "Unknown"),
                                    "symbol": coin.get("symbol", "UNKNOWN"),
                                    "description": coin.get("description", "")[:100],
                                    "market_cap": float(coin.get("usd_market_cap", 0)),
                                    "age_minutes": age_minutes,
                                    "age_seconds": age_seconds,
                                    "freshness_level": freshness_level,
                                    "platform": "Pump.fun",
                                    "created_time": created_time.strftime('%H:%M:%S'),
                                    "website": coin.get("website", ""),
                                    "twitter": coin.get("twitter", "")
                                })
                                
                                self.seen_tokens.add(mint)
                    
                    print(f"      ✅ Found {ultra_fresh} fresh tokens")
                
                else:
                    print(f"Status {response.status}")
        
        except Exception as e:
            print(f"Error: {str(e)[:50]}")
        
        return fresh_tokens
    
    async def display_fresh_launch(self, token):
        """Display fresh launch with enhanced info"""
        
        self.launch_count += 1
        
        age_minutes = token.get('age_minutes', 0)
        age_seconds = token.get('age_seconds', age_minutes * 60)
        freshness_level = token.get('freshness_level', 'Unknown')
        
        # Age display
        if age_seconds <= 60:
            age_display = f"{age_seconds:.0f} seconds ago"
            urgency = "🚨🚨🚨 ULTRA URGENT"
        elif age_minutes <= 5:
            age_display = f"{age_minutes:.1f} minutes ago"
            urgency = "🚨🚨 VERY URGENT"
        elif age_minutes <= 15:
            age_display = f"{age_minutes:.1f} minutes ago"
            urgency = "🚨 URGENT"
        else:
            age_display = f"{age_minutes:.1f} minutes ago"
            urgency = "⏰ Recent"
        
        print(f"\n" + "⚡" * 70)
        print(f"⚡ FRESH PUMP.FUN LAUNCH #{self.launch_count} - {datetime.now().strftime('%H:%M:%S')} ⚡")
        print(f"⚡" * 70)
        print(f"📍 FULL ADDRESS: {token['address']}")
        print(f"🏷️  NAME: {token.get('name', 'Unknown')}")
        print(f"🎯 SYMBOL: {token.get('symbol', 'UNKNOWN')}")
        print(f"⏰ AGE: {age_display}")
        print(f"🔥 FRESHNESS: {freshness_level}")
        print(f"🚨 URGENCY: {urgency}")
        print(f"🕐 CREATED: {token.get('created_time', 'Unknown')}")
        
        if token.get('market_cap'):
            print(f"📊 MARKET CAP: ${token['market_cap']:,.2f}")
        
        if token.get('description'):
            print(f"📝 DESCRIPTION: {token['description']}")
        
        if token.get('website'):
            print(f"🌐 WEBSITE: {token['website']}")
        
        if token.get('twitter'):
            print(f"🐦 TWITTER: {token['twitter']}")
        
        print(f"🔗 PUMP.FUN: https://pump.fun/{token['address']}")
        print(f"🔗 SOLSCAN: https://solscan.io/token/{token['address']}")
        print(f"🔗 DEXSCREENER: https://dexscreener.com/solana/{token['address']}")
        
        print(f"⏰ MONITOR RUNTIME: {(datetime.now() - self.start_time).total_seconds():.0f}s")
        print(f"⚡" * 70)
        
        self.save_launch(token)
    
    def save_launch(self, token):
        """Save launch"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/pump_fun_fresh.txt", "a") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            age = token.get('age_minutes', 0)
            f.write(f"{timestamp} | {token['address']} | {token.get('symbol', 'UNKNOWN')} | {age:.1f}min\n")
    
    async def monitor_pump_fun_live(self):
        """Main pump.fun monitoring"""
        
        print("🔥 PUMP.FUN LIVE LAUNCH MONITOR")
        print("=" * 60)
        print("⚡ Monitoring pump.fun for fresh launches")
        print("🎯 Direct API connection")
        print("🚨 Ultra-fresh detection (seconds old)")
        print("⏱️  Scanning every 15 seconds")
        print("")
        
        while True:
            try:
                scan_time = datetime.now().strftime('%H:%M:%S')
                runtime = (datetime.now() - self.start_time).total_seconds()
                
                print(f"\n🔥 PUMP.FUN SCAN - {scan_time} | Runtime: {runtime:.0f}s | Detected: {self.launch_count}")
                
                # Check for fresh tokens
                fresh_tokens = await self.check_pump_fun_direct()
                
                if fresh_tokens:
                    # Sort by freshness (newest first)
                    fresh_tokens.sort(key=lambda x: x.get('age_seconds', x.get('age_minutes', 999) * 60))
                    
                    print(f"🎉🎉🎉 FOUND {len(fresh_tokens)} FRESH LAUNCHES! 🎉🎉🎉")
                    
                    for token in fresh_tokens[:10]:  # Show top 10
                        await self.display_fresh_launch(token)
                        await asyncio.sleep(0.5)
                
                else:
                    print("   💤 No fresh launches detected")
                
                print(f"   ⏳ Next scan in 15 seconds...")
                await asyncio.sleep(15)
                
            except KeyboardInterrupt:
                print(f"\n⏹️ Pump.fun monitor stopped")
                print(f"📊 Total launches detected: {self.launch_count}")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(30)

async def main():
    """Main function"""
    
    async with PumpFunLiveMonitor() as monitor:
        await monitor.monitor_pump_fun_live()

if __name__ == "__main__":
    asyncio.run(main()) 