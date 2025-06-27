#!/usr/bin/env python3
"""
HYBRID LAUNCH MONITOR
====================

Shows recent launches first, then monitors for brand new ones.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class HybridLaunchMonitor:
    """Hybrid monitor for recent + new launches"""
    
    def __init__(self):
        self.session = None
        self.seen_tokens = set()
        self.launch_count = 0
        self.start_time = datetime.now()
        self.shown_recent = False
        
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
    
    async def show_recent_launches(self):
        """Show recent launches from the last few hours"""
        
        print("🕐 SHOWING RECENT LAUNCHES (Last 6 hours)")
        print("=" * 60)
        
        recent_tokens = []
        
        # Check pump.fun for recent launches
        try:
            pump_url = "https://frontend-api.pump.fun/coins?offset=0&limit=200&sort=created_timestamp&order=DESC"
            
            print("🔍 Loading recent pump.fun launches...", end=" ")
            
            async with self.session.get(pump_url) as response:
                if response.status == 200:
                    coins = await response.json()
                    
                    now = datetime.now()
                    found_recent = 0
                    
                    for coin in coins:
                        mint = coin.get("mint", "")
                        created_timestamp = coin.get("created_timestamp", 0)
                        
                        if mint and created_timestamp:
                            created_time = datetime.fromtimestamp(created_timestamp / 1000)
                            age_hours = (now - created_time).total_seconds() / 3600
                            
                            # Show tokens from last 6 hours
                            if age_hours <= 6:
                                found_recent += 1
                                
                                recent_tokens.append({
                                    "address": mint,
                                    "name": coin.get("name", "Unknown"),
                                    "symbol": coin.get("symbol", "UNKNOWN"),
                                    "description": coin.get("description", "")[:80],
                                    "market_cap": float(coin.get("usd_market_cap", 0)),
                                    "age_hours": age_hours,
                                    "platform": "Pump.fun",
                                    "created_time": created_time.strftime('%H:%M'),
                                    "website": coin.get("website", ""),
                                    "twitter": coin.get("twitter", "")
                                })
                                
                                self.seen_tokens.add(mint)
                    
                    print(f"Found {found_recent} recent tokens")
                
                else:
                    print(f"Error {response.status}")
        
        except Exception as e:
            print(f"Error: {str(e)[:50]}")
        
        # Sort by age (newest first)
        recent_tokens.sort(key=lambda x: x.get('age_hours', 999))
        
        # Display recent launches
        if recent_tokens:
            print(f"\n🎯 SHOWING TOP 15 RECENT LAUNCHES:")
            print("-" * 80)
            
            for i, token in enumerate(recent_tokens[:15], 1):
                age_hours = token.get('age_hours', 0)
                
                if age_hours < 1:
                    age_display = f"{age_hours * 60:.0f}m ago"
                    freshness = "🔥"
                elif age_hours < 3:
                    age_display = f"{age_hours:.1f}h ago"
                    freshness = "⭐"
                else:
                    age_display = f"{age_hours:.1f}h ago"
                    freshness = "📅"
                
                print(f"\n{freshness} #{i} - {token.get('created_time', 'Unknown')}")
                print(f"   ADDRESS: {token['address']}")
                print(f"   NAME: {token.get('name', 'Unknown')}")
                print(f"   SYMBOL: {token.get('symbol', 'UNKNOWN')}")
                print(f"   AGE: {age_display}")
                
                if token.get('market_cap'):
                    print(f"   MARKET CAP: ${token['market_cap']:,.0f}")
                
                if token.get('description'):
                    print(f"   DESC: {token['description']}")
                
                print(f"   🔗 https://pump.fun/{token['address']}")
        
        print(f"\n" + "=" * 60)
        self.shown_recent = True
    
    async def check_brand_new_launches(self):
        """Check for brand new launches (last 10 minutes)"""
        
        fresh_tokens = []
        
        # Check pump.fun for ultra fresh
        try:
            pump_url = "https://frontend-api.pump.fun/coins?offset=0&limit=50&sort=created_timestamp&order=DESC"
            
            async with self.session.get(pump_url) as response:
                if response.status == 200:
                    coins = await response.json()
                    
                    now = datetime.now()
                    
                    for coin in coins:
                        mint = coin.get("mint", "")
                        created_timestamp = coin.get("created_timestamp", 0)
                        
                        if mint and created_timestamp and mint not in self.seen_tokens:
                            created_time = datetime.fromtimestamp(created_timestamp / 1000)
                            age_minutes = (now - created_time).total_seconds() / 60
                            
                            # Only truly fresh launches
                            if age_minutes <= 15:
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
        
        except Exception as e:
            pass  # Silent fail for continuous monitoring
        
        return fresh_tokens
    
    async def display_brand_new_launch(self, token):
        """Display a brand new launch"""
        
        self.launch_count += 1
        
        age_minutes = token.get('age_minutes', 0)
        
        if age_minutes <= 2:
            freshness = "🔥🔥🔥 BRAND NEW"
            urgency = "ULTRA HOT"
        elif age_minutes <= 5:
            freshness = "🔥🔥 VERY FRESH"
            urgency = "VERY HOT"
        else:
            freshness = "🔥 FRESH"
            urgency = "HOT"
        
        print(f"\n" + "🚨" * 60)
        print(f"🚨 BRAND NEW LAUNCH #{self.launch_count} - {datetime.now().strftime('%H:%M:%S')} 🚨")
        print(f"🚨" * 60)
        print(f"📍 FULL ADDRESS: {token['address']}")
        print(f"🏷️  NAME: {token.get('name', 'Unknown')}")
        print(f"🎯 SYMBOL: {token.get('symbol', 'UNKNOWN')}")
        print(f"⏰ AGE: {age_minutes:.1f} minutes")
        print(f"🔥 STATUS: {freshness}")
        print(f"🚨 URGENCY: {urgency}")
        print(f"🏭 PLATFORM: {token.get('platform', 'Unknown')}")
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
        print(f"🚨" * 60)
        
        # Save
        self.save_launch(token)
    
    def save_launch(self, token):
        """Save launch to file"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/hybrid_launches.txt", "a") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            age = token.get('age_minutes', 0)
            f.write(f"{timestamp} | {token['address']} | {token.get('symbol', 'UNKNOWN')} | {age:.1f}min | {token.get('platform', 'Unknown')}\n")
    
    async def monitor_hybrid(self):
        """Main hybrid monitoring"""
        
        print("🔄 HYBRID LAUNCH MONITOR")
        print("=" * 50)
        print("📊 Phase 1: Show recent launches (last 6 hours)")
        print("⚡ Phase 2: Monitor for brand new launches")
        print("🚨 Will alert on ultra-fresh tokens")
        print("")
        
        # Phase 1: Show recent launches
        if not self.shown_recent:
            await self.show_recent_launches()
            print(f"\n✅ Recent launches loaded. Now monitoring for NEW launches...")
            print("🚨 You'll be alerted when brand new tokens launch!")
        
        # Phase 2: Monitor for brand new launches
        while True:
            try:
                scan_time = datetime.now().strftime('%H:%M:%S')
                runtime = (datetime.now() - self.start_time).total_seconds()
                
                print(f"\n⚡ LIVE SCAN - {scan_time} | Runtime: {runtime:.0f}s | New launches: {self.launch_count}")
                
                # Check for brand new launches
                new_tokens = await self.check_brand_new_launches()
                
                if new_tokens:
                    print(f"🚨🚨🚨 FOUND {len(new_tokens)} BRAND NEW LAUNCHES! 🚨🚨🚨")
                    
                    for token in new_tokens:
                        await self.display_brand_new_launch(token)
                        await asyncio.sleep(1)
                
                else:
                    print("   💤 No brand new launches this scan")
                
                print(f"   ⏳ Next scan in 30 seconds...")
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                print(f"\n⏹️ Hybrid monitor stopped")
                print(f"📊 Total new launches detected: {self.launch_count}")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(60)

async def main():
    """Main function"""
    
    async with HybridLaunchMonitor() as monitor:
        await monitor.monitor_hybrid()

if __name__ == "__main__":
    asyncio.run(main()) 