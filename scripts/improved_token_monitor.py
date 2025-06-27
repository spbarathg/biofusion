#!/usr/bin/env python3
"""
IMPROVED TOKEN LAUNCH MONITOR
============================

Real-time monitor that uses pump.fun's direct API for reliable detection.
Shows ONLY brand new token launches as they happen.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class ImprovedTokenMonitor:
    """Improved token monitor using direct pump.fun API"""
    
    def __init__(self):
        self.session = None
        self.seen_tokens = set()
        self.launch_count = 0
        self.start_time = datetime.now()
        self.last_check = None
        
    async def __aenter__(self):
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
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_for_new_launches(self):
        """Check pump.fun for brand new token launches"""
        
        new_launches = []
        
        try:
            url = "https://frontend-api.pump.fun/coins?offset=0&limit=50&sort=created_timestamp&order=DESC&includeNsfw=true"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    coins = await response.json()
                    now = datetime.now()
                    
                    for coin in coins:
                        mint = coin.get("mint", "")
                        created_timestamp = coin.get("created_timestamp", 0)
                        
                        if mint and created_timestamp and mint not in self.seen_tokens:
                            created_time = datetime.fromtimestamp(created_timestamp / 1000)
                            age_seconds = (now - created_time).total_seconds()
                            
                            # Only show tokens less than 10 minutes old
                            if age_seconds <= 600:
                                new_launches.append({
                                    "address": mint,
                                    "name": coin.get("name", "Unknown"),
                                    "symbol": coin.get("symbol", "UNKNOWN"),
                                    "description": coin.get("description", "")[:150],
                                    "market_cap": float(coin.get("usd_market_cap", 0)),
                                    "age_seconds": age_seconds,
                                    "created_time": created_time,
                                    "website": coin.get("website", ""),
                                    "twitter": coin.get("twitter", ""),
                                    "telegram": coin.get("telegram", ""),
                                    "virtual_sol_reserves": coin.get("virtual_sol_reserves", 0),
                                    "virtual_token_reserves": coin.get("virtual_token_reserves", 0),
                                    "reply_count": coin.get("reply_count", 0)
                                })
                                
                                self.seen_tokens.add(mint)
        
        except Exception as e:
            print(f"🚨 API Error: {str(e)[:50]}...")
        
        # Sort by creation time (newest first)
        new_launches.sort(key=lambda x: x["created_time"], reverse=True)
        return new_launches
    
    def display_launch(self, token):
        """Display new token launch with clean formatting"""
        
        self.launch_count += 1
        age_seconds = token["age_seconds"]
        created_time = token["created_time"]
        
        # Determine urgency level
        if age_seconds <= 30:
            urgency_emoji = "🚨🚨🚨"
            urgency_text = "ULTRA FRESH"
            age_display = f"{age_seconds:.0f} seconds ago"
        elif age_seconds <= 120:
            urgency_emoji = "🚨🚨"
            urgency_text = "VERY FRESH"
            age_display = f"{age_seconds:.0f} seconds ago"
        elif age_seconds <= 300:
            urgency_emoji = "🚨"
            urgency_text = "FRESH"
            age_display = f"{age_seconds/60:.1f} minutes ago"
        else:
            urgency_emoji = "⚡"
            urgency_text = "NEW"
            age_display = f"{age_seconds/60:.1f} minutes ago"
        
        print(f"\n{'🚀' * 60}")
        print(f"🚀 NEW TOKEN LAUNCH #{self.launch_count} - {datetime.now().strftime('%H:%M:%S')} 🚀")
        print(f"{'🚀' * 60}")
        
        print(f"📍 ADDRESS: {token['address']}")
        print(f"🏷️  NAME: {token['name']}")
        print(f"🎯 SYMBOL: {token['symbol']}")
        print(f"⏰ AGE: {age_display}")
        print(f"{urgency_emoji} FRESHNESS: {urgency_text}")
        print(f"🕐 CREATED: {created_time.strftime('%H:%M:%S')}")
        
        if token['market_cap'] > 0:
            print(f"💰 MARKET CAP: ${token['market_cap']:,.2f}")
        
        if token['virtual_sol_reserves']:
            sol_reserves = float(token['virtual_sol_reserves']) / 1e9
            print(f"💎 SOL RESERVES: {sol_reserves:.2f} SOL")
        
        if token['reply_count'] > 0:
            print(f"💬 REPLIES: {token['reply_count']}")
        
        if token['description']:
            print(f"📝 DESC: {token['description']}")
        
        # Links
        print(f"🔗 PUMP: https://pump.fun/{token['address']}")
        print(f"🔗 SCAN: https://solscan.io/token/{token['address']}")
        print(f"🔗 DEX: https://dexscreener.com/solana/{token['address']}")
        
        if token['website']:
            print(f"🌐 WEB: {token['website']}")
        if token['twitter']:
            print(f"🐦 X: {token['twitter']}")
        if token['telegram']:
            print(f"📱 TG: {token['telegram']}")
        
        # Stats
        runtime = (datetime.now() - self.start_time).total_seconds()
        rate = self.launch_count / (runtime / 60) if runtime > 0 else 0
        print(f"📊 TOTAL: {self.launch_count} | RUNTIME: {runtime:.0f}s | RATE: {rate:.1f}/min")
        
        print(f"{'🚀' * 60}")
        
        self.save_launch(token)
    
    def save_launch(self, token):
        """Save launch to log file"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/live_token_addresses.txt", "a") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            age = token["age_seconds"]
            market_cap = token["market_cap"]
            f.write(f"{timestamp} | {token['address']} | {token['symbol']} | {token['name']} | {age:.0f}s | ${market_cap:.2f}\n")
    
    async def monitor_launches(self):
        """Main monitoring loop"""
        
        print("🚀 IMPROVED TOKEN LAUNCH MONITOR")
        print("=" * 50)
        print("🎯 Real-time pump.fun API monitoring")
        print("⚡ Ultra-fast detection (3-second scans)")
        print("🚨 Shows only brand new launches")
        print("📊 Clean, focused display")
        print("")
        
        scan_count = 0
        no_new_count = 0
        
        try:
            while True:
                scan_count += 1
                scan_time = datetime.now().strftime('%H:%M:%S')
                
                # Check for new launches
                new_launches = await self.check_for_new_launches()
                
                if new_launches:
                    no_new_count = 0
                    print(f"\n🔥 SCAN #{scan_count} - {scan_time} | Found {len(new_launches)} new launches!")
                    
                    for token in new_launches:
                        self.display_launch(token)
                        await asyncio.sleep(0.5)  # Brief pause between displays
                else:
                    no_new_count += 1
                    if no_new_count % 10 == 0:
                        runtime = (datetime.now() - self.start_time).total_seconds()
                        print(f"📡 SCAN #{scan_count} - {scan_time} | No new launches | Runtime: {runtime:.0f}s | Total: {self.launch_count}")
                    else:
                        print(f"📡 SCAN #{scan_count} - {scan_time} | No new launches", end="\r")
                
                # Wait before next scan
                await asyncio.sleep(3)
        
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitor stopped")
            runtime = (datetime.now() - self.start_time).total_seconds()
            rate = self.launch_count / (runtime / 60) if runtime > 0 else 0
            print(f"📊 Final Stats: {self.launch_count} launches in {runtime:.0f}s ({rate:.1f}/min)")
        except Exception as e:
            print(f"\n❌ Error: {e}")

async def main():
    """Main function"""
    
    async with ImprovedTokenMonitor() as monitor:
        await monitor.monitor_launches()

if __name__ == "__main__":
    print("🚀 Starting Improved Token Launch Monitor!")
    asyncio.run(main()) 