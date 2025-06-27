#!/usr/bin/env python3
"""
DIRECT PUMP.FUN SCRAPER
======================

Scrapes pump.fun website directly to get real-time token launches.
This bypasses API issues and gets data from the actual website.
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class PumpFunScraper:
    """Direct pump.fun website scraper"""
    
    def __init__(self):
        self.session = None
        self.seen_tokens = set()
        self.launch_count = 0
        self.start_time = datetime.now()
        
    async def __aenter__(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none"
        }
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def try_api_endpoints(self):
        """Try multiple pump.fun API endpoints"""
        
        endpoints = [
            "https://frontend-api.pump.fun/coins?offset=0&limit=50&sort=created_timestamp&order=DESC&includeNsfw=true",
            "https://frontend-api.pump.fun/coins?offset=0&limit=50&sort=created_timestamp&order=DESC",
            "https://frontend-api.pump.fun/coins/latest",
            "https://api.pump.fun/coins?offset=0&limit=50&sort=created_timestamp&order=DESC",
            "https://pump.fun/api/coins?offset=0&limit=50&sort=created_timestamp&order=DESC"
        ]
        
        for i, endpoint in enumerate(endpoints, 1):
            print(f"🎯 Trying endpoint {i}/{len(endpoints)}: {endpoint.split('/')[-1]}")
            
            try:
                async with self.session.get(endpoint) as response:
                    print(f"   Status: {response.status}")
                    
                    if response.status == 200:
                        try:
                            data = await response.json()
                            if isinstance(data, list) and len(data) > 0:
                                print(f"   ✅ SUCCESS! Found {len(data)} tokens")
                                return data
                            elif isinstance(data, dict) and data.get('coins'):
                                coins = data['coins']
                                print(f"   ✅ SUCCESS! Found {len(coins)} tokens")
                                return coins
                        except:
                            text = await response.text()
                            print(f"   ❌ Invalid JSON response")
                    else:
                        print(f"   ❌ HTTP {response.status}")
                        
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:50]}")
            
            await asyncio.sleep(1)
        
        return None
    
    async def scrape_website_directly(self):
        """Try to scrape the pump.fun website directly"""
        
        print("🌐 Trying direct website scraping...")
        
        try:
            async with self.session.get("https://pump.fun") as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Look for JSON data in script tags
                    soup = BeautifulSoup(html, 'html.parser')
                    scripts = soup.find_all('script')
                    
                    for script in scripts:
                        if script.string and 'coins' in script.string:
                            # Try to extract JSON data
                            try:
                                # Look for patterns like window.__INITIAL_STATE__ or similar
                                content = script.string
                                
                                # Try to find JSON-like patterns
                                import re
                                json_matches = re.findall(r'\{[^{}]*"mint"[^{}]*\}', content)
                                
                                if json_matches:
                                    print(f"   ✅ Found {len(json_matches)} potential token patterns")
                                    return json_matches
                                    
                            except Exception as e:
                                continue
                    
                    print("   ❌ No token data found in website")
                else:
                    print(f"   ❌ Website returned {response.status}")
                    
        except Exception as e:
            print(f"   ❌ Website scraping error: {e}")
        
        return None
    
    async def try_websocket_approach(self):
        """Try WebSocket connection to pump.fun"""
        
        print("🔌 Trying WebSocket connections...")
        
        ws_urls = [
            "wss://pump.fun/ws",
            "wss://api.pump.fun/ws", 
            "wss://frontend-api.pump.fun/ws"
        ]
        
        for ws_url in ws_urls:
            try:
                print(f"   Trying: {ws_url}")
                
                import websockets
                async with websockets.connect(ws_url, timeout=5) as ws:
                    print(f"   ✅ Connected to {ws_url}")
                    
                    # Try to send subscription message
                    await ws.send(json.dumps({"type": "subscribe", "channel": "coins"}))
                    
                    # Wait for response
                    response = await asyncio.wait_for(ws.recv(), timeout=5)
                    print(f"   📨 Response: {response[:100]}...")
                    
                    return ws_url
                    
            except Exception as e:
                print(f"   ❌ {ws_url}: {str(e)[:50]}")
        
        return None
    
    async def get_fresh_tokens(self):
        """Try all methods to get fresh tokens"""
        
        print(f"\n🔍 SCANNING FOR FRESH TOKENS - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        # Method 1: Try API endpoints
        tokens = await self.try_api_endpoints()
        if tokens:
            return tokens
        
        # Method 2: Try website scraping
        website_data = await self.scrape_website_directly()
        if website_data:
            return website_data
        
        # Method 3: Try WebSocket
        ws_url = await self.try_websocket_approach()
        if ws_url:
            print(f"✅ WebSocket connection available: {ws_url}")
        
        return None
    
    async def process_tokens(self, tokens):
        """Process and display new tokens"""
        
        new_count = 0
        
        for token in tokens:
            try:
                # Handle different data formats
                if isinstance(token, str):
                    # Try to parse as JSON
                    try:
                        token = json.loads(token)
                    except:
                        continue
                
                mint = token.get('mint') or token.get('address') or token.get('id')
                name = token.get('name', 'Unknown')
                symbol = token.get('symbol', 'UNKNOWN')
                created_timestamp = token.get('created_timestamp') or token.get('createdAt') or token.get('timestamp')
                
                if mint and mint not in self.seen_tokens:
                    self.seen_tokens.add(mint)
                    new_count += 1
                    
                    # Calculate age
                    age_str = "Unknown age"
                    if created_timestamp:
                        try:
                            if created_timestamp > 1000000000000:  # Milliseconds
                                created_time = datetime.fromtimestamp(created_timestamp / 1000)
                            else:  # Seconds
                                created_time = datetime.fromtimestamp(created_timestamp)
                            
                            age_seconds = (datetime.now() - created_time).total_seconds()
                            if age_seconds < 60:
                                age_str = f"{age_seconds:.0f} seconds ago"
                            elif age_seconds < 3600:
                                age_str = f"{age_seconds/60:.1f} minutes ago"
                            else:
                                age_str = f"{age_seconds/3600:.1f} hours ago"
                        except:
                            pass
                    
                    self.launch_count += 1
                    
                    print(f"\n🚀 NEW TOKEN #{self.launch_count} - {datetime.now().strftime('%H:%M:%S')}")
                    print(f"📍 ADDRESS: {mint}")
                    print(f"🏷️  NAME: {name}")
                    print(f"🎯 SYMBOL: {symbol}")
                    print(f"⏰ AGE: {age_str}")
                    print(f"🔗 PUMP: https://pump.fun/{mint}")
                    print(f"🔗 SCAN: https://solscan.io/token/{mint}")
                    
            except Exception as e:
                print(f"⚠️ Error processing token: {e}")
        
        if new_count > 0:
            print(f"\n✅ Found {new_count} new tokens!")
        else:
            print("📡 No new tokens found")
        
        return new_count
    
    async def monitor_continuously(self):
        """Continuously monitor for new tokens"""
        
        print("🚀 DIRECT PUMP.FUN MONITOR")
        print("=" * 50)
        print("🎯 Direct website + API monitoring")
        print("🔄 Multiple fallback methods")
        print("⚡ Real-time token detection")
        print("📊 Scanning every 5 seconds")
        print("")
        
        scan_count = 0
        consecutive_failures = 0
        
        try:
            while True:
                scan_count += 1
                
                tokens = await self.get_fresh_tokens()
                
                if tokens:
                    consecutive_failures = 0
                    new_tokens = await self.process_tokens(tokens)
                    
                    if new_tokens == 0:
                        print(f"📡 SCAN #{scan_count} - No new tokens")
                else:
                    consecutive_failures += 1
                    print(f"❌ SCAN #{scan_count} - All methods failed ({consecutive_failures} consecutive)")
                    
                    if consecutive_failures >= 5:
                        print("🚨 Too many failures - checking connection...")
                        await asyncio.sleep(10)
                        consecutive_failures = 0
                
                print(f"💤 Waiting 5 seconds...")
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitor stopped")
            runtime = (datetime.now() - self.start_time).total_seconds()
            rate = self.launch_count / (runtime / 60) if runtime > 0 else 0
            print(f"📊 Final: {self.launch_count} tokens in {runtime:.0f}s ({rate:.1f}/min)")

async def main():
    """Main function"""
    
    async with PumpFunScraper() as scraper:
        await scraper.monitor_continuously()

if __name__ == "__main__":
    print("🚀 Starting Direct Pump.Fun Monitor!")
    asyncio.run(main()) 