#!/usr/bin/env python3
"""
DEBUG PUMP.FUN API
==================

Quick test to see what tokens pump.fun API is returning
"""

import asyncio
import aiohttp
import json
from datetime import datetime

async def test_pump_api():
    """Test pump.fun API directly"""
    
    print("🔍 Testing pump.fun API...")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        url = "https://frontend-api.pump.fun/coins?offset=0&limit=20&sort=created_timestamp&order=DESC&includeNsfw=true"
        
        try:
            async with session.get(url) as response:
                print(f"📡 Status: {response.status}")
                
                if response.status == 200:
                    coins = await response.json()
                    print(f"📊 Found {len(coins)} coins")
                    print("")
                    
                    now = datetime.now()
                    
                    for i, coin in enumerate(coins[:10], 1):
                        mint = coin.get("mint", "Unknown")
                        name = coin.get("name", "Unknown")
                        symbol = coin.get("symbol", "UNKNOWN")
                        created_timestamp = coin.get("created_timestamp", 0)
                        market_cap = coin.get("usd_market_cap", 0)
                        
                        if created_timestamp:
                            created_time = datetime.fromtimestamp(created_timestamp / 1000)
                            age_minutes = (now - created_time).total_seconds() / 60
                            age_display = f"{age_minutes:.1f} min ago" if age_minutes < 60 else f"{age_minutes/60:.1f} hrs ago"
                        else:
                            age_display = "Unknown age"
                        
                        print(f"🚀 #{i}")
                        print(f"   📍 Address: {mint[:20]}...")
                        print(f"   🏷️  Name: {name}")
                        print(f"   🎯 Symbol: {symbol}")
                        print(f"   ⏰ Age: {age_display}")
                        print(f"   💰 MC: ${market_cap:.2f}")
                        print("")
                        
                else:
                    print(f"❌ HTTP Error: {response.status}")
                    text = await response.text()
                    print(f"Response: {text[:200]}")
                    
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_pump_api()) 