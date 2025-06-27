#!/usr/bin/env python3
"""
Quick test to check pump.fun API and recent token activity
"""

import requests
import json
from datetime import datetime
import time

def test_pump_api():
    """Test pump.fun API for recent tokens"""
    print("🔍 Testing pump.fun API...")
    
    try:
        url = "https://frontend-api.pump.fun/coins?offset=0&limit=10&sort=created_timestamp&order=DESC"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Response: {len(data)} tokens")
            
            now = datetime.now()
            
            print("\n🕐 Recent Tokens (Last 24 hours):")
            print("-" * 80)
            
            recent_count = 0
            for i, coin in enumerate(data):
                created_timestamp = coin.get("created_timestamp", 0)
                if created_timestamp:
                    created_time = datetime.fromtimestamp(created_timestamp / 1000)
                    age_hours = (now - created_time).total_seconds() / 3600
                    age_minutes = (now - created_time).total_seconds() / 60
                    
                    if age_hours <= 24:  # Last 24 hours
                        recent_count += 1
                        name = coin.get("name", "Unknown")
                        symbol = coin.get("symbol", "???")
                        mint = coin.get("mint", "")
                        market_cap = coin.get("usd_market_cap", 0)
                        
                        if age_minutes < 60:
                            age_str = f"{age_minutes:.1f}min ago"
                        else:
                            age_str = f"{age_hours:.1f}h ago"
                        
                        print(f"{i+1:2}. {name} ({symbol}) - {age_str}")
                        print(f"    💰 ${market_cap:,.0f} | 🏷️ {mint[:8]}...{mint[-8:]}")
                        
                        if age_minutes <= 10:
                            print(f"    🔥 SUPER FRESH!")
                        elif age_minutes <= 60:
                            print(f"    ⚡ FRESH!")
                        
                        print()
            
            print(f"📊 Summary: {recent_count} tokens in last 24 hours")
            
            # Check activity rate
            if recent_count > 0:
                oldest_recent = None
                for coin in data:
                    created_timestamp = coin.get("created_timestamp", 0)
                    if created_timestamp:
                        created_time = datetime.fromtimestamp(created_timestamp / 1000)
                        age_hours = (now - created_time).total_seconds() / 3600
                        if age_hours <= 24:
                            oldest_recent = created_time
                
                if oldest_recent:
                    time_span_hours = (now - oldest_recent).total_seconds() / 3600
                    rate_per_hour = recent_count / time_span_hours if time_span_hours > 0 else 0
                    print(f"📈 Launch rate: {rate_per_hour:.1f} tokens/hour")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text[:200])
    
    except Exception as e:
        print(f"❌ Error: {e}")

def test_websocket_connectivity():
    """Test if we can connect to Helius WebSocket"""
    print("\n🔌 Testing WebSocket connectivity...")
    
    try:
        import asyncio
        import websockets
        
        async def test_connection():
            api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
            ws_url = f"wss://mainnet.helius-rpc.com/?api-key={api_key}"
            
            try:
                async with websockets.connect(ws_url, ping_timeout=5) as websocket:
                    print("✅ WebSocket connection successful")
                    
                    # Test subscription
                    subscribe = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "logsSubscribe",
                        "params": [
                            {"mentions": ["6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"]},
                            {"commitment": "processed"}
                        ]
                    }
                    
                    await websocket.send(json.dumps(subscribe))
                    print("✅ Subscription sent")
                    
                    # Wait for response
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    data = json.loads(response)
                    
                    if 'result' in data:
                        print("✅ Subscription confirmed")
                        return True
                    else:
                        print(f"❌ Subscription failed: {data}")
                        return False
                        
            except Exception as e:
                print(f"❌ WebSocket error: {e}")
                return False
        
        # Run the async test
        import asyncio
        if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        result = asyncio.run(test_connection())
        return result
        
    except ImportError:
        print("❌ websockets module not available")
        return False

if __name__ == "__main__":
    print("🚀 PUMP.FUN ACTIVITY TEST")
    print("=" * 50)
    
    # Test API
    test_pump_api()
    
    # Test WebSocket
    test_websocket_connectivity()
    
    print("\n✅ Test completed!") 