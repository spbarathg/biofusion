#!/usr/bin/env python3
"""
Test Script for Working Pump.fun Solutions
This script tests all the confirmed working solutions found from internet research.
"""

import asyncio
import websockets
import aiohttp
import json
import time
from datetime import datetime

async def test_pumpportal():
    """Test PumpPortal WebSocket API"""
    print("🧪 Testing PumpPortal WebSocket API...")
    try:
        async with websockets.connect("wss://pumpportal.fun/api/data", timeout=10) as websocket:
            print("✅ PumpPortal: Connection successful!")
            
            # Send subscription
            await websocket.send(json.dumps({"method": "subscribeNewToken"}))
            print("✅ PumpPortal: Subscription sent successfully!")
            
            # Wait for a few seconds to see if we get data
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"✅ PumpPortal: Received data: {message[:100]}...")
                return True
            except asyncio.TimeoutError:
                print("⚠️ PumpPortal: No data received in 5 seconds (connection works but no new tokens)")
                return True
                
    except Exception as e:
        print(f"❌ PumpPortal: Failed - {e}")
        return False

async def test_bitquery():
    """Test Bitquery GraphQL API"""
    print("\n🧪 Testing Bitquery GraphQL API...")
    
    # Simple test query
    query = """
    query {
      Solana {
        TokenSupplyUpdates(
          where: {
            Instruction: {
              Program: {
                Address: { is: "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P" }
                Method: { is: "create" }
              }
            }
          }
          limit: { count: 1 }
          orderBy: { descending: Block_Time }
        ) {
          TokenSupplyUpdate {
            Currency {
              MintAddress
              Name
              Symbol
            }
          }
        }
      }
    }
    """
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://graphql.bitquery.io",
                headers={
                    "Content-Type": "application/json",
                    "X-API-KEY": "BQYKNtaLjGVGlYLKCzNd6tZXITa6jKyz"  # Demo key
                },
                json={"query": query},
                timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data:
                        print("✅ Bitquery: API connection successful!")
                        print(f"✅ Bitquery: Sample data received: {json.dumps(data, indent=2)[:200]}...")
                        return True
                    else:
                        print(f"⚠️ Bitquery: API responded but with errors: {data}")
                        return False
                else:
                    error_text = await response.text()
                    print(f"❌ Bitquery: HTTP {response.status} - {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Bitquery: Failed - {e}")
        return False

async def test_solana_rpc():
    """Test Solana RPC endpoint"""
    print("\n🧪 Testing Solana RPC endpoint...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.mainnet-beta.solana.com",
                headers={"Content-Type": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getVersion"
                },
                timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'result' in data:
                        print(f"✅ Solana RPC: Connection successful! Version: {data['result']}")
                        return True
                    else:
                        print(f"⚠️ Solana RPC: Unexpected response: {data}")
                        return False
                else:
                    error_text = await response.text()
                    print(f"❌ Solana RPC: HTTP {response.status} - {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Solana RPC: Failed - {e}")
        return False

async def test_helius():
    """Test Helius RPC (alternative Solana RPC)"""
    print("\n🧪 Testing Helius RPC endpoint...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://mainnet.helius-rpc.com/?api-key=193ececa-6e42-4d84-b9bd-765c4813816d",
                headers={"Content-Type": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getVersion"
                },
                timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'result' in data:
                        print(f"✅ Helius RPC: Connection successful! Version: {data['result']}")
                        return True
                    else:
                        print(f"⚠️ Helius RPC: Unexpected response: {data}")
                        return False
                else:
                    error_text = await response.text()
                    print(f"❌ Helius RPC: HTTP {response.status} - {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ Helius RPC: Failed - {e}")
        return False

async def test_pump_fun_frontend():
    """Test pump.fun frontend API (known to be down)"""
    print("\n🧪 Testing pump.fun frontend API (expected to fail)...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://frontend-api.pump.fun/coins?offset=0&limit=10",
                timeout=10
            ) as response:
                if response.status == 200:
                    print("✅ Pump.fun Frontend: Surprisingly working!")
                    return True
                else:
                    print(f"❌ Pump.fun Frontend: Expected failure - HTTP {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ Pump.fun Frontend: Expected failure - {e}")
        return False

async def test_dexscreener():
    """Test DexScreener API"""
    print("\n🧪 Testing DexScreener API...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.dexscreener.com/latest/dex/pairs/solana",
                timeout=10
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ DexScreener: Connection successful! Found {len(data.get('pairs', []))} pairs")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ DexScreener: HTTP {response.status} - {error_text}")
                    return False
                    
    except Exception as e:
        print(f"❌ DexScreener: Failed - {e}")
        return False

async def main():
    """Run all tests"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                          🧪 TESTING WORKING SOLUTIONS 🧪                            ║
║                     Based on comprehensive internet research                         ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
""")
    
    start_time = time.time()
    results = {}
    
    # Test all solutions
    tests = [
        ("PumpPortal WebSocket", test_pumpportal),
        ("Bitquery GraphQL", test_bitquery),
        ("Solana RPC", test_solana_rpc),
        ("Helius RPC", test_helius),
        ("Pump.fun Frontend", test_pump_fun_frontend),
        ("DexScreener API", test_dexscreener),
    ]
    
    for name, test_func in tests:
        try:
            results[name] = await test_func()
        except Exception as e:
            print(f"❌ {name}: Test crashed - {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*90)
    print("📊 SUMMARY OF WORKING SOLUTIONS:")
    print("="*90)
    
    working_count = 0
    for name, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {name:<25} {'WORKING' if status else 'NOT WORKING'}")
        if status:
            working_count += 1
    
    print("="*90)
    print(f"🎯 RESULT: {working_count}/{len(tests)} solutions are working!")
    
    if working_count > 0:
        print(f"\n🚀 RECOMMENDATION: Use the working solutions above for monitoring!")
        print(f"💡 Start with: python working_pumpportal_monitor.py")
        if results.get("PumpPortal WebSocket"):
            print(f"💡 PumpPortal is working - this is your best bet!")
        if results.get("Bitquery GraphQL"):
            print(f"💡 Bitquery is working - use this as backup!")
    else:
        print(f"\n😞 Unfortunately, none of the tested solutions are working right now.")
        print(f"💡 This might be temporary - try again later.")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total test time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    # Install required packages
    try:
        import websockets
        import aiohttp
    except ImportError:
        print("Installing required packages...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "aiohttp"])
        import websockets
        import aiohttp
    
    asyncio.run(main()) 