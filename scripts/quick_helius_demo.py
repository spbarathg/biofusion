#!/usr/bin/env python3
"""
Quick Demo: Helius Connection Test
This shows that Helius RPC is working for pump.fun monitoring
"""

import asyncio
import aiohttp
import json

async def test_helius_connection():
    """Test Helius RPC connection"""
    print("🧪 Testing Helius RPC Connection...")
    
    url = "https://mainnet.helius-rpc.com/?api-key=193ececa-6e42-4d84-b9bd-765c4813816d"
    
    # Test basic connection
    request_data = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getVersion"
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=request_data, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Helius RPC Connected! Version: {data['result']}")
                    return True
                else:
                    print(f"❌ Connection failed: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

async def test_pump_fun_program_logs():
    """Test getting recent pump.fun program logs"""
    print("\n🔍 Testing pump.fun program log access...")
    
    url = "https://mainnet.helius-rpc.com/?api-key=193ececa-6e42-4d84-b9bd-765c4813816d"
    pump_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
    
    # Get recent signatures for pump.fun program
    request_data = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": [
            pump_program_id,
            {"limit": 5}
        ]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=request_data, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'result' in data:
                        signatures = data['result']
                        print(f"✅ Found {len(signatures)} recent pump.fun transactions!")
                        
                        for i, sig in enumerate(signatures[:3]):
                            print(f"  {i+1}. {sig['signature'][:16]}... (Block: {sig.get('slot', 'Unknown')})")
                        
                        if signatures:
                            # Test getting transaction details for the first one
                            await test_transaction_details(signatures[0]['signature'])
                        
                        return True
                    else:
                        print(f"❌ No signatures found: {data}")
                        return False
                else:
                    print(f"❌ Failed to get signatures: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Error getting signatures: {e}")
        return False

async def test_transaction_details(signature):
    """Test getting transaction details"""
    print(f"\n🔬 Testing transaction details for {signature[:16]}...")
    
    url = "https://mainnet.helius-rpc.com/?api-key=193ececa-6e42-4d84-b9bd-765c4813816d"
    
    request_data = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            signature,
            {
                "encoding": "json",
                "maxSupportedTransactionVersion": 0
            }
        ]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=request_data, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'result' in data and data['result']:
                        result = data['result']
                        
                        # Check for logs
                        if 'meta' in result and 'logMessages' in result['meta']:
                            logs = result['meta']['logMessages']
                            print(f"✅ Transaction has {len(logs)} log messages")
                            
                            # Show first few logs
                            for i, log in enumerate(logs[:3]):
                                print(f"  Log {i+1}: {log[:60]}...")
                        
                        # Check for token balances (indicating token operations)
                        if 'meta' in result and 'postTokenBalances' in result['meta']:
                            balances = result['meta']['postTokenBalances']
                            if balances:
                                print(f"✅ Found {len(balances)} token balance changes")
                                for balance in balances[:2]:
                                    mint = balance.get('mint', 'Unknown')
                                    print(f"  Token: {mint[:16]}...")
                        
                        return True
                    else:
                        print(f"❌ No transaction data: {data}")
                        return False
                else:
                    print(f"❌ Failed to get transaction: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Error getting transaction: {e}")
        return False

async def main():
    """Run all tests"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                        🚀 HELIUS PUMP.FUN DEMO 🚀                                   ║
║                    Demonstrating WORKING connection                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Run tests
    test1 = await test_helius_connection()
    test2 = await test_pump_fun_program_logs() if test1 else False
    
    print("\n" + "="*90)
    print("📊 DEMO RESULTS:")
    print("="*90)
    print(f"✅ Helius RPC Connection: {'WORKING' if test1 else 'FAILED'}")
    print(f"✅ Pump.fun Program Access: {'WORKING' if test2 else 'FAILED'}")
    print("="*90)
    
    if test1 and test2:
        print("\n🎉 SUCCESS! Helius can access pump.fun program data!")
        print("💡 The full monitor script (helius_working_monitor.py) will use WebSocket")
        print("   to get real-time notifications when new tokens are created.")
        print("\n🚀 NEXT STEP: Run the full monitor:")
        print("   python helius_working_monitor.py")
    else:
        print("\n😞 Some tests failed. Check your internet connection.")

if __name__ == "__main__":
    # Install aiohttp if needed
    try:
        import aiohttp
    except ImportError:
        print("Installing aiohttp...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
        import aiohttp
    
    asyncio.run(main()) 