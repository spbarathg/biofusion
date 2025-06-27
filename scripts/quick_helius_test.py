#!/usr/bin/env python3
"""
Quick Helius RPC Test
Simple verification script to quickly test if your Helius RPC is working
"""

import requests
import time
import json
import os

def test_helius_rpc(api_key: str):
    """Quick test of Helius RPC functionality"""
    
    # Your Helius endpoint
    url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
    
    # Headers
    headers = {
        "Content-Type": "application/json"
    }
    
    print("🟪 Quick Helius RPC Test")
    print("=" * 40)
    print(f"🔗 Endpoint: {url[:50]}...")
    print("")
    
    # Test 1: Basic Health Check
    print("1️⃣ Testing basic connectivity...")
    try:
        start_time = time.time()
        response = requests.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getHealth"
            },
            headers=headers,
            timeout=10
        )
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            if "result" in data and data["result"] == "ok":
                print(f"   ✅ PASSED - Health check OK ({latency:.1f}ms)")
            else:
                print(f"   ❌ FAILED - Unexpected response: {data}")
                return False
        else:
            print(f"   ❌ FAILED - HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ FAILED - Error: {e}")
        return False
    
    # Test 2: Get Version
    print("2️⃣ Testing version info...")
    try:
        start_time = time.time()
        response = requests.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getVersion"
            },
            headers=headers,
            timeout=10
        )
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                version_info = data["result"]
                solana_core = version_info.get("solana-core", "unknown")
                print(f"   ✅ PASSED - Solana version: {solana_core} ({latency:.1f}ms)")
            else:
                print(f"   ❌ FAILED - No version info: {data}")
                return False
        else:
            print(f"   ❌ FAILED - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ FAILED - Error: {e}")
        return False
    
    # Test 3: Get Current Slot
    print("3️⃣ Testing slot retrieval...")
    try:
        start_time = time.time()
        response = requests.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSlot"
            },
            headers=headers,
            timeout=10
        )
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                slot = data["result"]
                print(f"   ✅ PASSED - Current slot: {slot:,} ({latency:.1f}ms)")
            else:
                print(f"   ❌ FAILED - No slot info: {data}")
                return False
        else:
            print(f"   ❌ FAILED - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ FAILED - Error: {e}")
        return False
    
    # Test 4: Account Info (SOL token mint)
    print("4️⃣ Testing account info...")
    try:
        start_time = time.time()
        sol_mint = "So11111111111111111111111111111111111112"
        response = requests.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [sol_mint, {"encoding": "base64"}]
            },
            headers=headers,
            timeout=10
        )
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            if "result" in data and data["result"] is not None:
                print(f"   ✅ PASSED - Account info retrieved ({latency:.1f}ms)")
            else:
                print(f"   ❌ FAILED - No account info: {data}")
                return False
        else:
            print(f"   ❌ FAILED - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ FAILED - Error: {e}")
        return False
    
    # Test 5: Latency Test (5 quick requests)
    print("5️⃣ Testing latency (5 requests)...")
    latencies = []
    
    for i in range(5):
        try:
            start_time = time.time()
            response = requests.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getHealth"
                },
                headers=headers,
                timeout=5
            )
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                latencies.append(latency)
            else:
                print(f"   ⚠️ Request {i+1} failed")
        except Exception:
            print(f"   ⚠️ Request {i+1} timed out")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"   📊 Average: {avg_latency:.1f}ms")
        print(f"   📊 Min: {min_latency:.1f}ms")
        print(f"   📊 Max: {max_latency:.1f}ms")
        
        if avg_latency < 50:
            print("   🚀 Excellent latency for trading!")
        elif avg_latency < 100:
            print("   ✅ Good latency for most applications")
        elif avg_latency < 200:
            print("   ⚠️ Acceptable latency")
        else:
            print("   ❌ High latency - may affect trading performance")
    else:
        print("   ❌ All latency tests failed")
        return False
    
    print("")
    print("🎉 ALL TESTS PASSED!")
    print("✅ Your Helius RPC endpoint is working correctly")
    print("")
    print("💡 Next steps:")
    print("   • Update your .env file with PRIVATE_RPC_URL")
    print("   • Set USE_PRIVATE_RPC=true in your config")
    print("   • Your bot is ready to use the Helius endpoint!")
    
    return True

def main():
    """Main function"""
    
    # Check for API key
    api_key = os.getenv("HELIUS_API_KEY")
    
    if not api_key:
        print("🟪 Helius RPC Quick Test")
        print("=" * 40)
        print("Enter your Helius API key below.")
        print("You can find it in your Helius dashboard.")
        print("")
        api_key = input("Helius API Key: ").strip()
        
        if not api_key:
            print("❌ No API key provided. Exiting.")
            return
    
    # Run the test
    success = test_helius_rpc(api_key)
    
    if not success:
        print("")
        print("❌ Some tests failed!")
        print("💡 Troubleshooting:")
        print("   • Check your API key is correct")
        print("   • Verify your internet connection")
        print("   • Check if your Helius plan is active")
        print("   • Try again in a few minutes")

if __name__ == "__main__":
    main() 