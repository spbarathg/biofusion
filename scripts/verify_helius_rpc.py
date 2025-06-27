#!/usr/bin/env python3
"""
Helius RPC Verification Script
Comprehensive testing for Helius RPC endpoint functionality and performance
"""

import asyncio
import aiohttp
import json
import time
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Any
import os
from dataclasses import dataclass

@dataclass
class RPCTestResult:
    """Test result data structure"""
    test_name: str
    success: bool
    latency_ms: float
    response_data: Optional[Dict] = None
    error: Optional[str] = None

class HeliusRPCVerifier:
    """Comprehensive Helius RPC endpoint verification"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "HeliusRPCVerifier/1.0"
        }
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_rpc_call(self, method: str, params: List = None) -> RPCTestResult:
        """Make a single RPC call and measure performance"""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or []
        }
        
        start_time = time.time()
        try:
            async with self.session.post(self.base_url, json=payload) as response:
                latency_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    if "error" in data:
                        return RPCTestResult(
                            test_name=method,
                            success=False,
                            latency_ms=latency_ms,
                            error=f"RPC Error: {data['error']}"
                        )
                    
                    return RPCTestResult(
                        test_name=method,
                        success=True,
                        latency_ms=latency_ms,
                        response_data=data.get("result")
                    )
                else:
                    error_text = await response.text()
                    return RPCTestResult(
                        test_name=method,
                        success=False,
                        latency_ms=latency_ms,
                        error=f"HTTP {response.status}: {error_text}"
                    )
                    
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return RPCTestResult(
                test_name=method,
                success=False,
                latency_ms=latency_ms,
                error=str(e)
            )
    
    async def test_basic_connectivity(self) -> RPCTestResult:
        """Test basic RPC connectivity"""
        print("🔗 Testing basic connectivity...")
        return await self._make_rpc_call("getHealth")
    
    async def test_node_version(self) -> RPCTestResult:
        """Test node version endpoint"""
        print("📋 Testing node version...")
        return await self._make_rpc_call("getVersion")
    
    async def test_slot_info(self) -> RPCTestResult:
        """Test current slot information"""
        print("🎰 Testing slot information...")
        return await self._make_rpc_call("getSlot")
    
    async def test_block_height(self) -> RPCTestResult:
        """Test block height retrieval"""
        print("📏 Testing block height...")
        return await self._make_rpc_call("getBlockHeight")
    
    async def test_recent_performance(self) -> RPCTestResult:
        """Test recent performance samples"""
        print("⚡ Testing performance samples...")
        return await self._make_rpc_call("getRecentPerformanceSamples", [5])
    
    async def test_account_info(self) -> RPCTestResult:
        """Test account info retrieval (using SOL token mint)"""
        print("👤 Testing account info...")
        sol_mint = "So11111111111111111111111111111111111112"
        return await self._make_rpc_call("getAccountInfo", [sol_mint])
    
    async def test_token_accounts_by_owner(self) -> RPCTestResult:
        """Test token accounts retrieval"""
        print("🪙 Testing token accounts...")
        # Use a well-known wallet for testing
        test_wallet = "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"  # Raydium wallet
        return await self._make_rpc_call("getTokenAccountsByOwner", [
            test_wallet,
            {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
            {"encoding": "jsonParsed"}
        ])
    
    async def test_transaction_signature(self) -> RPCTestResult:
        """Test transaction signature verification"""
        print("📝 Testing transaction lookup...")
        # Use a known transaction hash for testing
        recent_result = await self._make_rpc_call("getRecentBlockhash")
        if recent_result.success and recent_result.response_data:
            return await self._make_rpc_call("getSignaturesForAddress", [
                "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM",
                {"limit": 1}
            ])
        return RPCTestResult("getSignaturesForAddress", False, 0, error="Could not get recent blockhash")
    
    async def test_multiple_requests(self, count: int = 10) -> List[float]:
        """Test multiple concurrent requests"""
        print(f"🚀 Testing {count} concurrent requests...")
        
        tasks = []
        for _ in range(count):
            tasks.append(self._make_rpc_call("getSlot"))
        
        results = await asyncio.gather(*tasks)
        latencies = [r.latency_ms for r in results if r.success]
        
        return latencies
    
    async def test_rate_limits(self) -> Dict[str, Any]:
        """Test rate limiting behavior"""
        print("⏱️ Testing rate limits...")
        
        # Send 30 requests rapidly to test rate limiting
        start_time = time.time()
        tasks = []
        for _ in range(30):
            tasks.append(self._make_rpc_call("getHealth"))
        
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        successful_requests = sum(1 for r in results if r.success)
        failed_requests = len(results) - successful_requests
        requests_per_second = len(results) / total_time
        
        return {
            "total_requests": len(results),
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "total_time_seconds": total_time,
            "requests_per_second": requests_per_second,
            "avg_latency_ms": statistics.mean([r.latency_ms for r in results if r.success]) if successful_requests > 0 else 0
        }
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all verification tests"""
        print("🔍 Starting Helius RPC Comprehensive Verification")
        print("=" * 60)
        
        results = {}
        
        # Basic functionality tests
        basic_tests = [
            self.test_basic_connectivity(),
            self.test_node_version(),
            self.test_slot_info(),
            self.test_block_height(),
            self.test_recent_performance(),
            self.test_account_info(),
            self.test_token_accounts_by_owner(),
            self.test_transaction_signature()
        ]
        
        print("\n📋 Running Basic Functionality Tests...")
        basic_results = await asyncio.gather(*basic_tests, return_exceptions=True)
        
        for result in basic_results:
            if isinstance(result, RPCTestResult):
                results[result.test_name] = {
                    "success": result.success,
                    "latency_ms": result.latency_ms,
                    "error": result.error
                }
                
                status = "✅" if result.success else "❌"
                print(f"  {status} {result.test_name}: {result.latency_ms:.1f}ms")
                if result.error:
                    print(f"      Error: {result.error}")
        
        # Performance tests
        print("\n⚡ Running Performance Tests...")
        latencies = await self.test_multiple_requests(10)
        
        if latencies:
            results["performance"] = {
                "avg_latency_ms": statistics.mean(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "median_latency_ms": statistics.median(latencies),
                "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0
            }
            
            print(f"  📊 Average latency: {results['performance']['avg_latency_ms']:.1f}ms")
            print(f"  📊 Min latency: {results['performance']['min_latency_ms']:.1f}ms")
            print(f"  📊 Max latency: {results['performance']['max_latency_ms']:.1f}ms")
            print(f"  📊 Median latency: {results['performance']['median_latency_ms']:.1f}ms")
        
        # Rate limit tests
        print("\n⏱️ Running Rate Limit Tests...")
        rate_limit_results = await self.test_rate_limits()
        results["rate_limits"] = rate_limit_results
        
        print(f"  📈 Requests per second: {rate_limit_results['requests_per_second']:.1f}")
        print(f"  ✅ Successful requests: {rate_limit_results['successful_requests']}")
        print(f"  ❌ Failed requests: {rate_limit_results['failed_requests']}")
        
        return results

def print_summary(results: Dict[str, Any]):
    """Print test summary and recommendations"""
    print("\n" + "=" * 60)
    print("📊 HELIUS RPC VERIFICATION SUMMARY")
    print("=" * 60)
    
    # Count successful tests
    basic_tests = [k for k in results.keys() if k not in ["performance", "rate_limits"]]
    successful_tests = sum(1 for test in basic_tests if results[test]["success"])
    
    print(f"\n✅ Basic Functionality: {successful_tests}/{len(basic_tests)} tests passed")
    
    if "performance" in results:
        perf = results["performance"]
        avg_latency = perf["avg_latency_ms"]
        
        if avg_latency < 50:
            latency_status = "🚀 Excellent"
        elif avg_latency < 100:
            latency_status = "✅ Good"
        elif avg_latency < 200:
            latency_status = "⚠️ Acceptable"
        else:
            latency_status = "❌ Slow"
            
        print(f"⚡ Performance: {latency_status} ({avg_latency:.1f}ms avg)")
    
    if "rate_limits" in results:
        rate_data = results["rate_limits"]
        success_rate = (rate_data["successful_requests"] / rate_data["total_requests"]) * 100
        
        if success_rate >= 95:
            rate_status = "✅ Excellent"
        elif success_rate >= 90:
            rate_status = "⚠️ Good"
        else:
            rate_status = "❌ Issues detected"
            
        print(f"⏱️ Rate Limiting: {rate_status} ({success_rate:.1f}% success rate)")
    
    print("\n💡 RECOMMENDATIONS:")
    
    if successful_tests == len(basic_tests):
        print("  ✅ Your Helius RPC endpoint is working perfectly!")
        print("  ✅ All basic functionality tests passed")
    else:
        print("  ❌ Some basic functionality tests failed")
        print("  💡 Check your API key and network connectivity")
    
    if "performance" in results:
        avg_latency = results["performance"]["avg_latency_ms"]
        if avg_latency > 100:
            print(f"  ⚠️ High latency detected ({avg_latency:.1f}ms)")
            print("  💡 Consider using a geographically closer RPC endpoint")
        else:
            print("  ✅ Latency is excellent for trading applications")
    
    if "rate_limits" in results:
        rps = results["rate_limits"]["requests_per_second"]
        if rps > 20:
            print(f"  ✅ High throughput capability ({rps:.1f} req/sec)")
        else:
            print(f"  ⚠️ Rate limiting detected ({rps:.1f} req/sec)")
            print("  💡 Consider upgrading your Helius plan for higher limits")

async def main():
    """Main verification function"""
    print("🟪 Helius RPC Verification Tool")
    print("=" * 60)
    
    # Try to get API key from environment or prompt
    api_key = os.getenv("HELIUS_API_KEY")
    
    if not api_key:
        print("🔑 Helius API Key not found in environment variables.")
        print("You can either:")
        print("1. Set HELIUS_API_KEY environment variable")
        print("2. Enter it manually below")
        print("3. Update your .env file with PRIVATE_RPC_URL=https://mainnet.helius-rpc.com/?api-key=YOUR_KEY")
        print("")
        
        api_key = input("Enter your Helius API key: ").strip()
        
        if not api_key:
            print("❌ No API key provided. Exiting.")
            return
    
    print(f"🔑 Using API key: {api_key[:8]}...")
    print("")
    
    try:
        async with HeliusRPCVerifier(api_key) as verifier:
            results = await verifier.run_comprehensive_test()
            print_summary(results)
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"logs/helius_verification_{timestamp}.json"
            
            os.makedirs("logs", exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {results_file}")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        print("💡 Check your API key and internet connectivity")

if __name__ == "__main__":
    asyncio.run(main()) 