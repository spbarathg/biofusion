#!/usr/bin/env python3
"""
TEST SOLANA TOKENS
=================

One-time test to see current Solana tokens on Raydium.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def test_solana_tokens():
    """Test Solana token detection"""
    
    print("🪙 TESTING SOLANA TOKEN DETECTION")
    print("=" * 50)
    
    # Create session
    connector = aiohttp.TCPConnector(
        resolver=aiohttp.AsyncResolver(),
        use_dns_cache=False
    )
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30),
        connector=connector
    ) as session:
        
        try:
            raydium_url = "https://api.raydium.io/v2/main/pairs"
            
            async with session.get(raydium_url) as response:
                if response.status == 200:
                    pairs = await response.json()
                    print(f"✅ Retrieved {len(pairs)} total pairs from Raydium")
                    
                    # SOL mint address
                    SOL_MINT = "So11111111111111111111111111111111111111112"
                    
                    sol_pairs = []
                    
                    for pair in pairs:
                        quote_mint = pair.get("quoteMint", "")
                        base_mint = pair.get("baseMint", "")
                        
                        # Only SOL pairs
                        if quote_mint == SOL_MINT:
                            base_symbol = pair.get("baseSymbol", "UNKNOWN")
                            liquidity = float(pair.get("liquidity", 0))
                            
                            if base_symbol != "UNKNOWN" and liquidity > 10:
                                sol_pairs.append({
                                    "symbol": base_symbol,
                                    "name": pair.get("baseName", "Unknown"),
                                    "address": base_mint,
                                    "liquidity": liquidity,
                                    "price": float(pair.get("price", 0))
                                })
                    
                    # Sort by liquidity
                    sol_pairs.sort(key=lambda x: x['liquidity'], reverse=True)
                    
                    print(f"🎯 Found {len(sol_pairs)} SOL trading pairs")
                    print("\n🔥 TOP 15 SOLANA TOKENS BY LIQUIDITY:")
                    print("-" * 80)
                    
                    for i, token in enumerate(sol_pairs[:15], 1):
                        print(f"{i:2d}. {token['symbol']:8s} | ${token['liquidity']:12,.2f} | ${token['price']:.8f} SOL")
                        print(f"    {token['name']}")
                        print(f"    {token['address'][:8]}...{token['address'][-8:]}")
                        print()
                    
                    print(f"💰 Total SOL liquidity in top 15: ${sum(t['liquidity'] for t in sol_pairs[:15]):,.2f}")
                    
                else:
                    print(f"❌ API Error: {response.status}")
        
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_solana_tokens()) 