#!/usr/bin/env python3
"""
EXAMINE RAYDIUM STRUCTURE
========================

Examine the actual API response to understand the data structure.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def examine_raydium_structure():
    """Examine Raydium API structure"""
    
    print("🔍 EXAMINING RAYDIUM API STRUCTURE")
    print("=" * 50)
    
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
                    print(f"✅ Retrieved {len(pairs)} total pairs")
                    
                    # Examine first few pairs
                    print("\n📊 FIRST 3 PAIR STRUCTURES:")
                    print("-" * 80)
                    
                    for i, pair in enumerate(pairs[:3]):
                        print(f"\n🔍 PAIR {i+1}:")
                        
                        # Print all keys and values
                        for key, value in pair.items():
                            print(f"  {key}: {value}")
                    
                    print("\n" + "=" * 80)
                    
                    # Look for SOL-related patterns
                    sol_mint = "So11111111111111111111111111111111111111112"
                    wrapped_sol = "So11111111111111111111111111111111111111112"
                    
                    print(f"\n🔎 SEARCHING FOR SOL PATTERNS:")
                    print(f"SOL mint: {sol_mint}")
                    
                    sol_count = 0
                    for pair in pairs[:100]:  # Check first 100
                        quote_mint = pair.get("quoteMint", "")
                        base_mint = pair.get("baseMint", "")
                        
                        if sol_mint in [quote_mint, base_mint]:
                            sol_count += 1
                            if sol_count <= 3:  # Show first 3 SOL pairs
                                print(f"\n🎯 SOL PAIR #{sol_count}:")
                                print(f"  Quote: {quote_mint}")
                                print(f"  Base: {base_mint}")
                                print(f"  Symbol: {pair.get('baseSymbol', 'N/A')}")
                                print(f"  Liquidity: {pair.get('liquidity', 'N/A')}")
                    
                    print(f"\n📊 Found {sol_count} SOL pairs in first 100 pairs")
                    
                    # Let's also check for different SOL representations
                    print(f"\n🔍 CHECKING DIFFERENT SOL PATTERNS:")
                    patterns = ["SOL", "WSOL", "sol", "wsol"]
                    
                    for pattern in patterns:
                        count = 0
                        for pair in pairs[:100]:
                            if (pattern in str(pair.get("baseSymbol", "")).upper() or 
                                pattern in str(pair.get("quoteSymbol", "")).upper()):
                                count += 1
                        print(f"  {pattern}: {count} matches")
                
                else:
                    print(f"❌ API Error: {response.status}")
        
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(examine_raydium_structure()) 