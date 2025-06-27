#!/usr/bin/env python3
"""
FAST SOLANA SCANNER
==================

Fast scanner that shows SOL pairs immediately without metadata lookup.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def fast_solana_scan():
    """Fast scan of Solana tokens"""
    
    print("⚡ FAST SOLANA TOKEN SCANNER")
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
            print("🔍 Fetching Raydium pairs...")
            raydium_url = "https://api.raydium.io/v2/main/pairs"
            
            async with session.get(raydium_url) as response:
                if response.status == 200:
                    pairs = await response.json()
                    print(f"✅ Retrieved {len(pairs)} total pairs")
                    
                    # SOL mint address
                    SOL_MINT = "So11111111111111111111111111111111111111112"
                    
                    print(f"\n🔎 Searching for SOL pairs...")
                    sol_pairs = []
                    
                    for pair in pairs:
                        quote_mint = pair.get("quoteMint", "")
                        base_mint = pair.get("baseMint", "")
                        
                        # Only SOL pairs
                        if quote_mint == SOL_MINT:
                            liquidity = float(pair.get("liquidity", 0))
                            
                            if liquidity > 100:  # At least $100 liquidity
                                sol_pairs.append({
                                    "base_mint": base_mint,
                                    "liquidity": liquidity,
                                    "price": float(pair.get("price", 0)),
                                    "volume24h": float(pair.get("volume24h", 0)),
                                    "name": pair.get("name", "Unknown"),
                                    "amm_id": pair.get("ammId", "")
                                })
                    
                    # Sort by liquidity
                    sol_pairs.sort(key=lambda x: x['liquidity'], reverse=True)
                    
                    print(f"🎯 Found {len(sol_pairs)} SOL trading pairs with >$100 liquidity")
                    
                    if sol_pairs:
                        print(f"\n🔥 TOP 20 SOL PAIRS BY LIQUIDITY:")
                        print("-" * 100)
                        print(f"{'#':<3} {'Address':<45} {'Liquidity':<15} {'Price (SOL)':<15} {'Volume 24h':<15}")
                        print("-" * 100)
                        
                        for i, pair in enumerate(sol_pairs[:20], 1):
                            address_short = f"{pair['base_mint'][:8]}...{pair['base_mint'][-8:]}"
                            
                            print(f"{i:<3} {address_short:<45} ${pair['liquidity']:<14,.0f} {pair['price']:<15.8f} ${pair['volume24h']:<14,.0f}")
                        
                        print("-" * 100)
                        print(f"💰 Total liquidity in top 20: ${sum(p['liquidity'] for p in sol_pairs[:20]):,.0f}")
                        
                        # Show some stats
                        high_volume = [p for p in sol_pairs if p['volume24h'] > 10000]
                        new_low_price = [p for p in sol_pairs if p['price'] < 0.001]
                        
                        print(f"\n📊 QUICK STATS:")
                        print(f"   High volume (>$10k): {len(high_volume)} pairs")
                        print(f"   Low price (<0.001 SOL): {len(new_low_price)} pairs")
                        print(f"   Total SOL pairs: {len(sol_pairs)}")
                        
                        # Show top 5 by different criteria
                        print(f"\n🚀 TOP 5 BY VOLUME:")
                        volume_sorted = sorted(sol_pairs, key=lambda x: x['volume24h'], reverse=True)
                        for i, pair in enumerate(volume_sorted[:5], 1):
                            address_short = f"{pair['base_mint'][:8]}...{pair['base_mint'][-8:]}"
                            print(f"   {i}. {address_short} - ${pair['volume24h']:,.0f} volume - ${pair['liquidity']:,.0f} liquidity")
                        
                        print(f"\n🎲 TOP 5 LOWEST PRICE (POTENTIAL GEMS):")
                        price_sorted = sorted([p for p in sol_pairs if p['price'] > 0], key=lambda x: x['price'])
                        for i, pair in enumerate(price_sorted[:5], 1):
                            address_short = f"{pair['base_mint'][:8]}...{pair['base_mint'][-8:]}"
                            print(f"   {i}. {address_short} - {pair['price']:.10f} SOL - ${pair['liquidity']:,.0f} liquidity")
                    
                    else:
                        print("❌ No SOL pairs found with sufficient liquidity")
                
                else:
                    print(f"❌ API Error: {response.status}")
        
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(fast_solana_scan()) 