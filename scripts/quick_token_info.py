#!/usr/bin/env python3
"""
QUICK TOKEN INFO
===============

Simple script to quickly get token information for any address.
Usage: python quick_token_info.py <token_address>
"""

import asyncio
import aiohttp
import sys
from datetime import datetime

async def get_token_info_quick(token_address: str):
    """Quick token info from DexScreener (most reliable when pump.fun is down)"""
    
    print(f"🔍 Getting info for: {token_address}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        try:
            # Try DexScreener first (most reliable)
            print("🎯 Checking DexScreener...")
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get("pairs", [])
                    
                    if pairs:
                        pair = pairs[0]
                        base_token = pair.get("baseToken", {})
                        
                        print(f"✅ FOUND ON DEXSCREENER!")
                        print(f"🏷️  NAME: {base_token.get('name', 'Unknown')}")
                        print(f"🎯 SYMBOL: {base_token.get('symbol', 'UNKNOWN')}")
                        print(f"💰 PRICE: ${pair.get('priceUsd', '0')}")
                        print(f"📊 MARKET CAP: ${float(pair.get('marketCap', 0)):,.2f}")
                        print(f"💧 LIQUIDITY: ${float(pair.get('liquidity', {}).get('usd', 0)):,.2f}")
                        print(f"📈 24H CHANGE: {pair.get('priceChange', {}).get('h24', '0')}%")
                        print(f"🔥 24H VOLUME: ${float(pair.get('volume', {}).get('h24', 0)):,.2f}")
                        print(f"🏊 DEX: {pair.get('dexId', 'Unknown').upper()}")
                        
                        created_at = pair.get('pairCreatedAt', 0)
                        if created_at:
                            created_time = datetime.fromtimestamp(created_at / 1000)
                            age = datetime.now() - created_time
                            if age.days > 0:
                                age_str = f"{age.days} days ago"
                            elif age.seconds > 3600:
                                age_str = f"{age.seconds // 3600} hours ago"
                            else:
                                age_str = f"{age.seconds // 60} minutes ago"
                            print(f"🕐 PAIR CREATED: {age_str}")
                        
                        print(f"\n🔗 LINKS:")
                        print(f"   🔗 PUMP: https://pump.fun/{token_address}")
                        print(f"   🔗 SCAN: https://solscan.io/token/{token_address}")
                        print(f"   🔗 DEX: https://dexscreener.com/solana/{token_address}")
                        
                        return True
                    else:
                        print("❌ Not found on DexScreener")
            
            # Try Solscan as backup
            print("\n🎯 Checking Solscan...")
            url = f"https://api.solscan.io/token/meta?token={token_address}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    print(f"✅ FOUND ON SOLSCAN!")
                    print(f"🏷️  NAME: {data.get('name', 'Unknown')}")
                    print(f"🎯 SYMBOL: {data.get('symbol', 'UNKNOWN')}")
                    print(f"🔢 DECIMALS: {data.get('decimals', 0)}")
                    print(f"📊 SUPPLY: {data.get('supply', '0')}")
                    
                    if data.get('website'):
                        print(f"🌐 WEBSITE: {data['website']}")
                    if data.get('twitter'):
                        print(f"🐦 TWITTER: {data['twitter']}")
                    
                    print(f"\n🔗 LINKS:")
                    print(f"   🔗 PUMP: https://pump.fun/{token_address}")
                    print(f"   🔗 SCAN: https://solscan.io/token/{token_address}")
                    print(f"   🔗 DEX: https://dexscreener.com/solana/{token_address}")
                    
                    return True
                else:
                    print("❌ Not found on Solscan")
            
            print("\n❌ Token not found on any source")
            print("💡 This might be a very new token that hasn't been indexed yet")
            print("🕐 Try again in a few minutes")
            
            return False
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

async def main():
    """Main function"""
    
    if len(sys.argv) != 2:
        print("Usage: python quick_token_info.py <token_address>")
        print("\nExample:")
        print("python quick_token_info.py EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
        return
    
    token_address = sys.argv[1]
    
    if len(token_address) != 44:
        print("❌ Invalid token address. Should be 44 characters long.")
        return
    
    await get_token_info_quick(token_address)

if __name__ == "__main__":
    asyncio.run(main()) 