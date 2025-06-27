#!/usr/bin/env python3
"""
GET TOKEN INFO - MULTIPLE SOURCES
================================

When pump.fun API is down, use these alternative methods to get token information.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import base64
import struct

class TokenInfoGetter:
    """Get token information from multiple sources"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_token_info_helius(self, token_address: str):
        """Get token info using Helius enhanced API"""
        
        print(f"🔍 Method 1: Helius Enhanced API")
        print(f"   Token: {token_address}")
        
        try:
            url = f"https://mainnet.helius-rpc.com/?api-key={self.helius_api_key}"
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAsset",
                "params": {
                    "id": token_address,
                    "displayOptions": {
                        "showFungible": True
                    }
                }
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    
                    if result:
                        content = result.get("content", {})
                        metadata = content.get("metadata", {})
                        
                        token_info = {
                            "source": "Helius",
                            "address": token_address,
                            "name": metadata.get("name", "Unknown"),
                            "symbol": metadata.get("symbol", "UNKNOWN"),
                            "description": metadata.get("description", ""),
                            "image": metadata.get("image", ""),
                            "supply": result.get("supply", {}).get("print_current_supply", 0),
                            "decimals": result.get("token_info", {}).get("decimals", 0)
                        }
                        
                        print(f"   ✅ Found: {token_info['name']} ({token_info['symbol']})")
                        return token_info
        
        except Exception as e:
            print(f"   ❌ Helius error: {e}")
        
        return None
    
    async def get_token_info_jupiter(self, token_address: str):
        """Get token info using Jupiter API"""
        
        print(f"🔍 Method 2: Jupiter API")
        
        try:
            url = f"https://token.jup.ag/strict"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    tokens = await response.json()
                    
                    # Find token in Jupiter's list
                    for token in tokens:
                        if token.get("address") == token_address:
                            token_info = {
                                "source": "Jupiter",
                                "address": token_address,
                                "name": token.get("name", "Unknown"),
                                "symbol": token.get("symbol", "UNKNOWN"),
                                "description": "",
                                "image": token.get("logoURI", ""),
                                "decimals": token.get("decimals", 0),
                                "tags": token.get("tags", [])
                            }
                            
                            print(f"   ✅ Found: {token_info['name']} ({token_info['symbol']})")
                            return token_info
                    
                    print(f"   ❌ Token not found in Jupiter list")
        
        except Exception as e:
            print(f"   ❌ Jupiter error: {e}")
        
        return None
    
    async def get_token_info_dexscreener(self, token_address: str):
        """Get token info using DexScreener API"""
        
        print(f"🔍 Method 3: DexScreener API")
        
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get("pairs", [])
                    
                    if pairs:
                        # Get info from first pair
                        pair = pairs[0]
                        base_token = pair.get("baseToken", {})
                        
                        token_info = {
                            "source": "DexScreener",
                            "address": token_address,
                            "name": base_token.get("name", "Unknown"),
                            "symbol": base_token.get("symbol", "UNKNOWN"),
                            "description": "",
                            "price_usd": pair.get("priceUsd", "0"),
                            "price_change_24h": pair.get("priceChange", {}).get("h24", "0"),
                            "volume_24h": pair.get("volume", {}).get("h24", "0"),
                            "market_cap": pair.get("marketCap", "0"),
                            "liquidity": pair.get("liquidity", {}).get("usd", "0"),
                            "pair_address": pair.get("pairAddress", ""),
                            "dex": pair.get("dexId", ""),
                            "pair_created_at": pair.get("pairCreatedAt", 0)
                        }
                        
                        print(f"   ✅ Found: {token_info['name']} ({token_info['symbol']})")
                        print(f"       Price: ${token_info['price_usd']}")
                        print(f"       Market Cap: ${float(token_info['market_cap']):,.2f}" if token_info['market_cap'] else "")
                        return token_info
                    else:
                        print(f"   ❌ No pairs found")
        
        except Exception as e:
            print(f"   ❌ DexScreener error: {e}")
        
        return None
    
    async def get_token_info_solscan(self, token_address: str):
        """Get token info using Solscan API"""
        
        print(f"🔍 Method 4: Solscan API")
        
        try:
            url = f"https://api.solscan.io/token/meta?token={token_address}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    token_info = {
                        "source": "Solscan",
                        "address": token_address,
                        "name": data.get("name", "Unknown"),
                        "symbol": data.get("symbol", "UNKNOWN"),
                        "description": "",
                        "decimals": data.get("decimals", 0),
                        "supply": data.get("supply", "0"),
                        "icon": data.get("icon", ""),
                        "website": data.get("website", ""),
                        "twitter": data.get("twitter", "")
                    }
                    
                    print(f"   ✅ Found: {token_info['name']} ({token_info['symbol']})")
                    return token_info
        
        except Exception as e:
            print(f"   ❌ Solscan error: {e}")
        
        return None
    
    async def get_token_info_solana_rpc(self, token_address: str):
        """Get token info using direct Solana RPC"""
        
        print(f"🔍 Method 5: Direct Solana RPC")
        
        try:
            url = f"https://mainnet.helius-rpc.com/?api-key={self.helius_api_key}"
            
            # Get token account info
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [
                    token_address,
                    {
                        "encoding": "jsonParsed"
                    }
                ]
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data.get("result", {})
                    
                    if result and result.get("value"):
                        account_data = result["value"]["data"]
                        
                        if account_data.get("program") == "spl-token":
                            parsed = account_data.get("parsed", {})
                            info = parsed.get("info", {})
                            
                            token_info = {
                                "source": "Solana RPC",
                                "address": token_address,
                                "mint_authority": info.get("mintAuthority"),
                                "supply": info.get("supply", "0"),
                                "decimals": info.get("decimals", 0),
                                "is_initialized": info.get("isInitialized", False),
                                "freeze_authority": info.get("freezeAuthority")
                            }
                            
                            print(f"   ✅ Found token account info")
                            print(f"       Decimals: {token_info['decimals']}")
                            print(f"       Supply: {token_info['supply']}")
                            return token_info
        
        except Exception as e:
            print(f"   ❌ Solana RPC error: {e}")
        
        return None
    
    async def get_comprehensive_token_info(self, token_address: str):
        """Get token info from all available sources"""
        
        print(f"\n🎯 GETTING COMPREHENSIVE TOKEN INFO")
        print(f"📍 Token: {token_address}")
        print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        # Try all methods
        methods = [
            self.get_token_info_helius,
            self.get_token_info_jupiter,
            self.get_token_info_dexscreener,
            self.get_token_info_solscan,
            self.get_token_info_solana_rpc
        ]
        
        all_info = {}
        
        for method in methods:
            try:
                info = await method(token_address)
                if info:
                    all_info[info["source"]] = info
                print()  # Spacing
                await asyncio.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"❌ Method failed: {e}")
        
        # Compile best info
        if all_info:
            print("🎉 COMPILED TOKEN INFORMATION")
            print("=" * 60)
            
            # Get the best name/symbol from available sources
            best_name = "Unknown"
            best_symbol = "UNKNOWN"
            
            for source, info in all_info.items():
                if info.get("name") and info["name"] != "Unknown":
                    best_name = info["name"]
                if info.get("symbol") and info["symbol"] != "UNKNOWN":
                    best_symbol = info["symbol"]
                break
            
            print(f"🏷️  NAME: {best_name}")
            print(f"🎯 SYMBOL: {best_symbol}")
            print(f"📍 ADDRESS: {token_address}")
            
            # Show info from each source
            for source, info in all_info.items():
                print(f"\n📊 {source.upper()} DATA:")
                for key, value in info.items():
                    if key not in ["source", "address"] and value:
                        print(f"   {key}: {value}")
            
            # Generate links
            print(f"\n🔗 USEFUL LINKS:")
            print(f"   🔗 PUMP: https://pump.fun/{token_address}")
            print(f"   🔗 SCAN: https://solscan.io/token/{token_address}")
            print(f"   🔗 DEX: https://dexscreener.com/solana/{token_address}")
            
            return all_info
        else:
            print("❌ No token information found from any source")
            return None

async def main():
    """Test token info gathering"""
    
    # Example token addresses (replace with actual tokens you want to check)
    test_tokens = [
        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC (known token)
        "So11111111111111111111111111111111111111112",   # SOL (wrapped)
        # Add actual pump.fun token addresses here
    ]
    
    helius_api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
    
    async with TokenInfoGetter(helius_api_key) as getter:
        for token_address in test_tokens:
            await getter.get_comprehensive_token_info(token_address)
            print("\n" + "="*80 + "\n")
            await asyncio.sleep(1)

if __name__ == "__main__":
    print("🔍 Starting Token Information Gathering!")
    print("💡 Multiple sources when pump.fun API is down")
    asyncio.run(main()) 