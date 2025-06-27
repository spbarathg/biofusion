#!/usr/bin/env python3
"""
ALTERNATIVE FRESH MONITOR
========================

Uses alternative APIs and blockchain monitoring for fresh launches.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class AlternativeFreshMonitor:
    """Alternative monitor using multiple sources"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.helius_url = f"https://mainnet.helius-rpc.com/?api-key={helius_api_key}"
        self.session = None
        self.seen_tokens = set()
        self.launch_count = 0
        self.start_time = datetime.now()
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            resolver=aiohttp.AsyncResolver(),
            use_dns_cache=False
        )
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*"
        }
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers=headers,
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_dexscreener_api(self):
        """Check DexScreener for new Solana tokens"""
        
        fresh_tokens = []
        
        try:
            # DexScreener new tokens
            url = "https://api.dexscreener.com/latest/dex/tokens/solana"
            
            print(f"      📊 Checking DexScreener...", end=" ")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    print(f"Found {len(pairs)} pairs")
                    
                    now = datetime.now()
                    found_fresh = 0
                    
                    for pair in pairs[:20]:  # Check first 20
                        base_token = pair.get('baseToken', {})
                        address = base_token.get('address', '')
                        created_at = pair.get('pairCreatedAt', 0)
                        
                        if address and created_at and address not in self.seen_tokens:
                            try:
                                created_time = datetime.fromtimestamp(created_at / 1000)
                                age_minutes = (now - created_time).total_seconds() / 60
                                
                                if age_minutes <= 120:  # Last 2 hours
                                    found_fresh += 1
                                    
                                    fresh_tokens.append({
                                        "address": address,
                                        "name": base_token.get('name', 'Unknown'),
                                        "symbol": base_token.get('symbol', 'UNKNOWN'),
                                        "age_minutes": age_minutes,
                                        "platform": "DexScreener",
                                        "created_time": created_time.strftime('%H:%M:%S'),
                                        "price": float(pair.get('priceNative', 0)),
                                        "liquidity": float(pair.get('liquidity', {}).get('usd', 0))
                                    })
                                    
                                    self.seen_tokens.add(address)
                            
                            except:
                                continue
                    
                    print(f" -> {found_fresh} fresh")
                
                else:
                    print(f"Error {response.status}")
        
        except Exception as e:
            print(f"Error: {str(e)[:30]}")
        
        return fresh_tokens
    
    async def check_helius_recent_transactions(self):
        """Use Helius to find recent token creation transactions"""
        
        fresh_tokens = []
        
        try:
            print(f"      ⚡ Checking Helius blockchain...", end=" ")
            
            # Get recent signatures for SPL token program
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [
                    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # SPL Token program
                    {"limit": 10}
                ]
            }
            
            async with self.session.post(self.helius_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    signatures = data.get('result', [])
                    
                    print(f"Found {len(signatures)} recent txs")
                    
                    found_fresh = 0
                    
                    for sig_info in signatures[:5]:  # Check recent transactions
                        signature = sig_info.get('signature', '')
                        block_time = sig_info.get('blockTime', 0)
                        
                        if block_time:
                            tx_time = datetime.fromtimestamp(block_time)
                            age_minutes = (datetime.now() - tx_time).total_seconds() / 60
                            
                            if age_minutes <= 10:  # Very recent
                                # Get transaction details for token creation
                                new_token = await self.get_transaction_token_details(signature, tx_time)
                                if new_token:
                                    found_fresh += 1
                                    fresh_tokens.append(new_token)
                    
                    print(f" -> {found_fresh} token creations")
                
                else:
                    print(f"Error {response.status}")
        
        except Exception as e:
            print(f"Error: {str(e)[:30]}")
        
        return fresh_tokens
    
    async def get_transaction_token_details(self, signature, tx_time):
        """Get details of a transaction to find new token creation"""
        
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    signature,
                    {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
                ]
            }
            
            async with self.session.post(self.helius_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "result" in data and data["result"]:
                        tx = data["result"]
                        
                        # Look for token mint instructions
                        if "transaction" in tx and "message" in tx["transaction"]:
                            instructions = tx["transaction"]["message"].get("instructions", [])
                            
                            for instruction in instructions:
                                if instruction.get("program") == "spl-token":
                                    parsed = instruction.get("parsed", {})
                                    
                                    if parsed.get("type") == "initializeMint":
                                        accounts = instruction.get("accounts", [])
                                        if accounts and accounts[0] not in self.seen_tokens:
                                            mint_address = accounts[0]
                                            
                                            age_minutes = (datetime.now() - tx_time).total_seconds() / 60
                                            
                                            self.seen_tokens.add(mint_address)
                                            
                                            return {
                                                "address": mint_address,
                                                "name": "New Token",
                                                "symbol": "NEW",
                                                "age_minutes": age_minutes,
                                                "platform": "Blockchain",
                                                "created_time": tx_time.strftime('%H:%M:%S'),
                                                "signature": signature
                                            }
        
        except Exception as e:
            pass
        
        return None
    
    async def display_fresh_launch(self, token):
        """Display fresh launch"""
        
        self.launch_count += 1
        
        age_minutes = token.get('age_minutes', 0)
        platform = token.get('platform', 'Unknown')
        
        if age_minutes <= 5:
            freshness = "🔥🔥🔥 ULTRA FRESH"
            urgency = "URGENT"
        elif age_minutes <= 30:
            freshness = "🔥🔥 VERY FRESH"
            urgency = "HIGH"
        elif age_minutes <= 120:
            freshness = "🔥 FRESH"
            urgency = "MEDIUM"
        else:
            freshness = "⏰ Recent"
            urgency = "LOW"
        
        print(f"\n" + "🚀" * 60)
        print(f"🚀 FRESH LAUNCH #{self.launch_count} - {datetime.now().strftime('%H:%M:%S')} 🚀")
        print(f"🚀" * 60)
        print(f"📍 ADDRESS: {token['address']}")
        print(f"🏷️  NAME: {token.get('name', 'Unknown')}")
        print(f"🎯 SYMBOL: {token.get('symbol', 'UNKNOWN')}")
        
        if age_minutes > 0:
            print(f"⏰ AGE: {age_minutes:.1f} minutes")
        else:
            print(f"⏰ AGE: Unknown")
        
        print(f"🔥 FRESHNESS: {freshness}")
        print(f"🚨 PRIORITY: {urgency}")
        print(f"🏭 SOURCE: {platform}")
        print(f"🕐 CREATED: {token.get('created_time', 'Unknown')}")
        
        if token.get('price'):
            print(f"💰 PRICE: ${token['price']:.8f}")
        
        if token.get('liquidity'):
            print(f"💎 LIQUIDITY: ${token['liquidity']:,.2f}")
        
        if token.get('market_cap'):
            print(f"📊 MARKET CAP: ${token['market_cap']:,.2f}")
        
        print(f"🔗 SOLSCAN: https://solscan.io/token/{token['address']}")
        print(f"🔗 DEXSCREENER: https://dexscreener.com/solana/{token['address']}")
        
        if platform == "Blockchain" and token.get('signature'):
            print(f"🔗 TRANSACTION: https://solscan.io/tx/{token['signature']}")
        
        print(f"⏰ RUNTIME: {(datetime.now() - self.start_time).total_seconds():.0f}s")
        print(f"🚀" * 60)
    
    async def monitor_alternative_sources(self):
        """Main monitoring using alternative sources"""
        
        print("🔄 ALTERNATIVE FRESH LAUNCH MONITOR")
        print("=" * 60)
        print("🎯 Alternative API sources:")
        print("   • DexScreener API") 
        print("   • Helius Blockchain direct")
        print("⚡ Scanning every 20 seconds")
        print("🚨 Working around pump.fun API issues")
        print("")
        
        while True:
            try:
                scan_time = datetime.now().strftime('%H:%M:%S')
                runtime = (datetime.now() - self.start_time).total_seconds()
                
                print(f"\n🔍 ALTERNATIVE SCAN - {scan_time} | Runtime: {runtime:.0f}s | Found: {self.launch_count}")
                
                # Check alternative sources
                all_fresh = []
                
                dex_tokens = await self.check_dexscreener_api()
                all_fresh.extend(dex_tokens)
                
                helius_tokens = await self.check_helius_recent_transactions()
                all_fresh.extend(helius_tokens)
                
                if all_fresh:
                    # Sort by freshness
                    all_fresh.sort(key=lambda x: x.get('age_minutes', 999))
                    
                    print(f"\n🎉 FOUND {len(all_fresh)} FRESH LAUNCHES!")
                    
                    for token in all_fresh[:10]:  # Show top 10
                        await self.display_fresh_launch(token)
                        await asyncio.sleep(0.5)
                
                else:
                    print("   💤 No fresh launches detected")
                
                print(f"   ⏳ Next scan in 20 seconds...")
                await asyncio.sleep(20)
                
            except KeyboardInterrupt:
                print(f"\n⏹️ Alternative monitor stopped")
                print(f"📊 Total fresh launches: {self.launch_count}")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(30)

async def main():
    """Main function"""
    
    helius_api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
    
    async with AlternativeFreshMonitor(helius_api_key) as monitor:
        await monitor.monitor_alternative_sources()

if __name__ == "__main__":
    asyncio.run(main()) 