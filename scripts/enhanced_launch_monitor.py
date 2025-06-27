#!/usr/bin/env python3
"""
ENHANCED LAUNCH MONITOR WITH TOKEN INFO
=======================================

Monitors for new token launches AND automatically gets comprehensive token info.
Works when pump.fun API is down by using multiple backup sources.
"""

import asyncio
import websockets
import aiohttp
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class EnhancedLaunchMonitor:
    """Enhanced monitor that detects launches and gets full token info"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.websocket_url = f"wss://mainnet.helius-rpc.com/?api-key={helius_api_key}"
        self.pump_fun_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        self.websocket = None
        self.session = None
        self.launch_count = 0
        self.start_time = datetime.now()
        
    async def connect(self):
        """Connect WebSocket and HTTP session"""
        
        print("🔗 Connecting to enhanced monitor...")
        
        try:
            # WebSocket connection
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=30,
                ping_timeout=10
            )
            
            # HTTP session for API calls
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
            
            print("✅ Connected!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def subscribe_to_launches(self):
        """Subscribe to pump.fun logs"""
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {"mentions": [self.pump_fun_program_id]},
                {"commitment": "processed"}
            ]
        }
        
        try:
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if "result" in response_data:
                print(f"✅ Subscribed to launches!")
                return True
            else:
                print(f"❌ Subscription failed: {response_data}")
                return False
                
        except Exception as e:
            print(f"❌ Subscription error: {e}")
            return False
    
    def extract_token_address(self, logs):
        """Extract potential token addresses from logs"""
        
        addresses = []
        for log in logs:
            words = log.split()
            for word in words:
                if len(word) == 44 and word.replace('_', '').replace('-', '').isalnum():
                    if word not in addresses:
                        addresses.append(word)
        
        return addresses
    
    async def get_token_info_dexscreener(self, token_address: str):
        """Get comprehensive token info from DexScreener"""
        
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get("pairs", [])
                    
                    if pairs:
                        pair = pairs[0]
                        base_token = pair.get("baseToken", {})
                        
                        return {
                            "found": True,
                            "source": "DexScreener",
                            "name": base_token.get("name", "Unknown"),
                            "symbol": base_token.get("symbol", "UNKNOWN"),
                            "price_usd": pair.get("priceUsd", "0"),
                            "market_cap": pair.get("marketCap", "0"),
                            "liquidity_usd": pair.get("liquidity", {}).get("usd", "0"),
                            "volume_24h": pair.get("volume", {}).get("h24", "0"),
                            "price_change_24h": pair.get("priceChange", {}).get("h24", "0"),
                            "pair_created_at": pair.get("pairCreatedAt", 0),
                            "dex": pair.get("dexId", "unknown")
                        }
        except Exception as e:
            pass
        
        return {"found": False, "source": "DexScreener"}
    
    async def get_token_info_solscan(self, token_address: str):
        """Get basic token info from Solscan"""
        
        try:
            url = f"https://api.solscan.io/token/meta?token={token_address}"
            headers = {"User-Agent": "Mozilla/5.0"}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    return {
                        "found": True,
                        "source": "Solscan",
                        "name": data.get("name", "Unknown"),
                        "symbol": data.get("symbol", "UNKNOWN"),
                        "decimals": data.get("decimals", 0),
                        "supply": data.get("supply", "0"),
                        "website": data.get("website", ""),
                        "twitter": data.get("twitter", "")
                    }
        except Exception as e:
            pass
        
        return {"found": False, "source": "Solscan"}
    
    async def get_comprehensive_token_info(self, token_address: str):
        """Get token info from multiple sources"""
        
        print(f"🔍 Getting comprehensive info for: {token_address[:20]}...")
        
        # Try both sources simultaneously
        dex_task = self.get_token_info_dexscreener(token_address)
        solscan_task = self.get_token_info_solscan(token_address)
        
        dex_info, solscan_info = await asyncio.gather(dex_task, solscan_task)
        
        # Combine the best information
        combined_info = {
            "address": token_address,
            "name": "Unknown",
            "symbol": "UNKNOWN",
            "found_sources": [],
            "timestamp": datetime.now()
        }
        
        if dex_info["found"]:
            combined_info.update(dex_info)
            combined_info["found_sources"].append("DexScreener")
        
        if solscan_info["found"]:
            # Use Solscan for basic info if DexScreener doesn't have it
            if combined_info["name"] == "Unknown":
                combined_info["name"] = solscan_info["name"]
            if combined_info["symbol"] == "UNKNOWN":
                combined_info["symbol"] = solscan_info["symbol"]
            
            combined_info.update({k: v for k, v in solscan_info.items() 
                                if k not in combined_info or not combined_info[k]})
            combined_info["found_sources"].append("Solscan")
        
        return combined_info
    
    def is_potential_launch(self, logs):
        """Check if logs indicate a potential token launch"""
        
        indicators = [
            "Program 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P invoke",
            "create",
            "initialize",
            "mint"
        ]
        
        for log in logs:
            for indicator in indicators:
                if indicator.lower() in log.lower():
                    return True
        
        # Also check for multiple addresses (typical in launches)
        addresses = self.extract_token_address(logs)
        return len(addresses) >= 2
    
    async def display_enhanced_launch(self, token_info, signature):
        """Display launch with comprehensive token information"""
        
        self.launch_count += 1
        
        print(f"\n{'🚀' * 70}")
        print(f"🚀 ENHANCED TOKEN LAUNCH #{self.launch_count} - {datetime.now().strftime('%H:%M:%S')} 🚀")
        print(f"{'🚀' * 70}")
        
        print(f"📍 ADDRESS: {token_info['address']}")
        print(f"🏷️  NAME: {token_info['name']}")
        print(f"🎯 SYMBOL: {token_info['symbol']}")
        
        if token_info.get("found_sources"):
            print(f"📊 DATA SOURCES: {', '.join(token_info['found_sources'])}")
        
        # DexScreener data (trading info)
        if "price_usd" in token_info:
            print(f"💰 PRICE: ${token_info['price_usd']}")
            
        if "market_cap" in token_info and token_info["market_cap"] != "0":
            mc = float(token_info["market_cap"])
            print(f"📊 MARKET CAP: ${mc:,.2f}")
            
        if "liquidity_usd" in token_info and token_info["liquidity_usd"] != "0":
            liq = float(token_info["liquidity_usd"])
            print(f"💧 LIQUIDITY: ${liq:,.2f}")
            
        if "volume_24h" in token_info and token_info["volume_24h"] != "0":
            vol = float(token_info["volume_24h"])
            print(f"🔥 24H VOLUME: ${vol:,.2f}")
            
        if "price_change_24h" in token_info and token_info["price_change_24h"] != "0":
            change = float(token_info["price_change_24h"])
            change_emoji = "📈" if change > 0 else "📉"
            print(f"{change_emoji} 24H CHANGE: {change:.2f}%")
        
        # Age information
        if "pair_created_at" in token_info and token_info["pair_created_at"]:
            created_time = datetime.fromtimestamp(token_info["pair_created_at"] / 1000)
            age = datetime.now() - created_time
            if age.total_seconds() < 3600:
                age_str = f"{int(age.total_seconds() / 60)} minutes ago"
                urgency = "🚨🚨🚨 ULTRA FRESH"
            elif age.total_seconds() < 86400:
                age_str = f"{int(age.total_seconds() / 3600)} hours ago"
                urgency = "🚨 FRESH"
            else:
                age_str = f"{age.days} days ago"
                urgency = "⏰ Established"
            
            print(f"🕐 PAIR AGE: {age_str}")
            print(f"🚨 URGENCY: {urgency}")
        
        # Solscan data
        if "decimals" in token_info:
            print(f"🔢 DECIMALS: {token_info['decimals']}")
            
        if "supply" in token_info and token_info["supply"] != "0":
            supply = float(token_info["supply"])
            print(f"📊 SUPPLY: {supply:,.0f}")
        
        # Social links
        if token_info.get("website"):
            print(f"🌐 WEBSITE: {token_info['website']}")
        if token_info.get("twitter"):
            print(f"🐦 TWITTER: {token_info['twitter']}")
        
        # Links
        print(f"\n🔗 QUICK LINKS:")
        print(f"   🔗 PUMP: https://pump.fun/{token_info['address']}")
        print(f"   🔗 SCAN: https://solscan.io/token/{token_info['address']}")
        print(f"   🔗 DEX: https://dexscreener.com/solana/{token_info['address']}")
        print(f"   📍 TX: {signature}")
        
        # Stats
        runtime = (datetime.now() - self.start_time).total_seconds()
        rate = self.launch_count / (runtime / 60) if runtime > 0 else 0
        print(f"\n📊 SESSION: {self.launch_count} launches | {runtime:.0f}s runtime | {rate:.1f}/min")
        
        print(f"{'🚀' * 70}")
        
        # Save to file
        await self.save_launch(token_info, signature)
    
    async def save_launch(self, token_info, signature):
        """Save launch with comprehensive info"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/enhanced_launches.txt", "a") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            name = token_info["name"]
            symbol = token_info["symbol"]
            address = token_info["address"]
            sources = ",".join(token_info.get("found_sources", []))
            price = token_info.get("price_usd", "0")
            
            f.write(f"{timestamp} | {address} | {symbol} | {name} | ${price} | {sources} | {signature}\n")
    
    async def process_message(self, message):
        """Process WebSocket messages and get comprehensive token info"""
        
        try:
            data = json.loads(message)
            
            if "method" in data and data["method"] == "logsNotification":
                params = data.get("params", {})
                result = params.get("result", {})
                logs = result.get("logs", [])
                signature = result.get("signature", "Unknown")
                
                if self.is_potential_launch(logs):
                    addresses = self.extract_token_address(logs)
                    
                    if addresses:
                        # Get info for the first address (likely the token)
                        token_address = addresses[0]
                        token_info = await self.get_comprehensive_token_info(token_address)
                        await self.display_enhanced_launch(token_info, signature)
        
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
    
    async def monitor_with_info(self):
        """Main monitoring with comprehensive token info"""
        
        print("🚀 ENHANCED LAUNCH MONITOR WITH TOKEN INFO")
        print("=" * 60)
        print("🎯 Detects launches + gets comprehensive token data")
        print("📊 Multiple API sources (DexScreener, Solscan)")
        print("💰 Price, market cap, liquidity, volume data")
        print("🚨 Age and urgency analysis")
        print("🔗 All relevant links included")
        print("")
        
        if not await self.connect():
            return
        
        if not await self.subscribe_to_launches():
            return
        
        print(f"🚀 ENHANCED MONITOR ACTIVE - {datetime.now().strftime('%H:%M:%S')}")
        print("🎯 Waiting for launches with comprehensive token info...")
        print("⚡ Press Ctrl+C to stop")
        print("")
        
        try:
            async for message in self.websocket:
                await self.process_message(message)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Enhanced monitor stopped")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            runtime = (datetime.now() - self.start_time).total_seconds()
            rate = self.launch_count / (runtime / 60) if runtime > 0 else 0
            print(f"📊 Final: {self.launch_count} launches in {runtime:.0f}s ({rate:.1f}/min)")
            
            if self.websocket:
                await self.websocket.close()
            if self.session:
                await self.session.close()

async def main():
    """Main function"""
    
    helius_api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
    
    monitor = EnhancedLaunchMonitor(helius_api_key)
    await monitor.monitor_with_info()

if __name__ == "__main__":
    print("🚀 Starting Enhanced Launch Monitor with Token Info!")
    asyncio.run(main()) 