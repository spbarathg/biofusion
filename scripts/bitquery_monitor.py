#!/usr/bin/env python3
"""
BITQUERY REAL-TIME MONITOR
==========================

Uses Bitquery's GraphQL API to get REAL-TIME pump.fun token launches.
This should actually work and show tokens launching!
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class BitqueryMonitor:
    """Real-time pump.fun monitor using Bitquery GraphQL"""
    
    def __init__(self):
        self.api_url = "https://streaming.bitquery.io/graphql"
        self.session = None
        self.launch_count = 0
        self.start_time = datetime.now()
        self.seen_tokens = set()
        
    async def __aenter__(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer ory_at_...",  # You'd need to get a free API key
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_recent_tokens(self):
        """Get recent pump.fun token launches"""
        
        # Bitquery GraphQL query for recent pump.fun tokens
        query = """
        {
          Solana {
            TokenSupplyUpdates(
              where: {
                Instruction: {
                  Program: {
                    Address: { is: "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P" }
                    Method: { is: "create" }
                  }
                }
                Block: { Time: { since: "2024-12-30T00:00:00Z" } }
              }
              orderBy: { descending: Block_Time }
              limit: { count: 20 }
            ) {
              Block {
                Time
              }
              Transaction {
                Signer
                Signature
              }
              TokenSupplyUpdate {
                Amount
                Currency {
                  Symbol
                  Name
                  MintAddress
                  MetadataAddress
                  Decimals
                  Uri
                }
                PostBalance
              }
            }
          }
        }
        """
        
        try:
            payload = {"query": query}
            
            async with self.session.post(self.api_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "data" in data and "Solana" in data["data"]:
                        updates = data["data"]["Solana"]["TokenSupplyUpdates"]
                        return updates
                    else:
                        print(f"⚠️ Unexpected response format: {data}")
                else:
                    text = await response.text()
                    print(f"❌ API Error {response.status}: {text[:200]}")
        
        except Exception as e:
            print(f"❌ Query error: {e}")
        
        return []
    
    def display_token_launch(self, token_data):
        """Display new token launch"""
        
        block = token_data.get("Block", {})
        transaction = token_data.get("Transaction", {})
        token_update = token_data.get("TokenSupplyUpdate", {})
        currency = token_update.get("Currency", {})
        
        mint_address = currency.get("MintAddress", "Unknown")
        name = currency.get("Name", "Unknown")
        symbol = currency.get("Symbol", "UNKNOWN")
        decimals = currency.get("Decimals", 0)
        uri = currency.get("Uri", "")
        creator = transaction.get("Signer", "")
        signature = transaction.get("Signature", "")
        supply = token_update.get("PostBalance", 0)
        block_time = block.get("Time", "")
        
        # Calculate age
        age_str = "Unknown"
        if block_time:
            try:
                created_time = datetime.fromisoformat(block_time.replace('Z', '+00:00'))
                age_seconds = (datetime.now().replace(tzinfo=created_time.tzinfo) - created_time).total_seconds()
                if age_seconds < 60:
                    age_str = f"{age_seconds:.0f} seconds ago"
                elif age_seconds < 3600:
                    age_str = f"{age_seconds/60:.1f} minutes ago"
                else:
                    age_str = f"{age_seconds/3600:.1f} hours ago"
            except:
                age_str = "Recent"
        
        self.launch_count += 1
        
        print(f"\n{'🚀' * 60}")
        print(f"🚀 PUMP.FUN TOKEN LAUNCH #{self.launch_count} - {datetime.now().strftime('%H:%M:%S')} 🚀")
        print(f"{'🚀' * 60}")
        
        print(f"📍 ADDRESS: {mint_address}")
        print(f"🏷️  NAME: {name}")
        print(f"🎯 SYMBOL: {symbol}")
        print(f"⏰ AGE: {age_str}")
        print(f"🔢 DECIMALS: {decimals}")
        
        if supply and supply != "0":
            supply_formatted = f"{float(supply):,.0f}"
            print(f"📊 SUPPLY: {supply_formatted}")
        
        if creator:
            print(f"👤 CREATOR: {creator}")
        
        if uri:
            print(f"📄 METADATA: {uri}")
        
        print(f"\n🔗 QUICK LINKS:")
        print(f"   🔗 PUMP: https://pump.fun/{mint_address}")
        print(f"   🔗 SCAN: https://solscan.io/token/{mint_address}")
        print(f"   🔗 DEX: https://dexscreener.com/solana/{mint_address}")
        print(f"   📍 TX: {signature}")
        
        # Stats
        runtime = (datetime.now() - self.start_time).total_seconds()
        rate = self.launch_count / (runtime / 60) if runtime > 0 else 0
        print(f"\n📊 SESSION: {self.launch_count} launches | {runtime:.0f}s | {rate:.1f}/min")
        
        print(f"{'🚀' * 60}")
        
        # Save to file
        self.save_launch(mint_address, name, symbol, creator, signature)
    
    def save_launch(self, mint_address, name, symbol, creator, signature):
        """Save launch to log file"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/bitquery_launches.txt", "a") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp} | {mint_address} | {symbol} | {name} | {creator} | {signature}\n")
    
    async def monitor_continuous(self):
        """Continuously monitor for new tokens"""
        
        print("🚀 BITQUERY PUMP.FUN MONITOR")
        print("=" * 50)
        print("🎯 Real-time pump.fun token detection")
        print("📊 GraphQL API via Bitquery")
        print("⚡ Polling every 10 seconds")
        print("🚀 Shows new token launches")
        print("")
        
        scan_count = 0
        
        try:
            while True:
                scan_count += 1
                scan_time = datetime.now().strftime('%H:%M:%S')
                
                print(f"🔍 SCAN #{scan_count} - {scan_time} | Getting recent tokens...")
                
                tokens = await self.get_recent_tokens()
                
                if tokens:
                    new_count = 0
                    for token in tokens:
                        mint_address = token.get("TokenSupplyUpdate", {}).get("Currency", {}).get("MintAddress", "")
                        
                        if mint_address and mint_address not in self.seen_tokens:
                            self.seen_tokens.add(mint_address)
                            self.display_token_launch(token)
                            new_count += 1
                    
                    if new_count == 0:
                        print(f"   📡 No new tokens (found {len(tokens)} existing)")
                    else:
                        print(f"   ✅ Found {new_count} new tokens!")
                else:
                    print(f"   ❌ No data received")
                
                print(f"💤 Waiting 10 seconds...")
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitor stopped")
            runtime = (datetime.now() - self.start_time).total_seconds()
            rate = self.launch_count / (runtime / 60) if runtime > 0 else 0
            print(f"📊 Final: {self.launch_count} launches in {runtime:.0f}s ({rate:.1f}/min)")

async def main():
    """Main function with fallback to free endpoint"""
    
    print("🚀 Starting Bitquery Monitor!")
    print("💡 Uses GraphQL to get pump.fun token data")
    print("")
    
    # Try without auth first (may work with free tier)
    monitor = BitqueryMonitor()
    monitor.session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        await monitor.monitor_continuous()
    finally:
        if monitor.session:
            await monitor.session.close()

if __name__ == "__main__":
    asyncio.run(main()) 