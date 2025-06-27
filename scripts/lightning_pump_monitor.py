#!/usr/bin/env python3
"""
⚡ LIGHTNING PUMP.FUN MONITOR ⚡
==============================

MAXIMUM SPEED pump.fun token detection:
- Real-time WebSocket monitoring
- Parallel API polling
- Instant mint extraction
- Zero latency display
- Multiple data sources
"""

import asyncio
import websockets
import aiohttp
import json
from datetime import datetime
import sys
import time
from typing import Set, Dict, Any
import signal

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class LightningPumpMonitor:
    def __init__(self, api_key="193ececa-6e42-4d84-b9bd-765c4813816d"):
        self.api_key = api_key
        self.ws_url = f"wss://mainnet.helius-rpc.com/?api-key={api_key}"
        self.http_url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        self.pump_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        
        # Speed tracking
        self.seen_tokens: Set[str] = set()
        self.seen_signatures: Set[str] = set()
        self.token_count = 0
        self.start_time = datetime.now()
        self.running = True
        
        # Session for HTTP requests
        self.session = None
        
        # Last known token from pump.fun API
        self.last_pump_check = 0
        
    async def setup_session(self):
        """Setup optimized HTTP session"""
        connector = aiohttp.TCPConnector(
            limit=200,
            limit_per_host=100,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
            force_close=False
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=3, connect=1),
            headers={
                "Connection": "keep-alive",
                "User-Agent": "LightningMonitor/1.0"
            }
        )
    
    async def start_lightning_monitoring(self):
        """Start lightning-fast multi-source monitoring"""
        print("⚡" * 70)
        print("⚡ LIGHTNING PUMP.FUN MONITOR - MAXIMUM SPEED MODE ⚡")
        print("⚡" * 70)
        print("📡 Multi-source real-time monitoring:")
        print("   🔌 WebSocket (Helius Geyser)")
        print("   🌐 Pump.fun API polling")
        print("   ⚡ Instant detection & display")
        print("⚡" * 70)
        
        await self.setup_session()
        
        try:
            # Start all monitoring tasks in parallel
            tasks = [
                asyncio.create_task(self.monitor_websocket()),
                asyncio.create_task(self.monitor_pump_api()),
                asyncio.create_task(self.print_speed_stats())
            ]
            
            # Wait for any task to complete (they should run forever)
            await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            self.running = False
            if self.session:
                await self.session.close()
    
    async def monitor_websocket(self):
        """Monitor WebSocket for instant detection"""
        retry_count = 0
        
        while self.running and retry_count < 5:
            try:
                print(f"🔌 Connecting to WebSocket...")
                
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=15,
                    ping_timeout=8,
                    close_timeout=5
                ) as websocket:
                    
                    # Subscribe to pump.fun program
                    subscribe = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "logsSubscribe",
                        "params": [
                            {"mentions": [self.pump_program_id]},
                            {"commitment": "processed"}
                        ]
                    }
                    
                    await websocket.send(json.dumps(subscribe))
                    print("✅ WebSocket subscribed to pump.fun program")
                    
                    retry_count = 0
                    
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            if 'params' in data and 'result' in data['params']:
                                await self.process_websocket_event(data['params']['result'])
                        except Exception as e:
                            continue
                            
            except Exception as e:
                retry_count += 1
                print(f"❌ WebSocket error: {e}")
                if retry_count < 5:
                    print(f"🔄 Retrying WebSocket in {retry_count * 2} seconds...")
                    await asyncio.sleep(retry_count * 2)
        
        print("❌ WebSocket monitoring stopped")
    
    async def monitor_pump_api(self):
        """Monitor pump.fun API for new tokens"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check pump.fun API every 2 seconds
                if current_time - self.last_pump_check >= 2:
                    await self.check_pump_api()
                    self.last_pump_check = current_time
                
                await asyncio.sleep(0.5)  # Small delay
                
            except Exception as e:
                print(f"❌ API monitoring error: {e}")
                await asyncio.sleep(1)
    
    async def check_pump_api(self):
        """Check pump.fun API for fresh tokens"""
        try:
            url = "https://frontend-api.pump.fun/coins?offset=0&limit=20&sort=created_timestamp&order=DESC"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    coins = await response.json()
                    
                    now = datetime.now()
                    
                    for coin in coins:
                        mint = coin.get("mint")
                        created_timestamp = coin.get("created_timestamp")
                        
                        if mint and mint not in self.seen_tokens and created_timestamp:
                            # Check if it's very fresh (last 10 minutes)
                            created_time = datetime.fromtimestamp(created_timestamp / 1000)
                            age_seconds = (now - created_time).total_seconds()
                            
                            if age_seconds <= 600:  # 10 minutes
                                self.seen_tokens.add(mint)
                                await self.display_api_token(coin, age_seconds)
                
        except Exception as e:
            pass  # Silent fail for API checks
    
    async def process_websocket_event(self, result):
        """Process WebSocket event for instant detection"""
        signature = result.get('signature')
        
        if signature and signature not in self.seen_signatures:
            self.seen_signatures.add(signature)
            self.token_count += 1
            
            # INSTANT display
            timestamp = datetime.now()
            elapsed = (timestamp - self.start_time).total_seconds()
            
            print(f"\n🚀 INSTANT #{self.token_count} | {timestamp.strftime('%H:%M:%S.%f')[:-3]} | +{elapsed:.1f}s")
            print(f"📜 {signature[:16]}...{signature[-16:]}")
            
            # Quick transaction parsing for mint
            asyncio.create_task(self.extract_mint_lightning_fast(signature))
    
    async def extract_mint_lightning_fast(self, signature):
        """Lightning-fast mint extraction"""
        try:
            # Try to get transaction with minimal data
            request_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    signature,
                    {
                        "encoding": "json",
                        "maxSupportedTransactionVersion": 0,
                        "commitment": "processed"
                    }
                ]
            }
            
            async with self.session.post(self.http_url, json=request_data) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'result' in data and data['result']:
                        mint = self.find_mint_super_fast(data['result'])
                        
                        if mint:
                            self.seen_tokens.add(mint)
                            print(f"⚡ MINT: {mint}")
                            print(f"🔗 https://pump.fun/{mint}")
                            print(f"🔗 https://dexscreener.com/solana/{mint}")
                            print("─" * 50)
                        
        except Exception:
            pass  # Silent fail for speed
    
    def find_mint_super_fast(self, transaction) -> str:
        """Super fast mint finding"""
        try:
            # Method 1: postTokenBalances (most reliable)
            if 'meta' in transaction and 'postTokenBalances' in transaction['meta']:
                for balance in transaction['meta']['postTokenBalances']:
                    mint = balance.get('mint')
                    if mint and len(mint) >= 32 and mint != 'So11111111111111111111111111111111111111112':
                        return mint
            
            # Method 2: New account creation
            if 'meta' in transaction:
                pre_balances = transaction['meta'].get('preTokenBalances', [])
                post_balances = transaction['meta'].get('postTokenBalances', [])
                
                pre_mints = {b['mint'] for b in pre_balances if 'mint' in b}
                post_mints = {b['mint'] for b in post_balances if 'mint' in b}
                
                new_mints = post_mints - pre_mints
                if new_mints:
                    return list(new_mints)[0]
            
        except:
            pass
        
        return None
    
    async def display_api_token(self, coin, age_seconds):
        """Display token found via API"""
        mint = coin.get("mint")
        name = coin.get("name", "Unknown")
        symbol = coin.get("symbol", "???")
        market_cap = coin.get("usd_market_cap", 0)
        
        if mint not in self.seen_tokens:
            return
        
        age_minutes = age_seconds / 60
        
        print(f"\n🌐 API DETECTION | Age: {age_minutes:.1f}min")
        print(f"📛 {name} ({symbol})")
        print(f"💰 Market Cap: ${market_cap:,.0f}")
        print(f"⚡ MINT: {mint}")
        print(f"🔗 https://pump.fun/{mint}")
        print("─" * 50)
    
    async def print_speed_stats(self):
        """Print real-time speed statistics"""
        while self.running:
            await asyncio.sleep(10)  # Every 10 seconds
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            ws_rate = self.token_count / elapsed if elapsed > 0 else 0
            
            print(f"\n📊 SPEED | Runtime: {elapsed:.0f}s | WS Events: {self.token_count} | Rate: {ws_rate:.2f}/sec")
            print(f"💾 Tracked: {len(self.seen_tokens)} unique tokens")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Stopping lightning monitor...")
    sys.exit(0)

async def main():
    """Main function"""
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    monitor = LightningPumpMonitor()
    
    try:
        await monitor.start_lightning_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Lightning monitor stopped")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        monitor.running = False

if __name__ == "__main__":
    asyncio.run(main()) 