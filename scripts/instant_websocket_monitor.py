#!/usr/bin/env python3
"""
🚀 INSTANT WEBSOCKET MONITOR 🚀
===============================

Pure WebSocket monitoring with maximum debug info to show exactly what's happening.
Shows ALL Solana activity, not just pump.fun, so you can see the system working.
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

class InstantWebSocketMonitor:
    def __init__(self, api_key="193ececa-6e42-4d84-b9bd-765c4813816d"):
        self.api_key = api_key
        self.ws_url = f"wss://mainnet.helius-rpc.com/?api-key={api_key}"
        self.http_url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        
        # Program IDs to monitor
        self.pump_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"  # Pump.fun
        self.token_program_id = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"  # SPL Token
        self.raydium_program_id = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"  # Raydium
        
        # Stats
        self.start_time = datetime.now()
        self.total_events = 0
        self.pump_events = 0
        self.token_events = 0
        self.seen_signatures = set()
        
        # Session for HTTP requests
        self.session = None
    
    async def start_monitoring(self):
        """Start instant WebSocket monitoring"""
        print("🚀" * 70)
        print("⚡ INSTANT WEBSOCKET MONITOR - LIVE SOLANA ACTIVITY ⚡")
        print("🚀" * 70)
        print("📡 Monitoring ALL Solana activity for instant feedback")
        print("🎯 Special focus on Pump.fun, Token Program, Raydium")
        print("⚡ Shows EVERYTHING so you see the system working!")
        print("🚀" * 70)
        
        # Setup session
        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=25,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=3)
        )
        
        try:
            # Start monitoring tasks
            tasks = [
                asyncio.create_task(self.monitor_pump_fun()),
                asyncio.create_task(self.monitor_general_activity()),
                asyncio.create_task(self.print_stats())
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            if self.session:
                await self.session.close()
    
    async def monitor_pump_fun(self):
        """Monitor pump.fun specific activity"""
        retry_count = 0
        
        while retry_count < 3:
            try:
                print(f"🔌 Connecting to Pump.fun monitoring...")
                
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10
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
                    print("✅ Subscribed to Pump.fun program logs")
                    
                    retry_count = 0
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self.process_pump_event(data)
                        except Exception as e:
                            continue
                            
            except Exception as e:
                retry_count += 1
                print(f"❌ Pump.fun WebSocket error: {e}")
                if retry_count < 3:
                    await asyncio.sleep(retry_count * 2)
    
    async def monitor_general_activity(self):
        """Monitor general Solana activity to show system is working"""
        retry_count = 0
        
        while retry_count < 3:
            try:
                print(f"🔌 Connecting to general Solana monitoring...")
                
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10
                ) as websocket:
                    
                    # Subscribe to token program (more active)
                    subscribe = {
                        "jsonrpc": "2.0", 
                        "id": 2,
                        "method": "logsSubscribe",
                        "params": [
                            {"mentions": [self.token_program_id]},
                            {"commitment": "processed"}
                        ]
                    }
                    
                    await websocket.send(json.dumps(subscribe))
                    print("✅ Subscribed to Token Program logs (to show activity)")
                    
                    retry_count = 0
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self.process_general_event(data)
                        except Exception as e:
                            continue
                            
            except Exception as e:
                retry_count += 1
                print(f"❌ General WebSocket error: {e}")
                if retry_count < 3:
                    await asyncio.sleep(retry_count * 2)
    
    async def process_pump_event(self, data):
        """Process pump.fun specific events"""
        if 'params' in data and 'result' in data['params']:
            result = data['params']['result']
            signature = result.get('signature')
            
            if signature and signature not in self.seen_signatures:
                self.seen_signatures.add(signature)
                self.pump_events += 1
                self.total_events += 1
                
                timestamp = datetime.now()
                elapsed = (timestamp - self.start_time).total_seconds()
                
                print(f"\n🚀 PUMP.FUN EVENT #{self.pump_events} | {timestamp.strftime('%H:%M:%S.%f')[:-3]}")
                print(f"📜 Signature: {signature[:16]}...{signature[-16:]}")
                print(f"🕐 Time: +{elapsed:.1f}s from start")
                
                # Show logs for analysis
                logs = result.get('logs', [])
                if logs:
                    print(f"📋 Logs ({len(logs)} entries):")
                    for i, log in enumerate(logs[:3]):  # Show first 3 logs
                        print(f"   {i+1}. {log[:100]}...")
                
                # Try to extract mint quickly
                asyncio.create_task(self.extract_mint_info(signature))
                print("─" * 50)
    
    async def process_general_event(self, data):
        """Process general Solana events to show activity"""
        if 'params' in data and 'result' in data['params']:
            result = data['params']['result']
            signature = result.get('signature')
            
            if signature and signature not in self.seen_signatures:
                self.seen_signatures.add(signature)
                self.token_events += 1
                self.total_events += 1
                
                # Only show every 10th general event to avoid spam
                if self.token_events % 10 == 0:
                    timestamp = datetime.now()
                    print(f"📊 General Activity: {self.token_events} token events | {timestamp.strftime('%H:%M:%S')}")
    
    async def extract_mint_info(self, signature):
        """Extract mint info from transaction"""
        try:
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
                        mint = self.find_mint_address(data['result'])
                        
                        if mint:
                            print(f"⚡ MINT EXTRACTED: {mint}")
                            print(f"🔗 https://pump.fun/{mint}")
                            print(f"🔗 https://dexscreener.com/solana/{mint}")
                        else:
                            print(f"🔍 No mint found in transaction")
                        
        except Exception as e:
            print(f"🔍 Mint extraction failed: {str(e)[:50]}")
    
    def find_mint_address(self, transaction):
        """Find mint address in transaction"""
        try:
            # Check postTokenBalances
            if 'meta' in transaction and 'postTokenBalances' in transaction['meta']:
                for balance in transaction['meta']['postTokenBalances']:
                    mint = balance.get('mint')
                    if mint and len(mint) >= 32 and mint != 'So11111111111111111111111111111111111111112':
                        return mint
            
            # Check for new mints
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
    
    async def print_stats(self):
        """Print live statistics"""
        while True:
            await asyncio.sleep(15)  # Every 15 seconds
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            print(f"\n📊 LIVE STATS | Runtime: {elapsed:.0f}s")
            print(f"   🚀 Pump.fun events: {self.pump_events}")
            print(f"   📊 Token events: {self.token_events}")
            print(f"   🔥 Total events: {self.total_events}")
            print(f"   ⚡ Rate: {self.total_events/elapsed:.2f} events/sec")
            print("─" * 30)

async def main():
    """Main function"""
    monitor = InstantWebSocketMonitor()
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 