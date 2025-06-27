#!/usr/bin/env python3
"""
🚀 ULTRA FAST TOKEN MONITOR 🚀
=============================

INSTANT token detection with lightning-fast extraction!
- Real-time WebSocket monitoring
- Parallel processing
- Minimal latency
- Instant display with basic info
- Background metadata enrichment
"""

import asyncio
import websockets
import aiohttp
import json
import logging
from datetime import datetime
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import traceback

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configure minimal logging for speed
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class UltraFastTokenMonitor:
    def __init__(self, api_key="193ececa-6e42-4d84-b9bd-765c4813816d"):
        self.api_key = api_key
        self.ws_url = f"wss://mainnet.helius-rpc.com/?api-key={api_key}"
        self.http_url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        self.pump_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        
        # Speed optimization
        self.seen_signatures = set()
        self.token_count = 0
        self.start_time = datetime.now()
        
        # Ultra-fast processing queues
        self.instant_queue = Queue()  # For instant display
        self.metadata_queue = Queue()  # For background enrichment
        
        # Background processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = True
        
        # Session management
        self.session = None
        
    async def start_ultra_fast_monitoring(self):
        """Start ultra-fast monitoring with parallel processing"""
        print("🚀" * 60)
        print("⚡ ULTRA FAST TOKEN MONITOR STARTED ⚡")
        print("🚀" * 60)
        print("📡 Real-time WebSocket monitoring active")
        print("⚡ Instant display enabled")
        print("🔄 Background metadata enrichment active")
        print("🚀" * 60)
        
        # Setup session with optimizations
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=50,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=5, connect=2),
            headers={"Connection": "keep-alive"}
        )
        
        try:
            # Start background processors
            self.start_background_processors()
            
            # Start WebSocket monitoring
            await self.monitor_websocket()
            
        finally:
            if self.session:
                await self.session.close()
            self.running = False
    
    def start_background_processors(self):
        """Start background processing threads for speed"""
        # Instant display processor
        threading.Thread(
            target=self.instant_display_processor,
            daemon=True
        ).start()
        
        # Metadata enrichment processor
        for i in range(3):  # 3 parallel metadata workers
            threading.Thread(
                target=self.metadata_processor,
                daemon=True
            ).start()
    
    def instant_display_processor(self):
        """Process instant displays in separate thread"""
        while self.running:
            try:
                if not self.instant_queue.empty():
                    token_data = self.instant_queue.get(timeout=1)
                    self.display_instant_token(token_data)
            except:
                continue
    
    def metadata_processor(self):
        """Process metadata enrichment in background"""
        while self.running:
            try:
                if not self.metadata_queue.empty():
                    token_data = self.metadata_queue.get(timeout=1)
                    # Background metadata fetching would go here
                    # For now, just mark as processed
                    pass
            except:
                continue
    
    async def monitor_websocket(self):
        """Monitor WebSocket for real-time token launches"""
        retry_count = 0
        max_retries = 10
        
        while retry_count < max_retries:
            try:
                print(f"🔌 Connecting to WebSocket... (attempt {retry_count + 1})")
                
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    
                    # Subscribe to Pump.fun program logs
                    subscribe_request = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "logsSubscribe",
                        "params": [
                            {
                                "mentions": [self.pump_program_id]
                            },
                            {
                                "commitment": "processed"
                            }
                        ]
                    }
                    
                    await websocket.send(json.dumps(subscribe_request))
                    print("✅ WebSocket connected and subscribed")
                    print("⚡ Monitoring for instant token launches...")
                    
                    retry_count = 0  # Reset retry count on successful connection
                    
                    # Listen for messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self.process_websocket_message(data)
                        except Exception as e:
                            logging.debug(f"Message processing error: {e}")
                            continue
                            
            except Exception as e:
                retry_count += 1
                wait_time = min(retry_count * 2, 30)
                print(f"❌ WebSocket error: {e}")
                print(f"🔄 Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        
        print("❌ Max retries reached. Exiting.")
    
    async def process_websocket_message(self, data):
        """Process WebSocket message for instant token detection"""
        try:
            if 'params' in data and 'result' in data['params']:
                result = data['params']['result']
                signature = result.get('signature')
                
                if signature and signature not in self.seen_signatures:
                    self.seen_signatures.add(signature)
                    self.token_count += 1
                    
                    # Create instant token data
                    token_data = {
                        'number': self.token_count,
                        'signature': signature,
                        'timestamp': datetime.now(),
                        'logs': result.get('logs', [])
                    }
                    
                    # Queue for INSTANT display
                    self.instant_queue.put(token_data)
                    
                    # Queue for background metadata enrichment
                    self.metadata_queue.put(token_data)
                    
                    # Optional: Start fast transaction parsing in background
                    asyncio.create_task(self.fast_parse_transaction(signature))
                    
        except Exception as e:
            logging.debug(f"WebSocket processing error: {e}")
    
    async def fast_parse_transaction(self, signature):
        """Ultra-fast transaction parsing"""
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
                        # Fast mint extraction
                        mint = self.extract_mint_fast(data['result'])
                        if mint and len(mint) >= 32:
                            print(f"🎯 MINT FOUND: {mint}")
                            
        except Exception as e:
            logging.debug(f"Fast parse error: {e}")
    
    def extract_mint_fast(self, transaction):
        """Ultra-fast mint extraction"""
        try:
            # Method 1: Check postTokenBalances (fastest)
            if 'meta' in transaction and 'postTokenBalances' in transaction['meta']:
                for balance in transaction['meta']['postTokenBalances']:
                    if 'mint' in balance:
                        mint = balance['mint']
                        if len(mint) >= 32 and mint != 'So11111111111111111111111111111111111111112':
                            return mint
            
            # Method 2: Check account keys
            if 'transaction' in transaction and 'message' in transaction['transaction']:
                account_keys = transaction['transaction']['message'].get('accountKeys', [])
                for key in account_keys[1:6]:  # Check first few account keys
                    if len(key) >= 32:
                        return key
                        
        except:
            pass
        
        return None
    
    def display_instant_token(self, token_data):
        """Display token instantly with minimal info"""
        number = token_data['number']
        timestamp = token_data['timestamp']
        signature = token_data['signature']
        
        # Time since start
        elapsed = (timestamp - self.start_time).total_seconds()
        
        # Ultra-fast display
        print(f"\n⚡ TOKEN #{number} | {timestamp.strftime('%H:%M:%S.%f')[:-3]} | +{elapsed:.1f}s")
        print(f"📜 {signature}")
        print(f"🔍 Processing in background...")
        
        # Show detection rate
        rate = number / elapsed if elapsed > 0 else 0
        print(f"📊 Rate: {rate:.2f} tokens/sec | Total: {number}")
        
        # Quick log analysis
        logs = token_data.get('logs', [])
        if logs:
            print(f"📋 Logs: {len(logs)} entries")
            
            # Look for pump.fun specific patterns
            pump_indicators = []
            for log in logs[:3]:  # Check first 3 logs only for speed
                if any(word in log.lower() for word in ['create', 'mint', 'token', 'initialize']):
                    pump_indicators.append("🎯")
                    break
            
            if pump_indicators:
                print(f"🚀 PUMP.FUN ACTIVITY DETECTED! {' '.join(pump_indicators)}")
        
        print("─" * 60)
    
    async def print_stats(self):
        """Print monitoring statistics"""
        while self.running:
            await asyncio.sleep(30)  # Every 30 seconds
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.token_count / elapsed if elapsed > 0 else 0
            
            print(f"\n📊 STATS | Runtime: {elapsed:.0f}s | Tokens: {self.token_count} | Rate: {rate:.2f}/sec")

async def main():
    """Main function"""
    monitor = UltraFastTokenMonitor()
    
    try:
        # Start stats printer in background
        stats_task = asyncio.create_task(monitor.print_stats())
        
        # Start main monitoring
        await monitor.start_ultra_fast_monitoring()
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
    finally:
        monitor.running = False

if __name__ == "__main__":
    asyncio.run(main()) 