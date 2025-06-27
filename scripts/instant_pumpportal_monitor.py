#!/usr/bin/env python3
"""
INSTANT PUMPPORTAL MONITOR
=========================

Ultra-fast pump.fun token detection using PumpPortal WebSocket API.
Optimized for Windows, instant display, super quick extraction.
"""

import asyncio
import websockets
import json
from datetime import datetime
import sys
import time

# Fix Windows asyncio and encoding issues
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

class InstantPumpPortalMonitor:
    def __init__(self):
        self.websocket_url = "wss://pumpportal.fun/api/data"
        self.total_tokens = 0
        self.start_time = datetime.now()
        self.running = True
        
    async def start_instant_monitoring(self):
        """Start ultra-fast PumpPortal monitoring"""
        print("=" * 70)
        print("INSTANT PUMPPORTAL MONITOR - MAXIMUM SPEED")
        print("=" * 70)
        print("Real-time pump.fun token detection via PumpPortal API")
        print("Instant display with immediate mint extraction")
        print("=" * 70)
        
        retry_count = 0
        max_retries = 5
        
        while self.running and retry_count < max_retries:
            try:
                print(f"Connecting to PumpPortal... (attempt {retry_count + 1})")
                
                async with websockets.connect(
                    self.websocket_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=1024*1024
                ) as websocket:
                    
                    print("SUCCESS: Connected to PumpPortal!")
                    retry_count = 0
                    
                    # Subscribe to new tokens
                    subscribe_payload = {"method": "subscribeNewToken"}
                    await websocket.send(json.dumps(subscribe_payload))
                    print("SUCCESS: Subscribed to new token events")
                    
                    # Subscribe to migrations (tokens graduating to Raydium)
                    migration_payload = {"method": "subscribeMigration"}
                    await websocket.send(json.dumps(migration_payload))
                    print("SUCCESS: Subscribed to migration events")
                    
                    print("LISTENING: Monitoring for instant token launches...")
                    print("-" * 70)
                    
                    # Start stats printer
                    stats_task = asyncio.create_task(self.print_stats())
                    
                    # Listen for messages
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            await self.process_instant_token(data)
                            
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            print(f"ERROR processing message: {str(e)[:50]}")
                            
            except Exception as e:
                retry_count += 1
                print(f"ERROR: Connection failed - {str(e)[:100]}")
                if retry_count < max_retries:
                    print(f"RETRY: Reconnecting in {retry_count * 2} seconds...")
                    await asyncio.sleep(retry_count * 2)
        
        print("STOPPED: Max retries reached or stopped by user")
    
    async def process_instant_token(self, data):
        """Process token with INSTANT display"""
        try:
            # Handle different data structures
            mint = None
            name = "Unknown"
            symbol = "???"
            creator = "Unknown"
            
            if isinstance(data, dict):
                # Extract mint address
                mint = data.get('mint') or data.get('mintAddress') or data.get('address')
                
                if mint:
                    self.total_tokens += 1
                    timestamp = datetime.now()
                    elapsed = (timestamp - self.start_time).total_seconds()
                    
                    # Extract token info
                    name = data.get('name', 'Unknown')
                    symbol = data.get('symbol', '???')
                    creator = data.get('tradeCreator') or data.get('creator') or data.get('dev', 'Unknown')
                    uri = data.get('uri', '')
                    
                    # INSTANT display
                    print(f"\nTOKEN #{self.total_tokens} | {timestamp.strftime('%H:%M:%S.%f')[:-3]} | +{elapsed:.1f}s")
                    print(f"NAME: {name}")
                    print(f"SYMBOL: {symbol}")
                    print(f"MINT: {mint}")
                    print(f"CREATOR: {creator[:16]}...{creator[-8:] if len(creator) > 24 else creator}")
                    
                    # Instant links
                    print(f"PUMP: https://pump.fun/{mint}")
                    print(f"DEX: https://dexscreener.com/solana/{mint}")
                    
                    # Quick analysis
                    if len(name) > 20:
                        print("NOTE: Long name (potential spam)")
                    if symbol.lower() in ['test', 'scam', 'rug']:
                        print("WARNING: Suspicious symbol")
                    
                    print("-" * 70)
                    
                    # Save to file
                    self.save_instant_token(timestamp, mint, name, symbol, creator)
                
                # Handle migration events
                elif data.get('type') == 'migration':
                    mint = data.get('mint', 'Unknown')
                    print(f"\nMIGRATION: Token {mint[:16]}... graduated to Raydium!")
                    
        except Exception as e:
            print(f"ERROR processing token: {str(e)[:50]}")
    
    def save_instant_token(self, timestamp, mint, name, symbol, creator):
        """Save token data instantly"""
        try:
            log_entry = f"{timestamp.isoformat()},{mint},{name},{symbol},{creator}\n"
            
            with open('../logs/instant_tokens.txt', 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as e:
            print(f"ERROR saving: {str(e)[:30]}")
    
    async def print_stats(self):
        """Print live statistics"""
        while self.running:
            await asyncio.sleep(30)  # Every 30 seconds
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.total_tokens / elapsed if elapsed > 0 else 0
            
            print(f"\nSTATS | Runtime: {elapsed:.0f}s | Tokens: {self.total_tokens} | Rate: {rate:.2f}/sec")

async def main():
    """Main function"""
    monitor = InstantPumpPortalMonitor()
    
    try:
        await monitor.start_instant_monitoring()
    except KeyboardInterrupt:
        print("\nSTOPPED by user")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
    finally:
        monitor.running = False

if __name__ == "__main__":
    print("Starting Instant PumpPortal Monitor...")
    asyncio.run(main()) 