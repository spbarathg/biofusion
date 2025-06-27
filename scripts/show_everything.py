#!/usr/bin/env python3
"""
SHOW EVERYTHING MONITOR
======================

Shows EVERY SINGLE Pump.fun event with full details - no filtering whatsoever
"""

import asyncio
import websockets
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class ShowEverythingMonitor:
    """Monitor that shows absolutely everything"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.websocket_url = f"wss://mainnet.helius-rpc.com/?api-key={helius_api_key}"
        self.pump_fun_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        self.websocket = None
        self.subscription_id = None
        self.event_count = 0
        self.start_time = datetime.now()
        
    async def connect_websocket(self):
        """Connect to Helius Geyser WebSocket"""
        
        print("🔗 Connecting...")
        
        try:
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            print("✅ Connected!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def subscribe_to_events(self):
        """Subscribe to Pump.fun activity"""
        
        subscription_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {
                    "mentions": [self.pump_fun_program_id]
                },
                {
                    "commitment": "processed"
                }
            ]
        }
        
        print("📡 Subscribing...")
        
        try:
            await self.websocket.send(json.dumps(subscription_request))
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if "result" in response_data:
                self.subscription_id = response_data["result"]
                print(f"✅ Subscribed! ID: {self.subscription_id}")
                return True
            else:
                print(f"❌ Failed: {response_data}")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def extract_all_addresses(self, logs):
        """Extract all potential addresses from logs"""
        
        addresses = []
        for log in logs:
            words = log.split()
            for word in words:
                # Look for 44-character base58 addresses or 32-character addresses
                if (len(word) == 44 or len(word) == 43 or len(word) == 32) and word.replace('_', '').replace('-', '').isalnum():
                    if word not in addresses:
                        addresses.append(word)
        
        return addresses
    
    def display_event(self, log_data):
        """Display EVERYTHING about this event"""
        
        self.event_count += 1
        
        # Extract ALL data
        logs = log_data.get("logs", [])
        signature = log_data.get("signature", "N/A")
        context = log_data.get("context", {})
        slot = context.get("slot", "N/A")
        
        # Check for special keywords
        create_found = any("create" in log.lower() for log in logs)
        initialize_found = any("initialize" in log.lower() for log in logs)
        invoke_found = any("invoke" in log for log in logs)
        program_found = any("Program 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P" in log for log in logs)
        
        # Special highlighting for potential launches
        if create_found or initialize_found:
            print(f"\n🚀🚀🚀 EVENT #{self.event_count} - POTENTIAL TOKEN ACTIVITY 🚀🚀🚀")
            print("=" * 80)
        else:
            print(f"\n⚡ EVENT #{self.event_count}")
            print("-" * 50)
        
        # Timestamp
        now = datetime.now()
        time_str = now.strftime('%H:%M:%S.%f')[:-3]
        print(f"⏰ TIME: {time_str}")
        
        # Transaction details
        print(f"📍 SIGNATURE: {signature}")
        print(f"🎯 SLOT: {slot}")
        
        # Event indicators
        indicators = []
        if create_found:
            indicators.append("CREATE")
        if initialize_found:
            indicators.append("INITIALIZE")
        if invoke_found:
            indicators.append("INVOKE")
        if program_found:
            indicators.append("PUMP.FUN")
        
        if indicators:
            print(f"🔥 INDICATORS: {', '.join(indicators)}")
        
        # Extract ALL addresses
        addresses = self.extract_all_addresses(logs)
        if addresses:
            print(f"🪙 ADDRESSES FOUND ({len(addresses)}):")
            for i, addr in enumerate(addresses):
                print(f"   {i+1:2d}. {addr}")
                if create_found or initialize_found:
                    print(f"       🔗 PUMP: https://pump.fun/{addr}")
                    print(f"       🔗 SCAN: https://solscan.io/token/{addr}")
        
        # Show ALL logs
        print(f"📋 ALL LOGS ({len(logs)} total):")
        for i, log in enumerate(logs):
            # Highlight important logs
            if any(keyword in log.lower() for keyword in ['create', 'initialize', 'program']):
                print(f"   🔥 {i+1:2d}. {log}")
            else:
                print(f"      {i+1:2d}. {log}")
        
        # Full context data
        if context and len(context) > 1:  # More than just slot
            print(f"📊 CONTEXT: {json.dumps(context, indent=2)}")
        
        # Runtime stats
        runtime = (now - self.start_time).total_seconds()
        events_per_sec = self.event_count / runtime if runtime > 0 else 0
        print(f"📈 STATS: Event #{self.event_count} | {events_per_sec:.1f} events/sec | Runtime: {runtime:.1f}s")
        
        if create_found or initialize_found:
            print("=" * 80)
        else:
            print("-" * 50)
    
    async def process_message(self, message):
        """Process incoming messages"""
        
        try:
            data = json.loads(message)
            
            # Process log notifications
            if "method" in data and data["method"] == "logsNotification":
                params = data.get("params", {})
                result = params.get("result", {})
                
                if result:  # Only process if we have data
                    self.display_event(result)
            
        except json.JSONDecodeError:
            pass  # Ignore malformed messages
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
    
    async def monitor_everything(self):
        """Main monitoring function"""
        
        print("🔥 SHOW EVERYTHING MONITOR")
        print("=" * 50)
        print("📡 Shows EVERY SINGLE Pump.fun event")
        print("🔍 Full details, no filtering")
        print("🚀 Real-time complete feed")
        print("")
        
        if not await self.connect_websocket():
            return
        
        if not await self.subscribe_to_events():
            return
        
        print(f"🔥 EVERYTHING MONITOR ACTIVE - {datetime.now().strftime('%H:%M:%S')}")
        print("📊 Showing ALL events with full details...")
        print("⚡ Press Ctrl+C to stop")
        print("")
        
        try:
            async for message in self.websocket:
                await self.process_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 WebSocket closed")
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitor stopped")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            print(f"📊 Total events: {self.event_count}")
            print(f"⏰ Total runtime: {(datetime.now() - self.start_time).total_seconds():.1f}s")
            
            if self.websocket:
                await self.websocket.close()

async def main():
    """Main function"""
    
    helius_api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
    
    monitor = ShowEverythingMonitor(helius_api_key)
    await monitor.monitor_everything()

if __name__ == "__main__":
    print("🔥 Starting Show Everything Monitor!")
    asyncio.run(main()) 