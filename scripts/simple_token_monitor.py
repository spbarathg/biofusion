#!/usr/bin/env python3
"""
SIMPLE TOKEN MONITOR
===================

Simple monitor that shows ALL Pump.fun events with proper log details
"""

import asyncio
import websockets
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class SimpleTokenMonitor:
    """Simple monitor to see all Pump.fun activity"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.websocket_url = f"wss://mainnet.helius-rpc.com/?api-key={helius_api_key}"
        self.pump_fun_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        self.websocket = None
        self.subscription_id = None
        self.event_count = 0
        self.launch_count = 0
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
    
    def check_for_launch(self, logs):
        """Check if this might be a token launch"""
        
        # Look for create/initialize patterns
        launch_patterns = [
            "Program log: create",
            "Program log: Create", 
            "Program log: initialize",
            "Program log: Initialize",
            "invoke [1]",
            "Program 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P invoke [1]"
        ]
        
        for log in logs:
            for pattern in launch_patterns:
                if pattern in log:
                    return True
        
        return False
    
    def extract_addresses_from_logs(self, logs):
        """Extract potential token addresses from logs"""
        
        addresses = []
        for log in logs:
            words = log.split()
            for word in words:
                # Look for 44-character base58 addresses
                if len(word) == 44 and word.replace('_', '').replace('-', '').isalnum():
                    if word not in addresses:
                        addresses.append(word)
        
        return addresses
    
    def display_event(self, log_data):
        """Display event information"""
        
        self.event_count += 1
        
        # Extract data
        logs = log_data.get("logs", [])
        signature = log_data.get("signature", "N/A")
        slot = log_data.get("context", {}).get("slot", "N/A")
        
        # Check if this might be a launch
        is_potential_launch = self.check_for_launch(logs)
        
        if is_potential_launch:
            self.launch_count += 1
            print(f"\n🚀🚀🚀 POTENTIAL LAUNCH #{self.launch_count} 🚀🚀🚀")
            print("=" * 60)
        else:
            # Show every 10th regular event to reduce spam
            if self.event_count % 10 == 0:
                print(f"\n⚡ Event #{self.event_count}")
                print("-" * 30)
            else:
                return  # Skip showing this event
        
        # Time
        now = datetime.now()
        time_str = now.strftime('%H:%M:%S.%f')[:-3]
        print(f"⏰ TIME: {time_str}")
        
        # Transaction details
        if signature != "N/A":
            print(f"📍 TX: {signature}")
            print(f"🎯 SLOT: {slot}")
        
        # Extract addresses
        addresses = self.extract_addresses_from_logs(logs)
        if addresses:
            print(f"🪙 ADDRESSES FOUND: {len(addresses)}")
            for i, addr in enumerate(addresses[:3]):  # Show first 3
                print(f"   {i+1}. {addr}")
                if is_potential_launch:
                    print(f"      🔗 PUMP: https://pump.fun/{addr}")
                    print(f"      🔗 SCAN: https://solscan.io/token/{addr}")
        
        # Show relevant logs
        print(f"📋 LOGS ({len(logs)} total):")
        relevant_logs = [log for log in logs if any(keyword in log.lower() for keyword in ['create', 'initialize', 'program', 'invoke'])]
        
        if relevant_logs:
            for i, log in enumerate(relevant_logs[:3]):  # Show first 3 relevant
                print(f"   {i+1}. {log}")
        else:
            # Show first 2 logs if no relevant ones found
            for i, log in enumerate(logs[:2]):
                print(f"   {i+1}. {log}")
        
        # Stats
        runtime = (now - self.start_time).total_seconds()
        print(f"📊 TOTAL: {self.event_count} | LAUNCHES: {self.launch_count} | TIME: {runtime:.1f}s")
        
        if is_potential_launch:
            print("=" * 60)
        else:
            print("-" * 30)
    
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
    
    async def monitor_tokens(self):
        """Main monitoring function"""
        
        print("🚀 SIMPLE TOKEN MONITOR")
        print("=" * 40)
        print("📡 Shows ALL Pump.fun activity")
        print("🚀 Highlights potential launches")
        print("🔍 Real-time event stream")
        print("")
        
        if not await self.connect_websocket():
            return
        
        if not await self.subscribe_to_events():
            return
        
        print(f"🚀 MONITOR ACTIVE - {datetime.now().strftime('%H:%M:%S')}")
        print("📊 Listening for events...")
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
            print(f"🚀 Potential launches: {self.launch_count}")
            print(f"⏰ Total runtime: {(datetime.now() - self.start_time).total_seconds():.1f}s")
            
            if self.websocket:
                await self.websocket.close()

async def main():
    """Main function"""
    
    helius_api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
    
    monitor = SimpleTokenMonitor(helius_api_key)
    await monitor.monitor_tokens()

if __name__ == "__main__":
    print("🚀 Starting Simple Token Monitor!")
    asyncio.run(main()) 