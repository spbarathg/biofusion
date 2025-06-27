#!/usr/bin/env python3
"""
DEBUG LAUNCH MONITOR
===================

Debug version to see what events we're actually receiving from Pump.fun
"""

import asyncio
import websockets
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class DebugLaunchMonitor:
    """Debug monitor to see all Pump.fun events"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.websocket_url = f"wss://mainnet.helius-rpc.com/?api-key={helius_api_key}"
        self.pump_fun_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        self.websocket = None
        self.subscription_id = None
        self.event_count = 0
        self.potential_launch_count = 0
        self.start_time = datetime.now()
        
    async def connect_websocket(self):
        """Connect to Helius Geyser WebSocket"""
        
        print("🔗 Connecting to Debug Monitor...")
        
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
        
        print("📡 Subscribing to all Pump.fun events...")
        
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
    
    def analyze_event(self, log_data):
        """Analyze event to see if it's a potential launch"""
        
        logs = log_data.get("logs", [])
        signature = log_data.get("signature", "Unknown")
        
        # Look for launch-like patterns
        launch_keywords = [
            "create", "initialize", "mint", "new", "token", 
            "curve", "bonding", "program", "invoke"
        ]
        
        keyword_matches = []
        for log in logs:
            for keyword in launch_keywords:
                if keyword.lower() in log.lower():
                    if keyword not in keyword_matches:
                        keyword_matches.append(keyword)
        
        # Extract addresses
        addresses = []
        for log in logs:
            words = log.split()
            for word in words:
                if len(word) == 44 and word.replace('_', '').replace('-', '').isalnum():
                    if word not in addresses:
                        addresses.append(word)
        
        # Check if this looks like a launch
        is_potential_launch = (
            len(keyword_matches) >= 2 or 
            "create" in keyword_matches or
            "initialize" in keyword_matches or
            len(addresses) >= 3
        )
        
        return {
            "signature": signature,
            "addresses": addresses,
            "keywords": keyword_matches,
            "logs": logs,
            "is_potential_launch": is_potential_launch,
            "timestamp": datetime.now()
        }
    
    def display_event(self, event_info):
        """Display event information"""
        
        self.event_count += 1
        
        if event_info["is_potential_launch"]:
            self.potential_launch_count += 1
            print(f"\n🚀 POTENTIAL LAUNCH #{self.potential_launch_count} (Event #{self.event_count})")
            print("=" * 70)
        else:
            print(f"\n⚡ Event #{self.event_count}")
            print("-" * 30)
        
        # Time
        time_str = event_info["timestamp"].strftime('%H:%M:%S.%f')[:-3]
        print(f"⏰ TIME: {time_str}")
        
        # Keywords found
        if event_info["keywords"]:
            print(f"🔑 KEYWORDS: {', '.join(event_info['keywords'])}")
        
        # Addresses found
        if event_info["addresses"]:
            print(f"🪙 ADDRESSES ({len(event_info['addresses'])}):")
            for i, addr in enumerate(event_info["addresses"][:3]):  # Show first 3
                print(f"   {i+1}. {addr}")
                if event_info["is_potential_launch"]:
                    print(f"      🔗 PUMP: https://pump.fun/{addr}")
        
        # Transaction signature
        print(f"📍 TX: {event_info['signature']}")
        
        # Sample of logs (first 2)
        if event_info["logs"]:
            print(f"📋 SAMPLE LOGS ({len(event_info['logs'])} total):")
            for i, log in enumerate(event_info["logs"][:2]):
                print(f"   {i+1}. {log}")
        
        # Stats
        runtime = (event_info["timestamp"] - self.start_time).total_seconds()
        print(f"📊 EVENTS: {self.event_count} | POTENTIAL LAUNCHES: {self.potential_launch_count} | TIME: {runtime:.1f}s")
        
        if event_info["is_potential_launch"]:
            print("=" * 70)
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
                    event_info = self.analyze_event(result)
                    self.display_event(event_info)
            
        except json.JSONDecodeError:
            pass  # Ignore malformed messages
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
    
    async def monitor_debug(self):
        """Debug monitor"""
        
        print("🔍 DEBUG LAUNCH MONITOR")
        print("=" * 50)
        print("🎯 Analyzing ALL Pump.fun events")
        print("🚀 Identifying potential launches")
        print("📊 Real-time event analysis")
        print("")
        
        if not await self.connect_websocket():
            return
        
        if not await self.subscribe_to_events():
            return
        
        print(f"🔍 DEBUG MONITOR ACTIVE - {datetime.now().strftime('%H:%M:%S')}")
        print("📡 Analyzing all events...")
        print("⚡ Press Ctrl+C to stop")
        print("")
        
        try:
            async for message in self.websocket:
                await self.process_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 WebSocket closed")
        except KeyboardInterrupt:
            print(f"\n⏹️ Debug monitor stopped")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            print(f"📊 Total events: {self.event_count}")
            print(f"🚀 Potential launches: {self.potential_launch_count}")
            print(f"⏰ Total runtime: {(datetime.now() - self.start_time).total_seconds():.1f}s")
            
            if self.websocket:
                await self.websocket.close()

async def main():
    """Main function"""
    
    helius_api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
    
    monitor = DebugLaunchMonitor(helius_api_key)
    await monitor.monitor_debug()

if __name__ == "__main__":
    print("🔍 Starting Debug Launch Monitor!")
    asyncio.run(main()) 