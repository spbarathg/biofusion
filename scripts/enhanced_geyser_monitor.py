#!/usr/bin/env python3
"""
ENHANCED HELIUS GEYSER MONITOR
=============================

Enhanced real-time monitor that shows ALL Pump.fun activity to help identify launch patterns.
"""

import asyncio
import websockets
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class EnhancedGeyserMonitor:
    """Enhanced monitor showing all Pump.fun activity"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.websocket_url = f"wss://mainnet.helius-rpc.com/?api-key={helius_api_key}"
        self.pump_fun_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        self.websocket = None
        self.subscription_id = None
        self.activity_count = 0
        self.launch_count = 0
        self.start_time = datetime.now()
        
    async def connect_websocket(self):
        """Connect to Helius Geyser WebSocket"""
        
        print("🔗 Connecting to Enhanced Geyser Monitor...")
        
        try:
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            print("✅ WebSocket connected!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def subscribe_to_pump_fun_logs(self):
        """Subscribe to all Pump.fun program logs"""
        
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
        
        print("📡 Subscribing to ALL Pump.fun activity...")
        
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
    
    def analyze_pump_fun_logs(self, log_data):
        """Analyze all Pump.fun logs to understand activity patterns"""
        
        logs = log_data.get("logs", [])
        signature = log_data.get("signature", "Unknown")
        
        # Check for different instruction types
        instruction_types = []
        token_addresses = []
        
        for log in logs:
            # Look for instruction patterns
            if "Instruction:" in log:
                parts = log.split("Instruction:")
                if len(parts) > 1:
                    instruction = parts[1].strip().split()[0]
                    instruction_types.append(instruction)
            
            # Look for addresses (44 character strings)
            words = log.split()
            for word in words:
                if len(word) == 44 and word.replace('_', '').replace('-', '').isalnum():
                    if word not in token_addresses:
                        token_addresses.append(word)
        
        return {
            "signature": signature,
            "instruction_types": instruction_types,
            "token_addresses": token_addresses,
            "logs": logs,
            "log_count": len(logs)
        }
    
    def is_token_creation(self, analysis):
        """Check if this looks like a token creation"""
        
        creation_keywords = ["Create", "create", "mint", "initialize", "init"]
        instruction_types = analysis.get("instruction_types", [])
        logs = analysis.get("logs", [])
        
        # Check instructions
        for instruction in instruction_types:
            if any(keyword.lower() in instruction.lower() for keyword in creation_keywords):
                return True
        
        # Check log content
        for log in logs:
            if "Create" in log and ("token" in log.lower() or "mint" in log.lower()):
                return True
        
        return False
    
    async def display_pump_fun_activity(self, analysis):
        """Display Pump.fun activity with detailed analysis"""
        
        self.activity_count += 1
        current_time = datetime.now()
        
        is_creation = self.is_token_creation(analysis)
        
        if is_creation:
            self.launch_count += 1
            print(f"\n" + "🚀" * 80)
            print(f"🚀 TOKEN LAUNCH DETECTED #{self.launch_count} - {current_time.strftime('%H:%M:%S.%f')[:-3]} 🚀")
            print(f"🚀" * 80)
        else:
            print(f"\n" + "💫" * 60)
            print(f"💫 PUMP.FUN ACTIVITY #{self.activity_count} - {current_time.strftime('%H:%M:%S.%f')[:-3]} 💫")
            print(f"💫" * 60)
        
        print(f"📍 SIGNATURE: {analysis['signature']}")
        print(f"📋 LOG COUNT: {analysis['log_count']}")
        
        if analysis['instruction_types']:
            print(f"🎯 INSTRUCTIONS: {', '.join(analysis['instruction_types'])}")
        
        if analysis['token_addresses']:
            print(f"🪙 ADDRESSES: {len(analysis['token_addresses'])} found")
            for i, addr in enumerate(analysis['token_addresses'][:3]):  # Show first 3
                print(f"   {i+1}. {addr}")
                if is_creation:
                    print(f"      🔗 PUMP.FUN: https://pump.fun/{addr}")
                    print(f"      🔗 SOLSCAN: https://solscan.io/token/{addr}")
        
        print(f"📋 RAW LOGS:")
        for i, log in enumerate(analysis['logs'][:5]):  # Show first 5 logs
            print(f"   {i+1}. {log}")
        
        if analysis['signature'] != "Unknown":
            print(f"🔗 TRANSACTION: https://solscan.io/tx/{analysis['signature']}")
        
        runtime = (current_time - self.start_time).total_seconds()
        print(f"⏰ RUNTIME: {runtime:.1f}s")
        
        if is_creation:
            print(f"🚀" * 80)
        else:
            print(f"💫" * 60)
        
        # Save all activity
        self.save_activity(analysis, is_creation)
    
    def save_activity(self, analysis, is_creation):
        """Save activity to file"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        filename = "logs/pump_fun_all_activity.txt"
        
        with open(filename, "a") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            activity_type = "CREATION" if is_creation else "ACTIVITY"
            signature = analysis['signature']
            instructions = ','.join(analysis['instruction_types'])
            f.write(f"{timestamp} | {activity_type} | {signature} | {instructions}\n")
    
    async def process_log_message(self, message):
        """Process all incoming log messages"""
        
        try:
            data = json.loads(message)
            
            if "method" in data and data["method"] == "logsNotification":
                params = data.get("params", {})
                result = params.get("result", {})
                
                # Analyze all Pump.fun activity
                analysis = self.analyze_pump_fun_logs(result)
                
                # Display the activity
                await self.display_pump_fun_activity(analysis)
            
            elif "id" in data:
                print(f"📨 Response: {data}")
            
        except json.JSONDecodeError:
            print(f"⚠️ Invalid JSON: {message[:100]}")
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
    
    async def monitor_enhanced(self):
        """Enhanced monitoring showing all activity"""
        
        print("🔥 ENHANCED HELIUS GEYSER MONITOR")
        print("=" * 70)
        print("⚡ Shows ALL Pump.fun activity")
        print("🎯 Analyzes instruction patterns")
        print("🔍 Identifies token creations")
        print("📊 Real-time activity classification")
        print("")
        
        if not await self.connect_websocket():
            return
        
        if not await self.subscribe_to_pump_fun_logs():
            return
        
        print(f"🔥 ENHANCED MONITORING STARTED - {datetime.now().strftime('%H:%M:%S')}")
        print("🎯 Showing ALL Pump.fun program activity...")
        print("🚀 Token launches will be highlighted")
        print("")
        
        try:
            async for message in self.websocket:
                await self.process_log_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 WebSocket closed")
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitor stopped")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            print(f"📊 Total activity: {self.activity_count}")
            print(f"🚀 Token launches: {self.launch_count}")
            print(f"⏰ Runtime: {(datetime.now() - self.start_time).total_seconds():.1f}s")
            
            if self.websocket:
                await self.websocket.close()

async def main():
    """Main function"""
    
    helius_api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
    
    monitor = EnhancedGeyserMonitor(helius_api_key)
    await monitor.monitor_enhanced()

if __name__ == "__main__":
    print("🚀 Starting Enhanced Geyser Monitor...")
    asyncio.run(main()) 