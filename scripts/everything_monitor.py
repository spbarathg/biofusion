#!/usr/bin/env python3
"""
EVERYTHING MONITOR
=================

Shows EVERYTHING happening on Pump.fun - no filtering, no conditions.
Every single transaction, every log, every activity in real-time.
"""

import asyncio
import websockets
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class EverythingMonitor:
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
        
        print("🔗 Connecting to EVERYTHING Monitor...")
        
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
    
    async def subscribe_to_everything(self):
        """Subscribe to ALL Pump.fun activity with no filtering"""
        
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
        
        print("📡 Subscribing to EVERYTHING on Pump.fun...")
        print("🚨 NO FILTERING - SHOWING ALL ACTIVITY")
        
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
    
    def extract_all_data(self, log_data):
        """Extract ALL data from logs with no filtering"""
        
        logs = log_data.get("logs", [])
        signature = log_data.get("signature", "Unknown")
        slot = log_data.get("slot", 0)
        err = log_data.get("err", None)
        
        # Extract all addresses
        all_addresses = []
        all_instructions = []
        all_keywords = []
        
        for log in logs:
            # Extract all instructions
            if "Instruction:" in log:
                instruction_part = log.split("Instruction:")[1].strip()
                all_instructions.append(instruction_part)
            
            # Extract all addresses (44 chars)
            words = log.split()
            for word in words:
                if len(word) == 44 and word.replace('_', '').replace('-', '').isalnum():
                    if word not in all_addresses:
                        all_addresses.append(word)
            
            # Extract interesting keywords
            keywords = ["Create", "Buy", "Sell", "Transfer", "Mint", "Burn", "Initialize", "Close"]
            for keyword in keywords:
                if keyword.lower() in log.lower():
                    all_keywords.append(keyword)
        
        return {
            "signature": signature,
            "slot": slot,
            "error": err,
            "logs": logs,
            "log_count": len(logs),
            "addresses": all_addresses,
            "instructions": all_instructions,
            "keywords": list(set(all_keywords)),  # Remove duplicates
            "timestamp": datetime.now()
        }
    
    async def display_everything(self, data):
        """Display everything with no filtering"""
        
        self.event_count += 1
        current_time = data["timestamp"]
        
        # Determine activity type based on content
        keywords = data.get("keywords", [])
        instructions = data.get("instructions", [])
        
        if "Create" in keywords or any("create" in inst.lower() for inst in instructions):
            activity_type = "🚀 TOKEN CREATION"
            border = "🚀" * 80
        elif "Buy" in keywords or any("buy" in inst.lower() for inst in instructions):
            activity_type = "💰 BUY ORDER"
            border = "💰" * 80
        elif "Sell" in keywords or any("sell" in inst.lower() for inst in instructions):
            activity_type = "💸 SELL ORDER"
            border = "💸" * 80
        else:
            activity_type = "⚡ PUMP.FUN ACTIVITY"
            border = "⚡" * 80
        
        print(f"\n{border}")
        print(f"{activity_type} #{self.event_count} - {current_time.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"{border}")
        
        # Basic info
        print(f"📍 SIGNATURE: {data['signature']}")
        print(f"🎯 SLOT: {data['slot']}")
        print(f"📋 LOG COUNT: {data['log_count']}")
        
        if data['error']:
            print(f"❌ ERROR: {data['error']}")
        
        # Instructions
        if data['instructions']:
            print(f"🎯 INSTRUCTIONS: {len(data['instructions'])}")
            for i, inst in enumerate(data['instructions']):
                print(f"   {i+1}. {inst}")
        
        # Keywords found
        if data['keywords']:
            print(f"🔑 KEYWORDS: {', '.join(data['keywords'])}")
        
        # Addresses
        if data['addresses']:
            print(f"🪙 ADDRESSES: {len(data['addresses'])}")
            for i, addr in enumerate(data['addresses'][:5]):  # Show first 5
                print(f"   {i+1}. {addr}")
                print(f"      🔗 PUMP: https://pump.fun/{addr}")
                print(f"      🔗 SCAN: https://solscan.io/token/{addr}")
        
        # ALL Raw logs
        print(f"📋 ALL LOGS:")
        for i, log in enumerate(data['logs']):
            print(f"   {i+1}. {log}")
        
        # Links
        if data['signature'] != "Unknown":
            print(f"🔗 TRANSACTION: https://solscan.io/tx/{data['signature']}")
        
        runtime = (current_time - self.start_time).total_seconds()
        print(f"⏰ RUNTIME: {runtime:.1f}s | EVENTS: {self.event_count}")
        print(f"{border}")
        
        # Save everything
        self.save_everything(data)
    
    def save_everything(self, data):
        """Save everything to file"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/everything_pump_fun.txt", "a") as f:
            timestamp = data["timestamp"].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            signature = data['signature']
            keywords = ','.join(data['keywords'])
            instructions = ' | '.join(data['instructions'])
            f.write(f"{timestamp} | {signature} | {keywords} | {instructions}\n")
    
    async def process_everything(self, message):
        """Process every single message with no filtering"""
        
        try:
            data = json.loads(message)
            
            # Process ALL log notifications
            if "method" in data and data["method"] == "logsNotification":
                params = data.get("params", {})
                result = params.get("result", {})
                
                # Extract and display everything
                extracted_data = self.extract_all_data(result)
                await self.display_everything(extracted_data)
            
            # Show all other messages too
            elif "id" in data:
                print(f"📨 SYSTEM: {data}")
            else:
                print(f"📨 OTHER: {data}")
            
        except json.JSONDecodeError:
            print(f"⚠️ RAW MESSAGE: {message}")
        except Exception as e:
            print(f"⚠️ ERROR: {e} | MESSAGE: {message[:200]}")
    
    async def monitor_everything(self):
        """Monitor everything with no limits"""
        
        print("🔥 PUMP.FUN EVERYTHING MONITOR")
        print("=" * 70)
        print("🚨 NO FILTERING - SHOWS EVERYTHING")
        print("⚡ Every transaction, every log, every activity")
        print("🎯 Real-time stream of all Pump.fun events")
        print("📊 Complete activity classification")
        print("💯 Zero conditions, zero limits")
        print("")
        
        if not await self.connect_websocket():
            return
        
        if not await self.subscribe_to_everything():
            return
        
        print(f"🔥 EVERYTHING MONITOR STARTED - {datetime.now().strftime('%H:%M:%S')}")
        print("🚨 SHOWING ABSOLUTELY EVERYTHING ON PUMP.FUN")
        print("⚡ Press Ctrl+C to stop")
        print("")
        
        try:
            async for message in self.websocket:
                await self.process_everything(message)
                
        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 WebSocket closed")
        except KeyboardInterrupt:
            print(f"\n⏹️ Everything monitor stopped")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            print(f"📊 Total events captured: {self.event_count}")
            print(f"⏰ Total runtime: {(datetime.now() - self.start_time).total_seconds():.1f}s")
            
            if self.websocket:
                await self.websocket.close()

async def main():
    """Main function"""
    
    helius_api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
    
    monitor = EverythingMonitor(helius_api_key)
    await monitor.monitor_everything()

if __name__ == "__main__":
    print("🚨 Starting EVERYTHING Monitor - No Limits, No Filtering!")
    asyncio.run(main()) 