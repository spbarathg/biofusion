#!/usr/bin/env python3
"""
HELIUS GEYSER REAL-TIME MONITOR
==============================

Real-time WebSocket streaming of Pump.fun token launches using Helius Geyser RPC.
Ultra-low latency detection via logsSubscribe to Pump.fun program ID.
"""

import asyncio
import websockets
import json
import base64
import struct
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class HeliusGeyserMonitor:
    """Real-time Pump.fun monitor using Helius Geyser WebSocket"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.websocket_url = f"wss://mainnet.helius-rpc.com/?api-key={helius_api_key}"
        self.pump_fun_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        self.websocket = None
        self.subscription_id = None
        self.launch_count = 0
        self.start_time = datetime.now()
        
    async def connect_websocket(self):
        """Connect to Helius Geyser WebSocket"""
        
        print("🔗 Connecting to Helius Geyser WebSocket...")
        print(f"🎯 Target: {self.websocket_url}")
        print(f"📡 Program ID: {self.pump_fun_program_id}")
        
        try:
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            print("✅ WebSocket connected successfully!")
            return True
            
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return False
    
    async def subscribe_to_pump_fun_logs(self):
        """Subscribe to Pump.fun program logs via logsSubscribe"""
        
        subscription_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {
                    "mentions": [self.pump_fun_program_id]
                },
                {
                    "commitment": "processed"  # Fastest confirmation level
                }
            ]
        }
        
        print("📡 Subscribing to Pump.fun logs...")
        print(f"🎯 Method: logsSubscribe")
        print(f"🚀 Commitment: processed (fastest)")
        
        try:
            await self.websocket.send(json.dumps(subscription_request))
            
            # Wait for subscription confirmation
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if "result" in response_data:
                self.subscription_id = response_data["result"]
                print(f"✅ Subscribed! Subscription ID: {self.subscription_id}")
                print("🔥 Now streaming real-time Pump.fun events...")
                return True
            else:
                print(f"❌ Subscription failed: {response_data}")
                return False
                
        except Exception as e:
            print(f"❌ Subscription error: {e}")
            return False
    
    def parse_pump_fun_instruction(self, log_data):
        """Parse Pump.fun instruction data to extract token details"""
        
        try:
            # Look for "Instruction: Create" in logs
            logs = log_data.get("logs", [])
            
            for log in logs:
                if "Instruction: Create" in log:
                    print(f"🔥 DETECTED: {log}")
                    return True
            
            # Also check for token creation patterns
            for log in logs:
                if any(keyword in log.lower() for keyword in ["create", "mint", "initialize"]):
                    if "pump" in log.lower():
                        print(f"🔥 PUMP CREATION: {log}")
                        return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Parse error: {e}")
            return False
    
    def extract_token_metadata(self, transaction_data):
        """Extract token metadata from transaction"""
        
        try:
            signature = transaction_data.get("signature", "Unknown")
            slot = transaction_data.get("slot", 0)
            
            # Extract accounts involved
            accounts = []
            if "transaction" in transaction_data:
                tx = transaction_data["transaction"]
                if "message" in tx and "accountKeys" in tx["message"]:
                    accounts = tx["message"]["accountKeys"]
            
            # Find potential mint address (usually first new account)
            mint_address = "Unknown"
            if accounts:
                # Look for new mint address (not known system accounts)
                for account in accounts:
                    if len(account) == 44:  # Solana address length
                        if account not in [
                            "11111111111111111111111111111112",  # System program
                            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token program
                            self.pump_fun_program_id
                        ]:
                            mint_address = account
                            break
            
            return {
                "signature": signature,
                "mint_address": mint_address,
                "slot": slot,
                "timestamp": datetime.now(),
                "accounts": accounts[:3]  # First 3 accounts for reference
            }
            
        except Exception as e:
            print(f"⚠️ Metadata extraction error: {e}")
            return None
    
    async def display_fresh_launch(self, token_data, log_data):
        """Display fresh token launch with full details"""
        
        self.launch_count += 1
        
        current_time = datetime.now()
        runtime = (current_time - self.start_time).total_seconds()
        
        print(f"\n" + "🚀" * 80)
        print(f"🚀 REAL-TIME PUMP.FUN LAUNCH #{self.launch_count} - {current_time.strftime('%H:%M:%S.%f')[:-3]} 🚀")
        print(f"🚀" * 80)
        
        # Token details
        if token_data:
            print(f"📍 SIGNATURE: {token_data['signature']}")
            print(f"🪙 MINT ADDRESS: {token_data['mint_address']}")
            print(f"🎯 SLOT: {token_data['slot']}")
            print(f"📅 DETECTED: {token_data['timestamp'].strftime('%H:%M:%S.%f')[:-3]}")
            
            if token_data['accounts']:
                print(f"👥 ACCOUNTS: {', '.join(token_data['accounts'])}")
        
        # Log analysis
        logs = log_data.get("logs", [])
        print(f"📋 LOG ENTRIES: {len(logs)}")
        
        for i, log in enumerate(logs[:5]):  # Show first 5 logs
            print(f"   {i+1}. {log}")
        
        # Links
        if token_data and token_data['mint_address'] != "Unknown":
            mint = token_data['mint_address']
            print(f"🔗 PUMP.FUN: https://pump.fun/{mint}")
            print(f"🔗 SOLSCAN: https://solscan.io/token/{mint}")
            print(f"🔗 DEXSCREENER: https://dexscreener.com/solana/{mint}")
        
        if token_data and token_data['signature'] != "Unknown":
            print(f"🔗 TRANSACTION: https://solscan.io/tx/{token_data['signature']}")
        
        print(f"⚡ LATENCY: Real-time WebSocket stream")
        print(f"⏰ RUNTIME: {runtime:.1f}s")
        print(f"🚀" * 80)
        
        # Save to log
        self.save_launch(token_data, log_data)
    
    def save_launch(self, token_data, log_data):
        """Save launch to file"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/helius_geyser_launches.txt", "a") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            signature = token_data['signature'] if token_data else "Unknown"
            mint = token_data['mint_address'] if token_data else "Unknown"
            f.write(f"{timestamp} | {signature} | {mint} | Geyser\n")
    
    async def process_log_message(self, message):
        """Process incoming log message from WebSocket"""
        
        try:
            data = json.loads(message)
            
            # Check if this is a log notification
            if "method" in data and data["method"] == "logsNotification":
                params = data.get("params", {})
                result = params.get("result", {})
                
                # Check if this contains Pump.fun activity
                if self.parse_pump_fun_instruction(result):
                    # Extract token metadata
                    token_data = self.extract_token_metadata(result)
                    
                    # Display the launch
                    await self.display_fresh_launch(token_data, result)
                
                else:
                    # Just show activity indicator
                    print(f"💫 Pump.fun activity detected (no creation)", end="\r")
            
            # Handle other message types
            elif "id" in data:
                print(f"📨 Response: {data}")
            
        except json.JSONDecodeError:
            print(f"⚠️ Invalid JSON: {message[:100]}")
        except Exception as e:
            print(f"⚠️ Message processing error: {e}")
    
    async def monitor_real_time(self):
        """Main real-time monitoring loop"""
        
        print("🔥 HELIUS GEYSER REAL-TIME MONITOR")
        print("=" * 70)
        print("⚡ Ultra-low latency WebSocket streaming")
        print("🎯 Target: Pump.fun program logs")
        print("📡 Method: logsSubscribe")
        print("🚀 Commitment: processed (fastest)")
        print("⏱️  Real-time event-driven detection")
        print("")
        
        # Connect to WebSocket
        if not await self.connect_websocket():
            print("❌ Failed to connect to WebSocket")
            return
        
        # Subscribe to logs
        if not await self.subscribe_to_pump_fun_logs():
            print("❌ Failed to subscribe to logs")
            return
        
        print(f"🔥 MONITORING STARTED - {datetime.now().strftime('%H:%M:%S')}")
        print("🎯 Waiting for real-time Pump.fun token launches...")
        print("⚡ Press Ctrl+C to stop")
        print("")
        
        try:
            # Listen for messages
            async for message in self.websocket:
                await self.process_log_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 WebSocket connection closed")
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitor stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            print(f"📊 Total launches detected: {self.launch_count}")
            print(f"⏰ Total runtime: {(datetime.now() - self.start_time).total_seconds():.1f}s")
            
            if self.websocket:
                await self.websocket.close()

async def main():
    """Main function"""
    
    helius_api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
    
    monitor = HeliusGeyserMonitor(helius_api_key)
    await monitor.monitor_real_time()

if __name__ == "__main__":
    print("🚀 Starting Helius Geyser Real-Time Monitor...")
    asyncio.run(main()) 