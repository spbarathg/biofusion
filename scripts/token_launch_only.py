#!/usr/bin/env python3
"""
TOKEN LAUNCH ONLY MONITOR
========================

Shows ONLY new token launches on Pump.fun with clean, focused details.
No noise, just the essential launch information.
"""

import asyncio
import websockets
import json
from datetime import datetime
import sys
import re

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class TokenLaunchMonitor:
    """Monitor that shows only token launches"""
    
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
        
        print("🔗 Connecting to Token Launch Monitor...")
        
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
    
    async def subscribe_to_launches(self):
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
        
        print("📡 Subscribing to token launches...")
        
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
    
    def is_token_launch(self, logs):
        """Check if this is a token launch event"""
        
        # More comprehensive launch detection
        launch_indicators = [
            "Program 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P invoke",  # Pump.fun program
            "create",
            "Create",
            "initialize", 
            "Initialize",
            "mint",
            "Mint",
            "InitializeMint",
            "CreateTokenAccount",
            "new token",
            "token created",
            "bonding curve",
            "curve created"
        ]
        
        # Check all logs for launch indicators
        for log in logs:
            for indicator in launch_indicators:
                if indicator.lower() in log.lower():
                    return True
                    
        # Also check if we have token-like addresses being created
        token_address_count = 0
        for log in logs:
            words = log.split()
            for word in words:
                if len(word) == 44 and word.replace('_', '').replace('-', '').isalnum():
                    token_address_count += 1
        
        # If we have multiple addresses and pump.fun program involvement, likely a launch
        return token_address_count >= 2
    
    def extract_token_info(self, log_data):
        """Extract token information from launch event"""
        
        logs = log_data.get("logs", [])
        signature = log_data.get("signature", "Unknown")
        slot = log_data.get("slot", 0)
        
        # Extract token addresses (44 character Solana addresses)
        token_addresses = []
        for log in logs:
            # Look for 44-character base58 addresses
            words = log.split()
            for word in words:
                if len(word) == 44 and word.replace('_', '').replace('-', '').isalnum():
                    if word not in token_addresses:
                        token_addresses.append(word)
        
        # Extract any metadata if available
        token_name = "Unknown"
        token_symbol = "Unknown"
        
        # Look for common patterns in logs that might contain name/symbol
        for log in logs:
            # Simple pattern matching for potential metadata
            if "name:" in log.lower() or "symbol:" in log.lower():
                # Extract basic info if available
                parts = log.split()
                for i, part in enumerate(parts):
                    if "name" in part.lower() and i + 1 < len(parts):
                        token_name = parts[i + 1].strip('"').strip("'")
                    elif "symbol" in part.lower() and i + 1 < len(parts):
                        token_symbol = parts[i + 1].strip('"').strip("'")
        
        return {
            "signature": signature,
            "slot": slot,
            "addresses": token_addresses,
            "name": token_name,
            "symbol": token_symbol,
            "timestamp": datetime.now(),
            "logs": logs
        }
    
    def display_token_launch(self, token_info):
        """Display token launch with clean formatting"""
        
        self.launch_count += 1
        
        print("\n" + "🚀" * 60)
        print(f"🚀 NEW TOKEN LAUNCH #{self.launch_count}")
        print("🚀" * 60)
        
        # Time
        time_str = token_info["timestamp"].strftime('%H:%M:%S')
        print(f"⏰ TIME: {time_str}")
        
        # Main token address (usually the first one)
        if token_info["addresses"]:
            main_address = token_info["addresses"][0]
            print(f"🪙 TOKEN: {main_address}")
            print(f"🔗 PUMP: https://pump.fun/{main_address}")
            print(f"🔗 SCAN: https://solscan.io/token/{main_address}")
            
            # Additional addresses if any
            if len(token_info["addresses"]) > 1:
                print(f"📋 OTHER ADDRESSES:")
                for i, addr in enumerate(token_info["addresses"][1:6], 1):  # Show up to 5 more
                    print(f"   {i}. {addr}")
        
        # Token details
        if token_info["name"] != "Unknown":
            print(f"🏷️ NAME: {token_info['name']}")
        if token_info["symbol"] != "Unknown":
            print(f"🎯 SYMBOL: {token_info['symbol']}")
        
        # Transaction details
        print(f"📍 TX: {token_info['signature']}")
        print(f"🎯 SLOT: {token_info['slot']}")
        
        # Runtime stats
        runtime = (token_info["timestamp"] - self.start_time).total_seconds()
        print(f"📊 TOTAL LAUNCHES: {self.launch_count} | RUNTIME: {runtime:.1f}s")
        
        print("🚀" * 60)
        
        # Save to file
        self.save_launch(token_info)
    
    def save_launch(self, token_info):
        """Save launch info to file"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/live_token_launches.txt", "a") as f:
            timestamp = token_info["timestamp"].strftime('%Y-%m-%d %H:%M:%S')
            main_address = token_info["addresses"][0] if token_info["addresses"] else "Unknown"
            signature = token_info["signature"]
            name = token_info["name"]
            symbol = token_info["symbol"]
            
            f.write(f"{timestamp} | {main_address} | {name} | {symbol} | {signature}\n")
    
    async def process_message(self, message):
        """Process incoming messages and filter for launches"""
        
        try:
            data = json.loads(message)
            
            # Process log notifications
            if "method" in data and data["method"] == "logsNotification":
                params = data.get("params", {})
                result = params.get("result", {})
                logs = result.get("logs", [])
                
                # Check if this is a token launch
                if self.is_token_launch(logs):
                    token_info = self.extract_token_info(result)
                    self.display_token_launch(token_info)
            
        except json.JSONDecodeError:
            pass  # Ignore malformed messages
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
    
    async def monitor_launches(self):
        """Monitor only token launches"""
        
        print("🚀 TOKEN LAUNCH MONITOR")
        print("=" * 50)
        print("🎯 Shows ONLY new token launches")
        print("🪙 Clean, focused token details")
        print("🔗 Direct links to Pump.fun & Solscan")
        print("📊 Real-time launch tracking")
        print("")
        
        if not await self.connect_websocket():
            return
        
        if not await self.subscribe_to_launches():
            return
        
        print(f"🚀 LAUNCH MONITOR ACTIVE - {datetime.now().strftime('%H:%M:%S')}")
        print("🎯 Waiting for new token launches...")
        print("⚡ Press Ctrl+C to stop")
        print("")
        
        try:
            async for message in self.websocket:
                await self.process_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 WebSocket closed")
        except KeyboardInterrupt:
            print(f"\n⏹️ Launch monitor stopped")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            print(f"📊 Total launches captured: {self.launch_count}")
            print(f"⏰ Total runtime: {(datetime.now() - self.start_time).total_seconds():.1f}s")
            
            if self.websocket:
                await self.websocket.close()

async def main():
    """Main function"""
    
    helius_api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
    
    monitor = TokenLaunchMonitor(helius_api_key)
    await monitor.monitor_launches()

if __name__ == "__main__":
    print("🚀 Starting Token Launch Monitor - Launches Only!")
    asyncio.run(main()) 