#!/usr/bin/env python3
"""
RAYDIUM LAUNCH MONITOR
=====================

Alternative monitor that watches for new Raydium pools when pump.fun API is down.
Monitors Raydium program for new token pools being created.
"""

import asyncio
import websockets
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class RaydiumLaunchMonitor:
    """Monitor new token launches via Raydium pool creation"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.websocket_url = f"wss://mainnet.helius-rpc.com/?api-key={helius_api_key}"
        self.raydium_program_id = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"  # Raydium AMM
        self.websocket = None
        self.subscription_id = None
        self.launch_count = 0
        self.start_time = datetime.now()
        
    async def connect_websocket(self):
        """Connect to Helius WebSocket"""
        
        print("🔗 Connecting to Raydium monitor...")
        
        try:
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            print("✅ Connected to Helius!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def subscribe_to_raydium(self):
        """Subscribe to Raydium AMM program"""
        
        subscription_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {
                    "mentions": [self.raydium_program_id]
                },
                {
                    "commitment": "processed"
                }
            ]
        }
        
        print("📡 Subscribing to Raydium pools...")
        
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
    
    def is_new_pool_creation(self, logs):
        """Check if this is a new pool creation"""
        
        pool_creation_indicators = [
            "initialize2",
            "InitializeInstruction",
            "create_pool", 
            "CreatePool",
            "ray_log_liquidity",
            "AddLiquidity",
            "initialize_pool",
            "InitializePool"
        ]
        
        # Check for pool creation indicators
        for log in logs:
            log_lower = log.lower()
            for indicator in pool_creation_indicators:
                if indicator.lower() in log_lower:
                    return True
        
        # Look for token addresses in logs (new tokens usually have multiple addresses)
        token_addresses = []
        for log in logs:
            words = log.split()
            for word in words:
                if len(word) == 44 and word.replace('_', '').replace('-', '').isalnum():
                    if word not in token_addresses:
                        token_addresses.append(word)
        
        return len(token_addresses) >= 3  # Pool creation involves multiple addresses
    
    def extract_pool_info(self, log_data):
        """Extract pool information from logs"""
        
        logs = log_data.get("logs", [])
        signature = log_data.get("signature", "Unknown")
        slot = log_data.get("slot", 0)
        
        # Extract token addresses
        token_addresses = []
        for log in logs:
            words = log.split()
            for word in words:
                if len(word) == 44 and word.replace('_', '').replace('-', '').isalnum():
                    if word not in token_addresses:
                        token_addresses.append(word)
        
        # Try to identify base and quote tokens
        base_token = token_addresses[0] if token_addresses else "Unknown"
        quote_token = token_addresses[1] if len(token_addresses) > 1 else "Unknown"
        
        return {
            "signature": signature,
            "slot": slot,
            "base_token": base_token,
            "quote_token": quote_token,
            "all_addresses": token_addresses,
            "timestamp": datetime.now(),
            "logs": logs,
            "pool_type": "Raydium"
        }
    
    def display_pool_launch(self, pool_info):
        """Display new pool launch"""
        
        self.launch_count += 1
        
        print(f"\n{'🌊' * 60}")
        print(f"🌊 NEW RAYDIUM POOL #{self.launch_count} - {datetime.now().strftime('%H:%M:%S')} 🌊")
        print(f"{'🌊' * 60}")
        
        print(f"📍 BASE TOKEN: {pool_info['base_token']}")
        print(f"💰 QUOTE TOKEN: {pool_info['quote_token']}")
        print(f"🏊 POOL TYPE: {pool_info['pool_type']}")
        print(f"🕐 TIME: {pool_info['timestamp'].strftime('%H:%M:%S')}")
        
        if len(pool_info['all_addresses']) > 2:
            print(f"📋 OTHER ADDRESSES:")
            for i, addr in enumerate(pool_info['all_addresses'][2:6], 1):
                print(f"   {i}. {addr}")
        
        # Links
        base_token = pool_info['base_token']
        print(f"🔗 PUMP: https://pump.fun/{base_token}")
        print(f"🔗 SCAN: https://solscan.io/token/{base_token}")
        print(f"🔗 DEX: https://dexscreener.com/solana/{base_token}")
        print(f"📍 TX: {pool_info['signature']}")
        
        # Stats
        runtime = (pool_info["timestamp"] - self.start_time).total_seconds()
        rate = self.launch_count / (runtime / 60) if runtime > 0 else 0
        print(f"📊 TOTAL: {self.launch_count} | RUNTIME: {runtime:.0f}s | RATE: {rate:.1f}/min")
        
        print(f"{'🌊' * 60}")
        
        self.save_launch(pool_info)
    
    def save_launch(self, pool_info):
        """Save launch info"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/raydium_pools.txt", "a") as f:
            timestamp = pool_info["timestamp"].strftime('%Y-%m-%d %H:%M:%S')
            base_token = pool_info["base_token"]
            signature = pool_info["signature"]
            
            f.write(f"{timestamp} | {base_token} | {signature}\n")
    
    async def process_message(self, message):
        """Process WebSocket messages"""
        
        try:
            data = json.loads(message)
            
            if "method" in data and data["method"] == "logsNotification":
                params = data.get("params", {})
                result = params.get("result", {})
                logs = result.get("logs", [])
                
                if self.is_new_pool_creation(logs):
                    pool_info = self.extract_pool_info(result)
                    self.display_pool_launch(pool_info)
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
    
    async def monitor_raydium_pools(self):
        """Monitor Raydium pool creation"""
        
        print("🌊 RAYDIUM POOL LAUNCH MONITOR")
        print("=" * 50)
        print("🎯 Alternative when pump.fun API is down")
        print("🌊 Monitors Raydium AMM for new pools")
        print("⚡ Real-time pool detection")
        print("📊 Clean pool launch display")
        print("")
        
        if not await self.connect_websocket():
            return
        
        if not await self.subscribe_to_raydium():
            return
        
        print(f"🌊 RAYDIUM MONITOR ACTIVE - {datetime.now().strftime('%H:%M:%S')}")
        print("🎯 Waiting for new pool creations...")
        print("⚡ Press Ctrl+C to stop")
        print("")
        
        try:
            async for message in self.websocket:
                await self.process_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 WebSocket closed")
        except KeyboardInterrupt:
            print(f"\n⏹️ Raydium monitor stopped")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            runtime = (datetime.now() - self.start_time).total_seconds()
            rate = self.launch_count / (runtime / 60) if runtime > 0 else 0
            print(f"📊 Final: {self.launch_count} pools in {runtime:.0f}s ({rate:.1f}/min)")
            
            if self.websocket:
                await self.websocket.close()

async def main():
    """Main function"""
    
    helius_api_key = "193ececa-6e42-4d84-b9bd-765c4813816d"
    
    monitor = RaydiumLaunchMonitor(helius_api_key)
    await monitor.monitor_raydium_pools()

if __name__ == "__main__":
    print("🌊 Starting Raydium Pool Monitor (Pump.fun Fallback)!")
    asyncio.run(main()) 