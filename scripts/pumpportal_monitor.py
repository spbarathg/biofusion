#!/usr/bin/env python3
"""
PUMPPORTAL REAL-TIME MONITOR
===========================

Uses PumpPortal's WebSocket API to get REAL-TIME pump.fun token launches.
This is the working solution that shows tokens launching every second!
"""

import asyncio
import websockets
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class PumpPortalMonitor:
    """Real-time pump.fun monitor using PumpPortal WebSocket"""
    
    def __init__(self):
        self.websocket_url = "wss://pumpportal.fun/api/data"
        self.websocket = None
        self.launch_count = 0
        self.trade_count = 0
        self.start_time = datetime.now()
        
    async def connect(self):
        """Connect to PumpPortal WebSocket"""
        
        print("🔗 Connecting to PumpPortal...")
        
        try:
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=30,
                ping_timeout=10
            )
            
            print("✅ Connected to PumpPortal!")
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    async def subscribe_to_events(self):
        """Subscribe to new token launches and trades"""
        
        print("📡 Subscribing to real-time events...")
        
        try:
            # Subscribe to new token creation events
            new_token_payload = {
                "method": "subscribeNewToken"
            }
            await self.websocket.send(json.dumps(new_token_payload))
            print("✅ Subscribed to new token launches!")
            
            # Subscribe to migration events (tokens moving to Raydium)
            migration_payload = {
                "method": "subscribeMigration"
            }
            await self.websocket.send(json.dumps(migration_payload))
            print("✅ Subscribed to token migrations!")
            
            return True
            
        except Exception as e:
            print(f"❌ Subscription failed: {e}")
            return False
    
    def display_new_token(self, token_data):
        """Display new token launch with clean formatting"""
        
        self.launch_count += 1
        
        # Extract token information
        mint = token_data.get("mint", "Unknown")
        name = token_data.get("name", "Unknown")
        symbol = token_data.get("symbol", "UNKNOWN")
        description = token_data.get("description", "")
        image_uri = token_data.get("image_uri", "")
        metadata_uri = token_data.get("metadata_uri", "")
        twitter = token_data.get("twitter", "")
        telegram = token_data.get("telegram", "")
        bonding_curve = token_data.get("bonding_curve", "")
        associated_bonding_curve = token_data.get("associated_bonding_curve", "")
        creator = token_data.get("creator", "")
        market_cap = token_data.get("market_cap", 0)
        reply_count = token_data.get("reply_count", 0)
        timestamp = token_data.get("timestamp", 0)
        
        # Calculate age
        age_str = "Just now"
        if timestamp:
            try:
                created_time = datetime.fromtimestamp(timestamp / 1000)
                age_seconds = (datetime.now() - created_time).total_seconds()
                if age_seconds < 60:
                    age_str = f"{age_seconds:.0f} seconds ago"
                else:
                    age_str = f"{age_seconds/60:.1f} minutes ago"
            except:
                pass
        
        print(f"\n{'🚀' * 70}")
        print(f"🚀 NEW TOKEN LAUNCH #{self.launch_count} - {datetime.now().strftime('%H:%M:%S')} 🚀")
        print(f"{'🚀' * 70}")
        
        print(f"📍 ADDRESS: {mint}")
        print(f"🏷️  NAME: {name}")
        print(f"🎯 SYMBOL: {symbol}")
        print(f"⏰ AGE: {age_str}")
        
        if description and len(description) > 0:
            # Truncate long descriptions
            desc = description[:150] + "..." if len(description) > 150 else description
            print(f"📝 DESC: {desc}")
        
        if market_cap and market_cap > 0:
            print(f"💰 MARKET CAP: ${market_cap:,.2f}")
        
        if reply_count > 0:
            print(f"💬 REPLIES: {reply_count}")
        
        if creator:
            print(f"👤 CREATOR: {creator}")
        
        # Social links
        if twitter:
            print(f"🐦 TWITTER: {twitter}")
        if telegram:
            print(f"📱 TELEGRAM: {telegram}")
        
        # Important links
        print(f"\n🔗 QUICK LINKS:")
        print(f"   🔗 PUMP: https://pump.fun/{mint}")
        print(f"   🔗 SCAN: https://solscan.io/token/{mint}")
        print(f"   🔗 DEX: https://dexscreener.com/solana/{mint}")
        
        if bonding_curve:
            print(f"   📈 CURVE: {bonding_curve}")
        
        # Stats
        runtime = (datetime.now() - self.start_time).total_seconds()
        rate = self.launch_count / (runtime / 60) if runtime > 0 else 0
        print(f"\n📊 SESSION: {self.launch_count} launches | {self.trade_count} trades | {runtime:.0f}s | {rate:.1f}/min")
        
        print(f"{'🚀' * 70}")
        
        # Save to file
        self.save_launch(token_data)
    
    def display_trade(self, trade_data):
        """Display trade with minimal info"""
        
        self.trade_count += 1
        
        # Only show every 10th trade to avoid spam
        if self.trade_count % 10 == 0:
            mint = trade_data.get("mint", "Unknown")
            tx_type = trade_data.get("txType", "unknown")
            sol_amount = trade_data.get("solAmount", 0)
            token_amount = trade_data.get("tokenAmount", 0)
            
            print(f"💹 TRADE #{self.trade_count}: {tx_type.upper()} | {mint[:8]}... | {sol_amount:.3f} SOL")
    
    def display_migration(self, migration_data):
        """Display token migration to Raydium"""
        
        mint = migration_data.get("mint", "Unknown")
        signature = migration_data.get("signature", "")
        
        print(f"\n🌊 TOKEN MIGRATION TO RAYDIUM 🌊")
        print(f"📍 TOKEN: {mint}")
        print(f"📍 TX: {signature}")
        print(f"🔗 PUMP: https://pump.fun/{mint}")
        print(f"🔗 RAYDIUM: https://raydium.io/")
        print("🌊" * 50)
    
    def save_launch(self, token_data):
        """Save launch to log file"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/pumpportal_launches.txt", "a") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            mint = token_data.get("mint", "Unknown")
            name = token_data.get("name", "Unknown")
            symbol = token_data.get("symbol", "UNKNOWN")
            market_cap = token_data.get("market_cap", 0)
            
            f.write(f"{timestamp} | {mint} | {symbol} | {name} | ${market_cap:.2f}\n")
    
    async def process_message(self, message):
        """Process incoming WebSocket messages"""
        
        try:
            data = json.loads(message)
            
            # Check message type
            if isinstance(data, dict):
                # New token creation
                if "mint" in data and "name" in data:
                    self.display_new_token(data)
                
                # Trade data
                elif "txType" in data and "solAmount" in data:
                    self.display_trade(data)
                
                # Migration data
                elif "signature" in data and not "txType" in data:
                    self.display_migration(data)
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
    
    async def monitor_realtime(self):
        """Main monitoring loop"""
        
        print("🚀 PUMPPORTAL REAL-TIME MONITOR")
        print("=" * 60)
        print("🎯 REAL-TIME pump.fun token launches")
        print("⚡ WebSocket stream from PumpPortal")
        print("🚀 Shows tokens launching every second")
        print("📊 Includes market data and social links")
        print("💹 Live trade notifications")
        print("")
        
        if not await self.connect():
            return
        
        if not await self.subscribe_to_events():
            return
        
        print(f"🚀 REAL-TIME MONITOR ACTIVE - {datetime.now().strftime('%H:%M:%S')}")
        print("🎯 Streaming live pump.fun launches...")
        print("⚡ Press Ctrl+C to stop")
        print("")
        
        try:
            async for message in self.websocket:
                await self.process_message(message)
                
        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 WebSocket closed")
        except KeyboardInterrupt:
            print(f"\n⏹️ Real-time monitor stopped")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            runtime = (datetime.now() - self.start_time).total_seconds()
            launch_rate = self.launch_count / (runtime / 60) if runtime > 0 else 0
            trade_rate = self.trade_count / (runtime / 60) if runtime > 0 else 0
            
            print(f"📊 Final Stats:")
            print(f"   🚀 Launches: {self.launch_count} ({launch_rate:.1f}/min)")
            print(f"   💹 Trades: {self.trade_count} ({trade_rate:.1f}/min)")
            print(f"   ⏰ Runtime: {runtime:.0f}s")
            
            if self.websocket:
                await self.websocket.close()

async def main():
    """Main function"""
    
    monitor = PumpPortalMonitor()
    await monitor.monitor_realtime()

if __name__ == "__main__":
    print("🚀 Starting PumpPortal Real-Time Monitor!")
    print("💡 This uses the working WebSocket API!")
    asyncio.run(main()) 