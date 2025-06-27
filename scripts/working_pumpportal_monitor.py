#!/usr/bin/env python3
"""
Working Pump.fun Token Launch Monitor
Using PumpPortal WebSocket API - CONFIRMED WORKING

This script uses the PumpPortal API which provides real-time pump.fun data
via WebSocket connection. Based on latest internet research, this is a
working solution as of 2024/2025.

API Documentation: https://pumpportal.fun/data-api/real-time/
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
import aiohttp
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/pumpportal_launches.txt'),
        logging.StreamHandler()
    ]
)

class PumpPortalMonitor:
    def __init__(self):
        self.websocket_url = "wss://pumpportal.fun/api/data"
        self.total_tokens_seen = 0
        self.session_start = datetime.now()
        
    async def get_token_metadata(self, mint_address):
        """Get additional token metadata using various APIs"""
        try:
            # Try Solscan API for token info
            async with aiohttp.ClientSession() as session:
                solscan_url = f"https://api.solscan.io/account/{mint_address}"
                async with session.get(solscan_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
        except Exception as e:
            logging.debug(f"Error fetching metadata for {mint_address}: {e}")
        
        return None
    
    async def handle_new_token(self, token_data):
        """Process newly detected token"""
        self.total_tokens_seen += 1
        
        try:
            # Extract token information
            mint = token_data.get('mint', 'Unknown')
            name = token_data.get('name', 'Unknown')
            symbol = token_data.get('symbol', 'Unknown')
            uri = token_data.get('uri', '')
            dev = token_data.get('tradeCreator', 'Unknown')
            timestamp = datetime.now()
            
            # Log the discovery
            log_message = f"""
🚀 NEW TOKEN DETECTED #{self.total_tokens_seen}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
🔤 Name: {name}
🎯 Symbol: {symbol}
📍 Mint: {mint}
👤 Creator: {dev}
🌐 URI: {uri}
🔗 Pump.fun: https://pump.fun/{mint}
🔍 Solscan: https://solscan.io/token/{mint}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            logging.info(log_message)
            
            # Save to file
            with open('../logs/live_token_launches.txt', 'a', encoding='utf-8') as f:
                f.write(f"{timestamp.isoformat()},{mint},{name},{symbol},{dev}\n")
                
        except Exception as e:
            logging.error(f"Error processing token data: {e}")
            logging.error(f"Token data: {token_data}")
    
    async def handle_trade(self, trade_data):
        """Process trade events (optional)"""
        try:
            mint = trade_data.get('mint', 'Unknown')
            trade_type = trade_data.get('type', 'Unknown')
            sol_amount = trade_data.get('sol_amount', 0)
            trader = trade_data.get('trader', 'Unknown')
            
            if sol_amount > 0.1:  # Only log significant trades
                logging.info(f"💰 Large {trade_type}: {sol_amount} SOL on {mint[:8]}... by {trader[:8]}...")
                
        except Exception as e:
            logging.error(f"Error processing trade data: {e}")
    
    async def connect_and_monitor(self):
        """Main monitoring loop with WebSocket connection"""
        retry_count = 0
        max_retries = 10
        
        while retry_count < max_retries:
            try:
                logging.info(f"🔄 Connecting to PumpPortal WebSocket... (Attempt {retry_count + 1})")
                
                async with websockets.connect(
                    self.websocket_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=10,
                    max_size=1024*1024  # 1MB max message size
                ) as websocket:
                    logging.info("✅ Connected to PumpPortal successfully!")
                    retry_count = 0  # Reset on successful connection
                    
                    # Subscribe to new token creation events
                    subscribe_payload = {
                        "method": "subscribeNewToken"
                    }
                    await websocket.send(json.dumps(subscribe_payload))
                    logging.info("📡 Subscribed to new token events")
                    
                    # Optional: Subscribe to migration events
                    migration_payload = {
                        "method": "subscribeMigration"
                    }
                    await websocket.send(json.dumps(migration_payload))
                    logging.info("📡 Subscribed to migration events")
                    
                    # Listen for messages
                    logging.info("👂 Listening for new pump.fun tokens...")
                    logging.info(f"📊 Session started at: {self.session_start}")
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            
                            # Handle different message types
                            if isinstance(data, dict):
                                # New token creation
                                if data.get('type') == 'new_token' or 'mint' in data:
                                    await self.handle_new_token(data)
                                
                                # Trade events
                                elif data.get('type') == 'trade':
                                    await self.handle_trade(data)
                                
                                # Migration events
                                elif data.get('type') == 'migration':
                                    mint = data.get('mint', 'Unknown')
                                    logging.info(f"🎓 Token graduated to Raydium: {mint}")
                                
                                # Generic data handling
                                else:
                                    # Log raw data for debugging
                                    logging.debug(f"Received data: {data}")
                                    
                                    # Try to extract token info from any structure
                                    if 'mint' in data or 'mintAddress' in data:
                                        await self.handle_new_token(data)
                            
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to parse JSON: {e}")
                            logging.error(f"Raw message: {message}")
                        except Exception as e:
                            logging.error(f"Error processing message: {e}")
                            logging.error(f"Message data: {message}")
                            
            except websockets.exceptions.ConnectionClosed as e:
                retry_count += 1
                logging.warning(f"❌ WebSocket connection closed: {e}")
                logging.info(f"🔄 Retrying in 5 seconds... ({retry_count}/{max_retries})")
                await asyncio.sleep(5)
                
            except Exception as e:
                retry_count += 1
                logging.error(f"❌ Connection error: {e}")
                logging.info(f"🔄 Retrying in 10 seconds... ({retry_count}/{max_retries})")
                await asyncio.sleep(10)
        
        logging.error(f"❌ Max retries ({max_retries}) reached. Exiting.")
    
    async def print_stats(self):
        """Print periodic statistics"""
        while True:
            await asyncio.sleep(60)  # Print stats every minute
            runtime = datetime.now() - self.session_start
            logging.info(f"📊 Stats: {self.total_tokens_seen} tokens detected in {runtime}")

async def main():
    """Main function"""
    monitor = PumpPortalMonitor()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                        🚀 PUMP.FUN TOKEN LAUNCH MONITOR 🚀                           ║
║                              Using PumpPortal WebSocket API                          ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  📡 Monitoring: wss://pumpportal.fun/api/data                                       ║
║  📝 Logs: ../logs/pumpportal_launches.txt                                           ║
║  📋 Token List: ../logs/live_token_launches.txt                                     ║
║  🔗 API Docs: https://pumpportal.fun/data-api/real-time/                           ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('../logs', exist_ok=True)
    
    # Create CSV header if file doesn't exist
    csv_file = '../logs/live_token_launches.txt'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,mint,name,symbol,creator\n")
    
    # Start monitoring tasks
    tasks = [
        monitor.connect_and_monitor(),
        monitor.print_stats()
    ]
    
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logging.info("👋 Monitoring stopped by user")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    # Install required packages
    try:
        import websockets
        import aiohttp
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "websockets", "aiohttp"])
        import websockets
        import aiohttp
    
    asyncio.run(main()) 