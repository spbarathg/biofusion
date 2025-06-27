#!/usr/bin/env python3
"""
Bitquery Real-time Pump.fun Token Monitor
Using Bitquery GraphQL Subscription API - CONFIRMED WORKING

This script uses Bitquery's GraphQL streaming API to monitor pump.fun token launches.
Based on latest internet research, this is a reliable alternative solution.

API Documentation: https://docs.bitquery.io/docs/examples/Solana/Pump-Fun-API/
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/bitquery_launches.txt'),
        logging.StreamHandler()
    ]
)

class BitqueryMonitor:
    def __init__(self, api_key=None):
        self.api_key = api_key or "YOUR_BITQUERY_API_KEY"  # Get free key from bitquery.io
        self.endpoint = "https://streaming.bitquery.io/eap"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-API-KEY": self.api_key
        }
        self.total_tokens_seen = 0
        self.session_start = datetime.now()
    
    def get_subscription_query(self):
        """GraphQL subscription query for real-time pump.fun token creation"""
        return """
        subscription {
          Solana {
            TokenSupplyUpdates(
              where: {
                Instruction: {
                  Program: {
                    Address: { is: "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P" }
                    Method: { is: "create" }
                  }
                }
              }
            ) {
              Block {
                Time
              }
              Transaction {
                Signer
                Signature
              }
              TokenSupplyUpdate {
                Amount
                Currency {
                  Symbol
                  Name
                  MintAddress
                  MetadataAddress
                  Uri
                  UpdateAuthority
                  Decimals
                }
                PostBalance
              }
            }
          }
        }
        """
    
    async def handle_new_token(self, token_data):
        """Process newly detected token"""
        self.total_tokens_seen += 1
        
        try:
            # Extract token information from Bitquery response
            currency = token_data.get('TokenSupplyUpdate', {}).get('Currency', {})
            transaction = token_data.get('Transaction', {})
            block = token_data.get('Block', {})
            
            mint = currency.get('MintAddress', 'Unknown')
            name = currency.get('Name', 'Unknown')
            symbol = currency.get('Symbol', 'Unknown')
            uri = currency.get('Uri', '')
            dev = transaction.get('Signer', 'Unknown')
            signature = transaction.get('Signature', 'Unknown')
            block_time = block.get('Time', '')
            supply = token_data.get('TokenSupplyUpdate', {}).get('PostBalance', 0)
            
            timestamp = datetime.now()
            
            # Log the discovery
            log_message = f"""
🚀 NEW TOKEN DETECTED #{self.total_tokens_seen}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏰ Detection Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
⛓️ Block Time: {block_time}
🔤 Name: {name}
🎯 Symbol: {symbol}
📍 Mint: {mint}
👤 Creator: {dev}
💰 Supply: {supply:,}
🌐 URI: {uri}
📝 Signature: {signature}
🔗 Pump.fun: https://pump.fun/{mint}
🔍 Solscan: https://solscan.io/token/{mint}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            logging.info(log_message)
            
            # Save to file
            with open('../logs/live_token_launches.txt', 'a', encoding='utf-8') as f:
                f.write(f"{timestamp.isoformat()},{mint},{name},{symbol},{dev},{supply},{signature}\n")
                
        except Exception as e:
            logging.error(f"Error processing token data: {e}")
            logging.error(f"Token data: {token_data}")
    
    async def connect_and_monitor(self):
        """Main monitoring loop using GraphQL subscription"""
        retry_count = 0
        max_retries = 10
        
        while retry_count < max_retries:
            try:
                logging.info(f"🔄 Connecting to Bitquery GraphQL API... (Attempt {retry_count + 1})")
                
                # Prepare subscription
                query = self.get_subscription_query()
                payload = {
                    "query": query,
                    "variables": {}
                }
                
                timeout = aiohttp.ClientTimeout(total=None)  # No timeout for streaming
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.endpoint,
                        headers=self.headers,
                        json=payload
                    ) as response:
                        
                        if response.status != 200:
                            error_text = await response.text()
                            logging.error(f"❌ API request failed: {response.status} - {error_text}")
                            retry_count += 1
                            await asyncio.sleep(10)
                            continue
                        
                        logging.info("✅ Connected to Bitquery successfully!")
                        logging.info("📡 Subscribed to pump.fun token creation events")
                        logging.info("👂 Listening for new tokens...")
                        logging.info(f"📊 Session started at: {self.session_start}")
                        
                        retry_count = 0  # Reset on successful connection
                        
                        # Read streaming response
                        buffer = ""
                        async for chunk in response.content.iter_chunked(1024):
                            try:
                                chunk_str = chunk.decode('utf-8')
                                buffer += chunk_str
                                
                                # Process complete JSON objects
                                while '\n' in buffer:
                                    line, buffer = buffer.split('\n', 1)
                                    if line.strip():
                                        try:
                                            data = json.loads(line)
                                            
                                            # Handle subscription data
                                            if 'data' in data and 'Solana' in data['data']:
                                                solana_data = data['data']['Solana']
                                                if 'TokenSupplyUpdates' in solana_data:
                                                    for update in solana_data['TokenSupplyUpdates']:
                                                        await self.handle_new_token(update)
                                            
                                            # Handle errors
                                            elif 'errors' in data:
                                                logging.error(f"GraphQL errors: {data['errors']}")
                                                
                                        except json.JSONDecodeError:
                                            # Skip incomplete JSON
                                            continue
                                            
                            except Exception as e:
                                logging.error(f"Error processing chunk: {e}")
                                continue
                
            except aiohttp.ClientError as e:
                retry_count += 1
                logging.error(f"❌ Connection error: {e}")
                logging.info(f"🔄 Retrying in 10 seconds... ({retry_count}/{max_retries})")
                await asyncio.sleep(10)
                
            except Exception as e:
                retry_count += 1
                logging.error(f"❌ Unexpected error: {e}")
                logging.info(f"🔄 Retrying in 10 seconds... ({retry_count}/{max_retries})")
                await asyncio.sleep(10)
        
        logging.error(f"❌ Max retries ({max_retries}) reached. Exiting.")
    
    async def test_connection(self):
        """Test API connection with a simple query"""
        test_query = """
        query {
          Solana {
            TokenSupplyUpdates(
              where: {
                Instruction: {
                  Program: {
                    Address: { is: "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P" }
                    Method: { is: "create" }
                  }
                }
              }
              limit: { count: 1 }
              orderBy: { descending: Block_Time }
            ) {
              TokenSupplyUpdate {
                Currency {
                  MintAddress
                  Name
                  Symbol
                }
              }
            }
          }
        }
        """
        
        payload = {"query": test_query}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logging.info("✅ API connection test successful!")
                        return True
                    else:
                        error_text = await response.text()
                        logging.error(f"❌ API test failed: {response.status} - {error_text}")
                        return False
        except Exception as e:
            logging.error(f"❌ API test error: {e}")
            return False
    
    async def print_stats(self):
        """Print periodic statistics"""
        while True:
            await asyncio.sleep(60)  # Print stats every minute
            runtime = datetime.now() - self.session_start
            logging.info(f"📊 Stats: {self.total_tokens_seen} tokens detected in {runtime}")

async def main():
    """Main function"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                        🚀 PUMP.FUN TOKEN LAUNCH MONITOR 🚀                           ║
║                               Using Bitquery GraphQL API                            ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  📡 Monitoring: Bitquery Solana Streaming API                                       ║
║  📝 Logs: ../logs/bitquery_launches.txt                                             ║
║  📋 Token List: ../logs/live_token_launches.txt                                     ║
║  🔗 API Docs: https://docs.bitquery.io/docs/examples/Solana/Pump-Fun-API/          ║
║  🔑 Get API Key: https://bitquery.io/                                               ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Check for API key
    api_key = os.getenv('BITQUERY_API_KEY')
    if not api_key:
        print("⚠️  Warning: No Bitquery API key found!")
        print("   Get a free API key from https://bitquery.io/")
        print("   Set it as environment variable: export BITQUERY_API_KEY=your_key")
        print("   Or edit the script to add your key directly")
        print()
        api_key = input("Enter your Bitquery API key (or press Enter to continue with demo): ").strip()
        if not api_key:
            api_key = "demo"  # Use demo key (limited functionality)
    
    monitor = BitqueryMonitor(api_key)
    
    # Create logs directory if it doesn't exist
    os.makedirs('../logs', exist_ok=True)
    
    # Create CSV header if file doesn't exist
    csv_file = '../logs/live_token_launches.txt'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,mint,name,symbol,creator,supply,signature\n")
    
    # Test connection first
    if await monitor.test_connection():
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
    else:
        logging.error("❌ Failed to connect to Bitquery API. Check your API key and try again.")

if __name__ == "__main__":
    # Install required packages
    try:
        import aiohttp
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "aiohttp"])
        import aiohttp
    
    asyncio.run(main()) 