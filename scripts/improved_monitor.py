#!/usr/bin/env python3
"""
Improved Pump.fun Token Monitor
Version with better transaction timing and error handling
"""

import asyncio
import websockets
import aiohttp
import json
import logging
from datetime import datetime
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/improved_monitor.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

class ImprovedTokenMonitor:
    def __init__(self, api_key="193ececa-6e42-4d84-b9bd-765c4813816d"):
        self.api_key = api_key
        self.ws_url = f"wss://mainnet.helius-rpc.com/?api-key={api_key}"
        self.http_url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        self.pump_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        self.total_tokens_seen = 0
        self.session_start = datetime.now()
        self.seen_signatures = set()
        
        # Queue for processing transactions with delay
        self.processing_queue = asyncio.Queue()
    
    async def handle_new_token(self, log_data):
        """Queue token for processing with delay"""
        try:
            signature = log_data.get('signature', 'Unknown')
            
            if signature in self.seen_signatures:
                return
            self.seen_signatures.add(signature)
            
            self.total_tokens_seen += 1
            timestamp = datetime.now()
            
            logging.info(f"[{self.total_tokens_seen}] Queuing signature: {signature}")
            
            # Add to processing queue with delay
            await self.processing_queue.put({
                'signature': signature,
                'timestamp': timestamp,
                'number': self.total_tokens_seen
            })
                
        except Exception as e:
            logging.error(f"Error handling token: {e}")
    
    async def process_queued_tokens(self):
        """Process tokens from queue with proper timing"""
        while True:
            try:
                # Wait for token to be queued
                token_data = await self.processing_queue.get()
                
                # Wait a bit for transaction to be fully processed
                await asyncio.sleep(2)
                
                signature = token_data['signature']
                logging.info(f"[{token_data['number']}] Processing: {signature}")
                
                # Get transaction details with multiple attempts
                token_info = await self.get_transaction_with_retries(signature)
                mint = token_info.get('mint', 'Unknown')
                creator = token_info.get('creator', 'Unknown')
                
                if mint and mint != 'Unknown' and len(mint) >= 32:
                    # Get metadata
                    metadata = await self.get_token_metadata(mint)
                    
                    # Display and save
                    display_data = {
                        'number': token_data['number'],
                        'timestamp': token_data['timestamp'],
                        'mint': mint,
                        'signature': signature,
                        'creator': creator,
                        'name': metadata.get('name', 'Unknown'),
                        'symbol': metadata.get('symbol', 'Unknown'),
                        'price': metadata.get('price_usd', 'Unknown')
                    }
                    
                    await self.display_token_info(display_data)
                    await self.save_token_data(display_data)
                    
                else:
                    logging.warning(f"Could not extract valid mint from: {signature}")
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                logging.error(f"Error processing queued token: {e}")
    
    async def get_transaction_with_retries(self, signature, max_retries=5):
        """Get transaction with multiple retries and progressive delays"""
        for attempt in range(max_retries):
            try:
                # Progressive delay: 1s, 2s, 3s, 4s, 5s
                if attempt > 0:
                    await asyncio.sleep(attempt + 1)
                
                result = await self.get_transaction_details(signature)
                
                if result['mint'] != 'Unknown':
                    return result
                elif attempt < max_retries - 1:
                    logging.debug(f"Attempt {attempt + 1}: No mint found, retrying...")
                    
            except Exception as e:
                logging.debug(f"Attempt {attempt + 1} failed: {e}")
        
        logging.warning(f"Failed to get transaction data after {max_retries} attempts: {signature}")
        return {'mint': 'Unknown', 'creator': 'Unknown'}
    
    async def get_transaction_details(self, signature):
        """Get transaction details from Helius"""
        try:
            request_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    signature,
                    {
                        "encoding": "json",
                        "maxSupportedTransactionVersion": 0,
                        "commitment": "confirmed"
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.http_url, json=request_data, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'result' in data and data['result']:
                            return await self.parse_transaction_data(data['result'])
                        else:
                            return {'mint': 'Unknown', 'creator': 'Unknown'}
                        
        except Exception as e:
            logging.debug(f"Error getting transaction: {e}")
            return {'mint': 'Unknown', 'creator': 'Unknown'}
    
    async def parse_transaction_data(self, transaction):
        """Parse transaction to extract mint and creator"""
        try:
            result = {'mint': 'Unknown', 'creator': 'Unknown'}
            
            if 'transaction' in transaction and 'message' in transaction['transaction']:
                message = transaction['transaction']['message']
                account_keys = message.get('accountKeys', [])
                
                # Get creator
                if account_keys:
                    result['creator'] = account_keys[0]
                
                # Method 1: Check postTokenBalances for new mints
                if 'meta' in transaction and 'postTokenBalances' in transaction['meta']:
                    post_balances = transaction['meta']['postTokenBalances']
                    
                    # Look for valid mint addresses
                    for balance in post_balances:
                        if 'mint' in balance:
                            mint = balance['mint']
                            # Check if it's a valid Solana address (32 chars, base58)
                            if len(mint) >= 32 and mint != 'So11111111111111111111111111111111111111112':  # Not SOL
                                result['mint'] = mint
                                logging.debug(f"Found mint: {mint}")
                                return result
                
                # Method 2: Look for differences in token balances
                if 'meta' in transaction:
                    pre_balances = transaction['meta'].get('preTokenBalances', [])
                    post_balances = transaction['meta'].get('postTokenBalances', [])
                    
                    pre_mints = {b['mint'] for b in pre_balances if 'mint' in b}
                    post_mints = {b['mint'] for b in post_balances if 'mint' in b}
                    
                    new_mints = post_mints - pre_mints
                    if new_mints:
                        mint = next(iter(new_mints))
                        if len(mint) >= 32:
                            result['mint'] = mint
                            logging.debug(f"Found new mint: {mint}")
                            return result
                
                # Method 3: Look through instructions for token program calls
                instructions = message.get('instructions', [])
                for instruction in instructions:
                    program_id_index = instruction.get('programIdIndex')
                    if program_id_index is not None and program_id_index < len(account_keys):
                        program_id = account_keys[program_id_index]
                        
                        # Token program
                        if program_id == 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA':
                            accounts = instruction.get('accounts', [])
                            if accounts:
                                mint_index = accounts[0]
                                if mint_index < len(account_keys):
                                    mint = account_keys[mint_index]
                                    if len(mint) >= 32:
                                        result['mint'] = mint
                                        logging.debug(f"Found mint via instruction: {mint}")
                                        return result
            
            return result
            
        except Exception as e:
            logging.debug(f"Error parsing transaction: {e}")
            return {'mint': 'Unknown', 'creator': 'Unknown'}
    
    async def get_token_metadata(self, mint_address):
        """Get token metadata from Helius"""
        try:
            request_data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAsset",
                "params": [mint_address]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.http_url, json=request_data, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'result' in data and data['result']:
                            asset = data['result']
                            content = asset.get('content', {})
                            metadata = content.get('metadata', {})
                            
                            return {
                                'name': metadata.get('name', 'Unknown'),
                                'symbol': metadata.get('symbol', 'Unknown'),
                                'price_usd': 'Unknown'
                            }
        except Exception as e:
            logging.debug(f"Error getting metadata: {e}")
        
        return {'name': 'Unknown', 'symbol': 'Unknown', 'price_usd': 'Unknown'}
    
    async def display_token_info(self, token_data):
        """Display token information"""
        try:
            display_text = f"""
===============================================================================
                    NEW PUMP.FUN TOKEN #{token_data['number']}
===============================================================================
Time: {token_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Name: {token_data['name']}
Symbol: {token_data['symbol']}
Price: ${token_data['price']}

Mint: {token_data['mint']}
Creator: {token_data['creator']}
Signature: {token_data['signature']}

Links:
- Pump.fun: https://pump.fun/{token_data['mint']}
- Solscan: https://solscan.io/token/{token_data['mint']}
- DexScreener: https://dexscreener.com/solana/{token_data['mint']}
===============================================================================
"""
            
            print(display_text)
            
        except Exception as e:
            logging.error(f"Error displaying token: {e}")
    
    async def save_token_data(self, token_data):
        """Save token data"""
        try:
            csv_line = f"{token_data['timestamp'].isoformat()},{token_data['mint']},{token_data['name']},{token_data['symbol']},{token_data['creator']},{token_data['signature']}\n"
            
            with open('../logs/improved_token_launches.csv', 'a', encoding='utf-8') as f:
                f.write(csv_line)
                
        except Exception as e:
            logging.error(f"Error saving data: {e}")
    
    async def monitor_pump_fun(self):
        """Main monitoring loop"""
        retry_count = 0
        max_retries = 10
        
        while retry_count < max_retries:
            try:
                logging.info(f"Connecting to Helius WebSocket... (Attempt {retry_count + 1})")
                
                async with websockets.connect(self.ws_url, ping_interval=30) as websocket:
                    logging.info("Connected successfully!")
                    
                    # Subscribe to pump.fun program logs
                    subscribe_request = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "logsSubscribe",
                        "params": [
                            {"mentions": [self.pump_program_id]},
                            {"commitment": "confirmed"}
                        ]
                    }
                    
                    await websocket.send(json.dumps(subscribe_request))
                    logging.info("Subscribed to pump.fun program logs")
                    logging.info("Listening for new token creations...")
                    
                    retry_count = 0
                    
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            
                            if 'params' in data and 'result' in data['params']:
                                result = data['params']['result']
                                
                                if 'value' in result:
                                    value = result['value']
                                    logs = value.get('logs', [])
                                    
                                    # Look for token creation indicators
                                    is_creation = any(
                                        'Program 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P invoke' in log or
                                        'CreateToken' in log or
                                        'initialize' in log.lower()
                                        for log in logs
                                    )
                                    
                                    if is_creation:
                                        await self.handle_new_token(value)
                            
                            elif 'result' in data and data.get('id') == 1:
                                logging.info(f"Subscription confirmed with ID: {data['result']}")
                                
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logging.error(f"Error processing message: {e}")
                            continue
                
            except Exception as e:
                retry_count += 1
                logging.error(f"Connection error: {e}")
                await asyncio.sleep(5)
        
        logging.error("Max retries reached")
    
    async def print_stats(self):
        """Print periodic statistics"""
        while True:
            await asyncio.sleep(60)  # Every minute
            runtime = datetime.now() - self.session_start
            rate = self.total_tokens_seen / (runtime.total_seconds() / 60) if runtime.total_seconds() > 0 else 0
            logging.info(f"STATS: {self.total_tokens_seen} tokens detected, queue size: {self.processing_queue.qsize()}, rate: {rate:.2f}/min")

async def main():
    """Main function"""
    print("""
===============================================================================
                    IMPROVED PUMP.FUN TOKEN MONITOR
===============================================================================
Features:
- Better transaction timing and retry logic
- Queue-based processing to handle delays
- Improved mint address extraction
- Windows console compatibility

Output: ../logs/improved_token_launches.csv
===============================================================================
""")
    
    # Create logs directory
    os.makedirs('../logs', exist_ok=True)
    
    # Create CSV header
    csv_file = '../logs/improved_token_launches.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,mint,name,symbol,creator,signature\n")
    
    monitor = ImprovedTokenMonitor()
    
    print("Starting improved monitor...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        tasks = [
            monitor.monitor_pump_fun(),
            monitor.process_queued_tokens(),
            monitor.print_stats()
        ]
        
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 