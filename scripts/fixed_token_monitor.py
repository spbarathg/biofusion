#!/usr/bin/env python3
"""
Fixed Pump.fun Token Monitor
Simplified version that focuses on reliable token detection and mint extraction
"""

import asyncio
import websockets
import aiohttp
import json
import logging
from datetime import datetime
import os
import sys

# Configure logging for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/fixed_monitor.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

class FixedTokenMonitor:
    def __init__(self, api_key="193ececa-6e42-4d84-b9bd-765c4813816d"):
        self.api_key = api_key
        self.ws_url = f"wss://mainnet.helius-rpc.com/?api-key={api_key}"
        self.http_url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        self.pump_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
        self.total_tokens_seen = 0
        self.session_start = datetime.now()
        self.seen_signatures = set()
    
    async def handle_new_token(self, log_data):
        """Process newly detected token"""
        try:
            signature = log_data.get('signature', 'Unknown')
            
            if signature in self.seen_signatures:
                return
            self.seen_signatures.add(signature)
            
            self.total_tokens_seen += 1
            timestamp = datetime.now()
            
            logging.info(f"[{self.total_tokens_seen}] Processing signature: {signature}")
            
            # Get transaction details with retry logic
            token_info = await self.get_transaction_details_with_retry(signature)
            mint = token_info.get('mint', 'Unknown')
            creator = token_info.get('creator', 'Unknown')
            
            if mint and mint != 'Unknown' and len(mint) > 20:
                # Get basic metadata
                metadata = await self.get_token_metadata(mint)
                
                # Display token info
                await self.display_token_info({
                    'number': self.total_tokens_seen,
                    'timestamp': timestamp,
                    'mint': mint,
                    'signature': signature,
                    'creator': creator,
                    'name': metadata.get('name', 'Unknown'),
                    'symbol': metadata.get('symbol', 'Unknown'),
                    'price': metadata.get('price_usd', 'Unknown')
                })
                
                # Save to file
                await self.save_token_data(timestamp, mint, signature, creator, metadata)
            else:
                logging.warning(f"Could not extract valid mint from signature: {signature}")
                
        except Exception as e:
            logging.error(f"Error processing token: {e}")
    
    async def get_transaction_details_with_retry(self, signature, max_retries=3):
        """Get transaction details with retry logic"""
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(0.5 * attempt)  # Progressive delay
                return await self.get_transaction_details(signature)
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed to get transaction after {max_retries} attempts: {e}")
                    return {'mint': 'Unknown', 'creator': 'Unknown'}
                logging.debug(f"Attempt {attempt + 1} failed, retrying...")
    
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
                        "maxSupportedTransactionVersion": 0
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.http_url, json=request_data, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'result' in data and data['result']:
                            return await self.parse_transaction_data(data['result'])
                        else:
                            logging.warning(f"No transaction data for {signature}")
                        
        except Exception as e:
            logging.debug(f"Error getting transaction details: {e}")
            
        return {'mint': 'Unknown', 'creator': 'Unknown'}
    
    async def parse_transaction_data(self, transaction):
        """Parse transaction to extract mint and creator"""
        try:
            result = {'mint': 'Unknown', 'creator': 'Unknown'}
            
            # Get account keys
            if 'transaction' in transaction and 'message' in transaction['transaction']:
                message = transaction['transaction']['message']
                account_keys = message.get('accountKeys', [])
                
                # Get creator (first signer)
                if account_keys:
                    result['creator'] = account_keys[0]
                
                # Method 1: Look for new token mint in postTokenBalances
                if 'meta' in transaction and 'postTokenBalances' in transaction['meta']:
                    post_balances = transaction['meta']['postTokenBalances']
                    
                    # Look for newly created tokens (high account index)
                    for balance in post_balances:
                        if 'mint' in balance and 'accountIndex' in balance:
                            mint = balance['mint']
                            if len(mint) >= 32:  # Valid Solana address
                                result['mint'] = mint
                                logging.debug(f"Found mint via postTokenBalances: {mint}")
                                return result
                
                # Method 2: Find mint by comparing pre vs post token balances
                if 'meta' in transaction:
                    pre_balances = transaction['meta'].get('preTokenBalances', [])
                    post_balances = transaction['meta'].get('postTokenBalances', [])
                    
                    pre_mints = {b['mint'] for b in pre_balances if 'mint' in b}
                    post_mints = {b['mint'] for b in post_balances if 'mint' in b}
                    
                    new_mints = post_mints - pre_mints
                    if new_mints:
                        mint = list(new_mints)[0]
                        result['mint'] = mint
                        logging.debug(f"Found mint via balance difference: {mint}")
                        return result
                
                # Method 3: Look through instructions for token creation
                instructions = message.get('instructions', [])
                for instruction in instructions:
                    program_id_index = instruction.get('programIdIndex')
                    if program_id_index is not None and program_id_index < len(account_keys):
                        program_id = account_keys[program_id_index]
                        
                        # Check if it's the token program
                        if program_id in ['TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA']:
                            accounts = instruction.get('accounts', [])
                            if accounts:
                                mint_index = accounts[0]
                                if mint_index < len(account_keys):
                                    mint = account_keys[mint_index]
                                    if len(mint) >= 32:
                                        result['mint'] = mint
                                        logging.debug(f"Found mint via instructions: {mint}")
                                        return result
            
            return result
            
        except Exception as e:
            logging.debug(f"Error parsing transaction: {e}")
            return {'mint': 'Unknown', 'creator': 'Unknown'}
    
    async def get_token_metadata(self, mint_address):
        """Get basic token metadata"""
        try:
            # Try Helius enhanced API first
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
                                'price_usd': 'Unknown'  # We'll add price APIs later
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
            logging.info(f"Token #{token_data['number']}: {token_data['name']} ({token_data['symbol']}) - {token_data['mint']}")
            
        except Exception as e:
            logging.error(f"Error displaying token info: {e}")
    
    async def save_token_data(self, timestamp, mint, signature, creator, metadata):
        """Save token data to file"""
        try:
            # Save to simple CSV format
            csv_line = f"{timestamp.isoformat()},{mint},{metadata.get('name', 'Unknown')},{metadata.get('symbol', 'Unknown')},{creator},{signature}\n"
            
            with open('../logs/fixed_token_launches.csv', 'a', encoding='utf-8') as f:
                f.write(csv_line)
                
        except Exception as e:
            logging.error(f"Error saving token data: {e}")
    
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
                                    
                                    # Simple filter for potential token creation
                                    is_creation = any(
                                        'create' in log.lower() or 
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
                logging.info(f"Retrying in 5 seconds... ({retry_count}/{max_retries})")
                await asyncio.sleep(5)
        
        logging.error(f"Max retries reached. Exiting.")
    
    async def print_stats(self):
        """Print periodic statistics"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            runtime = datetime.now() - self.session_start
            rate = self.total_tokens_seen / (runtime.total_seconds() / 60) if runtime.total_seconds() > 0 else 0
            logging.info(f"STATS: {self.total_tokens_seen} tokens detected in {runtime} ({rate:.2f}/min)")

async def main():
    """Main function"""
    print("""
===============================================================================
                    FIXED PUMP.FUN TOKEN MONITOR
===============================================================================
Features:
- Reliable real-time token detection
- Improved mint address extraction
- Basic metadata retrieval
- Windows console compatibility
- CSV data export

Output: ../logs/fixed_token_launches.csv
===============================================================================
""")
    
    # Create logs directory
    os.makedirs('../logs', exist_ok=True)
    
    # Create CSV header if needed
    csv_file = '../logs/fixed_token_launches.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,mint,name,symbol,creator,signature\n")
    
    monitor = FixedTokenMonitor()
    
    print("Starting fixed monitor...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        tasks = [
            monitor.monitor_pump_fun(),
            monitor.print_stats()
        ]
        
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 