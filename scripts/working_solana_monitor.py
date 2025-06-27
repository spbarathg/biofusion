#!/usr/bin/env python3
"""
WORKING SOLANA MONITOR
=====================

Working Solana token monitor based on actual Raydium API structure.
Uses token metadata from your Helius RPC to get symbol information.
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
import os
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class WorkingSolanaMonitor:
    """Working Solana token monitor using Helius RPC for metadata"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.helius_url = f"https://mainnet.helius-rpc.com/?api-key={helius_api_key}"
        self.session = None
        self.logger = logging.getLogger("WorkingSolanaMonitor")
        self.known_tokens = set()
        self.detection_count = 0
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            resolver=aiohttp.AsyncResolver(),
            use_dns_cache=False
        )
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Content-Type": "application/json"},
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_token_metadata(self, mint_address: str) -> dict:
        """Get token metadata from Helius RPC"""
        
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [
                    mint_address,
                    {"encoding": "jsonParsed"}
                ]
            }
            
            async with self.session.post(self.helius_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "result" in data and data["result"] and data["result"]["value"]:
                        account_data = data["result"]["value"]["data"]
                        
                        if account_data and "parsed" in account_data:
                            info = account_data["parsed"]["info"]
                            
                            # Try to get symbol from extensions
                            extensions = info.get("extensions", [])
                            symbol = "UNKNOWN"
                            name = "Unknown Token"
                            
                            for ext in extensions:
                                if ext.get("extension") == "tokenMetadata":
                                    metadata = ext.get("state", {})
                                    symbol = metadata.get("symbol", "UNKNOWN")
                                    name = metadata.get("name", "Unknown Token")
                                    break
                            
                            return {
                                "symbol": symbol,
                                "name": name,
                                "decimals": info.get("decimals", 9),
                                "supply": int(info.get("supply", 0))
                            }
        
        except Exception as e:
            self.logger.debug(f"Error getting metadata for {mint_address}: {e}")
        
        return {
            "symbol": "UNKNOWN",
            "name": "Unknown Token",
            "decimals": 9,
            "supply": 0
        }
    
    async def get_solana_pairs(self) -> list:
        """Get SOL trading pairs from Raydium"""
        
        new_tokens = []
        
        try:
            raydium_url = "https://api.raydium.io/v2/main/pairs"
            
            async with self.session.get(raydium_url) as response:
                if response.status == 200:
                    pairs = await response.json()
                    self.logger.info(f"Retrieved {len(pairs)} pairs from Raydium")
                    
                    # SOL mint address
                    SOL_MINT = "So11111111111111111111111111111111111111112"
                    
                    sol_pairs = []
                    
                    # Find SOL pairs
                    for pair in pairs:
                        quote_mint = pair.get("quoteMint", "")
                        base_mint = pair.get("baseMint", "")
                        
                        # Check if this is a SOL pair
                        if quote_mint == SOL_MINT and base_mint not in self.known_tokens:
                            liquidity = float(pair.get("liquidity", 0))
                            
                            # Filter by minimum liquidity
                            if liquidity > 50:  # At least $50 liquidity
                                sol_pairs.append({
                                    "base_mint": base_mint,
                                    "quote_mint": quote_mint,
                                    "liquidity": liquidity,
                                    "price": float(pair.get("price", 0)),
                                    "volume24h": float(pair.get("volume24h", 0)),
                                    "pair_name": pair.get("name", "Unknown"),
                                    "amm_id": pair.get("ammId", "")
                                })
                    
                    # Sort by liquidity (highest first)
                    sol_pairs.sort(key=lambda x: x['liquidity'], reverse=True)
                    
                    # Process top pairs and get metadata
                    for pair in sol_pairs[:20]:  # Top 20 by liquidity
                        
                        # Get token metadata
                        metadata = await self.get_token_metadata(pair["base_mint"])
                        
                        token_info = {
                            "address": pair["base_mint"],
                            "symbol": metadata["symbol"],
                            "name": metadata["name"],
                            "decimals": metadata["decimals"],
                            "supply": metadata["supply"],
                            "price": pair["price"],
                            "liquidity": pair["liquidity"],
                            "volume24h": pair["volume24h"],
                            "platform": "Raydium",
                            "quote_token": "SOL",
                            "pair_name": pair["pair_name"],
                            "amm_id": pair["amm_id"]
                        }
                        
                        new_tokens.append(token_info)
                        self.known_tokens.add(pair["base_mint"])
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.1)
                
                else:
                    self.logger.warning(f"Raydium API returned {response.status}")
        
        except Exception as e:
            self.logger.error(f"Error getting SOL pairs: {e}")
        
        return new_tokens
    
    async def analyze_token(self, token_info: dict) -> dict:
        """Analyze token for trading potential"""
        
        symbol = token_info.get("symbol", "").lower()
        liquidity = token_info.get("liquidity", 0)
        volume24h = token_info.get("volume24h", 0)
        price = token_info.get("price", 0)
        supply = token_info.get("supply", 0)
        
        # Scoring system
        score = 0.4  # Base score
        reasons = []
        
        # Liquidity scoring
        if liquidity > 10000:
            score += 0.3
            reasons.append("Very high liquidity")
        elif liquidity > 1000:
            score += 0.2
            reasons.append("High liquidity")
        elif liquidity > 100:
            score += 0.1
            reasons.append("Good liquidity")
        
        # Volume scoring
        if volume24h > 10000:
            score += 0.2
            reasons.append("High volume")
        elif volume24h > 1000:
            score += 0.1
            reasons.append("Good volume")
        
        # Price analysis
        if 0.000001 < price < 1:
            score += 0.1
            reasons.append("Good price range")
        
        # Symbol analysis
        good_patterns = ["cat", "dog", "pepe", "moon", "rocket", "inu", "shib", "bonk"]
        bad_patterns = ["test", "fake", "scam", "rug", "copy"]
        
        if any(pattern in symbol for pattern in good_patterns):
            score += 0.1
            reasons.append("Popular meme pattern")
        
        if any(pattern in symbol for pattern in bad_patterns):
            score -= 0.3
            reasons.append("Suspicious name")
        
        # SOL pair bonus
        score += 0.1
        reasons.append("SOL trading pair")
        
        # Generate recommendation
        if score > 0.8:
            recommendation = "STRONG_BUY"
        elif score > 0.6:
            recommendation = "BUY"
        elif score > 0.4:
            recommendation = "WATCH"
        else:
            recommendation = "SKIP"
        
        return {
            "score": min(max(score, 0.0), 1.0),
            "recommendation": recommendation,
            "reasons": reasons
        }
    
    async def run_scan(self):
        """Run one scan cycle"""
        
        print(f"\n🔍 SCANNING SOLANA TOKENS - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
        
        # Get SOL pairs
        tokens = await self.get_solana_pairs()
        
        if tokens:
            print(f"🔥 Found {len(tokens)} SOL trading pairs")
            
            for i, token in enumerate(tokens, 1):
                self.detection_count += 1
                
                # Analyze token
                analysis = await self.analyze_token(token)
                
                # Display results
                print(f"\n🪙 TOKEN #{i}")
                print(f"   Symbol: {token['symbol']}")
                print(f"   Name: {token['name']}")
                print(f"   Address: {token['address'][:8]}...{token['address'][-8:]}")
                print(f"   Price: {token['price']:.8f} SOL")
                print(f"   Liquidity: ${token['liquidity']:,.2f}")
                print(f"   Volume 24h: ${token['volume24h']:,.2f}")
                print(f"   Score: {analysis['score']:.2f}")
                print(f"   Recommendation: {analysis['recommendation']}")
                
                # Save to log
                self.save_detection(token, analysis)
                
                # Highlight high-potential tokens
                if analysis['recommendation'] in ['BUY', 'STRONG_BUY']:
                    print(f"   🎯 HIGH POTENTIAL!")
        
        else:
            print("💤 No SOL pairs found this scan")
    
    def save_detection(self, token_info: dict, analysis: dict):
        """Save detection to log file"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "detection_number": self.detection_count,
            "token_info": token_info,
            "analysis": analysis,
            "blockchain": "Solana"
        }
        
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/working_solana_detections.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

async def main():
    """Main function"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    helius_api_key = os.getenv("HELIUS_API_KEY", "193ececa-6e42-4d84-b9bd-765c4813816d")
    
    print("🪙 WORKING SOLANA TOKEN MONITOR")
    print("=" * 40)
    print("Features:")
    print("• SOL trading pairs from Raydium")
    print("• Token metadata from Helius RPC")
    print("• Liquidity and volume analysis")
    print("• Quality scoring system")
    print("")
    
    async with WorkingSolanaMonitor(helius_api_key) as monitor:
        try:
            # Run one scan to see results
            await monitor.run_scan()
            
        except KeyboardInterrupt:
            print("\n⏹️ Monitor stopped by user")

if __name__ == "__main__":
    asyncio.run(main()) 