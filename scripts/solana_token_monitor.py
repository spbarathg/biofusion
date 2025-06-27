#!/usr/bin/env python3
"""
SOLANA TOKEN MONITOR
===================

Monitor new Solana tokens launched on Raydium DEX.
Uses the correct API structure and focuses on SOL pairs only.
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
import os
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class SolanaTokenMonitor:
    """Monitor for new Solana tokens on Raydium"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.session = None
        self.logger = logging.getLogger("SolanaTokenMonitor")
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
    
    async def get_raydium_tokens(self) -> list:
        """Get tokens from Raydium with correct API parsing"""
        
        new_tokens = []
        
        try:
            # Use the correct Raydium API endpoint
            raydium_url = "https://api.raydium.io/v2/main/pairs"
            
            async with self.session.get(raydium_url) as response:
                if response.status == 200:
                    pairs = await response.json()
                    self.logger.info(f"Retrieved {len(pairs)} pairs from Raydium")
                    
                    for pair in pairs:
                        # Focus on SOL pairs only
                        quote_mint = pair.get("quoteMint", "")
                        base_mint = pair.get("baseMint", "")
                        
                        # SOL mint address
                        SOL_MINT = "So11111111111111111111111111111111111111112"
                        
                        # Only process if this is a SOL pair (quote token is SOL)
                        if quote_mint == SOL_MINT and base_mint not in self.known_tokens:
                            
                            # Get token info
                            base_symbol = pair.get("baseSymbol", "UNKNOWN")
                            base_name = pair.get("baseName", "Unknown Token")
                            liquidity = float(pair.get("liquidity", 0))
                            price = float(pair.get("price", 0))
                            
                            # Try different time fields that might exist
                            pool_time = None
                            age_hours = 999  # Default to very old
                            
                            # Check various possible timestamp fields
                            time_fields = ["poolOpenTime", "openTime", "createdAt", "timestamp"]
                            for field in time_fields:
                                if field in pair and pair[field]:
                                    try:
                                        timestamp = int(pair[field])
                                        if timestamp > 1000000000000:  # Milliseconds
                                            timestamp = timestamp / 1000
                                        
                                        pool_time = datetime.fromtimestamp(timestamp)
                                        age_hours = (datetime.now() - pool_time).total_seconds() / 3600
                                        break
                                    except:
                                        continue
                            
                            # Filter for recent tokens (last 24 hours) OR if no timestamp, include anyway
                            if age_hours < 24 or pool_time is None:
                                
                                # Basic quality filters
                                if (base_symbol != "UNKNOWN" and 
                                    len(base_symbol) > 1 and 
                                    liquidity > 1):  # At least $1 liquidity
                                    
                                    token_info = {
                                        "address": base_mint,
                                        "symbol": base_symbol,
                                        "name": base_name,
                                        "price": price,
                                        "liquidity": liquidity,
                                        "platform": "Raydium",
                                        "age_hours": age_hours if pool_time else None,
                                        "pool_open_time": pool_time.isoformat() if pool_time else None,
                                        "quote_token": "SOL"
                                    }
                                    
                                    new_tokens.append(token_info)
                                    self.known_tokens.add(base_mint)
                    
                    # Sort by liquidity (highest first)
                    new_tokens.sort(key=lambda x: x['liquidity'], reverse=True)
                    
                else:
                    self.logger.warning(f"Raydium API returned {response.status}")
        
        except Exception as e:
            self.logger.error(f"Error getting Raydium tokens: {e}")
        
        return new_tokens
    
    async def analyze_solana_token(self, token_info: dict) -> dict:
        """Analyze Solana token for trading potential"""
        
        symbol = token_info.get("symbol", "").lower()
        liquidity = token_info.get("liquidity", 0)
        age_hours = token_info.get("age_hours", 999)
        price = token_info.get("price", 0)
        
        # Scoring system
        score = 0.3  # Base score
        reasons = []
        
        # Age scoring (if available)
        if age_hours is not None:
            if age_hours < 1:  # Less than 1 hour
                score += 0.4
                reasons.append("Very new token")
            elif age_hours < 6:  # Less than 6 hours
                score += 0.2
                reasons.append("Recent token")
            elif age_hours < 24:  # Less than 24 hours
                score += 0.1
                reasons.append("Today's token")
        else:
            score += 0.1
            reasons.append("Unknown age")
        
        # Liquidity scoring
        if liquidity > 1000:
            score += 0.3
            reasons.append("High liquidity")
        elif liquidity > 100:
            score += 0.2
            reasons.append("Good liquidity")
        elif liquidity > 10:
            score += 0.1
            reasons.append("Decent liquidity")
        else:
            score -= 0.2
            reasons.append("Low liquidity")
        
        # Price analysis
        if 0.000001 < price < 10:  # Good price range for memecoins
            score += 0.1
            reasons.append("Good price range")
        
        # Symbol filtering
        good_patterns = ["cat", "dog", "pepe", "moon", "rocket", "inu", "shib"]
        bad_patterns = ["test", "fake", "scam", "rug", "copy"]
        
        if any(pattern in symbol for pattern in good_patterns):
            score += 0.1
            reasons.append("Popular symbol pattern")
        
        if any(pattern in symbol for pattern in bad_patterns):
            score -= 0.4
            reasons.append("Suspicious symbol")
        
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
    
    async def monitor_solana_tokens(self):
        """Main monitoring loop for Solana tokens"""
        
        self.logger.info("🚀 Starting Solana token monitoring...")
        
        while True:
            try:
                # Get new tokens
                new_tokens = await self.get_raydium_tokens()
                
                if new_tokens:
                    self.logger.info(f"🔥 Found {len(new_tokens)} new Solana tokens")
                    
                    for token in new_tokens[:10]:  # Show top 10 by liquidity
                        self.detection_count += 1
                        
                        # Analyze token
                        analysis = await self.analyze_solana_token(token)
                        
                        # Display results
                        print(f"\n🪙 SOLANA TOKEN #{self.detection_count}")
                        print(f"   Symbol: {token['symbol']}")
                        print(f"   Name: {token['name']}")
                        print(f"   Address: {token['address'][:8]}...")
                        print(f"   Price: ${token['price']:.8f} SOL")
                        print(f"   Liquidity: ${token['liquidity']:,.2f}")
                        if token['age_hours']:
                            print(f"   Age: {token['age_hours']:.1f} hours")
                        else:
                            print(f"   Age: Unknown")
                        print(f"   Score: {analysis['score']:.2f}")
                        print(f"   Recommendation: {analysis['recommendation']}")
                        print(f"   Platform: {token['platform']}")
                        
                        # Save detection
                        self.save_detection(token, analysis)
                        
                        # Alert for high-potential tokens
                        if analysis['recommendation'] in ['BUY', 'STRONG_BUY']:
                            print(f"   🎯 HIGH POTENTIAL SOLANA TOKEN!")
                
                else:
                    self.logger.debug("No new Solana tokens detected")
                
                # Wait before next scan
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
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
        
        with open("logs/solana_token_detections.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

async def main():
    """Main function"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    helius_api_key = os.getenv("HELIUS_API_KEY", "193ececa-6e42-4d84-b9bd-765c4813816d")
    
    print("🪙 SOLANA TOKEN MONITOR")
    print("=" * 40)
    print("Monitoring new Solana tokens on Raydium:")
    print("• SOL trading pairs only")
    print("• Real-time liquidity tracking")
    print("• Quality scoring system")
    print("• Recent launches prioritized")
    print("")
    
    async with SolanaTokenMonitor(helius_api_key) as monitor:
        try:
            await monitor.monitor_solana_tokens()
        except KeyboardInterrupt:
            print("\n⏹️ Solana monitor stopped by user")

if __name__ == "__main__":
    asyncio.run(main()) 