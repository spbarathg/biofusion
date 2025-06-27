#!/usr/bin/env python3
"""
LIVE TOKEN MONITOR
=================

Real-time monitor showing new Solana tokens being launched RIGHT NOW.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import sys

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class LiveTokenMonitor:
    """Live token monitor for new launches"""
    
    def __init__(self):
        self.session = None
        self.known_tokens = set()
        self.launch_count = 0
        self.start_time = datetime.now()
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            resolver=aiohttp.AsyncResolver(),
            use_dns_cache=False
        )
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20),
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scan_new_tokens(self):
        """Scan for new token launches"""
        
        try:
            raydium_url = "https://api.raydium.io/v2/main/pairs"
            
            async with self.session.get(raydium_url) as response:
                if response.status == 200:
                    pairs = await response.json()
                    
                    # SOL mint address
                    SOL_MINT = "So11111111111111111111111111111111111111112"
                    
                    new_tokens = []
                    
                    for pair in pairs:
                        quote_mint = pair.get("quoteMint", "")
                        base_mint = pair.get("baseMint", "")
                        
                        # Only SOL pairs we haven't seen before
                        if (quote_mint == SOL_MINT and 
                            base_mint not in self.known_tokens):
                            
                            liquidity = float(pair.get("liquidity", 0))
                            volume24h = float(pair.get("volume24h", 0))
                            price = float(pair.get("price", 0))
                            
                            # Filter for interesting new tokens
                            if (liquidity > 50 and  # At least $50 liquidity
                                price > 0):  # Valid price
                                
                                new_tokens.append({
                                    "address": base_mint,
                                    "liquidity": liquidity,
                                    "price": price,
                                    "volume24h": volume24h,
                                    "name": pair.get("name", "Unknown"),
                                    "amm_id": pair.get("ammId", "")
                                })
                                
                                self.known_tokens.add(base_mint)
                    
                    return new_tokens
                
                else:
                    print(f"⚠️ API returned {response.status}")
                    return []
        
        except Exception as e:
            print(f"❌ Scan error: {e}")
            return []
    
    async def display_new_token(self, token):
        """Display a newly detected token"""
        
        self.launch_count += 1
        
        # Calculate potential score
        liquidity = token['liquidity']
        price = token['price']
        volume = token['volume24h']
        
        # Simple scoring
        score = 0.5
        alerts = []
        
        if liquidity > 10000:
            score += 0.3
            alerts.append("HIGH LIQUIDITY")
        elif liquidity > 1000:
            score += 0.2
            alerts.append("GOOD LIQUIDITY")
        
        if volume > 1000:
            score += 0.2
            alerts.append("HIGH VOLUME")
        
        if 0.000001 < price < 0.01:
            score += 0.2
            alerts.append("LOW PRICE GEM")
        
        # Display format
        alert_text = " | ".join(alerts) if alerts else ""
        potential = "🔥 HOT" if score > 0.8 else "⭐ GOOD" if score > 0.6 else "👀 WATCH"
        
        print(f"\n🚀 NEW LAUNCH #{self.launch_count} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"   FULL ADDRESS: {token['address']}")
        print(f"   Price: {token['price']:.10f} SOL")
        print(f"   Liquidity: ${token['liquidity']:,.0f}")
        print(f"   Volume 24h: ${token['volume24h']:,.0f}")
        print(f"   Potential: {potential} ({score:.2f})")
        if alert_text:
            print(f"   🚨 {alert_text}")
        print(f"   ⏰ Runtime: {(datetime.now() - self.start_time).total_seconds():.0f}s")
        print(f"   🔗 Verify: https://solscan.io/token/{token['address']}")
        
        # Save to file for verification
        self.save_token_address(token)
    
    def save_token_address(self, token):
        """Save token address to file for verification"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        with open("logs/live_token_addresses.txt", "a") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{timestamp} | {token['address']} | ${token['liquidity']:,.0f} | {token['price']:.10f} SOL\n")
    
    async def monitor_live(self):
        """Main live monitoring loop"""
        
        print("🔴 LIVE TOKEN LAUNCH MONITOR")
        print("=" * 50)
        print("🚀 Monitoring Solana token launches in REAL-TIME")
        print("📊 Showing new SOL pairs as they appear")
        print("⚡ Scanning every 10 seconds")
        print("💎 Focus on potential gems")
        print("")
        
        cycle = 0
        
        while True:
            try:
                cycle += 1
                scan_time = datetime.now().strftime('%H:%M:%S')
                
                # Show scanning status
                if cycle % 6 == 1:  # Every minute
                    runtime = (datetime.now() - self.start_time).total_seconds()
                    print(f"📡 SCAN #{cycle} - {scan_time} | Runtime: {runtime:.0f}s | Detected: {self.launch_count} tokens")
                
                # Get new tokens
                new_tokens = await self.scan_new_tokens()
                
                if new_tokens:
                    # Sort by liquidity
                    new_tokens.sort(key=lambda x: x['liquidity'], reverse=True)
                    
                    for token in new_tokens:
                        await self.display_new_token(token)
                        
                        # Small delay between displays
                        await asyncio.sleep(0.5)
                
                # Wait before next scan
                await asyncio.sleep(10)  # Scan every 10 seconds
                
            except KeyboardInterrupt:
                print(f"\n⏹️ Monitor stopped. Total detected: {self.launch_count} new tokens")
                break
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                await asyncio.sleep(30)  # Wait longer on error

async def main():
    """Main function"""
    
    async with LiveTokenMonitor() as monitor:
        await monitor.monitor_live()

if __name__ == "__main__":
    asyncio.run(main()) 