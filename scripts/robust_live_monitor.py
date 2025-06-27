#!/usr/bin/env python3
"""
ROBUST LIVE MONITOR
==================

More robust real-time monitor with better error handling.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import sys
import traceback

# Fix Windows asyncio event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class RobustLiveMonitor:
    """Robust live token monitor"""
    
    def __init__(self):
        self.session = None
        self.known_tokens = set()
        self.launch_count = 0
        self.start_time = datetime.now()
        self.scan_count = 0
        self.error_count = 0
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            resolver=aiohttp.AsyncResolver(),
            use_dns_cache=False,
            limit=100,
            ttl_dns_cache=300
        )
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            connector=connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scan_new_tokens(self):
        """Scan for new token launches with better error handling"""
        
        try:
            self.scan_count += 1
            print(f"🔍 Scanning... #{self.scan_count}", end=" ", flush=True)
            
            raydium_url = "https://api.raydium.io/v2/main/pairs"
            
            async with self.session.get(raydium_url) as response:
                print(f"(Status: {response.status})", end=" ", flush=True)
                
                if response.status == 200:
                    pairs = await response.json()
                    print(f"(Found {len(pairs)} pairs)")
                    
                    # SOL mint address
                    SOL_MINT = "So11111111111111111111111111111111111111112"
                    
                    new_tokens = []
                    processed = 0
                    
                    for pair in pairs:
                        processed += 1
                        if processed % 10000 == 0:
                            print(f"   Processed {processed}/{len(pairs)} pairs...", flush=True)
                        
                        quote_mint = pair.get("quoteMint", "")
                        base_mint = pair.get("baseMint", "")
                        
                        # Only SOL pairs we haven't seen before
                        if (quote_mint == SOL_MINT and 
                            base_mint and
                            base_mint not in self.known_tokens):
                            
                            try:
                                liquidity = float(pair.get("liquidity", 0))
                                volume24h = float(pair.get("volume24h", 0))
                                price = float(pair.get("price", 0))
                                
                                # Filter for valid tokens
                                if (liquidity > 100 and  # At least $100 liquidity
                                    price > 0 and
                                    len(base_mint) == 44):  # Valid Solana address length
                                    
                                    new_tokens.append({
                                        "address": base_mint,
                                        "liquidity": liquidity,
                                        "price": price,
                                        "volume24h": volume24h,
                                        "name": pair.get("name", "Unknown"),
                                        "amm_id": pair.get("ammId", ""),
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    
                                    self.known_tokens.add(base_mint)
                            
                            except (ValueError, TypeError) as e:
                                continue  # Skip invalid data
                    
                    print(f"   ✅ Found {len(new_tokens)} new tokens")
                    return new_tokens
                
                else:
                    print(f"❌ API Error {response.status}")
                    if response.status == 429:
                        print("   Rate limited - waiting longer...")
                        await asyncio.sleep(60)
                    return []
        
        except asyncio.TimeoutError:
            self.error_count += 1
            print(f"⏰ Timeout error #{self.error_count}")
            return []
        except aiohttp.ClientError as e:
            self.error_count += 1
            print(f"🌐 Network error #{self.error_count}: {str(e)}")
            return []
        except Exception as e:
            self.error_count += 1
            print(f"❌ Unexpected error #{self.error_count}: {str(e)}")
            print(f"   Full error: {traceback.format_exc()}")
            return []
    
    async def display_new_token(self, token):
        """Display a newly detected token"""
        
        self.launch_count += 1
        
        # Calculate potential score
        liquidity = token['liquidity']
        price = token['price']
        volume = token['volume24h']
        
        # Scoring system
        score = 0.3
        alerts = []
        
        if liquidity > 50000:
            score += 0.4
            alerts.append("VERY HIGH LIQUIDITY")
        elif liquidity > 10000:
            score += 0.3
            alerts.append("HIGH LIQUIDITY")
        elif liquidity > 1000:
            score += 0.2
            alerts.append("GOOD LIQUIDITY")
        
        if volume > 10000:
            score += 0.3
            alerts.append("HIGH VOLUME")
        elif volume > 1000:
            score += 0.2
            alerts.append("GOOD VOLUME")
        
        if 0.0000001 < price < 0.001:
            score += 0.3
            alerts.append("ULTRA LOW PRICE")
        elif 0.000001 < price < 0.01:
            score += 0.2
            alerts.append("LOW PRICE GEM")
        
        # Potential rating
        if score >= 0.9:
            potential = "🔥🔥🔥 ULTRA HOT"
        elif score >= 0.7:
            potential = "🔥🔥 VERY HOT"
        elif score >= 0.5:
            potential = "🔥 HOT"
        else:
            potential = "👀 WATCH"
        
        # Display
        print(f"\n" + "="*80)
        print(f"🚀 NEW TOKEN LAUNCH #{self.launch_count} - {datetime.now().strftime('%H:%M:%S')}")
        print(f"="*80)
        print(f"📍 FULL ADDRESS: {token['address']}")
        print(f"💰 Price: {token['price']:.12f} SOL")
        print(f"💎 Liquidity: ${token['liquidity']:,.2f}")
        print(f"📊 Volume 24h: ${token['volume24h']:,.2f}")
        print(f"⭐ Potential: {potential} (Score: {score:.2f})")
        
        if alerts:
            print(f"🚨 ALERTS: {' | '.join(alerts)}")
        
        print(f"🔗 Verify on Solscan: https://solscan.io/token/{token['address']}")
        print(f"🔗 Dexscreener: https://dexscreener.com/solana/{token['address']}")
        print(f"⏰ Runtime: {(datetime.now() - self.start_time).total_seconds():.0f}s")
        print(f"="*80)
        
        # Save to file
        self.save_token_address(token, score, alerts)
    
    def save_token_address(self, token, score, alerts):
        """Save token to file with all details"""
        
        import os
        os.makedirs("logs", exist_ok=True)
        
        # Save to main log
        with open("logs/live_launches_full.txt", "a") as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            alert_text = "|".join(alerts) if alerts else "None"
            f.write(f"{timestamp} | {token['address']} | ${token['liquidity']:,.2f} | {token['price']:.12f} SOL | Score: {score:.2f} | Alerts: {alert_text}\n")
        
        # Save just addresses for easy copy
        with open("logs/addresses_only.txt", "a") as f:
            f.write(f"{token['address']}\n")
    
    async def monitor_live(self):
        """Main live monitoring loop"""
        
        print("🔴 ROBUST LIVE TOKEN LAUNCH MONITOR")
        print("=" * 60)
        print("🚀 Monitoring Solana token launches in REAL-TIME")
        print("📊 Enhanced error handling and debugging")
        print("⚡ Scanning every 15 seconds")
        print("💎 Full address verification included")
        print("📝 Saving all data to logs/")
        print("")
        
        while True:
            try:
                scan_time = datetime.now().strftime('%H:%M:%S')
                runtime = (datetime.now() - self.start_time).total_seconds()
                
                # Status update
                print(f"\n📡 SCAN - {scan_time} | Runtime: {runtime:.0f}s | Launches: {self.launch_count} | Errors: {self.error_count}")
                
                # Get new tokens
                new_tokens = await self.scan_new_tokens()
                
                if new_tokens:
                    # Sort by liquidity (highest first)
                    new_tokens.sort(key=lambda x: x['liquidity'], reverse=True)
                    
                    for token in new_tokens:
                        await self.display_new_token(token)
                        await asyncio.sleep(1)  # Small delay between displays
                
                else:
                    print("   💤 No new tokens this scan")
                
                # Wait before next scan
                print(f"   ⏳ Next scan in 15 seconds...")
                await asyncio.sleep(15)
                
            except KeyboardInterrupt:
                print(f"\n⏹️ Monitor stopped by user")
                print(f"📊 Total detected: {self.launch_count} new tokens")
                print(f"📊 Total scans: {self.scan_count}")
                print(f"📊 Total errors: {self.error_count}")
                break
            except Exception as e:
                self.error_count += 1
                print(f"❌ Monitor error #{self.error_count}: {str(e)}")
                print(f"   Traceback: {traceback.format_exc()}")
                await asyncio.sleep(30)

async def main():
    """Main function"""
    
    async with RobustLiveMonitor() as monitor:
        await monitor.monitor_live()

if __name__ == "__main__":
    asyncio.run(main()) 