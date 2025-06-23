#!/usr/bin/env python3
"""
Worker Ant V1 Runner
===================

Simple script to start the MVP trading bot.
"""

import os
import sys
import asyncio

# Add the parent directory to the path so we can import worker_ant_v1
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker_ant_v1.main import main


def banner():
    """Display startup banner"""
    print("""
🐜 ================================================== 🐜
   WORKER ANT V1 - MVP MEMECOIN TRADING BOT
🐜 ================================================== 🐜

    🎯 ONE ANT, ONE GOAL: PROFITABLE TRADES
    ⚡ Ultra-fast execution (<200ms)
    🛡️  Built-in safety limits
    📊 Real-time KPI tracking
    
Configuration:
  • Trade Size: 0.05 SOL (~$20-50)
  • Profit Target: 10%
  • Stop Loss: 10%
  • Timeout: 45 seconds
  
💡 Tip: Fund your wallet and set environment variables:
   export WALLET_PRIVATE_KEY="your_base58_key"
   export BIRDEYE_API_KEY="your_api_key" (optional)

🐜 ================================================== 🐜
""")


def check_environment():
    """Check required environment setup"""
    
    warnings = []
    
    # Check wallet
    if not os.getenv('WALLET_PRIVATE_KEY'):
        warnings.append("⚠️  WALLET_PRIVATE_KEY not set - will create new wallet")
    
    # Check APIs
    if not os.getenv('BIRDEYE_API_KEY'):
        warnings.append("⚠️  BIRDEYE_API_KEY not set - using free APIs only")
    
    # Check Solana RPC
    rpc_url = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')
    if 'mainnet-beta.solana.com' in rpc_url:
        warnings.append("⚠️  Using public RPC - consider private RPC for better speed")
        
    if warnings:
        print("Environment Warnings:")
        for warning in warnings:
            print(f"  {warning}")
        print()
        
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup your environment first, then restart.")
            sys.exit(1)


if __name__ == "__main__":
    try:
        banner()
        check_environment()
        
        print("🚀 Starting Worker Ant V1...")
        print("   Press Ctrl+C to stop gracefully")
        print()
        
        # Run the bot
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n👋 Worker Ant V1 stopped by user")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1) 