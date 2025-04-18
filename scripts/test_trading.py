#!/usr/bin/env python3
"""
Test script for AntBot trading functionality
"""
import os
import sys
import time
import asyncio
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Import the bridge
from src.bindings.worker_bridge import WorkerBridge
from src.core.wallet_manager import WalletManager

async def test_trading():
    """Test the trading functionality"""
    print("=== Testing Trading Functionality ===")
    
    # Create a wallet manager
    wallet_manager = WalletManager()
    
    # Create a test wallet
    print("Creating test wallet...")
    wallet_id = wallet_manager.create_wallet("test_trading", "worker")
    wallet_info = wallet_manager.wallets[wallet_id]
    
    print(f"Created wallet: {wallet_id}")
    print(f"Public key: {wallet_info['public_key']}")
    
    # Create a bridge instance
    bridge = WorkerBridge()
    
    # Start a worker with the test wallet
    worker_id = f"trading_test_{int(time.time())}"
    print(f"Starting worker {worker_id}...")
    
    start_result = await bridge.start_worker(
        worker_id, 
        wallet_info['public_key'], 
        1.0  # Initial capital
    )
    
    if not start_result:
        print("Error: Failed to start worker")
        return False
    
    print("Worker started successfully")
    
    # Wait for the worker to initialize
    print("Waiting for worker to initialize...")
    await asyncio.sleep(5)
    
    # Get worker status
    print("Getting worker status...")
    status = await bridge.get_worker_status(worker_id)
    
    if not status:
        print("Error: Failed to get worker status")
        return False
    
    print(f"Initial worker status: {json.dumps(status, indent=2)}")
    
    # Let it run for a while to perform trading
    print("Letting worker run for 30 seconds to perform trading...")
    await asyncio.sleep(30)
    
    # Get updated worker status
    print("Getting updated worker status...")
    status = await bridge.get_worker_status(worker_id)
    
    if not status:
        print("Error: Failed to get worker status")
        return False
    
    print(f"Updated worker status: {json.dumps(status, indent=2)}")
    
    # Check if trades were executed
    trades_executed = status.get('trades_executed', 0)
    print(f"Trades executed: {trades_executed}")
    
    # Check profit
    total_profit = float(status.get('total_profit', 0))
    print(f"Total profit: {total_profit:.6f} SOL")
    
    # Stop the worker
    print(f"Stopping worker {worker_id}...")
    stop_result = await bridge.stop_worker(worker_id)
    
    if not stop_result:
        print("Error: Failed to stop worker")
        return False
    
    print("Worker stopped successfully")
    
    return trades_executed > 0

async def main():
    # Test trading functionality
    trading_result = await test_trading()
    
    if trading_result:
        print("\nTrading test passed!")
        return 0
    else:
        print("\nTrading test failed. No trades were executed.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 