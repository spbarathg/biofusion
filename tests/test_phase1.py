#!/usr/bin/env python3
import os
import sys
import asyncio
from pathlib import Path
from loguru import logger

# Ensure src is in the path
sys.path.append(os.path.abspath("."))

from src.models.queen import Queen
from src.models.worker import Worker
from src.models.capital_manager import CapitalManager
from src.core.wallet_manager import WalletManager

async def test_queen():
    """Test the queen implementation"""
    print("\n=== Testing Queen Implementation ===")
    
    # Create a queen instance
    queen = Queen()
    
    # Initialize with mock capital
    await queen.initialize_colony(10.0)
    
    # Get colony state
    state = await queen.get_colony_state()
    print(f"Initial colony state: {state}")
    
    # Spawn some workers
    await queen.manage_workers()
    
    # Get updated state
    state = await queen.get_colony_state()
    print(f"Colony state after spawning workers: {state}")
    
    # Create a backup
    backup_path = await queen.backup_wallets()
    print(f"Created backup at: {backup_path}")
    
    # Stop the colony
    await queen.stop_colony()
    print("Colony stopped")

async def test_worker():
    """Test the worker implementation"""
    print("\n=== Testing Worker Implementation ===")
    
    # Create a wallet manager to create a test wallet
    wallet_manager = WalletManager()
    wallet_id = wallet_manager.create_wallet("test_worker", "worker")
    print(f"Created test worker wallet with ID: {wallet_id}")
    
    # Create a worker
    worker = Worker("test_worker_1", wallet_id=wallet_id)
    
    # Start the worker and let it run for a bit
    print("Starting worker...")
    worker_task = asyncio.create_task(worker.start())
    
    # Wait a bit for the worker to start and execute some trades
    await asyncio.sleep(5)  # Reduced from 30 to make testing faster
    
    # Stop the worker
    await worker.stop()
    
    # Get metrics
    metrics = worker.get_metrics()
    print(f"Worker metrics: {metrics}")

async def test_capital_manager():
    """Test the capital manager implementation"""
    print("\n=== Testing Capital Manager Implementation ===")
    
    # Create a capital manager
    capital_manager = CapitalManager()
    
    # Create a wallet manager to create test wallets
    wallet_manager = WalletManager()
    queen_wallet_id = wallet_manager.create_wallet("test_queen", "queen")
    print(f"Created test queen wallet with ID: {queen_wallet_id}")
    
    # Create a savings wallet to prevent the warning
    savings_wallet_id = wallet_manager.create_wallet("test_savings", "savings")
    print(f"Created test savings wallet with ID: {savings_wallet_id}")
    
    # Get savings metrics
    metrics = await capital_manager.get_savings_metrics()
    print(f"Initial savings metrics: {metrics}")
    
    # Test capital redistribution without actual transfer
    result = await capital_manager.redistribute_capital(
        queen_wallet_id,
        worker_allocation=0.4,
        princess_allocation=0.3
    )
    print(f"Capital redistribution: {result}")
    
    # We'll skip the process_profits test since it requires actual transfers
    # result = await capital_manager.process_profits(1.0, queen_wallet_id)
    # print(f"Processed profits: {result}")

async def main():
    """Main test function"""
    print("Starting Phase 1 tests...")
    
    try:
        # Test the wallet manager
        wallet_manager = WalletManager()
        wallets = wallet_manager.list_wallets()
        print(f"Existing wallets: {wallets}")
        
        # Run the queen test
        await test_queen()
        
        # Run the worker test
        await test_worker()
        
        # Run the capital manager test
        await test_capital_manager()
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"Error during tests: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the test
    asyncio.run(main()) 