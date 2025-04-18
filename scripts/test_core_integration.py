#!/usr/bin/env python3
"""
Test script for AntBot Core FFI integration
"""
import os
import sys
import time
import asyncio
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# Import the bridge
from src.bindings.worker_bridge import WorkerBridge

async def test_worker_lifecycle():
    """Test the basic worker lifecycle: create, start, get status, stop"""
    print("Testing worker lifecycle...")
    
    # Create a bridge instance
    bridge = WorkerBridge()
    
    # Generate a test worker ID
    worker_id = f"test_worker_{int(time.time())}"
    
    # Test wallet address (this won't be used in our mocked implementation)
    wallet_address = "3h1zGmCwsRJnVk5BuRNMLsPaQu1y2aqXqXDWYCgrp5UG"
    
    # Initial capital
    capital = 1.0
    
    # Start a worker
    print(f"Starting worker {worker_id}...")
    start_result = await bridge.start_worker(worker_id, wallet_address, capital)
    
    if not start_result:
        print("Error: Failed to start worker")
        return False
    
    print("Worker started successfully")
    
    # Wait for the worker to initialize
    print("Waiting for worker to initialize...")
    await asyncio.sleep(2)
    
    # Get worker status
    print("Getting worker status...")
    status = await bridge.get_worker_status(worker_id)
    
    if not status:
        print("Error: Failed to get worker status")
        return False
    
    print(f"Worker status: {status}")
    
    # Let it run for a bit
    print("Letting worker run for 5 seconds...")
    await asyncio.sleep(5)
    
    # Stop the worker
    print(f"Stopping worker {worker_id}...")
    stop_result = await bridge.stop_worker(worker_id)
    
    if not stop_result:
        print("Error: Failed to stop worker")
        return False
    
    print("Worker stopped successfully")
    
    return True

async def test_multiple_workers():
    """Test running multiple workers simultaneously"""
    print("Testing multiple workers...")
    
    # Create a bridge instance
    bridge = WorkerBridge()
    
    # Number of workers to create
    num_workers = 3
    worker_ids = [f"test_worker_{i}_{int(time.time())}" for i in range(num_workers)]
    
    # Test wallet address
    wallet_address = "3h1zGmCwsRJnVk5BuRNMLsPaQu1y2aqXqXDWYCgrp5UG"
    
    # Start workers
    for i, worker_id in enumerate(worker_ids):
        print(f"Starting worker {worker_id}...")
        start_result = await bridge.start_worker(worker_id, wallet_address, 1.0 + i)
        
        if not start_result:
            print(f"Error: Failed to start worker {worker_id}")
            return False
    
    print("All workers started successfully")
    
    # Wait for the workers to initialize
    await asyncio.sleep(2)
    
    # Get status of all workers
    print("Getting status of all workers...")
    statuses = await bridge.get_all_worker_statuses()
    
    if not statuses or len(statuses) != num_workers:
        print(f"Error: Expected {num_workers} worker statuses, got {len(statuses) if statuses else 0}")
        return False
    
    print(f"Worker statuses: {statuses}")
    
    # Let them run for a bit
    print("Letting workers run for 5 seconds...")
    await asyncio.sleep(5)
    
    # Stop all workers
    for worker_id in worker_ids:
        print(f"Stopping worker {worker_id}...")
        stop_result = await bridge.stop_worker(worker_id)
        
        if not stop_result:
            print(f"Error: Failed to stop worker {worker_id}")
            return False
    
    print("All workers stopped successfully")
    
    return True

async def main():
    print("=== AntBot Core Integration Test ===")
    
    # Test worker lifecycle
    lifecycle_result = await test_worker_lifecycle()
    print(f"Worker lifecycle test {'passed' if lifecycle_result else 'failed'}")
    
    # Test multiple workers
    multiple_result = await test_multiple_workers()
    print(f"Multiple workers test {'passed' if multiple_result else 'failed'}")
    
    if lifecycle_result and multiple_result:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 