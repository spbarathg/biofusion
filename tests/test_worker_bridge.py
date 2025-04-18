import unittest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

# Import the module to test
from src.bindings.worker_bridge import WorkerBridge, MockRustLib

class TestWorkerBridge(unittest.TestCase):
    """Test cases for the WorkerBridge class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_root = Path(self.temp_dir.name)
        
        # Create test directories
        self.config_dir = self.test_root / "config"
        self.logs_dir = self.test_root / "logs"
        
        # Create directories
        self.config_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create a test config file
        self.config_path = self.config_dir / "settings.yaml"
        with open(self.config_path, 'w') as f:
            f.write("# Test config file")
        
        # Set up patches
        self.paths_patch = patch.multiple(
            'src.bindings.worker_bridge',
            ROOT_DIR=self.test_root,
            CONFIG_PATH=self.config_path,
            LOGS_DIR=self.logs_dir
        )
        self.paths_patch.start()
        
        # Use mock Rust library for testing
        self.rust_lib_patch = patch('src.bindings.worker_bridge.rust_lib', MockRustLib())
        self.mock_rust_lib = self.rust_lib_patch.start()
        
        # Initialize worker bridge
        self.worker_bridge = WorkerBridge(str(self.config_path))
    
    def tearDown(self):
        """Tear down test fixtures"""
        # Stop all patches
        self.paths_patch.stop()
        self.rust_lib_patch.stop()
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    async def test_start_worker(self):
        """Test starting a worker"""
        # Define test inputs
        worker_id = "test_worker"
        wallet_address = "DYw8jCTfwHNRJhhmFcbXvVDTqWMEVFBX6ZKUmG5CNSKK"
        capital = 5.0
        
        # Start the worker
        success = await self.worker_bridge.start_worker(
            worker_id=worker_id,
            wallet_address=wallet_address,
            capital=capital
        )
        
        # Verify success
        self.assertTrue(success)
        
        # Verify worker is in active workers
        self.assertIn(worker_id, self.worker_bridge.active_workers)
        worker_info = self.worker_bridge.active_workers[worker_id]
        self.assertEqual(worker_info["wallet_address"], wallet_address)
        self.assertEqual(worker_info["capital"], capital)
        
        # Verify worker handle was stored
        self.assertIn(worker_id, self.worker_bridge.worker_handles)
        self.assertIsNotNone(self.worker_bridge.worker_handles[worker_id])
    
    async def test_stop_worker(self):
        """Test stopping a worker"""
        # First start a worker
        worker_id = "test_worker"
        wallet_address = "DYw8jCTfwHNRJhhmFcbXvVDTqWMEVFBX6ZKUmG5CNSKK"
        capital = 5.0
        
        await self.worker_bridge.start_worker(
            worker_id=worker_id,
            wallet_address=wallet_address,
            capital=capital
        )
        
        # Now stop the worker
        success = await self.worker_bridge.stop_worker(worker_id)
        
        # Verify success
        self.assertTrue(success)
        
        # Try stopping a non-existent worker
        success = await self.worker_bridge.stop_worker("non_existent_worker")
        self.assertFalse(success)
    
    async def test_get_worker_status(self):
        """Test getting worker status"""
        # First start a worker
        worker_id = "test_worker"
        wallet_address = "DYw8jCTfwHNRJhhmFcbXvVDTqWMEVFBX6ZKUmG5CNSKK"
        capital = 5.0
        
        await self.worker_bridge.start_worker(
            worker_id=worker_id,
            wallet_address=wallet_address,
            capital=capital
        )
        
        # Get worker status
        status = await self.worker_bridge.get_worker_status(worker_id)
        
        # Verify status
        self.assertIsNotNone(status)
        self.assertEqual(status["id"], worker_id)
        self.assertEqual(status["wallet_address"], wallet_address)
        self.assertIn("is_running", status)
        self.assertIn("trades_executed", status)
        self.assertIn("total_profit", status)
        self.assertIn("uptime", status)
        
        # Try getting status for a non-existent worker
        status = await self.worker_bridge.get_worker_status("non_existent_worker")
        self.assertIsNone(status)
    
    async def test_get_all_worker_statuses(self):
        """Test getting all worker statuses"""
        # Start multiple workers
        workers = [
            ("worker1", "address1", 1.0),
            ("worker2", "address2", 2.0),
            ("worker3", "address3", 3.0)
        ]
        
        for worker_id, wallet_address, capital in workers:
            await self.worker_bridge.start_worker(
                worker_id=worker_id,
                wallet_address=wallet_address,
                capital=capital
            )
        
        # Get all statuses
        statuses = await self.worker_bridge.get_all_worker_statuses()
        
        # Verify we have status for each worker
        self.assertEqual(len(statuses), 3)
        for worker_id, _, _ in workers:
            self.assertIn(worker_id, statuses)
    
    async def test_update_worker_metrics(self):
        """Test updating worker metrics"""
        # First start a worker
        worker_id = "test_worker"
        wallet_address = "DYw8jCTfwHNRJhhmFcbXvVDTqWMEVFBX6ZKUmG5CNSKK"
        capital = 5.0
        
        await self.worker_bridge.start_worker(
            worker_id=worker_id,
            wallet_address=wallet_address,
            capital=capital
        )
        
        # Update metrics
        trades = 10
        profit = 1.5
        success = await self.worker_bridge.update_worker_metrics(
            worker_id=worker_id,
            trades=trades,
            profit=profit
        )
        
        # Verify success
        self.assertTrue(success)
        
        # Get status to verify metrics were updated
        status = await self.worker_bridge.get_worker_status(worker_id)
        self.assertEqual(status["trades_executed"], trades)
        self.assertEqual(status["total_profit"], profit)
        
        # Try updating metrics for a non-existent worker
        success = await self.worker_bridge.update_worker_metrics(
            worker_id="non_existent_worker",
            trades=5,
            profit=0.5
        )
        self.assertFalse(success)

if __name__ == '__main__':
    # Run with asyncio support
    loop = asyncio.get_event_loop()
    loop.run_until_complete(unittest.main()) 