import os
import json
import asyncio
import ctypes
import time
from typing import Dict, List, Optional, Any
from loguru import logger
from pathlib import Path

from src.utils.logging.logger import setup_logging

# Configure library paths
if os.name == 'posix':  # Linux/Mac
    LIB_EXT = 'so'
elif os.name == 'nt':  # Windows
    LIB_EXT = 'dll'
else:
    raise RuntimeError(f"Unsupported OS: {os.name}")

# Define paths for testing and runtime
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONFIG_PATH = ROOT_DIR / "config" / "settings.yaml"
LOGS_DIR = ROOT_DIR / "logs"

# Module level rust_lib for testing
rust_lib = None

class WorkerBridge:
    """
    Bridges Python capital logic with the Rust trading engine.
    Directly communicates with the Rust core using FFI.
    Simplified version with error handling for missing libraries.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        # Convert config_path to absolute path if needed
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), config_path)
            
        self.config_path = config_path
        self.active_workers = {}
        self.worker_handles = {}
        
        # Setup logging
        setup_logging("worker_bridge", "worker_bridge.log")
        
        # Try to load the Rust library
        self.rust_lib = None
        try:
            self._load_rust_lib()
        except Exception as e:
            logger.warning(f"Could not load Rust library: {str(e)}")
            logger.warning("Worker bridge will operate in simulation mode")
    
    def _load_rust_lib(self):
        """Load the Rust library, if available"""
        # Find the Rust library
        def find_rust_lib():
            # Check in known locations
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            possible_paths = [
                os.path.join(base_dir, f"rust_core/target/release/libant_bot_core.{LIB_EXT}"),
                os.path.join(base_dir, f"target/release/libant_bot_core.{LIB_EXT}"),
                os.path.join(os.path.dirname(base_dir), f"rust_core/target/release/libant_bot_core.{LIB_EXT}")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return os.path.abspath(path)
            
            # If not found, try to build it
            logger.info("Rust library not found, attempting to build it...")
            build_cmd = f"cd {os.path.join(base_dir, 'rust_core')} && cargo build --release"
            build_status = os.system(build_cmd)
            
            if build_status != 0:
                raise RuntimeError("Failed to build Rust library")
            
            # Check if build succeeded
            if os.path.exists(possible_paths[0]):
                return os.path.abspath(possible_paths[0])
            
            raise RuntimeError("Could not find or build Rust library")

        # Load the Rust library
        try:
            rust_lib_path = find_rust_lib()
            rust_lib = ctypes.CDLL(rust_lib_path)
            
            # Define function signatures
            rust_lib.worker_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_char_p]
            rust_lib.worker_create.restype = ctypes.c_int
            
            rust_lib.worker_start.argtypes = [ctypes.c_int]
            rust_lib.worker_start.restype = ctypes.c_int
            
            rust_lib.worker_stop.argtypes = [ctypes.c_int]
            rust_lib.worker_stop.restype = ctypes.c_int
            
            rust_lib.worker_get_status.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            rust_lib.worker_get_status.restype = ctypes.c_int
            
            rust_lib.worker_update_metrics.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
            rust_lib.worker_update_metrics.restype = ctypes.c_int
            
            logger.info(f"Successfully loaded Rust library: {rust_lib_path}")
            self.rust_lib = rust_lib
            
        except Exception as e:
            logger.error(f"Failed to load Rust library: {str(e)}")
            self.rust_lib = None
            raise
    
    async def start_worker(self, worker_id: str, wallet_address: str, capital: float) -> bool:
        """
        Start a new Worker Ant
        
        Args:
            worker_id: Unique identifier for the worker
            wallet_address: Solana wallet address for the worker
            capital: Initial capital in SOL
            
        Returns:
            bool: True if worker started successfully, False otherwise
        """
        try:
            # Check if worker is already running
            if worker_id in self.active_workers:
                logger.warning(f"Worker {worker_id} is already running")
                return False
            
            # If Rust library is available, use it
            if self.rust_lib:
                # Call Rust function to create worker
                worker_id_bytes = worker_id.encode('utf-8')
                wallet_address_bytes = wallet_address.encode('utf-8')
                config_path_bytes = self.config_path.encode('utf-8')
                
                worker_handle = self.rust_lib.worker_create(
                    worker_id_bytes, 
                    wallet_address_bytes, 
                    capital, 
                    config_path_bytes
                )
                
                if worker_handle < 0:
                    logger.error(f"Failed to create Worker Ant {worker_id}")
                    return False
                
                # Store handle
                self.worker_handles[worker_id] = worker_handle
                
                # Start the worker
                result = self.rust_lib.worker_start(worker_handle)
                
                if result < 0:
                    logger.error(f"Failed to start Worker Ant {worker_id}")
                    del self.worker_handles[worker_id]
                    return False
            else:
                # Simulation mode - generate a fake handle
                self.worker_handles[worker_id] = len(self.worker_handles) + 1
            
            # Store worker info
            self.active_workers[worker_id] = {
                "wallet_address": wallet_address,
                "capital": capital,
                "start_time": time.time(),
                "trades": 0,
                "profit": 0.0,
                "simulation_mode": self.rust_lib is None
            }
            
            logger.info(f"Started Worker Ant {worker_id} with {capital} SOL")
            
            # Start monitoring task for simulation mode
            if self.rust_lib is None:
                asyncio.create_task(self._simulate_worker_activity(worker_id))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Worker Ant {worker_id}: {str(e)}")
            return False
    
    async def stop_worker(self, worker_id: str) -> bool:
        """
        Stop a running Worker Ant
        
        Args:
            worker_id: Unique identifier for the worker
            
        Returns:
            bool: True if worker stopped successfully, False otherwise
        """
        try:
            # Check if worker is running
            if worker_id not in self.active_workers or worker_id not in self.worker_handles:
                logger.warning(f"Worker {worker_id} is not running")
                return False
            
            # If Rust library is available, use it
            if self.rust_lib:
                # Get handle
                handle = self.worker_handles[worker_id]
                
                # Stop the worker
                result = self.rust_lib.worker_stop(handle)
                
                if result < 0:
                    logger.error(f"Failed to stop Worker Ant {worker_id}")
                    return False
            
            # Remove from active workers
            del self.worker_handles[worker_id]
            del self.active_workers[worker_id]
            
            logger.info(f"Stopped Worker Ant {worker_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Worker Ant {worker_id}: {str(e)}")
            return False
    
    async def get_worker_status(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a Worker Ant
        
        Args:
            worker_id: Unique identifier for the worker
            
        Returns:
            Optional[Dict[str, Any]]: Worker status or None if not found
        """
        if worker_id not in self.active_workers or worker_id not in self.worker_handles:
            return None
        
        worker_info = self.active_workers[worker_id].copy()
        
        # Add runtime
        worker_info["runtime"] = time.time() - worker_info["start_time"]
        
        # If in simulation mode, return simulated status
        if worker_info.get("simulation_mode", False):
            worker_info["is_running"] = True
            worker_info["total_profit"] = worker_info["profit"]
            return worker_info
        
        # If Rust library is available, get status from Rust
        if self.rust_lib:
            try:
                handle = self.worker_handles[worker_id]
                
                # Prepare buffer for JSON response
                buffer_size = 1024
                status_buffer = ctypes.create_string_buffer(buffer_size)
                
                # Get status
                result = self.rust_lib.worker_get_status(handle, status_buffer, buffer_size)
                
                if result >= 0:
                    # Parse JSON status
                    status_json = status_buffer.value.decode('utf-8')
                    rust_status = json.loads(status_json)
                    
                    # Update worker info
                    worker_info.update(rust_status)
                    worker_info["is_running"] = True
                else:
                    worker_info["is_running"] = False
                    logger.warning(f"Failed to get status for Worker Ant {worker_id}")
            
            except Exception as e:
                logger.error(f"Error getting status for Worker Ant {worker_id}: {str(e)}")
                worker_info["is_running"] = False
        
        return worker_info
    
    async def get_all_worker_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the status of all Worker Ants
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of worker statuses
        """
        statuses = {}
        
        for worker_id in self.active_workers:
            status = await self.get_worker_status(worker_id)
            if status:
                statuses[worker_id] = status
        
        return statuses
    
    async def update_worker_metrics(self, worker_id: str, trades: int, profit: float) -> bool:
        """
        Update worker metrics
        
        Args:
            worker_id: Unique identifier for the worker
            trades: Number of trades
            profit: Profit amount
            
        Returns:
            bool: True if metrics updated successfully, False otherwise
        """
        try:
            # Check if worker is running
            if worker_id not in self.active_workers or worker_id not in self.worker_handles:
                logger.warning(f"Worker {worker_id} is not running")
                return False
            
            # If in simulation mode, update simulated metrics
            if self.active_workers[worker_id].get("simulation_mode", False):
                self.active_workers[worker_id]["trades"] = trades
                self.active_workers[worker_id]["profit"] = profit
                return True
            
            # If Rust library is available, update metrics in Rust
            if self.rust_lib:
                handle = self.worker_handles[worker_id]
                
                # Update metrics
                result = self.rust_lib.worker_update_metrics(handle, trades, profit)
                
                if result < 0:
                    logger.error(f"Failed to update metrics for Worker Ant {worker_id}")
                    return False
            
            logger.info(f"Updated metrics for Worker Ant {worker_id}: trades={trades}, profit={profit}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metrics for Worker Ant {worker_id}: {str(e)}")
            return False
    
    async def _simulate_worker_activity(self, worker_id: str):
        """
        Simulate worker activity when Rust core is not available
        
        Args:
            worker_id: Worker ID to simulate
        """
        logger.info(f"Starting simulated activity for worker {worker_id}")
        
        while worker_id in self.active_workers:
            try:
                # Simulate trades and profits
                if worker_id in self.active_workers:
                    # Get current state
                    current_trades = self.active_workers[worker_id].get("trades", 0)
                    current_profit = self.active_workers[worker_id].get("profit", 0.0)
                    
                    # Add 1-3 trades randomly with a small profit
                    import random
                    new_trades = random.randint(1, 3)
                    profit_per_trade = random.uniform(0.001, 0.005)  # 0.1% to 0.5% profit
                    
                    # Update metrics
                    self.active_workers[worker_id]["trades"] = current_trades + new_trades
                    
                    # Calculate profit based on capital
                    capital = self.active_workers[worker_id].get("capital", 1.0)
                    new_profit = capital * profit_per_trade * new_trades
                    
                    self.active_workers[worker_id]["profit"] = current_profit + new_profit
                    
                    logger.debug(f"Simulated {new_trades} trades with {new_profit:.6f} SOL profit for worker {worker_id}")
                
                # Sleep for a random interval
                await asyncio.sleep(random.uniform(10, 30))
                
            except Exception as e:
                logger.error(f"Error in simulated activity for worker {worker_id}: {str(e)}")
                await asyncio.sleep(5)

class MockRustLib:
    """
    Mock implementation of the Rust library for testing.
    Simulates the behavior of the Rust FFI interface.
    """
    
    def __init__(self):
        self.workers = {}
        self.next_handle = 1
    
    def worker_create(self, worker_id, wallet_address, capital, config_path):
        """Create a worker and return a handle"""
        worker_id_str = worker_id.decode('utf-8')
        wallet_address_str = wallet_address.decode('utf-8')
        
        # Create worker record
        handle = self.next_handle
        self.next_handle += 1
        
        self.workers[handle] = {
            "id": worker_id_str,
            "wallet_address": wallet_address_str,
            "capital": capital,
            "config_path": config_path.decode('utf-8'),
            "is_running": False,
            "start_time": time.time(),
            "trades_executed": 0,
            "total_profit": 0.0
        }
        
        return handle
    
    def worker_start(self, handle):
        """Start a worker by handle"""
        if handle not in self.workers:
            return -1
        
        self.workers[handle]["is_running"] = True
        self.workers[handle]["start_time"] = time.time()
        return 0
    
    def worker_stop(self, handle):
        """Stop a worker by handle"""
        if handle not in self.workers:
            return -1
        
        self.workers[handle]["is_running"] = False
        return 0
    
    def worker_get_status(self, handle, buffer, buffer_size):
        """Get worker status by handle"""
        if handle not in self.workers:
            return -1
        
        worker = self.workers[handle]
        
        # Calculate uptime
        uptime = time.time() - worker["start_time"]
        
        # Prepare status JSON
        status = {
            "id": worker["id"],
            "wallet_address": worker["wallet_address"],
            "is_running": worker["is_running"],
            "uptime": uptime,
            "trades_executed": worker["trades_executed"],
            "total_profit": worker["total_profit"]
        }
        
        # Convert to JSON and copy to buffer
        json_str = json.dumps(status).encode('utf-8')
        
        # Truncate if too large
        if len(json_str) >= buffer_size:
            return -1
        
        ctypes.memmove(buffer, json_str, len(json_str))
        # Add null terminator
        buffer[len(json_str)] = 0
        
        return 0
    
    def worker_update_metrics(self, handle, trades, profit):
        """Update worker metrics by handle"""
        if handle not in self.workers:
            return -1
        
        self.workers[handle]["trades_executed"] = trades
        self.workers[handle]["total_profit"] = profit
        return 0

async def main():
    # Example usage
    bridge = WorkerBridge()
    worker_id = "test_worker_1"
    wallet_address = "SIMULATED_WALLET_ADDRESS"
    
    # Start a worker
    success = await bridge.start_worker(worker_id, wallet_address, 1.0)
    print(f"Started worker: {success}")
    
    # Wait for a moment
    await asyncio.sleep(5)
    
    # Get status
    status = await bridge.get_worker_status(worker_id)
    print(f"Worker status: {status}")
    
    # Stop the worker
    success = await bridge.stop_worker(worker_id)
    print(f"Stopped worker: {success}")

if __name__ == "__main__":
    asyncio.run(main()) 