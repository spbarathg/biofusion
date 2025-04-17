import os
import json
import asyncio
import ctypes
import time
from typing import Dict, List, Optional, Any
from loguru import logger
from pathlib import Path

# Import paths module
from src.core.paths import ROOT_DIR, CONFIG_PATH, LOGS_DIR

# Configure library paths
if os.name == 'posix':  # Linux/Mac
    LIB_EXT = 'so'
elif os.name == 'nt':  # Windows
    LIB_EXT = 'dll'
else:
    raise RuntimeError(f"Unsupported OS: {os.name}")

# Find the Rust library
def find_rust_lib():
    # Check in known locations
    possible_paths = [
        ROOT_DIR / f"rust_core/target/release/libant_bot_core.{LIB_EXT}",
        ROOT_DIR / f"rust_core/target/debug/libant_bot_core.{LIB_EXT}"
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path.absolute())
    
    # If not found, build it
    logger.info("Rust library not found, attempting to build it...")
    build_status = os.system(f"cd {ROOT_DIR / 'rust_core'} && cargo build --release")
    
    if build_status != 0:
        raise RuntimeError("Failed to build Rust library")
    
    # Check if build succeeded
    built_path = ROOT_DIR / f"rust_core/target/release/libant_bot_core.{LIB_EXT}"
    if built_path.exists():
        return str(built_path.absolute())
    
    raise RuntimeError("Could not find or build Rust library")

# Load the Rust library
try:
    RUST_LIB_PATH = find_rust_lib()
    rust_lib = ctypes.CDLL(RUST_LIB_PATH)
    
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
    
    logger.info(f"Successfully loaded Rust library: {RUST_LIB_PATH}")
    
except Exception as e:
    logger.error(f"Failed to load Rust library: {str(e)}")
    raise

class WorkerBridge:
    """
    Bridges Python capital logic with the Rust trading engine.
    Directly communicates with the Rust core using FFI instead of subprocess.
    """
    
    def __init__(self, config_path: str = None):
        # Use provided config path or default
        self.config_path = str(Path(config_path).absolute() if config_path else CONFIG_PATH)
        
        self.active_workers = {}
        self.worker_handles = {}
        
        # Setup logging
        logger.add(
            LOGS_DIR / "worker_bridge.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
    
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
            
            # Call Rust function to create worker
            worker_id_bytes = worker_id.encode('utf-8')
            wallet_address_bytes = wallet_address.encode('utf-8')
            config_path_bytes = self.config_path.encode('utf-8')
            
            worker_handle = rust_lib.worker_create(
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
            
            # Store worker info
            self.active_workers[worker_id] = {
                "wallet_address": wallet_address,
                "capital": capital,
                "start_time": time.time(),
                "trades": 0,
                "profit": 0.0
            }
            
            # Start the worker
            result = rust_lib.worker_start(worker_handle)
            
            if result < 0:
                logger.error(f"Failed to start Worker Ant {worker_id}")
                del self.worker_handles[worker_id]
                del self.active_workers[worker_id]
                return False
            
            logger.info(f"Started Worker Ant {worker_id} with {capital} SOL")
            
            # Start monitoring task
            asyncio.create_task(self._monitor_worker(worker_id))
            
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
            
            # Get handle
            handle = self.worker_handles[worker_id]
            
            # Stop the worker
            result = rust_lib.worker_stop(handle)
            
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
        
        # Get status from Rust
        try:
            handle = self.worker_handles[worker_id]
            
            # Prepare buffer for JSON response
            buffer_size = 1024
            status_buffer = ctypes.create_string_buffer(buffer_size)
            
            # Get status
            result = rust_lib.worker_get_status(handle, status_buffer, buffer_size)
            
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
        Update the metrics for a Worker Ant
        
        Args:
            worker_id: Unique identifier for the worker
            trades: Number of trades executed
            profit: Profit in SOL
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if worker_id not in self.active_workers or worker_id not in self.worker_handles:
                logger.warning(f"Worker {worker_id} not found for metrics update")
                return False
            
            # Update local cache
            self.active_workers[worker_id]["trades"] = trades
            self.active_workers[worker_id]["profit"] = profit
            
            # Update Rust worker
            handle = self.worker_handles[worker_id]
            result = rust_lib.worker_update_metrics(handle, trades, profit)
            
            if result < 0:
                logger.warning(f"Failed to update metrics for Worker Ant {worker_id}")
                return False
            
            logger.info(f"Updated metrics for Worker Ant {worker_id}: {trades} trades, {profit} SOL profit")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metrics for Worker Ant {worker_id}: {str(e)}")
            return False
    
    async def _monitor_worker(self, worker_id: str):
        """
        Monitor a Worker Ant
        
        Args:
            worker_id: Unique identifier for the worker
        """
        try:
            # Check every 10 seconds
            while worker_id in self.active_workers and worker_id in self.worker_handles:
                status = await self.get_worker_status(worker_id)
                
                if not status or not status.get("is_running", False):
                    logger.warning(f"Worker Ant {worker_id} is no longer running")
                    
                    # Clean up
                    if worker_id in self.worker_handles:
                        del self.worker_handles[worker_id]
                    if worker_id in self.active_workers:
                        del self.active_workers[worker_id]
                    
                    break
                
                # Sleep for a bit
                await asyncio.sleep(10)
                
        except Exception as e:
            logger.error(f"Error monitoring Worker Ant {worker_id}: {str(e)}")
        finally:
            # Clean up if process terminated
            if worker_id in self.active_workers:
                logger.info(f"Worker Ant {worker_id} monitoring stopped")
                if worker_id in self.worker_handles:
                    del self.worker_handles[worker_id]
                if worker_id in self.active_workers:
                    del self.active_workers[worker_id]

async def main():
    # Example usage
    bridge = WorkerBridge()
    
    # Start a worker
    success = await bridge.start_worker("worker_1", "wallet_address_1", 10.0)
    if success:
        # Wait a bit
        await asyncio.sleep(5)
        
        # Get status
        status = await bridge.get_worker_status("worker_1")
        print(f"Worker status: {status}")
        
        # Wait a bit more
        await asyncio.sleep(10)
        
        # Stop the worker
        await bridge.stop_worker("worker_1")
    
if __name__ == "__main__":
    asyncio.run(main()) 