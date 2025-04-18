import os
import yaml
import asyncio
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import time

from src.utils.logging.logger import setup_logging
from src.core.paths import CONFIG_PATH
from src.models.worker import Worker

@dataclass
class FailoverConfig:
    """Configuration for failover mechanism"""
    enabled: bool
    check_interval: int
    max_retries: int
    retry_delay: int
    failback_delay: int
    auto_restart: bool
    health_check_timeout: int
    notification_enabled: bool
    emergency_shutdown_threshold: int
    state_sync_interval: int

class FailoverManager:
    """
    Manages failover for worker ants to ensure continuous operation
    even if VPS instances or individual workers fail.
    """
    def __init__(self, config_path: str = None):
        # Use provided config path or default
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self.config = self._load_config(self.config_path)
        self.worker_states = {}  # worker_id -> state
        self.worker_health = {}  # worker_id -> health info
        self.failed_workers = set()  # Set of failed worker IDs
        self.recovery_attempts = {}  # worker_id -> recovery attempt count
        self.backup_workers = {}  # worker_id -> backup worker info
        self.state_snapshots = {}  # worker_id -> last state snapshot
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for failover manager"""
        setup_logging("failover_manager", "failover_manager.log")
        
    def _load_config(self, config_path: str) -> FailoverConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            fo_config = config.get("failover", {})
            
        return FailoverConfig(
            enabled=fo_config.get("enabled", True),
            check_interval=fo_config.get("check_interval", 60),
            max_retries=fo_config.get("max_retries", 3),
            retry_delay=fo_config.get("retry_delay", 5),
            failback_delay=fo_config.get("failback_delay", 300),
            auto_restart=fo_config.get("auto_restart", True),
            health_check_timeout=fo_config.get("health_check_timeout", 10),
            notification_enabled=fo_config.get("notification_enabled", True),
            emergency_shutdown_threshold=fo_config.get("emergency_shutdown_threshold", 10),
            state_sync_interval=fo_config.get("state_sync_interval", 60)
        )
        
    async def register_worker(self, worker_id: str, worker: Worker) -> None:
        """Register a worker for failover monitoring"""
        logger.info(f"Registering worker {worker_id} for failover monitoring")
        
        # Store worker state
        self.worker_states[worker_id] = {
            "worker": worker,
            "registered_time": time.time(),
            "last_active_time": time.time(),
            "status": "active",
            "vps_id": None,  # Will be set later if using worker distribution
            "capital": 0.0,
            "trades_executed": 0,
            "total_profit": 0.0,
        }
        
        # Initialize health info
        self.worker_health[worker_id] = {
            "last_check_time": time.time(),
            "consecutive_failures": 0,
            "total_failures": 0,
            "total_recoveries": 0,
            "is_healthy": True,
        }
        
        # Create initial state snapshot
        await self._create_state_snapshot(worker_id)
        
    async def update_worker_state(self, worker_id: str, state_update: Dict[str, Any]) -> None:
        """Update the state information for a worker"""
        if worker_id not in self.worker_states:
            logger.warning(f"Attempted to update state for unknown worker: {worker_id}")
            return
            
        # Update state
        for key, value in state_update.items():
            if key in self.worker_states[worker_id]:
                self.worker_states[worker_id][key] = value
                
        # Mark as active
        self.worker_states[worker_id]["last_active_time"] = time.time()
        
        # If this was a failed worker that's now active, update health status
        if worker_id in self.failed_workers and state_update.get("status") == "active":
            self.failed_workers.remove(worker_id)
            self.worker_health[worker_id]["consecutive_failures"] = 0
            self.worker_health[worker_id]["is_healthy"] = True
            self.worker_health[worker_id]["total_recoveries"] += 1
            logger.info(f"Worker {worker_id} has recovered from failure")
            
    async def _create_state_snapshot(self, worker_id: str) -> None:
        """Create a snapshot of worker state for recovery purposes"""
        if worker_id not in self.worker_states:
            return
            
        # Get current worker state
        worker_state = self.worker_states[worker_id]
        
        # Extract relevant information for snapshot
        snapshot = {
            "timestamp": time.time(),
            "worker_id": worker_id,
            "status": worker_state["status"],
            "capital": worker_state["capital"],
            "trades_executed": worker_state["trades_executed"],
            "total_profit": worker_state["total_profit"],
            "vps_id": worker_state["vps_id"],
        }
        
        # Add additional state from worker if available
        worker = worker_state.get("worker")
        if worker:
            metrics = await worker.get_metrics()
            snapshot.update(metrics)
            
        # Store snapshot
        self.state_snapshots[worker_id] = snapshot
        
    async def start_monitoring(self):
        """Start monitoring workers for failures"""
        if not self.config.enabled:
            logger.info("Failover manager is disabled")
            return
            
        logger.info("Starting failover monitoring")
        
        # Start background tasks
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._state_sync_loop())
        
    async def _health_check_loop(self):
        """Continuous health check loop"""
        logger.info("Starting health check loop")
        
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute on error
                
    async def _state_sync_loop(self):
        """Loop to periodically sync worker states"""
        logger.info("Starting state sync loop")
        
        while True:
            try:
                for worker_id in list(self.worker_states.keys()):
                    if worker_id not in self.failed_workers:
                        await self._create_state_snapshot(worker_id)
                        
                await asyncio.sleep(self.config.state_sync_interval)
            except Exception as e:
                logger.error(f"Error in state sync loop: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute on error
                
    async def _perform_health_checks(self):
        """Perform health checks on all registered workers"""
        current_time = time.time()
        failed_worker_count = 0
        
        for worker_id, state in list(self.worker_states.items()):
            # Skip already failed workers
            if worker_id in self.failed_workers:
                failed_worker_count += 1
                continue
                
            # Check worker health
            worker = state.get("worker")
            if not worker:
                logger.warning(f"No worker object found for {worker_id}")
                continue
                
            # Check if the worker is still active
            is_healthy = await self._check_worker_health(worker_id, worker)
            
            if not is_healthy:
                logger.warning(f"Worker {worker_id} failed health check")
                
                # Update health status
                health_info = self.worker_health[worker_id]
                health_info["consecutive_failures"] += 1
                health_info["total_failures"] += 1
                health_info["is_healthy"] = False
                health_info["last_check_time"] = current_time
                
                # If exceeded max retries, mark as failed
                if health_info["consecutive_failures"] >= self.config.max_retries:
                    logger.error(f"Worker {worker_id} has failed after {health_info['consecutive_failures']} consecutive failures")
                    
                    # Mark as failed
                    self.failed_workers.add(worker_id)
                    state["status"] = "failed"
                    
                    # Attempt recovery
                    await self._handle_worker_failure(worker_id)
                    failed_worker_count += 1
            else:
                # Update health status as healthy
                health_info = self.worker_health[worker_id]
                health_info["consecutive_failures"] = 0
                health_info["is_healthy"] = True
                health_info["last_check_time"] = current_time
                
        # Emergency shutdown check
        if failed_worker_count >= self.config.emergency_shutdown_threshold:
            logger.critical(f"Emergency shutdown threshold reached: {failed_worker_count} failed workers")
            # In a real implementation, this would trigger emergency procedures
            # For now, we'll just log it
                
    async def _check_worker_health(self, worker_id: str, worker: Worker) -> bool:
        """Check if a worker is healthy"""
        try:
            # Get worker status with timeout
            status_task = asyncio.create_task(worker.get_metrics())
            status = await asyncio.wait_for(status_task, timeout=self.config.health_check_timeout)
            
            # Check if worker is active
            return status.get("is_active", False)
            
        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout for worker {worker_id}")
            return False
        except Exception as e:
            logger.error(f"Error checking health of worker {worker_id}: {str(e)}")
            return False
            
    async def _handle_worker_failure(self, worker_id: str) -> bool:
        """Handle a worker failure and attempt recovery"""
        if worker_id not in self.worker_states:
            return False
            
        state = self.worker_states[worker_id]
        recovery_count = self.recovery_attempts.get(worker_id, 0)
        
        logger.info(f"Handling failure of worker {worker_id}, attempt {recovery_count + 1}")
        
        # Increment recovery attempts
        self.recovery_attempts[worker_id] = recovery_count + 1
        
        # If automatic restart is enabled, try to restart the worker
        if self.config.auto_restart:
            try:
                worker = state.get("worker")
                if worker:
                    logger.info(f"Attempting to restart worker {worker_id}")
                    
                    # Stop the worker first
                    await worker.stop()
                    
                    # Wait for retry delay
                    await asyncio.sleep(self.config.retry_delay)
                    
                    # Start the worker again
                    await worker.start()
                    
                    # Check if restart was successful
                    status = await worker.get_metrics()
                    if status.get("is_active", False):
                        logger.info(f"Successfully restarted worker {worker_id}")
                        
                        # Update state
                        state["status"] = "active"
                        state["last_active_time"] = time.time()
                        self.failed_workers.remove(worker_id)
                        
                        # Reset health info
                        self.worker_health[worker_id]["consecutive_failures"] = 0
                        self.worker_health[worker_id]["is_healthy"] = True
                        self.worker_health[worker_id]["total_recoveries"] += 1
                        
                        return True
                        
            except Exception as e:
                logger.error(f"Failed to restart worker {worker_id}: {str(e)}")
                
        # If restart failed or is disabled, create backup worker if needed
        if worker_id not in self.backup_workers:
            await self._create_backup_worker(worker_id)
            
        return False
            
    async def _create_backup_worker(self, worker_id: str) -> Optional[str]:
        """Create a backup worker to replace a failed one"""
        if worker_id not in self.worker_states:
            return None
            
        state = self.worker_states[worker_id]
        snapshot = self.state_snapshots.get(worker_id)
        
        if not snapshot:
            logger.warning(f"No state snapshot available for worker {worker_id}")
            return None
            
        logger.info(f"Creating backup worker for {worker_id}")
        
        try:
            # Create a new worker with same config but different ID
            backup_id = f"{worker_id}_backup_{int(time.time())}"
            
            # Get config path from original worker
            worker = state.get("worker")
            config_path = worker.config_path if worker else self.config_path
            
            # Create backup worker
            backup_worker = Worker(
                worker_id=backup_id,
                config_path=str(config_path),
                wallet_id=None  # Will create a new wallet
            )
            
            # Start the backup worker
            await backup_worker.start()
            
            # Register the backup worker
            await self.register_worker(backup_id, backup_worker)
            
            # Associate with original worker
            self.backup_workers[worker_id] = backup_id
            
            logger.info(f"Successfully created backup worker {backup_id} for {worker_id}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Failed to create backup worker for {worker_id}: {str(e)}")
            return None
            
    async def get_worker_status(self, worker_id: str) -> Dict[str, Any]:
        """Get the current status of a worker"""
        if worker_id not in self.worker_states:
            return {"error": "Worker not found"}
            
        state = self.worker_states[worker_id]
        health = self.worker_health.get(worker_id, {})
        
        # Return combined state and health info
        return {
            "worker_id": worker_id,
            "status": state["status"],
            "last_active_time": state["last_active_time"],
            "registered_time": state["registered_time"],
            "is_healthy": health.get("is_healthy", False),
            "consecutive_failures": health.get("consecutive_failures", 0),
            "total_failures": health.get("total_failures", 0),
            "total_recoveries": health.get("total_recoveries", 0),
            "last_check_time": health.get("last_check_time", 0),
            "has_backup": worker_id in self.backup_workers,
            "backup_worker_id": self.backup_workers.get(worker_id),
            "capital": state["capital"],
            "trades_executed": state["trades_executed"],
            "total_profit": state["total_profit"],
        }
        
    def get_failover_stats(self) -> Dict[str, Any]:
        """Get overall failover statistics"""
        total_workers = len(self.worker_states)
        active_workers = total_workers - len(self.failed_workers)
        
        return {
            "total_workers": total_workers,
            "active_workers": active_workers,
            "failed_workers": len(self.failed_workers),
            "backup_workers": len(self.backup_workers),
            "health_check_interval": self.config.check_interval,
            "auto_restart": self.config.auto_restart,
            "failover_enabled": self.config.enabled,
        }
        
    async def force_failover(self, worker_id: str) -> bool:
        """Force failover for a worker (for testing)"""
        if worker_id not in self.worker_states:
            logger.warning(f"Attempted to force failover for unknown worker: {worker_id}")
            return False
            
        logger.info(f"Forcing failover for worker {worker_id}")
        
        # Mark as failed
        self.failed_workers.add(worker_id)
        self.worker_states[worker_id]["status"] = "failed"
        
        # Handle failure
        return await self._handle_worker_failure(worker_id)
        
    async def cleanup(self):
        """Clean up and stop all managed workers"""
        logger.info("Cleaning up failover manager")
        
        for worker_id, state in list(self.worker_states.items()):
            worker = state.get("worker")
            if worker:
                try:
                    await worker.stop()
                    logger.info(f"Stopped worker {worker_id}")
                except Exception as e:
                    logger.error(f"Error stopping worker {worker_id}: {str(e)}")
                    
        # Clear all state
        self.worker_states.clear()
        self.worker_health.clear()
        self.failed_workers.clear()
        self.recovery_attempts.clear()
        self.backup_workers.clear()
        self.state_snapshots.clear() 