import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Set
from loguru import logger
from src.logging.log_config import setup_logging

class FailoverManager:
    def __init__(self, config_path: str = None):
        # Set up logging
        setup_logging("failover_manager", "failover_manager.log")
        logger.info("Initializing FailoverManager...")
        
        # State tracking
        self.is_running = False
        self.vps_instances: Dict[str, Dict] = {}
        self.failed_vps: Set[str] = set()
        self.worker_backups: Dict[str, Dict[str, Dict]] = {}
        self.health_check_timeouts: Dict[str, float] = {}
        self.retry_counts: Dict[str, int] = {}
        
        # Configuration
        self.config_path = config_path
        self.max_retries = 3
        self.health_check_timeout = 30  # seconds
        self.backup_retention_hours = 24
        self.state_file = "failover_state.json"
        
        # Load persisted state if exists
        self._load_state()
        
    def _load_state(self):
        """Load persisted state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.vps_instances = state.get('vps_instances', {})
                    self.failed_vps = set(state.get('failed_vps', []))
                    self.worker_backups = state.get('worker_backups', {})
                    logger.info("Loaded persisted state successfully")
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            
    def _save_state(self):
        """Save current state to file."""
        try:
            state = {
                'vps_instances': self.vps_instances,
                'failed_vps': list(self.failed_vps),
                'worker_backups': self.worker_backups,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug("Saved state successfully")
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            
    async def start_monitoring(self):
        """Start monitoring for failures."""
        logger.info("Starting failover monitoring...")
        self.is_running = True
        
        while self.is_running:
            try:
                # Check VPS health
                await self._check_vps_health()
                
                # Check worker health
                await self._check_worker_health()
                
                # Sync state if needed
                await self._sync_state()
                
                # Save state periodically
                self._save_state()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in failover monitoring: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def stop_monitoring(self):
        """Stop monitoring for failures."""
        logger.info("Stopping failover monitoring...")
        self.is_running = False
        self._save_state()  # Save final state
        
    async def register_vps(self, vps_instance: Dict):
        """Register a new VPS instance."""
        try:
            vps_id = vps_instance['id']
            if vps_id in self.vps_instances:
                logger.warning(f"VPS {vps_id} already registered, updating...")
            
            self.vps_instances[vps_id] = {
                **vps_instance,
                'last_health_check': datetime.now().isoformat(),
                'is_available': True,
                'workers': vps_instance.get('workers', [])
            }
            self.health_check_timeouts[vps_id] = 0
            self.retry_counts[vps_id] = 0
            
            logger.info(f"Registered VPS instance: {vps_id}")
            self._save_state()
            
        except Exception as e:
            logger.error(f"Error registering VPS: {str(e)}")
            raise
        
    async def _check_vps_health(self):
        """Check health of all VPS instances."""
        current_time = datetime.now().timestamp()
        
        for vps_id, vps in self.vps_instances.items():
            try:
                # Check if health check is overdue
                last_check = datetime.fromisoformat(vps['last_health_check']).timestamp()
                if current_time - last_check > self.health_check_timeout:
                    self.health_check_timeouts[vps_id] = current_time - last_check
                    
                    # Increment retry count
                    self.retry_counts[vps_id] = self.retry_counts.get(vps_id, 0) + 1
                    
                    if self.retry_counts[vps_id] >= self.max_retries:
                        if vps_id not in self.failed_vps:
                            logger.warning(f"VPS {vps_id} failed health check after {self.max_retries} retries")
                            await self._handle_vps_failure(vps_id)
                    else:
                        logger.warning(f"VPS {vps_id} health check overdue (attempt {self.retry_counts[vps_id]})")
                else:
                    # Reset retry count on successful check
                    self.retry_counts[vps_id] = 0
                    
            except Exception as e:
                logger.error(f"Error checking health of VPS {vps_id}: {str(e)}")
                
    async def _check_worker_health(self):
        """Check health of all workers."""
        for vps_id, vps in self.vps_instances.items():
            if vps_id in self.failed_vps:
                continue
                
            try:
                workers = vps.get('workers', [])
                for worker_id in workers:
                    try:
                        # Implement actual worker health check here
                        is_healthy = await self._perform_worker_health_check(worker_id)
                        
                        if not is_healthy:
                            logger.warning(f"Worker {worker_id} on VPS {vps_id} is unhealthy")
                            await self._handle_worker_failure(vps_id, worker_id)
                            
                    except Exception as e:
                        logger.error(f"Error checking health of worker {worker_id}: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error checking worker health on VPS {vps_id}: {str(e)}")
                
    async def _perform_worker_health_check(self, worker_id: str) -> bool:
        """Perform actual health check for a worker."""
        try:
            # Implement your actual worker health check logic here
            # This could involve checking process status, API responses, etc.
            return True
        except Exception as e:
            logger.error(f"Error in worker health check: {str(e)}")
            return False
                
    async def _handle_vps_failure(self, vps_id: str):
        """Handle VPS failure."""
        try:
            # Mark VPS as failed
            self.failed_vps.add(vps_id)
            self.vps_instances[vps_id]['is_available'] = False
            
            # Create backups of workers
            workers = self.vps_instances[vps_id].get('workers', [])
            for worker_id in workers:
                await self._create_worker_backup(vps_id, worker_id)
                
            logger.info(f"Created backups for {len(workers)} workers from failed VPS {vps_id}")
            self._save_state()
            
        except Exception as e:
            logger.error(f"Error handling VPS failure for {vps_id}: {str(e)}")
            
    async def _handle_vps_recovery(self, vps_id: str):
        """Handle VPS recovery."""
        try:
            # Mark VPS as available
            self.failed_vps.remove(vps_id)
            self.vps_instances[vps_id]['is_available'] = True
            self.vps_instances[vps_id]['last_health_check'] = datetime.now().isoformat()
            
            # Reset retry count
            self.retry_counts[vps_id] = 0
            
            # Restore workers from backups
            backups = self.worker_backups.get(vps_id, {})
            for worker_id, backup in backups.items():
                await self._restore_worker(vps_id, worker_id, backup)
                
            logger.info(f"Restored {len(backups)} workers to recovered VPS {vps_id}")
            self._save_state()
            
        except Exception as e:
            logger.error(f"Error handling VPS recovery for {vps_id}: {str(e)}")
            
    async def _handle_worker_failure(self, vps_id: str, worker_id: str):
        """Handle worker failure."""
        try:
            # Create backup of worker
            await self._create_worker_backup(vps_id, worker_id)
            
            # Remove failed worker
            if worker_id in self.vps_instances[vps_id].get('workers', []):
                self.vps_instances[vps_id]['workers'].remove(worker_id)
                
            logger.info(f"Created backup for failed worker {worker_id} on VPS {vps_id}")
            self._save_state()
            
        except Exception as e:
            logger.error(f"Error handling worker failure for {worker_id} on VPS {vps_id}: {str(e)}")
            
    async def _create_worker_backup(self, vps_id: str, worker_id: str):
        """Create a backup of a worker."""
        try:
            # Get worker state and configuration
            worker_state = await self._get_worker_state(worker_id)
            
            backup = {
                'worker_id': worker_id,
                'vps_id': vps_id,
                'state': worker_state,
                'timestamp': datetime.now().isoformat(),
                'config': self.vps_instances[vps_id].get('worker_configs', {}).get(worker_id, {})
            }
            
            if vps_id not in self.worker_backups:
                self.worker_backups[vps_id] = {}
                
            self.worker_backups[vps_id][worker_id] = backup
            logger.debug(f"Created backup for worker {worker_id} on VPS {vps_id}")
            
        except Exception as e:
            logger.error(f"Error creating backup for worker {worker_id} on VPS {vps_id}: {str(e)}")
            
    async def _get_worker_state(self, worker_id: str) -> Dict:
        """Get current state of a worker."""
        try:
            # Implement actual worker state collection here
            return {
                'status': 'unknown',
                'last_active': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting worker state: {str(e)}")
            return {}
            
    async def _restore_worker(self, vps_id: str, worker_id: str, backup: Dict):
        """Restore a worker from backup."""
        try:
            # Verify backup is still valid
            backup_time = datetime.fromisoformat(backup['timestamp'])
            if (datetime.now() - backup_time).total_seconds() > (self.backup_retention_hours * 3600):
                logger.warning(f"Backup for worker {worker_id} is too old, skipping restoration")
                return
                
            # Remove from backups
            if vps_id in self.worker_backups and worker_id in self.worker_backups[vps_id]:
                del self.worker_backups[vps_id][worker_id]
                
            # Add to VPS workers
            if worker_id not in self.vps_instances[vps_id].get('workers', []):
                self.vps_instances[vps_id]['workers'].append(worker_id)
                
            # Restore worker configuration
            if 'config' in backup:
                if 'worker_configs' not in self.vps_instances[vps_id]:
                    self.vps_instances[vps_id]['worker_configs'] = {}
                self.vps_instances[vps_id]['worker_configs'][worker_id] = backup['config']
                
            logger.debug(f"Restored worker {worker_id} to VPS {vps_id}")
            self._save_state()
            
        except Exception as e:
            logger.error(f"Error restoring worker {worker_id} to VPS {vps_id}: {str(e)}")
            
    async def _sync_state(self):
        """Synchronize state across VPS instances."""
        try:
            logger.debug("Syncing state across VPS instances...")
            
            # Update VPS states
            for vps_id, vps in self.vps_instances.items():
                if vps_id in self.failed_vps:
                    vps['is_available'] = False
                else:
                    vps['is_available'] = True
                    
            # Clean up old backups
            current_time = datetime.now()
            for vps_id in list(self.worker_backups.keys()):
                for worker_id in list(self.worker_backups[vps_id].keys()):
                    backup = self.worker_backups[vps_id][worker_id]
                    backup_time = datetime.fromisoformat(backup['timestamp'])
                    if (current_time - backup_time).total_seconds() > (self.backup_retention_hours * 3600):
                        del self.worker_backups[vps_id][worker_id]
                        
            logger.debug("State sync completed")
            self._save_state()
            
        except Exception as e:
            logger.error(f"Error syncing state: {str(e)}")
            
    def get_status(self) -> Dict:
        """Get the current status of the failover manager."""
        return {
            "is_running": self.is_running,
            "vps_count": len(self.vps_instances),
            "failed_vps_count": len(self.failed_vps),
            "backup_count": sum(
                len(backups) for backups in self.worker_backups.values()
            ),
            "health_check_timeouts": self.health_check_timeouts,
            "retry_counts": self.retry_counts
        }
        
    async def cleanup(self):
        """Clean up resources and stop monitoring."""
        try:
            await self.stop_monitoring()
            self._save_state()
            logger.info("Failover manager cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}") 