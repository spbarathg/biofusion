import os
import yaml
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from src.utils.logging.logger import setup_logging
from src.core.paths import CONFIG_PATH

@dataclass
class VpsInstance:
    """Represents a VPS instance for worker distribution"""
    id: str
    hostname: str
    ip_address: str
    max_workers: int
    active_workers: int
    cpu_usage: float
    memory_usage: float
    performance_score: float
    is_available: bool
    region: str
    cost_per_hour: float

@dataclass
class WorkerDistributionConfig:
    """Configuration for worker distribution across VPS instances"""
    max_workers_per_vps: int
    min_vps_instances: int
    max_vps_instances: int
    cpu_threshold: float
    memory_threshold: float
    auto_scaling: bool
    load_balancing_interval: int
    distribution_strategy: str  # "even", "performance", "cost-optimized"
    preferred_regions: List[str]
    failover_enabled: bool

class WorkerDistribution:
    """
    Handles distribution of worker ants across multiple VPS instances
    with load balancing and failover capabilities
    """
    def __init__(self, config_path: str = None):
        # Use provided config path or default
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self.config = self._load_config(self.config_path)
        self.vps_instances: Dict[str, VpsInstance] = {}
        self.worker_assignments: Dict[str, str] = {}  # worker_id -> vps_id
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for worker distribution"""
        setup_logging("worker_distribution", "worker_distribution.log")
        
    def _load_config(self, config_path: str) -> WorkerDistributionConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            distribution_config = config.get("worker_distribution", {})
            
        return WorkerDistributionConfig(
            max_workers_per_vps=distribution_config.get("max_workers_per_vps", 5),
            min_vps_instances=distribution_config.get("min_vps_instances", 1),
            max_vps_instances=distribution_config.get("max_vps_instances", 10),
            cpu_threshold=distribution_config.get("cpu_threshold", 80.0),
            memory_threshold=distribution_config.get("memory_threshold", 80.0),
            auto_scaling=distribution_config.get("auto_scaling", True),
            load_balancing_interval=distribution_config.get("load_balancing_interval", 300),
            distribution_strategy=distribution_config.get("distribution_strategy", "performance"),
            preferred_regions=distribution_config.get("preferred_regions", ["us-east", "eu-west"]),
            failover_enabled=distribution_config.get("failover_enabled", True)
        )
        
    async def register_vps(self, vps_data: Dict[str, Any]) -> str:
        """Register a new VPS instance for worker distribution"""
        vps_id = vps_data.get("id", f"vps-{len(self.vps_instances) + 1}")
        
        self.vps_instances[vps_id] = VpsInstance(
            id=vps_id,
            hostname=vps_data.get("hostname", ""),
            ip_address=vps_data.get("ip_address", ""),
            max_workers=vps_data.get("max_workers", self.config.max_workers_per_vps),
            active_workers=vps_data.get("active_workers", 0),
            cpu_usage=vps_data.get("cpu_usage", 0.0),
            memory_usage=vps_data.get("memory_usage", 0.0),
            performance_score=vps_data.get("performance_score", 0.0),
            is_available=vps_data.get("is_available", True),
            region=vps_data.get("region", "unknown"),
            cost_per_hour=vps_data.get("cost_per_hour", 0.0)
        )
        
        logger.info(f"Registered VPS instance {vps_id} ({self.vps_instances[vps_id].hostname})")
        return vps_id
        
    async def update_vps_metrics(self, vps_id: str, metrics: Dict[str, Any]) -> None:
        """Update metrics for a VPS instance"""
        if vps_id not in self.vps_instances:
            logger.warning(f"Attempted to update metrics for unknown VPS: {vps_id}")
            return
            
        vps = self.vps_instances[vps_id]
        vps.cpu_usage = metrics.get("cpu_usage", vps.cpu_usage)
        vps.memory_usage = metrics.get("memory_usage", vps.memory_usage)
        vps.performance_score = metrics.get("performance_score", vps.performance_score)
        vps.is_available = metrics.get("is_available", vps.is_available)
        vps.active_workers = metrics.get("active_workers", vps.active_workers)
        
        # Check if this VPS is overloaded
        if vps.cpu_usage > self.config.cpu_threshold or vps.memory_usage > self.config.memory_threshold:
            logger.warning(f"VPS {vps_id} is overloaded - CPU: {vps.cpu_usage}%, Memory: {vps.memory_usage}%")
            # This will trigger rebalancing in the next cycle
            
    async def assign_worker(self, worker_id: str) -> Optional[str]:
        """Assign a worker to the optimal VPS instance"""
        available_vps = [
            vps for vps in self.vps_instances.values()
            if vps.is_available and vps.active_workers < vps.max_workers
        ]
        
        if not available_vps:
            logger.warning("No available VPS instances for worker assignment")
            return None
            
        # Select VPS based on distribution strategy
        target_vps = None
        
        if self.config.distribution_strategy == "even":
            # Distribute workers evenly across VPS instances
            target_vps = min(available_vps, key=lambda vps: vps.active_workers)
            
        elif self.config.distribution_strategy == "performance":
            # Prioritize VPS instances with better performance
            target_vps = max(available_vps, key=lambda vps: vps.performance_score)
            
        elif self.config.distribution_strategy == "cost-optimized":
            # Optimize for cost (fill cheaper instances first)
            # Sort by cost, then by available capacity
            target_vps = min(available_vps, key=lambda vps: (vps.cost_per_hour, -vps.active_workers))
            
        else:
            # Default to even distribution
            target_vps = min(available_vps, key=lambda vps: vps.active_workers)
        
        if target_vps:
            self.worker_assignments[worker_id] = target_vps.id
            target_vps.active_workers += 1
            logger.info(f"Assigned worker {worker_id} to VPS {target_vps.id}")
            return target_vps.id
            
        return None
        
    async def reassign_worker(self, worker_id: str) -> Optional[str]:
        """Reassign a worker to a different VPS instance (for load balancing/failover)"""
        if worker_id not in self.worker_assignments:
            logger.warning(f"Attempted to reassign unknown worker: {worker_id}")
            return await self.assign_worker(worker_id)
            
        current_vps_id = self.worker_assignments[worker_id]
        
        # Remove worker from current assignment
        if current_vps_id in self.vps_instances:
            self.vps_instances[current_vps_id].active_workers -= 1
            
        # Find new VPS
        del self.worker_assignments[worker_id]
        return await self.assign_worker(worker_id)
        
    async def get_worker_vps(self, worker_id: str) -> Optional[VpsInstance]:
        """Get the VPS instance assigned to a worker"""
        if worker_id not in self.worker_assignments:
            return None
            
        vps_id = self.worker_assignments[worker_id]
        return self.vps_instances.get(vps_id)
        
    async def balance_load(self) -> int:
        """
        Balance the worker load across VPS instances
        Returns the number of workers reassigned
        """
        logger.info("Starting load balancing cycle")
        
        # Find overloaded VPS instances
        overloaded_vps = [
            vps for vps in self.vps_instances.values()
            if (vps.cpu_usage > self.config.cpu_threshold or 
                vps.memory_usage > self.config.memory_threshold) and 
                vps.active_workers > 0
        ]
        
        if not overloaded_vps:
            logger.info("No overloaded VPS instances found")
            return 0
            
        # Find workers to reassign
        workers_to_move = []
        
        for vps in overloaded_vps:
            # Find workers assigned to this VPS
            vps_workers = [
                worker_id for worker_id, vps_id in self.worker_assignments.items()
                if vps_id == vps.id
            ]
            
            # Calculate how many workers to move (at least 1, at most half)
            move_count = max(1, min(len(vps_workers) // 2, 
                                    vps.active_workers - (vps.max_workers // 2)))
            
            # Add those workers to the list to move
            workers_to_move.extend(vps_workers[:move_count])
            
        # Reassign selected workers
        reassigned_count = 0
        
        for worker_id in workers_to_move:
            new_vps_id = await self.reassign_worker(worker_id)
            if new_vps_id:
                reassigned_count += 1
                logger.info(f"Reassigned worker {worker_id} from overloaded VPS to {new_vps_id}")
                
        logger.info(f"Load balancing complete - reassigned {reassigned_count} workers")
        return reassigned_count
        
    async def handle_vps_failure(self, vps_id: str) -> int:
        """
        Handle a VPS instance failure by reassigning its workers
        Returns the number of workers successfully reassigned
        """
        if not self.config.failover_enabled:
            logger.warning(f"Failover is disabled - not handling VPS failure: {vps_id}")
            return 0
            
        if vps_id not in self.vps_instances:
            logger.warning(f"Attempted to handle failure for unknown VPS: {vps_id}")
            return 0
            
        # Mark the VPS as unavailable
        self.vps_instances[vps_id].is_available = False
        logger.warning(f"VPS {vps_id} marked as failed - initiating failover")
        
        # Find workers assigned to this VPS
        workers_to_reassign = [
            worker_id for worker_id, assigned_vps in self.worker_assignments.items()
            if assigned_vps == vps_id
        ]
        
        # Reassign each worker
        reassigned_count = 0
        
        for worker_id in workers_to_reassign:
            new_vps_id = await self.reassign_worker(worker_id)
            if new_vps_id:
                reassigned_count += 1
                logger.info(f"Failover: Reassigned worker {worker_id} to VPS {new_vps_id}")
            else:
                logger.error(f"Failover: Could not reassign worker {worker_id} - no available VPS")
                
        logger.info(f"Failover complete for VPS {vps_id} - reassigned {reassigned_count}/{len(workers_to_reassign)} workers")
        return reassigned_count
        
    async def start_monitoring(self):
        """Start background monitoring and load balancing"""
        logger.info("Starting worker distribution monitoring")
        
        while True:
            try:
                # Update VPS metrics
                await self._update_all_vps_metrics()
                
                # Perform load balancing
                await self.balance_load()
                
                # Check for failed VPS instances
                await self._check_vps_health()
                
                # Sleep for the configured interval
                await asyncio.sleep(self.config.load_balancing_interval)
                
            except Exception as e:
                logger.error(f"Error in worker distribution monitoring: {str(e)}")
                await asyncio.sleep(60)  # Sleep for a minute on error
    
    async def _update_all_vps_metrics(self):
        """Update metrics for all VPS instances"""
        for vps_id, vps in self.vps_instances.items():
            try:
                # In a real implementation, this would call an API on the VPS
                # to get real metrics. For now, we'll just simulate it.
                await self._get_vps_metrics(vps_id)
            except Exception as e:
                logger.error(f"Error updating metrics for VPS {vps_id}: {str(e)}")
    
    async def _get_vps_metrics(self, vps_id: str):
        """Get current metrics from a VPS instance"""
        if vps_id not in self.vps_instances:
            return
            
        vps = self.vps_instances[vps_id]
        
        try:
            # In a real implementation, this would make an HTTP request
            # to the VPS to get metrics. For now, we'll just simulate it.
            # Placeholder metrics collection logic
            await self.update_vps_metrics(vps_id, {
                "is_available": True,  # Assume it's available for this simulation
                "cpu_usage": vps.cpu_usage,  # Keep existing value for now
                "memory_usage": vps.memory_usage,  # Keep existing value for now
                "performance_score": vps.performance_score  # Keep existing value for now
            })
        except Exception as e:
            logger.error(f"Failed to get metrics from VPS {vps_id}: {str(e)}")
            
    async def _check_vps_health(self):
        """Check the health of all VPS instances"""
        for vps_id, vps in list(self.vps_instances.items()):
            if not vps.is_available:
                continue  # Already marked as unavailable
                
            try:
                # In a real implementation, this would ping the VPS or check
                # a health endpoint. For now, we'll just simulate it.
                healthy = await self._check_vps_connectivity(vps_id)
                
                if not healthy:
                    logger.warning(f"VPS {vps_id} is not responding - initiating failover")
                    await self.handle_vps_failure(vps_id)
                    
            except Exception as e:
                logger.error(f"Error checking health of VPS {vps_id}: {str(e)}")
                
    async def _check_vps_connectivity(self, vps_id: str) -> bool:
        """Check if a VPS instance is responsive"""
        if vps_id not in self.vps_instances:
            return False
            
        vps = self.vps_instances[vps_id]
        
        # In a real implementation, this would ping the VPS or check a health endpoint
        # For now, we'll just assume it's healthy
        return True
        
    def get_vps_list(self) -> List[Dict[str, Any]]:
        """Get a list of all VPS instances"""
        return [
            {
                "id": vps.id,
                "hostname": vps.hostname,
                "ip_address": vps.ip_address,
                "max_workers": vps.max_workers,
                "active_workers": vps.active_workers,
                "cpu_usage": vps.cpu_usage,
                "memory_usage": vps.memory_usage,
                "performance_score": vps.performance_score,
                "is_available": vps.is_available,
                "region": vps.region,
                "cost_per_hour": vps.cost_per_hour
            }
            for vps in self.vps_instances.values()
        ]
        
    def get_worker_assignments(self) -> Dict[str, str]:
        """Get the current worker assignments"""
        return self.worker_assignments.copy() 