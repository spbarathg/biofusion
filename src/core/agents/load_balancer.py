import os
import yaml
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import time
import random
import hashlib

from src.utils.logging.logger import setup_logging
from src.core.paths import CONFIG_PATH
from src.core.agents.worker_distribution import WorkerDistribution

@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer"""
    enabled: bool
    check_interval: int
    health_check_path: str
    max_retries: int
    retry_delay: int
    stickiness: bool
    distribution_method: str  # "round-robin", "least-connections", "ip-hash", "weighted"
    strategy_update_interval: int

class LoadBalancer:
    """
    Load balancer implementation for distributing workload across multiple VPS instances
    running worker ants. Provides traffic management, health checks and failover capabilities.
    """
    def __init__(self, worker_distribution: WorkerDistribution, config_path: str = None):
        # Use provided config path or default
        self.config_path = Path(config_path) if config_path else CONFIG_PATH
        self.config = self._load_config(self.config_path)
        self.worker_distribution = worker_distribution
        self.current_round_robin_index = 0
        self.connection_counts = {}
        self.sticky_mappings = {}
        self.vps_weights = {}
        self.last_health_checks = {}
        self.unhealthy_vps = set()
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for load balancer"""
        setup_logging("load_balancer", "load_balancer.log")
        
    def _load_config(self, config_path: str) -> LoadBalancerConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            lb_config = config.get("load_balancer", {})
            
        return LoadBalancerConfig(
            enabled=lb_config.get("enabled", True),
            check_interval=lb_config.get("check_interval", 60),
            health_check_path=lb_config.get("health_check_path", "/health"),
            max_retries=lb_config.get("max_retries", 3),
            retry_delay=lb_config.get("retry_delay", 5),
            stickiness=lb_config.get("stickiness", True),
            distribution_method=lb_config.get("distribution_method", "least-connections"),
            strategy_update_interval=lb_config.get("strategy_update_interval", 300)
        )
        
    async def start(self):
        """Start the load balancer"""
        if not self.config.enabled:
            logger.info("Load balancer is disabled")
            return
            
        logger.info(f"Starting load balancer with {self.config.distribution_method} distribution method")
        
        # Initialize weights based on performance score
        await self._update_vps_weights()
        
        # Start health check and monitoring tasks
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._update_strategy_loop())
        
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
                
    async def _update_strategy_loop(self):
        """Loop to periodically update load balancing strategy"""
        logger.info("Starting strategy update loop")
        
        while True:
            try:
                await self._update_vps_weights()
                await asyncio.sleep(self.config.strategy_update_interval)
            except Exception as e:
                logger.error(f"Error in strategy update loop: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute on error
    
    async def _perform_health_checks(self):
        """Perform health checks on all VPS instances"""
        vps_list = self.worker_distribution.get_vps_list()
        
        for vps in vps_list:
            vps_id = vps["id"]
            if not vps["is_available"]:
                continue  # Skip unavailable VPS
                
            is_healthy = await self._check_vps_health(vps)
            
            if not is_healthy and vps_id not in self.unhealthy_vps:
                logger.warning(f"VPS {vps_id} failed health check")
                self.unhealthy_vps.add(vps_id)
                
                # After several consecutive failures, mark as failed
                if self._get_consecutive_failures(vps_id) >= self.config.max_retries:
                    logger.error(f"VPS {vps_id} marked as unhealthy after {self.config.max_retries} failures")
                    # Initiate failover
                    asyncio.create_task(self.worker_distribution.handle_vps_failure(vps_id))
            
            elif is_healthy and vps_id in self.unhealthy_vps:
                logger.info(f"VPS {vps_id} recovered and is healthy again")
                self.unhealthy_vps.remove(vps_id)
                self.last_health_checks[vps_id] = {"status": "healthy", "time": time.time(), "failures": 0}
                
    def _get_consecutive_failures(self, vps_id: str) -> int:
        """Get the number of consecutive health check failures for a VPS"""
        if vps_id not in self.last_health_checks:
            return 0
            
        return self.last_health_checks.get(vps_id, {}).get("failures", 0)
                
    async def _check_vps_health(self, vps: Dict[str, Any]) -> bool:
        """Check the health of a VPS instance"""
        vps_id = vps["id"]
        ip_address = vps["ip_address"]
        
        # In a real implementation, this would make an HTTP request to a health endpoint
        # For now, we'll just simulate it with a basic check
        
        # Simulate a check with 95% success rate
        is_healthy = random.random() > 0.05
        
        # Update health check record
        if vps_id not in self.last_health_checks:
            self.last_health_checks[vps_id] = {"status": "unknown", "time": time.time(), "failures": 0}
            
        if is_healthy:
            self.last_health_checks[vps_id] = {"status": "healthy", "time": time.time(), "failures": 0}
        else:
            failures = self.last_health_checks[vps_id].get("failures", 0) + 1
            self.last_health_checks[vps_id] = {"status": "unhealthy", "time": time.time(), "failures": failures}
            
        return is_healthy
        
    async def _update_vps_weights(self):
        """Update weights for VPS instances based on performance metrics"""
        vps_list = self.worker_distribution.get_vps_list()
        
        for vps in vps_list:
            vps_id = vps["id"]
            
            # Calculate weight based on a combination of factors:
            # - Performance score (higher is better)
            # - CPU/Memory usage (lower is better)
            # - Region (preferred regions get bonus)
            
            # Base weight from performance score (0-100)
            base_weight = vps["performance_score"] * 100
            
            # Adjust based on resource usage (0-50 penalty)
            usage_penalty = (vps["cpu_usage"] + vps["memory_usage"]) / 4  # Average of CPU and memory, scaled
            
            # Region bonus (10 for preferred regions)
            region_bonus = 10 if vps["region"] in self.worker_distribution.config.preferred_regions else 0
            
            # Calculate final weight (minimum 1)
            weight = max(1, base_weight - usage_penalty + region_bonus)
            
            # Update weight
            self.vps_weights[vps_id] = weight
            
        logger.debug(f"Updated VPS weights: {self.vps_weights}")
        
    def select_vps(self, client_id: str = None, client_ip: str = None) -> Optional[str]:
        """
        Select a VPS instance based on the configured load balancing method
        
        Args:
            client_id: Optional client identifier for sticky sessions
            client_ip: Optional client IP for IP-hash based distribution
            
        Returns:
            str: Selected VPS ID or None if no suitable VPS is available
        """
        vps_list = [vps for vps in self.worker_distribution.get_vps_list() 
                    if vps["is_available"] and vps["id"] not in self.unhealthy_vps]
        
        if not vps_list:
            logger.warning("No healthy VPS instances available")
            return None
            
        # Check for sticky session if enabled
        if self.config.stickiness and client_id and client_id in self.sticky_mappings:
            sticky_vps_id = self.sticky_mappings[client_id]
            # Verify VPS is still healthy
            for vps in vps_list:
                if vps["id"] == sticky_vps_id:
                    return sticky_vps_id
                    
        # Apply load balancing strategy
        selected_vps_id = None
        
        if self.config.distribution_method == "round-robin":
            selected_vps_id = self._round_robin_select(vps_list)
            
        elif self.config.distribution_method == "least-connections":
            selected_vps_id = self._least_connections_select(vps_list)
            
        elif self.config.distribution_method == "ip-hash" and client_ip:
            selected_vps_id = self._ip_hash_select(vps_list, client_ip)
            
        elif self.config.distribution_method == "weighted":
            selected_vps_id = self._weighted_select(vps_list)
            
        else:
            # Default to least connections
            selected_vps_id = self._least_connections_select(vps_list)
            
        # Update sticky session mapping if enabled
        if self.config.stickiness and client_id and selected_vps_id:
            self.sticky_mappings[client_id] = selected_vps_id
            
        # Update connection count
        if selected_vps_id:
            self.connection_counts[selected_vps_id] = self.connection_counts.get(selected_vps_id, 0) + 1
            
        return selected_vps_id
        
    def _round_robin_select(self, vps_list: List[Dict[str, Any]]) -> str:
        """Round-robin VPS selection"""
        if not vps_list:
            return None
            
        # Increment index and wrap around
        self.current_round_robin_index = (self.current_round_robin_index + 1) % len(vps_list)
        return vps_list[self.current_round_robin_index]["id"]
        
    def _least_connections_select(self, vps_list: List[Dict[str, Any]]) -> str:
        """Select VPS with the least active connections"""
        if not vps_list:
            return None
            
        # Find VPS with the least active workers
        return min(vps_list, key=lambda vps: vps["active_workers"])["id"]
        
    def _ip_hash_select(self, vps_list: List[Dict[str, Any]], client_ip: str) -> str:
        """IP-hash based VPS selection"""
        if not vps_list:
            return None
            
        # Create a hash of the client IP
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        
        # Map hash to VPS index
        index = hash_value % len(vps_list)
        return vps_list[index]["id"]
        
    def _weighted_select(self, vps_list: List[Dict[str, Any]]) -> str:
        """Weighted random selection of VPS"""
        if not vps_list:
            return None
            
        # Get weights for available VPS instances
        weights = []
        for vps in vps_list:
            weight = self.vps_weights.get(vps["id"], 1)
            weights.append(weight)
            
        # Select based on weights
        total = sum(weights)
        if total == 0:
            # If all weights are 0, use equal weights
            return random.choice(vps_list)["id"]
            
        # Random value between 0 and total weight
        r = random.uniform(0, total)
        upto = 0
        
        # Find the VPS whose weight range contains r
        for i, w in enumerate(weights):
            if upto + w >= r:
                return vps_list[i]["id"]
            upto += w
            
        # Fallback to the last VPS
        return vps_list[-1]["id"]
        
    def release_connection(self, vps_id: str):
        """Mark a connection as completed/released"""
        if vps_id in self.connection_counts and self.connection_counts[vps_id] > 0:
            self.connection_counts[vps_id] -= 1
            
    def get_vps_stats(self) -> Dict[str, Any]:
        """Get statistics about VPS usage"""
        return {
            "connection_counts": self.connection_counts.copy(),
            "unhealthy_vps": list(self.unhealthy_vps),
            "weights": self.vps_weights.copy(),
            "health_checks": self.last_health_checks.copy()
        }
        
    def get_load_balancer_info(self) -> Dict[str, Any]:
        """Get information about the load balancer"""
        return {
            "enabled": self.config.enabled,
            "distribution_method": self.config.distribution_method,
            "stickiness": self.config.stickiness,
            "health_check_interval": self.config.check_interval,
            "active_connections": sum(self.connection_counts.values()),
            "sticky_sessions": len(self.sticky_mappings),
            "unhealthy_vps_count": len(self.unhealthy_vps)
        } 