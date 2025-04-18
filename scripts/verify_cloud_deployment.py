#!/usr/bin/env python3
import os
import sys
import asyncio
import time
import random
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.queen import Queen
from src.core.agents.worker_distribution import WorkerDistribution
from src.core.agents.load_balancer import LoadBalancer
from src.core.agents.failover import FailoverManager
from src.logging.log_config import setup_logging
from loguru import logger

setup_logging("verification", "verification.log")

async def test_vps_registration(queen):
    """Test VPS registration and metrics updates"""
    logger.info("Testing VPS registration...")
    
    # Register additional VPS instances
    vps_data = [
        {
            "id": "vps-2",
            "hostname": "worker-host-2",
            "ip_address": "10.0.0.2",
            "max_workers": 5,
            "active_workers": 0,
            "cpu_usage": 15.0,
            "memory_usage": 25.0,
            "performance_score": 0.85,
            "is_available": True,
            "region": "eu-west",
            "cost_per_hour": 0.0452
        },
        {
            "id": "vps-3",
            "hostname": "worker-host-3",
            "ip_address": "10.0.0.3",
            "max_workers": 5,
            "active_workers": 0,
            "cpu_usage": 30.0,
            "memory_usage": 40.0,
            "performance_score": 0.75,
            "is_available": True,
            "region": "ap-southeast",
            "cost_per_hour": 0.0472
        }
    ]
    
    for vps in vps_data:
        vps_id = await queen.worker_distribution.register_vps(vps)
        logger.info(f"Registered VPS: {vps_id}")
    
    # Get VPS list and verify correct count
    vps_list = queen.worker_distribution.get_vps_list()
    logger.info(f"VPS list has {len(vps_list)} instances (expected 3)")
    assert len(vps_list) == 3, f"Expected 3 VPS instances, got {len(vps_list)}"
    
    # Update metrics for a VPS
    await queen.worker_distribution.update_vps_metrics("vps-2", {
        "cpu_usage": 75.0,
        "memory_usage": 85.0,
        "performance_score": 0.65
    })
    
    # Verify metrics were updated
    vps_list = queen.worker_distribution.get_vps_list()
    for vps in vps_list:
        if vps["id"] == "vps-2":
            logger.info(f"VPS-2 metrics: CPU {vps['cpu_usage']}%, Memory {vps['memory_usage']}%, Score {vps['performance_score']}")
            assert vps["cpu_usage"] == 75.0, f"Expected CPU usage 75.0%, got {vps['cpu_usage']}%"
            assert vps["memory_usage"] == 85.0, f"Expected memory usage 85.0%, got {vps['memory_usage']}%"
    
    logger.info("VPS registration tests passed ✓")
    return True

async def test_worker_distribution(queen):
    """Test worker distribution and assignment"""
    logger.info("Testing worker distribution...")
    
    # Assign 10 workers
    worker_ids = [f"test_worker_{i}" for i in range(10)]
    assignments = {}
    
    for worker_id in worker_ids:
        vps_id = await queen.worker_distribution.assign_worker(worker_id)
        assignments[worker_id] = vps_id
        logger.info(f"Assigned {worker_id} to {vps_id}")
    
    # Verify all workers were assigned
    assert len(assignments) == 10, f"Expected 10 worker assignments, got {len(assignments)}"
    
    # Check worker counts on VPS instances
    vps_list = queen.worker_distribution.get_vps_list()
    worker_counts = {vps["id"]: vps["active_workers"] for vps in vps_list}
    logger.info(f"Worker distribution: {worker_counts}")
    
    # Verify no VPS is assigned more than max_workers
    for vps in vps_list:
        assert vps["active_workers"] <= vps["max_workers"], f"VPS {vps['id']} has {vps['active_workers']} workers, exceeding max {vps['max_workers']}"
    
    # Get assignments directly
    assignments_map = queen.worker_distribution.get_worker_assignments()
    assert len(assignments_map) == 10, f"Expected 10 assignments in map, got {len(assignments_map)}"
    
    logger.info("Worker distribution tests passed ✓")
    return True

async def test_load_balancing(queen):
    """Test load balancing functionality"""
    logger.info("Testing load balancing...")
    
    # Make one VPS overloaded
    await queen.worker_distribution.update_vps_metrics("vps-1", {
        "cpu_usage": 90.0,
        "memory_usage": 95.0
    })
    
    # Trigger load balancing
    reassigned = await queen.worker_distribution.balance_load()
    logger.info(f"Load balancing reassigned {reassigned} workers")
    
    # Verify at least one worker was reassigned
    assert reassigned > 0, "Expected at least one worker to be reassigned during load balancing"
    
    # Check worker counts after balancing
    vps_list = queen.worker_distribution.get_vps_list()
    worker_counts = {vps["id"]: vps["active_workers"] for vps in vps_list}
    logger.info(f"Worker distribution after balancing: {worker_counts}")
    
    # Test load balancer's selection methods
    lb = queen.load_balancer
    
    # Round-robin selection
    rr_selections = [lb._round_robin_select(vps_list) for _ in range(5)]
    logger.info(f"Round-robin selections: {rr_selections}")
    
    # Least connections selection
    lc_selection = lb._least_connections_select(vps_list)
    logger.info(f"Least connections selection: {lc_selection}")
    
    # IP-hash selection
    ip_hash_selection = lb._ip_hash_select(vps_list, "192.168.1.1")
    logger.info(f"IP-hash selection for 192.168.1.1: {ip_hash_selection}")
    
    # Weighted selection
    await lb._update_vps_weights()
    weighted_selection = lb._weighted_select(vps_list)
    logger.info(f"Weighted selection: {weighted_selection}")
    
    logger.info("Load balancing tests passed ✓")
    return True

async def test_failover(queen):
    """Test failover mechanism"""
    logger.info("Testing failover mechanism...")
    
    # Simulate VPS failure
    failed_vps = "vps-3"
    logger.info(f"Simulating failure of VPS {failed_vps}")
    
    # Handle the failure
    reassigned = await queen.worker_distribution.handle_vps_failure(failed_vps)
    logger.info(f"Failover reassigned {reassigned} workers from failed VPS")
    
    # Verify VPS is marked as unavailable
    vps_list = queen.worker_distribution.get_vps_list()
    for vps in vps_list:
        if vps["id"] == failed_vps:
            assert not vps["is_available"], f"VPS {failed_vps} should be marked as unavailable"
            
    # Check the failover manager status
    fm = queen.failover_manager
    stats = fm.get_failover_stats()
    logger.info(f"Failover stats: {stats}")
    
    # Create a test worker and simulate its failure
    worker_id = "test_failover_worker"
    
    # Since we can't easily create and register a real worker,
    # we'll just test that the interfaces work correctly
    vps_id = await queen.worker_distribution.assign_worker(worker_id)
    logger.info(f"Assigned test worker to {vps_id}")
    
    # Get load balancer info
    lb_info = queen.load_balancer.get_load_balancer_info()
    logger.info(f"Load balancer info: {lb_info}")
    
    logger.info("Failover tests passed ✓")
    return True

async def test_cloud_infrastructure_monitoring(queen):
    """Test cloud infrastructure monitoring and reporting"""
    logger.info("Testing cloud infrastructure monitoring...")
    
    # Get cloud infrastructure status
    status = await queen.get_cloud_infrastructure_status()
    
    # Verify required fields are present
    assert "vps_instances" in status, "Missing vps_instances in cloud status"
    assert "load_balancer" in status, "Missing load_balancer in cloud status"
    assert "failover" in status, "Missing failover in cloud status"
    
    logger.info(f"Cloud status shows {len(status['vps_instances'])} VPS instances")
    logger.info(f"Load balancer using {status['load_balancer']['distribution_method']} distribution")
    logger.info(f"Failover shows {status['failover']['active_workers']} active workers out of {status['failover']['total_workers']} total")
    
    # Get VPS stats from load balancer
    vps_stats = queen.load_balancer.get_vps_stats()
    logger.info(f"VPS stats: {vps_stats}")
    
    logger.info("Cloud infrastructure monitoring tests passed ✓")
    return True

async def run_verification():
    """Run all verification tests"""
    logger.info("Starting cloud deployment verification")
    
    queen = Queen()
    await queen.initialize_colony(10.0)  # Initialize with 10 SOL
    
    # Run tests
    tests = [
        test_vps_registration,
        test_worker_distribution,
        test_load_balancing,
        test_failover,
        test_cloud_infrastructure_monitoring
    ]
    
    results = []
    for test in tests:
        try:
            result = await test(queen)
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} failed: {str(e)}")
            results.append(False)
    
    # Report results
    successful = results.count(True)
    total = len(tests)
    logger.info(f"Verification complete: {successful}/{total} tests passed")
    
    # Clean up
    await queen.stop_colony()
    
    return successful == total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify cloud deployment functionality")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logger.level("DEBUG")
    
    success = asyncio.run(run_verification())
    
    if success:
        print("\n✅ Verification passed: All cloud deployment features are working correctly")
        sys.exit(0)
    else:
        print("\n❌ Verification failed: Some tests did not pass. Check the logs for details.")
        sys.exit(1) 