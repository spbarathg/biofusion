"""
Prometheus metrics collection for AntBot

This module provides a unified interface for collecting metrics
about bot operation, wallet balances, and system health.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Union
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
from prometheus_client import multiprocess, CollectorRegistry
import os

logger = logging.getLogger(__name__)

# Global registry
REGISTRY = CollectorRegistry()

# Metrics definitions
# Transaction metrics
TRANSACTION_COUNT = Counter(
    'antbot_transaction_count_total', 
    'Total number of transactions processed',
    ['type', 'status', 'market'],
    registry=REGISTRY
)

TRANSACTION_AMOUNT = Histogram(
    'antbot_transaction_amount_sol',
    'Transaction amounts in SOL',
    ['type', 'market'],
    buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0),
    registry=REGISTRY
)

TRANSACTION_DURATION = Histogram(
    'antbot_transaction_duration_seconds',
    'Time taken to complete transactions',
    ['type'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY
)

# Wallet metrics
WALLET_BALANCE = Gauge(
    'antbot_wallet_balance_sol',
    'Current wallet balance in SOL',
    ['wallet_name', 'wallet_type'],
    registry=REGISTRY
)

WALLET_COUNT = Gauge(
    'antbot_wallet_count',
    'Number of wallets by type',
    ['wallet_type'],
    registry=REGISTRY
)

# Worker metrics
WORKER_STATUS = Gauge(
    'antbot_worker_status',
    'Worker online status (1=online, 0=offline)',
    ['worker_id'],
    registry=REGISTRY
)

WORKER_JOBS = Gauge(
    'antbot_worker_jobs',
    'Number of active jobs on worker',
    ['worker_id'],
    registry=REGISTRY
)

WORKER_MEMORY_USAGE = Gauge(
    'antbot_worker_memory_usage_bytes',
    'Memory used by worker process',
    ['worker_id'],
    registry=REGISTRY
)

# System metrics
API_REQUEST_DURATION = Summary(
    'antbot_api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint', 'method', 'status'],
    registry=REGISTRY
)

ERROR_COUNT = Counter(
    'antbot_error_count_total',
    'Number of errors encountered',
    ['component', 'error_type'],
    registry=REGISTRY
)

BACKUP_SIZE = Gauge(
    'antbot_backup_size_bytes',
    'Size of the latest backup in bytes',
    [],
    registry=REGISTRY
)

BACKUP_DURATION = Gauge(
    'antbot_backup_duration_seconds',
    'Time taken to complete the last backup',
    [],
    registry=REGISTRY
)

# Key rotation metrics
KEY_ROTATION_STATUS = Gauge(
    'antbot_key_rotation_status',
    'Status of the last key rotation (1=success, 0=failure)',
    [],
    registry=REGISTRY
)

KEY_ROTATION_TIMESTAMP = Gauge(
    'antbot_key_rotation_timestamp',
    'Unix timestamp of the last successful key rotation',
    [],
    registry=REGISTRY
)

KEY_ROTATION_DURATION = Gauge(
    'antbot_key_rotation_duration_seconds',
    'Time taken to complete the last key rotation',
    [],
    registry=REGISTRY
)

KEY_ROTATION_COUNT = Counter(
    'antbot_key_rotation_count_total',
    'Total number of key rotations performed',
    ['status'],
    registry=REGISTRY
)

# Utility functions
def record_transaction(
    tx_type: str, 
    status: str, 
    market: str, 
    amount: float, 
    duration: float
) -> None:
    """
    Record a transaction with Prometheus metrics
    
    Args:
        tx_type: Type of transaction (buy, sell, transfer)
        status: Transaction status (success, failure, pending)
        market: Market where transaction occurred
        amount: Amount in SOL
        duration: Transaction duration in seconds
    """
    try:
        TRANSACTION_COUNT.labels(tx_type, status, market).inc()
        TRANSACTION_AMOUNT.labels(tx_type, market).observe(amount)
        TRANSACTION_DURATION.labels(tx_type).observe(duration)
    except Exception as e:
        logger.error(f"Failed to record transaction metrics: {e}")

def update_wallet_balance(wallet_name: str, wallet_type: str, balance: float) -> None:
    """
    Update a wallet's balance in the metrics
    
    Args:
        wallet_name: Name of the wallet
        wallet_type: Type of wallet (queen, worker, user)
        balance: Current balance in SOL
    """
    try:
        WALLET_BALANCE.labels(wallet_name, wallet_type).set(balance)
    except Exception as e:
        logger.error(f"Failed to update wallet balance metric: {e}")

def update_wallet_counts(wallet_counts: Dict[str, int]) -> None:
    """
    Update wallet count metrics
    
    Args:
        wallet_counts: Dictionary of wallet type to count
    """
    try:
        for wallet_type, count in wallet_counts.items():
            WALLET_COUNT.labels(wallet_type).set(count)
    except Exception as e:
        logger.error(f"Failed to update wallet count metrics: {e}")

def update_worker_status(worker_id: str, online: bool, jobs: int, memory_bytes: int) -> None:
    """
    Update worker status metrics
    
    Args:
        worker_id: ID of the worker
        online: Whether the worker is online
        jobs: Number of active jobs
        memory_bytes: Memory usage in bytes
    """
    try:
        WORKER_STATUS.labels(worker_id).set(1 if online else 0)
        WORKER_JOBS.labels(worker_id).set(jobs)
        WORKER_MEMORY_USAGE.labels(worker_id).set(memory_bytes)
    except Exception as e:
        logger.error(f"Failed to update worker status metrics: {e}")

def record_api_request(endpoint: str, method: str, status: str, duration: float) -> None:
    """
    Record API request duration
    
    Args:
        endpoint: API endpoint path
        method: HTTP method
        status: HTTP status code
        duration: Request duration in seconds
    """
    try:
        API_REQUEST_DURATION.labels(endpoint, method, status).observe(duration)
    except Exception as e:
        logger.error(f"Failed to record API request metric: {e}")

def record_error(component: str, error_type: str) -> None:
    """
    Increment error counter for a component
    
    Args:
        component: Component name where error occurred
        error_type: Type/classification of error
    """
    try:
        ERROR_COUNT.labels(component, error_type).inc()
    except Exception as e:
        logger.error(f"Failed to record error metric: {e}")

def update_backup_metrics(size_bytes: int, duration_seconds: float) -> None:
    """
    Update backup metrics
    
    Args:
        size_bytes: Size of backup in bytes
        duration_seconds: Duration of backup process
    """
    try:
        BACKUP_SIZE.set(size_bytes)
        BACKUP_DURATION.set(duration_seconds)
    except Exception as e:
        logger.error(f"Failed to update backup metrics: {e}")

def update_key_rotation_metrics(success: bool, duration_seconds: float) -> None:
    """
    Update key rotation metrics
    
    Args:
        success: Whether the key rotation was successful
        duration_seconds: Duration of key rotation process in seconds
    """
    try:
        KEY_ROTATION_STATUS.set(1 if success else 0)
        KEY_ROTATION_DURATION.set(duration_seconds)
        if success:
            KEY_ROTATION_TIMESTAMP.set(time.time())
        KEY_ROTATION_COUNT.labels("success" if success else "failure").inc()
    except Exception as e:
        logger.error(f"Failed to update key rotation metrics: {e}")

def start_metrics_server(port: int = 8000) -> None:
    """
    Start the Prometheus metrics HTTP server
    
    Args:
        port: HTTP port to listen on
    """
    try:
        # Setup for multi-process mode if needed
        if 'prometheus_multiproc_dir' in os.environ:
            multiprocess.MultiProcessCollector(REGISTRY)
            
        start_http_server(port, registry=REGISTRY)
        logger.info(f"Started Prometheus metrics server on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}") 