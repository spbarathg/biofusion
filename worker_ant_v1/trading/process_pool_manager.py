"""
PROCESS POOL MANAGER - ADVANCED CONCURRENCY FOR ML INFERENCE
===========================================================

ProcessPoolExecutor integration for CPU-intensive machine learning inference tasks.
Prevents the main asyncio event loop from being blocked during model computation.

Features:
- Separate processes for Oracle Ant, Hunter Ant, and Network Ant inference
- Efficient data serialization between processes
- Automatic load balancing and resource management
- Performance monitoring and bottleneck detection
- Integration with existing prediction engine
"""

import asyncio
import concurrent.futures
import logging
import multiprocessing
import os
import pickle
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import psutil
import queue
import threading
import uuid

from worker_ant_v1.utils.logger import setup_logger


@dataclass
class ProcessTaskData:
    """Serializable data for process tasks"""
    task_id: str
    task_type: str  # oracle, hunter, network
    input_data: Dict[str, Any]
    timestamp: datetime
    timeout: float = 30.0
    priority: int = 1  # 1=high, 2=medium, 3=low

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'input_data': self.input_data,
            'timestamp': self.timestamp.isoformat(),
            'timeout': self.timeout,
            'priority': self.priority
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessTaskData':
        """Create from dictionary"""
        return cls(
            task_id=data['task_id'],
            task_type=data['task_type'],
            input_data=data['input_data'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            timeout=data['timeout'],
            priority=data['priority']
        )


@dataclass
class ProcessTaskResult:
    """Result from process task execution"""
    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ProcessWorkerStats:
    """Statistics for process worker performance"""
    worker_id: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    last_activity: datetime = None
    tasks_per_minute: float = 0.0

    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()


class ProcessPoolManager:
    """Manager for CPU-intensive ML inference process pools"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = setup_logger("ProcessPoolManager")
        
        # Configuration
        self.config = config or {}
        self.max_workers = self.config.get('max_workers', min(multiprocessing.cpu_count(), 8))
        self.task_timeout = self.config.get('task_timeout', 30.0)
        self.queue_size = self.config.get('queue_size', 1000)
        self.enable_monitoring = self.config.get('enable_monitoring', True)
        
        # Process pools by task type
        self.process_pools: Dict[str, concurrent.futures.ProcessPoolExecutor] = {}
        self.pool_configs = {
            'oracle': {'max_workers': max(2, self.max_workers // 3)},
            'hunter': {'max_workers': max(2, self.max_workers // 3)},
            'network': {'max_workers': max(2, self.max_workers // 3)},
            'general': {'max_workers': max(1, self.max_workers // 4)}
        }
        
        # Task management
        self.pending_tasks: Dict[str, ProcessTaskData] = {}
        self.completed_tasks: Dict[str, ProcessTaskResult] = {}
        self.task_futures: Dict[str, concurrent.futures.Future] = {}
        
        # Performance monitoring
        self.worker_stats: Dict[str, ProcessWorkerStats] = {}
        self.performance_metrics = {
            'total_tasks_submitted': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'avg_queue_time': 0.0,
            'avg_execution_time': 0.0,
            'throughput_per_minute': 0.0,
            'cpu_utilization': 0.0,
            'memory_utilization': 0.0
        }
        
        # System state
        self.initialized = False
        self.running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Event loop reference
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def initialize(self) -> bool:
        """Initialize process pools and monitoring"""
        try:
            self.logger.info("🚀 Initializing Process Pool Manager...")
            
            # Store event loop reference
            self.loop = asyncio.get_event_loop()
            
            # Initialize process pools
            await self._initialize_process_pools()
            
            # Start monitoring if enabled
            if self.enable_monitoring:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.running = True
            self.initialized = True
            
            self.logger.info(f"✅ Process Pool Manager initialized with {self.max_workers} workers")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Process Pool Manager: {e}")
            return False

    async def _initialize_process_pools(self):
        """Initialize process pools for different task types"""
        for pool_name, config in self.pool_configs.items():
            try:
                # Create process pool
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=config['max_workers'],
                    initializer=_worker_initializer,
                    initargs=(pool_name,)
                )
                
                self.process_pools[pool_name] = executor
                
                # Initialize worker stats
                for i in range(config['max_workers']):
                    worker_id = f"{pool_name}_worker_{i}"
                    self.worker_stats[worker_id] = ProcessWorkerStats(worker_id=worker_id)
                
                self.logger.info(f"✅ Initialized {pool_name} pool with {config['max_workers']} workers")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize {pool_name} pool: {e}")
                raise

    async def submit_oracle_task(self, token_address: str, market_data: Dict[str, Any], 
                                horizon: int = 15) -> str:
        """Submit Oracle Ant inference task"""
        task_data = ProcessTaskData(
            task_id=str(uuid.uuid4()),
            task_type="oracle",
            input_data={
                "token_address": token_address,
                "market_data": market_data,
                "horizon": horizon
            },
            timestamp=datetime.utcnow(),
            priority=1
        )
        
        return await self._submit_task(task_data)

    async def submit_hunter_task(self, wallet_id: str, market_data: Dict[str, Any]) -> str:
        """Submit Hunter Ant inference task"""
        task_data = ProcessTaskData(
            task_id=str(uuid.uuid4()),
            task_type="hunter",
            input_data={
                "wallet_id": wallet_id,
                "market_data": market_data
            },
            timestamp=datetime.utcnow(),
            priority=1
        )
        
        return await self._submit_task(task_data)

    async def submit_network_task(self, token_address: str) -> str:
        """Submit Network Ant inference task"""
        task_data = ProcessTaskData(
            task_id=str(uuid.uuid4()),
            task_type="network",
            input_data={
                "token_address": token_address
            },
            timestamp=datetime.utcnow(),
            priority=2
        )
        
        return await self._submit_task(task_data)

    async def _submit_task(self, task_data: ProcessTaskData) -> str:
        """Submit task to appropriate process pool"""
        try:
            # Validate task
            if not self.initialized:
                raise RuntimeError("Process Pool Manager not initialized")
                
            task_type = task_data.task_type
            if task_type not in self.process_pools:
                task_type = "general"  # Fallback to general pool
            
            # Store task
            self.pending_tasks[task_data.task_id] = task_data
            
            # Submit to process pool
            executor = self.process_pools[task_type]
            future = self.loop.run_in_executor(
                executor,
                _execute_ml_task,
                task_data.to_dict()
            )
            
            self.task_futures[task_data.task_id] = future
            
            # Update metrics
            self.performance_metrics['total_tasks_submitted'] += 1
            
            self.logger.debug(f"📤 Submitted {task_type} task {task_data.task_id}")
            
            return task_data.task_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to submit task: {e}")
            raise

    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> ProcessTaskResult:
        """Get result for a submitted task"""
        try:
            # Check if task is already completed
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            # Check if task is pending
            if task_id not in self.task_futures:
                raise ValueError(f"Task {task_id} not found")
            
            # Wait for completion
            future = self.task_futures[task_id]
            timeout = timeout or self.task_timeout
            
            try:
                result_data = await asyncio.wait_for(future, timeout=timeout)
                result = ProcessTaskResult(**result_data)
                
                # Store completed result
                self.completed_tasks[task_id] = result
                
                # Cleanup
                del self.task_futures[task_id]
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
                
                # Update metrics
                if result.success:
                    self.performance_metrics['total_tasks_completed'] += 1
                else:
                    self.performance_metrics['total_tasks_failed'] += 1
                
                return result
                
            except asyncio.TimeoutError:
                self.logger.warning(f"⏰ Task {task_id} timed out after {timeout}s")
                
                # Cancel the future
                future.cancel()
                
                # Create timeout result
                result = ProcessTaskResult(
                    task_id=task_id,
                    success=False,
                    error=f"Task timed out after {timeout}s",
                    execution_time=timeout
                )
                
                self.completed_tasks[task_id] = result
                self.performance_metrics['total_tasks_failed'] += 1
                
                return result
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get task result: {e}")
            
            # Create error result
            return ProcessTaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                execution_time=0.0
            )

    async def get_task_result_nowait(self, task_id: str) -> Optional[ProcessTaskResult]:
        """Get task result if available, otherwise return None"""
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        
        if task_id in self.task_futures:
            future = self.task_futures[task_id]
            if future.done():
                try:
                    result_data = future.result()
                    result = ProcessTaskResult(**result_data)
                    self.completed_tasks[task_id] = result
                    
                    # Cleanup
                    del self.task_futures[task_id]
                    if task_id in self.pending_tasks:
                        del self.pending_tasks[task_id]
                    
                    return result
                except Exception as e:
                    error_result = ProcessTaskResult(
                        task_id=task_id,
                        success=False,
                        error=str(e)
                    )
                    self.completed_tasks[task_id] = error_result
                    return error_result
        
        return None

    async def batch_submit_tasks(self, tasks: List[ProcessTaskData]) -> List[str]:
        """Submit multiple tasks in batch"""
        task_ids = []
        
        for task in tasks:
            try:
                task_id = await self._submit_task(task)
                task_ids.append(task_id)
            except Exception as e:
                self.logger.error(f"❌ Failed to submit batch task: {e}")
                # Continue with other tasks
        
        return task_ids

    async def wait_for_tasks(self, task_ids: List[str], 
                           timeout: Optional[float] = None) -> List[ProcessTaskResult]:
        """Wait for multiple tasks to complete"""
        results = []
        
        for task_id in task_ids:
            try:
                result = await self.get_task_result(task_id, timeout)
                results.append(result)
            except Exception as e:
                self.logger.error(f"❌ Failed to get result for task {task_id}: {e}")
                error_result = ProcessTaskResult(
                    task_id=task_id,
                    success=False,
                    error=str(e)
                )
                results.append(error_result)
        
        return results

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                await self._update_performance_metrics()
                await self._cleanup_completed_tasks()
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate queue sizes
            pending_count = len(self.pending_tasks)
            completed_count = len(self.completed_tasks)
            
            # Calculate throughput
            current_time = time.time()
            if not hasattr(self, '_last_metric_time'):
                self._last_metric_time = current_time
                self._last_completed_count = completed_count
                return
                
            time_diff = current_time - self._last_metric_time
            if time_diff > 0:
                completed_diff = completed_count - self._last_completed_count
                self.performance_metrics['throughput_per_minute'] = (completed_diff / time_diff) * 60
            
            self._last_metric_time = current_time
            self._last_completed_count = completed_count
            
            # System resource utilization
            self.performance_metrics['cpu_utilization'] = psutil.cpu_percent()
            self.performance_metrics['memory_utilization'] = psutil.virtual_memory().percent
            
            # Log performance summary
            if pending_count > 0 or completed_count > 100:
                self.logger.info(
                    f"📊 Process Pool Stats: {pending_count} pending, "
                    f"{completed_count} completed, "
                    f"{self.performance_metrics['throughput_per_minute']:.1f} tasks/min"
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error updating performance metrics: {e}")

    async def _cleanup_completed_tasks(self):
        """Cleanup old completed tasks to prevent memory leaks"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            # Remove old completed tasks
            tasks_to_remove = [
                task_id for task_id, result in self.completed_tasks.items()
                if result.timestamp < cutoff_time
            ]
            
            for task_id in tasks_to_remove:
                del self.completed_tasks[task_id]
            
            if tasks_to_remove:
                self.logger.debug(f"🧹 Cleaned up {len(tasks_to_remove)} old completed tasks")
                
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up completed tasks: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'metrics': self.performance_metrics.copy(),
            'pool_stats': {
                pool_name: {
                    'workers': len([w for w in self.worker_stats.keys() if w.startswith(pool_name)]),
                    'max_workers': config['max_workers']
                }
                for pool_name, config in self.pool_configs.items()
            },
            'queue_stats': {
                'pending_tasks': len(self.pending_tasks),
                'completed_tasks': len(self.completed_tasks),
                'active_futures': len(self.task_futures)
            },
            'system_resources': {
                'cpu_count': multiprocessing.cpu_count(),
                'cpu_usage': self.performance_metrics['cpu_utilization'],
                'memory_usage': self.performance_metrics['memory_utilization']
            }
        }

    async def shutdown(self):
        """Shutdown process pools and cleanup"""
        try:
            self.logger.info("🛑 Shutting down Process Pool Manager...")
            
            self.running = False
            
            # Cancel monitoring task
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel all pending futures
            for task_id, future in self.task_futures.items():
                if not future.done():
                    future.cancel()
            
            # Shutdown process pools
            for pool_name, executor in self.process_pools.items():
                try:
                    executor.shutdown(wait=True, timeout=30)
                    self.logger.info(f"✅ Shutdown {pool_name} process pool")
                except Exception as e:
                    self.logger.error(f"❌ Error shutting down {pool_name} pool: {e}")
            
            # Clear data structures
            self.process_pools.clear()
            self.pending_tasks.clear()
            self.task_futures.clear()
            
            self.logger.info("✅ Process Pool Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")


# Worker process functions (must be at module level for pickling)

def _worker_initializer(pool_name: str):
    """Initialize worker process"""
    # Set process title for monitoring
    try:
        import setproctitle
        setproctitle.setproctitle(f"antbot-{pool_name}-worker")
    except ImportError:
        pass
    
    # Initialize logging for worker process
    logging.basicConfig(level=logging.INFO)


def _execute_ml_task(task_data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Execute ML inference task in worker process"""
    start_time = time.time()
    worker_id = f"worker_{os.getpid()}"
    
    try:
        # Reconstruct task data
        task_data = ProcessTaskData.from_dict(task_data_dict)
        
        # Route to appropriate inference function
        if task_data.task_type == "oracle":
            result = _execute_oracle_inference(task_data.input_data)
        elif task_data.task_type == "hunter":
            result = _execute_hunter_inference(task_data.input_data)
        elif task_data.task_type == "network":
            result = _execute_network_inference(task_data.input_data)
        else:
            raise ValueError(f"Unknown task type: {task_data.task_type}")
        
        execution_time = time.time() - start_time
        
        return {
            'task_id': task_data.task_id,
            'success': True,
            'result': result,
            'error': None,
            'execution_time': execution_time,
            'worker_id': worker_id,
            'timestamp': datetime.utcnow()
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        return {
            'task_id': task_data_dict['task_id'],
            'success': False,
            'result': None,
            'error': str(e),
            'execution_time': execution_time,
            'worker_id': worker_id,
            'timestamp': datetime.utcnow()
        }


def _execute_oracle_inference(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Oracle Ant inference in worker process"""
    # Import here to avoid loading in main process
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        from worker_ant_v1.trading.ml_architectures.oracle_ant import OracleAntPredictor
        
        # Initialize predictor (will use CPU in worker process)
        predictor = OracleAntPredictor()
        
        # Run prediction synchronously
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            prediction = loop.run_until_complete(
                predictor.predict(
                    input_data['token_address'],
                    input_data['market_data'],
                    input_data['horizon']
                )
            )
            
            # Convert to serializable format
            return {
                'token_address': prediction.token_address,
                'predicted_price': prediction.predicted_price,
                'predicted_volume': prediction.predicted_volume,
                'predicted_holders': prediction.predicted_holders,
                'price_confidence': prediction.price_confidence,
                'volume_confidence': prediction.volume_confidence,
                'holders_confidence': prediction.holders_confidence,
                'price_distribution': prediction.price_distribution,
                'prediction_horizon': prediction.prediction_horizon,
                'attention_weights': prediction.attention_weights,
                'timestamp': prediction.timestamp.isoformat()
            }
        finally:
            loop.close()
            
    except Exception as e:
        raise RuntimeError(f"Oracle Ant inference failed: {e}")


def _execute_hunter_inference(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Hunter Ant inference in worker process"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        from worker_ant_v1.trading.ml_architectures.hunter_ant import HunterAnt
        
        # Initialize hunter ant
        hunter = HunterAnt(input_data['wallet_id'])
        
        # Run action selection synchronously
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            action, value = loop.run_until_complete(
                hunter.act(input_data['market_data'])
            )
            
            return {
                'wallet_id': input_data['wallet_id'],
                'action': action,
                'value': value,
                'timestamp': datetime.utcnow().isoformat()
            }
        finally:
            loop.close()
            
    except Exception as e:
        raise RuntimeError(f"Hunter Ant inference failed: {e}")


def _execute_network_inference(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute Network Ant inference in worker process"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        from worker_ant_v1.trading.ml_architectures.network_ant import NetworkAntPredictor
        
        # Initialize predictor
        predictor = NetworkAntPredictor()
        
        # Run prediction synchronously
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            prediction = loop.run_until_complete(
                predictor.predict_network_intelligence(input_data['token_address'])
            )
            
            # Convert to serializable format
            return {
                'token_address': prediction.token_address,
                'contagion_score': prediction.contagion_score,
                'smart_money_probability': prediction.smart_money_probability,
                'manipulation_risk': prediction.manipulation_risk,
                'wallet_classifications': {
                    k: v.value for k, v in prediction.wallet_classifications.items()
                },
                'link_predictions': prediction.link_predictions,
                'influence_network': prediction.influence_network,
                'confidence': prediction.confidence,
                'timestamp': prediction.timestamp.isoformat()
            }
        finally:
            loop.close()
            
    except Exception as e:
        raise RuntimeError(f"Network Ant inference failed: {e}")


# Global process pool manager instance
_process_pool_manager: Optional[ProcessPoolManager] = None


def get_process_pool_config() -> Dict[str, Any]:
    """Get process pool configuration from environment"""
    return {
        'max_workers': int(os.getenv('PROCESS_POOL_MAX_WORKERS', multiprocessing.cpu_count())),
        'task_timeout': float(os.getenv('PROCESS_POOL_TASK_TIMEOUT', '30.0')),
        'queue_size': int(os.getenv('PROCESS_POOL_QUEUE_SIZE', '1000')),
        'enable_monitoring': os.getenv('PROCESS_POOL_MONITORING', 'true').lower() == 'true'
    }


async def get_process_pool_manager() -> ProcessPoolManager:
    """Get or create global process pool manager instance"""
    global _process_pool_manager
    
    if _process_pool_manager is None:
        config = get_process_pool_config()
        _process_pool_manager = ProcessPoolManager(config)
        await _process_pool_manager.initialize()
    
    return _process_pool_manager


async def shutdown_process_pool_manager():
    """Shutdown global process pool manager"""
    global _process_pool_manager
    
    if _process_pool_manager:
        await _process_pool_manager.shutdown()
        _process_pool_manager = None 