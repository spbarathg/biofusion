import os
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger

from utils.logger import log_alert

class WorkerMonitor:
    """
    Watches live performance of each Worker Ant.
    Tracks trade frequency, capital growth, and error rates.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.active_workers = {}
        self.worker_metrics = {}
        self.alert_thresholds = {
            "error_rate": 0.1,  # 10% error rate threshold
            "profit_threshold": 0.05,  # 5% profit threshold
            "trade_frequency": 60,  # 1 trade per minute threshold
            "capital_growth": 0.1,  # 10% capital growth threshold
        }
        
        # Load config
        with open(config_path, "r") as f:
            import yaml
            config = yaml.safe_load(f)
            if "monitoring" in config:
                monitoring_config = config["monitoring"]
                self.alert_thresholds.update(monitoring_config.get("alert_thresholds", {}))
    
    async def start_monitoring(self):
        """
        Start monitoring all workers
        """
        logger.info("Starting worker monitoring")
        
        while True:
            try:
                # Update worker metrics
                await self.update_worker_metrics()
                
                # Check for alerts
                await self.check_alerts()
                
                # Sleep for a bit
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in worker monitoring: {str(e)}")
                await asyncio.sleep(30)  # Wait longer on error
    
    async def register_worker(self, worker_id: str, wallet_address: str, initial_capital: float):
        """
        Register a new worker for monitoring
        
        Args:
            worker_id: Unique identifier for the worker
            wallet_address: Solana wallet address for the worker
            initial_capital: Initial capital in SOL
        """
        self.active_workers[worker_id] = {
            "wallet_address": wallet_address,
            "initial_capital": initial_capital,
            "current_capital": initial_capital,
            "start_time": datetime.now(),
            "trades": 0,
            "errors": 0,
            "last_trade_time": None,
            "profit": 0.0,
            "is_active": True
        }
        
        self.worker_metrics[worker_id] = {
            "trade_frequency": 0,
            "error_rate": 0,
            "capital_growth": 0,
            "profit_per_trade": 0
        }
        
        logger.info(f"Registered Worker Ant {worker_id} for monitoring")
    
    async def unregister_worker(self, worker_id: str):
        """
        Unregister a worker from monitoring
        
        Args:
            worker_id: Unique identifier for the worker
        """
        if worker_id in self.active_workers:
            del self.active_workers[worker_id]
            
        if worker_id in self.worker_metrics:
            del self.worker_metrics[worker_id]
            
        logger.info(f"Unregistered Worker Ant {worker_id} from monitoring")
    
    async def update_worker_metrics(self):
        """
        Update metrics for all workers
        """
        for worker_id, worker_info in self.active_workers.items():
            if not worker_info["is_active"]:
                continue
                
            # Calculate metrics
            runtime = (datetime.now() - worker_info["start_time"]).total_seconds() / 3600  # hours
            
            # Trade frequency (trades per hour)
            trade_frequency = worker_info["trades"] / runtime if runtime > 0 else 0
            
            # Error rate
            error_rate = worker_info["errors"] / worker_info["trades"] if worker_info["trades"] > 0 else 0
            
            # Capital growth
            capital_growth = (worker_info["current_capital"] - worker_info["initial_capital"]) / worker_info["initial_capital"] if worker_info["initial_capital"] > 0 else 0
            
            # Profit per trade
            profit_per_trade = worker_info["profit"] / worker_info["trades"] if worker_info["trades"] > 0 else 0
            
            # Update metrics
            self.worker_metrics[worker_id] = {
                "trade_frequency": trade_frequency,
                "error_rate": error_rate,
                "capital_growth": capital_growth,
                "profit_per_trade": profit_per_trade
            }
            
            # Log metrics
            logger.info(f"Worker {worker_id} metrics: {json.dumps(self.worker_metrics[worker_id])}")
    
    async def record_trade(self, worker_id: str, profit: float, current_capital: float):
        """
        Record a trade for a worker
        
        Args:
            worker_id: Unique identifier for the worker
            profit: Profit from the trade in SOL
            current_capital: Current capital in SOL
        """
        if worker_id not in self.active_workers:
            logger.warning(f"Worker {worker_id} not found for trade recording")
            return
            
        worker_info = self.active_workers[worker_id]
        worker_info["trades"] += 1
        worker_info["profit"] += profit
        worker_info["current_capital"] = current_capital
        worker_info["last_trade_time"] = datetime.now()
        
        logger.info(f"Recorded trade for Worker {worker_id}: {profit} SOL profit, {current_capital} SOL capital")
    
    async def record_error(self, worker_id: str):
        """
        Record an error for a worker
        
        Args:
            worker_id: Unique identifier for the worker
        """
        if worker_id not in self.active_workers:
            logger.warning(f"Worker {worker_id} not found for error recording")
            return
            
        worker_info = self.active_workers[worker_id]
        worker_info["errors"] += 1
        
        logger.info(f"Recorded error for Worker {worker_id}")
    
    async def check_alerts(self):
        """
        Check for alerts based on worker metrics
        """
        for worker_id, metrics in self.worker_metrics.items():
            worker_info = self.active_workers.get(worker_id)
            if not worker_info or not worker_info["is_active"]:
                continue
                
            # Check error rate
            if metrics["error_rate"] > self.alert_thresholds["error_rate"]:
                log_alert(
                    "warning",
                    f"Worker {worker_id} has high error rate: {metrics['error_rate']:.2%}",
                    {"worker_id": worker_id, "error_rate": metrics["error_rate"]}
                )
            
            # Check trade frequency
            if metrics["trade_frequency"] < self.alert_thresholds["trade_frequency"]:
                log_alert(
                    "warning",
                    f"Worker {worker_id} has low trade frequency: {metrics['trade_frequency']:.2f} trades/hour",
                    {"worker_id": worker_id, "trade_frequency": metrics["trade_frequency"]}
                )
            
            # Check capital growth
            if metrics["capital_growth"] > self.alert_thresholds["capital_growth"]:
                log_alert(
                    "info",
                    f"Worker {worker_id} has high capital growth: {metrics['capital_growth']:.2%}",
                    {"worker_id": worker_id, "capital_growth": metrics["capital_growth"]}
                )
            
            # Check profit per trade
            if metrics["profit_per_trade"] > self.alert_thresholds["profit_threshold"]:
                log_alert(
                    "info",
                    f"Worker {worker_id} has high profit per trade: {metrics['profit_per_trade']:.4f} SOL",
                    {"worker_id": worker_id, "profit_per_trade": metrics["profit_per_trade"]}
                )
    
    def get_worker_metrics(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a worker
        
        Args:
            worker_id: Unique identifier for the worker
            
        Returns:
            Optional[Dict[str, Any]]: Worker metrics or None if not found
        """
        if worker_id not in self.worker_metrics:
            return None
            
        return self.worker_metrics[worker_id].copy()
    
    def get_all_worker_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all workers
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of worker metrics
        """
        return {worker_id: metrics.copy() for worker_id, metrics in self.worker_metrics.items()}
    
    def get_worker_info(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Get info for a worker
        
        Args:
            worker_id: Unique identifier for the worker
            
        Returns:
            Optional[Dict[str, Any]]: Worker info or None if not found
        """
        if worker_id not in self.active_workers:
            return None
            
        worker_info = self.active_workers[worker_id].copy()
        
        # Add metrics
        if worker_id in self.worker_metrics:
            worker_info["metrics"] = self.worker_metrics[worker_id].copy()
        
        return worker_info
    
    def get_all_worker_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get info for all workers
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of worker info
        """
        result = {}
        
        for worker_id, worker_info in self.active_workers.items():
            info = worker_info.copy()
            
            # Add metrics
            if worker_id in self.worker_metrics:
                info["metrics"] = self.worker_metrics[worker_id].copy()
                
            result[worker_id] = info
        
        return result

async def main():
    # Example usage
    monitor = WorkerMonitor()
    
    # Register a worker
    await monitor.register_worker("worker_1", "wallet_address_1", 10.0)
    
    # Record some trades
    await monitor.record_trade("worker_1", 0.5, 10.5)
    await monitor.record_trade("worker_1", 0.3, 10.8)
    
    # Record an error
    await monitor.record_error("worker_1")
    
    # Update metrics
    await monitor.update_worker_metrics()
    
    # Get metrics
    metrics = monitor.get_worker_metrics("worker_1")
    print(f"Worker metrics: {metrics}")
    
    # Start monitoring
    await monitor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main()) 