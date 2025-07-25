"""
ENTERPRISE ERROR HANDLER - SYSTEM RESILIENCE & RECOVERY
======================================================

Advanced error handling system with circuit breaker patterns,
exponential backoff, and automatic recovery strategies.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import traceback

from worker_ant_v1.utils.logger import setup_logger


class ComponentHealth(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ErrorMetrics:
    """Error tracking metrics for a component"""
    component_name: str
    total_errors: int = 0
    error_rate: float = 0.0
    consecutive_failures: int = 0
    last_error_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    health_status: ComponentHealth = ComponentHealth.HEALTHY


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration"""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_factor: float = 2.0
    jitter: bool = True
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60


class EnterpriseErrorHandler:
    """Enterprise-grade error handling with circuit breaker and recovery strategies"""
    
    def __init__(self):
        self.logger = setup_logger("EnterpriseErrorHandler")
        
        # Component tracking
        self.component_metrics: Dict[str, ErrorMetrics] = {}
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        
        # Global settings
        self.global_circuit_breaker = False
        self.total_system_errors = 0
        self.system_health = ComponentHealth.HEALTHY
        
        # Default recovery strategy
        self.default_strategy = RecoveryStrategy()
        
        self.logger.info("🛡️ Enterprise Error Handler initialized")
    
    async def handle_error(
        self, 
        error: Exception, 
        component_name: str,
        operation_name: str = "unknown",
        context: Dict[str, Any] = None
    ) -> bool:
        """
        Handle an error with enterprise-grade recovery strategies
        
        Returns:
            bool: True if error was handled and operation should retry, False otherwise
        """
        try:
            # Initialize component metrics if not exists
            if component_name not in self.component_metrics:
                self.component_metrics[component_name] = ErrorMetrics(component_name=component_name)
            
            metrics = self.component_metrics[component_name]
            strategy = self.recovery_strategies.get(component_name, self.default_strategy)
            
            # Update error metrics
            self._update_error_metrics(metrics, error)
            
            # Log the error with full context
            error_context = {
                'component': component_name,
                'operation': operation_name,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'consecutive_failures': metrics.consecutive_failures,
                'circuit_breaker_state': metrics.circuit_breaker_state.value,
                'context': context or {}
            }
            
            self.logger.error(f"🚨 Error in {component_name}.{operation_name}: {error}", extra=error_context)
            
            # Check circuit breaker
            if strategy.circuit_breaker_enabled:
                if await self._check_circuit_breaker(metrics, strategy):
                    return False  # Circuit is open, fail fast
            
            # Determine if we should retry
            should_retry = metrics.consecutive_failures < strategy.max_retries
            
            if should_retry:
                # Calculate backoff delay
                delay = await self._calculate_backoff_delay(metrics, strategy)
                self.logger.info(f"🔄 Retrying {component_name}.{operation_name} in {delay:.2f}s (attempt {metrics.consecutive_failures + 1})")
                await asyncio.sleep(delay)
                return True
            else:
                self.logger.critical(f"💀 Maximum retries exceeded for {component_name}.{operation_name}")
                await self._escalate_error(component_name, error, metrics)
                return False
                
        except Exception as handler_error:
            self.logger.critical(f"CRITICAL: Error handler itself failed: {handler_error}")
            return False
    
    async def record_success(self, component_name: str, operation_name: str = "unknown"):
        """Record a successful operation"""
        if component_name not in self.component_metrics:
            self.component_metrics[component_name] = ErrorMetrics(component_name=component_name)
        
        metrics = self.component_metrics[component_name]
        metrics.consecutive_failures = 0
        metrics.last_success_time = datetime.now()
        
        # Update circuit breaker state
        if metrics.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            metrics.circuit_breaker_state = CircuitBreakerState.CLOSED
            self.logger.info(f"✅ Circuit breaker CLOSED for {component_name} - service recovered")
        
        # Update health status
        await self._update_component_health(metrics)
    
    def _update_error_metrics(self, metrics: ErrorMetrics, error: Exception):
        """Update error metrics for a component"""
        metrics.total_errors += 1
        metrics.consecutive_failures += 1
        metrics.last_error_time = datetime.now()
        self.total_system_errors += 1
        
        # Calculate error rate (errors per minute)
        if metrics.last_success_time:
            time_window = (datetime.now() - metrics.last_success_time).total_seconds() / 60.0
            metrics.error_rate = metrics.total_errors / max(time_window, 1.0)
    
    async def _check_circuit_breaker(self, metrics: ErrorMetrics, strategy: RecoveryStrategy) -> bool:
        """Check and update circuit breaker state"""
        if metrics.circuit_breaker_state == CircuitBreakerState.OPEN:
            # Check if we should transition to half-open
            if metrics.last_error_time:
                time_since_last_error = (datetime.now() - metrics.last_error_time).total_seconds()
                if time_since_last_error >= strategy.recovery_timeout_seconds:
                    metrics.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                    self.logger.warning(f"🔶 Circuit breaker HALF-OPEN for {metrics.component_name} - testing recovery")
                    return False
            return True  # Circuit is open, fail fast
        
        elif metrics.circuit_breaker_state == CircuitBreakerState.CLOSED:
            # Check if we should open the circuit
            if metrics.consecutive_failures >= strategy.failure_threshold:
                metrics.circuit_breaker_state = CircuitBreakerState.OPEN
                self.logger.critical(f"🔴 Circuit breaker OPEN for {metrics.component_name} - failing fast")
                return True
        
        return False
    
    async def _calculate_backoff_delay(self, metrics: ErrorMetrics, strategy: RecoveryStrategy) -> float:
        """Calculate exponential backoff delay with jitter"""
        delay = strategy.base_delay_seconds * (strategy.exponential_factor ** (metrics.consecutive_failures - 1))
        delay = min(delay, strategy.max_delay_seconds)
        
        # Add jitter to prevent thundering herd
        if strategy.jitter:
            import random
            jitter_factor = random.uniform(0.8, 1.2)
            delay *= jitter_factor
        
        return delay
    
    async def _update_component_health(self, metrics: ErrorMetrics):
        """Update component health status"""
        if metrics.consecutive_failures == 0:
            metrics.health_status = ComponentHealth.HEALTHY
        elif metrics.consecutive_failures < 3:
            metrics.health_status = ComponentHealth.DEGRADED
        elif metrics.consecutive_failures < 5:
            metrics.health_status = ComponentHealth.CRITICAL
        else:
            metrics.health_status = ComponentHealth.FAILED
    
    async def _escalate_error(self, component_name: str, error: Exception, metrics: ErrorMetrics):
        """Escalate critical errors to system administrators"""
        escalation_data = {
            'component': component_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'consecutive_failures': metrics.consecutive_failures,
            'total_errors': metrics.total_errors,
            'health_status': metrics.health_status.value,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.critical(f"🆘 ESCALATION: {component_name} has failed critically", extra=escalation_data)
        
        # TODO: Integration with alerting system (Slack, email, etc.)
    
    def register_component(self, component_name: str, strategy: RecoveryStrategy = None):
        """Register a component with custom recovery strategy"""
        if component_name not in self.component_metrics:
            self.component_metrics[component_name] = ErrorMetrics(component_name=component_name)
        
        if strategy:
            self.recovery_strategies[component_name] = strategy
        
        self.logger.info(f"📊 Registered component: {component_name}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        healthy_components = sum(1 for m in self.component_metrics.values() if m.health_status == ComponentHealth.HEALTHY)
        total_components = len(self.component_metrics)
        
        return {
            'system_health': self.system_health.value,
            'total_errors': self.total_system_errors,
            'total_components': total_components,
            'healthy_components': healthy_components,
            'health_percentage': (healthy_components / max(total_components, 1)) * 100,
            'global_circuit_breaker': self.global_circuit_breaker,
            'component_details': {
                name: {
                    'health_status': metrics.health_status.value,
                    'total_errors': metrics.total_errors,
                    'consecutive_failures': metrics.consecutive_failures,
                    'circuit_breaker_state': metrics.circuit_breaker_state.value,
                    'error_rate': metrics.error_rate
                }
                for name, metrics in self.component_metrics.items()
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown of error handler"""
        self.logger.info("🛑 Enterprise Error Handler shutting down")
        
        # Log final system health
        health_report = self.get_system_health()
        self.logger.info(f"📊 Final system health: {health_report}")


# Global instance
_error_handler = None

def get_enterprise_error_handler() -> EnterpriseErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = EnterpriseErrorHandler()
    return _error_handler 