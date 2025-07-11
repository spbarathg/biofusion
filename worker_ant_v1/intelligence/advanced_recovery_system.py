"""
ADVANCED RECOVERY SYSTEM - BATTLEFIELD RESILIENCE
==============================================

Advanced recovery system with microsecond-level response time,
pattern-based threat detection, and real-time battle intelligence.
"""

import asyncio
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from worker_ant_v1.utils.logger import setup_logger


@dataclass
class RecoveryState:
    """Recovery state tracking"""
    
    component: str
    status: str = "healthy"  # healthy, degraded, failed, recovering
    
    # Error statistics
    error_count_1h: int = 0
    error_count_24h: int = 0
    error_count_total: int = 0
    consecutive_errors: int = 0
    
    # Recovery tracking
    last_error_time: Optional[datetime] = None
    last_recovery_time: Optional[datetime] = None
    recovery_attempts: int = 0
    recovery_success_rate: float = 1.0


class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    
    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"


class AdvancedRecoverySystem:
    """Advanced recovery system with pattern detection"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize recovery system"""
        self.config = config
        self.logger = setup_logger("AdvancedRecoverySystem")
        
        # Component states
        self.component_states: Dict[str, RecoveryState] = {}
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Recovery thresholds
        self.error_thresholds = {
            'consecutive': 3,
            'hourly': 10,
            'daily': 50
        }
        
        # Recovery timeouts (seconds)
        self.recovery_timeouts = {
            'retry': 5,
            'circuit_breaker': 60,
            'fallback': 300,
            'graceful': 600
        }
    
    def register_component(self, component: str) -> None:
        """Register component for recovery tracking"""
        if component not in self.component_states:
            self.component_states[component] = RecoveryState(component=component)
            self.circuit_breakers[component] = {
                'state': 'closed',
                'failures': 0,
                'last_failure_time': None,
                'recovery_timeout': self.recovery_timeouts['circuit_breaker']
            }
    
    def record_error(self, component: str, error: Exception) -> None:
        """Record component error"""
        if component not in self.component_states:
            self.register_component(component)
        
        state = self.component_states[component]
        state.error_count_total += 1
        state.consecutive_errors += 1
        state.last_error_time = datetime.now()
        
        # Update hourly/daily counts
        if state.last_error_time:
            time_since_error = (datetime.now() - state.last_error_time).seconds
            if time_since_error <= 3600:
                state.error_count_1h += 1
            if time_since_error <= 86400:
                state.error_count_24h += 1
        
        # Check thresholds
        if self._should_trigger_recovery(state):
            self._trigger_recovery(component)
    
    def record_success(self, component: str) -> None:
        """Record component success"""
        if component not in self.component_states:
            return
        
        state = self.component_states[component]
        state.consecutive_errors = 0
        state.status = "healthy"
        
        # Update circuit breaker
        if component in self.circuit_breakers:
            breaker = self.circuit_breakers[component]
            if breaker['state'] == 'half_open':
                breaker['state'] = 'closed'
                breaker['failures'] = 0
                self.logger.info(f"🟢 Circuit breaker CLOSED for {component}")
    
    def is_circuit_open(self, component: str) -> bool:
        """Check if circuit breaker is open for component"""
        
        if component not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[component]
        
        if breaker['state'] == 'closed':
            return False
        
        if breaker['state'] == 'open':
            if breaker['last_failure_time']:
                time_since_failure = (
                    datetime.now() - breaker['last_failure_time']
                ).total_seconds()
                if time_since_failure > breaker['recovery_timeout']:
                    breaker['state'] = 'half_open'
                    self.logger.info(
                        f"🟡 Circuit breaker HALF-OPEN for {component}"
                    )
                    return False
            return True
        
        return False
    
    def _should_trigger_recovery(self, state: RecoveryState) -> bool:
        """Check if recovery should be triggered"""
        return (
            state.consecutive_errors >= self.error_thresholds['consecutive'] or
            state.error_count_1h >= self.error_thresholds['hourly'] or
            state.error_count_24h >= self.error_thresholds['daily']
        )
    
    def _trigger_recovery(self, component: str) -> None:
        """Trigger component recovery"""
        if component not in self.component_states:
            return
        
        state = self.component_states[component]
        state.status = "recovering"
        state.recovery_attempts += 1
        state.last_recovery_time = datetime.now()
        
        # Update circuit breaker
        if component in self.circuit_breakers:
            breaker = self.circuit_breakers[component]
            breaker['state'] = 'open'
            breaker['failures'] += 1
            breaker['last_failure_time'] = datetime.now()
            self.logger.warning(f"🔴 Circuit breaker OPEN for {component}")
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get component health status"""
        if component not in self.component_states:
            return {
                'status': 'unknown',
                'error_rate': 0.0,
                'recovery_rate': 0.0
            }
        
        state = self.component_states[component]
        
        return {
            'status': state.status,
            'error_rate': self._calculate_error_rate(state),
            'recovery_rate': state.recovery_success_rate,
            'consecutive_errors': state.consecutive_errors,
            'total_errors': state.error_count_total,
            'recovery_attempts': state.recovery_attempts
        }
    
    def _calculate_error_rate(self, state: RecoveryState) -> float:
        """Calculate component error rate"""
        if state.error_count_total == 0:
            return 0.0
        
        # Weight recent errors more heavily
        hourly_weight = 0.5
        daily_weight = 0.3
        total_weight = 0.2
        
        weighted_rate = (
            (state.error_count_1h / max(1, state.error_count_total)) * hourly_weight +
            (state.error_count_24h / max(1, state.error_count_total)) * daily_weight +
            1.0 * total_weight
        )
        
        return min(1.0, weighted_rate) 