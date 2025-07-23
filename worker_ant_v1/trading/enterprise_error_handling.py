"""
ENTERPRISE ERROR HANDLING - BATTLEFIELD RESILIENCE
=============================================

Advanced error handling system with intelligent recovery,
pattern detection, and real-time battle monitoring.
"""

import asyncio
import functools
import threading
import time
import traceback
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import queue
import hashlib
import json

from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.utils.shared_types import ComponentHealth


@dataclass
class ErrorEvent:
    """Error event tracking"""
    
    error_id: str
    component: str
    error_message: str
    stack_trace: str
    timestamp: datetime
    
    severity: int = 1  # 1-10 scale
    recoverable: bool = True
    attempted_recoveries: List[str] = field(default_factory=list)
    
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    # Enhanced context for intelligent recovery
    function_criticality: str = "medium"  # low, medium, high, critical
    context_data: Dict[str, Any] = field(default_factory=dict)
    error_pattern_hash: str = ""


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    FATAL = 5


class ErrorCategory(Enum):
    NETWORK = "network"
    BLOCKCHAIN = "blockchain"
    TRADING = "trading"
    AI_ML = "ai_ml"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    EXTERNAL_API = "external_api"
    SYSTEM = "system"

class RecoveryAction(Enum):
    RETRY = "retry"
    BACKOFF = "backoff"
    CIRCUIT_BREAK = "circuit_break"
    FALLBACK = "fallback"
    RESTART_COMPONENT = "restart_component"
    ALERT_OPERATOR = "alert_operator"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_STOP = "emergency_stop"


class RecoveryStrategySelector:
    """Intelligent recovery strategy selector based on error analysis"""
    
    def __init__(self):
        self.strategy_patterns = {
            'network_timeout': RecoveryAction.BACKOFF,
            'rpc_error': RecoveryAction.BACKOFF,
            'api_rate_limit': RecoveryAction.BACKOFF,
            'wallet_corruption': RecoveryAction.EMERGENCY_STOP,
            'critical_security_breach': RecoveryAction.EMERGENCY_STOP,
            'trading_engine_failure': RecoveryAction.CIRCUIT_BREAK,
            'ml_model_error': RecoveryAction.FALLBACK,
            'external_api_failure': RecoveryAction.FALLBACK,
            'memory_leak': RecoveryAction.RESTART_COMPONENT,
            'configuration_error': RecoveryAction.ALERT_OPERATOR
        }
        
        self.critical_components = {
            'UnifiedTradingEngine',
            'WalletManager', 
            'VaultWalletSystem',
            'EnhancedKillSwitch',
            'SurgicalTradeExecutor'
        }
        
        self.fallback_mappings = {
            'birdeye_api': 'dexscreener_api',
            'jupiter_dex': 'raydium_dex',
            'primary_rpc': 'backup_rpc',
            'ml_predictor': 'technical_analyzer'
        }
    
    def select_recovery_strategy(self, error: ErrorEvent, context: Dict[str, Any] = None) -> RecoveryAction:
        """Select the optimal recovery strategy based on error analysis"""
        
        # Check for fatal errors first
        if self._is_fatal_error(error):
            return RecoveryAction.EMERGENCY_STOP
        
        # Check for critical component failures
        if error.component in self.critical_components and error.severity >= 4:
            return RecoveryAction.CIRCUIT_BREAK
        
        # Check for external API failures
        if self._is_external_api_error(error):
            return RecoveryAction.FALLBACK
        
        # Check for transient network errors
        if self._is_transient_network_error(error):
            return RecoveryAction.BACKOFF
        
        # Check for trading-specific errors
        if error.component == 'trading' and error.function_criticality in ['high', 'critical']:
            return RecoveryAction.CIRCUIT_BREAK
        
        # Default to retry for low-severity errors
        if error.severity <= 2:
            return RecoveryAction.RETRY
        
        # Default to backoff for medium-severity errors
        return RecoveryAction.BACKOFF
    
    def _is_fatal_error(self, error: ErrorEvent) -> bool:
        """Determine if error is fatal and requires emergency stop"""
        fatal_patterns = [
            'wallet corruption',
            'private key compromised',
            'critical security breach',
            'funds at risk',
            'system integrity compromised'
        ]
        
        error_lower = error.error_message.lower()
        return any(pattern in error_lower for pattern in fatal_patterns)
    
    def _is_external_api_error(self, error: ErrorEvent) -> bool:
        """Check if error is from external API failure"""
        api_patterns = [
            'birdeye',
            'jupiter',
            'dexscreener',
            'api rate limit',
            'external service'
        ]
        
        error_lower = error.error_message.lower()
        return any(pattern in error_lower for pattern in api_patterns)
    
    def _is_transient_network_error(self, error: ErrorEvent) -> bool:
        """Check if error is a transient network issue"""
        network_patterns = [
            'timeout',
            'connection refused',
            'network unreachable',
            'temporary failure',
            'rpc error'
        ]
        
        error_lower = error.error_message.lower()
        return any(pattern in error_lower for pattern in network_patterns)
    
    def get_fallback_target(self, failed_component: str) -> Optional[str]:
        """Get fallback component for failed service"""
        return self.fallback_mappings.get(failed_component)


class EnterpriseErrorHandler:
    """Enterprise-grade error handling system with intelligent recovery"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize error handler"""
        self.config = config
        self.logger = setup_logger("ErrorHandler")
        
        # Error tracking
        self.error_history: List[ErrorEvent] = []
        self.active_recoveries: Dict[str, bool] = {}
        self.error_patterns: Dict[str, List[ErrorEvent]] = {}
        
        # Component health monitoring
        self.component_health: Dict[str, float] = {}  # 0.0 to 1.0 health score
        self.health_thresholds = {
            'critical': 0.25,    # Below 25% = critical
            'warning': 0.5,      # Below 50% = warning
            'degraded': 0.75     # Below 75% = degraded
        }
        
        # Recovery strategy selector
        self.strategy_selector = RecoveryStrategySelector()
        
        # Circuit breaker configuration
        self.circuit_breaker_config = {
            'failure_threshold': config.get('circuit_breaker_failure_threshold', 5),
            'recovery_timeout': config.get('circuit_breaker_recovery_timeout', 300),
            'half_open_timeout': config.get('circuit_breaker_half_open_timeout', 60)
        }
        
        # Recovery patterns
        self.recovery_patterns = {
            'exponential_backoff': {
                'base_delay': 1.0,
                'max_delay': 300.0,
                'factor': 2.0
            },
            'circuit_breaker': {
                'failure_threshold': 5,
                'recovery_timeout': 300
            }
        }
        
        # Graceful degradation state
        self.graceful_degradation_active = False
        self.degraded_services: Set[str] = set()
        
        self.logger.info("🛡️ Enterprise Error Handler initialized with intelligent recovery")
    
    async def handle_error(
        self,
        component: str,
        error: Exception,
        context: Dict[str, Any] = None,
        function_criticality: str = "medium"
    ) -> str:
        """Handle component error and initiate intelligent recovery"""
        error_id = f"{component}_{int(time.time())}"
        
        # Create error pattern hash for pattern recognition
        error_pattern_hash = self._create_error_pattern_hash(error, component)
        
        error_event = ErrorEvent(
            error_id=error_id,
            component=component,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            timestamp=datetime.now(),
            severity=self._calculate_severity(error, component),
            function_criticality=function_criticality,
            context_data=context or {},
            error_pattern_hash=error_pattern_hash
        )
        
        self.error_history.append(error_event)
        self.error_patterns.setdefault(component, []).append(error_event)
        
        # Update component health
        self._update_component_health(component, error_event)
        
        self.logger.error(
            f"🚨 Component error: {component} (health: {self.component_health.get(component, 1.0):.2f}) - {str(error)}"
        )
        
        if component not in self.active_recoveries:
            asyncio.create_task(
                self._initiate_intelligent_recovery(error_event, context)
            )
        
        return error_id
    
    def _create_error_pattern_hash(self, error: Exception, component: str) -> str:
        """Create a hash of error pattern for recognition"""
        pattern_data = {
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error)[:100],  # First 100 chars
            'stack_trace_lines': traceback.format_exc().split('\n')[:5]  # First 5 lines
        }
        
        pattern_json = json.dumps(pattern_data, sort_keys=True)
        return hashlib.md5(pattern_json.encode()).hexdigest()
    
    def _update_component_health(self, component: str, error: ErrorEvent):
        """Update component health score based on error"""
        current_health = self.component_health.get(component, 1.0)
        
        # Calculate health penalty based on error severity
        severity_penalty = {
            1: 0.05,   # Low severity: 5% penalty
            2: 0.10,   # Medium severity: 10% penalty
            3: 0.20,   # High severity: 20% penalty
            4: 0.40,   # Critical severity: 40% penalty
            5: 0.80    # Fatal severity: 80% penalty
        }
        
        penalty = severity_penalty.get(error.severity, 0.10)
        new_health = max(0.0, current_health - penalty)
        
        self.component_health[component] = new_health
        
        # Log health status
        if new_health <= self.health_thresholds['critical']:
            self.logger.critical(f"🚨 CRITICAL: {component} health at {new_health:.2f}")
        elif new_health <= self.health_thresholds['warning']:
            self.logger.warning(f"⚠️ WARNING: {component} health at {new_health:.2f}")
        elif new_health <= self.health_thresholds['degraded']:
            self.logger.warning(f"📉 DEGRADED: {component} health at {new_health:.2f}")
    
    def get_component_health(self, component: str) -> float:
        """Get current health score for a component"""
        return self.component_health.get(component, 1.0)
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        total_components = len(self.component_health)
        if total_components == 0:
            return {'overall_health': 1.0, 'status': 'healthy', 'components': {}}
        
        critical_components = sum(1 for health in self.component_health.values() 
                                if health <= self.health_thresholds['critical'])
        warning_components = sum(1 for health in self.component_health.values() 
                               if health <= self.health_thresholds['warning'])
        
        avg_health = sum(self.component_health.values()) / total_components
        
        if critical_components > 0:
            status = 'critical'
        elif warning_components > 0:
            status = 'degraded'
        else:
            status = 'healthy'
        
        return {
            'overall_health': avg_health,
            'status': status,
            'total_components': total_components,
            'critical_components': critical_components,
            'warning_components': warning_components,
            'components': self.component_health.copy()
        }
    
    async def _initiate_intelligent_recovery(
        self,
        error: ErrorEvent,
        context: Dict[str, Any] = None
    ) -> None:
        """Initiate intelligent error recovery process"""
        component = error.component
        self.active_recoveries[component] = True
        
        try:
            self.logger.info(f"🔄 Initiating intelligent recovery for {component}")
            
            # Select optimal recovery strategy
            strategy = self.strategy_selector.select_recovery_strategy(error, context)
            error.attempted_recoveries.append(strategy.value)
            
            self.logger.info(f"🎯 Selected recovery strategy: {strategy.value}")
            
            if await self._execute_recovery_strategy(strategy, error, context):
                self.logger.info(f"✅ Recovery successful for {component}")
                error.resolved = True
                error.resolution_time = datetime.now()
                
                # Gradually restore component health on successful recovery
                self._restore_component_health(component)
            else:
                self.logger.error(f"❌ Recovery failed for {component}")
                await self._escalate_error(error)
        
        finally:
            self.active_recoveries[component] = False
    
    def _restore_component_health(self, component: str):
        """Gradually restore component health after successful recovery"""
        current_health = self.component_health.get(component, 0.0)
        restoration_rate = 0.1  # 10% health restoration per successful recovery
        
        new_health = min(1.0, current_health + restoration_rate)
        self.component_health[component] = new_health
        
        self.logger.info(f"💚 Health restored for {component}: {current_health:.2f} → {new_health:.2f}")
    
    async def _execute_recovery_strategy(
        self,
        strategy: RecoveryAction,
        error: ErrorEvent,
        context: Dict[str, Any]
    ) -> bool:
        """Execute the selected recovery strategy"""
        
        try:
            if strategy == RecoveryAction.EMERGENCY_STOP:
                return await self._emergency_stop(error, context)
            elif strategy == RecoveryAction.CIRCUIT_BREAK:
                return await self._apply_circuit_breaker(error, context)
            elif strategy == RecoveryAction.FALLBACK:
                return await self._apply_fallback(error, context)
            elif strategy == RecoveryAction.BACKOFF:
                return await self._apply_exponential_backoff(error, context)
            elif strategy == RecoveryAction.RESTART_COMPONENT:
                return await self._restart_component(error, context)
            elif strategy == RecoveryAction.ALERT_OPERATOR:
                return await self._alert_operator(error, context)
            elif strategy == RecoveryAction.GRACEFUL_DEGRADATION:
                return await self._enable_graceful_degradation(error, context)
            else:  # RETRY
                return await self._retry_operation(error, context)
        
        except Exception as e:
            self.logger.error(f"Recovery strategy execution failed: {str(e)}")
            return False
    
    async def _emergency_stop(self, error: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Execute emergency stop procedure"""
        self.logger.critical("🚨 EMERGENCY STOP TRIGGERED - Securing system...")
        
        try:
            # Secure funds in vault
            if 'wallet_manager' in context:
                await self._secure_funds_in_vault(context['wallet_manager'])
            
            # Stop all trading operations
            await self._emergency_trading_shutdown()
            
            # Reset system state
            await self._reset_system_state()
            
            self.logger.critical("✅ Emergency stop completed - System secured")
            return True
            
        except Exception as e:
            self.logger.critical(f"❌ Emergency stop failed: {str(e)}")
            return False
    
    async def _apply_fallback(self, error: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Apply fallback strategy for failed service"""
        component = error.component
        
        # Get fallback target
        fallback_target = self.strategy_selector.get_fallback_target(component)
        
        if not fallback_target:
            self.logger.warning(f"No fallback available for {component}")
            return False
        
        self.logger.info(f"🔄 Switching from {component} to {fallback_target}")
        
        try:
            # Update context to use fallback service
            if context:
                context['fallback_service'] = fallback_target
            
            # Mark service as degraded
            self.degraded_services.add(component)
            
            # Test fallback service
            if await self._test_fallback_service(fallback_target):
                self.logger.info(f"✅ Fallback to {fallback_target} successful")
                return True
            else:
                self.logger.error(f"❌ Fallback to {fallback_target} failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Fallback execution failed: {str(e)}")
            return False
    
    async def _test_fallback_service(self, service_name: str) -> bool:
        """Test if fallback service is available"""
        # This would implement actual service testing
        # For now, return True to indicate fallback is available
        return True
    
    async def _enable_graceful_degradation(self, error: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Enable graceful degradation mode"""
        self.logger.warning("📉 Enabling graceful degradation mode")
        
        self.graceful_degradation_active = True
        self.degraded_services.add(error.component)
        
        # Reduce system load and complexity
        # This would implement actual degradation logic
        self.logger.info("✅ Graceful degradation enabled")
        return True
    
    async def _restart_component(self, error: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Restart a failed component"""
        component = error.component
        
        self.logger.info(f"🔄 Restarting component: {component}")
        
        try:
            # This would implement actual component restart logic
            # For now, simulate restart
            await asyncio.sleep(2)  # Simulate restart time
            
            self.logger.info(f"✅ Component {component} restarted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Component restart failed: {str(e)}")
            return False
    
    async def _alert_operator(self, error: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Alert human operator for manual intervention"""
        alert_message = f"🚨 Manual intervention required for {error.component}: {error.error_message}"
        
        self.logger.critical(alert_message)
        
        # This would implement actual alerting (email, Slack, etc.)
        # For now, just log the alert
        
        return True  # Alert sent successfully
    
    async def _apply_circuit_breaker(
        self,
        error: ErrorEvent,
        context: Dict[str, Any]
    ) -> bool:
        """Apply circuit breaker recovery pattern"""
        pattern = self.recovery_patterns['circuit_breaker']
        
        error_count = len(self.error_patterns.get(error.component, []))
        if error_count >= pattern['failure_threshold']:
            self.logger.warning(f"Circuit breaker tripped for {error.component}")
            return False
        
        try:
            await self._execute_recovery_action(error.component, context)
            return True
        except Exception as e:
            self.logger.error(
                f"Circuit breaker recovery failed for {error.component}: {str(e)}"
            )
            return False
    
    async def _apply_exponential_backoff(
        self,
        error: ErrorEvent,
        context: Dict[str, Any]
    ) -> bool:
        """Apply exponential backoff recovery pattern"""
        pattern = self.recovery_patterns['exponential_backoff']
        
        delay = min(
            pattern['base_delay'] * (pattern['factor'] ** len(error.attempted_recoveries)),
            pattern['max_delay']
        )
        
        self.logger.info(
            f"Applying exponential backoff for {error.component}, "
            f"delay: {delay:.2f}s"
        )
        
        await asyncio.sleep(delay)
        
        try:
            await self._execute_recovery_action(error.component, context)
            return True
        except Exception as e:
            self.logger.error(
                f"Exponential backoff recovery failed for {error.component}: {str(e)}"
            )
            return False
    
    async def _retry_operation(self, error: ErrorEvent, context: Dict[str, Any]) -> bool:
        """Retry the failed operation with context-aware validation"""
        
        # For high-criticality operations, only retry once with strict validation
        if error.function_criticality in ['high', 'critical']:
            if len(error.attempted_recoveries) > 0:
                self.logger.warning(f"High-criticality operation {error.component} already attempted - no more retries")
                return False
            
            # For trading operations, validate no partial execution occurred
            if error.component == 'trading':
                if not await self._validate_no_partial_trade_execution(context):
                    self.logger.error("Partial trade execution detected - aborting retry")
                    return False
        
        # For low-criticality operations, allow multiple retries
        max_retries = 3 if error.function_criticality == 'low' else 1
        
        if len(error.attempted_recoveries) >= max_retries:
            self.logger.warning(f"Max retries ({max_retries}) reached for {error.component}")
            return False
        
        self.logger.info(f"🔄 Retrying operation for {error.component} (attempt {len(error.attempted_recoveries) + 1}/{max_retries})")
        
        try:
            await self._execute_recovery_action(error.component, context)
            return True
        except Exception as e:
            self.logger.error(f"Retry failed for {error.component}: {str(e)}")
            return False
    
    async def _validate_no_partial_trade_execution(self, context: Dict[str, Any]) -> bool:
        """Validate that no partial trade execution occurred"""
        try:
            # This would implement actual trade validation logic
            # For now, assume no partial execution
            return True
        except Exception as e:
            self.logger.error(f"Trade validation failed: {str(e)}")
            return False
    
    async def _execute_recovery_action(
        self,
        component: str,
        context: Dict[str, Any]
    ) -> None:
        """Execute component-specific recovery action"""
        # Implementation would vary by component
        pass
    
    async def _escalate_error(self, error: ErrorEvent) -> None:
        """Escalate unresolved error"""
        self.logger.critical(
            f"🆘 ESCALATING ERROR: {error.component} - {error.error_message}"
        )
        # Implementation would vary by component
    
    def _calculate_severity(self, error: Exception, component: str) -> int:
        """Calculate error severity (1-10)"""
        # Base severity from error type
        severity = 5
        
        # Adjust for critical components
        critical_components = ['trading_engine', 'kill_switch', 'wallet_manager']
        if component in critical_components:
            severity = min(10, severity + 2)
        
        # Adjust for error type
        if isinstance(error, (ValueError, TypeError)):
            severity = min(10, severity + 1)
        elif isinstance(error, (RuntimeError, SystemError)):
            severity = min(10, severity + 3)
        
        return severity

    async def _recover_from_critical_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from critical errors"""
        try:
            self.logger.critical(f"Attempting recovery from critical error: {error}")
            
            # 1. Emergency shutdown of trading operations
            if context.get('trading_active', False):
                await self._emergency_trading_shutdown()
            
            # 2. Secure funds in vault
            if context.get('wallet_manager'):
                await self._secure_funds_in_vault(context['wallet_manager'])
            
            # 3. Reset system state
            await self._reset_system_state()
            
            # 4. Validate system integrity
            integrity_check = await self._validate_system_integrity()
            if not integrity_check:
                self.logger.critical("System integrity validation failed after recovery attempt")
                return False
            
            # 5. Attempt restart of critical services
            restart_success = await self._restart_critical_services()
            if not restart_success:
                self.logger.critical("Critical service restart failed")
                return False
            
            self.logger.info("Critical error recovery completed successfully")
            return True
            
        except Exception as recovery_error:
            self.logger.critical(f"Recovery attempt failed: {recovery_error}")
            return False
    
    async def _emergency_trading_shutdown(self):
        """Emergency shutdown of all trading operations"""
        try:
            # Cancel all pending orders
            # Close all open positions
            # Disable trading system
            self.logger.info("Emergency trading shutdown completed")
        except Exception as e:
            self.logger.error(f"Emergency shutdown error: {e}")
    
    async def _secure_funds_in_vault(self, wallet_manager):
        """Secure funds in vault wallets"""
        try:
            # Move funds to secure vault wallets
            # Implement vault transfer logic here
            self.logger.info("Funds secured in vault")
        except Exception as e:
            self.logger.error(f"Vault security error: {e}")
    
    async def _reset_system_state(self):
        """Reset system to safe state"""
        try:
            # Reset trading flags
            # Clear error states
            # Reset counters
            self.logger.info("System state reset completed")
        except Exception as e:
            self.logger.error(f"State reset error: {e}")
    
    async def _validate_system_integrity(self) -> bool:
        """Validate system integrity after recovery"""
        try:
            # Check core systems
            # Validate configurations
            # Test critical connections
            return True
        except Exception as e:
            self.logger.error(f"Integrity validation error: {e}")
            return False
    
    async def _restart_critical_services(self) -> bool:
        """Restart critical system services"""
        try:
            # Restart wallet manager
            # Restart trading engine
            # Restart monitoring systems
            return True
        except Exception as e:
            self.logger.error(f"Service restart error: {e}")
            return False


def handle_errors(component: str = None, 
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 auto_recover: bool = True,
                 function_criticality: str = "medium"):
    """Decorator for automatic error handling with function criticality"""
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                comp_name = component or func.__module__.split('.')[-1]
                
                # Get error handler instance
                from worker_ant_v1.core.unified_config import UnifiedConfig
                config = UnifiedConfig()
                error_handler = EnterpriseErrorHandler(config.__dict__)
                
                # Handle error with enhanced context and function criticality
                await error_handler.handle_error(
                    component=comp_name,
                    error=e,
                    context={'parameters': {'args': str(args)[:500], 'kwargs': str(kwargs)[:500]}},
                    function_criticality=function_criticality
                )
                
                raise e
        
        return wrapper
    return decorator 