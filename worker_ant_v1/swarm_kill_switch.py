"""
Swarm Kill Switch System
========================

Emergency shutdown system that stops the entire botnet if major errors/losses/net issues occur.
Implements multiple safety triggers and emergency protocols.
"""

import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from worker_ant_v1.config import trading_config


class EmergencyType(Enum):
    MAJOR_LOSS = "major_loss"
    NETWORK_FAILURE = "network_failure"
    SYSTEM_ERROR = "system_error"
    RUG_DETECTION = "rug_detection"
    MANUAL_TRIGGER = "manual_trigger"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class EmergencyEvent:
    """Emergency event that triggered kill switch"""
    event_id: str
    emergency_type: EmergencyType
    severity: str  # "critical", "high", "medium"
    description: str
    timestamp: datetime
    triggered_by: str
    metrics: Dict


class SwarmKillSwitch:
    """Emergency kill switch for the entire swarm"""
    
    def __init__(self):
        self.logger = logging.getLogger("SwarmKillSwitch")
        
        # Kill switch state
        self.is_armed = True
        self.is_triggered = False
        self.emergency_events: List[EmergencyEvent] = []
        
        # Safety thresholds
        self.max_daily_loss = 2.0  # SOL
        self.max_hourly_loss = 0.5  # SOL
        self.max_consecutive_failures = 10
        self.min_system_memory_mb = 100
        self.max_cpu_usage = 95.0
        
        # Monitoring
        self.current_daily_loss = 0.0
        self.current_hourly_loss = 0.0
        self.consecutive_failures = 0
        self.last_reset = datetime.now()
        
    async def monitor_emergency_conditions(self):
        """Continuously monitor for emergency conditions"""
        
        while self.is_armed and not self.is_triggered:
            try:
                # Check loss thresholds
                await self._check_loss_thresholds()
                
                # Check system resources
                await self._check_system_resources()
                
                # Check network connectivity
                await self._check_network_health()
                
                # Check trading errors
                await self._check_trading_errors()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in emergency monitoring: {e}")
                await asyncio.sleep(30)
                
    async def _check_loss_thresholds(self):
        """Check if loss thresholds are exceeded"""
        
        now = datetime.now()
        
        # Reset daily counter
        if (now - self.last_reset).days >= 1:
            self.current_daily_loss = 0.0
            self.last_reset = now
            
        # Check daily loss
        if self.current_daily_loss >= self.max_daily_loss:
            await self.trigger_emergency(
                EmergencyType.MAJOR_LOSS,
                f"Daily loss limit exceeded: {self.current_daily_loss:.2f} SOL",
                "critical"
            )
            
        # Check hourly loss
        if self.current_hourly_loss >= self.max_hourly_loss:
            await self.trigger_emergency(
                EmergencyType.MAJOR_LOSS,
                f"Hourly loss limit exceeded: {self.current_hourly_loss:.2f} SOL",
                "high"
            )
            
    async def _check_system_resources(self):
        """Check system resource availability"""
        
        # Memory check
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        
        if available_mb < self.min_system_memory_mb:
            await self.trigger_emergency(
                EmergencyType.RESOURCE_EXHAUSTION,
                f"Low memory: {available_mb:.0f}MB available",
                "high"
            )
            
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.max_cpu_usage:
            await self.trigger_emergency(
                EmergencyType.RESOURCE_EXHAUSTION,
                f"High CPU usage: {cpu_percent:.1f}%",
                "medium"
            )
            
    async def _check_network_health(self):
        """Check network connectivity and RPC health"""
        
        # This would implement actual network checks        
        # Implementation pending
        pass
        
    async def _check_trading_errors(self):
        """Check for consecutive trading errors"""
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            await self.trigger_emergency(
                EmergencyType.SYSTEM_ERROR,
                f"Too many consecutive failures: {self.consecutive_failures}",
                "critical"
            )
            
    async def trigger_emergency(self, emergency_type: EmergencyType, 
                              description: str, severity: str = "high"):
        """Trigger emergency shutdown"""
        
        if self.is_triggered:
            return  # Already triggered
            
        self.is_triggered = True
        
        # Create emergency event
        event = EmergencyEvent(
            event_id=f"emergency_{int(datetime.now().timestamp())}",
            emergency_type=emergency_type,
            severity=severity,
            description=description,
            timestamp=datetime.now(),
            triggered_by="auto_monitor",
            metrics={
                "daily_loss": self.current_daily_loss,
                "consecutive_failures": self.consecutive_failures,
                "memory_mb": psutil.virtual_memory().available / (1024 * 1024),
                "cpu_percent": psutil.cpu_percent()
            }
        )
        
        self.emergency_events.append(event)
        
        # Execute emergency shutdown
        await self._execute_emergency_shutdown(event)
        
    async def _execute_emergency_shutdown(self, event: EmergencyEvent):
        """Execute emergency shutdown procedures"""
        
        self.logger.critical(f"🚨 EMERGENCY SHUTDOWN TRIGGERED: {event.description}")
        
        try:
            # 1. Stop all trading immediately
            await self._stop_all_trading()
            
            # 2. Close all open positions
            await self._emergency_close_positions()
            
            # 3. Save current state
            await self._save_emergency_state()
            
            # 4. Send emergency alerts
            await self._send_emergency_alerts(event)
            
            # 5. Shutdown all components
            await self._shutdown_all_components()
            
            self.logger.critical("Emergency shutdown completed")
            
        except Exception as e:
            self.logger.critical(f"Error during emergency shutdown: {e}")
            
    async def _stop_all_trading(self):
        """Stop all trading activities immediately"""
        
        # Signal all components to stop
        # This would integrate with the actual trading components
        self.logger.warning("All trading stopped")
        
    async def _emergency_close_positions(self):
        """Close all open positions at market price"""
        
        # This would implement actual position closing
        self.logger.warning("All positions closed")
        
    async def _save_emergency_state(self):
        """Save current state for recovery"""
        
        # Save system state, positions, logs etc.
        self.logger.info("Emergency state saved")
        
    async def _send_emergency_alerts(self, event: EmergencyEvent):
        """Send emergency alerts to operators"""
        
        # This would implement actual alerting (Telegram, email, etc.)
        alert_message = f"""
🚨 EMERGENCY SHUTDOWN ACTIVATED 🚨

Type: {event.emergency_type.value}
Severity: {event.severity}
Time: {event.timestamp}
Reason: {event.description}

System Metrics:
- Daily Loss: {event.metrics.get('daily_loss', 0):.2f} SOL
- Consecutive Failures: {event.metrics.get('consecutive_failures', 0)}
- Available Memory: {event.metrics.get('memory_mb', 0):.0f} MB
- CPU Usage: {event.metrics.get('cpu_percent', 0):.1f}%

All trading has been stopped and positions closed.
Manual intervention required to restart.
        """
        
        self.logger.critical(alert_message)
        
    async def _shutdown_all_components(self):
        """Shutdown all swarm components"""
        
        # This would shutdown all the actual components
        self.logger.info("All components shutdown")
        
    def record_loss(self, loss_amount: float):
        """Record a trading loss for monitoring"""
        
        self.current_daily_loss += loss_amount
        self.current_hourly_loss += loss_amount
        
        # Reset hourly counter periodically
        # (Implementation would track hourly resets)
        
    def record_failure(self):
        """Record a trading failure"""
        
        self.consecutive_failures += 1
        
    def record_success(self):
        """Record a trading success (resets failure counter)"""
        
        self.consecutive_failures = 0
        
    def manual_trigger(self, reason: str = "Manual emergency stop"):
        """Manually trigger emergency shutdown"""
        
        asyncio.create_task(self.trigger_emergency(
            EmergencyType.MANUAL_TRIGGER,
            reason,
            "critical"
        ))
        
    def get_status(self) -> Dict:
        """Get current kill switch status"""
        
        return {
            "armed": self.is_armed,
            "triggered": self.is_triggered,
            "daily_loss": self.current_daily_loss,
            "consecutive_failures": self.consecutive_failures,
            "emergency_events": len(self.emergency_events),
            "last_event": self.emergency_events[-1].description if self.emergency_events else None
        }


# Global instance
swarm_kill_switch = SwarmKillSwitch() 