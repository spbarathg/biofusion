"""
BATTLEFIELD ALERT SYSTEM
====================

Core alert system for battlefield survival:
- Critical pattern alerts
- Emergency kill switch notifications
- Threat detection warnings
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import os

from worker_ant_v1.utils.logger import get_logger

class AlertPriority(Enum):
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertChannel(Enum):
    CONSOLE = "console"
    FILE_LOG = "file_log"

@dataclass
class Alert:
    """Alert message structure"""
    alert_id: str
    title: str
    message: str
    priority: AlertPriority
    timestamp: datetime
    data: Dict[str, Any]

class BattlefieldAlertSystem:
    """Core battlefield alert system"""
    
    def __init__(self):
        self.logger = get_logger("BattlefieldAlerts")
        
        # Configuration
        self.alert_log_path = "data/alerts.log"
        
        # Alert queue and history
        self.alert_queue = asyncio.Queue()
        self.alert_history = []
        self.max_history = 1000
        
        # Rate limiting
        self.rate_limits = {
            AlertChannel.CONSOLE: {'count': 0, 'window_start': datetime.now()},
            AlertChannel.FILE_LOG: {'count': 0, 'window_start': datetime.now()}
        }
        self.max_alerts_per_minute = 10
        
        # Alert templates
        self.templates = {
            'pattern_detected': {
                'title': '🎯 Critical Pattern Detected',
                'template': 'Pattern detected: {pattern_type} - Confidence: {confidence:.1f}%'
            },
            'kill_switch_triggered': {
                'title': '🚨 Kill Switch Activated',
                'template': 'Emergency kill switch triggered: {reason}'
            },
            'rug_detected': {
                'title': '🔍 Rug Pull Detected',
                'template': 'Rug pull detected for token {token_symbol} ({token_address})'
            },
            'threat_detected': {
                'title': '⚠️ Threat Detected',
                'template': 'Threat detected: {threat_type} - Level: {threat_level}'
            }
        }
        
        # Processing active flag
        self.processing_active = False
        
    async def initialize(self):
        """Initialize alert system"""
        self.logger.info("📡 Initializing Battlefield Alert System")
        
        # Start alert processing
        await self._start_alert_processing()
        
        self.logger.info("✅ Battlefield Alert System ready")
        
    async def _start_alert_processing(self):
        """Start background alert processing"""
        self.processing_active = True
        asyncio.create_task(self._process_alert_queue())
        
    async def _process_alert_queue(self):
        """Process alerts from queue"""
        while self.processing_active:
            try:
                # Get alert from queue
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                
                # Process alert
                await self._send_alert(alert)
                
                # Add to history
                self.alert_history.append(alert)
                
                # Maintain history size
                if len(self.alert_history) > self.max_history:
                    self.alert_history = self.alert_history[-self.max_history:]
                
                # Mark task done
                self.alert_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"💥 Alert processing error: {e}")
    
    async def send_alert(self, title: str, message: str, 
                        priority: AlertPriority = AlertPriority.WARNING,
                        data: Dict[str, Any] = None):
        """Send battlefield alert"""
        if data is None:
            data = {}
        
        # Create alert object
        alert = Alert(
            alert_id=f"alert_{int(datetime.now().timestamp())}",
            title=title,
            message=message,
            priority=priority,
            timestamp=datetime.now(),
            data=data
        )
        
        # Add to queue
        await self.alert_queue.put(alert)
        
    async def send_templated_alert(self, template_name: str, 
                                 priority: AlertPriority = AlertPriority.WARNING,
                                 **kwargs):
        """Send alert using predefined template"""
        if template_name not in self.templates:
            self.logger.error(f"💥 Unknown template: {template_name}")
            return
        
        template = self.templates[template_name]
        
        try:
            # Format message with provided data
            formatted_message = template['template'].format(**kwargs)
            
            await self.send_alert(
                title=template['title'],
                message=formatted_message,
                priority=priority,
                data=kwargs
            )
            
        except KeyError as e:
            self.logger.error(f"💥 Template formatting error: Missing key {e}")
    
    async def _send_alert(self, alert: Alert):
        """Send alert to all channels"""
        try:
            # Check rate limits
            if not await self._check_rate_limits():
                self.logger.warning(f"⚠️ Rate limit exceeded for alert: {alert.title}")
                return
            
            # Console alert
            self.logger.warning(f"{alert.title}: {alert.message}")
            
            # File alert
            try:
                with open(self.alert_log_path, 'a') as f:
                    f.write(f"[{alert.timestamp}] {alert.priority.name} - {alert.title}: {alert.message}\n")
            except Exception as e:
                self.logger.error(f"💥 Failed to write alert to file: {e}")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to send alert: {e}")
    
    async def _check_rate_limits(self) -> bool:
        """Check alert rate limits"""
        now = datetime.now()
        
        for channel in self.rate_limits:
            # Reset counter if window expired
            if (now - self.rate_limits[channel]['window_start']).total_seconds() >= 60:
                self.rate_limits[channel]['count'] = 0
                self.rate_limits[channel]['window_start'] = now
            
            # Check limit
            if self.rate_limits[channel]['count'] >= self.max_alerts_per_minute:
                return False
            
            # Increment counter
            self.rate_limits[channel]['count'] += 1
        
        return True
    
    async def send_emergency_alert(self, title: str, message: str, data: Dict[str, Any] = None):
        """Send emergency priority alert"""
        await self.send_alert(
            title=title,
            message=message,
            priority=AlertPriority.EMERGENCY,
            data=data
        )
    
    async def shutdown(self):
        """Shutdown alert system"""
        self.logger.info("Shutting down alert system")
        self.processing_active = False
        
        # Wait for queue to empty
        if not self.alert_queue.empty():
            await self.alert_queue.join()

async def create_alert_system() -> BattlefieldAlertSystem:
    """Create and initialize alert system"""
    alert_system = BattlefieldAlertSystem()
    await alert_system.initialize()
    return alert_system
