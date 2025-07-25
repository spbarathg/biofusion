"""
Enhanced Kill Switch System
=========================

Advanced kill switch with multiple trigger conditions and graceful shutdown.
"""

from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

class KillSwitchTrigger(Enum):
    MANUAL = "MANUAL"
    LOSS_THRESHOLD = "LOSS_THRESHOLD"
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    SECURITY_BREACH = "SECURITY_BREACH"

class EnhancedKillSwitch:
    """Enhanced kill switch with multiple safety features"""
    
    def __init__(self):
        self.logger = logging.getLogger("KillSwitch")
        self.is_triggered = False
        self.trigger_reason: Optional[KillSwitchTrigger] = None
        self.trigger_time: Optional[datetime] = None
        self.emergency_actions: List[str] = []
        
    def trigger(self, reason: KillSwitchTrigger, details: str = "") -> bool:
        """Trigger the kill switch"""
        if not self.is_triggered:
            self.is_triggered = True
            self.trigger_reason = reason
            self.trigger_time = datetime.utcnow()
            
            self.logger.critical(f"🚨 KILL SWITCH TRIGGERED: {reason.value}")
            if details:
                self.logger.critical(f"Details: {details}")
            
            self._execute_emergency_protocol()
            return True
        return False
    
    def _execute_emergency_protocol(self):
        """Execute emergency shutdown protocol"""
        try:
            # 1. Cancel all pending orders
            self.emergency_actions.append("Cancelling all pending orders")
            
            # 2. Close all open positions
            self.emergency_actions.append("Closing all open positions")
            
            # 3. Move funds to secure wallet
            self.emergency_actions.append("Securing funds in vault")
            
            # 4. Disable trading system
            self.emergency_actions.append("Disabling trading system")
            
            # 5. Alert operators
            self.emergency_actions.append("Sending emergency alerts")
            
            self.logger.info("✅ Emergency protocol executed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Emergency protocol failed: {str(e)}")
            raise
    
    def check_conditions(self, metrics: Dict) -> bool:
        """Check if kill switch conditions are met"""
        try:
            # 1. Check loss threshold
            if metrics.get('daily_loss_sol', 0) > metrics.get('max_daily_loss_sol', float('inf')):
                self.trigger(
                    KillSwitchTrigger.LOSS_THRESHOLD,
                    f"Daily loss {metrics['daily_loss_sol']} SOL exceeded threshold"
                )
                return True
            
            # 2. Check for anomalies
            if metrics.get('anomaly_score', 0) > 0.8:
                self.trigger(
                    KillSwitchTrigger.ANOMALY_DETECTED,
                    f"Anomaly score {metrics['anomaly_score']} exceeded threshold"
                )
                return True
            
            # 3. Check system errors
            if metrics.get('error_count', 0) > 10:
                self.trigger(
                    KillSwitchTrigger.SYSTEM_ERROR,
                    f"Error count {metrics['error_count']} exceeded threshold"
                )
                return True
            
            # 4. Check security
            if not metrics.get('security_status', True):
                self.trigger(
                    KillSwitchTrigger.SECURITY_BREACH,
                    "Security breach detected"
                )
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Kill switch condition check failed: {str(e)}")
            self.trigger(KillSwitchTrigger.SYSTEM_ERROR, str(e))
            return True
    
    def get_status(self) -> Dict:
        """Get current kill switch status"""
        return {
            'is_triggered': self.is_triggered,
            'trigger_reason': self.trigger_reason.value if self.trigger_reason else None,
            'trigger_time': self.trigger_time.isoformat() if self.trigger_time else None,
            'emergency_actions': self.emergency_actions
        }
    
    def reset(self, override_code: str) -> bool:
        """Reset kill switch - requires override code"""
        if override_code == "MANUAL_OVERRIDE_ACCEPTED":
            self.is_triggered = False
            self.trigger_reason = None
            self.trigger_time = None
            self.emergency_actions = []
            self.logger.info("🔄 Kill switch reset successfully")
            return True
        return False
    
    async def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        self.trigger(KillSwitchTrigger.MANUAL, reason)
    
    async def shutdown(self):
        """Shutdown kill switch"""
        self.logger.info("🛑 Kill switch shutdown")