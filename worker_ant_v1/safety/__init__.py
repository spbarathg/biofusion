"""
Smart Ape Mode - Safety Module
==============================

Safety and protection systems including:
- Kill switch mechanisms
- Alert systems for notifications
- Recovery systems for fault tolerance
- Rug detection and protection
"""

from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch, KillSwitchTrigger, SafetyAlert, ThreatLevel
from worker_ant_v1.safety.alert_system import AdvancedAlertSystem, AlertPriority, AlertChannel
from worker_ant_v1.safety.recovery_system import AutonomousRecoverySystem, RecoveryAction
from worker_ant_v1.safety.rug_detector import AdvancedRugDetector, RugDetectionResult, RugConfidence

__all__ = [
    'EnhancedKillSwitch',
    'KillSwitchTrigger',
    'SafetyAlert',
    'ThreatLevel',
    'AdvancedAlertSystem',
    'AlertPriority', 
    'AlertChannel',
    'AutonomousRecoverySystem',
    'RecoveryAction',
    'AdvancedRugDetector',
    'RugDetectionResult',
    'RugConfidence'
]
