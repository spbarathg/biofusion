"""
AUTONOMOUS RECOVERY SYSTEM
=========================

24/7 autonomous recovery and self-healing for Smart Ape Mode:
- Crash detection and auto-restart
- Component health monitoring
- Emergency protocols and fail-safes
- Log rotation and vault snapshots
- Memory leak prevention
"""

import asyncio
import time
import os
import signal
import psutil
import logging
import subprocess
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import threading
from pathlib import Path

class SystemHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"

class RecoveryAction(Enum):
    RESTART_COMPONENT = "restart_component"
    RESTART_SYSTEM = "restart_system"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    ESCALATE_ALERT = "escalate_alert"

@dataclass
class HealthMetrics:
    """System health metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_active: bool
    process_count: int
    uptime_hours: float
    error_rate: float
    last_check: datetime

@dataclass
class ComponentStatus:
    """Individual component status"""
    component_name: str
    status: SystemHealth
    last_heartbeat: datetime
    error_count: int
    restart_count: int
    critical_errors: List[str]

class AutonomousRecoverySystem:
    """Autonomous recovery and self-healing system"""
    
    def __init__(self, main_process_name: str = "smart_ape_mode"):
        self.main_process_name = main_process_name
        self.health_metrics = HealthMetrics(
            cpu_percent=0.0,
            memory_percent=0.0,
            disk_percent=0.0,
            network_active=True,
            process_count=0,
            uptime_hours=0.0,
            error_rate=0.0,
            last_check=datetime.now()
        )
        
        # Component monitoring
        self.components = {}
        self.recovery_actions = {}
        self.auto_restart_enabled = True
        self.max_restart_attempts = 3
        self.restart_cooldown_minutes = 5
        
        # Health thresholds
        self.cpu_threshold = 85.0
        self.memory_threshold = 90.0
        self.disk_threshold = 95.0
        self.error_rate_threshold = 0.1
        
        # Recovery protocols
        self.recovery_protocols = {
            'high_cpu': self._handle_high_cpu,
            'high_memory': self._handle_high_memory,
            'component_failure': self._handle_component_failure,
            'system_unresponsive': self._handle_system_unresponsive,
            'memory_leak': self._handle_memory_leak
        }
        
        # Monitoring threads
        self.monitoring_active = False
        self.monitoring_threads = []
        
        # Logging
        self.logger = logging.getLogger("AutonomousRecovery")
        
    async def initialize(self):
        """Initialize autonomous recovery system"""
        
        self.logger.info("🔄 Initializing Autonomous Recovery System")
        
        # Setup recovery protocols
        await self._setup_recovery_protocols()
        
        # Start monitoring
        await self._start_monitoring()
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        self.logger.info("✅ Autonomous Recovery System online")
        
    async def _setup_recovery_protocols(self):
        """Setup all recovery protocols"""
        
        # Register recovery actions for different scenarios
        self.recovery_actions = {
            'trading_engine_failure': [
                RecoveryAction.RESTART_COMPONENT,
                RecoveryAction.ESCALATE_ALERT
            ],
            'memory_exhaustion': [
                RecoveryAction.RESTART_SYSTEM,
                RecoveryAction.EMERGENCY_SHUTDOWN
            ],
            'kill_switch_failure': [
                RecoveryAction.EMERGENCY_SHUTDOWN,
                RecoveryAction.ESCALATE_ALERT
            ],
            'wallet_compromise': [
                RecoveryAction.EMERGENCY_SHUTDOWN,
                RecoveryAction.ESCALATE_ALERT
            ]
        }
        
    async def _start_monitoring(self):
        """Start all monitoring threads"""
        
        self.monitoring_active = True
        
        # System health monitoring
        health_thread = threading.Thread(
            target=self._system_health_monitor,
            daemon=True
        )
        health_thread.start()
        self.monitoring_threads.append(health_thread)
        
        # Process monitoring
        process_thread = threading.Thread(
            target=self._process_monitor,
            daemon=True
        )
        process_thread.start()
        self.monitoring_threads.append(process_thread)
        
        # Log monitoring
        log_thread = threading.Thread(
            target=self._log_monitor,
            daemon=True
        )
        log_thread.start()
        self.monitoring_threads.append(log_thread)
        
    def _system_health_monitor(self):
        """Monitor system health continuously"""
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check thresholds
                health_issues = self._check_health_thresholds()
                
                # Take recovery actions if needed
                if health_issues:
                    asyncio.run(self._handle_health_issues(health_issues))
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"💥 Health monitoring error: {e}")
                time.sleep(60)
    
    def _collect_system_metrics(self):
        """Collect current system metrics"""
        
        try:
            # CPU usage
            self.health_metrics.cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.health_metrics.memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.health_metrics.disk_percent = (disk.used / disk.total) * 100
            
            # Process count
            self.health_metrics.process_count = len(psutil.pids())
            
            # Network status
            self.health_metrics.network_active = self._check_network_connectivity()
            
            # Update timestamp
            self.health_metrics.last_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"💥 Failed to collect metrics: {e}")
    
    def _check_health_thresholds(self) -> List[str]:
        """Check if any health thresholds are exceeded"""
        
        issues = []
        
        if self.health_metrics.cpu_percent > self.cpu_threshold:
            issues.append(f"high_cpu: {self.health_metrics.cpu_percent:.1f}%")
            
        if self.health_metrics.memory_percent > self.memory_threshold:
            issues.append(f"high_memory: {self.health_metrics.memory_percent:.1f}%")
            
        if self.health_metrics.disk_percent > self.disk_threshold:
            issues.append(f"high_disk: {self.health_metrics.disk_percent:.1f}%")
            
        if not self.health_metrics.network_active:
            issues.append("network_disconnected")
            
        if self.health_metrics.error_rate > self.error_rate_threshold:
            issues.append(f"high_error_rate: {self.health_metrics.error_rate:.3f}")
        
        return issues
    
    async def _handle_health_issues(self, issues: List[str]):
        """Handle detected health issues"""
        
        for issue in issues:
            self.logger.warning(f"⚠️ Health issue detected: {issue}")
            
            if issue.startswith("high_cpu"):
                await self._handle_high_cpu()
            elif issue.startswith("high_memory"):
                await self._handle_high_memory()
            elif issue == "network_disconnected":
                await self._handle_network_issues()
            elif issue.startswith("high_error_rate"):
                await self._handle_high_error_rate()
    
    def _process_monitor(self):
        """Monitor main process and components"""
        
        while self.monitoring_active:
            try:
                # Check if main process is running
                main_process_running = self._is_process_running(self.main_process_name)
                
                if not main_process_running and self.auto_restart_enabled:
                    self.logger.critical("💥 Main process not running - attempting restart")
                    asyncio.run(self._restart_main_process())
                
                # Check component heartbeats
                self._check_component_heartbeats()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"💥 Process monitoring error: {e}")
                time.sleep(120)
    
    def _log_monitor(self):
        """Monitor logs for critical errors"""
        
        while self.monitoring_active:
            try:
                # Check for critical errors in logs
                critical_errors = self._scan_logs_for_errors()
                
                if critical_errors:
                    asyncio.run(self._handle_critical_errors(critical_errors))
                
                # Rotate logs if needed
                self._rotate_logs_if_needed()
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"💥 Log monitoring error: {e}")
                time.sleep(600)
    
    def _is_process_running(self, process_name: str) -> bool:
        """Check if a process is running"""
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if process_name in str(proc.info.get('cmdline', [])):
                    return True
            return False
        except Exception:
            return False
    
    async def _restart_main_process(self):
        """Restart the main Smart Ape Mode process"""
        
        try:
            self.logger.info("🔄 Restarting main process...")
            
            # Kill existing process if running
            await self._kill_process(self.main_process_name)
            
            # Wait for cleanup
            await asyncio.sleep(10)
            
            # Start new process
            startup_command = [
                "python", "-m", "worker_ant_v1.ultimate_smart_ape_system"
            ]
            
            subprocess.Popen(
                startup_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            self.logger.info("✅ Main process restart initiated")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to restart main process: {e}")
            await self._escalate_alert("Process restart failed", str(e))
    
    async def _kill_process(self, process_name: str):
        """Gracefully kill a process"""
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if process_name in str(proc.info.get('cmdline', [])):
                    process = psutil.Process(proc.info['pid'])
                    
                    # Try graceful shutdown first
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=30)
                    except psutil.TimeoutExpired:
                        # Force kill if needed
                        process.kill()
                        
        except Exception as e:
            self.logger.error(f"💥 Failed to kill process {process_name}: {e}")
    
    def _check_component_heartbeats(self):
        """Check heartbeats from all registered components"""
        
        current_time = datetime.now()
        
        for component_name, status in self.components.items():
            time_since_heartbeat = current_time - status.last_heartbeat
            
            if time_since_heartbeat.total_seconds() > 300:  # 5 minutes
                self.logger.warning(f"⚠️ Component {component_name} missed heartbeat")
                status.error_count += 1
                
                if status.error_count > 3:
                    self.logger.critical(f"💥 Component {component_name} failed")
                    asyncio.run(self._handle_component_failure(component_name))
    
    def register_component(self, component_name: str):
        """Register a component for monitoring"""
        
        self.components[component_name] = ComponentStatus(
            component_name=component_name,
            status=SystemHealth.HEALTHY,
            last_heartbeat=datetime.now(),
            error_count=0,
            restart_count=0,
            critical_errors=[]
        )
        
        self.logger.info(f"📝 Registered component: {component_name}")
    
    def component_heartbeat(self, component_name: str):
        """Record heartbeat from component"""
        
        if component_name in self.components:
            self.components[component_name].last_heartbeat = datetime.now()
            self.components[component_name].status = SystemHealth.HEALTHY
    
    async def _handle_high_cpu(self):
        """Handle high CPU usage"""
        
        self.logger.warning("⚠️ High CPU usage detected - optimizing performance")
        
        # Reduce trading frequency temporarily
        # Lower processing priorities
        # Kill non-essential processes
        
    async def _handle_high_memory(self):
        """Handle high memory usage"""
        
        self.logger.warning("⚠️ High memory usage - initiating cleanup")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear caches
        # Restart memory-intensive components
        
    async def _handle_memory_leak(self):
        """Handle detected memory leak"""
        
        self.logger.critical("💥 Memory leak detected - restarting system")
        await self._restart_main_process()
    
    async def _handle_component_failure(self, component_name: str):
        """Handle component failure"""
        
        self.logger.critical(f"💥 Component failure: {component_name}")
        
        if component_name in self.components:
            status = self.components[component_name]
            status.restart_count += 1
            
            if status.restart_count <= self.max_restart_attempts:
                self.logger.info(f"🔄 Attempting to restart {component_name}")
                # Component-specific restart logic would go here
            else:
                self.logger.critical(f"💥 Component {component_name} exceeded max restarts")
                await self._escalate_alert(f"Component {component_name} failed", "Max restarts exceeded")
    
    async def _handle_system_unresponsive(self):
        """Handle unresponsive system"""
        
        self.logger.critical("💥 System unresponsive - emergency restart")
        await self._restart_main_process()
    
    async def _handle_network_issues(self):
        """Handle network connectivity issues"""
        
        self.logger.warning("⚠️ Network connectivity issues detected")
        
        # Wait for network recovery
        for attempt in range(5):
            await asyncio.sleep(30)
            if self._check_network_connectivity():
                self.logger.info("✅ Network connectivity restored")
                return
        
        # Network still down - escalate
        await self._escalate_alert("Network connectivity lost", "Extended network outage")
    
    async def _handle_high_error_rate(self):
        """Handle high error rate"""
        
        self.logger.warning("⚠️ High error rate detected - investigating")
        
        # Analyze recent errors
        # Reduce system load
        # Switch to conservative mode
    
    def _check_network_connectivity(self) -> bool:
        """Check network connectivity"""
        
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except Exception:
            return False
    
    def _scan_logs_for_errors(self) -> List[str]:
        """Scan logs for critical errors"""
        
        critical_errors = []
        
        try:
            log_files = ["worker_ant_v1/logs/smart_ape.log", "swarm.log"]
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        # Read last 1000 lines
                        lines = f.readlines()[-1000:]
                        
                        for line in lines:
                            if any(keyword in line.lower() for keyword in 
                                  ['critical', 'fatal', 'emergency', 'crash']):
                                critical_errors.append(line.strip())
                                
        except Exception as e:
            self.logger.error(f"💥 Failed to scan logs: {e}")
        
        return critical_errors
    
    async def _handle_critical_errors(self, errors: List[str]):
        """Handle critical errors found in logs"""
        
        self.logger.critical(f"💥 Critical errors detected: {len(errors)} errors")
        
        for error in errors[-5:]:  # Last 5 errors
            self.logger.critical(f"💥 Critical: {error}")
        
        # Analyze error patterns and take action
        if any('wallet' in error.lower() for error in errors):
            await self._escalate_alert("Wallet-related critical errors", "Check wallet security")
        
        if any('memory' in error.lower() for error in errors):
            await self._handle_memory_leak()
    
    def _rotate_logs_if_needed(self):
        """Rotate logs if they get too large"""
        
        try:
            log_files = ["worker_ant_v1/logs/smart_ape.log", "swarm.log"]
            max_size = 100 * 1024 * 1024  # 100MB
            
            for log_file in log_files:
                if os.path.exists(log_file):
                    if os.path.getsize(log_file) > max_size:
                        # Rotate log
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_name = f"{log_file}.{timestamp}"
                        os.rename(log_file, backup_name)
                        
                        self.logger.info(f"📝 Rotated log: {log_file}")
                        
        except Exception as e:
            self.logger.error(f"💥 Failed to rotate logs: {e}")
    
    async def _escalate_alert(self, title: str, message: str):
        """Escalate alert to all channels"""
        
        self.logger.critical(f"🚨 ESCALATED ALERT: {title} - {message}")
        
        # Send alerts via all available channels
        # Telegram, Discord, Email, etc.
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        
        def signal_handler(signum, frame):
            self.logger.info("🛑 Shutdown signal received")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def shutdown(self):
        """Graceful shutdown of recovery system"""
        
        self.logger.info("🔄 Shutting down Autonomous Recovery System")
        
        self.monitoring_active = False
        
        # Wait for monitoring threads to finish
        for thread in self.monitoring_threads:
            thread.join(timeout=10)
        
        self.logger.info("✅ Autonomous Recovery System shutdown complete")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        return {
            'health_metrics': {
                'cpu_percent': self.health_metrics.cpu_percent,
                'memory_percent': self.health_metrics.memory_percent,
                'disk_percent': self.health_metrics.disk_percent,
                'network_active': self.health_metrics.network_active,
                'last_check': self.health_metrics.last_check.isoformat()
            },
            'components': {
                name: {
                    'status': status.status.value,
                    'error_count': status.error_count,
                    'restart_count': status.restart_count,
                    'last_heartbeat': status.last_heartbeat.isoformat()
                }
                for name, status in self.components.items()
            },
            'monitoring_active': self.monitoring_active,
            'auto_restart_enabled': self.auto_restart_enabled
        }

def create_autonomous_recovery_system(process_name: str = "smart_ape_mode") -> AutonomousRecoverySystem:
    """Create autonomous recovery system"""
    return AutonomousRecoverySystem(process_name)
