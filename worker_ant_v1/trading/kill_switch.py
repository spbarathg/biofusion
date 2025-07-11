"""
KILL SWITCH - EMERGENCY SHUTDOWN
==============================

Emergency shutdown system with multiple trigger conditions
and graceful shutdown procedures.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from worker_ant_v1.core.unified_config import get_security_config
from worker_ant_v1.utils.logger import setup_logger

kill_logger = setup_logger(__name__)


class KillSwitchTrigger(Enum):
    """Kill switch trigger conditions"""
    
    MANUAL = "manual"
    LOSS_THRESHOLD = "loss_threshold"
    RAPID_LOSS = "rapid_loss"
    SYSTEM_ERROR = "system_error"
    NETWORK_ISSUE = "network_issue"
    SECURITY_BREACH = "security_breach"
    API_ERROR = "api_error"
    LIQUIDITY_LOW = "liquidity_low"
    HONEYPOT_DETECTED = "honeypot_detected"
    MARKET_MANIPULATION = "market_manipulation"
    SLIPPAGE_HIGH = "slippage_high"
    VOLUME_DROP = "volume_drop"
    PRICE_IMPACT_HIGH = "price_impact_high"
    TRANSACTION_FAILURE = "transaction_failure"
    WALLET_ISSUE = "wallet_issue"


@dataclass
class KillSwitchConfig:
    """Kill switch configuration"""
    
    # Loss thresholds
    max_loss_percent: float = 10.0
    rapid_loss_percent: float = 5.0
    rapid_loss_timeframe_seconds: int = 60
    
    # System health
    max_error_rate: float = 0.1
    min_success_rate: float = 0.8
    max_latency_ms: int = 5000
    
    # Network conditions
    min_network_reliability: float = 0.95
    max_rpc_failures: int = 3
    rpc_timeout_seconds: float = 10
    
    # Market conditions
    min_liquidity_sol: float = 10.0
    max_price_impact_percent: float = 3.0
    min_volume_24h_sol: float = 100.0
    max_slippage_percent: float = 2.0
    
    # Security settings
    enable_honeypot_detection: bool = True
    enable_manipulation_detection: bool = True
    max_wallet_exposure_percent: float = 20.0
    
    # Recovery settings
    auto_restart_enabled: bool = False
    restart_delay_seconds: int = 300
    max_restart_attempts: int = 3


class KillSwitch:
    """Emergency kill switch with comprehensive monitoring"""
    
    def __init__(self):
        self.logger = kill_logger
        self.config = KillSwitchConfig()
        
        # State tracking
        self.active = False
        self.triggered_by = None
        self.trigger_time = None
        self.trigger_reason = None
        
        # Performance metrics
        self.total_trades = 0
        self.failed_trades = 0
        self.total_errors = 0
        self.total_loss_sol = 0.0
        
        # Network metrics
        self.rpc_failures = 0
        self.network_latency_ms = 0
        self.last_network_check = None
        
        # Market metrics
        self.current_liquidity_sol = 0.0
        self.current_volume_sol = 0.0
        self.current_price_impact = 0.0
        self.current_slippage = 0.0
        
        # Load config
        self._load_config()
    
    def _load_config(self):
        """Load configuration from settings"""
        
        try:
            config = get_security_config()
            if config and "kill_switch" in config:
                kill_config = config["kill_switch"]
                
                # Load loss thresholds
                self.config.max_loss_percent = float(
                    kill_config.get(
                        "max_loss_percent",
                        self.config.max_loss_percent
                    )
                )
                self.config.rapid_loss_percent = float(
                    kill_config.get(
                        "rapid_loss_percent",
                        self.config.rapid_loss_percent
                    )
                )
                self.config.rapid_loss_timeframe_seconds = int(
                    kill_config.get(
                        "rapid_loss_timeframe_seconds",
                        self.config.rapid_loss_timeframe_seconds
                    )
                )
                
                # Load system health settings
                self.config.max_error_rate = float(
                    kill_config.get(
                        "max_error_rate",
                        self.config.max_error_rate
                    )
                )
                self.config.min_success_rate = float(
                    kill_config.get(
                        "min_success_rate",
                        self.config.min_success_rate
                    )
                )
                self.config.max_latency_ms = int(
                    kill_config.get(
                        "max_latency_ms",
                        self.config.max_latency_ms
                    )
                )
                
                # Load network settings
                self.config.min_network_reliability = float(
                    kill_config.get(
                        "min_network_reliability",
                        self.config.min_network_reliability
                    )
                )
                self.config.max_rpc_failures = int(
                    kill_config.get(
                        "max_rpc_failures",
                        self.config.max_rpc_failures
                    )
                )
                self.config.rpc_timeout_seconds = float(
                    kill_config.get(
                        "rpc_timeout_seconds",
                        self.config.rpc_timeout_seconds
                    )
                )
                
                # Load market condition settings
                self.config.min_liquidity_sol = float(
                    kill_config.get(
                        "min_liquidity_sol",
                        self.config.min_liquidity_sol
                    )
                )
                self.config.max_price_impact_percent = float(
                    kill_config.get(
                        "max_price_impact_percent",
                        self.config.max_price_impact_percent
                    )
                )
                self.config.min_volume_24h_sol = float(
                    kill_config.get(
                        "min_volume_24h_sol",
                        self.config.min_volume_24h_sol
                    )
                )
                self.config.max_slippage_percent = float(
                    kill_config.get(
                        "max_slippage_percent",
                        self.config.max_slippage_percent
                    )
                )
                
                # Load security settings
                self.config.enable_honeypot_detection = bool(
                    kill_config.get(
                        "enable_honeypot_detection",
                        self.config.enable_honeypot_detection
                    )
                )
                self.config.enable_manipulation_detection = bool(
                    kill_config.get(
                        "enable_manipulation_detection",
                        self.config.enable_manipulation_detection
                    )
                )
                self.config.max_wallet_exposure_percent = float(
                    kill_config.get(
                        "max_wallet_exposure_percent",
                        self.config.max_wallet_exposure_percent
                    )
                )
                
                # Load recovery settings
                self.config.auto_restart_enabled = bool(
                    kill_config.get(
                        "auto_restart_enabled",
                        self.config.auto_restart_enabled
                    )
                )
                self.config.restart_delay_seconds = int(
                    kill_config.get(
                        "restart_delay_seconds",
                        self.config.restart_delay_seconds
                    )
                )
                self.config.max_restart_attempts = int(
                    kill_config.get(
                        "max_restart_attempts",
                        self.config.max_restart_attempts
                    )
                )
                
        except Exception as e:
            self.logger.error(f"Failed to load kill switch config: {e}")
    
    def activate(
        self,
        trigger: KillSwitchTrigger,
        reason: str = "Emergency shutdown"
    ):
        """Activate kill switch"""
        
        if not self.active:
            self.active = True
            self.triggered_by = trigger
            self.trigger_time = datetime.utcnow()
            self.trigger_reason = reason
            
            self.logger.warning(
                f"Kill switch activated: {trigger.value} - {reason}"
            )
    
    def deactivate(self):
        """Deactivate kill switch"""
        
        if self.active:
            self.active = False
            self.triggered_by = None
            self.trigger_time = None
            self.trigger_reason = None
            
            self.logger.info("Kill switch deactivated")
    
    def is_active(self) -> bool:
        """Check if kill switch is active"""
        return self.active
    
    def get_status(self) -> Dict:
        """Get current kill switch status"""
        
        return {
            "active": self.active,
            "triggered_by": self.triggered_by.value if self.triggered_by else None,
            "trigger_time": self.trigger_time.isoformat() if self.trigger_time else None,
            "trigger_reason": self.trigger_reason,
            "total_trades": self.total_trades,
            "failed_trades": self.failed_trades,
            "total_errors": self.total_errors,
            "total_loss_sol": self.total_loss_sol,
            "rpc_failures": self.rpc_failures,
            "network_latency_ms": self.network_latency_ms,
            "current_liquidity_sol": self.current_liquidity_sol,
            "current_volume_sol": self.current_volume_sol,
            "current_price_impact": self.current_price_impact,
            "current_slippage": self.current_slippage
        }
    
    def update_metrics(
        self,
        trades: int = 0,
        failed: int = 0,
        errors: int = 0,
        loss_sol: float = 0.0,
        rpc_fails: int = 0,
        latency_ms: int = 0,
        liquidity_sol: float = 0.0,
        volume_sol: float = 0.0,
        price_impact: float = 0.0,
        slippage: float = 0.0
    ):
        """Update performance metrics"""
        
        self.total_trades += trades
        self.failed_trades += failed
        self.total_errors += errors
        self.total_loss_sol += loss_sol
        
        self.rpc_failures += rpc_fails
        self.network_latency_ms = latency_ms
        self.last_network_check = datetime.utcnow()
        
        self.current_liquidity_sol = liquidity_sol
        self.current_volume_sol = volume_sol
        self.current_price_impact = price_impact
        self.current_slippage = slippage
        
        # Check trigger conditions
        self._check_trigger_conditions()
    
    def _check_trigger_conditions(self):
        """Check if any kill switch conditions are met"""
        
        if self.active:
            return
        
        # Check loss thresholds
        if self.total_loss_sol >= self.config.max_loss_percent:
            self.activate(
                KillSwitchTrigger.LOSS_THRESHOLD,
                f"Total loss exceeded {self.config.max_loss_percent}%"
            )
            return
        
        # Check error rate
        if self.total_trades > 0:
            error_rate = self.total_errors / self.total_trades
            if error_rate >= self.config.max_error_rate:
                self.activate(
                    KillSwitchTrigger.SYSTEM_ERROR,
                    f"Error rate {error_rate:.1%} exceeded threshold"
                )
                return
        
        # Check network conditions
        if self.rpc_failures >= self.config.max_rpc_failures:
            self.activate(
                KillSwitchTrigger.NETWORK_ISSUE,
                f"RPC failures {self.rpc_failures} exceeded threshold"
            )
            return
        
        if self.network_latency_ms >= self.config.max_latency_ms:
            self.activate(
                KillSwitchTrigger.NETWORK_ISSUE,
                f"Network latency {self.network_latency_ms}ms too high"
            )
            return
        
        # Check market conditions
        if self.current_liquidity_sol < self.config.min_liquidity_sol:
            self.activate(
                KillSwitchTrigger.LIQUIDITY_LOW,
                f"Liquidity {self.current_liquidity_sol} SOL too low"
            )
            return
        
        if self.current_price_impact > self.config.max_price_impact_percent:
            self.activate(
                KillSwitchTrigger.PRICE_IMPACT_HIGH,
                f"Price impact {self.current_price_impact:.1f}% too high"
            )
            return
        
        if self.current_slippage > self.config.max_slippage_percent:
            self.activate(
                KillSwitchTrigger.SLIPPAGE_HIGH,
                f"Slippage {self.current_slippage:.1f}% too high"
            )
            return