"""
Smart Ape Mode - Simplified Configuration
========================================

Lightweight configuration management without external dependencies.
For production use, replace with the full config.py after installing dependencies.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class TradingMode(Enum):
    LIVE = "live"
    SIMULATION = "simulation"
    PAPER = "paper"
    TESTING = "testing"


@dataclass
class TradingConfig:
    """Simplified trading configuration"""
    
    # Basic trading parameters
    trade_amount_sol: float = 0.1
    min_trade_amount_sol: float = 0.05
    max_trade_amount_sol: float = 0.25
    
    # Profit and loss targets
    base_profit_target_percent: float = 5.0
    base_stop_loss_percent: float = 3.0
    max_daily_loss_sol: float = 0.5
    
    # Trading limits
    max_trades_per_hour: int = 6
    max_concurrent_positions: int = 2
    max_wallet_risk_percent: float = 2.0
    
    # Technical settings
    entry_timeout_ms: int = 1000
    max_slippage_percent: float = 2.0
    rpc_timeout_seconds: int = 5
    
    # RPC endpoints
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    backup_rpc_urls: List[str] = None
    
    def __post_init__(self):
        if self.backup_rpc_urls is None:
            self.backup_rpc_urls = [
                "https://solana-api.projectserum.com",
                "https://rpc.ankr.com/solana"
            ]


@dataclass  
class SecurityConfig:
    """Simplified security configuration"""
    
    # Basic security settings
    enable_encryption: bool = True
    enable_stealth_mode: bool = True
    enable_kill_switch: bool = True
    
    # Wallet management
    wallet_rotation_enabled: bool = True
    wallet_rotation_interval_hours: int = 12
    max_wallet_exposure_sol: float = 5.0
    
    # Rate limiting
    max_api_calls_per_minute: int = 30
    max_transactions_per_minute: int = 5
    
    # Emergency stops
    emergency_stop_enabled: bool = True
    max_drawdown_stop_percent: float = 10.0
    
    # Logging
    log_sensitive_data: bool = False
    mask_private_keys_in_logs: bool = True


class SimpleConfigManager:
    """Simplified configuration manager"""
    
    def __init__(self):
        self.logger = logging.getLogger("SimpleConfig")
        self._trading_config = None
        self._security_config = None
        self._loaded = False
    
    def load_configuration(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        
        if self._loaded:
            return self.get_all_config()
        
        try:
            # Load trading configuration
            self._trading_config = self._load_trading_config()
            
            # Load security configuration  
            self._security_config = self._load_security_config()
            
            self._loaded = True
            self.logger.info("✅ Configuration loaded successfully")
            
            return self.get_all_config()
            
        except Exception as e:
            self.logger.error(f"Configuration loading failed: {e}")
            # Use defaults
            self._trading_config = TradingConfig()
            self._security_config = SecurityConfig()
            self._loaded = True
            return self.get_all_config()
    
    def _load_trading_config(self) -> TradingConfig:
        """Load trading config from environment variables"""
        
        config = TradingConfig()
        
        # Override with environment variables if present
        config.trade_amount_sol = float(os.getenv("TRADE_AMOUNT_SOL", config.trade_amount_sol))
        config.base_profit_target_percent = float(os.getenv("PROFIT_TARGET_PERCENT", config.base_profit_target_percent))
        config.base_stop_loss_percent = float(os.getenv("STOP_LOSS_PERCENT", config.base_stop_loss_percent))
        config.max_trades_per_hour = int(os.getenv("MAX_TRADES_PER_HOUR", config.max_trades_per_hour))
        config.max_concurrent_positions = int(os.getenv("MAX_CONCURRENT_POSITIONS", config.max_concurrent_positions))
        config.rpc_url = os.getenv("RPC_URL", config.rpc_url)
        
        return config
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security config from environment variables"""
        
        config = SecurityConfig()
        
        # Override with environment variables if present
        config.enable_stealth_mode = os.getenv("ENABLE_STEALTH_MODE", "true").lower() == "true"
        config.wallet_rotation_enabled = os.getenv("WALLET_ROTATION_ENABLED", "true").lower() == "true"
        config.wallet_rotation_interval_hours = int(os.getenv("WALLET_ROTATION_HOURS", config.wallet_rotation_interval_hours))
        config.max_drawdown_stop_percent = float(os.getenv("MAX_DRAWDOWN_PERCENT", config.max_drawdown_stop_percent))
        
        return config
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration"""
        if not self._loaded:
            self.load_configuration()
        return self._trading_config
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        if not self._loaded:
            self.load_configuration()
        return self._security_config
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            'trading': self._trading_config,
            'security': self._security_config
        }
    
    def get_config(self, section: str = None) -> Any:
        """Get specific configuration section"""
        if not self._loaded:
            self.load_configuration()
            
        if section == "trading":
            return self._trading_config
        elif section == "security":
            return self._security_config
        else:
            return self.get_all_config()


# Global configuration manager instance
_config_manager = SimpleConfigManager()


def get_trading_config() -> TradingConfig:
    """Get the current trading configuration"""
    return _config_manager.get_trading_config()


def get_security_config() -> SecurityConfig:
    """Get the current security configuration"""
    return _config_manager.get_security_config()


def get_config_manager() -> SimpleConfigManager:
    """Get the configuration manager instance"""
    return _config_manager


def reload_configuration() -> bool:
    """Reload configuration from environment"""
    try:
        _config_manager._loaded = False
        _config_manager.load_configuration()
        return True
    except Exception:
        return False


def validate_solana_address(address: str) -> bool:
    """Basic Solana address validation"""
    if not address or not isinstance(address, str):
        return False
    
    # Basic length check for Solana addresses (44 characters in base58)
    if len(address) < 32 or len(address) > 44:
        return False
    
    # Check for valid base58 characters
    valid_chars = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
    if not all(c in valid_chars for c in address):
        return False
    
    return True


def mask_sensitive_value(value: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """Mask sensitive values for logging"""
    if not value or len(value) <= visible_chars * 2:
        return mask_char * 8
    
    start = value[:visible_chars]
    end = value[-visible_chars:]
    middle = mask_char * (len(value) - visible_chars * 2)
    
    return f"{start}{middle}{end}" 