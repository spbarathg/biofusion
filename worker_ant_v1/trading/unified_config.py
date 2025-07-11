"""
UNIFIED CONFIGURATION SYSTEM
============================

Single source of truth for all trading bot configuration.
Consolidates scattered config files into one coherent system.
"""

import os
import secrets
import hashlib
import base64
import json
import gc
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pathlib import Path
import logging

# Import validation libraries
try:
    from pydantic import BaseModel, Field, validator, ValidationError
    from pydantic_settings import BaseSettings
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base58
    VALIDATION_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        f"Critical dependencies missing: {e}. "
        "Install with: pip install pydantic cryptography base58"
    ) from e

class TradingMode(Enum):
    LIVE = "live"
    SIMULATION = "simulation"
    PAPER = "paper"
    TESTING = "testing"

class SecurityLevel(Enum):
    HIGH = "high"
    MAXIMUM = "maximum"

class TradingStrategy(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    SCALPING = "scalping"
    MOMENTUM = "momentum"

# === CORE TRADING CONFIGURATION ===

@dataclass
class TradingConfig:
    """Unified trading configuration with validation"""
    
    # Capital Management
    initial_capital_sol: float = 10.0
    max_position_size_percent: float = 20.0
    min_position_size_sol: float = 0.1
    emergency_reserve_percent: float = 10.0
    
    # Risk Management
    stop_loss_percent: float = 5.0
    take_profit_percent: float = 15.0
    max_daily_loss_percent: float = 10.0
    max_concurrent_positions: int = 3
    
    # Entry/Exit Parameters
    entry_timeout_seconds: int = 30
    exit_timeout_seconds: int = 300
    max_slippage_percent: float = 2.0
    min_liquidity_sol: float = 100.0
    
    # Trading Frequency
    max_trades_per_hour: int = 10
    min_time_between_trades_seconds: int = 60
    
    # Strategy Specific Settings
    strategy: TradingStrategy = TradingStrategy.CONSERVATIVE
    enable_scalping: bool = False
    enable_momentum_trading: bool = True
    enable_social_signals: bool = False  # DISABLED - Pure on-chain only
    
    # AI/ML Settings
    enable_ml_predictions: bool = True
    ml_confidence_threshold: float = 0.65
    sentiment_weight: float = 0.3
    technical_weight: float = 0.4
    fundamental_weight: float = 0.3

@dataclass
class SecurityConfig:
    """Unified security configuration"""
    
    security_level: SecurityLevel = SecurityLevel.HIGH
    enable_encryption: bool = True
    
    # Wallet Security
    wallet_rotation_enabled: bool = True
    wallet_rotation_interval_hours: int = 12
    max_wallet_exposure_sol: float = 5.0
    
    # API Security
    max_api_calls_per_minute: int = 30
    max_transactions_per_minute: int = 5
    api_key_rotation_days: int = 7
    
    # Kill Switch
    enable_kill_switch: bool = True
    max_drawdown_stop_percent: float = 15.0
    emergency_stop_enabled: bool = True
    
    # Monitoring
    enable_anomaly_detection: bool = True
    log_sensitive_data: bool = False
    mask_private_keys_in_logs: bool = True

@dataclass
class NetworkConfig:
    """Solana network and RPC configuration"""
    
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    backup_rpc_urls: List[str] = field(default_factory=lambda: [
        "https://solana-api.projectserum.com",
        "https://rpc.ankr.com/solana"
    ])
    rpc_timeout_seconds: int = 5
    max_rpc_retries: int = 3
    use_private_rpc: bool = True
    max_tx_fee_sol: float = 0.001

@dataclass 
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_grafana: bool = True
    grafana_port: int = 3000
    
    # Alerting
    enable_email_alerts: bool = False
    # SOCIAL MEDIA ALERTS REMOVED - PURE ON-CHAIN + AI ONLY
    alert_on_large_loss: bool = True
    alert_threshold_percent: float = 5.0

@dataclass
class APIConfig:
    """API keys and external service configuration"""
    
    # Core Trading APIs
    helius_api_key: Optional[str] = None
    solana_tracker_api_key: Optional[str] = None
    
    # Optional Premium APIs
    quicknode_rpc_url: Optional[str] = None
    dexscreener_api_key: Optional[str] = None
    birdeye_api_key: Optional[str] = None
    coingecko_api_key: Optional[str] = None
    
    # Twitter sentiment is sourced via Discord server feed
    discord_bot_token: Optional[str] = None
    discord_server_id: Optional[str] = None
    discord_channel_id: Optional[str] = None
    
    # Alert APIs - Local only (no external social platforms)
    local_alerts_enabled: bool = True
    console_alerts_enabled: bool = True
    
    def validate_required_apis(self) -> List[str]:
        """Validate required API keys and return missing ones"""
        missing = []
        
        # Check core APIs (on-chain only)
        if not self.helius_api_key:
            missing.append("HELIUS_API_KEY")
        if not self.solana_tracker_api_key:
            missing.append("SOLANA_TRACKER_API_KEY")
        # Check Discord for Twitter sentiment
        if not self.discord_bot_token:
            missing.append("DISCORD_BOT_TOKEN")
        if not self.discord_server_id:
            missing.append("DISCORD_SERVER_ID")
        if not self.discord_channel_id:
            missing.append("DISCORD_CHANNEL_ID")
        return missing
    
    def get_masked_config(self) -> Dict[str, str]:
        """Get API config with masked keys for logging"""
        return {
            'helius_api_key': mask_sensitive_value(self.helius_api_key or ""),
            'solana_tracker_api_key': mask_sensitive_value(self.solana_tracker_api_key or ""),
            'quicknode_rpc_url': mask_sensitive_value(self.quicknode_rpc_url or ""),
            'discord_bot_token': mask_sensitive_value(self.discord_bot_token or ""),
            'discord_server_id': mask_sensitive_value(self.discord_server_id or ""),
            'discord_channel_id': mask_sensitive_value(self.discord_channel_id or ""),
            'local_alerts_enabled': str(self.local_alerts_enabled),
            'console_alerts_enabled': str(self.console_alerts_enabled),
        }

# === UNIFIED CONFIGURATION MANAGER ===

class UnifiedConfigManager:
    """Single configuration manager for the entire system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger("UnifiedConfig")
        self.config_file = config_file or ".env.production"
        self._config_cache = {}
        self._last_modified = None
        
        # Configuration instances
        self.trading = TradingConfig()
        self.security = SecurityConfig()
        self.network = NetworkConfig()
        self.monitoring = MonitoringConfig()
        self.api = APIConfig()
        
        self._load_from_environment()
        self._validate_configuration()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        # Trading Config
        if os.getenv("INITIAL_CAPITAL_SOL"):
            self.trading.initial_capital_sol = float(os.getenv("INITIAL_CAPITAL_SOL"))
        if os.getenv("MAX_POSITION_SIZE_PERCENT"):
            self.trading.max_position_size_percent = float(os.getenv("MAX_POSITION_SIZE_PERCENT"))
        if os.getenv("STOP_LOSS_PERCENT"):
            self.trading.stop_loss_percent = float(os.getenv("STOP_LOSS_PERCENT"))
        if os.getenv("TAKE_PROFIT_PERCENT"):
            self.trading.take_profit_percent = float(os.getenv("TAKE_PROFIT_PERCENT"))
        if os.getenv("TRADING_STRATEGY"):
            self.trading.strategy = TradingStrategy(os.getenv("TRADING_STRATEGY"))
        
        # Security Config  
        if os.getenv("SECURITY_LEVEL"):
            self.security.security_level = SecurityLevel(os.getenv("SECURITY_LEVEL"))
        if os.getenv("WALLET_ROTATION_ENABLED"):
            self.security.wallet_rotation_enabled = os.getenv("WALLET_ROTATION_ENABLED").lower() == "true"
        if os.getenv("MAX_DRAWDOWN_STOP_PERCENT"):
            self.security.max_drawdown_stop_percent = float(os.getenv("MAX_DRAWDOWN_STOP_PERCENT"))
        
        # Network Config
        if os.getenv("RPC_URL"):
            self.network.rpc_url = os.getenv("RPC_URL")
        if os.getenv("RPC_TIMEOUT_SECONDS"):
            self.network.rpc_timeout_seconds = int(os.getenv("RPC_TIMEOUT_SECONDS"))
        
        # Monitoring Config
        if os.getenv("ENABLE_PROMETHEUS"):
            self.monitoring.enable_prometheus = os.getenv("ENABLE_PROMETHEUS").lower() == "true"
        if os.getenv("PROMETHEUS_PORT"):
            self.monitoring.prometheus_port = int(os.getenv("PROMETHEUS_PORT"))
        
        # API Config
        if os.getenv("HELIUS_API_KEY"):
            self.api.helius_api_key = os.getenv("HELIUS_API_KEY")
        if os.getenv("SOLANA_TRACKER_API_KEY"):
            self.api.solana_tracker_api_key = os.getenv("SOLANA_TRACKER_API_KEY")
        if os.getenv("QUICKNODE_RPC_URL"):
            self.api.quicknode_rpc_url = os.getenv("QUICKNODE_RPC_URL")
        if os.getenv("DEXSCREENER_API_KEY"):
            self.api.dexscreener_api_key = os.getenv("DEXSCREENER_API_KEY")
        if os.getenv("BIRDEYE_API_KEY"):
            self.api.birdeye_api_key = os.getenv("BIRDEYE_API_KEY")
        if os.getenv("COINGECKO_API_KEY"):
            self.api.coingecko_api_key = os.getenv("COINGECKO_API_KEY")
        # Twitter API - ENABLED for sentiment analysis
        if os.getenv("TWITTER_BEARER_TOKEN"):
            self.api.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        if os.getenv("TWITTER_API_KEY"):
            self.api.twitter_api_key = os.getenv("TWITTER_API_KEY")
        if os.getenv("TWITTER_API_SECRET"):
            self.api.twitter_api_secret = os.getenv("TWITTER_API_SECRET")
        if os.getenv("TWITTER_ACCESS_TOKEN"):
            self.api.twitter_access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        if os.getenv("TWITTER_ACCESS_TOKEN_SECRET"):
            self.api.twitter_access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        
        # Discord, Telegram, Reddit - DISABLED (Twitter only mode)
        if os.getenv("DISCORD_BOT_TOKEN"):
            self.api.discord_bot_token = os.getenv("DISCORD_BOT_TOKEN")
        if os.getenv("DISCORD_SERVER_ID"):
            self.api.discord_server_id = os.getenv("DISCORD_SERVER_ID")
        if os.getenv("DISCORD_CHANNEL_ID"):
            self.api.discord_channel_id = os.getenv("DISCORD_CHANNEL_ID")
        # if os.getenv("TELEGRAM_BOT_TOKEN"):
        #     self.api.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        # if os.getenv("TELEGRAM_CHAT_ID"):
        #     self.api.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        # if os.getenv("REDDIT_CLIENT_ID"):
        #     self.api.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        # if os.getenv("REDDIT_CLIENT_SECRET"):
        #     self.api.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    
    def _validate_configuration(self):
        """Validate all configuration parameters"""
        
        # Trading validation
        assert 0.01 <= self.trading.initial_capital_sol <= 1000, "Initial capital must be between 0.01 and 1000 SOL"
        assert 1 <= self.trading.max_position_size_percent <= 50, "Max position size must be between 1% and 50%"
        assert 1 <= self.trading.stop_loss_percent <= 20, "Stop loss must be between 1% and 20%"
        assert 1 <= self.trading.take_profit_percent <= 100, "Take profit must be between 1% and 100%"
        
        # Security validation
        assert self.security.max_drawdown_stop_percent >= self.trading.max_daily_loss_percent, \
            "Drawdown stop must be >= daily loss limit"
        
        # Network validation
        assert self.network.rpc_timeout_seconds >= 1, "RPC timeout must be at least 1 second"
        assert len(self.network.backup_rpc_urls) >= 1, "Must have at least one backup RPC"
        
        # API validation
        missing_apis = self.api.validate_required_apis()
        if missing_apis:
            self.logger.warning(f"⚠️  Missing API keys: {', '.join(missing_apis)}")
            self.logger.warning("Bot will run with limited functionality. Add API keys to .env.production")
        else:
            self.logger.info("✅ All required API keys configured")
        
        self.logger.info("✅ Configuration validation passed")
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration"""
        return self.trading
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration"""
        return self.security
    
    def get_network_config(self) -> NetworkConfig:
        """Get network configuration"""
        return self.network
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return self.monitoring
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration"""
        return self.api
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            'trading': self.trading.__dict__,
            'security': self.security.__dict__,
            'network': self.network.__dict__,
            'monitoring': self.monitoring.__dict__,
            'api': self.api.get_masked_config()  # Use masked version for security
        }
    
    def reload_if_changed(self) -> bool:
        """Reload configuration if file has changed"""
        try:
            if self.config_file and os.path.exists(self.config_file):
                current_modified = os.path.getmtime(self.config_file)
                if self._last_modified is None or current_modified > self._last_modified:
                    self._load_from_environment()
                    self._validate_configuration()
                    self._last_modified = current_modified
                    self.logger.info("🔄 Configuration reloaded")
                    return True
            return False
        except Exception as e:
            self.logger.error(f"❌ Failed to reload configuration: {e}")
            return False

# === GLOBAL CONFIGURATION INSTANCE ===

_config_manager = None

def get_config_manager() -> UnifiedConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = UnifiedConfigManager()
    return _config_manager

def get_trading_config() -> TradingConfig:
    """Get trading configuration"""
    return get_config_manager().get_trading_config()

def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return get_config_manager().get_security_config()

def get_network_config() -> NetworkConfig:
    """Get network configuration"""
    return get_config_manager().get_network_config()

def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return get_config_manager().get_monitoring_config()

def get_api_config() -> APIConfig:
    """Get API configuration"""
    return get_config_manager().get_api_config()

def reload_configuration() -> bool:
    """Reload configuration from file"""
    return get_config_manager().reload_if_changed()

# === UTILITY FUNCTIONS ===

def mask_sensitive_value(value: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """Mask sensitive values for logging"""
    if not value or len(value) <= visible_chars * 2:
        return mask_char * 8
    return f"{value[:visible_chars]}{mask_char * (len(value) - visible_chars * 2)}{value[-visible_chars:]}"

def validate_solana_address(address: str) -> bool:
    """Validate Solana address format"""
    try:
        if not address or not isinstance(address, str):
            return False
        decoded = base58.b58decode(address)
        return len(decoded) == 32
    except Exception:
        return False

def get_secure_random_string(length: int = 32) -> str:
    """Generate cryptographically secure random string"""
    return secrets.token_urlsafe(length)[:length]

@dataclass
class BattleConfig:
    # Core execution parameters
    max_slippage: float = 0.01  # 1% max slippage
    execution_timeout: int = 2  # seconds
    gas_multiplier: float = 1.2  # 20% gas boost for faster execution
    
    # Safety thresholds
    min_liquidity_usd: float = 50000  # $50k minimum liquidity
    max_wallet_concentration: float = 0.15  # 15% max wallet concentration
    min_unique_holders: int = 100
    min_pool_age_hours: int = 1
    
    # Pattern detection
    pattern_confidence_threshold: float = 0.85
    lookback_blocks: int = 100
    min_pattern_occurrences: int = 3
    
    # Position management
    max_position_size: float = 0.1  # 10% of available capital
    take_profit_threshold: float = 0.3  # 30% profit target
    stop_loss_threshold: float = 0.05  # 5% stop loss
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'BattleConfig':
        """Load configuration from file or use defaults"""
        if config_path and config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
                return cls(**config_dict)
        return cls()
        
    def save(self, config_path: Path):
        """Save current configuration"""
        config_dict = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    def validate(self) -> bool:
        """Validate configuration parameters"""
        checks = [
            0 < self.max_slippage < 0.1,  # Reasonable slippage range
            0 < self.execution_timeout < 10,  # Reasonable timeout range
            self.gas_multiplier > 1,  # Must boost gas
            self.min_liquidity_usd > 10000,  # Minimum safety threshold
            0 < self.max_wallet_concentration < 0.5,  # Reasonable concentration
            self.pattern_confidence_threshold > 0.8,  # High confidence required
            0 < self.max_position_size < 0.5,  # Reasonable position size
            self.take_profit_threshold > self.stop_loss_threshold  # Logical profit/loss
        ]
        return all(checks)
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        } 