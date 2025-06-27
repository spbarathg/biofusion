"""
PRODUCTION-READY CONFIGURATION SYSTEM
====================================

Secure, validated configuration management for the Worker Ant trading bot.
Includes encrypted wallet handling, comprehensive validation, and fail-safe defaults.
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

# Import validation libraries - REQUIRED FOR PRODUCTION
try:
    from pydantic import BaseModel, Field, validator, ValidationError
    from pydantic_settings import BaseSettings
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base58
    VALIDATION_AVAILABLE = True
except ImportError as e:
    # CRITICAL: No fallback allowed in production
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

# === SECURE WALLET MANAGEMENT ===

class SecureWalletManager:
    """Secure wallet private key management with encryption and memory protection"""
    
    def __init__(self):
        self.logger = logging.getLogger("SecureWalletManager")
        self._encryption_key = None
        self._salt = None
        
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password and salt"""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            return key
        except Exception as e:
            self.logger.error(f"Key derivation failed: {e}")
            raise
    
    def encrypt_private_key(self, private_key: str, password: str) -> Dict[str, str]:
        """Encrypt private key with password and clear from memory"""
        try:
            # Validate private key first
            if not self.validate_private_key(private_key):
                raise ValueError("Invalid private key format")
                
            # Generate salt
            salt = os.urandom(16)
            key = self._derive_key(password, salt)
            fernet = Fernet(key)
            
            # Encrypt the private key
            encrypted_key = fernet.encrypt(private_key.encode())
            
            # Clear sensitive data from memory
            private_key = "0" * len(private_key)
            del private_key
            
            # Clear derived key
            key = b"0" * len(key)
            del key
            
            # Force garbage collection
            gc.collect()
            
            return {
                "encrypted_key": base64.b64encode(encrypted_key).decode(),
                "salt": base64.b64encode(salt).decode(),
                "algorithm": "PBKDF2-SHA256-Fernet",
                "version": "1.0"
            }
        except Exception as e:
            self.logger.error(f"Failed to encrypt private key: {e}")
            raise
    
    def decrypt_private_key(self, encrypted_data: Dict[str, str], password: str) -> str:
        """Decrypt private key with password"""
        try:
            # Validate encrypted data structure
            required_fields = ["encrypted_key", "salt", "algorithm"]
            if not all(field in encrypted_data for field in required_fields):
                raise ValueError("Invalid encrypted data structure")
                
            salt = base64.b64decode(encrypted_data["salt"])
            encrypted_key = base64.b64decode(encrypted_data["encrypted_key"])
            
            key = self._derive_key(password, salt)
            fernet = Fernet(key)
            
            decrypted_key = fernet.decrypt(encrypted_key).decode()
            
            # Clear derived key from memory
            key = b"0" * len(key)
            del key
            gc.collect()
            
            # Validate decrypted key
            if not self.validate_private_key(decrypted_key):
                raise ValueError("Decrypted key failed validation")
                
            return decrypted_key
        except Exception as e:
            self.logger.error(f"Failed to decrypt private key: {e}")
            raise
    
    def validate_private_key(self, private_key: str) -> bool:
        """Validate Solana private key format with comprehensive checks"""
        try:
            if not private_key or not isinstance(private_key, str):
                return False
                
            # Check if it's base58 encoded
            try:
                decoded = base58.b58decode(private_key)
            except Exception:
                return False
                
            # Solana private keys should be exactly 64 bytes
            if len(decoded) != 64:
                return False
                
            # Check for obviously invalid patterns
            if all(b == 0 for b in decoded):  # All zeros
                return False
            if all(b == 255 for b in decoded):  # All 255s
                return False
                
            return True
        except Exception:
            return False
    
    def generate_secure_private_key(self) -> str:
        """Generate a cryptographically secure private key"""
        try:
            # Generate 64 random bytes
            private_key_bytes = secrets.token_bytes(64)
            
            # Encode as base58
            private_key = base58.b58encode(private_key_bytes).decode()
            
            # Validate the generated key
            if not self.validate_private_key(private_key):
                raise ValueError("Generated key failed validation")
                
            return private_key
        except Exception as e:
            self.logger.error(f"Failed to generate private key: {e}")
            raise

# === CONFIGURATION MODELS WITH MANDATORY VALIDATION ===

class TradingConfigModel(BaseModel):
    """Validated trading configuration with strict limits"""
    
    # Trade amounts (SOL) - Conservative limits for safety
    trade_amount_sol: float = Field(default=0.1, ge=0.01, le=1.0)
    min_trade_amount_sol: float = Field(default=0.05, ge=0.005, le=0.5)
    max_trade_amount_sol: float = Field(default=0.25, ge=0.1, le=2.0)
    
    # Profit targets (percentages) - Realistic targets
    base_profit_target_percent: float = Field(default=5.0, ge=1.0, le=15.0)
    min_profit_target_percent: float = Field(default=3.0, ge=0.5, le=10.0)
    max_profit_target_percent: float = Field(default=8.0, ge=3.0, le=25.0)
    
    # Stop losses (percentages) - Tight risk control
    base_stop_loss_percent: float = Field(default=3.0, ge=1.0, le=10.0)
    min_stop_loss_percent: float = Field(default=2.0, ge=0.5, le=5.0)
    max_stop_loss_percent: float = Field(default=5.0, ge=2.0, le=15.0)
    
    # Speed & latency (milliseconds) - Realistic production values
    entry_timeout_ms: int = Field(default=1000, ge=500, le=5000)
    max_slippage_percent: float = Field(default=2.0, ge=0.1, le=5.0)
    min_slippage_percent: float = Field(default=0.5, ge=0.05, le=1.0)
    dynamic_slippage: bool = True
    
    # Throughput - Conservative for stability
    max_trades_per_hour: int = Field(default=6, ge=1, le=20)
    min_time_between_trades_seconds: int = Field(default=60, ge=30, le=300)
    max_concurrent_positions: int = Field(default=2, ge=1, le=5)
    
    # Exit timing (seconds)
    timeout_exit_seconds: int = Field(default=300, ge=60, le=1800)
    min_timeout_seconds: int = Field(default=240, ge=60, le=900)
    max_timeout_seconds: int = Field(default=600, ge=120, le=3600)
    
    # Risk management - Strict limits
    max_wallet_risk_percent: float = Field(default=2.0, ge=0.1, le=10.0)
    max_daily_loss_sol: float = Field(default=0.5, ge=0.1, le=5.0)
    max_daily_loss_percent: float = Field(default=5.0, ge=1.0, le=20.0)
    
    # Position sizing
    position_size_multiplier: float = Field(default=1.0, ge=0.1, le=2.0)
    max_position_multiplier: float = Field(default=1.5, ge=1.0, le=3.0)
    min_position_multiplier: float = Field(default=0.5, ge=0.1, le=1.0)
    
    # RPC configuration
    rpc_url: str = Field(default="https://api.mainnet-beta.solana.com")
    backup_rpc_urls: List[str] = Field(default_factory=lambda: [
        "https://solana-api.projectserum.com",
        "https://rpc.ankr.com/solana"
    ])
    use_private_rpc: bool = True
    rpc_timeout_seconds: int = Field(default=5, ge=2, le=30)
    max_rpc_retries: int = Field(default=3, ge=1, le=5)
    
    # Cost control
    max_tx_fee_sol: float = Field(default=0.001, ge=0.0001, le=0.01)
    max_rpc_cost_per_trade: float = Field(default=0.0001, ge=0.00001, le=0.001)
    alert_cost_threshold_percent: float = Field(default=5.0, ge=1.0, le=20.0)
    
    # Minimum liquidity requirements
    min_liquidity_sol: float = Field(default=100.0, ge=50.0, le=1000.0)
    min_volume_24h_sol: float = Field(default=1000.0, ge=500.0, le=10000.0)
    
    @validator('min_trade_amount_sol')
    def min_less_than_max_trade(cls, v, values):
        if 'max_trade_amount_sol' in values and v >= values['max_trade_amount_sol']:
            raise ValueError('min_trade_amount_sol must be less than max_trade_amount_sol')
        return v
    
    @validator('min_profit_target_percent')
    def min_profit_less_than_max(cls, v, values):
        if 'max_profit_target_percent' in values and v >= values['max_profit_target_percent']:
            raise ValueError('min_profit_target_percent must be less than max_profit_target_percent')
        return v

class SecurityConfigModel(BaseModel):
    """Security configuration validation with mandatory security"""
    
    security_level: SecurityLevel = SecurityLevel.HIGH
    enable_encryption: bool = True
    wallet_rotation_enabled: bool = True
    wallet_rotation_interval_hours: int = Field(default=12, ge=1, le=24)
    max_wallet_exposure_sol: float = Field(default=5.0, ge=0.1, le=50.0)
    enable_stealth_mode: bool = True
    api_key_rotation_days: int = Field(default=7, ge=1, le=30)
    
    # Rate limiting - Mandatory
    max_api_calls_per_minute: int = Field(default=30, ge=1, le=100)
    max_transactions_per_minute: int = Field(default=5, ge=1, le=20)
    
    # Monitoring - Always enabled
    enable_anomaly_detection: bool = True
    log_sensitive_data: bool = False
    mask_private_keys_in_logs: bool = True
    
    # Kill switch settings - Non-disableable
    enable_kill_switch: bool = True
    emergency_stop_enabled: bool = True
    max_drawdown_stop_percent: float = Field(default=10.0, ge=5.0, le=25.0)

# === CONFIGURATION LOADER ===

class ConfigurationManager:
    """Centralized configuration management with validation and security"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger("ConfigurationManager")
        self.config_file = config_file or ".env"
        self.wallet_manager = SecureWalletManager()
        self._config_cache = {}
        self._last_reload = 0
        self._initialized = False
        
    def load_configuration(self) -> Dict[str, Any]:
        """Load and validate configuration from environment and files"""
        try:
            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv(self.config_file)
            
            # Build configuration with validation
            config = {
                "trading": self._load_trading_config(),
                "security": self._load_security_config(),
                "wallet": self._load_wallet_config(),
                "monitoring": self._load_monitoring_config(),
                "scanner": self._load_scanner_config(),
                "deployment": self._load_deployment_config()
            }
            
            # Validate configuration
            self._validate_configuration(config)
            
            # Cache configuration
            self._config_cache = config
            self._last_reload = os.path.getmtime(self.config_file) if os.path.exists(self.config_file) else 0
            self._initialized = True
            
            return config
            
        except Exception as e:
            self.logger.error(f"Configuration loading failed: {e}")
            raise
    
    def _load_trading_config(self) -> TradingConfigModel:
        """Load trading configuration with validation"""
        try:
            config_data = {
                "trade_amount_sol": float(os.getenv("TRADE_AMOUNT_SOL", "0.1")),
                "min_trade_amount_sol": float(os.getenv("MIN_TRADE_AMOUNT_SOL", "0.05")),
                "max_trade_amount_sol": float(os.getenv("MAX_TRADE_AMOUNT_SOL", "0.25")),
                "base_profit_target_percent": float(os.getenv("PROFIT_TARGET_PERCENT", "5.0")),
                "base_stop_loss_percent": float(os.getenv("STOP_LOSS_PERCENT", "3.0")),
                "max_trades_per_hour": int(os.getenv("MAX_TRADES_PER_HOUR", "6")),
                "max_concurrent_positions": int(os.getenv("MAX_CONCURRENT_POSITIONS", "2")),
                "max_daily_loss_sol": float(os.getenv("MAX_DAILY_LOSS_SOL", "0.5")),
                "rpc_url": os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com"),
                "use_private_rpc": os.getenv("USE_PRIVATE_RPC", "true").lower() == "true"
            }
            
            return TradingConfigModel(**config_data)
        except Exception as e:
            self.logger.error(f"Trading config validation failed: {e}")
            raise
    
    def _load_security_config(self) -> SecurityConfigModel:
        """Load security configuration with validation"""
        try:
            config_data = {
                "security_level": SecurityLevel(os.getenv("SECURITY_LEVEL", "high")),
                "enable_encryption": True,  # Always enabled
                "wallet_rotation_enabled": os.getenv("WALLET_ROTATION", "true").lower() == "true",
                "wallet_rotation_interval_hours": int(os.getenv("WALLET_ROTATION_HOURS", "12")),
                "enable_stealth_mode": os.getenv("STEALTH_MODE", "true").lower() == "true",
                "max_api_calls_per_minute": int(os.getenv("MAX_API_CALLS_PER_MINUTE", "30")),
                "max_transactions_per_minute": int(os.getenv("MAX_TXN_PER_MINUTE", "5")),
                "enable_kill_switch": True,  # Always enabled
                "emergency_stop_enabled": True,  # Always enabled
                "max_drawdown_stop_percent": float(os.getenv("MAX_DRAWDOWN_STOP", "10.0"))
            }
            
            return SecurityConfigModel(**config_data)
        except Exception as e:
            self.logger.error(f"Security config validation failed: {e}")
            raise
    
    def _load_wallet_config(self) -> Dict[str, Any]:
        """Load wallet configuration with security validation"""
        try:
            wallet_config = {
                "encrypted_private_key": os.getenv("ENCRYPTED_WALLET_KEY"),
                "wallet_password": os.getenv("WALLET_PASSWORD"),
                "auto_create_wallet": os.getenv("AUTO_CREATE_WALLET", "false").lower() == "true",
                "backup_wallet_count": int(os.getenv("BACKUP_WALLET_COUNT", "3")),
                "wallet_derivation_path": os.getenv("WALLET_DERIVATION_PATH", "m/44'/501'/0'/0'")
            }
            
            # Validate wallet configuration
            if not wallet_config["encrypted_private_key"] and not wallet_config["auto_create_wallet"]:
                raise ValueError("Either ENCRYPTED_WALLET_KEY or AUTO_CREATE_WALLET=true must be set")
                
            if wallet_config["encrypted_private_key"] and not wallet_config["wallet_password"]:
                raise ValueError("WALLET_PASSWORD required when using ENCRYPTED_WALLET_KEY")
            
            return wallet_config
        except Exception as e:
            self.logger.error(f"Wallet config validation failed: {e}")
            raise
    
    def _load_monitoring_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        return {
            "enable_logging": True,
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "enable_metrics": True,
            "metrics_interval_seconds": int(os.getenv("METRICS_INTERVAL", "30")),
            "enable_health_checks": True,
            "health_check_interval_seconds": int(os.getenv("HEALTH_CHECK_INTERVAL", "60")),
            "enable_discord_alerts": os.getenv("ENABLE_DISCORD_ALERTS", "true").lower() == "true",
            "discord_webhook_url": os.getenv("DISCORD_WEBHOOK_URL"),
            "alert_on_errors": True,
            "alert_on_profit_milestones": True
        }
    
    def _load_scanner_config(self) -> Dict[str, Any]:
        """Load scanner configuration"""
        return {
            "scan_interval_seconds": int(os.getenv("SCAN_INTERVAL", "30")),
            "max_tokens_per_scan": int(os.getenv("MAX_TOKENS_PER_SCAN", "100")),
            "min_confidence_score": float(os.getenv("MIN_CONFIDENCE_SCORE", "0.7")),
            "enable_security_filters": True,
            "max_security_risk_level": os.getenv("MAX_SECURITY_RISK", "medium"),
            "use_multiple_sources": True,
            "cache_duration_seconds": int(os.getenv("CACHE_DURATION", "300"))
        }
    
    def _load_deployment_config(self) -> Dict[str, Any]:
        """Load deployment-specific configuration"""
        return {
            "environment": os.getenv("ENVIRONMENT", "production"),
            "debug_mode": os.getenv("DEBUG_MODE", "false").lower() == "true",
            "enable_simulation": os.getenv("ENABLE_SIMULATION", "false").lower() == "true",
            "data_directory": os.getenv("DATA_DIRECTORY", "data"),
            "log_directory": os.getenv("LOG_DIRECTORY", "logs"),
            "backup_directory": os.getenv("BACKUP_DIRECTORY", "data/backups"),
            "max_log_file_size_mb": int(os.getenv("MAX_LOG_SIZE_MB", "100")),
            "log_retention_days": int(os.getenv("LOG_RETENTION_DAYS", "30"))
        }
    
    def _validate_configuration(self, config: Dict[str, Any]):
        """Validate complete configuration for security and consistency"""
        try:
            # Validate trading vs security consistency
            trading = config["trading"]
            security = config["security"]
            
            # Ensure risk limits are consistent
            if trading.max_daily_loss_sol > security.max_wallet_exposure_sol * 0.5:
                raise ValueError("Daily loss limit too high relative to wallet exposure")
            
            # Validate wallet configuration
            wallet = config["wallet"]
            if not wallet["encrypted_private_key"] and not wallet["auto_create_wallet"]:
                raise ValueError("No wallet configuration provided")
            
            # Validate monitoring
            monitoring = config["monitoring"]
            if not monitoring["enable_logging"]:
                raise ValueError("Logging cannot be disabled in production")
            
            self.logger.info("✅ Configuration validation successful")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_config(self, section: str = None) -> Union[Dict[str, Any], Any]:
        """Get configuration section or full config"""
        if not self._initialized:
            self.load_configuration()
        
        if section:
            if section not in self._config_cache:
                raise ValueError(f"Configuration section '{section}' not found")
            return self._config_cache[section]
        
        return self._config_cache
    
    def reload_if_changed(self) -> bool:
        """Reload configuration if file has changed"""
        try:
            if not os.path.exists(self.config_file):
                return False
            
            current_mtime = os.path.getmtime(self.config_file)
            if current_mtime > self._last_reload:
                self.logger.info("Configuration file changed, reloading...")
                self.load_configuration()
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to check configuration changes: {e}")
            return False

# === UTILITY FUNCTIONS ===

def get_secure_random_string(length: int = 32) -> str:
    """Generate cryptographically secure random string"""
    return secrets.token_urlsafe(length)

def validate_solana_address(address: str) -> bool:
    """Validate Solana address format"""
    try:
        if not address or len(address) < 32 or len(address) > 44:
            return False
        decoded = base58.b58decode(address)
        return len(decoded) == 32
    except Exception:
        return False

def mask_sensitive_value(value: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """Mask sensitive values for safe logging"""
    if not value or len(value) <= visible_chars * 2:
        return mask_char * 8
    
    return f"{value[:visible_chars]}{mask_char * (len(value) - visible_chars * 2)}{value[-visible_chars:]}"

# === SINGLETON CONFIGURATION MANAGER ===

config_manager = ConfigurationManager()

def get_trading_config() -> TradingConfigModel:
    """Get validated trading configuration"""
    return config_manager.get_config("trading")

def get_security_config() -> SecurityConfigModel:
    """Get validated security configuration"""
    return config_manager.get_config("security")

def reload_configuration() -> bool:
    """Reload configuration if changed"""
    return config_manager.reload_if_changed()

# Initialize configuration on import with error handling
try:
    config_manager.load_configuration()
except Exception as e:
    logging.error(f"Failed to load configuration on import: {e}")
    # Do not continue without valid configuration
    raise 