from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel
import os
from pathlib import Path
import json

class TradingMode(Enum):
    LIVE = "LIVE"
    SIMULATION = "SIMULATION"

class SecurityLevel(Enum):
    STANDARD = "STANDARD"
    HIGH = "HIGH"
    MAXIMUM = "MAXIMUM"

class UnifiedConfig(BaseModel):
    """Core configuration for the trading system"""
    
    # Trading parameters
    trading_mode: TradingMode
    security_level: SecurityLevel
    max_trade_size_sol: float
    min_trade_size_sol: float
    max_slippage_percent: float
    profit_target_percent: float
    stop_loss_percent: float
    
    # Safety settings
    enable_kill_switch: bool = True
    emergency_stop_enabled: bool = True
    max_daily_loss_sol: float
    
    # API configuration
    helius_api_key: str
    solana_tracker_api_key: str
    jupiter_api_key: str
    raydium_api_key: str
    quicknode_rpc_url: Optional[str] = None
    dexscreener_api_key: Optional[str] = None
    birdeye_api_key: Optional[str] = None
    
    class Config:
        use_enum_values = True

class UnifiedConfigManager:
    """Manager for unified trading configuration"""
    
    def __init__(self):
        self.config: Optional[UnifiedConfig] = None
        # Get the project root directory (where worker_ant_v1 is located)
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = self.project_root / "config"
        self.config_file = self.config_dir / ".env.production"
        self.template_file = self.config_dir / "env.template"
        self.load_config()
    
    def load_config(self):
        """Load configuration from environment file"""
        if not self.config_file.exists():
            if self.template_file.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_file}\n"
                    f"Please copy {self.template_file} to {self.config_file} "
                    "and fill in your API keys and settings."
                )
            else:
                raise FileNotFoundError(
                    f"Neither configuration file {self.config_file} "
                    f"nor template {self.template_file} found."
                )
        
        # Load environment variables
        config_data = {}
        with open(self.config_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key and value and '\x00' not in value:
                        os.environ[key] = value
                        config_data[key.lower()] = value
        
        # Create config object
        self.config = UnifiedConfig(
            trading_mode=os.getenv('TRADING_MODE', 'SIMULATION'),
            security_level=os.getenv('SECURITY_LEVEL', 'HIGH'),
            max_trade_size_sol=float(os.getenv('MAX_TRADE_SIZE_SOL', '5.0')),
            min_trade_size_sol=float(os.getenv('MIN_TRADE_SIZE_SOL', '0.1')),
            max_slippage_percent=float(os.getenv('MAX_SLIPPAGE_PERCENT', '1.0')),
            profit_target_percent=float(os.getenv('PROFIT_TARGET_PERCENT', '2.5')),
            stop_loss_percent=float(os.getenv('STOP_LOSS_PERCENT', '1.0')),
            max_daily_loss_sol=float(os.getenv('MAX_DAILY_LOSS_SOL', '10.0')),
            enable_kill_switch=os.getenv('ENABLE_KILL_SWITCH', 'true').lower() == 'true',
            emergency_stop_enabled=os.getenv('EMERGENCY_STOP_ENABLED', 'true').lower() == 'true',
            helius_api_key=os.getenv('HELIUS_API_KEY', ''),
            solana_tracker_api_key=os.getenv('SOLANA_TRACKER_API_KEY', ''),
            jupiter_api_key=os.getenv('JUPITER_API_KEY', ''),
            raydium_api_key=os.getenv('RAYDIUM_API_KEY', ''),
            quicknode_rpc_url=os.getenv('QUICKNODE_RPC_URL'),
            dexscreener_api_key=os.getenv('DEXSCREENER_API_KEY'),
            birdeye_api_key=os.getenv('BIRDEYE_API_KEY')
        )
    
    def get_config(self) -> UnifiedConfig:
        """Get current configuration"""
        if not self.config:
            self.load_config()
        return self.config
    
    def validate_config(self) -> List[str]:
        """Validate current configuration"""
        errors = []
        
        if not self.config:
            return ["Configuration not loaded"]
        
        # Check critical API keys
        if not self.config.helius_api_key:
            errors.append("Missing HELIUS_API_KEY")
        elif self.config.helius_api_key == "your_helius_api_key_here":
            errors.append("HELIUS_API_KEY is still set to template value")
            
        if not self.config.solana_tracker_api_key:
            errors.append("Missing SOLANA_TRACKER_API_KEY")
        elif self.config.solana_tracker_api_key == "your_solana_tracker_api_key":
            errors.append("SOLANA_TRACKER_API_KEY is still set to template value")
            
        if not self.config.jupiter_api_key:
            errors.append("Missing JUPITER_API_KEY")
        elif self.config.jupiter_api_key == "your_jupiter_api_key":
            errors.append("JUPITER_API_KEY is still set to template value")
            
        if not self.config.raydium_api_key:
            errors.append("Missing RAYDIUM_API_KEY")
        elif self.config.raydium_api_key == "your_raydium_api_key":
            errors.append("RAYDIUM_API_KEY is still set to template value")
        
        # Validate trading parameters
        if self.config.max_trade_size_sol < self.config.min_trade_size_sol:
            errors.append("MAX_TRADE_SIZE_SOL must be greater than MIN_TRADE_SIZE_SOL")
        
        if self.config.max_slippage_percent <= 0 or self.config.max_slippage_percent > 100:
            errors.append("Invalid MAX_SLIPPAGE_PERCENT (must be between 0 and 100)")
        
        if self.config.profit_target_percent <= 0:
            errors.append("PROFIT_TARGET_PERCENT must be positive")
        
        if self.config.stop_loss_percent <= 0:
            errors.append("STOP_LOSS_PERCENT must be positive")
        
        return errors
    
    def save_config(self):
        """Save current configuration to file"""
        if not self.config:
            raise ValueError("No configuration to save")
        
        config_dict = self.config.dict()
        
        with open(self.config_file, 'w') as f:
            for key, value in config_dict.items():
                if value is not None:
                    f.write(f"{key.upper()}={value}\n")

    def check_api_keys_configured(self) -> bool:
        """Check if all required API keys are properly configured"""
        if not self.config:
            self.load_config()
            
        template_values = {
            "your_helius_api_key_here",
            "your_solana_tracker_api_key",
            "your_jupiter_api_key",
            "your_raydium_api_key"
        }
        
        keys_to_check = [
            self.config.helius_api_key,
            self.config.solana_tracker_api_key,
            self.config.jupiter_api_key,
            self.config.raydium_api_key
        ]
        
        return all(
            key and key not in template_values
            for key in keys_to_check
        )

def get_trading_config() -> UnifiedConfig:
    """Get current trading configuration"""
    manager = UnifiedConfigManager()
    return manager.get_config()

def get_security_config() -> Dict[str, any]:
    """Get security configuration"""
    config = get_trading_config()
    return {
        'security_level': config.security_level,
        'enable_kill_switch': config.enable_kill_switch,
        'emergency_stop_enabled': config.emergency_stop_enabled,
        'max_daily_loss_sol': config.max_daily_loss_sol
    }

def get_network_config() -> Dict[str, any]:
    """Get network configuration"""
    config = get_trading_config()
    return {
        'helius_api_key': config.helius_api_key,
        'solana_tracker_api_key': config.solana_tracker_api_key,
        'jupiter_api_key': config.jupiter_api_key,
        'raydium_api_key': config.raydium_api_key,
        'quicknode_rpc_url': config.quicknode_rpc_url,
        'dexscreener_api_key': config.dexscreener_api_key,
        'birdeye_api_key': config.birdeye_api_key
    }

def mask_sensitive_value(value: str, mask_char: str = '*', visible_chars: int = 4) -> str:
    """Mask sensitive values like API keys for logging"""
    if not value or len(value) <= visible_chars:
        return mask_char * len(value) if value else ''
    
    return value[:visible_chars] + mask_char * (len(value) - visible_chars)

def get_config_manager() -> UnifiedConfigManager:
    """Get global config manager instance"""
    return UnifiedConfigManager() 