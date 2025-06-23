"""
Configuration for Worker Ant V1
===============================

All trading parameters, thresholds, and safety settings.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradingConfig:
    """Core trading parameters"""
    
    # Trade sizing
    trade_amount_sol: float = 0.05  # $20-50 at current SOL prices
    max_position_size_sol: float = 0.2  # Max position size
    
    # Entry criteria
    min_liquidity_sol: float = 5.0  # Minimum pool liquidity
    max_slippage_percent: float = 1.5  # Maximum allowed slippage
    entry_timeout_ms: int = 200  # Must execute within 200ms
    
    # Exit strategy
    profit_target_percent: float = 10.0  # Target profit (5-15%)
    stop_loss_percent: float = 10.0  # Stop loss threshold
    timeout_exit_seconds: int = 45  # Exit if no movement
    
    # Safety limits
    max_trades_per_hour: int = 20
    min_time_between_trades_seconds: int = 30
    max_daily_loss_sol: float = 1.0  # Circuit breaker
    
    # RPC settings
    rpc_url: str = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    use_private_rpc: bool = False
    rpc_timeout_seconds: int = 5


@dataclass
class WalletConfig:
    """Wallet and security settings"""
    
    wallet_private_key: Optional[str] = None  # Base58 encoded
    wallet_path: str = "worker_ant_v1/wallet.json"
    auto_create_wallet: bool = True


@dataclass
class MonitoringConfig:
    """Logging and monitoring settings"""
    
    log_level: str = "INFO"
    log_file: str = "worker_ant_v1/trades.log"
    db_path: str = "worker_ant_v1/trades.db"
    enable_detailed_logging: bool = True
    
    # KPI tracking
    track_win_rate: bool = True
    track_slippage: bool = True
    track_latency: bool = True


@dataclass
class ScannerConfig:
    """Token scanning parameters"""
    
    # Data sources
    use_birdeye_api: bool = True
    use_dexscreener_api: bool = True
    birdeye_api_key: Optional[str] = os.getenv("BIRDEYE_API_KEY")
    
    # Filtering
    min_pool_age_seconds: int = 10  # Avoid ultra-fresh pools
    max_pool_age_seconds: int = 300  # Focus on new tokens (5 min)
    blacklisted_tokens: list = None  # Token addresses to avoid
    
    # Performance
    scan_interval_seconds: float = 0.5  # How often to check for new tokens
    max_tokens_per_scan: int = 5  # Process top N tokens per scan


# Default configuration instance
config = TradingConfig()
wallet_config = WalletConfig()
monitoring_config = MonitoringConfig()
scanner_config = ScannerConfig()

# Initialize blacklist if None
if scanner_config.blacklisted_tokens is None:
    scanner_config.blacklisted_tokens = [
        # Add known rug pulls or problematic tokens here
        "11111111111111111111111111111111",  # System program
    ]


def load_config_from_env():
    """Load configuration from environment variables"""
    global config, wallet_config, monitoring_config, scanner_config
    
    # Trading config
    if os.getenv("TRADE_AMOUNT_SOL"):
        config.trade_amount_sol = float(os.getenv("TRADE_AMOUNT_SOL"))
    if os.getenv("PROFIT_TARGET_PERCENT"):
        config.profit_target_percent = float(os.getenv("PROFIT_TARGET_PERCENT"))
    if os.getenv("STOP_LOSS_PERCENT"):
        config.stop_loss_percent = float(os.getenv("STOP_LOSS_PERCENT"))
    
    # Wallet config
    if os.getenv("WALLET_PRIVATE_KEY"):
        wallet_config.wallet_private_key = os.getenv("WALLET_PRIVATE_KEY")
    
    # Scanner config
    if os.getenv("BIRDEYE_API_KEY"):
        scanner_config.birdeye_api_key = os.getenv("BIRDEYE_API_KEY")


def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    # Trading validation
    if config.trade_amount_sol <= 0:
        errors.append("trade_amount_sol must be positive")
    if config.profit_target_percent <= 0:
        errors.append("profit_target_percent must be positive")
    if config.stop_loss_percent <= 0:
        errors.append("stop_loss_percent must be positive")
    if config.max_slippage_percent < 0 or config.max_slippage_percent > 50:
        errors.append("max_slippage_percent must be between 0 and 50")
    
    # Safety validation
    if config.max_daily_loss_sol <= 0:
        errors.append("max_daily_loss_sol must be positive")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True


# Load environment variables on import
load_config_from_env() 