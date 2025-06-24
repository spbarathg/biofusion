"""
Optimized Configuration for Profitable Worker Ant V1
===================================================

High-performance, deployment-ready trading configuration.
Optimized for 12+ trades/hour with maximum profitability.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

class TradingMode(Enum):
    LIVE = "live"
    SIMULATION = "simulation"
    PAPER = "paper"

@dataclass
class TradingConfig:
    """Optimized trading parameters for maximum profitability"""
    
    # === PROFITABILITY SETTINGS ===
    trade_amount_sol: float = 0.2  # $50-100 at ~$200 SOL
    min_trade_amount_sol: float = 0.1  # Minimum for low liquidity
    max_trade_amount_sol: float = 0.5  # Maximum for high confidence
    
    # Dynamic profit targets (5-15% range)
    base_profit_target_percent: float = 8.0  # Base target
    min_profit_target_percent: float = 5.0   # Low volatility
    max_profit_target_percent: float = 15.0  # High volatility
    
    # Dynamic stop losses (-5 to -10% range) 
    base_stop_loss_percent: float = 7.0   # Base stop loss
    min_stop_loss_percent: float = 5.0    # Tight for stable tokens
    max_stop_loss_percent: float = 10.0   # Wider for volatile tokens
    
    # === SPEED & LATENCY OPTIMIZATION ===
    entry_timeout_ms: int = 300  # Must execute within 300ms
    max_slippage_percent: float = 1.5     # Maximum allowed
    min_slippage_percent: float = 0.3     # Minimum for good pools
    dynamic_slippage: bool = True          # Adjust based on pool depth
    
    # === THROUGHPUT OPTIMIZATION ===
    max_trades_per_hour: int = 12          # Target throughput
    min_time_between_trades_seconds: int = 15  # Reduced for speed
    max_concurrent_positions: int = 3      # Conservative risk
    
    # Exit timing (2-3 minutes as requested)
    timeout_exit_seconds: int = 150        # 2.5 minutes average
    min_timeout_seconds: int = 120         # 2 minutes minimum  
    max_timeout_seconds: int = 180         # 3 minutes maximum
    
    # === RISK MANAGEMENT ===
    max_wallet_risk_percent: float = 5.0   # Never risk >5% of wallet per trade
    max_daily_loss_sol: float = 2.0        # Daily circuit breaker
    max_daily_loss_percent: float = 10.0   # % of wallet daily limit
    
    # Position sizing risk
    position_size_multiplier: float = 1.0   # Base sizing
    max_position_multiplier: float = 2.0    # Max increase on wins
    min_position_multiplier: float = 0.5    # Min decrease on losses
    
    # === LIQUIDITY & FILTERING ===
    min_liquidity_sol: float = 15.0        # Higher minimum (3-5K+ USD)
    min_liquidity_usd: float = 3000        # Absolute minimum
    max_liquidity_usd: float = 500000      # Avoid oversized pools
    
    # === PERFORMANCE SCALING ===
    performance_window_trades: int = 10     # Look at last N trades
    win_rate_scale_threshold: float = 70.0  # Scale up if >70% win rate
    loss_rate_scale_threshold: float = 30.0 # Scale down if <30% win rate
    
    # === RPC OPTIMIZATION ===
    rpc_url: str = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    backup_rpc_urls: List[str] = field(default_factory=lambda: [
        "https://solana-api.projectserum.com",
        "https://api.mainnet-beta.solana.com"
    ])
    use_private_rpc: bool = True
    rpc_timeout_seconds: int = 3
    max_rpc_retries: int = 3
    
    # === COST CONTROL ===
    max_tx_fee_sol: float = 0.001          # Cap transaction fees
    max_rpc_cost_per_trade: float = 0.0001  # Cost monitoring
    alert_cost_threshold_percent: float = 10.0  # Alert if costs >10% profit


@dataclass  
class ScannerConfig:
    """Optimized scanning for cost efficiency and speed"""
    
    # === API EFFICIENCY ===
    scan_interval_seconds: float = 0.5      # 2 scans/second for cost control
    max_tokens_per_scan: int = 8            # Process top tokens only
    api_rate_limit_delay: float = 0.1       # Prevent rate limiting
    
    # Data sources (prioritized by cost/quality)
    use_raydium_api: bool = True            # Free, fast
    use_orca_api: bool = True               # Free, good coverage  
    use_dexscreener_api: bool = True        # Free tier available
    use_birdeye_api: bool = False           # Paid, use if API key available
    
    # API keys
    birdeye_api_key: Optional[str] = os.getenv("BIRDEYE_API_KEY")
    dexscreener_pro: bool = os.getenv("DEXSCREENER_PRO", "false").lower() == "true"
    
    # === FILTERING OPTIMIZATION ===
    # Pool age filtering (catch early but avoid ultra-risky)
    min_pool_age_seconds: int = 30          # Avoid immediate launches
    max_pool_age_seconds: int = 300         # Focus on 5min window
    
    # Quality filters
    min_holder_count: int = 50              # Avoid concentrated ownership
    max_holder_concentration: float = 30.0   # Top holder <30%
    min_volume_24h_usd: float = 10000       # $10K daily volume minimum
    
    # Blacklist management
    blacklisted_tokens: List[str] = field(default_factory=lambda: [
        "11111111111111111111111111111111",  # System program
        # Add known rugs dynamically
    ])
    auto_blacklist_rugs: bool = True        # Auto-add failed tokens
    
    # === HONEYPOT DETECTION ===
    enable_honeypot_check: bool = True
    max_buy_tax_percent: float = 5.0
    max_sell_tax_percent: float = 5.0
    require_liquidity_locked: bool = False   # Too restrictive for memecoins


@dataclass
class MonitoringConfig:
    """Optimized logging and cost tracking"""
    
    # === LOGGING EFFICIENCY ===
    log_level: str = "INFO"
    log_file: str = "worker_ant_v1/trades.log"
    db_path: str = "worker_ant_v1/trades.db"
    
    # Batch logging for efficiency
    batch_logs: bool = True
    batch_size: int = 10
    batch_timeout_seconds: int = 60
    
    # === COST TRACKING ===
    track_rpc_costs: bool = True
    track_gas_costs: bool = True
    track_api_costs: bool = True
    cost_alert_threshold_sol: float = 0.1
    
    # === PERFORMANCE MONITORING ===
    track_latency: bool = True
    track_slippage: bool = True
    track_win_rate: bool = True
    
    # Hourly summaries
    enable_hourly_summary: bool = True
    hourly_summary_file: str = "worker_ant_v1/hourly_reports.log"


@dataclass
class WalletConfig:
    """Secure wallet configuration"""
    
    wallet_private_key: Optional[str] = os.getenv("WALLET_PRIVATE_KEY")
    wallet_path: str = "worker_ant_v1/wallet.json"
    auto_create_wallet: bool = False  # Require explicit wallet setup
    
    # Backup configuration
    backup_wallet_path: str = "worker_ant_v1/backup_wallet.json"
    enable_wallet_backup: bool = True


@dataclass
class DeploymentConfig:
    """Deployment and operational settings"""
    
    # Operating mode
    trading_mode: TradingMode = TradingMode.SIMULATION  # Default to safe mode
    
    # Resource limits (for VPS deployment)
    max_memory_mb: int = 512
    max_cpu_percent: float = 50.0
    
    # Health monitoring
    enable_health_check: bool = True
    health_check_interval: int = 30
    auto_restart_on_error: bool = True
    max_restart_attempts: int = 3
    
    # Alerts
    enable_telegram_alerts: bool = False
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    
    enable_email_alerts: bool = False
    alert_email: Optional[str] = os.getenv("ALERT_EMAIL")


# === GLOBAL CONFIG INSTANCES ===
trading_config = TradingConfig()
scanner_config = ScannerConfig()
monitoring_config = MonitoringConfig()
wallet_config = WalletConfig()
deployment_config = DeploymentConfig()


def load_env_config():
    """Load all configuration from environment variables"""
    global trading_config, scanner_config, monitoring_config, wallet_config, deployment_config
    
    # === TRADING CONFIG ===
    if os.getenv("TRADE_AMOUNT_SOL"):
        trading_config.trade_amount_sol = float(os.getenv("TRADE_AMOUNT_SOL"))
    
    if os.getenv("PROFIT_TARGET_PERCENT"):
        trading_config.base_profit_target_percent = float(os.getenv("PROFIT_TARGET_PERCENT"))
        
    if os.getenv("STOP_LOSS_PERCENT"):
        trading_config.base_stop_loss_percent = float(os.getenv("STOP_LOSS_PERCENT"))
        
    if os.getenv("MAX_DAILY_LOSS_SOL"):
        trading_config.max_daily_loss_sol = float(os.getenv("MAX_DAILY_LOSS_SOL"))
        
    if os.getenv("TRADING_MODE"):
        mode_str = os.getenv("TRADING_MODE").lower()
        if mode_str in [m.value for m in TradingMode]:
            deployment_config.trading_mode = TradingMode(mode_str)
    
    # === RPC CONFIG ===
    if os.getenv("PRIVATE_RPC_URL"):
        trading_config.rpc_url = os.getenv("PRIVATE_RPC_URL")
        trading_config.use_private_rpc = True
        
    if os.getenv("BACKUP_RPC_URLS"):
        urls = os.getenv("BACKUP_RPC_URLS").split(",")
        trading_config.backup_rpc_urls = [url.strip() for url in urls]
    
    # === API KEYS ===
    if os.getenv("BIRDEYE_API_KEY"):
        scanner_config.birdeye_api_key = os.getenv("BIRDEYE_API_KEY")
        scanner_config.use_birdeye_api = True
        
    # === WALLET CONFIG ===
    if os.getenv("WALLET_PRIVATE_KEY"):
        wallet_config.wallet_private_key = os.getenv("WALLET_PRIVATE_KEY")


def validate_config():
    """Comprehensive configuration validation"""
    errors = []
    
    # === TRADING VALIDATION ===
    if trading_config.trade_amount_sol <= 0:
        errors.append("trade_amount_sol must be positive")
        
    if trading_config.base_profit_target_percent <= 0:
        errors.append("profit_target_percent must be positive")
        
    if trading_config.base_stop_loss_percent <= 0:
        errors.append("stop_loss_percent must be positive")
        
    if trading_config.max_slippage_percent > 5.0:
        errors.append("max_slippage_percent too high (>5%)")
        
    if trading_config.max_daily_loss_sol <= 0:
        errors.append("max_daily_loss_sol must be positive")
    
    # === WALLET VALIDATION ===
    if deployment_config.trading_mode == TradingMode.LIVE:
        if not wallet_config.wallet_private_key:
            errors.append("wallet_private_key required for live trading")
    
    # === SCANNER VALIDATION ===
    if scanner_config.scan_interval_seconds < 0.5:
        errors.append("scan_interval too fast (risk of rate limiting)")
        
    if scanner_config.min_liquidity_sol < 5.0:
        errors.append("min_liquidity_sol too low (rug risk)")
    
    # === RISK VALIDATION ===
    if trading_config.max_wallet_risk_percent > 10.0:
        errors.append("wallet risk too high (>10%)")
        
    if trading_config.max_concurrent_positions > 5:
        errors.append("too many concurrent positions (>5)")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True


def get_optimized_trade_size(win_rate: float, recent_performance: float) -> float:
    """Calculate optimized trade size based on performance"""
    base_size = trading_config.trade_amount_sol
    
    # Performance-based scaling
    if win_rate > trading_config.win_rate_scale_threshold:
        multiplier = min(trading_config.max_position_multiplier, 
                        1.0 + (win_rate - 50) / 100)
    elif win_rate < trading_config.loss_rate_scale_threshold:
        multiplier = max(trading_config.min_position_multiplier,
                        1.0 - (50 - win_rate) / 100)
    else:
        multiplier = 1.0
        
    return base_size * multiplier


def get_dynamic_slippage(liquidity_usd: float) -> float:
    """Calculate dynamic slippage based on pool liquidity"""
    if not trading_config.dynamic_slippage:
        return trading_config.max_slippage_percent
        
    # Lower slippage for higher liquidity
    if liquidity_usd > 100000:  # $100K+
        return trading_config.min_slippage_percent
    elif liquidity_usd > 50000:  # $50K+
        return (trading_config.min_slippage_percent + trading_config.max_slippage_percent) / 2
    else:
        return trading_config.max_slippage_percent


def get_dynamic_timeout(volatility_score: float) -> int:
    """Calculate dynamic timeout based on token volatility"""
    base = trading_config.timeout_exit_seconds
    
    if volatility_score > 0.8:  # High volatility
        return trading_config.min_timeout_seconds
    elif volatility_score < 0.3:  # Low volatility  
        return trading_config.max_timeout_seconds
    else:
        return base


# Initialize on import
load_env_config() 