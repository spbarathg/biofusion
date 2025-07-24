"""
CONSTANTS AND ENUMS
==================

Centralized constants and Enums to replace magic strings throughout the codebase.
This ensures consistency, type safety, and easier maintenance.
"""

from enum import Enum, auto
from typing import Dict, List, Any


class TradingMode(Enum):
    """Trading mode enumeration"""
    SIMULATION = "simulation"
    LIVE = "live"
    PRODUCTION = "production"
    TEST = "test"


class SecurityLevel(Enum):
    """Security level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAXIMUM = "maximum"


class ValidationLevel(Enum):
    """Validation severity levels"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class DataSource(Enum):
    """Available data sources"""
    JUPITER = "jupiter"
    BIRDEYE = "birdeye"
    DEXSCREENER = "dexscreener"
    HELIUS = "helius"
    RAYDIUM = "raydium"


class TradingStatus(Enum):
    """Trading operation status"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class OrderType(Enum):
    """Order type enumeration"""
    BUY = "buy"
    SELL = "sell"
    SWAP = "swap"


class SentimentDecision(Enum):
    """Sentiment analysis decision"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class MemecoinPattern(Enum):
    """Memecoin pattern types"""
    HYPE_CYCLE = "hype_cycle"
    FOMO_PATTERN = "fomo_pattern"
    DUMP_PATTERN = "dump_pattern"
    ACCUMULATION = "accumulation"
    PUMP_AND_DUMP = "pump_and_dump"
    STEADY_GROWTH = "steady_growth"


class VaultType(Enum):
    """Vault type enumeration"""
    PROFIT_VAULT = "profit_vault"
    EMERGENCY_VAULT = "emergency_vault"
    COMPOUNDING_VAULT = "compounding_vault"


class CompoundCycle(Enum):
    """Compounding cycle speeds"""
    LIGHTNING = "lightning"
    RAPID = "rapid"
    AGGRESSIVE = "aggressive"
    STANDARD = "standard"


class WalletTier(Enum):
    """Wallet performance tier"""
    ELITE = "elite"
    VETERAN = "veteran"
    ROOKIE = "rookie"
    RETIRED = "retired"


class AlertType(Enum):
    """Alert type enumeration"""
    TRADE_EXECUTED = "trade_executed"
    PROFIT_TAKEN = "profit_taken"
    STOP_LOSS_HIT = "stop_loss_hit"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    SYSTEM_ERROR = "system_error"
    WALLET_EVOLUTION = "wallet_evolution"
    VAULT_DEPOSIT = "vault_deposit"


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DatabaseType(Enum):
    """Database type enumeration"""
    SQLITE = "sqlite"
    REDIS = "redis"
    MONGODB = "mongodb"


class NetworkType(Enum):
    """Network type enumeration"""
    MAINNET = "mainnet"
    DEVNET = "devnet"
    TESTNET = "testnet"


class TokenStandard(Enum):
    """Token standard enumeration"""
    SPL_TOKEN = "spl_token"
    SOL = "sol"
    USDC = "usdc"
    USDT = "usdt"


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class PerformanceMetric(Enum):
    """Performance metric types"""
    WIN_RATE = "win_rate"
    TOTAL_PROFIT = "total_profit"
    AVERAGE_PROFIT = "average_profit"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    TOTAL_TRADES = "total_trades"


class ConfigurationSection(Enum):
    """Configuration section names"""
    TRADING = "trading"
    SECURITY = "security"
    NETWORK = "network"
    MONITORING = "monitoring"
    DATABASE = "database"
    INTELLIGENCE = "intelligence"
    VAULT = "vault"


# API Endpoints
class APIEndpoints:
    """Centralized API endpoint constants"""
    
    # Jupiter DEX
    JUPITER_BASE_URL = "https://quote-api.jup.ag/v6"
    JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
    JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"
    JUPITER_PRICE_URL = "https://price.jup.ag/v4/price"
    
    # Birdeye
    BIRDEYE_BASE_URL = "https://public-api.birdeye.so"
    BIRDEYE_TOKEN_URL = "https://public-api.birdeye.so/public/token"
    BIRDEYE_TRENDING_URL = "https://public-api.birdeye.so/public/trending"
    
    # DexScreener
    DEXSCREENER_BASE_URL = "https://api.dexscreener.com/latest/dex"
    DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens"
    DEXSCREENER_TRENDING_URL = "https://api.dexscreener.com/latest/dex/tokens/trending"
    
    # Helius
    HELIUS_BASE_URL = "https://api.helius.xyz/v0"
    HELIUS_TOKEN_METADATA_URL = "https://api.helius.xyz/v0/token-metadata"
    
    # Solana RPC
    SOLANA_MAINNET_RPC = "https://api.mainnet-beta.solana.com"
    SOLANA_DEVNET_RPC = "https://api.devnet.solana.com"
    SOLANA_TESTNET_RPC = "https://api.testnet.solana.com"


# Token Mints
class TokenMints:
    """Common token mint addresses"""
    
    SOL = "So11111111111111111111111111111111111111112"
    USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
    WSOL = "So11111111111111111111111111111111111111112"


# Trading Constants
class TradingConstants:
    """Trading-related constants"""
    
    # Position sizing
    MIN_POSITION_SIZE_SOL = 0.01
    MAX_POSITION_SIZE_SOL = 1000.0
    DEFAULT_POSITION_SIZE_PERCENT = 0.2
    
    # Risk management
    MIN_STOP_LOSS_PERCENT = 0.5
    MAX_STOP_LOSS_PERCENT = 50.0
    DEFAULT_STOP_LOSS_PERCENT = 5.0
    
    MIN_PROFIT_TARGET_PERCENT = 0.5
    MAX_PROFIT_TARGET_PERCENT = 100.0
    DEFAULT_PROFIT_TARGET_PERCENT = 10.0
    
    # Slippage
    MIN_SLIPPAGE_PERCENT = 0.1
    MAX_SLIPPAGE_PERCENT = 5.0
    DEFAULT_SLIPPAGE_PERCENT = 1.0
    
    # Time limits
    MIN_HOLD_TIME_MINUTES = 1
    MAX_HOLD_TIME_HOURS = 24
    DEFAULT_HOLD_TIME_HOURS = 4
    
    # Capital limits
    MIN_INITIAL_CAPITAL_SOL = 0.1
    MAX_INITIAL_CAPITAL_SOL = 10000.0
    
    # Concurrent positions
    MIN_CONCURRENT_POSITIONS = 1
    MAX_CONCURRENT_POSITIONS = 20
    DEFAULT_CONCURRENT_POSITIONS = 5


# Market Scanner Constants
class MarketScannerConstants:
    """Market scanner related constants"""
    
    # Liquidity thresholds
    MIN_LIQUIDITY_SOL = 5.0
    MAX_LIQUIDITY_SOL = 10000.0
    
    # Volume thresholds
    MIN_VOLUME_24H_SOL = 50.0
    MAX_VOLUME_24H_SOL = 1000000.0
    
    # Price impact
    MIN_PRICE_IMPACT_PERCENT = 0.1
    MAX_PRICE_IMPACT_PERCENT = 10.0
    
    # Token age
    MIN_TOKEN_AGE_HOURS = 0.1
    MAX_TOKEN_AGE_HOURS = 8760  # 1 year
    
    # Holder count
    MIN_HOLDER_COUNT = 10
    MAX_HOLDER_COUNT = 1000000
    
    # Trending tokens
    MIN_TRENDING_TOKENS = 10
    MAX_TRENDING_TOKENS = 100
    DEFAULT_TRENDING_TOKENS = 30


# Sentiment Analysis Constants
class SentimentConstants:
    """Sentiment analysis related constants"""
    
    # Decision thresholds
    STRONG_BUY_THRESHOLD = 0.6
    BUY_THRESHOLD = 0.3
    NEUTRAL_HIGH_THRESHOLD = 0.1
    NEUTRAL_LOW_THRESHOLD = -0.1
    SELL_THRESHOLD = -0.3
    STRONG_SELL_THRESHOLD = -0.6
    
    # Weights
    IMMEDIATE_WEIGHT = 0.5
    TREND_WEIGHT = 0.3
    STABILITY_WEIGHT = 0.1
    STRENGTH_WEIGHT = 0.1
    
    # Confidence thresholds
    MIN_CONFIDENCE = 0.0
    MAX_CONFIDENCE = 1.0
    DEFAULT_CONFIDENCE_THRESHOLD = 0.6
    
    # Blacklist duration
    DEFAULT_BLACKLIST_DURATION_HOURS = 24


# Safety Constants
class SafetyConstants:
    """Safety and risk management constants"""
    
    # Kill switch
    DEFAULT_KILL_SWITCH_ENABLED = True
    DEFAULT_EMERGENCY_STOP_ENABLED = True
    
    # Drawdown limits
    MIN_DRAWDOWN_STOP_PERCENT = 5.0
    MAX_DRAWDOWN_STOP_PERCENT = 50.0
    DEFAULT_DRAWDOWN_STOP_PERCENT = 20.0
    
    # Daily loss limits
    MIN_DAILY_LOSS_PERCENT = 1.0
    MAX_DAILY_LOSS_PERCENT = 50.0
    DEFAULT_DAILY_LOSS_PERCENT = 15.0
    
    # Trade limits
    MIN_TRADES_PER_HOUR = 1
    MAX_TRADES_PER_HOUR = 1000
    DEFAULT_MAX_TRADES_PER_HOUR = 100


# Vault Constants
class VaultConstants:
    """Vault system constants"""
    
    # Profit percentages
    MIN_PROFIT_VAULT_PERCENT = 0.05
    MAX_PROFIT_VAULT_PERCENT = 0.5
    DEFAULT_PROFIT_VAULT_PERCENT = 0.2
    
    # Minimum balances
    MIN_VAULT_BALANCE_SOL = 0.01
    MAX_VAULT_BALANCE_SOL = 1000.0
    DEFAULT_MIN_VAULT_BALANCE_SOL = 1.0
    
    # Compounding
    MIN_COMPOUNDING_PERCENT = 0.1
    MAX_COMPOUNDING_PERCENT = 0.95
    DEFAULT_COMPOUNDING_PERCENT = 0.8


# Network Constants
class NetworkConstants:
    """Network and RPC constants"""
    
    # Timeouts
    MIN_RPC_TIMEOUT_SECONDS = 1
    MAX_RPC_TIMEOUT_SECONDS = 60
    DEFAULT_RPC_TIMEOUT_SECONDS = 10
    
    # Retries
    MIN_RETRIES = 1
    MAX_RETRIES = 10
    DEFAULT_MAX_RETRIES = 3
    
    # Retry delays
    MIN_RETRY_DELAY_SECONDS = 0.1
    MAX_RETRY_DELAY_SECONDS = 10
    DEFAULT_RETRY_DELAY_SECONDS = 1
    
    # Transaction fees
    MIN_TX_FEE_SOL = 0.000005
    MAX_TX_FEE_SOL = 0.01
    DEFAULT_TX_FEE_SOL = 0.0005


# Monitoring Constants
class MonitoringConstants:
    """Monitoring and logging constants"""
    
    # Log levels
    DEFAULT_LOG_LEVEL = LogLevel.INFO
    
    # Performance intervals
    MIN_PERFORMANCE_INTERVAL_MINUTES = 1
    MAX_PERFORMANCE_INTERVAL_MINUTES = 60
    DEFAULT_PERFORMANCE_INTERVAL_MINUTES = 5
    
    # Cache durations
    MIN_CACHE_DURATION_SECONDS = 10
    MAX_CACHE_DURATION_SECONDS = 3600
    DEFAULT_CACHE_DURATION_SECONDS = 300
    
    # Rate limiting
    MIN_RATE_LIMIT_REQUESTS_PER_SECOND = 1
    MAX_RATE_LIMIT_REQUESTS_PER_SECOND = 100
    DEFAULT_RATE_LIMIT_REQUESTS_PER_SECOND = 10


# File Paths
class FilePaths:
    """Common file paths"""
    
    # Configuration
    ENV_TEMPLATE = "config/env.template"
    ENV_PRODUCTION = ".env.production"
    ENV_DEVELOPMENT = ".env.development"
    
    # Logs
    LOG_DIRECTORY = "logs"
    DEFAULT_LOG_FILE = "logs/trading_bot.log"
    
    # Data
    DATA_DIRECTORY = "data"
    DEFAULT_DATABASE_PATH = "data/trading_bot.db"
    
    # Backups
    BACKUP_DIRECTORY = "backups"
    
    # Config
    CONFIG_DIRECTORY = "config"
    REQUIREMENTS_FILE = "config/requirements.txt"


# Error Messages
class ErrorMessages:
    """Standardized error messages"""
    
    # Configuration
    MISSING_API_KEY = "Missing required API key: {}"
    INVALID_CONFIG_VALUE = "Invalid configuration value for {}: {}"
    CONFIG_FILE_NOT_FOUND = "Configuration file not found: {}"
    
    # Trading
    INSUFFICIENT_BALANCE = "Insufficient balance for trade: required {}, available {}"
    TRADE_EXECUTION_FAILED = "Trade execution failed: {}"
    INVALID_TOKEN_ADDRESS = "Invalid token address: {}"
    
    # Network
    RPC_CONNECTION_FAILED = "RPC connection failed: {}"
    TIMEOUT_ERROR = "Operation timed out: {}"
    RATE_LIMIT_EXCEEDED = "Rate limit exceeded for {}"
    
    # Validation
    VALIDATION_FAILED = "Validation failed: {}"
    MISSING_REQUIRED_FIELD = "Missing required field: {}"
    INVALID_VALUE_RANGE = "Value {} is outside valid range [{}, {}]"


# Success Messages
class SuccessMessages:
    """Standardized success messages"""
    
    # Initialization
    SYSTEM_INITIALIZED = "System initialized successfully"
    COMPONENT_READY = "{} component ready"
    VALIDATION_PASSED = "Validation passed successfully"
    
    # Trading
    TRADE_EXECUTED = "Trade executed successfully: {}"
    PROFIT_TAKEN = "Profit taken: {} SOL"
    POSITION_CLOSED = "Position closed: {}"
    
    # Safety
    KILL_SWITCH_ACTIVATED = "Kill switch activated: {}"
    VAULT_DEPOSIT_SUCCESS = "Vault deposit successful: {} SOL"
    EMERGENCY_STOP_ACTIVATED = "Emergency stop activated: {}"


# Default Values
class DefaultValues:
    """Default configuration values"""
    
    # Trading
    DEFAULT_TRADING_MODE = TradingMode.SIMULATION
    DEFAULT_SECURITY_LEVEL = SecurityLevel.HIGH
    DEFAULT_INITIAL_CAPITAL_SOL = 1.0
    DEFAULT_POSITION_SIZE_PERCENT = 0.2
    DEFAULT_MAX_POSITION_SIZE_SOL = 50.0
    DEFAULT_MIN_PROFIT_TARGET = 0.05
    DEFAULT_MAX_LOSS_PERCENT = 0.15
    DEFAULT_MAX_HOLD_TIME_HOURS = 4
    DEFAULT_SCAN_INTERVAL_SECONDS = 30
    DEFAULT_SENTIMENT_THRESHOLD = 0.3
    
    # Wallet
    DEFAULT_WALLET_COUNT = 10
    DEFAULT_EVOLUTION_INTERVAL_HOURS = 24
    DEFAULT_EVOLUTION_MUTATION_RATE = 0.1
    DEFAULT_RETIREMENT_THRESHOLD = 0.3
    
    # Market Scanner
    DEFAULT_MIN_LIQUIDITY_SOL = 5.0
    DEFAULT_MIN_VOLUME_SOL = 50.0
    DEFAULT_MAX_PRICE_IMPACT_PERCENT = 5.0
    DEFAULT_MAX_TOKEN_AGE_HOURS = 168
    DEFAULT_MIN_HOLDER_COUNT = 50
    DEFAULT_TRENDING_TOKENS_LIMIT = 30
    
    # Safety
    DEFAULT_KILL_SWITCH_ENABLED = True
    DEFAULT_EMERGENCY_STOP_ENABLED = True
    DEFAULT_VAULT_ENABLED = True
    DEFAULT_MAX_DRAWDOWN_STOP_PERCENT = 20.0
    
    # Network
    DEFAULT_SOLANA_RPC_URL = APIEndpoints.SOLANA_MAINNET_RPC
    DEFAULT_RPC_TIMEOUT_SECONDS = 10
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY_SECONDS = 1
    
    # Monitoring
    DEFAULT_LOG_LEVEL = LogLevel.INFO
    DEFAULT_PERFORMANCE_MONITORING = True
    DEFAULT_PERFORMANCE_INTERVAL_MINUTES = 5
    DEFAULT_DETAILED_LOGGING = True
    
    # Database
    DEFAULT_DATABASE_TYPE = DatabaseType.SQLITE
    DEFAULT_SQLITE_DB_PATH = "data/trading_bot.db"
    DEFAULT_REDIS_HOST = "localhost"
    DEFAULT_REDIS_PORT = 6379
    DEFAULT_REDIS_DB = 0
    
    # Vault
    DEFAULT_PROFIT_VAULT_PERCENTAGE = 0.2
    DEFAULT_MIN_VAULT_BALANCE_SOL = 1.0
    DEFAULT_COMPOUNDING_PERCENTAGE = 0.8
    DEFAULT_VAULT_PERCENTAGE = 0.2
    DEFAULT_MIN_COMPOUNDING_PROFIT_SOL = 0.01
    
    # Sentiment
    DEFAULT_SENTIMENT_MODEL = "finbert"
    DEFAULT_MIN_SENTIMENT_CONFIDENCE = 0.6
    DEFAULT_SENTIMENT_UPDATE_INTERVAL = 5
    DEFAULT_SENTIMENT_BLACKLIST_ENABLED = True
    DEFAULT_BLACKLIST_DURATION_HOURS = 24
    
    # Pattern Detection
    DEFAULT_PATTERN_DETECTION_ENABLED = True
    DEFAULT_HYPE_CYCLE_SENSITIVITY = 0.7
    DEFAULT_FOMO_PATTERN_SENSITIVITY = 0.6
    DEFAULT_DUMP_PATTERN_SENSITIVITY = 0.8


# Validation Rules
class ValidationRules:
    """Validation rules for configuration values"""
    
    @staticmethod
    def get_trading_rules() -> Dict[str, tuple]:
        """Get trading parameter validation rules"""
        return {
            "initial_capital_sol": (TradingConstants.MIN_INITIAL_CAPITAL_SOL, TradingConstants.MAX_INITIAL_CAPITAL_SOL),
            "position_size_percent": (0.01, 1.0),
            "max_position_size_sol": (TradingConstants.MIN_POSITION_SIZE_SOL, TradingConstants.MAX_POSITION_SIZE_SOL),
            "min_profit_target": (TradingConstants.MIN_PROFIT_TARGET_PERCENT, TradingConstants.MAX_PROFIT_TARGET_PERCENT),
            "max_loss_percent": (TradingConstants.MIN_STOP_LOSS_PERCENT, TradingConstants.MAX_STOP_LOSS_PERCENT),
            "max_hold_time_hours": (TradingConstants.MIN_HOLD_TIME_MINUTES / 60, TradingConstants.MAX_HOLD_TIME_HOURS),
            "scan_interval_seconds": (1, 3600),
            "sentiment_threshold": (-1.0, 1.0),
            "max_concurrent_positions": (TradingConstants.MIN_CONCURRENT_POSITIONS, TradingConstants.MAX_CONCURRENT_POSITIONS),
        }
    
    @staticmethod
    def get_safety_rules() -> Dict[str, tuple]:
        """Get safety parameter validation rules"""
        return {
            "max_drawdown_stop_percent": (SafetyConstants.MIN_DRAWDOWN_STOP_PERCENT, SafetyConstants.MAX_DRAWDOWN_STOP_PERCENT),
            "max_daily_loss_percent": (SafetyConstants.MIN_DAILY_LOSS_PERCENT, SafetyConstants.MAX_DAILY_LOSS_PERCENT),
            "max_trades_per_hour": (SafetyConstants.MIN_TRADES_PER_HOUR, SafetyConstants.MAX_TRADES_PER_HOUR),
        }
    
    @staticmethod
    def get_market_scanner_rules() -> Dict[str, tuple]:
        """Get market scanner parameter validation rules"""
        return {
            "min_liquidity_sol": (MarketScannerConstants.MIN_LIQUIDITY_SOL, MarketScannerConstants.MAX_LIQUIDITY_SOL),
            "min_volume_sol": (MarketScannerConstants.MIN_VOLUME_24H_SOL, MarketScannerConstants.MAX_VOLUME_24H_SOL),
            "max_price_impact_percent": (MarketScannerConstants.MIN_PRICE_IMPACT_PERCENT, MarketScannerConstants.MAX_PRICE_IMPACT_PERCENT),
            "max_token_age_hours": (MarketScannerConstants.MIN_TOKEN_AGE_HOURS, MarketScannerConstants.MAX_TOKEN_AGE_HOURS),
            "min_holder_count": (MarketScannerConstants.MIN_HOLDER_COUNT, MarketScannerConstants.MAX_HOLDER_COUNT),
            "trending_tokens_limit": (MarketScannerConstants.MIN_TRENDING_TOKENS, MarketScannerConstants.MAX_TRENDING_TOKENS),
        }
    
    @staticmethod
    def get_vault_rules() -> Dict[str, tuple]:
        """Get vault parameter validation rules"""
        return {
            "profit_vault_percentage": (VaultConstants.MIN_PROFIT_VAULT_PERCENT, VaultConstants.MAX_PROFIT_VAULT_PERCENT),
            "min_vault_balance_sol": (VaultConstants.MIN_VAULT_BALANCE_SOL, VaultConstants.MAX_VAULT_BALANCE_SOL),
            "compounding_percentage": (VaultConstants.MIN_COMPOUNDING_PERCENT, VaultConstants.MAX_COMPOUNDING_PERCENT),
        }
    
    @staticmethod
    def get_network_rules() -> Dict[str, tuple]:
        """Get network parameter validation rules"""
        return {
            "rpc_timeout_seconds": (NetworkConstants.MIN_RPC_TIMEOUT_SECONDS, NetworkConstants.MAX_RPC_TIMEOUT_SECONDS),
            "max_retries": (NetworkConstants.MIN_RETRIES, NetworkConstants.MAX_RETRIES),
            "retry_delay_seconds": (NetworkConstants.MIN_RETRY_DELAY_SECONDS, NetworkConstants.MAX_RETRY_DELAY_SECONDS),
            "max_tx_fee_sol": (NetworkConstants.MIN_TX_FEE_SOL, NetworkConstants.MAX_TX_FEE_SOL),
        } 