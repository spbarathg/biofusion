from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel
import os
from pathlib import Path
import json
import asyncio
import multiprocessing

from worker_ant_v1.monitoring.secrets_manager import get_secrets_manager

class BotMode(Enum):
    SIMPLIFIED = "SIMPLIFIED"
    ADVANCED = "ADVANCED"

class TradingMode(Enum):
    LIVE = "LIVE"
    SIMULATION = "SIMULATION"

class SecurityLevel(Enum):
    STANDARD = "STANDARD"
    HIGH = "HIGH"
    MAXIMUM = "MAXIMUM"

class UnifiedConfig(BaseModel):
    """Core configuration for the trading system - SINGLE SOURCE OF TRUTH"""
    
    # Core system configuration
    bot_mode: BotMode = BotMode.SIMPLIFIED
    trading_mode: TradingMode
    security_level: SecurityLevel
    
    # Capital and position sizing
    initial_capital_sol: float = 1.5
    max_trade_size_sol: float
    min_trade_size_sol: float
    max_slippage_percent: float
    profit_target_percent: float
    stop_loss_percent: float
    
    # Three-Stage Mathematical Core Configuration
    acceptable_rel_threshold: float = 0.1  # Stage 1: Survival Filter
    hunt_threshold: float = 0.6           # Stage 2: Win-Rate Engine
    kelly_fraction: float = 0.25          # Stage 3: Growth Maximizer
    max_position_percent: float = 0.2
    
    # Trading behavior
    max_hold_time_hours: float = 4.0
    compound_rate: float = 0.8
    compound_threshold_sol: float = 0.2
    scan_interval_seconds: int = 30
    
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
    
    # Network/RPC configuration
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    
    # Database configuration
    timescaledb_host: str = "localhost"
    timescaledb_port: int = 5432
    timescaledb_database: str = "antbot_trading"
    timescaledb_username: str = "antbot"
    timescaledb_password: str = ""
    timescaledb_pool_min_size: int = 10
    timescaledb_pool_max_size: int = 20
    timescaledb_ssl_mode: str = "prefer"
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Wallet management configuration
    max_wallets: int = 10
    min_wallets: int = 5
    evolution_interval_hours: int = 24
    retirement_threshold: float = 0.3
    evolution_mutation_rate: float = 0.1
    wallet_encryption_enabled: bool = True
    wallet_password: str = ""
    encrypted_wallet_key: str = ""
    auto_create_wallet: bool = False
    
    # Process pool configuration
    process_pool_max_workers: int = 8
    process_pool_task_timeout: float = 30.0
    process_pool_queue_size: int = 1000
    process_pool_monitoring: bool = True
    
    # NATS message bus configuration
    nats_servers: str = "nats://localhost:4222"
    nats_connection_timeout: float = 10.0
    nats_reconnect_wait: float = 2.0
    nats_max_reconnect: int = 60
    nats_max_message_size: int = 1048576
    nats_enable_compression: bool = True
    nats_compression_threshold: int = 1024
    nats_enable_metrics: bool = True
    nats_stats_interval: float = 30.0
    nats_dead_letter_queue: str = "antbot.dlq"
    nats_enable_persistence: bool = False
    nats_stream_name: str = "ANTBOT_STREAM"
    nats_retention_policy: str = "limits"
    
    # Secrets manager configuration
    secrets_provider: str = "vault"
    vault_url: str = "http://localhost:8200"
    vault_token: Optional[str] = None
    vault_mount_path: str = "secret"
    secrets_cache_ttl: int = 3600
    secrets_cache_size: int = 1000
    secrets_allow_env_fallback: bool = True
    secrets_enable_cache_encryption: bool = True
    
    # Social signals and monitoring
    enable_social_signals: bool = False
    discord_bot_token: Optional[str] = None
    discord_channel_id: int = 0
    
    # High availability
    disable_ha: bool = False
    
    # Component identification
    component_id: Optional[str] = None
    
    class Config:
        use_enum_values = True

class UnifiedConfigManager:
    """Manager for unified trading configuration"""
    
    def __init__(self):
        self.config: Optional[UnifiedConfig] = None
        # Get the project root directory (where worker_ant_v1 is located)
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = self.project_root / "config"
        self.config_file = self.config_dir / "env.production"
        self.template_file = self.config_dir / "env.template"
        self._secrets_manager = None
        self._config_loaded = False
    
    async def _get_secrets_manager(self):
        """Get or create secrets manager instance"""
        if self._secrets_manager is None:
            self._secrets_manager = await get_secrets_manager()
        return self._secrets_manager
    
    async def _get_secret(self, key: str, default: str = '') -> str:
        """Get secret from secrets manager with fallback"""
        try:
            secrets_manager = await self._get_secrets_manager()
            return await secrets_manager.get_secret(key.lower())
        except Exception:
            # Fallback to environment variable for development/testing
            return os.getenv(key, default)
    
    def load_config(self):
        """Load configuration from environment file (synchronous version)"""
        if not self._config_loaded:
            # Try to load asynchronously if possible
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, we should use load_config_async instead
                    self._load_config_sync_fallback()
                else:
                    # Run async loading in a new event loop
                    asyncio.run(self.load_config_async())
            except RuntimeError:
                # No event loop, use synchronous fallback
                self._load_config_sync_fallback()
    
    def _load_config_sync_fallback(self):
        """Synchronous fallback configuration loading"""
        if not self.config_file.exists():
            # Check for unified template
            unified_template = self.config_dir / "unified.env.template"
            if unified_template.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_file}\n"
                    f"Please copy {unified_template} to {self.config_file} "
                    "and fill in your API keys and settings."
                )
            elif self.template_file.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_file}\n"
                    f"Please copy {self.template_file} to {self.config_file} "
                    "and fill in your API keys and settings."
                )
            else:
                raise FileNotFoundError(
                    f"Neither configuration file {self.config_file} "
                    f"nor template found."
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
        
        # Create config object with ALL environment variables - SINGLE SOURCE OF TRUTH
        self.config = UnifiedConfig(
            # Core system configuration
            bot_mode=os.getenv('BOT_MODE', 'SIMPLIFIED'),
            trading_mode=os.getenv('TRADING_MODE', 'SIMULATION'),
            security_level=os.getenv('SECURITY_LEVEL', 'HIGH'),
            
            # Capital and position sizing
            initial_capital_sol=float(os.getenv('INITIAL_CAPITAL_SOL', '1.5')),
            max_trade_size_sol=float(os.getenv('MAX_TRADE_SIZE_SOL', '0.5')),
            min_trade_size_sol=float(os.getenv('MIN_TRADE_SIZE_SOL', '0.01')),
            max_slippage_percent=float(os.getenv('MAX_SLIPPAGE_PERCENT', '5.0')),
            profit_target_percent=float(os.getenv('PROFIT_TARGET_PERCENT', '15.0')),
            stop_loss_percent=float(os.getenv('STOP_LOSS_PERCENT', '5.0')),
            
            # Three-Stage Mathematical Core Configuration
            acceptable_rel_threshold=float(os.getenv('ACCEPTABLE_REL_THRESHOLD', '0.1')),
            hunt_threshold=float(os.getenv('HUNT_THRESHOLD', '0.6')),
            kelly_fraction=float(os.getenv('KELLY_FRACTION', '0.25')),
            max_position_percent=float(os.getenv('MAX_POSITION_PERCENT', '0.2')),
            
            # Trading behavior
            max_hold_time_hours=float(os.getenv('MAX_HOLD_TIME_HOURS', '4.0')),
            compound_rate=float(os.getenv('COMPOUND_RATE', '0.8')),
            compound_threshold_sol=float(os.getenv('COMPOUND_THRESHOLD_SOL', '0.2')),
            scan_interval_seconds=int(os.getenv('SCAN_INTERVAL_SECONDS', '30')),
            
            # Safety settings
            max_daily_loss_sol=float(os.getenv('MAX_DAILY_LOSS_SOL', '1.0')),
            enable_kill_switch=os.getenv('ENABLE_KILL_SWITCH', 'true').lower() == 'true',
            emergency_stop_enabled=os.getenv('EMERGENCY_STOP_ENABLED', 'true').lower() == 'true',
            
            # API configuration
            helius_api_key=os.getenv('HELIUS_API_KEY', ''),
            solana_tracker_api_key=os.getenv('SOLANA_TRACKER_API_KEY', ''),
            jupiter_api_key=os.getenv('JUPITER_API_KEY', ''),
            raydium_api_key=os.getenv('RAYDIUM_API_KEY', ''),
            quicknode_rpc_url=os.getenv('QUICKNODE_RPC_URL'),
            dexscreener_api_key=os.getenv('DEXSCREENER_API_KEY'),
            birdeye_api_key=os.getenv('BIRDEYE_API_KEY'),
            
            # Network/RPC configuration
            solana_rpc_url=os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com'),
            
            # Database configuration
            timescaledb_host=os.getenv('TIMESCALEDB_HOST', 'localhost'),
            timescaledb_port=int(os.getenv('TIMESCALEDB_PORT', '5432')),
            timescaledb_database=os.getenv('TIMESCALEDB_DATABASE', 'antbot_trading'),
            timescaledb_username=os.getenv('TIMESCALEDB_USERNAME', 'antbot'),
            timescaledb_password=os.getenv('TIMESCALEDB_PASSWORD', ''),
            timescaledb_pool_min_size=int(os.getenv('TIMESCALEDB_POOL_MIN_SIZE', '10')),
            timescaledb_pool_max_size=int(os.getenv('TIMESCALEDB_POOL_MAX_SIZE', '20')),
            timescaledb_ssl_mode=os.getenv('TIMESCALEDB_SSL_MODE', 'prefer'),
            
            # Redis configuration
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', '6379')),
            redis_password=os.getenv('REDIS_PASSWORD'),
            redis_db=int(os.getenv('REDIS_DB', '0')),
            
            # Wallet management configuration
            max_wallets=int(os.getenv('MAX_WALLETS', '10')),
            min_wallets=int(os.getenv('MIN_WALLETS', '5')),
            evolution_interval_hours=int(os.getenv('EVOLUTION_INTERVAL_HOURS', '24')),
            retirement_threshold=float(os.getenv('RETIREMENT_THRESHOLD', '0.3')),
            evolution_mutation_rate=float(os.getenv('EVOLUTION_MUTATION_RATE', '0.1')),
            wallet_encryption_enabled=os.getenv('WALLET_ENCRYPTION_ENABLED', 'true').lower() == 'true',
            wallet_password=os.getenv('WALLET_PASSWORD', ''),
            encrypted_wallet_key=os.getenv('ENCRYPTED_WALLET_KEY', ''),
            auto_create_wallet=os.getenv('AUTO_CREATE_WALLET', 'false').lower() == 'true',
            
            # Process pool configuration
            process_pool_max_workers=int(os.getenv('PROCESS_POOL_MAX_WORKERS', str(multiprocessing.cpu_count()))),
            process_pool_task_timeout=float(os.getenv('PROCESS_POOL_TASK_TIMEOUT', '30.0')),
            process_pool_queue_size=int(os.getenv('PROCESS_POOL_QUEUE_SIZE', '1000')),
            process_pool_monitoring=os.getenv('PROCESS_POOL_MONITORING', 'true').lower() == 'true',
            
            # NATS message bus configuration
            nats_servers=os.getenv('NATS_SERVERS', 'nats://localhost:4222'),
            nats_connection_timeout=float(os.getenv('NATS_CONNECTION_TIMEOUT', '10.0')),
            nats_reconnect_wait=float(os.getenv('NATS_RECONNECT_WAIT', '2.0')),
            nats_max_reconnect=int(os.getenv('NATS_MAX_RECONNECT', '60')),
            nats_max_message_size=int(os.getenv('NATS_MAX_MESSAGE_SIZE', '1048576')),
            nats_enable_compression=os.getenv('NATS_ENABLE_COMPRESSION', 'true').lower() == 'true',
            nats_compression_threshold=int(os.getenv('NATS_COMPRESSION_THRESHOLD', '1024')),
            nats_enable_metrics=os.getenv('NATS_ENABLE_METRICS', 'true').lower() == 'true',
            nats_stats_interval=float(os.getenv('NATS_STATS_INTERVAL', '30.0')),
            nats_dead_letter_queue=os.getenv('NATS_DEAD_LETTER_QUEUE', 'antbot.dlq'),
            nats_enable_persistence=os.getenv('NATS_ENABLE_PERSISTENCE', 'false').lower() == 'true',
            nats_stream_name=os.getenv('NATS_STREAM_NAME', 'ANTBOT_STREAM'),
            nats_retention_policy=os.getenv('NATS_RETENTION_POLICY', 'limits'),
            
            # Secrets manager configuration
            secrets_provider=os.getenv('SECRETS_PROVIDER', 'vault'),
            vault_url=os.getenv('VAULT_URL', 'http://localhost:8200'),
            vault_token=os.getenv('VAULT_TOKEN'),
            vault_mount_path=os.getenv('VAULT_MOUNT_PATH', 'secret'),
            secrets_cache_ttl=int(os.getenv('SECRETS_CACHE_TTL', '3600')),
            secrets_cache_size=int(os.getenv('SECRETS_CACHE_SIZE', '1000')),
            secrets_allow_env_fallback=os.getenv('SECRETS_ALLOW_ENV_FALLBACK', 'true').lower() == 'true',
            secrets_enable_cache_encryption=os.getenv('SECRETS_ENABLE_CACHE_ENCRYPTION', 'true').lower() == 'true',
            
            # Social signals and monitoring
            enable_social_signals=os.getenv('ENABLE_SOCIAL_SIGNALS', 'false').lower() == 'true',
            discord_bot_token=os.getenv('DISCORD_BOT_TOKEN'),
            discord_channel_id=int(os.getenv('DISCORD_CHANNEL_ID', '0')),
            
            # High availability
            disable_ha=os.getenv('DISABLE_HA', 'false').lower() == 'true',
            
            # Component identification
            component_id=os.getenv('COMPONENT_ID')
        )
        self._config_loaded = True
    
    async def load_config_async(self):
        """Load configuration asynchronously using secrets manager"""
        if self._config_loaded:
            return
            
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
        
        # Load environment variables from file for non-sensitive config
        config_data = {}
        with open(self.config_file, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key and value and '\x00' not in value:
                        os.environ[key] = value
                        config_data[key.lower()] = value
        
        # Load API keys from secrets manager
        try:
            helius_key = await self._get_secret('HELIUS_API_KEY')
            solana_tracker_key = await self._get_secret('SOLANA_TRACKER_API_KEY')
            jupiter_key = await self._get_secret('JUPITER_API_KEY')
            raydium_key = await self._get_secret('RAYDIUM_API_KEY')
            quicknode_url = await self._get_secret('QUICKNODE_RPC_URL')
            dexscreener_key = await self._get_secret('DEXSCREENER_API_KEY')
            birdeye_key = await self._get_secret('BIRDEYE_API_KEY')
        except Exception as e:
            # Log warning and use environment fallback
            print(f"Warning: Could not load secrets from manager: {e}. Using environment variables.")
            helius_key = os.getenv('HELIUS_API_KEY', '')
            solana_tracker_key = os.getenv('SOLANA_TRACKER_API_KEY', '')
            jupiter_key = os.getenv('JUPITER_API_KEY', '')
            raydium_key = os.getenv('RAYDIUM_API_KEY', '')
            quicknode_url = os.getenv('QUICKNODE_RPC_URL')
            dexscreener_key = os.getenv('DEXSCREENER_API_KEY')
            birdeye_key = os.getenv('BIRDEYE_API_KEY')
        
        # Create config object with ALL environment variables - SINGLE SOURCE OF TRUTH
        self.config = UnifiedConfig(
            # Core system configuration
            bot_mode=os.getenv('BOT_MODE', 'SIMPLIFIED'),
            trading_mode=os.getenv('TRADING_MODE', 'SIMULATION'),
            security_level=os.getenv('SECURITY_LEVEL', 'HIGH'),
            
            # Capital and position sizing
            initial_capital_sol=float(os.getenv('INITIAL_CAPITAL_SOL', '1.5')),
            max_trade_size_sol=float(os.getenv('MAX_TRADE_SIZE_SOL', '0.5')),
            min_trade_size_sol=float(os.getenv('MIN_TRADE_SIZE_SOL', '0.01')),
            max_slippage_percent=float(os.getenv('MAX_SLIPPAGE_PERCENT', '5.0')),
            profit_target_percent=float(os.getenv('PROFIT_TARGET_PERCENT', '15.0')),
            stop_loss_percent=float(os.getenv('STOP_LOSS_PERCENT', '5.0')),
            
            # Three-Stage Mathematical Core Configuration
            acceptable_rel_threshold=float(os.getenv('ACCEPTABLE_REL_THRESHOLD', '0.1')),
            hunt_threshold=float(os.getenv('HUNT_THRESHOLD', '0.6')),
            kelly_fraction=float(os.getenv('KELLY_FRACTION', '0.25')),
            max_position_percent=float(os.getenv('MAX_POSITION_PERCENT', '0.2')),
            
            # Trading behavior
            max_hold_time_hours=float(os.getenv('MAX_HOLD_TIME_HOURS', '4.0')),
            compound_rate=float(os.getenv('COMPOUND_RATE', '0.8')),
            compound_threshold_sol=float(os.getenv('COMPOUND_THRESHOLD_SOL', '0.2')),
            scan_interval_seconds=int(os.getenv('SCAN_INTERVAL_SECONDS', '30')),
            
            # Safety settings
            max_daily_loss_sol=float(os.getenv('MAX_DAILY_LOSS_SOL', '1.0')),
            enable_kill_switch=os.getenv('ENABLE_KILL_SWITCH', 'true').lower() == 'true',
            emergency_stop_enabled=os.getenv('EMERGENCY_STOP_ENABLED', 'true').lower() == 'true',
            
            # API configuration (from secrets manager)
            helius_api_key=helius_key,
            solana_tracker_api_key=solana_tracker_key,
            jupiter_api_key=jupiter_key,
            raydium_api_key=raydium_key,
            quicknode_rpc_url=quicknode_url,
            dexscreener_api_key=dexscreener_key,
            birdeye_api_key=birdeye_key,
            
            # Network/RPC configuration
            solana_rpc_url=os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com'),
            
            # Database configuration
            timescaledb_host=os.getenv('TIMESCALEDB_HOST', 'localhost'),
            timescaledb_port=int(os.getenv('TIMESCALEDB_PORT', '5432')),
            timescaledb_database=os.getenv('TIMESCALEDB_DATABASE', 'antbot_trading'),
            timescaledb_username=os.getenv('TIMESCALEDB_USERNAME', 'antbot'),
            timescaledb_password=os.getenv('TIMESCALEDB_PASSWORD', ''),
            timescaledb_pool_min_size=int(os.getenv('TIMESCALEDB_POOL_MIN_SIZE', '10')),
            timescaledb_pool_max_size=int(os.getenv('TIMESCALEDB_POOL_MAX_SIZE', '20')),
            timescaledb_ssl_mode=os.getenv('TIMESCALEDB_SSL_MODE', 'prefer'),
            
            # Redis configuration
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', '6379')),
            redis_password=os.getenv('REDIS_PASSWORD'),
            redis_db=int(os.getenv('REDIS_DB', '0')),
            
            # Wallet management configuration
            max_wallets=int(os.getenv('MAX_WALLETS', '10')),
            min_wallets=int(os.getenv('MIN_WALLETS', '5')),
            evolution_interval_hours=int(os.getenv('EVOLUTION_INTERVAL_HOURS', '24')),
            retirement_threshold=float(os.getenv('RETIREMENT_THRESHOLD', '0.3')),
            evolution_mutation_rate=float(os.getenv('EVOLUTION_MUTATION_RATE', '0.1')),
            wallet_encryption_enabled=os.getenv('WALLET_ENCRYPTION_ENABLED', 'true').lower() == 'true',
            wallet_password=os.getenv('WALLET_PASSWORD', ''),
            encrypted_wallet_key=os.getenv('ENCRYPTED_WALLET_KEY', ''),
            auto_create_wallet=os.getenv('AUTO_CREATE_WALLET', 'false').lower() == 'true',
            
            # Process pool configuration
            process_pool_max_workers=int(os.getenv('PROCESS_POOL_MAX_WORKERS', str(multiprocessing.cpu_count()))),
            process_pool_task_timeout=float(os.getenv('PROCESS_POOL_TASK_TIMEOUT', '30.0')),
            process_pool_queue_size=int(os.getenv('PROCESS_POOL_QUEUE_SIZE', '1000')),
            process_pool_monitoring=os.getenv('PROCESS_POOL_MONITORING', 'true').lower() == 'true',
            
            # NATS message bus configuration
            nats_servers=os.getenv('NATS_SERVERS', 'nats://localhost:4222'),
            nats_connection_timeout=float(os.getenv('NATS_CONNECTION_TIMEOUT', '10.0')),
            nats_reconnect_wait=float(os.getenv('NATS_RECONNECT_WAIT', '2.0')),
            nats_max_reconnect=int(os.getenv('NATS_MAX_RECONNECT', '60')),
            nats_max_message_size=int(os.getenv('NATS_MAX_MESSAGE_SIZE', '1048576')),
            nats_enable_compression=os.getenv('NATS_ENABLE_COMPRESSION', 'true').lower() == 'true',
            nats_compression_threshold=int(os.getenv('NATS_COMPRESSION_THRESHOLD', '1024')),
            nats_enable_metrics=os.getenv('NATS_ENABLE_METRICS', 'true').lower() == 'true',
            nats_stats_interval=float(os.getenv('NATS_STATS_INTERVAL', '30.0')),
            nats_dead_letter_queue=os.getenv('NATS_DEAD_LETTER_QUEUE', 'antbot.dlq'),
            nats_enable_persistence=os.getenv('NATS_ENABLE_PERSISTENCE', 'false').lower() == 'true',
            nats_stream_name=os.getenv('NATS_STREAM_NAME', 'ANTBOT_STREAM'),
            nats_retention_policy=os.getenv('NATS_RETENTION_POLICY', 'limits'),
            
            # Secrets manager configuration
            secrets_provider=os.getenv('SECRETS_PROVIDER', 'vault'),
            vault_url=os.getenv('VAULT_URL', 'http://localhost:8200'),
            vault_token=os.getenv('VAULT_TOKEN'),
            vault_mount_path=os.getenv('VAULT_MOUNT_PATH', 'secret'),
            secrets_cache_ttl=int(os.getenv('SECRETS_CACHE_TTL', '3600')),
            secrets_cache_size=int(os.getenv('SECRETS_CACHE_SIZE', '1000')),
            secrets_allow_env_fallback=os.getenv('SECRETS_ALLOW_ENV_FALLBACK', 'true').lower() == 'true',
            secrets_enable_cache_encryption=os.getenv('SECRETS_ENABLE_CACHE_ENCRYPTION', 'true').lower() == 'true',
            
            # Social signals and monitoring
            enable_social_signals=os.getenv('ENABLE_SOCIAL_SIGNALS', 'false').lower() == 'true',
            discord_bot_token=os.getenv('DISCORD_BOT_TOKEN'),
            discord_channel_id=int(os.getenv('DISCORD_CHANNEL_ID', '0')),
            
            # High availability
            disable_ha=os.getenv('DISABLE_HA', 'false').lower() == 'true',
            
            # Component identification
            component_id=os.getenv('COMPONENT_ID')
        )
        self._config_loaded = True
    
    def get_config(self) -> UnifiedConfig:
        """Get current configuration (synchronous)"""
        if not self.config:
            self.load_config()
        return self.config
    
    async def get_config_async(self) -> UnifiedConfig:
        """Get current configuration (asynchronous with secrets manager)"""
        if not self.config:
            await self.load_config_async()
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
    """Get current trading configuration (synchronous)"""
    manager = UnifiedConfigManager()
    return manager.get_config()

async def get_trading_config_async() -> UnifiedConfig:
    """Get current trading configuration (asynchronous with secrets manager)"""
    manager = UnifiedConfigManager()
    return await manager.get_config_async()

def get_security_config() -> Dict[str, any]:
    """Get security configuration (synchronous)"""
    config = get_trading_config()
    return {
        'security_level': config.security_level,
        'enable_kill_switch': config.enable_kill_switch,
        'emergency_stop_enabled': config.emergency_stop_enabled,
        'max_daily_loss_sol': config.max_daily_loss_sol
    }

async def get_security_config_async() -> Dict[str, any]:
    """Get security configuration (asynchronous)"""
    config = await get_trading_config_async()
    return {
        'security_level': config.security_level,
        'enable_kill_switch': config.enable_kill_switch,
        'emergency_stop_enabled': config.emergency_stop_enabled,
        'max_daily_loss_sol': config.max_daily_loss_sol
    }

def get_network_config() -> Dict[str, any]:
    """Get network configuration (synchronous)"""
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

async def get_network_config_async() -> Dict[str, any]:
    """Get network configuration (asynchronous with secrets manager)"""
    config = await get_trading_config_async()
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

def get_redis_config() -> Dict[str, any]:
    """Get Redis configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG"""
    config = get_trading_config()
    return {
        'host': config.redis_host,
        'port': config.redis_port,
        'password': config.redis_password,
        'db': config.redis_db
    }

def get_api_config() -> Dict[str, str]:
    """Get API configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG"""
    config = get_trading_config()
    return {
        'birdeye_api_key': config.birdeye_api_key or '',
        'jupiter_api_key': config.jupiter_api_key or '',
        'helius_api_key': config.helius_api_key or '',
        'dexscreener_api_key': config.dexscreener_api_key or '',
        'solana_tracker_api_key': config.solana_tracker_api_key or '',
        'raydium_api_key': config.raydium_api_key or ''
    }

def get_network_rpc_url() -> str:
    """Get Solana RPC URL - CANONICAL ACCESS THROUGH UNIFIED CONFIG"""
    config = get_trading_config()
    return config.solana_rpc_url

def get_social_signals_config() -> Dict[str, any]:
    """Get social signals configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG"""
    config = get_trading_config()
    return {
        'enable_social_signals': config.enable_social_signals,
        'discord_bot_token': config.discord_bot_token,
        'discord_channel_id': config.discord_channel_id
    }

def get_ha_config() -> Dict[str, any]:
    """Get high availability configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG"""
    config = get_trading_config()
    return {
        'disable_ha': config.disable_ha
    }

def get_config_manager() -> UnifiedConfigManager:
    """Get global config manager instance"""
    return UnifiedConfigManager()

def get_wallet_config() -> Dict[str, any]:
    """Get wallet configuration - CANONICAL ACCESS THROUGH UNIFIED CONFIG"""
    config = get_trading_config()  # Force through unified config
    return {
        'solana_rpc_url': config.solana_rpc_url,
        'max_wallets': config.max_wallets,
        'min_wallets': config.min_wallets,
        'evolution_interval_hours': config.evolution_interval_hours,
        'retirement_threshold': config.retirement_threshold,
        'evolution_mutation_rate': config.evolution_mutation_rate,
        'wallet_encryption_enabled': config.wallet_encryption_enabled,
        'wallet_password': config.wallet_password,
        'encrypted_wallet_key': config.encrypted_wallet_key,
        'auto_create_wallet': config.auto_create_wallet
    } 