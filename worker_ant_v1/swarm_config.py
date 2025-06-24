"""
Hyper-Compounding Swarm Configuration
===================================

Configuration for the 3-hour flywheel compounding trading swarm.
Turns $1,000 into $20,000+ in 72 hours through autonomous replication.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

class SwarmMode(Enum):
    GENESIS = "genesis"           # Single ant start
    COMPOUNDING = "compounding"   # Active swarm growth
    AGGRESSIVE = "aggressive"     # Maximum expansion mode
    DEFENSIVE = "defensive"       # Risk reduction mode
    HARVEST = "harvest"           # Capital extraction mode

@dataclass
class WorkerAntConfig:
    """Individual Worker Ant configuration for hyper-compounding"""
    
    # === HYPER-AGGRESSIVE TRADING ===
    starting_capital_usd: float = 1000.0      # $1,000 genesis capital
    trade_size_usd_min: float = 100.0         # $100 minimum trade
    trade_size_usd_max: float = 150.0         # $150 maximum trade
    trades_per_hour_target: int = 25           # 25 trades/hour per ant
    
    # === PROFIT TARGETS (AGGRESSIVE) ===
    profit_target_percent: float = 3.0        # 3% profit per trade
    profit_target_min: float = 2.0            # Minimum 2%
    profit_target_max: float = 5.0            # Maximum 5%
    
    # === RISK MANAGEMENT ===
    stop_loss_percent: float = 7.0            # 5-10% stop loss range
    stop_loss_min: float = 5.0
    stop_loss_max: float = 10.0
    
    # === SPEED REQUIREMENTS ===
    entry_timeout_ms: int = 500               # 500ms max entry time
    trade_duration_max_seconds: int = 120     # <2 minutes per trade
    slippage_threshold_percent: float = 1.25  # 1-1.5% slippage cap
    
    # === PERFORMANCE THRESHOLDS ===
    target_win_rate: float = 67.5             # 65-70% win rate target
    min_win_rate_pause: float = 50.0          # Pause if <50% over 20 trades
    performance_window_trades: int = 20       # Performance evaluation window
    
    # === ADAPTIVE SCALING ===
    scale_up_after_wins: int = 3              # Scale up 10% after 3 wins
    scale_up_multiplier: float = 1.10         # 10% trade size increase
    max_scale_multiplier: float = 2.0         # Maximum 2x scaling
    
    # === DAILY OPERATION ===
    active_hours_per_day: int = 20            # 20 hours active trading
    rest_period_hours: int = 4                # 4 hours rest/maintenance


@dataclass
class CompoundingConfig:
    """3-Hour flywheel compounding configuration"""
    
    # === COMPOUNDING CYCLE ===
    compounding_interval_hours: int = 3       # Every 3 hours
    split_threshold_multiplier: float = 2.0   # Split when 2x capital
    merge_threshold_multiplier: float = 0.8   # Merge/pause when <80% capital
    
    # === ANT LIFECYCLE ===
    max_ants_total: int = 16                  # Maximum 16 ants in swarm
    min_ants_active: int = 1                  # Minimum 1 ant always active
    ant_death_threshold: float = 0.5          # Kill ant if <50% starting capital
    
    # === CAPITAL ALLOCATION ===
    split_capital_ratio: float = 0.5          # 50/50 split on replication
    reserve_capital_percent: float = 10.0     # 10% hedging reserve
    emergency_reserve_percent: float = 5.0    # 5% emergency fund
    
    # === TARGET GOALS ===
    target_capital_72h: float = 20000.0       # $20K target in 72 hours
    target_ants_72h: int = 12                 # 12-16 ants target
    exponential_growth_rate: float = 1.44     # ~44% growth per 3-hour cycle


@dataclass
class SwarmControllerConfig:
    """Queen Bot swarm controller configuration"""
    
    # === SWARM MANAGEMENT ===
    max_concurrent_operations: int = 5        # Max simultaneous operations
    health_check_interval_seconds: int = 30   # Health check frequency
    rebalancing_interval_hours: int = 6       # Capital rebalancing
    
    # === SAFETY SYSTEMS ===
    global_kill_switch_drawdown: float = 30.0 # Kill if >30% total drawdown
    max_trades_per_second: float = 2.0        # Rate limiting
    gas_fee_threshold_percent: float = 10.0   # Throttle if gas >10% profit
    
    # === WALLET MANAGEMENT ===
    wallets_per_ant: int = 1                  # 1 wallet per ant initially
    wallet_rotation_interval_hours: int = 24  # Rotate wallets daily
    max_wallet_exposure_usd: float = 5000.0   # Max $5K per wallet
    
    # === MONITORING & ALERTS ===
    enable_discord_alerts: bool = True
    enable_performance_tracking: bool = True
    enable_real_time_dashboard: bool = True
    
    # === SCALING TRIGGERS ===
    auto_scale_threshold_usd: float = 5000.0  # Auto-scale at $5K milestones
    scale_mode_aggressive_threshold: float = 15000.0  # Aggressive mode at $15K
    scale_mode_defensive_threshold: float = 25000.0   # Defensive mode at $25K


@dataclass
class TradingOptimizationConfig:
    """Advanced trading optimization for swarm"""
    
    # === TOKEN FILTERING (HYPER-SELECTIVE) ===
    min_liquidity_usd: float = 50000.0        # $50K minimum liquidity
    max_liquidity_usd: float = 2000000.0      # $2M maximum (avoid manipulation)
    min_volume_24h_usd: float = 500000.0      # $500K daily volume
    
    # === POOL AGE OPTIMIZATION ===
    min_pool_age_seconds: int = 60             # 1 minute minimum
    max_pool_age_seconds: int = 900            # 15 minutes maximum
    sweet_spot_age_seconds: int = 300          # 5 minutes sweet spot
    
    # === HOLDER ANALYSIS ===
    max_top_holder_percent: float = 15.0      # Top holder <15%
    min_holder_count: int = 200               # Minimum 200 holders
    max_whale_concentration: float = 40.0     # Top 10 holders <40%
    
    # === TECHNICAL INDICATORS ===
    enable_momentum_filter: bool = True       # Momentum-based filtering
    enable_volume_spike_detection: bool = True # Volume spike trading
    enable_social_sentiment: bool = True      # Social sentiment scoring
    
    # === EXECUTION OPTIMIZATION ===
    use_jupiter_v6: bool = True               # Latest Jupiter aggregator
    enable_priority_fees: bool = True         # Priority fee optimization
    max_priority_fee_sol: float = 0.01        # Max 0.01 SOL priority fee
    
    # === MEV PROTECTION ===
    enable_mev_protection: bool = True        # Anti-MEV measures
    sandwich_protection: bool = True          # Sandwich attack protection
    private_mempool: bool = True              # Private mempool usage


@dataclass
class AlertConfig:
    """Alert and notification configuration"""
    
    # === DISCORD INTEGRATION ===
    discord_webhook_url: Optional[str] = os.getenv("DISCORD_WEBHOOK_URL")
    enable_ant_birth_alerts: bool = True      # New ant spawned
    enable_ant_death_alerts: bool = True      # Ant death/merge
    enable_milestone_alerts: bool = True      # Capital milestones
    enable_top_gainer_alerts: bool = True     # Best performing trades
    
    # === MILESTONE THRESHOLDS ===
    milestone_thresholds_usd: List[float] = field(default_factory=lambda: [
        2000, 5000, 10000, 15000, 20000, 25000, 50000
    ])
    
    # === PERFORMANCE ALERTS ===
    alert_on_low_win_rate: bool = True        # Alert on poor performance
    alert_on_high_gas_fees: bool = True       # Alert on expensive trades
    alert_on_system_errors: bool = True       # Alert on technical issues
    
    # === TELEGRAM INTEGRATION ===
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")


@dataclass
class AdvancedStrategyConfig:
    """Advanced trading strategies for different ant types"""
    
    # === STRATEGY TYPES ===
    enable_sniper_ants: bool = True           # Ultra-fast entry ants
    enable_confirmation_ants: bool = True     # Confirmation-based ants
    enable_dip_buyer_ants: bool = True        # Dip buying specialists
    enable_momentum_ants: bool = True         # Momentum trading ants
    
    # === STRATEGY ALLOCATION ===
    sniper_ant_ratio: float = 0.4             # 40% sniper ants
    confirmation_ant_ratio: float = 0.3       # 30% confirmation ants
    dip_buyer_ant_ratio: float = 0.2          # 20% dip buyers
    momentum_ant_ratio: float = 0.1           # 10% momentum traders
    
    # === TIERED EXECUTION ===
    enable_staggered_entries: bool = True     # Staggered entry execution
    entry_tiers: int = 3                      # 3-tier entry system
    tier_delay_ms: int = 100                  # 100ms between tiers


@dataclass
class AggressiveMemeStrategy:
    """Aggressive memecoin trading strategy configuration"""
    
    # === AGGRESSIVE TRADING PARAMETERS ===
    trades_per_day_target: int = 15           # 10-20 trades/day target
    capital_allocation_percent: float = 12.5  # 10-15% per trade (12.5% default)
    profit_target_min: float = 5.0           # 5% minimum profit target
    profit_target_max: float = 10.0          # 10% maximum profit target
    max_holding_hours: int = 48               # Up to 48 hours for high conviction
    
    # === RISK MANAGEMENT ===
    stop_loss_percent: float = 2.5           # 2-3% tight stop losses
    daily_loss_limit_percent: float = 10.0   # 10% daily loss cap
    trading_pause_hours: int = 24             # 24h pause after loss limit
    basket_size: int = 7                     # 5-10 memecoins (7 default)
    
    # === SENTIMENT ANALYSIS ===
    enable_twitter_sentiment: bool = True     # Twitter sentiment analysis
    enable_reddit_sentiment: bool = True      # Reddit sentiment analysis
    enable_telegram_monitoring: bool = True   # Telegram channel monitoring
    sentiment_weight: float = 0.4            # 40% sentiment weight in decisions
    
    # === TECHNICAL INDICATORS ===
    enable_rsi_signals: bool = True           # RSI for entry/exit
    rsi_oversold_threshold: float = 30.0      # RSI oversold level
    rsi_overbought_threshold: float = 70.0    # RSI overbought level
    enable_macd_signals: bool = True          # MACD for momentum
    macd_signal_threshold: float = 0.1        # MACD signal sensitivity
    
    # === MACHINE LEARNING ===
    enable_lstm_prediction: bool = True       # LSTM price prediction
    enable_reinforcement_learning: bool = True # RL for strategy optimization
    prediction_weight: float = 0.3           # 30% ML prediction weight
    model_retrain_hours: int = 24            # Retrain models every 24h
    
    # === DATA SOURCES ===
    price_data_sources: List[str] = field(default_factory=lambda: [
        "binance", "coinbase", "dexscreener", "birdeye"
    ])
    sentiment_data_sources: List[str] = field(default_factory=lambda: [
        "twitter", "reddit", "telegram", "discord"
    ])
    
    # === PERFORMANCE OPTIMIZATION ===
    parallel_analysis_enabled: bool = True    # Parallel sentiment/technical analysis
    real_time_updates_ms: int = 5000          # 5-second update intervals
    gpu_acceleration: bool = True             # GPU for ML computations
    

@dataclass
class MLModelConfig:
    """Machine Learning model configuration"""
    
    # === LSTM CONFIGURATION ===
    lstm_sequence_length: int = 60            # 60-period lookback
    lstm_hidden_layers: int = 3               # 3 hidden layers
    lstm_units_per_layer: int = 50            # 50 LSTM units per layer
    lstm_dropout_rate: float = 0.2            # 20% dropout for regularization
    
    # === REINFORCEMENT LEARNING ===
    rl_environment: str = "continuous"        # Continuous action space
    rl_algorithm: str = "ppo"                 # Proximal Policy Optimization
    rl_training_episodes: int = 1000          # Training episodes
    rl_learning_rate: float = 0.0003          # Learning rate
    
    # === FEATURE ENGINEERING ===
    price_features: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume", "market_cap"
    ])
    technical_features: List[str] = field(default_factory=lambda: [
        "rsi", "macd", "bb_upper", "bb_lower", "ema_12", "ema_26"
    ])
    sentiment_features: List[str] = field(default_factory=lambda: [
        "twitter_sentiment", "reddit_sentiment", "mention_volume", "social_dominance"
    ])
    
    # === TRAINING CONFIGURATION ===
    training_data_days: int = 90              # 3 months training data
    validation_split: float = 0.2             # 20% validation split
    batch_size: int = 32                      # Training batch size
    epochs: int = 100                         # Training epochs
    early_stopping_patience: int = 10         # Early stopping patience


# === GLOBAL CONFIGURATION INSTANCES ===
worker_ant_config = WorkerAntConfig()
compounding_config = CompoundingConfig()
swarm_controller_config = SwarmControllerConfig()
trading_optimization_config = TradingOptimizationConfig()
alert_config = AlertConfig()
advanced_strategy_config = AdvancedStrategyConfig()


def load_swarm_env_config():
    """Load swarm configuration from environment variables"""
    
    # === CAPITAL SETTINGS ===
    if os.getenv("STARTING_CAPITAL_USD"):
        worker_ant_config.starting_capital_usd = float(os.getenv("STARTING_CAPITAL_USD"))
    
    if os.getenv("TRADE_SIZE_USD_MIN"):
        worker_ant_config.trade_size_usd_min = float(os.getenv("TRADE_SIZE_USD_MIN"))
        
    if os.getenv("TRADE_SIZE_USD_MAX"):
        worker_ant_config.trade_size_usd_max = float(os.getenv("TRADE_SIZE_USD_MAX"))
    
    # === PERFORMANCE TARGETS ===
    if os.getenv("TRADES_PER_HOUR"):
        worker_ant_config.trades_per_hour_target = int(os.getenv("TRADES_PER_HOUR"))
        
    if os.getenv("PROFIT_TARGET_PERCENT"):
        worker_ant_config.profit_target_percent = float(os.getenv("PROFIT_TARGET_PERCENT"))
    
    # === COMPOUNDING SETTINGS ===
    if os.getenv("COMPOUNDING_INTERVAL_HOURS"):
        compounding_config.compounding_interval_hours = int(os.getenv("COMPOUNDING_INTERVAL_HOURS"))
        
    if os.getenv("MAX_ANTS_TOTAL"):
        compounding_config.max_ants_total = int(os.getenv("MAX_ANTS_TOTAL"))
    
    # === ALERT SETTINGS ===
    if os.getenv("DISCORD_WEBHOOK_URL"):
        alert_config.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL")


def validate_swarm_config():
    """Validate swarm configuration for safety and performance"""
    errors = []
    
    # === CAPITAL VALIDATION ===
    if worker_ant_config.starting_capital_usd < 500:
        errors.append("Starting capital too low (<$500)")
        
    if worker_ant_config.trade_size_usd_max > worker_ant_config.starting_capital_usd * 0.2:
        errors.append("Trade size too large (>20% of capital)")
    
    # === PERFORMANCE VALIDATION ===
    if worker_ant_config.trades_per_hour_target > 30:
        errors.append("Trades per hour too aggressive (>30/hour risk)")
        
    if worker_ant_config.target_win_rate > 80:
        errors.append("Target win rate unrealistic (>80%)")
    
    # === SWARM VALIDATION ===
    if compounding_config.max_ants_total > 20:
        errors.append("Too many ants (>20 may cause issues)")
        
    if swarm_controller_config.max_trades_per_second > 5:
        errors.append("Trade rate too high (RPC rate limit risk)")
    
    # === SAFETY VALIDATION ===
    if swarm_controller_config.global_kill_switch_drawdown > 50:
        errors.append("Kill switch threshold too high (>50%)")
    
    if errors:
        raise ValueError(f"Swarm configuration errors: {', '.join(errors)}")
    
    return True


def get_swarm_status_summary() -> Dict:
    """Get comprehensive swarm configuration summary"""
    return {
        "genesis_capital": worker_ant_config.starting_capital_usd,
        "target_72h": compounding_config.target_capital_72h,
        "trades_per_hour": worker_ant_config.trades_per_hour_target,
        "profit_target": worker_ant_config.profit_target_percent,
        "compounding_interval": compounding_config.compounding_interval_hours,
        "max_ants": compounding_config.max_ants_total,
        "split_threshold": compounding_config.split_threshold_multiplier,
        "safety_drawdown": swarm_controller_config.global_kill_switch_drawdown
    } 

# === GLOBAL INSTANCES ===
aggressive_meme_strategy = AggressiveMemeStrategy()
ml_model_config = MLModelConfig() 