"""
UNIFIED DATA SCHEMAS - SINGLE SOURCE OF TRUTH
===========================================

Centralized data structures and schemas used across the entire system.
Eliminates circular dependencies and ensures data consistency.
"""

import hashlib
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class TradeType(Enum):
    """Trade operation types"""
    BUY = "BUY"
    SELL = "SELL"
    SWAP = "SWAP"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "PENDING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ValidationLevel(Enum):
    """Validation issue severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class TradeRecord:
    """Unified trade record used across logging, database, and analytics systems"""
    
    # Core identification
    timestamp: datetime
    trade_id: str
    session_id: str = field(default_factory=lambda: secrets.token_hex(8))
    wallet_id: str = ""
    
    # Token information
    token_address: str = ""
    token_symbol: str = ""
    token_name: Optional[str] = None
    
    # Trade execution details
    trade_type: str = TradeType.BUY.value  # BUY, SELL, SWAP
    success: bool = False
    amount_sol: float = 0.0
    amount_tokens: float = 0.0
    price: float = 0.0
    slippage_percent: float = 0.0
    
    # Performance metrics
    latency_ms: int = 0
    gas_cost_sol: float = 0.0
    rpc_cost_sol: float = 0.0
    api_cost_sol: float = 0.0
    
    # P&L metrics (for sell orders)
    profit_loss_sol: Optional[float] = None
    profit_loss_percent: Optional[float] = None
    hold_time_seconds: Optional[int] = None
    
    # Technical details (secured)
    tx_signature: Optional[str] = None
    tx_signature_hash: Optional[str] = None
    retry_count: int = 0
    exit_reason: Optional[str] = None
    error_message: Optional[str] = None
    
    # Market context (for ML training)
    market_cap_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    price_change_24h_percent: Optional[float] = None
    
    # AI Signal Snapshot for Naive Bayes Training
    signal_snapshot: Optional[Dict[str, Any]] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Generate trade_id if not provided
        if not self.trade_id:
            self.trade_id = str(uuid.uuid4())
        
        # Auto-mask sensitive data if tx_signature is present
        if self.tx_signature and not self.tx_signature_hash:
            self.tx_signature_hash = hashlib.sha256(
                self.tx_signature.encode()
            ).hexdigest()[:16]
    
    def mask_sensitive_data(self):
        """Mask sensitive transaction data for secure logging"""
        if self.tx_signature:
            self.tx_signature_hash = hashlib.sha256(
                self.tx_signature.encode()
            ).hexdigest()[:16]
            self.tx_signature = None  # Remove original signature
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecord':
        """Create TradeRecord from dictionary"""
        # Handle datetime conversion
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        
        return cls(**data)


@dataclass
class SystemEvent:
    """Unified system event record for monitoring and alerting"""
    
    timestamp: datetime
    event_id: str
    event_type: str
    component: str
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    event_data: Dict[str, Any]
    session_id: Optional[str] = None
    wallet_id: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())


@dataclass
class PerformanceMetric:
    """Performance metric for system monitoring"""
    
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    component: str
    labels: Dict[str, str]
    aggregation_period: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MarketOpportunity:
    """Market trading opportunity data structure"""
    
    token_address: str
    token_symbol: str
    token_name: str
    detected_at: datetime
    opportunity_score: float
    risk_level: str
    expected_profit: float
    recommended_position_size: float
    max_slippage_percent: float
    market_data: Dict[str, Any]
    analysis_metadata: Optional[Dict[str, Any]] = None


@dataclass
class TokenMarketData:
    """Token market data structure"""
    
    token_address: str
    symbol: str
    name: str
    price: float
    volume_24h: float
    liquidity: float
    price_change_1h: float
    price_change_24h: float
    holder_count: int
    age_hours: float
    market_cap_usd: Optional[float] = None
    last_updated: Optional[datetime] = None


@dataclass
class ValidationResult:
    """System validation result"""
    
    passed: bool
    issues: List[Dict[str, Any]]
    start_time: datetime
    end_time: Optional[datetime] = None
    summary: Optional[Dict[str, Any]] = None


@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    
    # Capital management
    initial_capital_sol: float = 1.5
    max_position_percent: float = 0.20
    
    # Risk management  
    acceptable_rel_threshold: float = 0.1
    stop_loss_percent: float = 0.05
    max_hold_time_hours: float = 4.0
    
    # Trading logic
    hunt_threshold: float = 0.6
    kelly_fraction: float = 0.25
    compound_rate: float = 0.8
    compound_threshold_sol: float = 0.2
    
    # Scanning
    scan_interval_seconds: int = 30


@dataclass
class WalletInfo:
    """Wallet information structure"""
    
    wallet_id: str
    public_key: str
    balance_sol: float
    active: bool
    created_at: datetime
    last_used: Optional[datetime] = None
    genetics: Optional[Dict[str, Any]] = None
    risk_profile: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


# Constants for validation and configuration
class SecurityLevel(Enum):
    """Security configuration levels"""
    STANDARD = "STANDARD"
    HIGH = "HIGH"
    MAXIMUM = "MAXIMUM"


class TradingMode(Enum):
    """Trading operation modes"""
    LIVE = "LIVE"
    SIMULATION = "SIMULATION"


# Default values and constants
DEFAULT_VALUES = {
    'max_trade_size_sol': 5.0,
    'min_trade_size_sol': 0.1,
    'max_slippage_percent': 1.0,
    'profit_target_percent': 2.5,
    'stop_loss_percent': 1.0,
    'max_daily_loss_sol': 10.0,
}

VALIDATION_RULES = {
    'trading': {
        'MAX_TRADE_SIZE_SOL': (0.1, 100.0),
        'MIN_TRADE_SIZE_SOL': (0.01, 10.0),
        'MAX_SLIPPAGE_PERCENT': (0.1, 10.0),
        'PROFIT_TARGET_PERCENT': (1.0, 50.0),
        'STOP_LOSS_PERCENT': (0.5, 20.0),
    },
    'safety': {
        'MAX_DAILY_LOSS_SOL': (1.0, 1000.0),
        'KILL_SWITCH_THRESHOLD': (0.05, 0.5),
        'EMERGENCY_STOP_THRESHOLD': (0.1, 0.8),
    }
}

# API Endpoints
class APIEndpoints:
    """API endpoint constants"""
    SOLANA_MAINNET_RPC = "https://api.mainnet-beta.solana.com"
    JUPITER_PRICE_URL = "https://quote-api.jup.ag/v6/price"
    BIRDEYE_BASE_URL = "https://public-api.birdeye.so"
    DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/latest/dex/tokens"

# Token Mints
class TokenMints:
    """Common token mint addresses"""
    SOL = "So11111111111111111111111111111111111111112"
    USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    USDT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"

# Validation Rules Helper Class
class ValidationRules:
    """Validation rules helper"""
    @staticmethod
    def get_trading_rules():
        return VALIDATION_RULES['trading']
    
    @staticmethod
    def get_safety_rules():
        return VALIDATION_RULES['safety'] 