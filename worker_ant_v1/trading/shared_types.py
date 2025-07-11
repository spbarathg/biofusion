"""
SHARED TYPES - BATTLEFIELD STANDARDS
================================

Common type definitions and data structures used across
the battlefield trading system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

@dataclass
class TokenMetadata:
    """Token metadata information"""
    
    address: str
    symbol: str
    name: str
    decimals: int
    
    total_supply: float = 0.0
    market_cap: float = 0.0
    volume_24h: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class PriceData:
    """Token price information"""
    
    token_address: str
    price_usd: float
    price_sol: float
    timestamp: datetime = field(default_factory=datetime.now)

class TokenStatus(Enum):
    """Token trading status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BLACKLISTED = "blacklisted"
    WHITELISTED = "whitelisted"
    MONITORING = "monitoring"

@dataclass
class TokenState:
    """Token state tracking"""
    
    address: str
    status: TokenStatus
    metadata: TokenMetadata
    current_price: PriceData
    
    position_size: float = 0.0
    entry_price: float = 0.0
    last_trade_time: datetime = field(default_factory=datetime.now)
    
    is_tracked: bool = False
    track_reason: str = "" 