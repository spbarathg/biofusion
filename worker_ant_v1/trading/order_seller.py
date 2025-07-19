"""
PRODUCTION-READY SELLER MODULE
=============================

Ultra-fast, secure sell execution with intelligent exit strategies,
profit optimization, emergency exits, and comprehensive monitoring.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from solana.rpc.async_api import AsyncClient
    from solana.rpc.commitment import Confirmed, Finalized, Processed
    from solana.transaction import Transaction
except ImportError:
    from ..utils.solana_compat import AsyncClient, Confirmed, Finalized, Processed, Transaction
try:
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
except ImportError:
    from ..utils.solana_compat import Keypair as Keypair, PublicKey as Pubkey
try:
    from spl.token.constants import TOKEN_PROGRAM_ID
except ImportError:
    TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
import base58

# Internal imports
from worker_ant_v1.core.unified_config import (
    get_network_config,
    get_security_config,
    get_trading_config
)
from worker_ant_v1.utils.jupiter_dex_integration import JupiterAggregator
from worker_ant_v1.utils.logger import setup_logger

sell_logger = setup_logger(__name__)


@dataclass
class SellConfig:
    """Configuration for sell operations"""
    
    # Price thresholds
    min_profit_percent: float = 5.0
    max_loss_percent: float = 10.0
    emergency_loss_percent: float = 20.0
    
    # Timing settings
    max_hold_time_seconds: int = 3600
    min_hold_time_seconds: int = 60
    price_check_interval_seconds: float = 1.0
    
    # Transaction settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    max_slippage_percent: float = 1.0
    
    # Safety settings
    enable_stop_loss: bool = True
    enable_trailing_stop: bool = True
    trailing_stop_percent: float = 2.0
    
    # Liquidity settings
    min_liquidity_sol: float = 10.0
    max_impact_percent: float = 3.0
    
    # Advanced settings
    use_limit_orders: bool = False
    partial_fills_enabled: bool = True
    min_fill_percent: float = 50.0


@dataclass
class SellPosition:
    """Active sell position tracking"""
    
    token_address: str
    token_symbol: str
    amount: float
    entry_price_sol: float
    current_price_sol: float
    highest_price_sol: float
    entry_time: datetime
    last_update_time: datetime
    stop_loss_price: float
    trailing_stop_price: float
    total_profit_loss: float
    status: str = "active"


class SellStatus(Enum):
    """Sell operation status"""
    
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EMERGENCY = "emergency"


class ProductionSeller:
    """Production-grade seller with comprehensive monitoring and safety features"""
    
    def __init__(self):
        self.logger = sell_logger
        self.config = SellConfig()
        self.network_config = get_network_config()
        
        # RPC client setup
        self.rpc_client = None
        self.wallet = None
        
        # Position tracking
        self.active_positions = {}
        self.position_history = []
        
        # Performance metrics
        self.total_sells = 0
        self.successful_sells = 0
        self.failed_sells = 0
        self.total_profit_sol = 0.0
        self.average_sell_latency_ms = 0.0
        self.average_hold_time_seconds = 0.0
        
        # Safety settings
        self.stop_loss_enabled = True
        self.trailing_stop_enabled = True
        
        # Initialize Jupiter
        self.jupiter = JupiterAggregator()
        
        # Load config
        self._load_config()
    
    def _load_config(self):
        """Load configuration from settings"""
        
        try:
            config = get_trading_config()
            if config and "sell" in config:
                sell_config = config["sell"]
                
                # Load price thresholds
                self.config.min_profit_percent = float(
                    sell_config.get(
                        "min_profit_percent",
                        self.config.min_profit_percent
                    )
                )
                self.config.max_loss_percent = float(
                    sell_config.get(
                        "max_loss_percent",
                        self.config.max_loss_percent
                    )
                )
                self.config.emergency_loss_percent = float(
                    sell_config.get(
                        "emergency_loss_percent",
                        self.config.emergency_loss_percent
                    )
                )
                
                # Load timing settings
                self.config.max_hold_time_seconds = int(
                    sell_config.get(
                        "max_hold_time_seconds",
                        self.config.max_hold_time_seconds
                    )
                )
                self.config.min_hold_time_seconds = int(
                    sell_config.get(
                        "min_hold_time_seconds",
                        self.config.min_hold_time_seconds
                    )
                )
                self.config.price_check_interval_seconds = float(
                    sell_config.get(
                        "price_check_interval_seconds",
                        self.config.price_check_interval_seconds
                    )
                )
                
                # Load transaction settings
                self.config.max_retries = int(
                    sell_config.get(
                        "max_retries",
                        self.config.max_retries
                    )
                )
                self.config.retry_delay_seconds = float(
                    sell_config.get(
                        "retry_delay_seconds",
                        self.config.retry_delay_seconds
                    )
                )
                self.config.max_slippage_percent = float(
                    sell_config.get(
                        "max_slippage_percent",
                        self.config.max_slippage_percent
                    )
                )
                
                # Load safety settings
                self.config.enable_stop_loss = bool(
                    sell_config.get(
                        "enable_stop_loss",
                        self.config.enable_stop_loss
                    )
                )
                self.config.enable_trailing_stop = bool(
                    sell_config.get(
                        "enable_trailing_stop",
                        self.config.enable_trailing_stop
                    )
                )
                self.config.trailing_stop_percent = float(
                    sell_config.get(
                        "trailing_stop_percent",
                        self.config.trailing_stop_percent
                    )
                )
                
                # Load liquidity settings
                self.config.min_liquidity_sol = float(
                    sell_config.get(
                        "min_liquidity_sol",
                        self.config.min_liquidity_sol
                    )
                )
                self.config.max_impact_percent = float(
                    sell_config.get(
                        "max_impact_percent",
                        self.config.max_impact_percent
                    )
                )
                
                # Load advanced settings
                self.config.use_limit_orders = bool(
                    sell_config.get(
                        "use_limit_orders",
                        self.config.use_limit_orders
                    )
                )
                self.config.partial_fills_enabled = bool(
                    sell_config.get(
                        "partial_fills_enabled",
                        self.config.partial_fills_enabled
                    )
                )
                self.config.min_fill_percent = float(
                    sell_config.get(
                        "min_fill_percent",
                        self.config.min_fill_percent
                    )
                )
                
        except Exception as e:
            self.logger.error(f"Failed to load sell config: {e}")
    
    async def connect(self):
        """Connect to Solana network"""
        
        try:
            if not self.rpc_client:
                self.rpc_client = AsyncClient(
                    self.network_config["rpc_url"],
                    commitment=Processed
                )
                self.logger.info("Connected to Solana network")
            
            if not self.wallet:
                private_key = base58.b58decode(
                    self.network_config["private_key"]
                )
                self.wallet = Keypair.from_bytes(private_key)
                self.logger.info("Wallet loaded successfully")
            
            await self.jupiter.connect()
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Solana network"""
        
        try:
            if self.rpc_client:
                await self.rpc_client.close()
                self.rpc_client = None
                self.logger.info("Disconnected from Solana network")
            
            await self.jupiter.disconnect()
            
        except Exception as e:
            self.logger.error(f"Failed to disconnect: {e}")
    
    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get current token price in SOL"""
        
        try:
            price = await self.jupiter.get_token_price(token_address)
            return price
            
        except Exception as e:
            self.logger.error(f"Failed to get price for {token_address}: {e}")
        
        return None
    
    async def get_token_prices(
        self,
        token_addresses: List[str]
    ) -> Dict[str, Optional[float]]:
        """Get current prices for multiple tokens"""
        
        results = {}
        
        for token_address in token_addresses:
            try:
                price = await self.get_token_price(token_address)
                results[token_address] = price
            except Exception as e:
                self.logger.error(
                    f"Failed to get price for {token_address}: {e}"
                )
                results[token_address] = None
        
        return results
    
    async def check_liquidity(
        self,
        token_address: str,
        amount_sol: float
    ) -> bool:
        """Check if token has sufficient liquidity"""
        
        try:
            liquidity = await self.jupiter.get_token_liquidity(token_address)
            
            if liquidity < self.config.min_liquidity_sol:
                self.logger.warning(
                    f"Insufficient liquidity for {token_address}: "
                    f"{liquidity:.3f} SOL"
                )
                return False
            
            impact = await self.jupiter.get_price_impact(
                token_address,
                amount_sol
            )
            
            if impact > self.config.max_impact_percent:
                self.logger.warning(
                    f"Price impact too high for {token_address}: "
                    f"{impact:.1f}%"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to check liquidity for {token_address}: {e}"
            )
            return False
    
    def _calculate_stop_loss(
        self,
        entry_price: float,
        current_price: float
    ) -> float:
        """Calculate stop loss price"""
        
        stop_loss = entry_price * (
            1 - (self.config.max_loss_percent / 100)
        )
        
        if self.config.enable_trailing_stop:
            trailing_stop = current_price * (
                1 - (self.config.trailing_stop_percent / 100)
            )
            stop_loss = max(stop_loss, trailing_stop)
        
        return stop_loss
    
    def _should_sell(
        self,
        position: SellPosition,
        current_price: float
    ) -> bool:
        """Check if position should be sold"""
        
        # Check stop loss
        if (
            self.config.enable_stop_loss and
            current_price <= position.stop_loss_price
        ):
            self.logger.warning(
                f"Stop loss triggered for {position.token_symbol}"
            )
            return True
        
        # Check trailing stop
        if (
            self.config.enable_trailing_stop and
            current_price <= position.trailing_stop_price
        ):
            self.logger.warning(
                f"Trailing stop triggered for {position.token_symbol}"
            )
            return True
        
        # Check profit target
        profit_percent = (
            (current_price - position.entry_price_sol) /
            position.entry_price_sol * 100
        )
        if profit_percent >= self.config.min_profit_percent:
            self.logger.info(
                f"Profit target reached for {position.token_symbol}: "
                f"{profit_percent:.1f}%"
            )
            return True
        
        # Check hold time
        hold_time = (
            datetime.utcnow() - position.entry_time
        ).total_seconds()
        if hold_time >= self.config.max_hold_time_seconds:
            self.logger.warning(
                f"Max hold time reached for {position.token_symbol}"
            )
            return True
        
        return False
    
    async def execute_sell(
        self,
        token_address: str,
        amount: float
    ) -> bool:
        """Execute sell order"""
        
        try:
            # Check liquidity
            if not await self.check_liquidity(token_address, amount):
                return False
            
            # Get current price
            price = await self.get_token_price(token_address)
            if not price:
                return False
            
            # Execute trade
            success = await self.jupiter.sell_token(
                token_address,
                amount,
                self.config.max_slippage_percent
            )
            
            if success:
                self.logger.info(
                    f"Successfully sold {amount} tokens at {price:.6f} SOL"
                )
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to execute sell: {e}")
            return False
    
    async def monitor_positions(self):
        """Monitor active sell positions"""
        
        while True:
            try:
                if not self.active_positions:
                    await asyncio.sleep(1)
                    continue
                
                # Get current prices
                token_addresses = list(self.active_positions.keys())
                prices = await self.get_token_prices(token_addresses)
                
                # Update positions
                for token_address, position in self.active_positions.items():
                    current_price = prices.get(token_address)
                    if not current_price:
                        continue
                    
                    # Update position
                    position.current_price_sol = current_price
                    position.highest_price_sol = max(
                        position.highest_price_sol,
                        current_price
                    )
                    position.last_update_time = datetime.utcnow()
                    
                    # Update trailing stop
                    if self.config.enable_trailing_stop:
                        position.trailing_stop_price = max(
                            position.trailing_stop_price,
                            current_price * (
                                1 - self.config.trailing_stop_percent / 100
                            )
                        )
                    
                    # Check if should sell
                    if self._should_sell(position, current_price):
                        success = await self.execute_sell(
                            token_address,
                            position.amount
                        )
                        
                        if success:
                            position.status = "completed"
                            self.position_history.append(position)
                            del self.active_positions[token_address]
                
                await asyncio.sleep(
                    self.config.price_check_interval_seconds
                )
                
            except Exception as e:
                self.logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(5)
    
    def get_status(self) -> Dict:
        """Get current seller status"""
        
        return {
            "active_positions": len(self.active_positions),
            "total_sells": self.total_sells,
            "successful_sells": self.successful_sells,
            "failed_sells": self.failed_sells,
            "total_profit_sol": self.total_profit_sol,
            "average_sell_latency_ms": self.average_sell_latency_ms,
            "average_hold_time_seconds": self.average_hold_time_seconds,
            "stop_loss_enabled": self.stop_loss_enabled,
            "trailing_stop_enabled": self.trailing_stop_enabled
        } 