"""
Seller Module for Worker Ant V1
===============================

Handles dynamic exit strategies: profit targets, stop losses, and timeout exits.
"""

import asyncio
import time
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime

from .config import config
from .logger import trading_logger, TradeResult
from .buyer import trade_buyer, BuyResult


@dataclass
class Position:
    """Represents an open trading position"""
    
    token_address: str
    token_symbol: str
    entry_time: float
    entry_price: float
    amount_tokens: float
    amount_sol_invested: float
    buy_signature: str
    
    # Tracking
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    last_price_check: float = 0.0


@dataclass
class SellResult:
    """Result of a sell operation"""
    
    success: bool
    signature: Optional[str] = None
    error_message: Optional[str] = None
    
    # Trade details
    sell_price: float = 0.0
    amount_sol_received: float = 0.0
    profit_loss_sol: float = 0.0
    profit_loss_percent: float = 0.0
    hold_time_seconds: int = 0
    
    # Exit reason
    exit_reason: str = "unknown"  # 'profit_target', 'stop_loss', 'timeout', 'manual'


class PositionManager:
    """Manages open positions and exit strategies"""
    
    def __init__(self):
        self.open_positions: Dict[str, Position] = {}
        self.monitoring = False
        self.jupiter_url = "https://quote-api.jup.ag/v6"
        
    async def start_monitoring(self):
        """Start monitoring open positions for exit signals"""
        
        self.monitoring = True
        trading_logger.logger.info("Position monitoring started")
        
        # Start the monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
    async def stop_monitoring(self):
        """Stop position monitoring"""
        
        self.monitoring = False
        trading_logger.logger.info("Position monitoring stopped")
        
    def add_position(self, buy_result: BuyResult, token_address: str, token_symbol: str):
        """Add a new position from a successful buy"""
        
        if not buy_result.success:
            return
            
        position = Position(
            token_address=token_address,
            token_symbol=token_symbol,
            entry_time=time.time(),
            entry_price=buy_result.price,
            amount_tokens=buy_result.amount_tokens,
            amount_sol_invested=buy_result.amount_sol,
            buy_signature=buy_result.signature,
            highest_price=buy_result.price,
            lowest_price=buy_result.price,
            last_price_check=time.time()
        )
        
        self.open_positions[token_address] = position
        
        trading_logger.logger.info(
            f"Added position: {token_symbol} - {buy_result.amount_tokens:.4f} tokens "
            f"at {buy_result.price:.8f} SOL per token"
        )
        
    async def _monitoring_loop(self):
        """Main monitoring loop for all positions"""
        
        while self.monitoring:
            try:
                if not self.open_positions:
                    await asyncio.sleep(5)  # Wait if no positions
                    continue
                    
                # Check each position
                for token_address in list(self.open_positions.keys()):
                    position = self.open_positions[token_address]
                    
                    try:
                        await self._check_position_exit_signals(position)
                    except Exception as e:
                        trading_logger.logger.error(
                            f"Error checking position {position.token_symbol}: {e}"
                        )
                        
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                trading_logger.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
                
    async def _check_position_exit_signals(self, position: Position):
        """Check if a position should be exited"""
        
        current_time = time.time()
        
        # Get current price
        current_price = await self._get_current_price(position.token_address)
        if current_price is None:
            return
            
        # Update price tracking
        position.highest_price = max(position.highest_price, current_price)
        position.lowest_price = min(position.lowest_price, current_price)
        position.last_price_check = current_time
        
        # Calculate current metrics
        profit_percent = ((current_price - position.entry_price) / position.entry_price) * 100
        hold_time = current_time - position.entry_time
        
        # Check exit conditions
        exit_reason = None
        
        # 1. Profit target
        if profit_percent >= config.profit_target_percent:
            exit_reason = "profit_target"
            trading_logger.logger.info(
                f"Profit target hit for {position.token_symbol}: {profit_percent:.2f}%"
            )
            
        # 2. Stop loss
        elif profit_percent <= -config.stop_loss_percent:
            exit_reason = "stop_loss"
            trading_logger.logger.warning(
                f"Stop loss triggered for {position.token_symbol}: {profit_percent:.2f}%"
            )
            
        # 3. Timeout exit
        elif hold_time >= config.timeout_exit_seconds:
            exit_reason = "timeout"
            trading_logger.logger.info(
                f"Timeout exit for {position.token_symbol}: {hold_time:.0f}s held"
            )
            
        # 4. Emergency conditions (optional)
        elif await self._check_emergency_exit(position, current_price):
            exit_reason = "emergency"
            trading_logger.logger.warning(
                f"Emergency exit for {position.token_symbol}"
            )
            
        # Execute exit if needed
        if exit_reason:
            await self._execute_exit(position, current_price, exit_reason)
            
    async def _check_emergency_exit(self, position: Position, current_price: float) -> bool:
        """Check for emergency exit conditions (rug pull indicators)"""
        
        # Large sudden drop (>30% in 10 seconds)
        if current_price < position.entry_price * 0.7:
            time_since_entry = time.time() - position.entry_time
            if time_since_entry < 10:
                return True
                
        # Price volatility too high (rapid swings)
        price_range = position.highest_price - position.lowest_price
        if price_range > position.entry_price * 0.5:  # 50% range
            return True
            
        return False
        
    async def _execute_exit(self, position: Position, current_price: float, exit_reason: str):
        """Execute the exit trade for a position"""
        
        try:
            trading_logger.logger.info(
                f"Executing exit for {position.token_symbol} - Reason: {exit_reason}"
            )
            
            # Get quote for selling tokens back to SOL
            quote = await self._get_sell_quote(position.token_address, position.amount_tokens)
            
            if not quote:
                trading_logger.logger.error(f"Failed to get sell quote for {position.token_symbol}")
                return
                
            # Execute the sell
            sell_result = await self._execute_sell_transaction(position, quote, exit_reason)
            
            # Remove from active positions
            if position.token_address in self.open_positions:
                del self.open_positions[position.token_address]
                
            # Log the completed trade
            if sell_result.success:
                trading_logger.logger.info(
                    f"Exit completed: {position.token_symbol} - "
                    f"P&L: {sell_result.profit_loss_percent:.2f}% "
                    f"({sell_result.profit_loss_sol:.4f} SOL)"
                )
            else:
                trading_logger.logger.error(
                    f"Exit failed: {position.token_symbol} - {sell_result.error_message}"
                )
                
        except Exception as e:
            trading_logger.logger.error(f"Error executing exit for {position.token_symbol}: {e}")
            
    async def _get_current_price(self, token_address: str) -> Optional[float]:
        """Get current price for a token"""
        
        try:
            import aiohttp
            
            # Use Jupiter for price discovery
            url = f"{self.jupiter_url}/quote"
            params = {
                'inputMint': token_address,
                'outputMint': "So11111111111111111111111111111111111112",  # SOL
                'amount': 1000000,  # 1 token (assuming 6 decimals)
                'onlyDirectRoutes': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        out_amount = float(data.get('outAmount', 0))
                        # Convert back to price per token
                        price = out_amount / 1e9  # Convert lamports to SOL
                        return price
                    else:
                        return None
                        
        except Exception as e:
            trading_logger.logger.debug(f"Error getting price for {token_address}: {e}")
            return None
            
    async def _get_sell_quote(self, token_address: str, amount_tokens: float) -> Optional[dict]:
        """Get quote for selling tokens"""
        
        try:
            import aiohttp
            
            # Convert token amount to raw units (assuming 6 decimals)
            amount_raw = int(amount_tokens * 1e6)
            
            url = f"{self.jupiter_url}/quote"
            params = {
                'inputMint': token_address,
                'outputMint': "So11111111111111111111111111111111111112",  # SOL
                'amount': amount_raw,
                'slippageBps': int(config.max_slippage_percent * 100 * 2),  # Allow 2x slippage for exits
                'onlyDirectRoutes': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        return None
                        
        except Exception as e:
            trading_logger.logger.error(f"Error getting sell quote: {e}")
            return None
            
    async def _execute_sell_transaction(self, position: Position, quote: dict, exit_reason: str) -> SellResult:
        """Execute the actual sell transaction"""
        
        start_time = time.time()
        
        try:
            # Use the same Jupiter swap execution as buyer
            import aiohttp
            
            # Get swap transaction from Jupiter
            swap_url = f"{self.jupiter_url}/swap"
            swap_data = {
                'quoteResponse': quote,
                'userPublicKey': str(trade_buyer.wallet.public_key),
                'wrapAndUnwrapSol': True,
                'useSharedAccounts': True,
                'feeAccount': None
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    swap_url,
                    json=swap_data,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status != 200:
                        return SellResult(
                            success=False,
                            error_message=f"Jupiter API error: {resp.status}",
                            exit_reason=exit_reason
                        )
                        
                    swap_response = await resp.json()
                    
            # Execute the transaction (reuse buyer's logic)
            swap_transaction = swap_response.get('swapTransaction')
            if not swap_transaction:
                return SellResult(
                    success=False,
                    error_message="No swap transaction returned",
                    exit_reason=exit_reason
                )
                
            # Sign and submit transaction
            import base64
            from solana.transaction import Transaction
            
            tx_bytes = base64.b64decode(swap_transaction)
            transaction = Transaction.deserialize(tx_bytes)
            transaction.sign(trade_buyer.wallet)
            
            response = await trade_buyer.client.send_transaction(transaction)
            
            if response.value:
                # Wait for confirmation
                confirmed = await trade_buyer._wait_for_confirmation(response.value, timeout=30)
                
                if confirmed:
                    # Calculate results
                    sol_received = float(quote.get('outAmount', 0)) / 1e9
                    profit_loss_sol = sol_received - position.amount_sol_invested
                    profit_loss_percent = (profit_loss_sol / position.amount_sol_invested) * 100
                    hold_time = int(time.time() - position.entry_time)
                    
                    result = SellResult(
                        success=True,
                        signature=response.value,
                        sell_price=sol_received / position.amount_tokens,
                        amount_sol_received=sol_received,
                        profit_loss_sol=profit_loss_sol,
                        profit_loss_percent=profit_loss_percent,
                        hold_time_seconds=hold_time,
                        exit_reason=exit_reason
                    )
                    
                    # Log the complete trade cycle
                    trade_record = TradeResult(
                        timestamp=datetime.utcnow().isoformat(),
                        token_address=position.token_address,
                        token_symbol=position.token_symbol,
                        trade_type='SELL',
                        amount_sol=sol_received,
                        amount_tokens=position.amount_tokens,
                        price=result.sell_price,
                        slippage_percent=quote.get('priceImpactPct', 0),
                        latency_ms=int((time.time() - start_time) * 1000),
                        tx_signature=response.value,
                        success=True,
                        profit_loss_sol=profit_loss_sol,
                        profit_loss_percent=profit_loss_percent,
                        hold_time_seconds=hold_time
                    )
                    
                    trading_logger.log_trade(trade_record)
                    
                    return result
                else:
                    return SellResult(
                        success=False,
                        error_message="Transaction confirmation timeout",
                        exit_reason=exit_reason
                    )
            else:
                return SellResult(
                    success=False,
                    error_message="Transaction submission failed",
                    exit_reason=exit_reason
                )
                
        except Exception as e:
            error_msg = f"Sell execution error: {str(e)}"
            trading_logger.logger.error(error_msg)
            
            return SellResult(
                success=False,
                error_message=error_msg,
                exit_reason=exit_reason
            )
            
    def get_position_summary(self) -> Dict:
        """Get summary of all open positions"""
        
        summary = {
            'total_positions': len(self.open_positions),
            'total_invested_sol': 0.0,
            'positions': []
        }
        
        for position in self.open_positions.values():
            summary['total_invested_sol'] += position.amount_sol_invested
            
            position_info = {
                'token_symbol': position.token_symbol,
                'token_address': position.token_address,
                'entry_time': position.entry_time,
                'hold_time_seconds': int(time.time() - position.entry_time),
                'amount_tokens': position.amount_tokens,
                'amount_sol_invested': position.amount_sol_invested,
                'entry_price': position.entry_price,
                'highest_price': position.highest_price,
                'lowest_price': position.lowest_price
            }
            
            summary['positions'].append(position_info)
            
        return summary
        
    async def manual_exit_position(self, token_address: str) -> SellResult:
        """Manually exit a specific position"""
        
        if token_address not in self.open_positions:
            return SellResult(
                success=False,
                error_message="Position not found",
                exit_reason="manual"
            )
            
        position = self.open_positions[token_address]
        current_price = await self._get_current_price(token_address)
        
        if current_price is None:
            return SellResult(
                success=False,
                error_message="Could not get current price",
                exit_reason="manual"
            )
            
        return await self._execute_exit(position, current_price, "manual")
        
    async def emergency_exit_all(self) -> List[SellResult]:
        """Emergency exit all positions"""
        
        results = []
        
        for token_address in list(self.open_positions.keys()):
            result = await self.manual_exit_position(token_address)
            results.append(result)
            
        return results


# Global position manager instance
position_manager = PositionManager() 