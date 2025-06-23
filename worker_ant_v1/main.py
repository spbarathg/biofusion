"""
Main Orchestrator for Worker Ant V1
===================================

The main trading loop that coordinates scanning, buying, and selling.
"""

import asyncio
import signal
import time
from typing import Optional

from .config import config, validate_config
from .logger import trading_logger
from .scanner import token_scanner
from .buyer import trade_buyer
from .seller import position_manager


class WorkerAntV1:
    """Main Worker Ant trading bot orchestrator"""
    
    def __init__(self):
        self.running = False
        self.scan_count = 0
        self.trade_count = 0
        
    async def start(self):
        """Start the trading bot"""
        
        try:
            # Validate configuration
            validate_config()
            trading_logger.logger.info("Configuration validated")
            
            # Setup components
            await self._setup_components()
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start the main trading loop
            self.running = True
            trading_logger.logger.info("Worker Ant V1 started - Beginning hunt for opportunities...")
            
            await self._main_trading_loop()
            
        except KeyboardInterrupt:
            trading_logger.logger.info("Received shutdown signal")
        except Exception as e:
            trading_logger.logger.error(f"Fatal error: {e}")
            raise
        finally:
            await self._cleanup()
            
    async def _setup_components(self):
        """Initialize all trading components"""
        
        trading_logger.logger.info("Setting up trading components...")
        
        # Setup buyer (wallet and RPC connection)
        await trade_buyer.setup()
        
        # Start token scanner
        await token_scanner.start()
        
        # Start position monitoring
        await position_manager.start_monitoring()
        
        trading_logger.logger.info("All components ready")
        
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        
        def signal_handler(signum, frame):
            trading_logger.logger.info(f"Received signal {signum}, shutting down...")
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def _main_trading_loop(self):
        """Main trading loop - the heart of the bot"""
        
        from .config import scanner_config
        
        while self.running:
            try:
                loop_start = time.time()
                
                # 1. Scan for new opportunities
                opportunities = await token_scanner.scan_for_opportunities()
                self.scan_count += 1
                
                if opportunities:
                    trading_logger.logger.info(
                        f"Found {len(opportunities)} opportunities: "
                        f"{[f'{opp.token_symbol}({opp.confidence_score:.2f})' for opp in opportunities[:3]]}"
                    )
                    
                    # 2. Evaluate and execute best opportunity
                    best_opportunity = opportunities[0]  # Already sorted by confidence
                    
                    if await self._should_trade(best_opportunity):
                        await self._execute_trade(best_opportunity)
                        
                else:
                    trading_logger.logger.debug(f"Scan #{self.scan_count}: No opportunities found")
                    
                # 3. Log periodic status
                if self.scan_count % 20 == 0:  # Every 20 scans
                    await self._log_status()
                    
                # 4. Sleep until next scan
                loop_time = time.time() - loop_start
                sleep_time = max(0, scanner_config.scan_interval_seconds - loop_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
            except Exception as e:
                trading_logger.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Backoff on error
                
    async def _should_trade(self, opportunity) -> bool:
        """Determine if we should trade this opportunity"""
        
        # Check safety limits
        if not trading_logger.check_safety_limits():
            return False
            
        # Check if we have sufficient confidence
        if opportunity.confidence_score < 0.5:  # 50% minimum confidence for MVP
            trading_logger.logger.debug(
                f"Skipping {opportunity.token_symbol}: confidence too low ({opportunity.confidence_score:.2f})"
            )
            return False
            
        # Check position limits
        positions = position_manager.get_position_summary()
        if positions['total_positions'] >= 3:  # Max 3 positions at once for MVP
            trading_logger.logger.debug("Skipping trade: maximum positions reached")
            return False
            
        # Check wallet balance
        balance = await trade_buyer.get_sol_balance()
        if balance < config.trade_amount_sol * 1.2:  # 20% buffer
            trading_logger.logger.warning(f"Insufficient balance for trade: {balance:.4f} SOL")
            return False
            
        return True
        
    async def _execute_trade(self, opportunity):
        """Execute a trade for the given opportunity"""
        
        try:
            trading_logger.logger.info(
                f"🚀 EXECUTING TRADE: {opportunity.token_symbol} "
                f"(Confidence: {opportunity.confidence_score:.2f}, "
                f"Liquidity: {opportunity.liquidity_sol:.2f} SOL)"
            )
            
            # Execute the buy
            buy_result = await trade_buyer.execute_buy(opportunity)
            
            if buy_result.success:
                self.trade_count += 1
                
                # Add to position manager for monitoring
                position_manager.add_position(
                    buy_result,
                    opportunity.token_address,
                    opportunity.token_symbol
                )
                
                trading_logger.logger.info(
                    f"✅ BUY SUCCESS: {opportunity.token_symbol} "
                    f"- {buy_result.amount_tokens:.4f} tokens @ {buy_result.price:.8f} SOL "
                    f"- Latency: {buy_result.latency_ms}ms "
                    f"- Slippage: {buy_result.slippage_percent:.2f}%"
                )
                
            else:
                trading_logger.logger.warning(
                    f"❌ BUY FAILED: {opportunity.token_symbol} - {buy_result.error_message}"
                )
                
        except Exception as e:
            trading_logger.logger.error(f"Error executing trade for {opportunity.token_symbol}: {e}")
            
    async def _log_status(self):
        """Log periodic status information"""
        
        # Get current metrics
        metrics = trading_logger.get_session_metrics()
        positions = position_manager.get_position_summary()
        balance = await trade_buyer.get_sol_balance()
        
        trading_logger.logger.info(
            f"📊 STATUS UPDATE:\n"
            f"  Scans: {self.scan_count} | Trades: {self.trade_count}\n"
            f"  Balance: {balance:.4f} SOL\n"
            f"  Positions: {positions['total_positions']} "
            f"(Invested: {positions['total_invested_sol']:.4f} SOL)\n"
            f"  Win Rate: {metrics.win_rate_percent:.1f}% | "
            f"  Total P&L: {metrics.total_profit_sol:.4f} SOL\n"
            f"  Avg Latency: {metrics.average_latency_ms:.0f}ms | "
            f"  Avg Slippage: {metrics.average_slippage_percent:.2f}%"
        )
        
    async def _cleanup(self):
        """Clean up resources on shutdown"""
        
        trading_logger.logger.info("Shutting down Worker Ant V1...")
        
        try:
            # Stop scanning
            await token_scanner.stop()
            
            # Stop position monitoring
            await position_manager.stop_monitoring()
            
            # Emergency exit all positions (optional - comment out if you want to keep positions)
            # positions = position_manager.get_position_summary()
            # if positions['total_positions'] > 0:
            #     trading_logger.logger.info(f"Emergency exit of {positions['total_positions']} positions")
            #     await position_manager.emergency_exit_all()
            
            # Close buyer connections
            await trade_buyer.close()
            
            # Export final metrics
            metrics_file = f"worker_ant_v1/session_metrics_{int(time.time())}.json"
            trading_logger.export_metrics(metrics_file)
            
            # Close logger
            trading_logger.close()
            
            trading_logger.logger.info("Shutdown complete")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")


async def main():
    """Entry point for the Worker Ant V1 trading bot"""
    
    print("🐜 Worker Ant V1 - Simplified Memecoin Trading Bot")
    print("=" * 50)
    
    bot = WorkerAntV1()
    await bot.start()


if __name__ == "__main__":
    # For direct execution
    asyncio.run(main()) 