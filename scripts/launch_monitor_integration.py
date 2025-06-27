#!/usr/bin/env python3
"""
LAUNCH MONITOR INTEGRATION
=========================

Connects the new token monitor with your existing trading bot for automatic trading.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the worker_ant_v1 to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.new_token_monitor import NewTokenDetector, TokenLaunchAnalyzer
from worker_ant_v1.trading.market_scanner import production_scanner
from worker_ant_v1.trading.order_buyer import ProductionBuyer
from worker_ant_v1.core.simple_config import get_trading_config

class IntegratedLaunchTrader:
    """Integrated system that monitors launches and executes trades"""
    
    def __init__(self, helius_api_key: str):
        self.helius_api_key = helius_api_key
        self.logger = logging.getLogger("IntegratedLaunchTrader")
        
        # Components
        self.detector = None
        self.analyzer = TokenLaunchAnalyzer()
        self.scanner = production_scanner
        self.buyer = ProductionBuyer()
        
        # Configuration
        self.config = get_trading_config()
        
        # Trading state
        self.active_trades = {}
        self.daily_trades = 0
        self.max_daily_trades = 10
        
    async def initialize(self):
        """Initialize all components"""
        
        self.logger.info("🔄 Initializing integrated launch trader...")
        
        # Initialize detector
        self.detector = NewTokenDetector(self.helius_api_key)
        await self.detector.__aenter__()
        
        # Initialize scanner and buyer
        await self.scanner.initialize()
        await self.buyer.initialize()
        
        # Add callback for new launches
        self.detector.add_callback(self.handle_new_launch)
        
        self.logger.info("✅ Integrated launch trader initialized")
    
    async def shutdown(self):
        """Shutdown all components"""
        
        self.logger.info("🛑 Shutting down integrated launch trader...")
        
        if self.detector:
            await self.detector.__aexit__(None, None, None)
        
        await self.scanner.shutdown()
        await self.buyer.shutdown()
        
        self.logger.info("✅ Shutdown complete")
    
    async def handle_new_launch(self, launch):
        """Handle newly detected token launch"""
        
        try:
            # Check daily trade limits
            if self.daily_trades >= self.max_daily_trades:
                self.logger.info(f"⏹️ Daily trade limit reached ({self.max_daily_trades})")
                return
            
            # Analyze the launch
            analysis = await self.analyzer.analyze_launch(launch)
            
            self.logger.info(
                f"🔍 Analyzing launch: {launch.symbol} "
                f"Score: {analysis['trading_score']:.2f} "
                f"Recommendation: {analysis['recommendation']}"
            )
            
            # Check if we should trade
            if analysis['recommendation'] == 'BUY' and analysis['trading_score'] > 0.7:
                await self.execute_launch_trade(launch, analysis)
            elif analysis['recommendation'] == 'WATCH':
                await self.add_to_watchlist(launch, analysis)
            else:
                self.logger.debug(f"⏭️ Skipping {launch.symbol}: {analysis['reasons']}")
                
        except Exception as e:
            self.logger.error(f"Error handling launch: {e}")
    
    async def execute_launch_trade(self, launch, analysis):
        """Execute a trade on a new launch"""
        
        try:
            self.logger.info(f"🎯 Executing trade on {launch.symbol}")
            
            # Create a trading opportunity for the scanner
            opportunity_data = {
                "token_address": launch.token_address,
                "symbol": launch.symbol,
                "name": launch.name,
                "price_sol": launch.initial_price_sol,
                "liquidity_sol": launch.initial_liquidity_sol,
                "market_cap_sol": launch.market_cap_sol,
                "volume_24h_sol": launch.volume_1h_sol,
                "confidence_score": analysis['trading_score'],
                "urgency_score": analysis['speed_score'],
                "risk_level": analysis['risk_score'],
                "source": f"launch_monitor_{launch.detection_method}"
            }
            
            # Use the existing scanner to create a proper opportunity
            from worker_ant_v1.trading.market_scanner import TradingOpportunity
            from worker_ant_v1.trading.market_scanner import SecurityRisk
            
            opportunity = TradingOpportunity(
                token_address=launch.token_address,
                token_symbol=launch.symbol,
                token_name=launch.name,
                current_price_sol=launch.initial_price_sol,
                market_cap_sol=launch.market_cap_sol,
                liquidity_sol=launch.initial_liquidity_sol,
                volume_24h_sol=launch.volume_1h_sol,
                confidence_score=analysis['trading_score'],
                urgency_score=analysis['speed_score'],
                profit_potential=analysis.get('profit_potential', 10.0),
                risk_level=analysis['risk_score'],
                security_risk=SecurityRisk.MEDIUM,  # Default to medium
                rug_probability=0.3,  # Default
                honeypot_risk=0.2,   # Default
                price_momentum=0.1,   # Default
                volume_spike=0.5,     # Default
                liquidity_score=0.7,  # Default
                source="launch_monitor"
            )
            
            # Create buy signal
            from worker_ant_v1.trading.order_buyer import BuySignal
            
            # Calculate trade amount based on confidence
            base_amount = self.config.trade_amount_sol
            confidence_multiplier = min(analysis['trading_score'] * 1.5, 2.0)
            trade_amount = min(base_amount * confidence_multiplier, self.config.max_trade_amount_sol)
            
            buy_signal = BuySignal(
                token_address=launch.token_address,
                amount_sol=trade_amount,
                max_slippage=self.config.max_slippage_percent,
                urgency=analysis['speed_score'],
                source="launch_monitor",
                metadata={
                    "launch_time": launch.launch_time.isoformat(),
                    "detection_method": launch.detection_method,
                    "detection_latency_ms": launch.detection_latency_ms,
                    "dex_platform": launch.dex_platform,
                    "analysis": analysis
                }
            )
            
            # Execute the buy
            buy_result = await self.buyer.execute_buy(buy_signal)
            
            if buy_result.success:
                self.daily_trades += 1
                self.active_trades[launch.token_address] = {
                    "launch": launch,
                    "analysis": analysis,
                    "buy_result": buy_result,
                    "entry_time": datetime.now()
                }
                
                self.logger.info(
                    f"✅ Successfully bought {launch.symbol} "
                    f"Amount: {trade_amount:.4f} SOL "
                    f"Signature: {buy_result.signature[:8]}..."
                )
                
                # Log successful trade
                self._log_successful_trade(launch, analysis, buy_result)
                
            else:
                self.logger.error(
                    f"❌ Failed to buy {launch.symbol}: {buy_result.error_message}"
                )
                
        except Exception as e:
            self.logger.error(f"Error executing launch trade: {e}")
    
    async def add_to_watchlist(self, launch, analysis):
        """Add token to watchlist for monitoring"""
        
        try:
            # Add to the scanner's watchlist
            self.scanner.add_token_to_watchlist(launch.token_address)
            
            self.logger.info(
                f"👀 Added {launch.symbol} to watchlist "
                f"(Score: {analysis['trading_score']:.2f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error adding to watchlist: {e}")
    
    def _log_successful_trade(self, launch, analysis, buy_result):
        """Log successful trade for analysis"""
        
        trade_log = {
            "timestamp": datetime.now().isoformat(),
            "token_address": launch.token_address,
            "symbol": launch.symbol,
            "trade_amount_sol": buy_result.amount_sol,
            "entry_price_sol": buy_result.average_price,
            "signature": buy_result.signature,
            "detection_method": launch.detection_method,
            "detection_latency_ms": launch.detection_latency_ms,
            "dex_platform": launch.dex_platform,
            "trading_score": analysis['trading_score'],
            "risk_score": analysis['risk_score'],
            "speed_score": analysis['speed_score']
        }
        
        # Append to trade log
        os.makedirs("logs", exist_ok=True)
        with open("logs/launch_trades.jsonl", "a") as f:
            import json
            f.write(json.dumps(trade_log) + "\n")
    
    async def run(self):
        """Run the integrated system"""
        
        self.logger.info("🚀 Starting integrated launch trader")
        
        try:
            await self.initialize()
            
            # Start continuous monitoring
            await self.detector.continuous_monitoring()
            
        except KeyboardInterrupt:
            self.logger.info("⏹️ Stopping due to user interrupt")
        except Exception as e:
            self.logger.error(f"💥 System error: {e}")
        finally:
            await self.shutdown()

async def main():
    """Main function"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/launch_trader.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("LaunchTraderMain")
    
    # Get Helius API key
    helius_api_key = os.getenv("HELIUS_API_KEY")
    if not helius_api_key:
        print("🔑 Helius API Key not found in environment")
        helius_api_key = input("Enter your Helius API key: ").strip()
        
        if not helius_api_key:
            print("❌ No API key provided. Exiting.")
            return
    
    print("🤖 LAUNCH MONITOR + TRADING BOT INTEGRATION")
    print("=" * 60)
    print("This system will:")
    print("• Monitor new token launches in real-time")
    print("• Analyze launch potential automatically")
    print("• Execute trades on high-potential launches")
    print("• Maintain risk management and limits")
    print("")
    
    # Create and run the integrated trader
    trader = IntegratedLaunchTrader(helius_api_key)
    await trader.run()

if __name__ == "__main__":
    asyncio.run(main()) 