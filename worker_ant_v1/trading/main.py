"""
HYPER-INTELLIGENT MEMECOIN TRADING BOT - PRODUCTION READY
=======================================================

Main entry point for the hyper-intelligent memecoin trading bot.
Implements aggressive sentiment-driven trading with rapid compounding.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import traceback
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List

# Import core systems
from worker_ant_v1.core.unified_trading_engine import get_trading_engine
from worker_ant_v1.core.vault_wallet_system import get_vault_system
from worker_ant_v1.core.wallet_manager import get_wallet_manager
from worker_ant_v1.intelligence.sentiment_first_ai import get_sentiment_first_ai
from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch
from worker_ant_v1.trading.market_scanner import get_market_scanner
from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.utils.constants import SentimentDecision as SentimentDecisionEnum

# Three-Stage Decision Pipeline Imports
from worker_ant_v1.trading.devils_advocate_synapse import DevilsAdvocateSynapse
from worker_ant_v1.core.swarm_decision_engine import SwarmDecisionEngine
from worker_ant_v1.trading.hyper_compound_engine import HyperCompoundEngine


class HyperIntelligentTradingSwarm:
    """Hyper-intelligent trading swarm that orchestrates all systems."""

    def __init__(self, config_file: str = None, initial_capital: float = 300.0):
        """Initialize the trading swarm.
        
        Args:
            config_file: Path to swarm-specific configuration file
            initial_capital: Initial capital allocation for this swarm
        """
        self.logger = get_logger("TradingSwarm")
        
        # Swarm configuration
        self.config_file = config_file
        self.initial_capital = initial_capital
        self.swarm_id = os.getenv('SWARM_ID', 'default_swarm')

        # Core systems
        self.trading_engine = None
        self.wallet_manager = None
        self.vault_system = None
        self.sentiment_ai = None
        self.market_scanner = None
        self.kill_switch = None

        # System state
        self.initialized = False
        self.trading_active = False
        self.start_time = None
        
        # Blitzscaling mode
        self.blitzscaling_active = False

        self.logger.info(f"🚀 HyperIntelligentTradingSwarm {self.swarm_id} initialized with {initial_capital} SOL")

    async def initialize_all_systems(self) -> bool:
        """Initialize all trading systems.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info(f"🚀 Initializing HyperIntelligentTradingSwarm {self.swarm_id}...")

            # Load swarm-specific configuration if provided
            if self.config_file and os.path.exists(self.config_file):
                await self._load_swarm_config()
                self.logger.info(f"📋 Loaded swarm configuration from {self.config_file}")

            # Initialize core systems
            self.trading_engine = await get_trading_engine()
            self.wallet_manager = await get_wallet_manager()
            self.vault_system = await get_vault_system()
            self.sentiment_ai = await get_sentiment_first_ai()
            self.market_scanner = await get_market_scanner()

            # Initialize kill switch
            self.kill_switch = EnhancedKillSwitch()
            await self.kill_switch.initialize()

            # Set up signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            self.initialized = True
            self.trading_active = True
            self.start_time = datetime.now()

            self.logger.info(f"✅ HyperIntelligentTradingSwarm {self.swarm_id} initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Swarm {self.swarm_id} initialization failed: {e}")
            return False

    async def run(self):
        """Run the trading swarm."""
        try:
            if not self.initialized:
                self.logger.error("❌ Swarm not initialized")
                return

            self.logger.info("🔥 Starting HyperIntelligentTradingSwarm...")

            # Start the main trading bot
            from worker_ant_v1.trading.main import MemecoinTradingBot

            bot = MemecoinTradingBot(initial_capital=300.0)

            if await bot.initialize():
                # Keep the bot running
                while bot.trading_active and self.trading_active:
                    await asyncio.sleep(1)
            else:
                self.logger.error("❌ Failed to initialize trading bot")

        except Exception as e:
            self.logger.error(f"❌ Swarm error: {e}")
            await self.emergency_shutdown()

    async def shutdown(self):
        """Shutdown the trading swarm."""
        try:
            self.logger.info("🛑 Shutting down HyperIntelligentTradingSwarm...")

            self.trading_active = False

            # Shutdown core systems
            if self.trading_engine:
                await self.trading_engine.shutdown()
            if self.wallet_manager:
                await self.wallet_manager.shutdown()
            if self.vault_system:
                await self.vault_system.shutdown()
            if self.kill_switch:
                await self.kill_switch.shutdown()

            self.logger.info("✅ HyperIntelligentTradingSwarm shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def emergency_shutdown(self):
        """Emergency shutdown."""
        try:
            self.logger.error("🚨 EMERGENCY SHUTDOWN TRIGGERED")
            await self.shutdown()
        except Exception as e:
            self.logger.error(f"Emergency shutdown error: {e}")

    async def _load_swarm_config(self):
        """Load swarm-specific configuration from file"""
        try:
            if not self.config_file or not os.path.exists(self.config_file):
                return
            
            # Load environment variables from config file
            with open(self.config_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        if key and value:
                            os.environ[key] = value
            
            self.logger.info(f"📋 Swarm configuration loaded from {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading swarm config: {e}")
    
    async def set_blitzscaling_mode(self, active: bool):
        """Set blitzscaling mode for this swarm"""
        self.blitzscaling_active = active
        self.logger.info(f"🚀 Blitzscaling mode {'ACTIVATED' if active else 'DEACTIVATED'} for swarm {self.swarm_id}")
        
        # Notify core systems of blitzscaling mode change
        if self.trading_engine and hasattr(self.trading_engine, 'set_blitzscaling_mode'):
            await self.trading_engine.set_blitzscaling_mode(active)
        
        if self.wallet_manager and hasattr(self.wallet_manager, 'set_blitzscaling_mode'):
            await self.wallet_manager.set_blitzscaling_mode(active)
    
    def get_status(self) -> Dict[str, Any]:
        """Get swarm status for colony monitoring"""
        return {
            'swarm_id': self.swarm_id,
            'initialized': self.initialized,
            'trading_active': self.trading_active,
            'blitzscaling_active': self.blitzscaling_active,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'initial_capital': self.initial_capital,
            'current_capital': self.initial_capital,  # This would be updated from actual trading
            'total_trades': 0,  # This would be updated from actual trading
            'successful_trades': 0,  # This would be updated from actual trading
            'total_profit': 0.0,  # This would be updated from actual trading
        }

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.logger.info(f"🛑 Received signal {signum}, shutting down swarm {self.swarm_id}...")
        asyncio.create_task(self.shutdown())


class MemecoinTradingBot:
    """Hyper-intelligent memecoin trading bot with aggressive compounding."""

    def __init__(self, initial_capital: float = 300.0):
        """Initialize the trading bot.

        Args:
            initial_capital: Initial capital in SOL
        """
        self.logger = get_logger("MemecoinBot")
        self.initial_capital = initial_capital

        # Core systems
        self.trading_engine = None
        self.wallet_manager = None
        self.vault_system = None
        self.sentiment_ai = None
        self.market_scanner = None
        self.kill_switch = None
        
        # Three-Stage Decision Pipeline Components
        self.devils_advocate = None  # Stage 1: Survival Filter (WCCA)
        self.swarm_decision_engine = None  # Stage 2: Win-Rate Engine (Naive Bayes)
        self.hyper_compound_engine = None  # Stage 3: Growth Maximizer (Kelly Criterion)

        # Trading state
        self.trading_active = False
        self.initialized = False
        self.start_time = None

        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit = 0.0
        self.current_capital = initial_capital

        # Position tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.position_history: List[Dict[str, Any]] = []

        # Trading configuration
        self.trading_config = {
            "max_concurrent_positions": 5,
            "position_size_percent": 0.2,  # 20% of capital per position
            "max_position_size_sol": 50.0,  # Maximum 50 SOL per position
            "min_profit_target": 0.05,  # 5% minimum profit
            "max_loss_percent": 0.15,  # 15% maximum loss
            "max_hold_time_hours": 4,  # 4 hours maximum hold
            "scan_interval_seconds": 30,  # Scan every 30 seconds
            "sentiment_threshold": 0.3,  # Minimum sentiment for buying
            "compounding_enabled": True,
            "aggressive_mode": True,
            
            # Three-Stage Pipeline Configuration
            "acceptable_rel_threshold": 0.1,  # WCCA: Max acceptable R-EL in SOL
            "hunt_threshold": 0.6,  # Naive Bayes: Min win probability to hunt
            "kelly_fraction": 0.25,  # Kelly Criterion: Fraction of full Kelly to use
        }

        # Statistics
        self.stats = {
            "trades_executed": 0,
            "total_profit_sol": 0.0,
            "win_rate": 0.0,
            "avg_profit_per_trade": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "opportunities_scanned": 0,
            "positions_opened": 0,
            "positions_closed": 0,
        }

        self.logger.info(
            f"🚀 Memecoin Trading Bot initialized with ${initial_capital} capital"
        )

    async def initialize(self) -> bool:
        """Initialize all trading systems.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("🚀 Initializing Memecoin Trading Bot...")

            # Initialize core systems
            self.trading_engine = await get_trading_engine()
            self.wallet_manager = await get_wallet_manager()
            self.vault_system = await get_vault_system()
            self.sentiment_ai = await get_sentiment_first_ai()
            self.market_scanner = await get_market_scanner()

            # Initialize kill switch
            self.kill_switch = EnhancedKillSwitch()
            await self.kill_switch.initialize()
            
            # Initialize Three-Stage Decision Pipeline
            self.devils_advocate = DevilsAdvocateSynapse()  # Stage 1: WCCA Survival Filter
            await self.devils_advocate.initialize()
            
            self.swarm_decision_engine = SwarmDecisionEngine()  # Stage 2: Naive Bayes Win-Rate Engine
            
            self.hyper_compound_engine = HyperCompoundEngine()  # Stage 3: Kelly Criterion Growth Maximizer
            await self.hyper_compound_engine.initialize()
            
            # Connect components for optimal integration
            self.hyper_compound_engine.trading_engine = self.trading_engine
            self.hyper_compound_engine.wallet_manager = self.wallet_manager
            self.hyper_compound_engine.vault_system = self.vault_system

            # Set up signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Start background tasks
            asyncio.create_task(self._trading_loop())
            asyncio.create_task(self._position_monitoring_loop())
            asyncio.create_task(self._performance_tracking_loop())
            asyncio.create_task(self._capital_management_loop())

            self.initialized = True
            self.trading_active = True
            self.start_time = datetime.now()

            self.logger.info("✅ Memecoin Trading Bot initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Bot initialization failed: {e}")
            return False

    async def _trading_loop(self):
        """Main trading loop"""
        while self.trading_active:
            try:
                # Check kill switch
                if self.kill_switch and self.kill_switch.is_triggered:
                    self.logger.warning("🛑 Kill switch triggered, stopping trading")
                    break

                # Scan for opportunities
                opportunities = await self.market_scanner.scan_markets()
                self.stats["opportunities_scanned"] += len(opportunities)

                # Process opportunities
                for opportunity in opportunities[:3]:  # Top 3 opportunities
                    await self._process_opportunity(opportunity)

                # Wait before next scan
                await asyncio.sleep(self.trading_config["scan_interval_seconds"])

            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)

    async def _process_opportunity(self, opportunity: Dict[str, Any]):
        """
        Process trading opportunity through Three-Stage Decision Pipeline
        
        Stage 1: Survival Filter (WCCA) - Risk-Adjusted Expected Loss veto
        Stage 2: Win-Rate Engine (Naive Bayes) - Probabilistic win assessment  
        Stage 3: Growth Maximizer (Kelly Criterion) - Optimal position sizing
        """
        try:
            token_address = opportunity['token_address']
            token_symbol = opportunity.get('token_symbol', 'Unknown')

            # Check if we already have a position
            if token_address in self.active_positions:
                return

            # Check if we have enough capital
            if not self._can_open_position():
                return

            self.logger.info(f"🔍 Three-stage analysis for {token_symbol}")
            
            # Prepare comprehensive trade parameters
            trade_params = {
                'token_address': token_address,
                'token_symbol': token_symbol,
                'amount': self._get_initial_position_estimate(),  # Initial estimate for R-EL calculation
                'token_age_hours': opportunity.get('age_hours', 24),
                'liquidity_concentration': opportunity.get('liquidity_concentration', 0.5),
                'dev_holdings_percent': opportunity.get('dev_holdings_percent', 0.0),
                'contract_verified': opportunity.get('contract_verified', True),
                'has_transfer_restrictions': opportunity.get('has_transfer_restrictions', False),
                'sell_buy_ratio': opportunity.get('sell_buy_ratio', 1.0),
                'has_blacklist_function': opportunity.get('has_blacklist_function', False),
                'volume_24h_sol': opportunity.get('volume_24h_sol', 0),
                'market_cap_usd': opportunity.get('market_cap_usd', 0),
            }
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 1: SURVIVAL FILTER (WCCA) - VETO CATASTROPHIC RISKS
            # ═══════════════════════════════════════════════════════════
            
            wcca_result = await self.devils_advocate.conduct_pre_mortem_analysis(trade_params)
            
            if wcca_result.get('veto', False):
                self.logger.warning(f"🚫 STAGE 1 VETO: {token_symbol} | {wcca_result.get('reason', 'Unknown')}")
                return  # Hard stop - trade vetoed
            
            self.logger.info(f"✅ STAGE 1 CLEAR: {token_symbol} | R-EL: {wcca_result.get('max_rel', 0):.4f} SOL")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 2: WIN-RATE ENGINE (NAIVE BAYES) - PROBABILITY CALC
            # ═══════════════════════════════════════════════════════════
            
            # Gather comprehensive market data for Naive Bayes analysis
            market_data = await self._gather_market_data(opportunity)
            
            # Calculate win probability using Naive Bayes
            win_probability = await self.swarm_decision_engine.analyze_opportunity(
                token_address, market_data, narrative_weight=1.0
            )
            
            # Check if win probability meets hunt threshold
            hunt_threshold = self.trading_config["hunt_threshold"]
            if win_probability < hunt_threshold:
                self.logger.info(f"⏸️ STAGE 2 SKIP: {token_symbol} | Win prob {win_probability:.3f} < {hunt_threshold}")
                return  # Don't hunt low-probability trades
            
            self.logger.info(f"✅ STAGE 2 HUNT: {token_symbol} | Win prob: {win_probability:.3f}")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 3: GROWTH MAXIMIZER (KELLY CRITERION) - POSITION SIZE
            # ═══════════════════════════════════════════════════════════
            
            # Calculate optimal position size using Kelly Criterion
            optimal_position_size = await self.hyper_compound_engine.calculate_optimal_position_size(
                win_probability=win_probability,
                current_capital=self.current_capital
            )
            
            if optimal_position_size <= 0:
                self.logger.warning(f"⏸️ STAGE 3 SKIP: {token_symbol} | Optimal size: {optimal_position_size:.4f} SOL")
                return  # No position recommended
            
            self.logger.info(f"✅ STAGE 3 SIZE: {token_symbol} | Kelly optimal: {optimal_position_size:.4f} SOL")
            
            # ═══════════════════════════════════════════════════════════
            # EXECUTION: ALL STAGES PASSED - EXECUTE TRADE
            # ═══════════════════════════════════════════════════════════
            
            await self._execute_optimized_trade(
                opportunity=opportunity,
                position_size=optimal_position_size,
                win_probability=win_probability,
                wcca_result=wcca_result,
                market_data=market_data
            )

        except Exception as e:
            self.logger.error(f"❌ Three-stage pipeline error for {opportunity.get('token_symbol', 'Unknown')}: {e}")
    
    async def _gather_market_data(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather comprehensive market data for Naive Bayes analysis
        Consolidates data from various AI ants and market sources
        """
        try:
            token_address = opportunity['token_address']
            
            # Base market data from opportunity
            market_data = {
                'current_price': opportunity.get('price', 0.0),
                'volume_24h_sol': opportunity.get('volume_24h_sol', 0),
                'market_cap_usd': opportunity.get('market_cap_usd', 0),
                'liquidity_sol': opportunity.get('liquidity_sol', 0),
                'price_change_24h': opportunity.get('price_change_24h_percent', 0),
                'age_hours': opportunity.get('age_hours', 24),
            }
            
            # Get sentiment analysis from AI
            if self.sentiment_ai:
                sentiment_result = await self.sentiment_ai.analyze_and_decide(token_address, market_data)
                market_data['sentiment_score'] = sentiment_result.sentiment_score
                market_data['social_buzz'] = sentiment_result.confidence
            
            # Add technical indicators
            market_data['volume_momentum'] = 1.0 if market_data['volume_24h_sol'] > 1000 else 0.5
            market_data['price_momentum'] = 1.0 if market_data['price_change_24h'] > 0 else 0.3
            market_data['liquidity_health'] = 1.0 if market_data['liquidity_sol'] > 500 else 0.5
            
            # Calculate additional signals
            market_data['rug_risk_score'] = opportunity.get('rug_risk_score', 0.3)
            market_data['narrative_strength'] = opportunity.get('narrative_strength', 0.5)
            market_data['whale_activity'] = opportunity.get('whale_activity', 0.5)
            
            # RSI and breakout signals (placeholder - would be calculated from price data)
            market_data['rsi'] = 45  # Placeholder
            market_data['volume_change_24h'] = 1.5  # Placeholder
            market_data['price_breakout_signal'] = 0.7  # Placeholder
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error gathering market data: {e}")
            return {}
    
    def _get_initial_position_estimate(self) -> float:
        """Get initial position size estimate for R-EL calculation in Stage 1"""
        return min(0.1, self.current_capital * 0.05)  # Conservative 5% estimate

    async def _execute_optimized_trade(self, opportunity: Dict[str, Any], position_size: float, win_probability: float, wcca_result: Dict[str, Any], market_data: Dict[str, Any]):
        """Execute the trade based on optimized parameters."""
        try:
            token_address = opportunity['token_address']
            token_symbol = opportunity['token_symbol']

            # Get best wallet for this trade
            wallet = await self.wallet_manager.get_best_wallet(
                {
                    "behavior": "sniper",  # Use sniper behavior for memecoins
                    "aggression": 0.8,  # High aggression
                    "risk_level": "high",
                }
            )

            if not wallet:
                self.logger.warning("No suitable wallet available")
                return

            # Execute buy order
            trade_params = {
                "amount": position_size,
                "wallet_id": wallet.wallet_id,
                "order_type": "buy",
                "metadata": {
                    "opportunity": opportunity,
                    "win_probability": win_probability,
                    "wcca_result": wcca_result,
                    "market_data": market_data,
                    "signal_snapshot": {
                        "timestamp": datetime.now().isoformat(),
                        "token_address": token_address,
                        "token_symbol": token_symbol,
                        "position_size": position_size,
                        "win_probability": win_probability,
                        "wcca_max_rel": wcca_result.get('max_rel', 0),
                        "market_data_price": market_data.get('current_price', 0),
                        "market_data_volume": market_data.get('volume_24h_sol', 0),
                        "market_data_liquidity": market_data.get('liquidity_sol', 0),
                        "market_data_price_change": market_data.get('price_change_24h', 0),
                        "market_data_sentiment_score": market_data.get('sentiment_score', 0),
                        "market_data_social_buzz": market_data.get('social_buzz', 0),
                        "market_data_volume_momentum": market_data.get('volume_momentum', 0),
                        "market_data_price_momentum": market_data.get('price_momentum', 0),
                        "market_data_liquidity_health": market_data.get('liquidity_health', 0),
                        "market_data_rug_risk": market_data.get('rug_risk_score', 0),
                        "market_data_narrative_strength": market_data.get('narrative_strength', 0),
                        "market_data_whale_activity": market_data.get('whale_activity', 0),
                        "market_data_rsi": market_data.get('rsi', 0),
                        "market_data_volume_change": market_data.get('volume_change_24h', 0),
                        "market_data_price_breakout": market_data.get('price_breakout_signal', 0),
                    }
                },
            }

            result = await self.trading_engine.execute_trade(token_address, trade_params)

            if result.success:
                # Record position
                position = {
                    "token_address": token_address,
                    "token_symbol": token_symbol,
                    "wallet_id": wallet.wallet_id,
                    "amount_sol": position_size,
                    "entry_price": 0.0,  # Will be updated
                    "entry_time": datetime.now(),
                    "win_probability": win_probability,
                    "wcca_result": wcca_result,
                    "market_data": market_data,
                    "transaction_signature": result.transaction_signature,
                }

                self.active_positions[token_address] = position
                self.stats["positions_opened"] += 1

                self.logger.info(f"💰 Opened position: {token_symbol} ({position_size:.4f} SOL)")

                # Update wallet performance
                await self.wallet_manager.update_wallet_performance(
                    wallet.wallet_id,
                    {
                        "success": True,
                        "profit_sol": 0.0,  # Will be calculated on exit
                        "trade_type": "buy",
                    },
                )

            else:
                self.logger.error(f"Failed to open position for {token_symbol}: {result.error_message}")

        except Exception as e:
            self.logger.error(f"Error executing optimized trade: {e}")

    def _calculate_position_size(self, opportunity: Dict[str, Any], sentiment_decision: Any) -> float:
        """Calculate position size based on opportunity and sentiment.

        Args:
            opportunity: Trading opportunity data
            sentiment_decision: Sentiment analysis result

        Returns:
            Calculated position size in SOL
        """
        try:
            # Base position size
            base_size = self.current_capital * self.trading_config["position_size_percent"]

            # Adjust based on sentiment score
            sentiment_multiplier = 1.0 + (sentiment_decision.sentiment_score - 0.3) * 2.0

            # Adjust based on confidence
            confidence_multiplier = 1.0 + (sentiment_decision.confidence - 0.5) * 1.0

            # Adjust based on expected profit
            profit_multiplier = 1.0 + sentiment_decision.expected_profit * 2.0

            # Calculate final position size
            position_size = (
                base_size * sentiment_multiplier * confidence_multiplier * profit_multiplier
            )

            # Apply limits
            position_size = min(position_size, self.trading_config["max_position_size_sol"])
            position_size = min(position_size, self.current_capital * 0.5)  # Max 50% of capital

            return max(position_size, 0.1)  # Minimum 0.1 SOL

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.1

    def _can_open_position(self) -> bool:
        """Check if we can open a new position.

        Returns:
            True if position can be opened, False otherwise
        """
        try:
            # Check number of active positions
            if len(self.active_positions) >= self.trading_config["max_concurrent_positions"]:
                return False

            # Check available capital
            if self.current_capital < 1.0:  # Need at least 1 SOL
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking position availability: {e}")
            return False

    async def _position_monitoring_loop(self):
        """Monitor active positions for exit conditions"""
        while self.trading_active:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                current_time = datetime.now()
                positions_to_close = []

                for token_address, position in self.active_positions.items():
                    # Check hold time
                    hold_time = current_time - position['entry_time']
                    if hold_time.total_seconds() > self.trading_config['max_hold_time_hours'] * 3600:
                        positions_to_close.append((token_address, 'time_limit'))
                        continue

                    # Check sentiment for exit
                    await self._check_sentiment_exit(token_address, position)

                    # Check profit/loss
                    await self._check_profit_loss_exit(token_address, position)

                # Close positions
                for token_address, reason in positions_to_close:
                    await self._close_position(token_address, reason)

            except Exception as e:
                self.logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(30)

    async def _check_sentiment_exit(self, token_address: str, position: Dict[str, Any]):
        """Check if sentiment has deteriorated enough to exit"""
        try:
            # Get current sentiment
            market_data = {
                'symbol': position['token_symbol'],
                'price': 0.0,
                'volume': position['opportunity']['volume_24h_sol'],
                'liquidity': position['opportunity']['liquidity_sol']
            }

            sentiment_decision = await self.sentiment_ai.analyze_and_decide(
                token_address, market_data
            )

            # Exit if sentiment has turned negative
            if sentiment_decision.decision == SentimentDecisionEnum.SELL.value and sentiment_decision.confidence >= 0.7:
                await self._close_position(token_address, 'sentiment_exit')

        except Exception as e:
            self.logger.error(f"Error checking sentiment exit: {e}")

    async def _check_profit_loss_exit(self, token_address: str, position: Dict[str, Any]):
        """Check profit/loss exit conditions"""
        try:
            # Get current price
            current_price = await self.market_scanner.get_token_price(token_address)
            if not current_price:
                return

            # Calculate profit/loss
            entry_price = position.get('entry_price', 0)
            if entry_price > 0:
                profit_loss_pct = (current_price - entry_price) / entry_price

                # Check profit target
                if profit_loss_pct >= self.trading_config['min_profit_target']:
                    await self._close_position(token_address, 'profit_target')
                    return

                # Check stop loss
                if profit_loss_pct <= -self.trading_config['max_loss_percent']:
                    await self._close_position(token_address, 'stop_loss')
                    return

        except Exception as e:
            self.logger.error(f"Error checking profit/loss exit: {e}")

    async def _close_position(self, token_address: str, reason: str):
        """Close a trading position"""
        try:
            position = self.active_positions[token_address]
            wallet_id = position['wallet_id']

            # Get wallet keypair
            wallet = await self.wallet_manager.get_wallet_keypair(wallet_id)
            if not wallet:
                self.logger.error(f"Could not get wallet keypair for {wallet_id}")
                return

            # Execute sell order
            trade_params = {
                'amount': position['amount_sol'],
                'wallet_id': wallet_id,
                'order_type': 'sell',
                'metadata': {
                    'close_reason': reason,
                    'position': position
                }
            }

            result = await self.trading_engine.execute_trade(token_address, trade_params)

            if result.success:
                # Calculate profit/loss
                profit_sol = result.profit_sol

                # Update statistics
                self.stats['trades_executed'] += 1
                self.stats['total_profit_sol'] += profit_sol
                self.current_capital += profit_sol

                if profit_sol > 0:
                    self.stats["successful_trades"] += 1

                # Update position history
                position["exit_time"] = datetime.now()
                position["exit_price"] = 0.0  # Would be calculated from result
                position["profit_sol"] = profit_sol
                position["close_reason"] = reason
                position["transaction_signature"] = result.transaction_signature

                self.position_history.append(position)
                del self.active_positions[token_address]

                self.stats["positions_closed"] += 1

                self.logger.info(f"💸 Closed position: {position['token_symbol']} - Profit: {profit_sol:.4f} SOL ({reason})")

                # Update wallet performance
                await self.wallet_manager.update_wallet_performance(
                    wallet_id,
                    {
                        "success": profit_sol > 0,
                        "profit_sol": profit_sol,
                        "trade_type": "sell",
                    },
                )

                # Compound profits if enabled
                if self.trading_config["compounding_enabled"] and profit_sol > 0:
                    await self._compound_profits(profit_sol)

            else:
                self.logger.error(f"Failed to close position for {position['token_symbol']}: {result.error_message}")

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

    async def _compound_profits(self, profit_sol: float):
        """Compound profits back into trading capital.

        Args:
            profit_sol: Profit amount in SOL to compound
        """
        try:
            # Add 80% of profits back to trading capital
            compound_amount = profit_sol * 0.8
            self.current_capital += compound_amount

            # Send 20% to vault for safety
            vault_amount = profit_sol * 0.2
            await self.vault_system.deposit_profits(vault_amount)

            self.logger.info(
                f"💰 Compounded {compound_amount:.4f} SOL, vaulted {vault_amount:.4f} SOL"
            )

        except Exception as e:
            self.logger.error(f"Error compounding profits: {e}")

    async def _performance_tracking_loop(self):
        """Track and log performance metrics"""
        while self.trading_active:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Calculate statistics
                if self.stats["trades_executed"] > 0:
                    self.stats["win_rate"] = (
                        self.stats["successful_trades"] / self.stats["trades_executed"]
                    )
                    self.stats["avg_profit_per_trade"] = (
                        self.stats["total_profit_sol"] / self.stats["trades_executed"]
                    )

                # Calculate drawdown
                if self.current_capital < self.initial_capital:
                    drawdown = (
                        (self.initial_capital - self.current_capital) / self.initial_capital
                    )
                    self.stats["max_drawdown"] = max(self.stats["max_drawdown"], drawdown)

                # Log performance summary
                await self._log_performance_summary()

            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(300)

    async def _capital_management_loop(self):
        """Manage trading capital and risk"""
        while self.trading_active:
            try:
                await asyncio.sleep(60)  # Every minute

                # Check if we need to reduce risk
                if self.current_capital < self.initial_capital * 0.8:  # 20% drawdown
                    self.logger.warning(
                        "⚠️ Significant drawdown detected, reducing position sizes"
                    )
                    self.trading_config["position_size_percent"] *= 0.5

                # Check if we can increase risk
                elif self.current_capital > self.initial_capital * 1.5:  # 50% profit
                    self.logger.info("📈 Significant profits, increasing position sizes")
                    self.trading_config["position_size_percent"] = min(
                        self.trading_config["position_size_percent"] * 1.2,
                        0.3,  # Max 30% per position
                    )

            except Exception as e:
                self.logger.error(f"Capital management error: {e}")
                await asyncio.sleep(60)

    async def _log_performance_summary(self):
        """Log performance summary."""
        try:
            runtime = datetime.now() - self.start_time
            runtime_hours = runtime.total_seconds() / 3600

            self.logger.info("📊 Performance Summary:")
            self.logger.info(f"   Runtime: {runtime_hours:.1f} hours")
            self.logger.info(
                f"   Capital: ${self.current_capital:.2f} (${self.initial_capital:.2f} initial)"
            )
            self.logger.info(f"   Total Profit: {self.stats['total_profit_sol']:.4f} SOL")
            self.logger.info(f"   Win Rate: {self.stats['win_rate']:.1%}")
            self.logger.info(f"   Trades: {self.stats['trades_executed']}")
            self.logger.info(f"   Active Positions: {len(self.active_positions)}")
            self.logger.info(f"   Max Drawdown: {self.stats['max_drawdown']:.1%}")

        except Exception as e:
            self.logger.error(f"Error logging performance summary: {e}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.logger.info(f"🛑 Received signal {signum}, shutting down...")
        asyncio.create_task(self.shutdown())

    async def shutdown(self):
        """Shutdown the trading bot."""
        try:
            self.logger.info("🛑 Shutting down Memecoin Trading Bot...")

            # Stop trading
            self.trading_active = False

            # Close all active positions
            for token_address in list(self.active_positions.keys()):
                await self._close_position(token_address, "shutdown")

            # Shutdown core systems
            if self.trading_engine:
                await self.trading_engine.shutdown()
            if self.wallet_manager:
                await self.wallet_manager.shutdown()
            if self.vault_system:
                await self.vault_system.shutdown()
            if self.kill_switch:
                await self.kill_switch.shutdown()

            # Log final statistics
            await self._log_performance_summary()

            self.logger.info("✅ Memecoin Trading Bot shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get bot status.

        Returns:
            Dictionary containing bot status information
        """
        try:
            runtime = (
                datetime.now() - self.start_time if self.start_time else timedelta(0)
            )

            return {
                "initialized": self.initialized,
                "trading_active": self.trading_active,
                "runtime_hours": runtime.total_seconds() / 3600,
                "initial_capital": self.initial_capital,
                "current_capital": self.current_capital,
                "total_profit_sol": self.stats["total_profit_sol"],
                "win_rate": self.stats["win_rate"],
                "active_positions": len(self.active_positions),
                "stats": self.stats,
                "config": self.trading_config,
            }

        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {"error": str(e)}

