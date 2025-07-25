"""
SIMPLIFIED TRADING BOT - LEAN MATHEMATICAL CORE
==============================================

🎯 MISSION: Three-Stage Decision Pipeline - Survive -> Quantify Edge -> Bet Optimally
💰 STRATEGY: Mathematical precision without complex abstraction layers
🧬 EVOLUTION: Lean, maintainable, and mathematically pure

This simplified bot retains the mathematical core that gives Antbot its edge:
1. STAGE 1 - SURVIVAL FILTER: Enhanced Rug Detection + WCCA Risk Analysis
2. STAGE 2 - WIN-RATE ENGINE: Naive Bayes probability calculation  
3. STAGE 3 - GROWTH MAXIMIZER: Kelly Criterion position sizing

Eliminates all complex management layers, multi-agent systems, and abstraction
while preserving the profitable mathematical foundation.
"""

import asyncio
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Core mathematical components (preserved)
from worker_ant_v1.intelligence.enhanced_rug_detector import EnhancedRugDetector
from worker_ant_v1.trading.devils_advocate_synapse import DevilsAdvocateSynapse
from worker_ant_v1.intelligence.sentiment_first_ai import SentimentFirstAI
from worker_ant_v1.intelligence.technical_analyzer import TechnicalAnalyzer

# Essential systems
from worker_ant_v1.core.unified_trading_engine import UnifiedTradingEngine
from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
from worker_ant_v1.safety.vault_wallet_system import VaultWalletSystem
from worker_ant_v1.trading.market_scanner import RealMarketScanner
from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch
from worker_ant_v1.utils.logger import get_logger


@dataclass
class SimplifiedConfig:
    """Lean configuration for simplified bot"""
    initial_capital_sol: float = 1.5  # ~$300
    
    # Stage 1: Survival Filter
    acceptable_rel_threshold: float = 0.1  # Max Risk-Adjusted Expected Loss
    
    # Stage 2: Win-Rate Engine  
    hunt_threshold: float = 0.6  # Minimum win probability to trade
    
    # Stage 3: Growth Maximizer
    kelly_fraction: float = 0.25  # Fraction of full Kelly to use
    max_position_percent: float = 0.20  # 20% max position size
    
    # Compounding (simplified)
    compound_rate: float = 0.8  # Fixed 80% reinvestment
    compound_threshold_sol: float = 0.2  # Compound when vault has 0.2 SOL
    
    # Risk management
    stop_loss_percent: float = 0.05  # 5% stop loss
    max_hold_time_hours: float = 4.0  # 4 hour max hold
    
    # Scanning
    scan_interval_seconds: int = 30


@dataclass
class TradingMetrics:
    """Simple performance tracking"""
    trades_executed: int = 0
    successful_trades: int = 0
    total_profit_sol: float = 0.0
    current_capital_sol: float = 0.0
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    total_compounds: int = 0


class SimplifiedTradingBot:
    """
    Lean trading bot implementing the three-stage mathematical pipeline
    without complex management layers or multi-agent abstractions
    """
    
    def __init__(self, config: SimplifiedConfig = None):
        self.logger = get_logger("SimplifiedTradingBot")
        self.config = config or SimplifiedConfig()
        
        # Core mathematical components (THE ESSENTIAL TRIO)
        self.rug_detector: Optional[EnhancedRugDetector] = None
        self.devils_advocate: Optional[DevilsAdvocateSynapse] = None  # WCCA
        self.sentiment_ai: Optional[SentimentFirstAI] = None
        self.technical_analyzer: Optional[TechnicalAnalyzer] = None
        
        # Essential systems (minimal required)
        self.trading_engine: Optional[UnifiedTradingEngine] = None
        self.wallet_manager: Optional[UnifiedWalletManager] = None
        self.vault_system: Optional[VaultWalletSystem] = None
        self.market_scanner: Optional[RealMarketScanner] = None
        self.kill_switch: Optional[EnhancedKillSwitch] = None
        
        # Simple state tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.metrics = TradingMetrics(current_capital_sol=self.config.initial_capital_sol)
        self.running = False
        self.initialized = False
        
        # Enhanced Naive Bayes with dynamic learning
        self.signal_probabilities = {
            'p_win_base': 0.55,  # Base win probability (updated from historical data)
            'p_loss_base': 0.45, # Base loss probability (updated from historical data)
            'signal_conditionals': {},  # P(Signal|Win), P(Signal|Loss) - learned from history
            'total_trades': 0,
            'successful_trades': 0,
            'signal_history': [],  # Store recent signal outcomes for learning
            'last_update': datetime.now()
        }
        
        # Kelly Criterion parameters
        self.cached_win_loss_ratio = 1.5  # Default, updated from historical data
        
        self.logger.info("🎯 SimplifiedTradingBot initialized - Mathematical core only")
    
    async def initialize(self) -> bool:
        """Initialize essential systems only"""
        try:
            self.logger.info("🚀 Initializing simplified trading bot...")
            
            # Initialize essential systems
            self.trading_engine = UnifiedTradingEngine()
            await self.trading_engine.initialize()
            
            from worker_ant_v1.core.wallet_manager import get_wallet_manager
            from worker_ant_v1.safety.vault_wallet_system import get_vault_system
            from worker_ant_v1.trading.market_scanner import get_market_scanner
            
            self.wallet_manager = await get_wallet_manager()
            self.vault_system = await get_vault_system()
            self.market_scanner = await get_market_scanner()
            
            # Initialize kill switch
            self.kill_switch = EnhancedKillSwitch()
            await self.kill_switch.initialize()
            
            # Initialize mathematical core (THE ESSENTIAL TRIO)
            self.rug_detector = EnhancedRugDetector()
            await self.rug_detector.initialize()
            
            self.devils_advocate = DevilsAdvocateSynapse()
            await self.devils_advocate.initialize()
            
            self.sentiment_ai = SentimentFirstAI()
            await self.sentiment_ai.initialize()
            
            self.technical_analyzer = TechnicalAnalyzer()
            await self.technical_analyzer.initialize()
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.initialized = True
            self.logger.info("✅ Simplified trading bot initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize simplified bot: {e}")
            return False
    
    async def run(self):
        """Main trading loop - simplified and direct"""
        try:
            if not self.initialized:
                self.logger.error("❌ Bot not initialized")
                return
            
            self.running = True
            self.logger.info("🔥 Starting simplified trading bot...")
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self._trading_loop()),
                asyncio.create_task(self._position_monitoring_loop()),
                asyncio.create_task(self._compounding_loop()),
                asyncio.create_task(self._metrics_update_loop())
            ]
            
            # Run until shutdown
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in trading bot: {e}")
            await self.emergency_shutdown()
    
    async def _trading_loop(self):
        """Main trading loop - scan and process opportunities"""
        while self.running:
            try:
                # Scan for opportunities
                opportunities = await self.market_scanner.scan_opportunities()
                
                # Process each opportunity through three-stage pipeline
                for opportunity in opportunities:
                    await self._process_opportunity_pipeline(opportunity)
                
                await asyncio.sleep(self.config.scan_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _process_opportunity_pipeline(self, opportunity: Dict[str, Any]):
        """
        Three-Stage Decision Pipeline - The Mathematical Core
        
        STAGE 1: SURVIVAL FILTER - Enhanced Rug Detection + WCCA 
        STAGE 2: WIN-RATE ENGINE - Naive Bayes probability calculation
        STAGE 3: GROWTH MAXIMIZER - Kelly Criterion position sizing
        """
        try:
            token_address = opportunity['token_address']
            token_symbol = opportunity.get('token_symbol', 'Unknown')
            
            # Skip if we already have a position
            if token_address in self.active_positions:
                return
            
            # Skip if insufficient capital
            if self.metrics.current_capital_sol < 0.01:  # Need at least 0.01 SOL
                return
            
            self.logger.info(f"🔍 Three-stage pipeline for {token_symbol}")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 1: SURVIVAL FILTER - VETO CATASTROPHIC RISKS
            # ═══════════════════════════════════════════════════════════
            
            # Enhanced rug detection
            rug_result = await self.rug_detector.analyze_token(
                token_address, 
                opportunity.get('token_name', ''),
                token_symbol
            )
            
            if rug_result.detection_level.value in ['high', 'critical']:
                self.logger.warning(f"🚫 STAGE 1 RUG VETO: {token_symbol} | Level: {rug_result.detection_level.value}")
                return
            
            # WCCA Risk-Adjusted Expected Loss analysis
            trade_params = {
                'token_address': token_address,
                'token_symbol': token_symbol,
                'amount': 0.1,  # Initial estimate for R-EL calculation
                'token_age_hours': opportunity.get('age_hours', 24),
                'liquidity_concentration': opportunity.get('liquidity_concentration', 0.5),
                'dev_holdings_percent': opportunity.get('dev_holdings_percent', 0.0),
                'contract_verified': opportunity.get('contract_verified', True),
                'has_transfer_restrictions': opportunity.get('has_transfer_restrictions', False),
                'sell_buy_ratio': opportunity.get('sell_buy_ratio', 1.0),
                'rug_detector_score': rug_result.overall_risk
            }
            
            wcca_result = await self.devils_advocate.conduct_pre_mortem_analysis(trade_params)
            
            if wcca_result.get('veto', False):
                self.logger.warning(f"🚫 STAGE 1 WCCA VETO: {token_symbol} | {wcca_result.get('reason', 'Unknown')}")
                return
            
            self.logger.info(f"✅ STAGE 1 CLEAR: {token_symbol} | Max R-EL: {wcca_result.get('max_rel', 0):.4f} SOL")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 2: WIN-RATE ENGINE - NAIVE BAYES PROBABILITY
            # ═══════════════════════════════════════════════════════════
            
            # Gather signals for Naive Bayes
            market_data = await self._gather_market_signals(opportunity)
            
            # Calculate win probability using Enhanced Naive Bayes
            win_probability = await self._calculate_naive_bayes_probability(market_data)
            
            if win_probability < self.config.hunt_threshold:
                self.logger.info(f"⏸️ STAGE 2 SKIP: {token_symbol} | Win prob {win_probability:.3f} < {self.config.hunt_threshold}")
                return
            
            self.logger.info(f"✅ STAGE 2 HUNT: {token_symbol} | Win probability: {win_probability:.3f}")
            
            # ═══════════════════════════════════════════════════════════
            # STAGE 3: GROWTH MAXIMIZER - KELLY CRITERION SIZING
            # ═══════════════════════════════════════════════════════════
            
            # Calculate optimal position size using Kelly Criterion
            optimal_position_size = self._calculate_kelly_position_size(
                win_probability, 
                self.metrics.current_capital_sol
            )
            
            if optimal_position_size <= 0.005:  # Minimum viable position
                self.logger.warning(f"⏸️ STAGE 3 SKIP: {token_symbol} | Position too small: {optimal_position_size:.4f} SOL")
                return
            
            self.logger.info(f"✅ STAGE 3 SIZE: {token_symbol} | Kelly optimal: {optimal_position_size:.4f} SOL")
            
            # ═══════════════════════════════════════════════════════════
            # EXECUTION: ALL STAGES PASSED - EXECUTE TRADE
            # ═══════════════════════════════════════════════════════════
            
            await self._execute_trade(
                opportunity=opportunity,
                position_size=optimal_position_size,
                win_probability=win_probability,
                entry_signals=market_data  # Store signals for learning
            )
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline error for {opportunity.get('token_symbol', 'Unknown')}: {e}")
    
    async def _gather_market_signals(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Gather market signals for Naive Bayes analysis"""
        try:
            token_address = opportunity['token_address']
            
            # Get sentiment analysis
            sentiment_result = await self.sentiment_ai.analyze_sentiment(token_address)
            
            # Get technical analysis  
            technical_result = await self.technical_analyzer.analyze_token(token_address)
            
            # Compile signals
            signals = {
                'sentiment_score': sentiment_result.sentiment_score if sentiment_result else 0.5,
                'confidence': sentiment_result.confidence if sentiment_result else 0.5,
                'social_buzz': sentiment_result.social_buzz_score if sentiment_result else 0.5,
                'volume_momentum': opportunity.get('volume_change_24h', 1.0),
                'price_momentum': opportunity.get('price_change_24h', 0.0),
                'liquidity_health': 1.0 - opportunity.get('liquidity_concentration', 0.5),
                'rsi_oversold': 1.0 if technical_result and getattr(technical_result, 'rsi', 50) < 30 else 0.0,
                'volume_spike': 1.0 if opportunity.get('volume_change_24h', 1.0) > 2.0 else 0.0
            }
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error gathering market signals: {e}")
            return {}
    
    async def _calculate_naive_bayes_probability(self, signals: Dict[str, Any]) -> float:
        """
        Enhanced Naive Bayes probability calculation with dynamic learning
        P(Win | Signals) ∝ P(Win) * Π P(Signal_i | Win)
        """
        try:
            if not signals:
                return 0.5  # Default probability
            
            # Update learning from historical data periodically
            await self._update_signal_probabilities_if_needed()
            
            # Get base probabilities (updated from historical performance)
            p_win = self.signal_probabilities['p_win_base']
            p_loss = self.signal_probabilities['p_loss_base']
            
            # Initialize likelihoods
            likelihood_win = p_win
            likelihood_loss = p_loss
            
            # Apply each signal using learned conditional probabilities
            signal_contributions = 0
            
            for signal_name, signal_value in signals.items():
                if self._is_positive_signal(signal_value):
                    # Get learned conditional probabilities or use defaults
                    conditionals = self.signal_probabilities['signal_conditionals'].get(signal_name, {
                        'p_signal_given_win': 0.7,  # Default if not learned
                        'p_signal_given_loss': 0.3,
                        'sample_count': 0
                    })
                    
                    p_signal_given_win = conditionals['p_signal_given_win']
                    p_signal_given_loss = conditionals['p_signal_given_loss']
                    
                    # Apply confidence weighting based on sample size
                    sample_count = conditionals.get('sample_count', 0)
                    confidence = min(1.0, sample_count / 100.0)  # Full confidence after 100 samples
                    
                    # Blend learned probabilities with defaults based on confidence
                    default_win = 0.7
                    default_loss = 0.3
                    
                    weighted_p_win = (confidence * p_signal_given_win) + ((1 - confidence) * default_win)
                    weighted_p_loss = (confidence * p_signal_given_loss) + ((1 - confidence) * default_loss)
                    
                    likelihood_win *= weighted_p_win
                    likelihood_loss *= weighted_p_loss
                    signal_contributions += 1
                    
                    self.logger.debug(f"Signal {signal_name}: P(Win)={weighted_p_win:.3f}, P(Loss)={weighted_p_loss:.3f}, Confidence={confidence:.2f}")
            
            # Normalize and calculate final probability
            total_likelihood = likelihood_win + likelihood_loss
            if total_likelihood > 0:
                final_probability = likelihood_win / total_likelihood
            else:
                final_probability = 0.5
            
            # Apply smoothing to prevent extreme probabilities
            final_probability = max(0.05, min(0.95, final_probability))
            
            self.logger.debug(f"📊 Enhanced Naive Bayes: {final_probability:.3f} ({signal_contributions} signals, "
                            f"base P(Win)={p_win:.3f})")
            return final_probability
            
        except Exception as e:
            self.logger.error(f"Error in Enhanced Naive Bayes calculation: {e}")
            return 0.5
    
    def _is_positive_signal(self, value: Any) -> bool:
        """Determine if a signal value is positive"""
        try:
            if isinstance(value, (int, float)):
                return value > 0.5  # Threshold for positive signal
            return bool(value)
        except:
            return False
    
    async def _update_signal_probabilities_if_needed(self):
        """Update signal probabilities from historical data if needed"""
        try:
            current_time = datetime.now()
            last_update = self.signal_probabilities.get('last_update', current_time)
            
            # Update every hour or if we have new trade data
            if (current_time - last_update).total_seconds() > 3600:  # 1 hour
                await self._learn_from_historical_data()
                self.signal_probabilities['last_update'] = current_time
                
        except Exception as e:
            self.logger.error(f"Error updating signal probabilities: {e}")
    
    async def _learn_from_historical_data(self):
        """Learn signal probabilities from historical trade data"""
        try:
            # Update base win/loss probabilities from recent performance
            if self.metrics.trades_executed > 0:
                historical_win_rate = self.metrics.successful_trades / self.metrics.trades_executed
                
                # Use exponential moving average to update base probabilities
                alpha = 0.1  # Learning rate
                current_p_win = self.signal_probabilities['p_win_base']
                new_p_win = alpha * historical_win_rate + (1 - alpha) * current_p_win
                
                self.signal_probabilities['p_win_base'] = max(0.1, min(0.9, new_p_win))
                self.signal_probabilities['p_loss_base'] = 1.0 - self.signal_probabilities['p_win_base']
                
                self.logger.debug(f"Updated base probabilities: P(Win)={new_p_win:.3f} from {self.metrics.trades_executed} trades")
            
            # Learn conditional probabilities from signal history
            await self._update_conditional_probabilities()
            
        except Exception as e:
            self.logger.error(f"Error learning from historical data: {e}")
    
    async def _update_conditional_probabilities(self):
        """Update conditional probabilities P(Signal|Win) and P(Signal|Loss) from signal history"""
        try:
            signal_history = self.signal_probabilities.get('signal_history', [])
            
            if len(signal_history) < 10:  # Need minimum data
                return
            
            # Group by signal name and outcome
            signal_stats = {}
            
            for record in signal_history:
                signals = record.get('signals', {})
                was_successful = record.get('successful', False)
                
                for signal_name, signal_value in signals.items():
                    if signal_name not in signal_stats:
                        signal_stats[signal_name] = {
                            'positive_wins': 0,
                            'positive_losses': 0,
                            'total_wins': 0,
                            'total_losses': 0
                        }
                    
                    is_positive = self._is_positive_signal(signal_value)
                    
                    if was_successful:
                        signal_stats[signal_name]['total_wins'] += 1
                        if is_positive:
                            signal_stats[signal_name]['positive_wins'] += 1
                    else:
                        signal_stats[signal_name]['total_losses'] += 1
                        if is_positive:
                            signal_stats[signal_name]['positive_losses'] += 1
            
            # Calculate conditional probabilities
            for signal_name, stats in signal_stats.items():
                total_wins = stats['total_wins']
                total_losses = stats['total_losses']
                
                if total_wins > 0 and total_losses > 0:
                    p_signal_given_win = stats['positive_wins'] / total_wins
                    p_signal_given_loss = stats['positive_losses'] / total_losses
                    
                    # Apply smoothing to prevent extreme values
                    p_signal_given_win = max(0.1, min(0.9, p_signal_given_win))
                    p_signal_given_loss = max(0.1, min(0.9, p_signal_given_loss))
                    
                    self.signal_probabilities['signal_conditionals'][signal_name] = {
                        'p_signal_given_win': p_signal_given_win,
                        'p_signal_given_loss': p_signal_given_loss,
                        'sample_count': total_wins + total_losses
                    }
                    
                    self.logger.debug(f"Learned {signal_name}: P(+|Win)={p_signal_given_win:.3f}, "
                                    f"P(+|Loss)={p_signal_given_loss:.3f} from {total_wins + total_losses} samples")
            
        except Exception as e:
            self.logger.error(f"Error updating conditional probabilities: {e}")
    
    def _record_signal_outcome(self, signals: Dict[str, Any], successful: bool):
        """Record signal outcome for learning"""
        try:
            record = {
                'timestamp': datetime.now(),
                'signals': signals.copy(),
                'successful': successful
            }
            
            # Add to history (keep last 1000 records)
            signal_history = self.signal_probabilities.get('signal_history', [])
            signal_history.append(record)
            
            # Keep only recent history
            if len(signal_history) > 1000:
                signal_history = signal_history[-1000:]
            
            self.signal_probabilities['signal_history'] = signal_history
            
        except Exception as e:
            self.logger.error(f"Error recording signal outcome: {e}")
    
    def _calculate_kelly_position_size(self, win_probability: float, current_capital: float) -> float:
        """
        Kelly Criterion position sizing calculation
        f* = p - ((1 - p) / b) where p = win probability, b = win/loss ratio
        """
        try:
            # Ensure valid inputs
            win_probability = max(0.01, min(0.99, win_probability))
            
            # Get win/loss ratio (simplified - could be updated from historical data)
            win_loss_ratio = self.cached_win_loss_ratio
            
            # Calculate Kelly fraction
            kelly_fraction_full = win_probability - ((1 - win_probability) / win_loss_ratio)
            
            # Apply safety factor (fractional Kelly)
            kelly_fraction_safe = kelly_fraction_full * self.config.kelly_fraction
            
            # Calculate position size
            position_size = kelly_fraction_safe * current_capital
            
            # Apply maximum position size limit
            max_position = current_capital * self.config.max_position_percent
            position_size = min(position_size, max_position)
            
            # Ensure minimum viable position
            position_size = max(0.0, position_size)
            
            self.logger.debug(f"💰 Kelly: p={win_probability:.3f}, b={win_loss_ratio:.2f}, "
                            f"Kelly={kelly_fraction_full:.3f}, Safe={kelly_fraction_safe:.3f}, "
                            f"Size={position_size:.4f} SOL")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error in Kelly Criterion calculation: {e}")
            return current_capital * 0.01  # Conservative fallback
    
    async def _execute_trade(self, opportunity: Dict[str, Any], position_size: float, win_probability: float, entry_signals: Dict[str, Any] = None):
        """Execute trade with simplified, direct execution"""
        try:
            token_address = opportunity['token_address']
            token_symbol = opportunity.get('token_symbol', 'Unknown')
            
            # Execute buy order
            result = await self.trading_engine.execute_buy_order(
                token_address=token_address,
                amount_sol=position_size,
                max_slippage_bps=500  # 5% max slippage
            )
            
            if result.get('success', False):
                # Track position with entry signals for learning
                position = {
                    'token_address': token_address,
                    'token_symbol': token_symbol,
                    'position_size_sol': position_size,
                    'entry_price': result.get('execution_price', 0),
                    'entry_time': datetime.now(),
                    'win_probability': win_probability,
                    'stop_loss_price': result.get('execution_price', 0) * (1 - self.config.stop_loss_percent),
                    'target_profit': result.get('execution_price', 0) * 1.15,  # 15% target
                    'max_hold_until': datetime.now() + timedelta(hours=self.config.max_hold_time_hours),
                    'entry_signals': entry_signals or {}  # Store for learning
                }
                
                self.active_positions[token_address] = position
                self.metrics.current_capital_sol -= position_size
                
                self.logger.info(f"✅ TRADE EXECUTED: {token_symbol} | Size: {position_size:.4f} SOL | "
                               f"Price: {result.get('execution_price', 0):.6f}")
                
            else:
                self.logger.error(f"❌ Trade execution failed for {token_symbol}: {result.get('error', 'Unknown')}")
                
        except Exception as e:
            self.logger.error(f"❌ Trade execution error: {e}")
    
    async def _position_monitoring_loop(self):
        """Monitor active positions for exits"""
        while self.running:
            try:
                positions_to_close = []
                
                for token_address, position in self.active_positions.items():
                    should_close, reason = await self._should_close_position(position)
                    if should_close:
                        positions_to_close.append((token_address, position, reason))
                
                # Close positions
                for token_address, position, reason in positions_to_close:
                    await self._close_position(token_address, position, reason)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in position monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _should_close_position(self, position: Dict[str, Any]) -> tuple[bool, str]:
        """Determine if position should be closed"""
        try:
            token_address = position['token_address']
            
            # Get current price
            current_price = await self.trading_engine.get_token_price(token_address)
            entry_price = position['entry_price']
            
            # Check stop loss
            if current_price <= position['stop_loss_price']:
                return True, "stop_loss"
            
            # Check profit target
            if current_price >= position['target_profit']:
                return True, "profit_target"
            
            # Check max hold time
            if datetime.now() >= position['max_hold_until']:
                return True, "max_hold_time"
            
            return False, ""
            
        except Exception as e:
            self.logger.error(f"Error checking position close condition: {e}")
            return True, "error"  # Close on error for safety
    
    async def _close_position(self, token_address: str, position: Dict[str, Any], reason: str):
        """Close position and update metrics"""
        try:
            token_symbol = position['token_symbol']
            
            # Execute sell order
            result = await self.trading_engine.execute_sell_order(
                token_address=token_address,
                max_slippage_bps=500
            )
            
            if result.get('success', False):
                # Calculate profit/loss
                exit_price = result.get('execution_price', 0)
                entry_price = position['entry_price']
                position_size = position['position_size_sol']
                
                # Simple P&L calculation (simplified)
                pnl_sol = position_size * (exit_price / entry_price - 1) if entry_price > 0 else 0
                
                # Update metrics
                self.metrics.trades_executed += 1
                self.metrics.total_profit_sol += pnl_sol
                self.metrics.current_capital_sol += position_size + pnl_sol
                
                if pnl_sol > 0:
                    self.metrics.successful_trades += 1
                
                # Update win rate
                self.metrics.win_rate = self.metrics.successful_trades / self.metrics.trades_executed if self.metrics.trades_executed > 0 else 0
                self.metrics.avg_profit_per_trade = self.metrics.total_profit_sol / self.metrics.trades_executed if self.metrics.trades_executed > 0 else 0
                
                # Record signal outcome for Naive Bayes learning
                if 'entry_signals' in position:
                    was_successful = pnl_sol > 0
                    self._record_signal_outcome(position['entry_signals'], was_successful)
                    self.logger.debug(f"📚 Recorded signal outcome for learning: {'WIN' if was_successful else 'LOSS'}")
                
                # Remove from active positions
                del self.active_positions[token_address]
                
                profit_status = "PROFIT" if pnl_sol > 0 else "LOSS"
                self.logger.info(f"✅ POSITION CLOSED: {token_symbol} | {profit_status}: {pnl_sol:.4f} SOL | Reason: {reason}")
                
                # Send profit to vault for compounding
                if pnl_sol > 0:
                    await self.vault_system.deposit_profit(pnl_sol)
                
            else:
                self.logger.error(f"❌ Failed to close position for {token_symbol}")
                
        except Exception as e:
            self.logger.error(f"❌ Error closing position: {e}")
    
    async def _compounding_loop(self):
        """Simple compounding loop - 80% reinvestment rule"""
        while self.running:
            try:
                vault_balance = await self.vault_system.get_balance()
                
                if vault_balance >= self.config.compound_threshold_sol:
                    # Simple rule: Reinvest 80% of vault balance
                    compound_amount = vault_balance * self.config.compound_rate
                    
                    # Transfer from vault to active capital
                    await self.vault_system.withdraw(compound_amount)
                    self.metrics.current_capital_sol += compound_amount
                    self.metrics.total_compounds += 1
                    
                    self.logger.info(f"💰 COMPOUND: {compound_amount:.4f} SOL reinvested | "
                                   f"Active capital: {self.metrics.current_capital_sol:.4f} SOL")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Error in compounding loop: {e}")
                await asyncio.sleep(600)
    
    async def _metrics_update_loop(self):
        """Periodically log performance metrics"""
        while self.running:
            try:
                self.logger.info(f"📊 METRICS | Trades: {self.metrics.trades_executed} | "
                               f"Win Rate: {self.metrics.win_rate:.1%} | "
                               f"Total P&L: {self.metrics.total_profit_sol:.4f} SOL | "
                               f"Active Capital: {self.metrics.current_capital_sol:.4f} SOL | "
                               f"Active Positions: {len(self.active_positions)} | "
                               f"Compounds: {self.metrics.total_compounds}")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Error in metrics update: {e}")
                await asyncio.sleep(600)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"🛑 Received signal {signum}, shutting down...")
        asyncio.create_task(self.shutdown())
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            self.logger.info("🛑 Shutting down simplified trading bot...")
            self.running = False
            
            # Close all active positions
            for token_address, position in list(self.active_positions.items()):
                await self._close_position(token_address, position, "shutdown")
            
            # Shutdown systems
            if self.kill_switch:
                await self.kill_switch.shutdown()
            
            self.logger.info("✅ Simplified trading bot shutdown complete")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
    
    async def emergency_shutdown(self):
        """Emergency shutdown"""
        self.logger.error("🚨 EMERGENCY SHUTDOWN TRIGGERED")
        await self.shutdown()
    
    def get_status(self) -> Dict[str, Any]:
        """Get bot status"""
        return {
            'running': self.running,
            'initialized': self.initialized,
            'active_positions': len(self.active_positions),
            'metrics': {
                'trades_executed': self.metrics.trades_executed,
                'win_rate': self.metrics.win_rate,
                'total_profit_sol': self.metrics.total_profit_sol,
                'current_capital_sol': self.metrics.current_capital_sol,
                'total_compounds': self.metrics.total_compounds
            }
        }


# Factory function for easy initialization
async def create_simplified_bot(initial_capital_sol: float = 1.5) -> SimplifiedTradingBot:
    """Create and initialize simplified trading bot"""
    config = SimplifiedConfig(initial_capital_sol=initial_capital_sol)
    bot = SimplifiedTradingBot(config)
    
    if await bot.initialize():
        return bot
    else:
        raise RuntimeError("Failed to initialize simplified trading bot") 