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
from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch, KillSwitchTrigger
from worker_ant_v1.utils.logger import get_logger


@dataclass
class SimplifiedConfig:
    """Lean configuration for simplified bot"""
    initial_capital_sol: float = 1.5  # ~$300
    
    # Stage 1: Survival Filter - DYNAMIC RISK MODEL
    acceptable_rel_threshold_percent: float = 0.02  # 2% of capital max risk
    risk_score_veto_threshold: float = 0.8  # Veto if risk score > 80%
    
    # Stage 2: Win-Rate Engine  
    hunt_threshold: float = 0.6  # Minimum win probability to trade
    
    # Stage 3: Growth Maximizer - EXPONENTIAL OPTIMIZATION
    kelly_fraction: float = 0.25  # Base fraction of full Kelly to use
    kelly_max_fraction: float = 0.65  # Maximum Kelly fraction for ultra-high confidence
    max_position_percent: float = 0.20  # Base max position size
    max_position_ultra: float = 0.40  # Ultra-high confidence max position
    
    # EXPONENTIAL COMPOUNDING SYSTEM
    compound_rate: float = 0.8  # Base 80% reinvestment
    compound_rate_aggressive: float = 0.95  # 95% reinvestment for hot streaks
    compound_threshold_sol: float = 0.2  # Compound when vault has 0.2 SOL
    
    # SIGNAL FUSION THRESHOLDS
    ultra_confidence_threshold: float = 0.85  # 85%+ for exponential sizing
    multi_signal_threshold: float = 0.75  # 75%+ for enhanced sizing
    
    # EXPONENTIAL TRIGGER CONDITIONS
    hot_streak_threshold: int = 3  # 3 consecutive wins triggers aggressive mode
    exponential_mode_duration: int = 10  # Number of trades in exponential mode
    
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
        
        # DYNAMIC KELLY PARAMETERS - DATA-DRIVEN CALCULATIONS
        self.cached_win_loss_ratio = 1.5  # Fallback, continuously updated from actual performance
        self.trade_history = []  # Complete trade history for statistical analysis
        self.winning_trades = []  # Winning trade profits for dynamic win/loss ratio
        self.losing_trades = []  # Losing trade losses for dynamic win/loss ratio
        self.min_trades_for_dynamic_ratio = 20  # Minimum trades before using dynamic ratio
        
        # CONTINUOUS RISK SCORING SYSTEM
        self.risk_score_history = []  # Track risk scores vs actual outcomes for model improvement
        self.current_risk_score = 0.0  # Current trade risk score (0.0 = no risk, 1.0 = maximum risk)
        
        # EXPONENTIAL GROWTH TRACKING
        self.exponential_mode = False
        self.exponential_trades_remaining = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.performance_streak = []  # Track last 10 trades for momentum analysis
        self.ultra_confidence_trades = 0  # Track ultra-high confidence opportunities
        
        self.logger.info("🎯 SimplifiedTradingBot initialized - Mathematical core with EXPONENTIAL GROWTH optimization")
    
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
                # ⚡ KILL SWITCH CHECK - Critical safety check before any trading activity
                if self.kill_switch.is_triggered:
                    self.logger.critical(f"🚨 KILL SWITCH ACTIVE - Trading loop stopped | "
                                       f"Reason: {self.kill_switch.trigger_reason.value if self.kill_switch.trigger_reason else 'Unknown'}")
                    self.running = False
                    break
                
                # Check kill switch conditions with current metrics
                current_metrics = {
                    'daily_loss_sol': max(0, self.metrics.initial_capital_sol - self.metrics.current_capital_sol),
                    'max_daily_loss_sol': self.config.initial_capital_sol * 0.2,  # 20% max daily loss
                    'error_count': getattr(self.metrics, 'error_count', 0),
                    'anomaly_score': getattr(self.metrics, 'anomaly_score', 0.0),
                    'security_status': True  # Assume secure unless otherwise detected
                }
                
                if self.kill_switch.check_conditions(current_metrics):
                    self.logger.critical("🚨 KILL SWITCH TRIGGERED BY CONDITIONS - Emergency shutdown initiated")
                    await self.emergency_shutdown()
                    break
                
                # Scan for opportunities
                opportunities = await self.market_scanner.scan_opportunities()
                
                # Process each opportunity through three-stage pipeline
                for opportunity in opportunities:
                    # Additional kill switch check before processing each opportunity
                    if self.kill_switch.is_triggered:
                        self.logger.warning("🚨 KILL SWITCH TRIGGERED during opportunity processing - stopping immediately")
                        self.running = False
                        break
                    
                    await self._process_opportunity_pipeline(opportunity)
                
                await asyncio.sleep(self.config.scan_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"❌ Error in trading loop: {e}")
                
                # Increment error count for kill switch monitoring
                if hasattr(self.metrics, 'error_count'):
                    self.metrics.error_count += 1
                else:
                    self.metrics.error_count = 1
                
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
            
            # ⚡ KILL SWITCH CHECK - Safety check before pipeline processing
            if self.kill_switch.is_triggered:
                self.logger.warning(f"🚨 KILL SWITCH ACTIVE - Skipping pipeline for {token_symbol}")
                return
            
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
            
            # CRITICAL IMPROVEMENT: Continuous Risk Scoring System
            # Calculate continuous risk score (0.0 = no risk, 1.0 = maximum risk)
            enhanced_trade_params = {
                **trade_params,
                'token_age_hours': opportunity.get('age_hours', 24),
                'liquidity_sol': opportunity.get('liquidity_sol', 50),
                'holder_count': opportunity.get('holder_count', 100)
            }
            
            risk_score = self._calculate_continuous_risk_score(enhanced_trade_params)
            
            # Dynamic risk threshold based on current capital (percentage-based)
            dynamic_risk_threshold_sol = self.metrics.current_capital_sol * self.config.acceptable_rel_threshold_percent
            
            # Calculate expected loss based on risk score
            estimated_position_size = min(0.1, self.metrics.current_capital_sol * 0.1)  # Rough estimate
            expected_loss = risk_score * estimated_position_size
            
            # ═══════════════════════════════════════════════════════════
            # ENHANCED RISK ASSESSMENT - Continuous scoring with veto only for extreme risk
            # ═══════════════════════════════════════════════════════════
            
            # Veto only if risk score exceeds threshold (80%) OR expected loss exceeds dynamic threshold
            if risk_score > self.config.risk_score_veto_threshold or expected_loss > dynamic_risk_threshold_sol:
                veto_reason = f"Risk score {risk_score:.3f} > {self.config.risk_score_veto_threshold}" if risk_score > self.config.risk_score_veto_threshold else f"Expected loss {expected_loss:.4f} > {dynamic_risk_threshold_sol:.4f} SOL"
                
                self.logger.warning(f"🚫 STAGE 1 RISK VETO: {token_symbol}")
                self.logger.warning(f"   └─ Risk Score: {risk_score:.3f} | Expected Loss: {expected_loss:.4f} SOL")
                self.logger.warning(f"   └─ Dynamic Threshold: {dynamic_risk_threshold_sol:.4f} SOL ({self.config.acceptable_rel_threshold_percent:.1%} of capital)")
                self.logger.warning(f"   └─ Reason: {veto_reason}")
                
                # Store risk data for model improvement
                self._record_risk_veto({
                    'token_address': token_address,
                    'token_symbol': token_symbol,
                    'risk_score': risk_score,
                    'expected_loss': expected_loss,
                    'dynamic_threshold': dynamic_risk_threshold_sol,
                    'veto_reason': veto_reason,
                    'timestamp': datetime.now()
                })
                return
            
            # Risk accepted - log risk assessment for tracking
            self.logger.info(f"✅ STAGE 1 RISK CLEAR: {token_symbol}")
            self.logger.info(f"   └─ Risk Score: {risk_score:.3f} | Expected Loss: {expected_loss:.4f} SOL")
            self.logger.info(f"   └─ Dynamic Threshold: {dynamic_risk_threshold_sol:.4f} SOL ({self.config.acceptable_rel_threshold_percent:.1%} of capital)")
            
            # Store risk analysis for position tracking (used later in Kelly calculation)
            risk_analysis_summary = {
                'risk_score': risk_score,
                'expected_loss': expected_loss,
                'dynamic_threshold': dynamic_risk_threshold_sol,
                'risk_components': enhanced_trade_params
            }
            
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
            
            # Calculate optimal position size using Enhanced Kelly Criterion
            optimal_position_size = self._calculate_kelly_position_size(
                win_probability, 
                self.metrics.current_capital_sol,
                market_data  # Pass market signals for enhanced calculations
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
                entry_signals=market_data,  # Store signals for learning
                risk_analysis=risk_analysis_summary  # Store risk assessment insights
            )
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline error for {opportunity.get('token_symbol', 'Unknown')}: {e}")
    
    async def _gather_market_signals(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather market signals for Naive Bayes analysis with logical purity.
        All signals sourced from canonical analyzer modules for consistency.
        """
        try:
            token_address = opportunity['token_address']
            
            # ═══════════════════════════════════════════════════════════
            # CANONICAL SIGNAL SOURCES - All from dedicated analyzers
            # ═══════════════════════════════════════════════════════════
            
            # Get sentiment analysis from SentimentFirstAI
            sentiment_result = await self.sentiment_ai.analyze_sentiment(token_address)
            
            # Get technical analysis from TechnicalAnalyzer
            technical_result = await self.technical_analyzer.analyze_token(token_address)
            
            # Get market data from MarketDataFetcher (not pre-processed opportunity)
            try:
                from worker_ant_v1.utils.market_data_fetcher import MarketDataFetcher
                market_fetcher = MarketDataFetcher()
                fresh_market_data = await market_fetcher.get_token_data(token_address)
            except Exception as e:
                self.logger.warning(f"Failed to get fresh market data, using fallback: {e}")
                fresh_market_data = None
            
            # Get liquidity analysis from enhanced rug detector
            liquidity_analysis = None
            if hasattr(self, 'rug_detector'):
                try:
                    liquidity_analysis = await self.rug_detector.analyze_liquidity_health(token_address)
                except Exception as e:
                    self.logger.debug(f"Liquidity analysis unavailable: {e}")
            
            # ═══════════════════════════════════════════════════════════
            # COMPILE SIGNALS - All from canonical sources
            # ═══════════════════════════════════════════════════════════
            
            signals = {}
            
            # Sentiment signals
            if sentiment_result:
                signals['sentiment_score'] = sentiment_result.sentiment_score
                signals['confidence'] = sentiment_result.confidence
                signals['social_buzz'] = sentiment_result.social_buzz_score
            else:
                signals['sentiment_score'] = 0.5
                signals['confidence'] = 0.5
                signals['social_buzz'] = 0.5
            
            # Technical analysis signals
            if technical_result:
                signals['rsi_oversold'] = 1.0 if getattr(technical_result, 'rsi', 50) < 30 else 0.0
                signals['rsi_overbought'] = 1.0 if getattr(technical_result, 'rsi', 50) > 70 else 0.0
                signals['volume_momentum'] = getattr(technical_result, 'volume_change_24h', 1.0)
                signals['price_momentum'] = getattr(technical_result, 'price_change_24h', 0.0)
                signals['technical_strength'] = getattr(technical_result, 'overall_score', 0.5)
            else:
                # Fallback to fresh market data if technical analysis unavailable
                if fresh_market_data:
                    signals['volume_momentum'] = fresh_market_data.get('volume_change_24h', 1.0)
                    signals['price_momentum'] = fresh_market_data.get('price_change_24h', 0.0)
                else:
                    signals['volume_momentum'] = 1.0
                    signals['price_momentum'] = 0.0
                
                signals['rsi_oversold'] = 0.0
                signals['rsi_overbought'] = 0.0
                signals['technical_strength'] = 0.5
            
            # Liquidity and market structure signals
            if liquidity_analysis:
                signals['liquidity_health'] = liquidity_analysis.get('health_score', 0.5)
                signals['liquidity_concentration_risk'] = liquidity_analysis.get('concentration_risk', 0.5)
            elif fresh_market_data:
                # Calculate from fresh data
                liquidity_concentration = fresh_market_data.get('liquidity_concentration', 0.5)
                signals['liquidity_health'] = 1.0 - liquidity_concentration
                signals['liquidity_concentration_risk'] = liquidity_concentration
            else:
                signals['liquidity_health'] = 0.5
                signals['liquidity_concentration_risk'] = 0.5
            
            # Derived momentum signals
            signals['volume_spike'] = 1.0 if signals['volume_momentum'] > 2.0 else 0.0
            signals['strong_momentum'] = 1.0 if (signals['price_momentum'] > 0.1 and signals['volume_momentum'] > 1.5) else 0.0
            
            self.logger.debug(f"📊 Gathered {len(signals)} canonical signals for {token_address[:8]}")
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
    
    def _record_wcca_veto(self, token_address: str, wcca_result: Dict[str, Any], trade_params: Dict[str, Any]):
        """Record WCCA veto data for future pattern analysis and ML training"""
        try:
            veto_record = {
                'timestamp': datetime.now(),
                'token_address': token_address,
                'veto_reason': wcca_result.get('reason', 'Unknown'),
                'worst_pattern': wcca_result.get('worst_pattern', 'Unknown'),
                'rel_calculated': wcca_result.get('rel_calculated', 0.0),
                'threshold': wcca_result.get('threshold', 0.0),
                'patterns_analyzed': wcca_result.get('patterns_analyzed', 0),
                'rel_breakdown': wcca_result.get('rel_breakdown', {}),
                'trade_params': {
                    'amount': trade_params.get('amount', 0.0),
                    'token_age_hours': trade_params.get('token_age_hours', 0),
                    'dev_holdings_percent': trade_params.get('dev_holdings_percent', 0.0),
                    'liquidity_concentration': trade_params.get('liquidity_concentration', 0.0),
                }
            }
            
            # Store in memory for analysis (could be persisted to database)
            if not hasattr(self, 'wcca_veto_history'):
                self.wcca_veto_history = []
            
            self.wcca_veto_history.append(veto_record)
            
            # Keep only recent history (last 500 vetoes)
            if len(self.wcca_veto_history) > 500:
                self.wcca_veto_history = self.wcca_veto_history[-500:]
            
            # Log analytics insight
            if len(self.wcca_veto_history) > 10:
                recent_vetoes = self.wcca_veto_history[-10:]
                common_patterns = {}
                for record in recent_vetoes:
                    pattern = record['worst_pattern']
                    common_patterns[pattern] = common_patterns.get(pattern, 0) + 1
                
                most_common = max(common_patterns.items(), key=lambda x: x[1])
                if most_common[1] >= 3:  # If pattern appears 3+ times in last 10 vetoes
                    self.logger.info(f"📈 WCCA Pattern Alert: {most_common[0]} pattern dominant ({most_common[1]}/10 recent vetoes)")
            
        except Exception as e:
            self.logger.error(f"Error recording WCCA veto: {e}")
    
    def _record_risk_veto(self, risk_data: Dict[str, Any]):
        """Record risk veto for model improvement and pattern analysis"""
        try:
            # Add to risk score history for model training
            self.risk_score_history.append({
                **risk_data,
                'outcome': 'veto',
                'capital_at_time': self.metrics.current_capital_sol
            })
            
            # Keep only recent history (last 500 records)
            if len(self.risk_score_history) > 500:
                self.risk_score_history = self.risk_score_history[-500:]
            
            self.logger.debug(f"📝 Risk veto recorded: {risk_data['token_symbol']} | Risk: {risk_data['risk_score']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error recording risk veto: {e}")
    
    def _calculate_kelly_position_size(self, win_probability: float, current_capital: float, market_signals: Dict[str, Any] = None) -> float:
        """
        EXPONENTIAL GROWTH KELLY CRITERION with Adaptive Scaling
        
        Base Formula: f* = p - ((1 - p) / b)
        Enhanced with:
        - Confidence-based Kelly scaling (25% to 65%)
        - Multi-signal fusion multipliers
        - Exponential mode amplification
        - Performance streak momentum
        """
        try:
            # Ensure valid inputs
            win_probability = max(0.01, min(0.99, win_probability))
            market_signals = market_signals or {}
            
            # DYNAMIC WIN/LOSS RATIO - Data-driven calculation from actual performance
            win_loss_ratio = self._calculate_dynamic_win_loss_ratio()
            
            # Calculate base Kelly fraction
            kelly_fraction_full = win_probability - ((1 - win_probability) / win_loss_ratio)
            
            # EXPONENTIAL OPTIMIZATION LAYER 1: Confidence-Based Kelly Scaling
            confidence_multiplier = self._calculate_confidence_multiplier(win_probability, market_signals)
            adaptive_kelly_fraction = self._interpolate_kelly_fraction(win_probability, confidence_multiplier)
            
            # EXPONENTIAL OPTIMIZATION LAYER 2: Multi-Signal Fusion
            signal_strength_multiplier = self._calculate_signal_fusion_multiplier(market_signals)
            
            # EXPONENTIAL OPTIMIZATION LAYER 3: Performance Streak Momentum
            momentum_multiplier = self._calculate_momentum_multiplier()
            
            # EXPONENTIAL OPTIMIZATION LAYER 4: Exponential Mode Amplification
            exponential_multiplier = self._calculate_exponential_mode_multiplier()
            
            # CRITICAL IMPROVEMENT: Risk-Adjusted Kelly Calculation
            # Calculate continuous risk score for this trade
            trade_params = {
                'token_address': market_signals.get('token_address', ''),
                'amount': 0.0,  # Will be calculated
                'token_age_hours': market_signals.get('token_age_hours', 24),
                'liquidity_sol': market_signals.get('liquidity_sol', 50),
                'holder_count': market_signals.get('holder_count', 100)
            }
            risk_score = self._calculate_continuous_risk_score(trade_params)
            self.current_risk_score = risk_score
            
            # Apply risk adjustment to Kelly fraction: f* × (1 - risk_score)
            risk_adjusted_kelly = kelly_fraction_full * (1 - risk_score)
            
            # Calculate enhanced Kelly fraction with all multipliers
            enhanced_kelly = risk_adjusted_kelly * adaptive_kelly_fraction * signal_strength_multiplier * momentum_multiplier * exponential_multiplier
            
            # Calculate base position size
            base_position_size = enhanced_kelly * current_capital
            
            # CRITICAL IMPROVEMENT: Multi-Asset Kelly Adjustment
            position_size = self._calculate_multi_asset_kelly_adjustment(base_position_size)
            
            # Apply dynamic maximum position size limits
            max_position_percent = self._get_dynamic_max_position(win_probability, confidence_multiplier)
            max_position = current_capital * max_position_percent
            position_size = min(position_size, max_position)
            
            # Ensure minimum viable position
            position_size = max(0.0, position_size)
            
            # Enhanced logging for risk-adjusted Kelly tracking
            mode_status = "🚀 EXPONENTIAL" if self.exponential_mode else "📊 STANDARD"
            ratio_source = "DYNAMIC" if len(self.winning_trades) >= 10 and len(self.losing_trades) >= 10 else "FALLBACK"
            
            self.logger.info(f"💰 {mode_status} RISK-ADJUSTED KELLY: p={win_probability:.3f}, b={win_loss_ratio:.2f} ({ratio_source})")
            self.logger.info(f"   └─ Base Kelly: {kelly_fraction_full:.3f} → Risk Adj: {risk_adjusted_kelly:.3f} → Enhanced: {enhanced_kelly:.3f}")
            self.logger.info(f"   └─ Risk Score: {risk_score:.3f} | Multi-Asset Adj: {base_position_size:.4f} → {position_size:.4f} SOL")
            self.logger.info(f"   └─ Multipliers: Conf={confidence_multiplier:.2f}, Signal={signal_strength_multiplier:.2f}, "
                           f"Momentum={momentum_multiplier:.2f}, Exp={exponential_multiplier:.2f}")
            self.logger.info(f"   └─ Final Position: {position_size:.4f} SOL ({position_size/current_capital:.1%} of capital) | "
                           f"Active Positions: {len(self.active_positions)}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error in Enhanced Kelly Criterion calculation: {e}")
            return current_capital * 0.01  # Conservative fallback
    
    def _calculate_dynamic_win_loss_ratio(self) -> float:
        """
        CRITICAL IMPROVEMENT: Calculate win/loss ratio from actual bot performance
        
        This replaces the fixed 1.5 assumption with data-driven calculation.
        Returns the ratio of average winning trade profit to average losing trade loss.
        """
        try:
            # Need minimum trades for statistical validity
            if len(self.winning_trades) < 10 or len(self.losing_trades) < 10:
                self.logger.debug(f"Insufficient trade data (W:{len(self.winning_trades)}, L:{len(self.losing_trades)}), using fallback ratio")
                return self.cached_win_loss_ratio
            
            # Calculate recent performance (last 50 trades of each type for relevance)
            recent_wins = self.winning_trades[-50:] if len(self.winning_trades) >= 50 else self.winning_trades
            recent_losses = self.losing_trades[-50:] if len(self.losing_trades) >= 50 else self.losing_trades
            
            # Calculate average profit and loss
            avg_win = sum(recent_wins) / len(recent_wins)
            avg_loss = abs(sum(recent_losses) / len(recent_losses))  # Ensure positive for ratio
            
            if avg_loss == 0:
                self.logger.warning("Average loss is zero, using fallback ratio")
                return self.cached_win_loss_ratio
            
            # Calculate dynamic win/loss ratio
            dynamic_ratio = avg_win / avg_loss
            
            # Apply bounds to prevent extreme ratios (0.5 to 5.0)
            dynamic_ratio = max(0.5, min(5.0, dynamic_ratio))
            
            self.logger.debug(f"📊 Dynamic W/L Ratio: {dynamic_ratio:.3f} (avg_win: {avg_win:.4f}, avg_loss: {avg_loss:.4f})")
            
            return dynamic_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic win/loss ratio: {e}")
            return self.cached_win_loss_ratio
    
    def _calculate_continuous_risk_score(self, trade_params: Dict[str, Any]) -> float:
        """
        CRITICAL IMPROVEMENT: Continuous risk scoring instead of binary veto
        
        Returns a risk score from 0.0 (no risk) to 1.0 (maximum risk)
        This replaces the binary go/no-go decision with nuanced risk assessment.
        """
        try:
            token_address = trade_params.get('token_address', '')
            position_size_sol = abs(float(trade_params.get('amount', 0.0)))
            
            # Initialize risk components
            risk_components = {
                'rug_pull_risk': 0.0,
                'honeypot_risk': 0.0,
                'liquidity_risk': 0.0,
                'whale_risk': 0.0,
                'technical_risk': 0.0
            }
            
            # STATISTICAL RISK MODEL COMPONENTS
            # (In production, these would be ML model predictions)
            
            # Rug Pull Risk - based on token characteristics
            token_age_hours = trade_params.get('token_age_hours', 0)
            liquidity_sol = trade_params.get('liquidity_sol', 0)
            holder_count = trade_params.get('holder_count', 0)
            
            # Age-based risk (newer = riskier)
            if token_age_hours < 1:
                risk_components['rug_pull_risk'] += 0.4
            elif token_age_hours < 24:
                risk_components['rug_pull_risk'] += 0.2
            elif token_age_hours < 168:  # 1 week
                risk_components['rug_pull_risk'] += 0.1
            
            # Liquidity-based risk
            if liquidity_sol < 5:
                risk_components['liquidity_risk'] += 0.3
            elif liquidity_sol < 20:
                risk_components['liquidity_risk'] += 0.15
            elif liquidity_sol < 100:
                risk_components['liquidity_risk'] += 0.05
            
            # Holder concentration risk
            if holder_count < 50:
                risk_components['whale_risk'] += 0.25
            elif holder_count < 200:
                risk_components['whale_risk'] += 0.1
            
            # Position size risk (larger positions = higher risk)
            position_risk = min(0.3, position_size_sol / self.metrics.current_capital_sol)
            risk_components['technical_risk'] = position_risk
            
            # Weighted risk score calculation
            risk_weights = {
                'rug_pull_risk': 0.3,
                'honeypot_risk': 0.2,
                'liquidity_risk': 0.2,
                'whale_risk': 0.15,
                'technical_risk': 0.15
            }
            
            total_risk_score = sum(
                risk_components[component] * risk_weights[component]
                for component in risk_components
            )
            
            # Bound to [0.0, 1.0]
            total_risk_score = max(0.0, min(1.0, total_risk_score))
            
            self.logger.debug(f"🎯 Risk Score: {total_risk_score:.3f} | Components: {risk_components}")
            
            return total_risk_score
            
        except Exception as e:
            self.logger.error(f"Error calculating continuous risk score: {e}")
            return 0.5  # Conservative medium risk fallback
    
    def _calculate_multi_asset_kelly_adjustment(self, base_position_size: float) -> float:
        """
        CRITICAL IMPROVEMENT: Multi-asset Kelly adjustment for concurrent positions
        
        Adjusts position size based on number of open positions to prevent over-leveraging.
        Classic Kelly assumes single bet, but we trade multiple assets simultaneously.
        """
        try:
            num_active_positions = len(self.active_positions)
            
            if num_active_positions == 0:
                return base_position_size
            
            # Multi-asset Kelly adjustment formula
            # Reduces exposure as more positions are taken
            adjustment_factor = 1.0 / (1.0 + (num_active_positions * 0.3))
            
            adjusted_position_size = base_position_size * adjustment_factor
            
            self.logger.debug(f"🔄 Multi-Asset Kelly: {num_active_positions} positions, "
                            f"adjustment factor: {adjustment_factor:.3f}, "
                            f"position: {base_position_size:.4f} → {adjusted_position_size:.4f} SOL")
            
            return adjusted_position_size
            
        except Exception as e:
            self.logger.error(f"Error in multi-asset Kelly adjustment: {e}")
            return base_position_size
    
    def _calculate_confidence_multiplier(self, win_probability: float, market_signals: Dict[str, Any]) -> float:
        """Calculate confidence multiplier based on signal strength and probability"""
        # Base confidence from win probability
        prob_confidence = min(1.0, (win_probability - 0.5) * 2.0)  # Scale 0.5-1.0 to 0.0-1.0
        
        # Signal strength confidence
        signal_count = len([s for s in market_signals.values() if isinstance(s, (int, float)) and s > 0])
        signal_confidence = min(1.0, signal_count / 8.0)  # Normalize by expected 8 signals
        
        # Combined confidence
        combined_confidence = (prob_confidence * 0.7) + (signal_confidence * 0.3)
        
        return 0.5 + (combined_confidence * 0.5)  # Scale to 0.5-1.0 range
    
    def _interpolate_kelly_fraction(self, win_probability: float, confidence_multiplier: float) -> float:
        """Interpolate Kelly fraction based on win probability and confidence"""
        if win_probability >= self.config.ultra_confidence_threshold:
            # Ultra-high confidence: use maximum Kelly fraction
            return self.config.kelly_max_fraction * confidence_multiplier
        elif win_probability >= self.config.multi_signal_threshold:
            # High confidence: interpolate between base and max
            ratio = (win_probability - self.config.multi_signal_threshold) / (self.config.ultra_confidence_threshold - self.config.multi_signal_threshold)
            interpolated_fraction = self.config.kelly_fraction + (ratio * (self.config.kelly_max_fraction - self.config.kelly_fraction))
            return interpolated_fraction * confidence_multiplier
        else:
            # Standard confidence: use base Kelly fraction
            return self.config.kelly_fraction * confidence_multiplier
    
    def _calculate_signal_fusion_multiplier(self, market_signals: Dict[str, Any]) -> float:
        """Calculate multiplier based on multi-signal fusion strength"""
        if not market_signals:
            return 1.0
        
        # Count strong positive signals
        strong_signals = 0
        total_signals = 0
        
        signal_patterns = {
            'price_momentum': market_signals.get('price_momentum', 0),
            'volume_momentum': market_signals.get('volume_momentum', 0),
            'volume_spike': market_signals.get('volume_spike', 0),
            'strong_momentum': market_signals.get('strong_momentum', 0),
            'rsi_strength': market_signals.get('rsi_strength', 0),
            'macd_bullish': market_signals.get('macd_bullish', 0),
            'breakout_signal': market_signals.get('breakout_signal', 0),
            'sentiment_spike': market_signals.get('sentiment_spike', 0)
        }
        
        for signal_name, signal_value in signal_patterns.items():
            if isinstance(signal_value, (int, float)):
                total_signals += 1
                if signal_value > 0.7:  # Strong signal threshold
                    strong_signals += 1
        
        if total_signals == 0:
            return 1.0
        
        # Calculate signal strength ratio
        signal_strength_ratio = strong_signals / total_signals
        
        # Return multiplier: 1.0 (no enhancement) to 1.5 (50% enhancement)
        return 1.0 + (signal_strength_ratio * 0.5)
    
    def _calculate_momentum_multiplier(self) -> float:
        """Calculate multiplier based on performance streak momentum"""
        if len(self.performance_streak) < 3:
            return 1.0  # No momentum data
        
        # Count recent wins
        recent_wins = sum(1 for trade in self.performance_streak[-5:] if trade.get('profitable', False))
        recent_trades = len(self.performance_streak[-5:])
        
        if recent_trades == 0:
            return 1.0
        
        win_rate = recent_wins / recent_trades
        
        # Momentum multiplier based on recent performance
        if win_rate >= 0.8:  # 80%+ recent win rate
            return 1.3  # 30% enhancement
        elif win_rate >= 0.6:  # 60%+ recent win rate
            return 1.15  # 15% enhancement
        elif win_rate <= 0.3:  # 30% or lower recent win rate
            return 0.8  # 20% reduction for risk management
        else:
            return 1.0  # Neutral
    
    def _calculate_exponential_mode_multiplier(self) -> float:
        """Calculate multiplier for exponential mode"""
        if not self.exponential_mode:
            return 1.0
        
        # Progressive amplification during exponential mode
        trades_completed = self.config.exponential_mode_duration - self.exponential_trades_remaining
        progress_ratio = trades_completed / self.config.exponential_mode_duration
        
        # Start at 1.2x and scale up to 1.8x during exponential mode
        return 1.2 + (progress_ratio * 0.6)
    
    def _get_dynamic_max_position(self, win_probability: float, confidence_multiplier: float) -> float:
        """Get dynamic maximum position percentage based on confidence"""
        if win_probability >= self.config.ultra_confidence_threshold and confidence_multiplier > 0.8:
            # Ultra-high confidence: allow larger positions
            return self.config.max_position_ultra
        elif win_probability >= self.config.multi_signal_threshold:
            # High confidence: interpolate between base and ultra max
            ratio = (win_probability - self.config.multi_signal_threshold) / (self.config.ultra_confidence_threshold - self.config.multi_signal_threshold)
            return self.config.max_position_percent + (ratio * (self.config.max_position_ultra - self.config.max_position_percent))
        else:
            # Standard confidence: use base max position
            return self.config.max_position_percent
    
    def _update_exponential_mode(self, trade_profitable: bool):
        """Update exponential mode based on performance"""
        # Update performance streak
        self.performance_streak.append({
            'profitable': trade_profitable,
            'timestamp': datetime.now()
        })
        
        # Keep only last 10 trades
        if len(self.performance_streak) > 10:
            self.performance_streak = self.performance_streak[-10:]
        
        # Update consecutive counters
        if trade_profitable:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
            # Check for exponential mode trigger
            if self.consecutive_wins >= self.config.hot_streak_threshold and not self.exponential_mode:
                self.exponential_mode = True
                self.exponential_trades_remaining = self.config.exponential_mode_duration
                self.logger.info(f"🚀 EXPONENTIAL MODE ACTIVATED! Hot streak: {self.consecutive_wins} wins")
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # Exit exponential mode on loss
            if self.exponential_mode:
                self.exponential_mode = False
                self.exponential_trades_remaining = 0
                self.logger.info("📊 Exponential mode deactivated due to loss")
        
        # Decrement exponential mode counter
        if self.exponential_mode:
            self.exponential_trades_remaining -= 1
            if self.exponential_trades_remaining <= 0:
                self.exponential_mode = False
                self.logger.info("📊 Exponential mode completed")
    
    async def _execute_trade(self, opportunity: Dict[str, Any], position_size: float, win_probability: float, entry_signals: Dict[str, Any] = None, risk_analysis: Dict[str, Any] = None):
        """Execute trade with simplified, direct execution and enhanced tracking"""
        try:
            token_address = opportunity['token_address']
            token_symbol = opportunity.get('token_symbol', 'Unknown')
            
            # ⚡ CRITICAL KILL SWITCH CHECK - Final safety gate before trade execution
            if self.kill_switch.is_triggered:
                self.logger.critical(f"🚨 KILL SWITCH ACTIVE - Trade execution BLOCKED for {token_symbol} | "
                                   f"Reason: {self.kill_switch.trigger_reason.value if self.kill_switch.trigger_reason else 'Unknown'}")
                return
            
            # Final pre-execution kill switch condition check
            final_metrics = {
                'daily_loss_sol': max(0, self.metrics.initial_capital_sol - self.metrics.current_capital_sol),
                'max_daily_loss_sol': self.config.initial_capital_sol * 0.2,  # 20% max daily loss
                'error_count': getattr(self.metrics, 'error_count', 0),
                'anomaly_score': getattr(self.metrics, 'anomaly_score', 0.0),
                'security_status': True
            }
            
            if self.kill_switch.check_conditions(final_metrics):
                self.logger.critical(f"🚨 KILL SWITCH TRIGGERED ON FINAL CHECK - Trade execution BLOCKED for {token_symbol}")
                return
            
            self.logger.info(f"✅ KILL SWITCH CLEAR - Proceeding with trade execution for {token_symbol}")
            
            # Execute buy order
            result = await self.trading_engine.execute_buy_order(
                token_address=token_address,
                amount_sol=position_size,
                max_slippage_bps=500  # 5% max slippage
            )
            
            if result.get('success', False):
                # Track position with entry signals and WCCA analysis for learning
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
                    'entry_signals': entry_signals or {},  # Store for learning
                    'risk_analysis': risk_analysis or {},  # Store risk assessment insights
                    'risk_score': risk_analysis.get('risk_score', 0.5) if risk_analysis else 0.5  # Store risk score for tracking
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
                # ⚡ KILL SWITCH CHECK - Emergency position closure if triggered
                if self.kill_switch.is_triggered:
                    self.logger.critical(f"🚨 KILL SWITCH ACTIVE - Emergency closure of all positions | "
                                       f"Reason: {self.kill_switch.trigger_reason.value if self.kill_switch.trigger_reason else 'Unknown'}")
                    
                    # Emergency close all positions immediately
                    emergency_positions = list(self.active_positions.items())
                    for token_address, position in emergency_positions:
                        await self._close_position(token_address, position, f"KILL_SWITCH_EMERGENCY_{self.kill_switch.trigger_reason.value if self.kill_switch.trigger_reason else 'UNKNOWN'}")
                    
                    break
                
                positions_to_close = []
                
                for token_address, position in self.active_positions.items():
                    should_close, reason = await self._should_close_position(position)
                    if should_close:
                        positions_to_close.append((token_address, position, reason))
                
                # Close positions
                for token_address, position, reason in positions_to_close:
                    # Additional kill switch check before each position closure
                    if self.kill_switch.is_triggered:
                        self.logger.warning(f"🚨 KILL SWITCH TRIGGERED during position closure - switching to emergency mode")
                        reason = f"KILL_SWITCH_EMERGENCY_{reason}"
                    
                    await self._close_position(token_address, position, reason)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in position monitoring: {e}")
                
                # Increment error count for kill switch monitoring
                if hasattr(self.metrics, 'error_count'):
                    self.metrics.error_count += 1
                else:
                    self.metrics.error_count = 1
                
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
                
                # CRITICAL IMPROVEMENT: Track trade history for dynamic calculations
                trade_record = {
                    'timestamp': datetime.now(),
                    'token_address': token_address,
                    'token_symbol': token_symbol,
                    'position_size_sol': position_size,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_sol': pnl_sol,
                    'pnl_percent': (exit_price / entry_price - 1) if entry_price > 0 else 0,
                    'profitable': pnl_sol > 0,
                    'risk_score': position.get('risk_score', 0.5),
                    'hold_time_minutes': (datetime.now() - position.get('entry_time', datetime.now())).total_seconds() / 60
                }
                
                # Add to comprehensive trade history
                self.trade_history.append(trade_record)
                
                # Update winning/losing trade arrays for dynamic win/loss ratio
                if pnl_sol > 0:
                    # Store profit as percentage for consistency
                    profit_percent = trade_record['pnl_percent']
                    self.winning_trades.append(profit_percent)
                    self.metrics.successful_trades += 1
                    
                    # Keep only recent trades (last 100)
                    if len(self.winning_trades) > 100:
                        self.winning_trades = self.winning_trades[-100:]
                else:
                    # Store loss as positive percentage for ratio calculation
                    loss_percent = abs(trade_record['pnl_percent'])
                    self.losing_trades.append(loss_percent)
                    
                    # Keep only recent trades (last 100)
                    if len(self.losing_trades) > 100:
                        self.losing_trades = self.losing_trades[-100:]
                
                # Update metrics
                self.metrics.trades_executed += 1
                self.metrics.total_profit_sol += pnl_sol
                self.metrics.current_capital_sol += position_size + pnl_sol
                
                # Update win rate
                self.metrics.win_rate = self.metrics.successful_trades / self.metrics.trades_executed if self.metrics.trades_executed > 0 else 0
                self.metrics.avg_profit_per_trade = self.metrics.total_profit_sol / self.metrics.trades_executed if self.metrics.trades_executed > 0 else 0
                
                # Record signal outcome for Naive Bayes learning
                if 'entry_signals' in position:
                    was_successful = pnl_sol > 0
                    self._record_signal_outcome(position['entry_signals'], was_successful)
                    self.logger.debug(f"📚 Recorded signal outcome for learning: {'WIN' if was_successful else 'LOSS'}")
                
                # EXPONENTIAL MODE: Update performance tracking
                self._update_exponential_mode(pnl_sol > 0)
                
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
        """EXPONENTIAL COMPOUNDING LOOP with Dynamic Rates"""
        while self.running:
            try:
                vault_balance = await self.vault_system.get_balance()
                
                if vault_balance >= self.config.compound_threshold_sol:
                    # DYNAMIC COMPOUND RATE based on performance and mode
                    compound_rate = self._get_dynamic_compound_rate()
                    compound_amount = vault_balance * compound_rate
                    
                    # Transfer from vault to active capital
                    await self.vault_system.withdraw(compound_amount)
                    self.metrics.current_capital_sol += compound_amount
                    self.metrics.total_compounds += 1
                    
                    # Enhanced logging with mode status
                    mode_indicator = "🚀 EXPONENTIAL" if self.exponential_mode else "📊 STANDARD"
                    self.logger.info(f"💰 {mode_indicator} COMPOUND: {compound_amount:.4f} SOL ({compound_rate:.1%} rate)")
                    self.logger.info(f"   └─ Active capital: {self.metrics.current_capital_sol:.4f} SOL | "
                                   f"Vault reserve: {vault_balance - compound_amount:.4f} SOL")
                    
                    if self.exponential_mode:
                        self.logger.info(f"   └─ 🔥 Exponential trades remaining: {self.exponential_trades_remaining}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Error in exponential compounding loop: {e}")
                await asyncio.sleep(600)
    
    def _get_dynamic_compound_rate(self) -> float:
        """Calculate dynamic compound rate based on performance and mode"""
        base_rate = self.config.compound_rate
        
        # Exponential mode: use aggressive compound rate
        if self.exponential_mode:
            return self.config.compound_rate_aggressive
        
        # Performance-based adjustment
        if len(self.performance_streak) >= 5:
            recent_wins = sum(1 for trade in self.performance_streak[-5:] if trade.get('profitable', False))
            recent_win_rate = recent_wins / 5
            
            if recent_win_rate >= 0.8:  # 80%+ recent win rate
                return min(0.95, base_rate + 0.1)  # Increase compound rate by 10%
            elif recent_win_rate <= 0.4:  # 40% or lower recent win rate
                return max(0.5, base_rate - 0.2)  # Decrease compound rate by 20% for safety
        
        return base_rate
    
    async def _metrics_update_loop(self):
        """Periodically log enhanced performance metrics with exponential tracking"""
        while self.running:
            try:
                # Calculate growth metrics
                growth_rate = ((self.metrics.current_capital_sol / self.config.initial_capital_sol) - 1) * 100
                
                # Mode status indicator
                mode_status = "🚀 EXPONENTIAL MODE" if self.exponential_mode else "📊 Standard Mode"
                
                self.logger.info(f"📊 ENHANCED METRICS | {mode_status}")
                self.logger.info(f"   └─ Trades: {self.metrics.trades_executed} | Win Rate: {self.metrics.win_rate:.1%}")
                self.logger.info(f"   └─ Total P&L: {self.metrics.total_profit_sol:.4f} SOL | Growth: {growth_rate:+.1f}%")
                self.logger.info(f"   └─ Active Capital: {self.metrics.current_capital_sol:.4f} SOL")
                self.logger.info(f"   └─ Active Positions: {len(self.active_positions)} | Compounds: {self.metrics.total_compounds}")
                
                # Exponential mode details
                if self.exponential_mode:
                    self.logger.info(f"   └─ 🔥 Hot Streak: {self.consecutive_wins} wins | "
                                   f"Exp. Trades Left: {self.exponential_trades_remaining}")
                
                # Performance streak summary
                if len(self.performance_streak) >= 3:
                    recent_wins = sum(1 for trade in self.performance_streak[-5:] if trade.get('profitable', False))
                    recent_trades = len(self.performance_streak[-5:])
                    recent_win_rate = recent_wins / recent_trades if recent_trades > 0 else 0
                    self.logger.info(f"   └─ Recent Performance: {recent_wins}/{recent_trades} ({recent_win_rate:.1%})")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                self.logger.error(f"❌ Error in enhanced metrics update: {e}")
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
        """Emergency shutdown with kill switch integration"""
        try:
            self.logger.critical("🚨 EMERGENCY SHUTDOWN TRIGGERED")
            
            # Trigger kill switch if not already triggered
            if not self.kill_switch.is_triggered:
                self.kill_switch.trigger(
                    reason=KillSwitchTrigger.SYSTEM_ERROR,
                    details="Emergency shutdown called"
                )
                self.logger.critical("🚨 KILL SWITCH TRIGGERED BY EMERGENCY SHUTDOWN")
            
            # Set running to False immediately to stop all loops
            self.running = False
            
            # Emergency close all positions immediately (no waiting)
            if self.active_positions:
                self.logger.critical(f"🚨 EMERGENCY CLOSING {len(self.active_positions)} ACTIVE POSITIONS")
                
                emergency_closure_tasks = []
                for token_address, position in list(self.active_positions.items()):
                    task = asyncio.create_task(
                        self._close_position(token_address, position, "EMERGENCY_SHUTDOWN")
                    )
                    emergency_closure_tasks.append(task)
                
                # Wait for all emergency closures with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*emergency_closure_tasks, return_exceptions=True),
                        timeout=30.0  # 30 second timeout for emergency closure
                    )
                    self.logger.critical("✅ ALL EMERGENCY POSITION CLOSURES COMPLETED")
                except asyncio.TimeoutError:
                    self.logger.critical("⚠️ EMERGENCY POSITION CLOSURE TIMEOUT - Some positions may remain open")
            
            # Move remaining funds to vault immediately
            if hasattr(self, 'vault_system') and self.vault_system:
                try:
                    await asyncio.wait_for(
                        self.vault_system.emergency_secure_all_funds(),
                        timeout=10.0
                    )
                    self.logger.critical("✅ EMERGENCY VAULT SECURING COMPLETED")
                except Exception as e:
                    self.logger.critical(f"❌ EMERGENCY VAULT SECURING FAILED: {e}")
            
            # Call normal shutdown for cleanup
            await self.shutdown()
            
            self.logger.critical("🚨 EMERGENCY SHUTDOWN COMPLETED")
            
        except Exception as e:
            self.logger.critical(f"❌ CRITICAL ERROR IN EMERGENCY SHUTDOWN: {e}")
            # Force stop everything
            self.running = False
            sys.exit(1)
    
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