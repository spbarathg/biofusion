"""
Unified Trading Engine
=====================

Advanced trading engine that combines pattern recognition, surgical execution,
and battlefield survival mechanisms into a single, powerful system.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import traceback

from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.core.wallet_manager import UnifiedWalletManager as WalletManager
from worker_ant_v1.intelligence.token_intelligence_system import TokenIntelligenceSystem
from worker_ant_v1.intelligence.battle_pattern_intelligence import BattlePatternIntelligence


@dataclass
class PatternPrediction:
    pattern_confidence: float
    time_sensitivity: float
    expected_move_size: float
    supporting_patterns: List[Any]

@dataclass 
class ChainPattern:
    pattern_type: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityAssessment:
    threat_level: Any
    risk_factors: List[str]
    warning_messages: List[str]


# Import trading components
try:
    from worker_ant_v1.trading.surgical_trade_executor import SurgicalTradeExecutor, ExecutionResult
except ImportError:
    SurgicalTradeExecutor = None
    ExecutionResult = dict

try:
    from worker_ant_v1.safety.enhanced_rug_detector import EnhancedRugDetector, ThreatLevel
except ImportError:
    EnhancedRugDetector = None
    ThreatLevel = None

try:
    from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch as KillSwitch
except ImportError:
    KillSwitch = None

try:
    from worker_ant_v1.monitoring.production_monitoring_system import EnhancedProductionMonitoringSystem as ProductionMonitoring
except ImportError:
    ProductionMonitoring = None

try:
    from worker_ant_v1.core.unified_config import UnifiedConfigManager
except ImportError:
    UnifiedConfigManager = None

try:
    from worker_ant_v1.intelligence.discord_intelligence_system import DiscordIntelligenceSystem, DiscordSignal, SignalType
except ImportError:
    DiscordIntelligenceSystem = None
    DiscordSignal = None
    SignalType = None

engine_logger = setup_logger(__name__)

class TradingState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    TRADING = "trading"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class TradingSignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class SignalConfidence(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class TradingSignal:
    """Trading signal with confidence and metadata"""
    signal_type: TradingSignalType
    token_address: str
    confidence: SignalConfidence
    timestamp: datetime
    price: float = 0.0
    volume: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActivePosition:
    """Active trading position"""
    token_address: str
    entry_price: float
    quantity: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BattlefieldState:
    """Real-time battlefield state"""
    active_trades: Dict[str, Dict]
    monitored_tokens: Dict[str, Dict]
    detected_patterns: Dict[str, List[ChainPattern]]
    active_threats: Dict[str, SecurityAssessment]
    performance_metrics: Dict[str, float]
    last_update: datetime

@dataclass
class ExecutionContext:
    """Context for trade execution"""
    token_address: str
    operation_id: str
    pattern_confidence: float
    security_assessment: SecurityAssessment
    position_size: float
    max_slippage: float
    timeout: float
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        self.created_at = datetime.now()
    
    def should_retry(self) -> bool:
        return (
            self.retry_count < self.max_retries and
            (datetime.now() - self.created_at).total_seconds() < self.timeout
        )

class UnifiedTradingEngine:
    """Advanced unified trading engine with battlefield intelligence"""
    
    def __init__(self):
        self.logger = setup_logger("UnifiedTradingEngine")
        
        # Core components
        self.wallet_manager = None
        self.pattern_intelligence = None
        self.trade_executor = None
        self.rug_detector = None
        self.kill_switch = None
        self.monitoring = None
        
        # Discord intelligence
        self.discord_intelligence = None
        
        # System state
        self.state = TradingState.INITIALIZING
        self.battlefield_state = BattlefieldState(
            active_trades={},
            monitored_tokens={},
            detected_patterns={},
            active_threats={},
            performance_metrics={
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'survival_rate': 0.0
            },
            last_update=datetime.now()
        )
        
        # Position limits
        self.position_limits = {
            'max_position_size': 0.1,  # 10% of capital
            'max_open_positions': 5,
            'max_exposure_per_token': 0.05  # 5% of capital
        }
        
        # Trading thresholds
        self.min_pattern_confidence = 0.7
        self.max_threat_level = ThreatLevel.SUSPICIOUS
        self.min_timing_accuracy = 0.8
        
        # Execution management
        self.active_operations: Dict[str, ExecutionContext] = {}
        self.operation_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.execution_lock = asyncio.Lock()
        
        # Performance tracking
        self.execution_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'average_execution_time': 0.0,
            'retry_count': 0
        }

    async def initialize(self):
        """Initialize the trading engine"""
        
        self.logger.info("🚀 Initializing Unified Trading Engine...")
        
        try:
            # Initialize core components
            if UnifiedConfigManager:
                self.config = UnifiedConfigManager()
                await self.config.initialize()
            
            if WalletManager:
                self.wallet_manager = WalletManager()
                await self.wallet_manager.initialize()
            
            if BattlePatternIntelligence:
                self.pattern_intelligence = BattlePatternIntelligence()
                await self.pattern_intelligence.initialize()
            
            if TokenIntelligenceSystem:
                self.token_intelligence = TokenIntelligenceSystem()
                await self.token_intelligence.initialize()
            
            if SurgicalTradeExecutor:
                self.trade_executor = SurgicalTradeExecutor()
                await self.trade_executor.initialize()
            
            if EnhancedRugDetector:
                self.rug_detector = EnhancedRugDetector()
                await self.rug_detector.initialize()
            
            if KillSwitch:
                self.kill_switch = KillSwitch()
                await self.kill_switch.initialize()
            
            if ProductionMonitoring:
                self.monitoring = ProductionMonitoring()
                await self.monitoring.initialize()
            
            # Initialize Discord intelligence if available
            if DiscordIntelligenceSystem:
                try:
                    self.discord_intelligence = DiscordIntelligenceSystem()
                    # Load Discord config if available
                    import json
                    import os
                    discord_config_path = "config/discord_config.json"
                    if os.path.exists(discord_config_path):
                        with open(discord_config_path, 'r') as f:
                            discord_config = json.load(f)
                        
                        await self.discord_intelligence.initialize(
                            bot_token=discord_config['discord_bot_token'],
                            server_configs=discord_config['servers']
                        )
                        
                        # Add signal callback
                        self.discord_intelligence.add_signal_callback(self.on_discord_signal)
                        
                        self.logger.info("✅ Discord intelligence initialized")
                    else:
                        self.logger.info("⚠️  Discord config not found - Discord intelligence disabled")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to initialize Discord intelligence: {e}")
            
            self.state = TradingState.READY
            self.logger.info("✅ Unified Trading Engine ready")
            
        except Exception as e:
            self.state = TradingState.ERROR
            self.logger.error(f"❌ Failed to initialize trading engine: {e}")
            raise

    async def analyze_opportunity(self, token_address: str) -> Optional[Dict]:
        """Analyze trading opportunity with full intelligence"""
        try:
            token_analysis = await self.pattern_intelligence.analyze_token(token_address, {})
            if not token_analysis:
                return None
            
            prediction = PatternPrediction(
                pattern_confidence=0.7,  # Default for now
                time_sensitivity=0.5,
                expected_move_size=0.1,
                supporting_patterns=[]
            )
            
            if prediction.pattern_confidence < self.min_pattern_confidence:
                return None
            
            rug_score = await self.rug_detector.analyze_token(token_address, {})
            
            security = SecurityAssessment(
                threat_level=ThreatLevel.LOW if rug_score < 0.3 else ThreatLevel.HIGH,
                risk_factors=[],
                warning_messages=[]
            )
            
            if security.threat_level > self.max_threat_level:
                return None
            
            position_size = self._calculate_position_size(prediction, security)
            
            opportunity = {
                'token_address': token_address,
                'pattern_confidence': prediction.pattern_confidence,
                'time_sensitivity': prediction.time_sensitivity,
                'position_size': position_size,
                'expected_move': prediction.expected_move_size,
                'supporting_patterns': [p.pattern_type for p in prediction.supporting_patterns],
                'security_assessment': {
                    'threat_level': security.threat_level.name,
                    'risk_factors': security.risk_factors,
                    'warning_messages': security.warning_messages
                },
                'analysis_time': datetime.now()
            }
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Opportunity analysis failed: {str(e)}")
            return None

    async def execute_trade(self, token_address: str, trade_params: Dict) -> Dict:
        """Execute a trade with advanced pattern recognition and risk management"""
        operation_id = None
        try:
            opportunity = await self.analyze_opportunity(token_address)
            if not opportunity:
                return {'success': False, 'error': 'No valid opportunity found'}
            
            operation_id = f"trade_{int(datetime.now().timestamp())}"
            context = ExecutionContext(
                token_address=token_address,
                operation_id=operation_id,
                pattern_confidence=opportunity['pattern_confidence'],
                security_assessment=opportunity['security_assessment'],
                position_size=opportunity['position_size'],
                max_slippage=trade_params.get('max_slippage', 0.01),
                timeout=30.0  # 30 seconds timeout
            )
            
            self.active_operations[operation_id] = context
            
            result = await self._execute_trade_operation(context)
            
            if result['success']:
                await self._process_successful_trade(context, result)
            else:
                await self._process_failed_trade(context, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            if operation_id and operation_id in self.active_operations:
                del self.active_operations[operation_id]

    async def run(self):
        """Main run loop for the trading swarm"""
        
        if not self.initialization_complete:
            self.logger.error("❌ Cannot run: System not properly initialized")
            return
        
        try:
            self.system_running = True
            self.logger.info("🔄 Starting main trading swarm loop...")
            
            # Start background tasks
            asyncio.create_task(self._system_monitoring_loop())
            asyncio.create_task(self._market_scanning_loop())
            asyncio.create_task(self._position_management_loop())
            asyncio.create_task(self._sentiment_analysis_loop())
            
            # Main trading loop
            while self.system_running and not self.emergency_shutdown_triggered:
                try:
                    # Check for new opportunities
                    await self._scan_for_opportunities()
                    
                    # Manage existing positions
                    await self._manage_positions()
                    
                    # Periodic system check
                    await self._periodic_system_check()
                    
                    # Adaptive sleep based on market activity
                    sleep_time = self._calculate_adaptive_sleep()
                    await asyncio.sleep(sleep_time)
                    
                except Exception as e:
                    self.logger.error(f"❌ Trading loop error: {e}")
                    await asyncio.sleep(10)  # Brief pause on error
                    
        except Exception as e:
            self.logger.error(f"❌ Main loop error: {e}")
            self.logger.error(traceback.format_exc())
            await self.emergency_shutdown()
        
        finally:
            if self.system_running:
                await self.shutdown()

    async def _scan_for_opportunities(self):
        """Scan for new trading opportunities"""
        try:
            # Get market scanner results
            if hasattr(self, 'market_scanner'):
                opportunities = await self.market_scanner.scan_markets()
                
                for opp in opportunities:
                    if await self._validate_opportunity(opp):
                        await self._execute_opportunity(opp)
                        
        except Exception as e:
            self.logger.error(f"Opportunity scanning error: {e}")

    async def _validate_opportunity(self, opportunity: Dict) -> bool:
        """Validate if opportunity meets criteria"""
        try:
            # Check sentiment score
            sentiment_score = opportunity.get('sentiment_score', 0)
            if sentiment_score < 0.7:  # Minimum sentiment threshold
                return False
                
            # Check liquidity
            liquidity = opportunity.get('liquidity_sol', 0)
            if liquidity < 10.0:  # Minimum liquidity
                return False
                
            # Check volume
            volume_24h = opportunity.get('volume_24h_sol', 0)
            if volume_24h < 100.0:  # Minimum volume
                return False
                
            # Check price impact
            price_impact = opportunity.get('price_impact_percent', 0)
            if price_impact > 3.0:  # Maximum price impact
                return False
                
            # Check if we already have position
            token_address = opportunity.get('token_address')
            if token_address in self.battlefield_state.active_trades:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Opportunity validation error: {e}")
            return False

    async def _execute_opportunity(self, opportunity: Dict):
        """Execute a trading opportunity"""
        try:
            token_address = opportunity['token_address']
            amount_sol = self._calculate_position_size(opportunity)
            
            # Get best wallet for this trade
            wallet = await self.wallet_manager.get_best_wallet(opportunity.get('risk_score', 0.5))
            if not wallet:
                return
                
            # Execute buy
            buy_result = await self.trade_executor.execute_buy(
                token_address=token_address,
                amount_sol=amount_sol,
                wallet=wallet,
                max_slippage=2.0
            )
            
            if buy_result.success:
                # Track position
                self.battlefield_state.active_trades[token_address] = {
                    'entry_price': buy_result.price,
                    'amount_sol': amount_sol,
                    'amount_tokens': buy_result.amount_tokens,
                    'entry_time': datetime.now(),
                    'wallet': wallet,
                    'sentiment_score': opportunity.get('sentiment_score', 0)
                }
                
                self.logger.info(f"✅ Position opened: {token_address} | {amount_sol} SOL")
                
        except Exception as e:
            self.logger.error(f"Opportunity execution error: {e}")

    async def _manage_positions(self):
        """Manage existing positions"""
        try:
            for token_address, position in list(self.battlefield_state.active_trades.items()):
                # Check exit conditions
                if await self._should_exit_position(token_address, position):
                    await self._exit_position(token_address, "Exit condition met")
                    
        except Exception as e:
            self.logger.error(f"Position management error: {e}")

    async def _should_exit_position(self, token_address: str, position: Dict) -> bool:
        """Check if position should be exited"""
        try:
            # Get current price
            current_price = await self._get_current_price(token_address)
            if not current_price:
                return False
                
            entry_price = position['entry_price']
            price_change = (current_price - entry_price) / entry_price
            
            # Take profit at 10%
            if price_change >= 0.10:
                return True
                
            # Stop loss at -5%
            if price_change <= -0.05:
                return True
                
            # Check sentiment change
            current_sentiment = await self._get_current_sentiment(token_address)
            if current_sentiment and current_sentiment < 0.3:  # Sentiment dropped
                return True
                
            # Time-based exit (max 2 hours)
            entry_time = position['entry_time']
            if (datetime.now() - entry_time).total_seconds() > 7200:
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Exit condition check error: {e}")
            return False

    async def _exit_position(self, token_address: str, reason: str):
        """Exit a position"""
        try:
            position = self.battlefield_state.active_trades[token_address]
            wallet = position['wallet']
            
            # Execute sell
            sell_result = await self.trade_executor.execute_sell(
                token_address=token_address,
                amount_tokens=position['amount_tokens'],
                wallet=wallet,
                max_slippage=2.0
            )
            
            if sell_result.success:
                # Calculate P&L
                entry_value = position['amount_sol']
                exit_value = sell_result.amount_sol
                pnl = exit_value - entry_value
                
                # Update wallet performance
                await self.wallet_manager.update_wallet_performance(wallet, {
                    'success': pnl > 0,
                    'profit': pnl,
                    'trade_count': 1
                })
                
                self.logger.info(f"✅ Position closed: {token_address} | P&L: {pnl:.4f} SOL | Reason: {reason}")
                
                # Remove from active trades
                del self.battlefield_state.active_trades[token_address]
                
        except Exception as e:
            self.logger.error(f"Position exit error: {e}")

    async def _market_scanning_loop(self):
        """Background market scanning loop"""
        while self.system_running:
            try:
                if hasattr(self, 'market_scanner'):
                    await self.market_scanner.update_market_data()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                self.logger.error(f"Market scanning error: {e}")
                await asyncio.sleep(60)

    async def _position_management_loop(self):
        """Background position management loop"""
        while self.system_running:
            try:
                await self._manage_positions()
                await asyncio.sleep(15)  # Check every 15 seconds
            except Exception as e:
                self.logger.error(f"Position management error: {e}")
                await asyncio.sleep(30)

    async def _sentiment_analysis_loop(self):
        """Background sentiment analysis loop"""
        while self.system_running:
            try:
                if hasattr(self, 'sentiment_analyzer'):
                    # Update sentiment for active positions
                    for token_address in self.battlefield_state.active_trades:
                        await self.sentiment_analyzer.update_token_sentiment(token_address)
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                self.logger.error(f"Sentiment analysis error: {e}")
                await asyncio.sleep(120)

    def _calculate_adaptive_sleep(self) -> int:
        """Calculate adaptive sleep time based on market activity"""
        try:
            active_positions = len(self.battlefield_state.active_trades)
            
            if active_positions == 0:
                return 30  # More frequent scanning when no positions
            elif active_positions < 3:
                return 15  # Moderate frequency
            else:
                return 10  # High frequency when managing multiple positions
                
        except Exception:
            return 30  # Default fallback

    async def _get_current_price(self, token_address: str) -> Optional[float]:
        """Get current token price"""
        try:
            if hasattr(self, 'market_scanner'):
                return await self.market_scanner.get_token_price(token_address)
            return None
        except Exception as e:
            self.logger.error(f"Price fetch error: {e}")
            return None

    async def _get_current_sentiment(self, token_address: str) -> Optional[float]:
        """Get current token sentiment"""
        try:
            if hasattr(self, 'sentiment_analyzer'):
                sentiment_data = await self.sentiment_analyzer.get_token_sentiment(token_address)
                return sentiment_data.get('sentiment', {}).get('positive', 0)
            return None
        except Exception as e:
            self.logger.error(f"Sentiment fetch error: {e}")
            return None

    async def _monitor_battlefield(self):
        """Real-time battlefield monitoring"""
        while True:
            try:
                if self.state != TradingState.TRADING:
                    await asyncio.sleep(5)
                    continue
                
                
                for token_address in list(self.battlefield_state.active_trades.keys()):
                    await self._monitor_position(token_address)
                
                
                for token_address in self.battlefield_state.monitored_tokens:
                    if token_address not in self.battlefield_state.active_trades:
                        opportunity = await self.analyze_opportunity(token_address)
                        if opportunity:
                            await self.execute_trade(token_address, {})
                
                
                self.battlefield_state.last_update = datetime.now()
                
                await asyncio.sleep(1)  # 1 second monitoring interval
                
            except Exception as e:
                self.logger.error(f"Battlefield monitoring error: {str(e)}")
                await asyncio.sleep(5)

    async def _monitor_position(self, token_address: str) -> Dict:
        """Monitor active position with pattern intelligence"""
        try:
            if token_address not in self.battlefield_state.active_trades:
                return None
            
            position = self.battlefield_state.active_trades[token_address]
            
            
            patterns = await self.pattern_intelligence.detect_patterns(token_address)
            
            
            security = await self.rug_detector.analyze_security(token_address)
            
            
            exit_required = await self._check_exit_conditions(
                token_address, patterns, security
            )
            
            if exit_required:
                await self._exit_position(token_address, "Pattern-based exit")
            
            
            position.update({
                'current_patterns': [p.pattern_type for p in patterns],
                'threat_level': security.threat_level.name,
                'last_update': datetime.now()
            })
            
            return position
            
        except Exception as e:
            self.logger.error(f"Position monitoring failed: {str(e)}")
            return None

    def _calculate_position_size(self, prediction: PatternPrediction,
                               security: SecurityAssessment) -> float:
        """Calculate optimal position size based on multiple factors"""
        try:
            base_size = prediction.pattern_confidence * self.position_limits['max_position_size']
            
            threat_multiplier = 1.0 - (security.threat_level.value / 10.0)
            adjusted_size = base_size * threat_multiplier
            
            final_size = min(
                adjusted_size,
                self.position_limits['max_exposure_per_token']
            )
            
            return final_size
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}")
            return 0.0

    async def _check_exit_conditions(self, token_address: str,
                                   patterns: List[ChainPattern],
                                   security: SecurityAssessment) -> bool:
        """Check if position should be exited"""
        try:
            position = self.battlefield_state.active_trades.get(token_address)
            if not position:
                return False
            
            pattern_breakdown = any(
                p.pattern_type == position['pattern_id'] and p.is_breaking_down
                for p in patterns
            )
            
            security_deterioration = (
                security.threat_level.value >
                self.max_threat_level.value
            )
            
            return pattern_breakdown or security_deterioration
            
        except Exception as e:
            self.logger.error(f"Exit condition check failed: {str(e)}")
            return True  # Exit on error for safety

    async def _exit_position(self, token_address: str, reason: str):
        """Exit a position with proper cleanup"""
        try:
            position = self.battlefield_state.active_trades.get(token_address)
            if not position:
                return
            
            result = await self.trade_executor.execute_exit(
                token_address=token_address,
                wallet_address=self.wallet_manager.active_wallet,
                position_size=position['position_size']
            )
            
            if result and result.success:
                self._update_performance_metrics(token_address, result)
                del self.battlefield_state.active_trades[token_address]
                self.logger.info(f"Successfully exited position: {token_address}")
            else:
                self.logger.error(f"Failed to exit position: {token_address}")
            
        except Exception as e:
            self.logger.error(f"Error exiting position: {str(e)}")

    def _update_performance_metrics(self, token_address: str, result: ExecutionResult):
        """Update performance metrics after trade completion"""
        try:
            metrics = self.battlefield_state.performance_metrics
            
            
            profit = result.effective_price - self.battlefield_state.active_trades[token_address]['entry_price']
            profit_pct = profit / self.battlefield_state.active_trades[token_address]['entry_price']
            
            
            metrics['total_trades'] = metrics.get('total_trades', 0) + 1
            metrics['profitable_trades'] = metrics.get('profitable_trades', 0) + (1 if profit > 0 else 0)
            metrics['win_rate'] = metrics['profitable_trades'] / metrics['total_trades']
            metrics['avg_profit'] = (metrics.get('avg_profit', 0) * (metrics['total_trades'] - 1) + profit_pct) / metrics['total_trades']
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")

    async def shutdown(self):
        """Shutdown the trading engine"""
        try:
            self.logger.info("Initiating trading engine shutdown")
            
            
            await self._update_state(TradingState.SHUTDOWN)
            
            
            for token_address in list(self.battlefield_state.active_trades.keys()):
                await self._exit_position(token_address, "System shutdown")
            
            
            await self.wallet_manager.shutdown()
            await self.pattern_intelligence.shutdown()
            await self.trade_executor.shutdown()
            await self.rug_detector.shutdown()
            await self.kill_switch.shutdown()
            await self.monitoring.shutdown()
            
            self.logger.info("Trading engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

async def get_trading_engine() -> UnifiedTradingEngine:
    """Get or create trading engine instance"""
    engine = UnifiedTradingEngine()
    await engine.initialize()
    return engine 

    def on_discord_signal(self, signal: DiscordSignal):
        """Handle incoming Discord signal"""
        
        try:
            self.logger.info(
                f"📡 Discord signal: {signal.signal_type.value.upper()} "
                f"from {signal.author_name} - {signal.token_name or signal.token_address} "
                f"(confidence: {signal.confidence:.2f})"
            )
            
            # Process high-confidence signals
            if signal.confidence >= 0.7:
                asyncio.create_task(self.process_discord_signal(signal))
            
        except Exception as e:
            self.logger.error(f"Error handling Discord signal: {e}")
    
    async def process_discord_signal(self, signal: DiscordSignal):
        """Process Discord signal for trading"""
        
        try:
            # Validate signal
            if not await self._validate_discord_signal(signal):
                return
            
            # Create trading opportunity
            opportunity = {
                'token_address': signal.token_address,
                'token_name': signal.token_name,
                'signal_type': signal.signal_type.value,
                'confidence': signal.confidence,
                'source': 'discord',
                'author': signal.author_name,
                'timestamp': signal.timestamp,
                'price_target': signal.price_target,
                'stop_loss': signal.stop_loss,
                'reasoning': signal.reasoning
            }
            
            # Execute if confidence is high enough
            if signal.confidence >= 0.8:
                await self._execute_opportunity(opportunity)
            else:
                # Add to monitoring
                self.battlefield_state.monitored_tokens[signal.token_address] = opportunity
                
        except Exception as e:
            self.logger.error(f"Error processing Discord signal: {e}")
    
    async def _validate_discord_signal(self, signal: DiscordSignal) -> bool:
        """Validate Discord signal"""
        
        # Check if we have token information
        if not signal.token_address and not signal.token_name:
            return False
        
        # Check if signal is recent (within last 5 minutes)
        signal_age = (datetime.now() - signal.timestamp).total_seconds()
        if signal_age > 300:  # 5 minutes
            return False
        
        # Check for duplicate signals
        if signal.token_address in self.battlefield_state.monitored_tokens:
            existing = self.battlefield_state.monitored_tokens[signal.token_address]
            if existing.get('source') == 'discord':
                return False
        
        return True 