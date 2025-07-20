"""
SWARM DECISION ENGINE - NEURAL COMMAND CENTER BRAIN (NATS Enhanced)
==================================================================

The core decision-making engine for the 10-wallet neural swarm.
Integrates all intelligence sources and makes final trading decisions using NATS message bus.

Enhanced with NATS messaging for distributed communication:
- Multi-source consensus validation
- Pattern-based opportunity analysis  
- Risk-adjusted position sizing
- Stealth execution coordination
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from worker_ant_v1.trading.colony_commander import ColonyCommander
from worker_ant_v1.trading.stealth_operations import StealthOperationsSystem
from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
from worker_ant_v1.core.vault_wallet_system import VaultWalletSystem
from worker_ant_v1.core.unified_trading_engine import UnifiedTradingEngine
from worker_ant_v1.intelligence.enhanced_rug_detector import EnhancedRugDetector
from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch
from worker_ant_v1.trading.squad_manager import SquadManager
from worker_ant_v1.trading.ml_predictor import MLPredictor
from worker_ant_v1.trading.ml_architectures.prediction_engine import PredictionEngine, UnifiedPrediction
from worker_ant_v1.core.message_bus import get_message_bus, MessageBus, MessageEnvelope, MessageType, MessagePriority, swarm_subject, intelligence_subject
from worker_ant_v1.utils.logger import setup_logger
from worker_ant_v1.utils.constants import SentimentDecision as SentimentDecisionEnum

class SwarmState(Enum):
    """Swarm operational states"""
    INITIALIZING = "initializing"
    HUNTING = "hunting"
    FEASTING = "feasting"
    STALKING = "stalking"
    RETREATING = "retreating"
    EVOLVING = "evolving"
    HIBERNATING = "hibernating"
    EMERGENCY = "emergency"


@dataclass
class SwarmDecision:
    """Swarm decision with comprehensive analysis"""
    decision_id: str
    token_address: str
    swarm_state: SwarmState
    consensus_level: float
    confidence_score: float
    recommended_action: str
    position_size: float
    squad_assignment: Optional[str]
    risk_level: float
    execution_priority: int
    timestamp: datetime
    reasoning: str
    intelligence_sources: Dict[str, Any]
    wallet_assignments: Dict[str, str]
    expected_profit: float
    max_drawdown: float
    time_horizon_minutes: int


class SwarmDecisionEngine:
    """Orchestrates decisions across the 10-wallet swarm using NATS message bus"""
    
    def __init__(self, swarm_id: str = "default"):
        self.logger = setup_logger("SwarmDecisionEngine")
        self.swarm_id = swarm_id
        
        # Message bus for distributed communication
        self.message_bus: Optional[MessageBus] = None
        
        # Core components
        self.neural_command_center: Optional[ColonyCommander] = None
        self.stealth_operations: Optional[StealthOperationsSystem] = None
        self.wallet_manager: Optional[UnifiedWalletManager] = None
        self.vault_system: Optional[VaultWalletSystem] = None
        self.trading_engine: Optional[UnifiedTradingEngine] = None
        self.rug_detector: Optional[EnhancedRugDetector] = None
        self.kill_switch: Optional[EnhancedKillSwitch] = None
        self.squad_manager: Optional[SquadManager] = None
        self.ml_predictor: Optional[MLPredictor] = None
        self.prediction_engine: Optional[PredictionEngine] = None
        
        # Swarm state
        self.current_state = SwarmState.HUNTING
        self.swarm_health = 1.0
        self.decision_history: List[SwarmDecision] = []
        
        # Performance tracking
        self.swarm_metrics = {
            'total_opportunities': 0,
            'executed_opportunities': 0,
            'successful_executions': 0,
            'total_profit': 0.0,
            'avg_consensus_level': 0.0,
            'swarm_efficiency': 0.0
        }
        
        # Configuration
        self.config = {
            'min_consensus_threshold': 0.7,
            'max_risk_per_opportunity': 0.3,
            'opportunity_timeout_minutes': 30,
            'swarm_coordination_delay': 2.0,  # seconds
            'evolution_trigger_threshold': 0.6
        }
        
        # System state
        self.initialized = False
        self.decision_making_active = False
        
    async def initialize_swarm(self) -> bool:
        """Initialize all swarm components with NATS message bus"""
        try:
            self.logger.info("🧬 Initializing Swarm Decision Engine with NATS message bus...")
            
            # Initialize message bus
            self.message_bus = await get_message_bus()
            
            # Initialize core components
            self.neural_command_center = ColonyCommander()
            self.stealth_operations = StealthOperationsSystem()
            
            # Get wallet and vault systems
            from worker_ant_v1.core.wallet_manager import get_wallet_manager
            from worker_ant_v1.core.vault_wallet_system import get_vault_system
            
            self.wallet_manager = await get_wallet_manager()
            self.vault_system = await get_vault_system()
            
            # Initialize trading engine
            self.trading_engine = UnifiedTradingEngine()
            await self.trading_engine.initialize()
            
            # Initialize safety systems
            self.rug_detector = EnhancedRugDetector(
                solana_client=None,  # Will be injected
                jupiter_dex=None     # Will be injected
            )
            await self.rug_detector.initialize()
            
            self.kill_switch = EnhancedKillSwitch()
            await self.kill_switch.initialize()
            
            # Initialize squad manager
            self.squad_manager = SquadManager()
            await self.squad_manager.initialize(self.wallet_manager)
            
            # Initialize ML predictor with state-of-the-art architectures
            self.ml_predictor = MLPredictor()
            
            # Initialize prediction engine
            self.prediction_engine = PredictionEngine()
            await self.prediction_engine.initialize()
            
            # Initialize Hunter Ants for all wallets
            active_wallets = await self.wallet_manager.get_all_wallets()
            wallet_ids = list(active_wallets.keys())
            await self.ml_predictor.initialize_hunter_ants(wallet_ids)
            await self.prediction_engine.initialize_hunter_ants(wallet_ids)
            
            # Connect components
            await self.neural_command_center.initialize(self.wallet_manager, self.vault_system)
            
            # Setup NATS message subscriptions
            await self._setup_message_subscriptions()
            
            # Start background tasks
            asyncio.create_task(self._decision_making_loop())
            asyncio.create_task(self._performance_tracking_loop())
            asyncio.create_task(self._swarm_health_monitoring())
            
            self.current_state = SwarmState.HUNTING
            self.initialized = True
            self.decision_making_active = True
            
            self.logger.info("✅ Swarm Decision Engine initialized and hunting with NATS messaging")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize swarm: {e}")
            return False

    async def _setup_message_subscriptions(self):
        """Setup NATS message subscriptions for swarm communication"""
        try:
            # Subscribe to market opportunities
            await self.message_bus.subscribe(
                intelligence_subject("market"),
                self._handle_market_opportunity,
                queue_group=f"swarm_{self.swarm_id}"
            )
            
            # Subscribe to sentiment signals
            await self.message_bus.subscribe(
                intelligence_subject("sentiment"),
                self._handle_sentiment_signal,
                queue_group=f"swarm_{self.swarm_id}"
            )
            
            # Subscribe to ML predictions
            await self.message_bus.subscribe(
                intelligence_subject("ml"),
                self._handle_ml_prediction,
                queue_group=f"swarm_{self.swarm_id}"
            )
            
            # Subscribe to colony commands
            await self.message_bus.subscribe(
                f"colony.command.{self.swarm_id}",
                self._handle_colony_command
            )
            
            # Subscribe to colony-wide commands
            await self.message_bus.subscribe(
                "colony.command.all",
                self._handle_colony_command
            )
            
            # Subscribe to safety alerts
            await self.message_bus.subscribe(
                "safety.alert.*",
                self._handle_safety_alert
            )
            
            # Subscribe to kill switch
            await self.message_bus.subscribe(
                "safety.kill_switch.*",
                self._handle_kill_switch
            )
            
            self.logger.info("✅ NATS message subscriptions setup complete")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup message subscriptions: {e}")

    async def _handle_market_opportunity(self, message: MessageEnvelope):
        """Handle market opportunity messages from intelligence systems"""
        try:
            opportunity_data = message.data
            token_address = opportunity_data.get('token_address')
            
            if not token_address:
                self.logger.warning("⚠️ Received market opportunity without token address")
                return
            
            self.swarm_metrics['total_opportunities'] += 1
            
            # Analyze opportunity using neural command center
            consensus_signal = await self.neural_command_center.analyze_opportunity(
                token_address, opportunity_data
            )
            
            # Make swarm decision
            decision = await self._make_swarm_decision(consensus_signal, opportunity_data)
            
            if decision and decision.recommended_action != "HOLD":
                # Publish decision to execution systems
                await self._publish_swarm_decision(decision)
                
            self.logger.info(f"📊 Processed market opportunity for {token_address}: {decision.recommended_action if decision else 'REJECTED'}")
            
        except Exception as e:
            self.logger.error(f"❌ Error handling market opportunity: {e}")

    async def _handle_sentiment_signal(self, message: MessageEnvelope):
        """Handle sentiment analysis signals"""
        try:
            sentiment_data = message.data
            self.logger.debug(f"📈 Received sentiment signal: {sentiment_data.get('sentiment', 'unknown')}")
            
            # Forward to neural command center for integration
            # This will be used in the next opportunity analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error handling sentiment signal: {e}")

    async def _handle_ml_prediction(self, message: MessageEnvelope):
        """Handle ML prediction messages"""
        try:
            prediction_data = message.data
            token_address = prediction_data.get('token_address')
            
            if token_address:
                self.logger.debug(f"🤖 Received ML prediction for {token_address}")
                # Process ML prediction as market opportunity
                await self._handle_market_opportunity(message)
            
        except Exception as e:
            self.logger.error(f"❌ Error handling ML prediction: {e}")

    async def _handle_colony_command(self, message: MessageEnvelope):
        """Handle commands from Colony Commander"""
        try:
            command_data = message.data
            command = command_data.get('command')
            
            self.logger.info(f"📢 Received colony command: {command}")
            
            if command == "activate_blitzscaling":
                await self._activate_blitzscaling_mode()
            elif command == "deactivate_blitzscaling":
                await self._deactivate_blitzscaling_mode()
            elif command == "change_state":
                new_state = command_data.get('state')
                if new_state:
                    self.current_state = SwarmState(new_state)
                    self.logger.info(f"🔄 Swarm state changed to: {new_state}")
            elif command == "shutdown":
                await self._shutdown_swarm()
            elif command == "rebalance":
                await self._rebalance_swarm_capital()
            
        except Exception as e:
            self.logger.error(f"❌ Error handling colony command: {e}")

    async def _handle_safety_alert(self, message: MessageEnvelope):
        """Handle safety alerts"""
        try:
            alert_data = message.data
            alert_type = alert_data.get('type')
            severity = alert_data.get('severity', 'medium')
            
            self.logger.warning(f"🚨 Safety alert received: {alert_type} (severity: {severity})")
            
            if severity == 'high' or alert_type == 'rug_detected':
                # Immediate risk mitigation
                self.current_state = SwarmState.RETREATING
                await self._execute_emergency_procedures()
            
        except Exception as e:
            self.logger.error(f"❌ Error handling safety alert: {e}")

    async def _handle_kill_switch(self, message: MessageEnvelope):
        """Handle kill switch activation"""
        try:
            kill_data = message.data
            reason = kill_data.get('reason', 'Unknown')
            
            self.logger.error(f"🛑 KILL SWITCH ACTIVATED: {reason}")
            
            self.current_state = SwarmState.EMERGENCY
            self.decision_making_active = False
            
            # Immediately stop all trading activities
            await self._emergency_shutdown()
            
        except Exception as e:
            self.logger.error(f"❌ Error handling kill switch: {e}")

    async def _make_swarm_decision(self, consensus_signal, opportunity_data: Dict[str, Any]) -> Optional[SwarmDecision]:
        """Make swarm-level trading decision based on consensus"""
        try:
            if not consensus_signal.passes_consensus:
                return None
            
            token_address = opportunity_data.get('token_address')
            
            # Get ML prediction for additional intelligence
            ml_prediction = await self.prediction_engine.predict(
                token_address, 
                opportunity_data,
                prediction_horizon=15
            )
            
            # Determine squad assignment
            squad_assignment = None
            if self.squad_manager:
                squad_assignment = await self.squad_manager.determine_squad_for_opportunity(opportunity_data)
            
            # Calculate position sizing based on risk and confidence
            position_size = self._calculate_position_size(
                consensus_signal.confidence_level,
                ml_prediction.overall_confidence,
                opportunity_data.get('risk_score', 0.5)
            )
            
            # Create swarm decision
            decision = SwarmDecision(
                decision_id=f"swarm_{self.swarm_id}_{int(time.time())}",
                token_address=token_address,
                swarm_state=self.current_state,
                consensus_level=consensus_signal.consensus_strength,
                confidence_score=consensus_signal.confidence_level,
                recommended_action=consensus_signal.recommended_action,
                position_size=position_size,
                squad_assignment=squad_assignment,
                risk_level=opportunity_data.get('risk_score', 0.5),
                execution_priority=self._calculate_execution_priority(consensus_signal),
                timestamp=datetime.utcnow(),
                reasoning=consensus_signal.reasoning,
                intelligence_sources={
                    'ai_prediction': consensus_signal.ai_prediction,
                    'onchain_sentiment': consensus_signal.onchain_sentiment,
                    'twitter_sentiment': consensus_signal.twitter_sentiment,
                    'technical_score': consensus_signal.technical_score,
                    'ml_prediction': ml_prediction.trading_recommendation
                },
                wallet_assignments={},  # Will be filled by squad manager
                expected_profit=ml_prediction.oracle_price_prediction * position_size,
                max_drawdown=position_size * 0.1,  # 10% max drawdown
                time_horizon_minutes=15
            )
            
            # Store decision in history
            self.decision_history.append(decision)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"❌ Error making swarm decision: {e}")
            return None

    async def _publish_swarm_decision(self, decision: SwarmDecision):
        """Publish swarm decision to execution systems"""
        try:
            # Send to trading engine for execution
            await self.message_bus.send_trade_order(
                {
                    'decision_id': decision.decision_id,
                    'token_address': decision.token_address,
                    'action': decision.recommended_action,
                    'position_size': decision.position_size,
                    'risk_level': decision.risk_level,
                    'squad_assignment': decision.squad_assignment,
                    'priority': decision.execution_priority
                },
                "trading_engine"
            )
            
            # Send to squad manager if assigned
            if decision.squad_assignment:
                await self.message_bus.send_squad_signal(
                    {
                        'decision_id': decision.decision_id,
                        'token_address': decision.token_address,
                        'action': decision.recommended_action,
                        'position_size': decision.position_size
                    },
                    decision.squad_assignment
                )
            
            # Broadcast decision to interested systems
            decision_message = MessageEnvelope(
                message_id=decision.decision_id,
                message_type=MessageType.SWARM_DECISION,
                subject=swarm_subject(self.swarm_id, "decision"),
                sender_id=f"swarm_{self.swarm_id}",
                timestamp=datetime.utcnow(),
                priority=MessagePriority.HIGH,
                data={
                    'decision': {
                        'decision_id': decision.decision_id,
                        'token_address': decision.token_address,
                        'action': decision.recommended_action,
                        'position_size': decision.position_size,
                        'confidence_score': decision.confidence_score,
                        'squad_assignment': decision.squad_assignment
                    }
                },
                broadcast=True
            )
            
            await self.message_bus.publish(decision_message)
            
            self.swarm_metrics['executed_opportunities'] += 1
            self.logger.info(f"📡 Published swarm decision {decision.decision_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error publishing swarm decision: {e}")

    def _calculate_position_size(self, consensus_confidence: float, ml_confidence: float, risk_score: float) -> float:
        """Calculate position size based on multiple confidence factors"""
        try:
            # Base position size on combined confidence
            combined_confidence = (consensus_confidence + ml_confidence) / 2
            
            # Risk adjustment
            risk_adjustment = 1.0 - risk_score
            
            # Position size calculation
            base_size = combined_confidence * 0.3  # Max 30% base allocation
            adjusted_size = base_size * risk_adjustment
            
            # Apply swarm state modifiers
            if self.current_state == SwarmState.FEASTING:
                adjusted_size *= 1.5  # Increase size during profitable periods
            elif self.current_state == SwarmState.RETREATING:
                adjusted_size *= 0.5  # Reduce size during risky periods
            
            # Cap at maximum risk per opportunity
            max_size = self.config['max_risk_per_opportunity']
            
            return min(adjusted_size, max_size)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating position size: {e}")
            return 0.1  # Conservative fallback

    def _calculate_execution_priority(self, consensus_signal) -> int:
        """Calculate execution priority (1=highest, 5=lowest)"""
        try:
            # High consensus and confidence = high priority
            if consensus_signal.consensus_strength > 0.9 and consensus_signal.confidence_level > 0.8:
                return 1
            elif consensus_signal.consensus_strength > 0.8 and consensus_signal.confidence_level > 0.7:
                return 2
            elif consensus_signal.consensus_strength > 0.7 and consensus_signal.confidence_level > 0.6:
                return 3
            elif consensus_signal.consensus_strength > 0.6:
                return 4
            else:
                return 5
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating execution priority: {e}")
            return 5

    async def _decision_making_loop(self):
        """Main decision-making loop (replaced queue-based with event-driven)"""
        while self.decision_making_active:
            try:
                # Monitor swarm health and adjust state
                await self._monitor_swarm_health()
                
                # Process any pending analysis
                await self._process_pending_analysis()
                
                # Sleep and wait for message-driven events
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Decision making loop error: {e}")
                await asyncio.sleep(10)

    async def _monitor_swarm_health(self):
        """Monitor overall swarm health and performance"""
        try:
            # Calculate success rate
            if self.swarm_metrics['executed_opportunities'] > 0:
                success_rate = self.swarm_metrics['successful_executions'] / self.swarm_metrics['executed_opportunities']
            else:
                success_rate = 0.0
            
            # Update swarm health
            self.swarm_health = success_rate
            
            # State transitions based on performance
            if success_rate > 0.8 and self.current_state != SwarmState.FEASTING:
                self.current_state = SwarmState.FEASTING
                self.logger.info("🍯 Swarm state: FEASTING (high success rate)")
            elif success_rate < 0.4 and self.current_state != SwarmState.RETREATING:
                self.current_state = SwarmState.RETREATING
                self.logger.info("🛡️ Swarm state: RETREATING (low success rate)")
            elif 0.4 <= success_rate <= 0.8 and self.current_state not in [SwarmState.HUNTING, SwarmState.STALKING]:
                self.current_state = SwarmState.HUNTING
                self.logger.info("🎯 Swarm state: HUNTING (moderate performance)")
            
        except Exception as e:
            self.logger.error(f"❌ Error monitoring swarm health: {e}")

    async def _process_pending_analysis(self):
        """Process any pending analysis or maintenance tasks"""
        try:
            # Clean up old decisions
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.decision_history = [
                d for d in self.decision_history 
                if d.timestamp > cutoff_time
            ]
            
            # Update performance metrics
            await self._update_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"❌ Error in pending analysis: {e}")

    async def _performance_tracking_loop(self):
        """Background performance tracking"""
        while self.decision_making_active:
            try:
                await self._update_performance_metrics()
                
                # Publish performance status
                await self._publish_swarm_status()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"❌ Performance tracking error: {e}")
                await asyncio.sleep(60)

    async def _update_performance_metrics(self):
        """Update swarm performance metrics"""
        try:
            # Calculate efficiency
            if self.swarm_metrics['executed_opportunities'] > 0:
                self.swarm_metrics['swarm_efficiency'] = (
                    self.swarm_metrics['successful_executions'] / 
                    self.swarm_metrics['executed_opportunities']
                )
            
            # Calculate average consensus level
            if self.decision_history:
                recent_decisions = self.decision_history[-100:]  # Last 100 decisions
                self.swarm_metrics['avg_consensus_level'] = sum(
                    d.consensus_level for d in recent_decisions
                ) / len(recent_decisions)
            
        except Exception as e:
            self.logger.error(f"❌ Error updating performance metrics: {e}")

    async def _publish_swarm_status(self):
        """Publish current swarm status to message bus"""
        try:
            status_message = MessageEnvelope(
                message_id=f"status_{self.swarm_id}_{int(time.time())}",
                message_type=MessageType.SWARM_STATUS,
                subject=swarm_subject(self.swarm_id, "status"),
                sender_id=f"swarm_{self.swarm_id}",
                timestamp=datetime.utcnow(),
                priority=MessagePriority.LOW,
                data={
                    'swarm_id': self.swarm_id,
                    'current_state': self.current_state.value,
                    'swarm_health': self.swarm_health,
                    'metrics': self.swarm_metrics,
                    'active_decisions': len(self.decision_history)
                }
            )
            
            await self.message_bus.publish(status_message)
            
        except Exception as e:
            self.logger.error(f"❌ Error publishing swarm status: {e}")

    async def _swarm_health_monitoring(self):
        """Background swarm health monitoring"""
        while self.decision_making_active:
            try:
                await self._monitor_swarm_health()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Swarm health monitoring error: {e}")
                await asyncio.sleep(30)

    async def _activate_blitzscaling_mode(self):
        """Activate aggressive blitzscaling mode"""
        try:
            self.config['max_risk_per_opportunity'] = 0.5  # Increase max risk
            self.current_state = SwarmState.FEASTING
            self.logger.info("🚀 BLITZSCALING MODE ACTIVATED")
            
        except Exception as e:
            self.logger.error(f"❌ Error activating blitzscaling mode: {e}")

    async def _deactivate_blitzscaling_mode(self):
        """Deactivate blitzscaling mode"""
        try:
            self.config['max_risk_per_opportunity'] = 0.3  # Reset to conservative
            self.current_state = SwarmState.HUNTING
            self.logger.info("🛡️ Blitzscaling mode deactivated")
            
        except Exception as e:
            self.logger.error(f"❌ Error deactivating blitzscaling mode: {e}")

    async def _execute_emergency_procedures(self):
        """Execute emergency risk mitigation procedures"""
        try:
            self.logger.warning("🚨 Executing emergency procedures")
            
            # Reduce all position sizes
            self.config['max_risk_per_opportunity'] = 0.1
            
            # Pause decision making temporarily
            self.decision_making_active = False
            await asyncio.sleep(60)  # 1 minute pause
            self.decision_making_active = True
            
        except Exception as e:
            self.logger.error(f"❌ Error in emergency procedures: {e}")

    async def _emergency_shutdown(self):
        """Emergency shutdown of swarm operations"""
        try:
            self.logger.error("🛑 EMERGENCY SHUTDOWN INITIATED")
            
            self.decision_making_active = False
            
            # Send emergency stop to all connected systems
            await self.message_bus.send_kill_switch(
                "Swarm emergency shutdown",
                f"swarm_{self.swarm_id}"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error during emergency shutdown: {e}")

    async def _shutdown_swarm(self):
        """Graceful shutdown of swarm operations"""
        try:
            self.logger.info("🛑 Shutting down swarm...")
            
            self.decision_making_active = False
            
            # Publish shutdown status
            await self._publish_swarm_status()
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")

    async def _rebalance_swarm_capital(self):
        """Rebalance capital allocation across swarm"""
        try:
            self.logger.info("⚖️ Rebalancing swarm capital...")
            
            # Implementation would coordinate with vault system
            # for capital reallocation based on performance
            
        except Exception as e:
            self.logger.error(f"❌ Error during capital rebalancing: {e}")

    def get_swarm_metrics(self) -> Dict[str, Any]:
        """Get current swarm performance metrics"""
        return {
            'swarm_id': self.swarm_id,
            'current_state': self.current_state.value,
            'swarm_health': self.swarm_health,
            'metrics': self.swarm_metrics,
            'recent_decisions': len([d for d in self.decision_history if d.timestamp > datetime.utcnow() - timedelta(hours=1)]),
            'message_bus_stats': self.message_bus.get_stats() if self.message_bus else None
        } 