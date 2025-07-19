"""
SWARM DECISION ENGINE - 10-WALLET SWARM ORCHESTRATOR
==================================================

Orchestrates decisions across the 10-wallet swarm, managing
opportunity distribution, risk allocation, and swarm coordination.
"""

import asyncio
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.trading.neural_command_center import NeuralCommandCenter
from worker_ant_v1.trading.stealth_operations import StealthOperationsSystem
from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
from worker_ant_v1.core.vault_wallet_system import VaultWalletSystem
from worker_ant_v1.trading.unified_trading_engine import UnifiedTradingEngine
from worker_ant_v1.intelligence.enhanced_rug_detector import EnhancedRugDetector
from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch

class SwarmState(Enum):
    HUNTING = "hunting"
    STALKING = "stalking"
    FEASTING = "feasting"
    RETREATING = "retreating"
    EVOLVING = "evolving"
    HIBERNATING = "hibernating"

class OpportunityType(Enum):
    HIGH_CONFIDENCE = "high_confidence"
    MEDIUM_CONFIDENCE = "medium_confidence"
    LOW_CONFIDENCE = "low_confidence"
    EXPERIMENTAL = "experimental"

@dataclass
class SwarmOpportunity:
    """Opportunity for the swarm"""
    token_address: str
    opportunity_type: OpportunityType
    confidence_score: float
    risk_level: float
    expected_profit: float
    time_sensitivity: float
    swarm_consensus: float
    wallet_assignments: Dict[str, float]  # wallet_id -> allocation_percentage
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SwarmDecision:
    """Swarm-level decision"""
    decision_id: str
    opportunity: SwarmOpportunity
    action: str  # "execute", "pass", "stake", "evolve"
    execution_plan: Dict[str, Any]
    risk_assessment: Dict[str, float]
    consensus_level: float
    timestamp: datetime
    reasoning: List[str] = field(default_factory=list)

class SwarmDecisionEngine:
    """Orchestrates decisions across the 10-wallet swarm"""
    
    def __init__(self):
        self.logger = get_logger("SwarmDecisionEngine")
        
        # Core components
        self.neural_command_center: Optional[NeuralCommandCenter] = None
        self.stealth_operations: Optional[StealthOperationsSystem] = None
        self.wallet_manager: Optional[UnifiedWalletManager] = None
        self.vault_system: Optional[VaultWalletSystem] = None
        self.trading_engine: Optional[UnifiedTradingEngine] = None
        self.rug_detector: Optional[EnhancedRugDetector] = None
        self.kill_switch: Optional[EnhancedKillSwitch] = None
        
        # Swarm state
        self.current_state = SwarmState.HUNTING
        self.swarm_health = 1.0
        self.opportunity_queue: asyncio.Queue = asyncio.Queue()
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
        """Initialize all swarm components"""
        try:
            self.logger.info("🧬 Initializing Swarm Decision Engine...")
            
            # Initialize core components
            self.neural_command_center = NeuralCommandCenter()
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
            
            # Connect components
            self.neural_command_center.wallet_manager = self.wallet_manager
            self.neural_command_center.vault_system = self.vault_system
            
            # Start background tasks
            asyncio.create_task(self._opportunity_monitoring_loop())
            asyncio.create_task(self._decision_making_loop())
            asyncio.create_task(self._performance_tracking_loop())
            
            self.current_state = SwarmState.HUNTING
            self.logger.info("✅ Swarm Decision Engine initialized and hunting")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize swarm: {e}")
            return False
    
    async def submit_opportunity(self, token_address: str, market_data: Dict[str, Any]) -> str:
        """Submit an opportunity for swarm evaluation"""
        try:
            # Generate opportunity ID
            opportunity_id = f"opp_{token_address}_{int(datetime.now().timestamp())}"
            
            # Create swarm opportunity
            opportunity = await self._create_swarm_opportunity(token_address, market_data)
            
            # Add to queue
            await self.opportunity_queue.put(opportunity)
            
            self.swarm_metrics['total_opportunities'] += 1
            self.logger.info(f"🎯 Opportunity submitted: {opportunity_id} for {token_address}")
            
            return opportunity_id
            
        except Exception as e:
            self.logger.error(f"Error submitting opportunity: {e}")
            return ""
    
    async def _create_swarm_opportunity(self, token_address: str, market_data: Dict[str, Any]) -> SwarmOpportunity:
        """Create a swarm opportunity from market data"""
        try:
            # Get neural consensus
            consensus_signal = await self.neural_command_center.analyze_opportunity(token_address, market_data)
            
            # Determine opportunity type and confidence
            if consensus_signal.consensus_strength >= 0.8:
                opportunity_type = OpportunityType.HIGH_CONFIDENCE
                confidence_score = consensus_signal.consensus_strength
            elif consensus_signal.consensus_strength >= 0.6:
                opportunity_type = OpportunityType.MEDIUM_CONFIDENCE
                confidence_score = consensus_signal.consensus_strength
            elif consensus_signal.consensus_strength >= 0.4:
                opportunity_type = OpportunityType.LOW_CONFIDENCE
                confidence_score = consensus_signal.consensus_strength
            else:
                opportunity_type = OpportunityType.EXPERIMENTAL
                confidence_score = consensus_signal.consensus_strength
            
            # Calculate risk level
            risk_level = self._calculate_opportunity_risk(token_address, market_data, consensus_signal)
            
            # Estimate expected profit
            expected_profit = self._estimate_expected_profit(market_data, consensus_signal)
            
            # Calculate time sensitivity
            time_sensitivity = self._calculate_time_sensitivity(market_data)
            
            # Get swarm consensus
            swarm_consensus = await self._get_swarm_consensus(token_address, consensus_signal)
            
            # Assign wallets
            wallet_assignments = await self._assign_wallets_to_opportunity(opportunity_type, confidence_score, risk_level)
            
            opportunity = SwarmOpportunity(
                token_address=token_address,
                opportunity_type=opportunity_type,
                confidence_score=confidence_score,
                risk_level=risk_level,
                expected_profit=expected_profit,
                time_sensitivity=time_sensitivity,
                swarm_consensus=swarm_consensus,
                wallet_assignments=wallet_assignments,
                timestamp=datetime.now(),
                metadata={
                    'consensus_signal': consensus_signal,
                    'market_data': market_data
                }
            )
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Error creating swarm opportunity: {e}")
            # Return minimal opportunity
            return SwarmOpportunity(
                token_address=token_address,
                opportunity_type=OpportunityType.EXPERIMENTAL,
                confidence_score=0.1,
                risk_level=1.0,
                expected_profit=0.0,
                time_sensitivity=0.0,
                swarm_consensus=0.0,
                wallet_assignments={},
                timestamp=datetime.now()
            )
    
    def _calculate_opportunity_risk(self, token_address: str, market_data: Dict[str, Any], 
                                  consensus_signal) -> float:
        """Calculate risk level for an opportunity"""
        try:
            risk_factors = []
            
            # Consensus risk
            consensus_risk = 1.0 - consensus_signal.consensus_strength
            risk_factors.append(consensus_risk * 0.4)
            
            # Market volatility risk
            volatility = market_data.get('volatility', 0.5)
            risk_factors.append(volatility * 0.3)
            
            # Liquidity risk
            liquidity = market_data.get('liquidity', 0.0)
            liquidity_risk = max(0.0, 1.0 - liquidity / 1000.0)  # Normalize to 1000 SOL
            risk_factors.append(liquidity_risk * 0.2)
            
            # Time sensitivity risk
            time_sensitivity = market_data.get('time_sensitivity', 0.5)
            risk_factors.append(time_sensitivity * 0.1)
            
            return sum(risk_factors)
            
        except Exception as e:
            self.logger.error(f"Error calculating opportunity risk: {e}")
            return 0.5
    
    def _estimate_expected_profit(self, market_data: Dict[str, Any], consensus_signal) -> float:
        """Estimate expected profit for an opportunity"""
        try:
            # Base profit from consensus strength
            base_profit = consensus_signal.consensus_strength * 0.1  # 10% max
            
            # Adjust for market conditions
            market_multiplier = market_data.get('market_multiplier', 1.0)
            
            # Adjust for volatility (higher volatility = higher potential profit)
            volatility = market_data.get('volatility', 0.5)
            volatility_multiplier = 1.0 + volatility * 0.5
            
            expected_profit = base_profit * market_multiplier * volatility_multiplier
            
            return min(expected_profit, 0.5)  # Cap at 50%
            
        except Exception as e:
            self.logger.error(f"Error estimating expected profit: {e}")
            return 0.05
    
    def _calculate_time_sensitivity(self, market_data: Dict[str, Any]) -> float:
        """Calculate time sensitivity of an opportunity"""
        try:
            # Factors that increase time sensitivity
            factors = []
            
            # Volume spike
            volume_spike = market_data.get('volume_spike', 0.0)
            factors.append(volume_spike * 0.3)
            
            # Price momentum
            price_momentum = market_data.get('price_momentum', 0.0)
            factors.append(abs(price_momentum) * 0.3)
            
            # Social sentiment velocity
            sentiment_velocity = market_data.get('sentiment_velocity', 0.0)
            factors.append(abs(sentiment_velocity) * 0.2)
            
            # Market attention
            market_attention = market_data.get('market_attention', 0.0)
            factors.append(market_attention * 0.2)
            
            return min(1.0, sum(factors))
            
        except Exception as e:
            self.logger.error(f"Error calculating time sensitivity: {e}")
            return 0.5
    
    async def _get_swarm_consensus(self, token_address: str, consensus_signal) -> float:
        """Get swarm consensus level"""
        try:
            # Get individual wallet opinions
            wallet_opinions = []
            
            for wallet_id in self.wallet_manager.active_wallets:
                wallet = self.wallet_manager.wallets[wallet_id]
                
                # Calculate wallet-specific opinion based on genetics
                opinion = self._calculate_wallet_opinion(wallet, consensus_signal)
                wallet_opinions.append(opinion)
            
            if not wallet_opinions:
                return 0.0
            
            # Calculate swarm consensus as weighted average
            swarm_consensus = sum(wallet_opinions) / len(wallet_opinions)
            
            return swarm_consensus
            
        except Exception as e:
            self.logger.error(f"Error getting swarm consensus: {e}")
            return 0.0
    
    def _calculate_wallet_opinion(self, wallet, consensus_signal) -> float:
        """Calculate individual wallet opinion"""
        try:
            # Base opinion from consensus signal
            base_opinion = consensus_signal.consensus_strength
            
            # Adjust based on wallet genetics
            genetics = wallet.genetics
            
            # Signal trust adjustment
            signal_adjustment = (genetics.signal_trust - 0.5) * 0.2
            
            # Aggression adjustment
            aggression_adjustment = (genetics.aggression - 0.5) * 0.1
            
            # Memory strength adjustment (learn from past)
            memory_adjustment = (genetics.memory_strength - 0.5) * 0.05
            
            # Calculate final opinion
            final_opinion = base_opinion + signal_adjustment + aggression_adjustment + memory_adjustment
            
            return max(0.0, min(1.0, final_opinion))
            
        except Exception as e:
            self.logger.error(f"Error calculating wallet opinion: {e}")
            return 0.5
    
    async def _assign_wallets_to_opportunity(self, opportunity_type: OpportunityType, 
                                           confidence_score: float, risk_level: float) -> Dict[str, float]:
        """Assign wallets to an opportunity"""
        try:
            wallet_assignments = {}
            
            # Get available wallets
            available_wallets = self.wallet_manager.active_wallets.copy()
            
            if not available_wallets:
                return wallet_assignments
            
            # Determine number of wallets to assign based on opportunity type
            if opportunity_type == OpportunityType.HIGH_CONFIDENCE:
                num_wallets = min(5, len(available_wallets))
                allocation_per_wallet = 0.2  # 20% each
            elif opportunity_type == OpportunityType.MEDIUM_CONFIDENCE:
                num_wallets = min(3, len(available_wallets))
                allocation_per_wallet = 0.15  # 15% each
            elif opportunity_type == OpportunityType.LOW_CONFIDENCE:
                num_wallets = min(2, len(available_wallets))
                allocation_per_wallet = 0.1  # 10% each
            else:  # EXPERIMENTAL
                num_wallets = min(1, len(available_wallets))
                allocation_per_wallet = 0.05  # 5% each
            
            # Select best wallets for this opportunity
            selected_wallets = await self._select_best_wallets_for_opportunity(
                available_wallets, opportunity_type, confidence_score, risk_level, num_wallets
            )
            
            # Assign allocations
            for wallet_id in selected_wallets:
                wallet_assignments[wallet_id] = allocation_per_wallet
            
            return wallet_assignments
            
        except Exception as e:
            self.logger.error(f"Error assigning wallets: {e}")
            return {}
    
    async def _select_best_wallets_for_opportunity(self, available_wallets: List[str], 
                                                 opportunity_type: OpportunityType,
                                                 confidence_score: float, risk_level: float,
                                                 num_wallets: int) -> List[str]:
        """Select best wallets for an opportunity"""
        try:
            wallet_scores = []
            
            for wallet_id in available_wallets:
                wallet = self.wallet_manager.wallets[wallet_id]
                
                # Calculate fitness score
                score = self._calculate_wallet_fitness_for_opportunity(
                    wallet, opportunity_type, confidence_score, risk_level
                )
                
                wallet_scores.append((wallet_id, score))
            
            # Sort by score and select top wallets
            wallet_scores.sort(key=lambda x: x[1], reverse=True)
            selected_wallets = [wallet_id for wallet_id, _ in wallet_scores[:num_wallets]]
            
            return selected_wallets
            
        except Exception as e:
            self.logger.error(f"Error selecting wallets: {e}")
            return available_wallets[:num_wallets] if available_wallets else []
    
    def _calculate_wallet_fitness_for_opportunity(self, wallet, opportunity_type: OpportunityType,
                                                confidence_score: float, risk_level: float) -> float:
        """Calculate wallet fitness for a specific opportunity"""
        try:
            genetics = wallet.genetics
            performance = wallet.performance
            
            # Base fitness from performance
            performance_fitness = performance.win_rate * 0.4 + performance.sharpe_ratio * 0.2
            
            # Genetics fitness
            genetics_fitness = 0.0
            
            if opportunity_type == OpportunityType.HIGH_CONFIDENCE:
                # High confidence opportunities prefer high signal trust and patience
                genetics_fitness = (genetics.signal_trust * 0.6 + genetics.patience * 0.4)
            elif opportunity_type == OpportunityType.MEDIUM_CONFIDENCE:
                # Medium confidence opportunities prefer balanced genetics
                genetics_fitness = (genetics.aggression * 0.4 + genetics.adaptation_rate * 0.6)
            elif opportunity_type == OpportunityType.LOW_CONFIDENCE:
                # Low confidence opportunities prefer high aggression and memory
                genetics_fitness = (genetics.aggression * 0.7 + genetics.memory_strength * 0.3)
            else:  # EXPERIMENTAL
                # Experimental opportunities prefer high adaptation and herd immunity
                genetics_fitness = (genetics.adaptation_rate * 0.5 + genetics.herd_immunity * 0.5)
            
            # Risk tolerance fitness
            risk_tolerance_fitness = 1.0 - abs(genetics.aggression - risk_level)
            
            # Final fitness score
            final_fitness = (
                performance_fitness * 0.4 +
                genetics_fitness * 0.4 +
                risk_tolerance_fitness * 0.2
            )
            
            return final_fitness
            
        except Exception as e:
            self.logger.error(f"Error calculating wallet fitness: {e}")
            return 0.5
    
    async def _opportunity_monitoring_loop(self):
        """Monitor for new opportunities"""
        while self.initialized:
            try:
                await asyncio.sleep(1)  # Check every second
                
                # Process opportunities in queue
                while not self.opportunity_queue.empty():
                    opportunity = await self.opportunity_queue.get()
                    await self._process_opportunity(opportunity)
                
            except Exception as e:
                self.logger.error(f"Opportunity monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _process_opportunity(self, opportunity: SwarmOpportunity):
        """Process a swarm opportunity"""
        try:
            # Create swarm decision
            decision = await self._make_swarm_decision(opportunity)
            
            # Execute decision
            if decision.action == "execute":
                await self._execute_opportunity(decision)
            elif decision.action == "stake":
                await self._stake_opportunity(decision)
            elif decision.action == "pass":
                self.logger.info(f"🚫 Passed opportunity: {opportunity.token_address}")
            
            # Record decision
            self.decision_history.append(decision)
            
            # Keep only last 100 decisions
            if len(self.decision_history) > 100:
                self.decision_history = self.decision_history[-100:]
            
        except Exception as e:
            self.logger.error(f"Error processing opportunity: {e}")
    
    async def _make_swarm_decision(self, opportunity: SwarmOpportunity) -> SwarmDecision:
        """Make a swarm decision for an opportunity"""
        try:
            decision_id = f"decision_{opportunity.token_address}_{int(datetime.now().timestamp())}"
            
            # Check consensus threshold
            if opportunity.swarm_consensus < self.config['min_consensus_threshold']:
                action = "pass"
                reasoning = ["Insufficient swarm consensus"]
            elif opportunity.risk_level > self.config['max_risk_per_opportunity']:
                action = "pass"
                reasoning = ["Risk level too high"]
            elif opportunity.time_sensitivity > 0.8:
                action = "execute"
                reasoning = ["High time sensitivity - immediate execution"]
            elif opportunity.confidence_score > 0.8:
                action = "execute"
                reasoning = ["High confidence opportunity"]
            elif opportunity.confidence_score > 0.6:
                action = "stake"
                reasoning = ["Medium confidence - staking position"]
            else:
                action = "pass"
                reasoning = ["Low confidence - passing"]
            
            # Create execution plan
            execution_plan = await self._create_execution_plan(opportunity, action)
            
            # Risk assessment
            risk_assessment = {
                'overall_risk': opportunity.risk_level,
                'consensus_risk': 1.0 - opportunity.swarm_consensus,
                'market_risk': opportunity.metadata.get('market_data', {}).get('volatility', 0.5),
                'execution_risk': 0.1  # Base execution risk
            }
            
            decision = SwarmDecision(
                decision_id=decision_id,
                opportunity=opportunity,
                action=action,
                execution_plan=execution_plan,
                risk_assessment=risk_assessment,
                consensus_level=opportunity.swarm_consensus,
                timestamp=datetime.now(),
                reasoning=reasoning
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error making swarm decision: {e}")
            # Return pass decision on error
            return SwarmDecision(
                decision_id=f"error_decision_{int(datetime.now().timestamp())}",
                opportunity=opportunity,
                action="pass",
                execution_plan={},
                risk_assessment={'overall_risk': 1.0},
                consensus_level=0.0,
                timestamp=datetime.now(),
                reasoning=[f"Error in decision making: {str(e)}"]
            )
    
    async def _create_execution_plan(self, opportunity: SwarmOpportunity, action: str) -> Dict[str, Any]:
        """Create execution plan for an opportunity"""
        try:
            if action == "execute":
                return {
                    'execution_type': 'immediate',
                    'wallet_assignments': opportunity.wallet_assignments,
                    'position_sizing': 'aggressive',
                    'stealth_level': 'normal',
                    'timeout_seconds': 30
                }
            elif action == "stake":
                return {
                    'execution_type': 'stake',
                    'wallet_assignments': {k: v * 0.5 for k, v in opportunity.wallet_assignments.items()},
                    'position_sizing': 'conservative',
                    'stealth_level': 'high',
                    'timeout_seconds': 60
                }
            else:
                return {
                    'execution_type': 'none',
                    'wallet_assignments': {},
                    'position_sizing': 'none',
                    'stealth_level': 'none',
                    'timeout_seconds': 0
                }
                
        except Exception as e:
            self.logger.error(f"Error creating execution plan: {e}")
            return {}
    
    async def _execute_opportunity(self, decision: SwarmDecision):
        """Execute an opportunity"""
        try:
            self.logger.info(f"🚀 Executing opportunity: {decision.opportunity.token_address}")
            
            # Coordinate wallet execution
            execution_results = []
            
            for wallet_id, allocation in decision.execution_plan['wallet_assignments'].items():
                # Apply stealth operations
                stealth_params = await self.stealth_operations.prepare_stealth_transaction(
                    wallet_id, {
                        'token_address': decision.opportunity.token_address,
                        'allocation': allocation,
                        'execution_type': decision.execution_plan['execution_type']
                    }
                )
                
                # Execute trade
                result = await self.trading_engine.execute_trade(
                    decision.opportunity.token_address,
                    {
                        'wallet_id': wallet_id,
                        'allocation': allocation,
                        'stealth_params': stealth_params,
                        'timeout': decision.execution_plan['timeout_seconds']
                    }
                )
                
                execution_results.append(result)
            
            # Update metrics
            self.swarm_metrics['executed_opportunities'] += 1
            
            # Check for success
            successful_executions = sum(1 for r in execution_results if r.get('success', False))
            if successful_executions > 0:
                self.swarm_metrics['successful_executions'] += 1
            
            self.logger.info(f"✅ Opportunity execution complete: {successful_executions}/{len(execution_results)} successful")
            
        except Exception as e:
            self.logger.error(f"Error executing opportunity: {e}")
    
    async def _stake_opportunity(self, decision: SwarmDecision):
        """Stake an opportunity (small position to monitor)"""
        try:
            self.logger.info(f"📊 Staking opportunity: {decision.opportunity.token_address}")
            
            # Similar to execute but with smaller positions and longer timeout
            await self._execute_opportunity(decision)
            
        except Exception as e:
            self.logger.error(f"Error staking opportunity: {e}")
    
    async def _decision_making_loop(self):
        """Main decision making loop"""
        while self.initialized:
            try:
                await asyncio.sleep(0.1)  # High frequency decision making
                
                # Update swarm state based on performance
                await self._update_swarm_state()
                
            except Exception as e:
                self.logger.error(f"Decision making error: {e}")
                await asyncio.sleep(1)
    
    async def _update_swarm_state(self):
        """Update swarm state based on performance and conditions"""
        try:
            # Calculate swarm health
            recent_decisions = [d for d in self.decision_history[-10:] if d.action == "execute"]
            
            if recent_decisions:
                success_rate = sum(1 for d in recent_decisions if d.opportunity.expected_profit > 0) / len(recent_decisions)
                self.swarm_health = success_rate
            
            # Update swarm state based on health
            if self.swarm_health > 0.8:
                if self.current_state != SwarmState.FEASTING:
                    self.current_state = SwarmState.FEASTING
                    self.logger.info("🦁 Swarm state: FEASTING")
            elif self.swarm_health > 0.6:
                if self.current_state != SwarmState.HUNTING:
                    self.current_state = SwarmState.HUNTING
                    self.logger.info("🐺 Swarm state: HUNTING")
            elif self.swarm_health > 0.4:
                if self.current_state != SwarmState.STALKING:
                    self.current_state = SwarmState.STALKING
                    self.logger.info("🦊 Swarm state: STALKING")
            else:
                if self.current_state != SwarmState.RETREATING:
                    self.current_state = SwarmState.RETREATING
                    self.logger.info("🦘 Swarm state: RETREATING")
            
        except Exception as e:
            self.logger.error(f"Error updating swarm state: {e}")
    
    async def _performance_tracking_loop(self):
        """Track swarm performance"""
        while self.initialized:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Calculate swarm efficiency
                if self.swarm_metrics['executed_opportunities'] > 0:
                    self.swarm_metrics['swarm_efficiency'] = (
                        self.swarm_metrics['successful_executions'] / 
                        self.swarm_metrics['executed_opportunities']
                    )
                
                # Calculate average consensus level
                if self.decision_history:
                    consensus_levels = [d.consensus_level for d in self.decision_history[-20:]]
                    self.swarm_metrics['avg_consensus_level'] = sum(consensus_levels) / len(consensus_levels)
                
                # Log performance summary
                await self._log_performance_summary()
                
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(60)
    
    async def _log_performance_summary(self):
        """Log swarm performance summary"""
        try:
            self.logger.info("📊 Swarm Performance Summary:")
            self.logger.info(f"   Opportunities: {self.swarm_metrics['total_opportunities']}")
            self.logger.info(f"   Executed: {self.swarm_metrics['executed_opportunities']}")
            self.logger.info(f"   Success Rate: {self.swarm_metrics['swarm_efficiency']:.2%}")
            self.logger.info(f"   Avg Consensus: {self.swarm_metrics['avg_consensus_level']:.2f}")
            self.logger.info(f"   Swarm Health: {self.swarm_health:.2f}")
            self.logger.info(f"   State: {self.current_state.value}")
            
        except Exception as e:
            self.logger.error(f"Error logging performance summary: {e}")
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status"""
        try:
            return {
                'current_state': self.current_state.value,
                'swarm_health': self.swarm_health,
                'metrics': self.swarm_metrics,
                'active_wallets': len(self.wallet_manager.active_wallets) if self.wallet_manager else 0,
                'opportunity_queue_size': self.opportunity_queue.qsize(),
                'recent_decisions': len(self.decision_history[-10:]),
                'initialized': self.initialized
            }
            
        except Exception as e:
            self.logger.error(f"Error getting swarm status: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the swarm decision engine"""
        try:
            self.logger.info("🛑 Shutting down swarm decision engine...")
            
            self.initialized = False
            
            # Shutdown components
            if self.trading_engine:
                await self.trading_engine.shutdown()
            if self.rug_detector:
                await self.rug_detector.shutdown()
            if self.kill_switch:
                await self.kill_switch.shutdown()
            
            self.logger.info("✅ Swarm decision engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during swarm shutdown: {e}")

# Global instance
_swarm_decision_engine = None

async def get_swarm_decision_engine() -> SwarmDecisionEngine:
    """Get global swarm decision engine instance"""
    global _swarm_decision_engine
    if _swarm_decision_engine is None:
        _swarm_decision_engine = SwarmDecisionEngine()
        await _swarm_decision_engine.initialize_swarm()
    return _swarm_decision_engine 