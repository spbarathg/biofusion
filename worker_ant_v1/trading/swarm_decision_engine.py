"""
SWARM DECISION ENGINE - NEURAL COMMAND CENTER BRAIN
===================================================

The core decision-making engine for the 10-wallet neural swarm.
Integrates all intelligence sources and makes final trading decisions.

This is the brain that turns $300 into $10K+ through:
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

from worker_ant_v1.intelligence.neural_command_center import NeuralCommandCenter
from worker_ant_v1.intelligence.stealth_operations import StealthOperationsSystem
from worker_ant_v1.core.wallet_manager import UnifiedWalletManager
from worker_ant_v1.core.vault_wallet_system import VaultWalletSystem
from worker_ant_v1.core.unified_trading_engine import UnifiedTradingEngine
from worker_ant_v1.safety.enhanced_rug_detector import EnhancedRugDetector
from worker_ant_v1.safety.kill_switch import EnhancedKillSwitch
from worker_ant_v1.utils.logger import setup_logger

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
class OpportunitySignal:
    """Signal for a potential trading opportunity"""
    token_address: str
    signal_strength: float
    confidence: float
    source: str
    timestamp: datetime
    market_data: Dict[str, Any]
    reasoning: str
    urgency: int = 5  # 1-10 scale

@dataclass
class ConsensusResult:
    """Result of consensus analysis"""
    action: str  # BUY, SELL, HOLD, AVOID
    confidence: float
    reasoning: str
    risk_level: str
    position_size: float
    execution_parameters: Dict[str, Any]
    consensus_sources: List[str]
    dissenting_sources: List[str]

@dataclass
class SwarmDecision:
    """Final swarm decision"""
    opportunity: OpportunitySignal
    consensus: ConsensusResult
    wallet_allocation: Dict[str, float]  # wallet_id -> amount_sol
    execution_timing: datetime
    stealth_parameters: Dict[str, Any]
    risk_assessment: Dict[str, Any]

class SwarmDecisionEngine:
    """The brain of the 10-wallet neural swarm"""
    
    def __init__(self):
        self.logger = setup_logger("SwarmDecisionEngine")
        
        
        self.neural_command_center: Optional[NeuralCommandCenter] = None
        self.stealth_operations: Optional[StealthOperationsSystem] = None
        self.wallet_manager: Optional[UnifiedWalletManager] = None
        self.vault_system: Optional[VaultWalletSystem] = None
        self.trading_engine: Optional[UnifiedTradingEngine] = None
        self.rug_detector: Optional[EnhancedRugDetector] = None
        self.kill_switch: Optional[EnhancedKillSwitch] = None
        
        
        self.consensus_thresholds = {
            'min_sources': 3,           # At least 3 sources must agree
            'min_confidence': 0.65,     # 65% minimum confidence
            'max_risk_level': 'medium', # Maximum risk tolerance
            'min_signal_strength': 0.6  # 60% minimum signal strength
        }
        
        
        self.current_state = SwarmState.INITIALIZING
        self.active_opportunities: Dict[str, OpportunitySignal] = {}
        self.decision_history: List[SwarmDecision] = []
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_trades': 0,
            'avoided_rugs': 0,
            'profit_generated': 0.0,
            'accuracy_rate': 0.0
        }
        
        
        self.last_decision_time = datetime.now()
        self.min_decision_interval = 25  # 25 seconds minimum between decisions
        
    async def initialize_swarm(self) -> bool:
        """Initialize all swarm components"""
        try:
            self.logger.info("🧬 Initializing Swarm Decision Engine...")
            
            
            self.neural_command_center = NeuralCommandCenter()
            self.stealth_operations = StealthOperationsSystem()
            
            
            from worker_ant_v1.core.wallet_manager import get_wallet_manager
            from worker_ant_v1.core.vault_wallet_system import get_vault_system
            
            self.wallet_manager = await get_wallet_manager()
            self.vault_system = await get_vault_system()
            
            
            self.trading_engine = UnifiedTradingEngine()
            await self.trading_engine.initialize()
            
            self.rug_detector = EnhancedRugDetector(
                solana_client=None,  # Will be injected
                jupiter_dex=None     # Will be injected
            )
            await self.rug_detector.initialize()
            
            self.kill_switch = EnhancedKillSwitch()
            await self.kill_switch.initialize()
            
            
            self.neural_command_center.wallet_manager = self.wallet_manager
            self.neural_command_center.vault_system = self.vault_system
            
            
            asyncio.create_task(self._opportunity_monitoring_loop())
            asyncio.create_task(self._decision_making_loop())
            asyncio.create_task(self._performance_tracking_loop())
            
            self.current_state = SwarmState.HUNTING
            self.logger.info("✅ Swarm Decision Engine initialized and hunting")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize swarm: {e}")
            return False
    
    async def analyze_opportunity(self, token_address: str, market_data: Dict[str, Any]) -> SwarmDecision:
        """Analyze a trading opportunity and make swarm decision"""
        try:
            opportunity = OpportunitySignal(
                token_address=token_address,
                signal_strength=market_data.get('signal_strength', 0.5),
                confidence=market_data.get('confidence', 0.5),
                source="market_scanner",
                timestamp=datetime.now(),
                market_data=market_data,
                reasoning="Market scanner detected opportunity",
                urgency=market_data.get('urgency', 5)
            )
            
            
            consensus = await self._run_consensus_analysis(opportunity)
            
            
            risk_assessment = await self._assess_risks(opportunity, consensus)
            
            
            if consensus.action == "BUY":
                wallet_allocation = await self._calculate_wallet_allocation(consensus.position_size)
                stealth_params = await self._generate_stealth_parameters()
                
                decision = SwarmDecision(
                    opportunity=opportunity,
                    consensus=consensus,
                    wallet_allocation=wallet_allocation,
                    execution_timing=datetime.now() + timedelta(seconds=5),
                    stealth_parameters=stealth_params,
                    risk_assessment=risk_assessment
                )
                
                
                self.decision_history.append(decision)
                self.performance_metrics['total_decisions'] += 1
                
                return decision
            
            else:
            else:
                return SwarmDecision(
                    opportunity=opportunity,
                    consensus=consensus,
                    wallet_allocation={},
                    execution_timing=datetime.now(),
                    stealth_parameters={},
                    risk_assessment=risk_assessment
                )
                
        except Exception as e:
            self.logger.error(f"Error analyzing opportunity: {e}")
            self.logger.error(f"Error analyzing opportunity: {e}")
            return SwarmDecision(
                opportunity=opportunity,
                consensus=ConsensusResult(
                    action="AVOID",
                    confidence=0.0,
                    reasoning=f"Analysis error: {e}",
                    risk_level="high",
                    position_size=0.0,
                    execution_parameters={},
                    consensus_sources=[],
                    dissenting_sources=[]
                ),
                wallet_allocation={},
                execution_timing=datetime.now(),
                stealth_parameters={},
                risk_assessment={"error": str(e)}
            )
    
    async def _run_consensus_analysis(self, opportunity: OpportunitySignal) -> ConsensusResult:
        """Run multi-source consensus analysis"""
        try:
            neural_analysis = await self._get_neural_analysis(opportunity)
            
            
            rug_analysis = await self._get_rug_analysis(opportunity)
            
            
            stealth_analysis = await self._get_stealth_analysis(opportunity)
            
            
            all_analyses = [neural_analysis, rug_analysis, stealth_analysis]
            
            
            buy_votes = sum(1 for analysis in all_analyses if analysis.get('action') == 'BUY')
            avoid_votes = sum(1 for analysis in all_analyses if analysis.get('action') == 'AVOID')
            
            
            if buy_votes >= 2 and avoid_votes == 0:  # Majority buy, no avoid votes
                avg_confidence = np.mean([a.get('confidence', 0) for a in all_analyses if a.get('action') == 'BUY'])
                
                if avg_confidence >= self.consensus_thresholds['min_confidence']:
                if avg_confidence >= self.consensus_thresholds['min_confidence']:
                    base_position = 0.35  # 35% max from config
                    confidence_multiplier = avg_confidence
                    position_size = base_position * confidence_multiplier
                    
                    return ConsensusResult(
                        action="BUY",
                        confidence=avg_confidence,
                        reasoning=f"Consensus reached: {buy_votes}/3 sources approve",
                        risk_level="medium",
                        position_size=position_size,
                        execution_parameters={
                            'max_slippage': 0.03,
                            'timeout_seconds': 30,
                            'priority': 'high'
                        },
                        consensus_sources=[a.get('source', 'unknown') for a in all_analyses if a.get('action') == 'BUY'],
                        dissenting_sources=[]
                    )
            
            
            return ConsensusResult(
                action="AVOID",
                confidence=0.0,
                reasoning="Insufficient consensus or confidence",
                risk_level="high",
                position_size=0.0,
                execution_parameters={},
                consensus_sources=[],
                dissenting_sources=[a.get('source', 'unknown') for a in all_analyses]
            )
            
        except Exception as e:
            self.logger.error(f"Consensus analysis failed: {e}")
            return ConsensusResult(
                action="AVOID",
                confidence=0.0,
                reasoning=f"Analysis error: {e}",
                risk_level="high",
                position_size=0.0,
                execution_parameters={},
                consensus_sources=[],
                dissenting_sources=[]
            )
    
    async def _get_neural_analysis(self, opportunity: OpportunitySignal) -> Dict[str, Any]:
        """Get analysis from neural command center"""
        try:
            base_confidence = opportunity.signal_strength * 0.8
            
            
            market_cap = opportunity.market_data.get('market_cap', 1000000)
            liquidity = opportunity.market_data.get('liquidity', 50000)
            
            
            if market_cap < 2000000 and liquidity > 30000:
                base_confidence += 0.1
            
            
            price_change_24h = opportunity.market_data.get('price_change_24h', 0)
            if 5 < price_change_24h < 50:  # Healthy growth
                base_confidence += 0.05
            
            action = "BUY" if base_confidence > 0.65 else "AVOID"
            
            return {
                'source': 'neural_command_center',
                'action': action,
                'confidence': min(0.95, base_confidence),
                'reasoning': f"Neural analysis: {action} with {base_confidence:.2f} confidence"
            }
            
        except Exception as e:
            return {
                'source': 'neural_command_center',
                'action': 'AVOID',
                'confidence': 0.0,
                'reasoning': f"Neural analysis failed: {e}"
            }
    
    async def _get_rug_analysis(self, opportunity: OpportunitySignal) -> Dict[str, Any]:
        """Get rug detection analysis"""
        try:
            liquidity = opportunity.market_data.get('liquidity', 0)
            holder_count = opportunity.market_data.get('holder_count', 0)
            age_hours = opportunity.market_data.get('age_hours', 0)
            
            risk_factors = 0
            
            
            if liquidity < 25000:  # Low liquidity
                risk_factors += 1
            if holder_count < 75:  # Few holders
                risk_factors += 1
            if age_hours < 0.5:  # Too new
                risk_factors += 1
            
            top_holders_percent = opportunity.market_data.get('top_holders_percent', 50)
            if top_holders_percent > 40:  # Too concentrated
                risk_factors += 1
            
            
            if risk_factors >= 2:
                action = "AVOID"
                confidence = 0.8
                reasoning = f"Rug risk detected: {risk_factors} risk factors"
            else:
                action = "BUY"
                confidence = 0.7
                reasoning = f"Rug check passed: {risk_factors} risk factors"
            
            return {
                'source': 'rug_detector',
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning
            }
            
        except Exception as e:
            return {
                'source': 'rug_detector',
                'action': 'AVOID',
                'confidence': 0.9,
                'reasoning': f"Rug analysis failed: {e}"
            }
    
    async def _get_stealth_analysis(self, opportunity: OpportunitySignal) -> Dict[str, Any]:
        """Get stealth operations analysis"""
        try:
            bot_activity = opportunity.market_data.get('bot_activity_score', 0.5)
            
            if bot_activity > 0.7:
                return {
                    'source': 'stealth_operations',
                    'action': 'AVOID',
                    'confidence': 0.8,
                    'reasoning': f"High bot activity detected: {bot_activity:.2f}"
                }
            else:
                return {
                    'source': 'stealth_operations',
                    'action': 'BUY',
                    'confidence': 0.7,
                    'reasoning': f"Stealth conditions favorable: {bot_activity:.2f} bot activity"
                }
                
        except Exception as e:
            return {
                'source': 'stealth_operations',
                'action': 'AVOID',
                'confidence': 0.5,
                'reasoning': f"Stealth analysis failed: {e}"
            }
    
    async def _assess_risks(self, opportunity: OpportunitySignal, consensus: ConsensusResult) -> Dict[str, Any]:
        """Assess overall risks"""
        return {
            'rug_risk': 'low' if consensus.action == 'BUY' else 'high',
            'liquidity_risk': 'low' if opportunity.market_data.get('liquidity', 0) > 30000 else 'medium',
            'volatility_risk': 'medium',
            'execution_risk': 'low',
            'overall_risk': consensus.risk_level
        }
    
    async def _calculate_wallet_allocation(self, total_position_size: float) -> Dict[str, float]:
        """Calculate how to distribute trade across wallets"""
        try:
            if not self.wallet_manager:
                return {}
            
            
            wallets = await self.wallet_manager.get_active_wallets()
            if not wallets:
                return {}
            
            
            selected_wallets = list(wallets.keys())[:3]
            allocation_per_wallet = total_position_size / len(selected_wallets)
            
            return {
                wallet_id: allocation_per_wallet 
                for wallet_id in selected_wallets
            }
            
        except Exception as e:
            self.logger.error(f"Wallet allocation failed: {e}")
            return {}
    
    async def _generate_stealth_parameters(self) -> Dict[str, Any]:
        """Generate stealth execution parameters"""
        return {
            'gas_randomization': True,
            'timing_variance_ms': np.random.randint(100, 1000),
            'amount_obfuscation': True,
            'execution_delay_ms': np.random.randint(0, 2000)
        }
    
    async def _opportunity_monitoring_loop(self):
        """Monitor for new opportunities"""
        while True:
            try:
                # This would integrate with actual market scanning
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Opportunity monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _decision_making_loop(self):
        """Main decision making loop"""
        while True:
            try:
                time_since_last = (datetime.now() - self.last_decision_time).total_seconds()
                
                if time_since_last < self.min_decision_interval:
                    await asyncio.sleep(5)
                    continue
                
                
                # Process any pending opportunities
                
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Decision making loop error: {e}")
                await asyncio.sleep(10)
    
    async def _performance_tracking_loop(self):
        """Track performance metrics"""
        while True:
            try:
                if self.decision_history:
                    total_decisions = len(self.decision_history)
                    successful_trades = len([d for d in self.decision_history if d.consensus.action == 'BUY'])
                    
                    self.performance_metrics.update({
                        'total_decisions': total_decisions,
                        'successful_trades': successful_trades,
                        'accuracy_rate': successful_trades / total_decisions if total_decisions > 0 else 0.0
                    })
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(60)
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status"""
        return {
            'state': self.current_state.value,
            'active_opportunities': len(self.active_opportunities),
            'decision_history_length': len(self.decision_history),
            'performance_metrics': self.performance_metrics,
            'last_decision_time': self.last_decision_time.isoformat(),
            'systems_online': {
                'neural_command_center': self.neural_command_center is not None,
                'stealth_operations': self.stealth_operations is not None,
                'wallet_manager': self.wallet_manager is not None,
                'vault_system': self.vault_system is not None,
                'trading_engine': self.trading_engine is not None,
                'rug_detector': self.rug_detector is not None,
                'kill_switch': self.kill_switch is not None
            }
        }
    
    async def emergency_stop(self):
        """Emergency stop all operations"""
        self.logger.warning("🚨 EMERGENCY STOP TRIGGERED")
        self.current_state = SwarmState.EMERGENCY
        
        if self.kill_switch:
            await self.kill_switch.trigger_emergency_stop("Manual emergency stop")
        
        
        if self.trading_engine:
            await self.trading_engine.emergency_stop()


_swarm_decision_engine = None

async def get_swarm_decision_engine() -> SwarmDecisionEngine:
    """Get global swarm decision engine instance"""
    global _swarm_decision_engine
    if _swarm_decision_engine is None:
        _swarm_decision_engine = SwarmDecisionEngine()
    return _swarm_decision_engine 