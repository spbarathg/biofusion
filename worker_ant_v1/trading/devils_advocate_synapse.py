"""
DEVIL'S ADVOCATE SYNAPSE - PRE-MORTEM ANALYSIS SYSTEM
====================================================

The Devil's Advocate Synapse conducts rapid pre-mortem analysis for every trade,
stress-testing against known failure patterns and issuing vetoes when necessary.

This system embodies the Colony's second-order thinking capability, asking:
"This trade failed catastrophically. What was the most probable cause?"
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from worker_ant_v1.utils.logger import setup_logger

class FailurePattern(Enum):
    """Known failure patterns for trade analysis"""
    RUG_PULL = "rug_pull"
    SLOW_RUG = "slow_rug"
    SANDWICH_ATTACK = "sandwich_attack"
    MEV_FRONT_RUN = "mev_front_run"
    LIQUIDITY_DRAIN = "liquidity_drain"
    WHALE_DUMP = "whale_dump"
    COORDINATED_SELL = "coordinated_sell"
    FAKE_VOLUME = "fake_volume"
    PUMP_AND_DUMP = "pump_and_dump"
    HONEYPOT = "honeypot"
    CONTRACT_EXPLOIT = "contract_exploit"
    IMPERMANENT_LOSS = "impermanent_loss"


class VetoReason(Enum):
    """Reasons for trade veto"""
    HIGH_FAILURE_PROBABILITY = "high_failure_probability"
    INSUFFICIENT_LIQUIDITY = "insufficient_liquidity"
    SUSPICIOUS_WHALE_ACTIVITY = "suspicious_whale_activity"
    RAPID_PRICE_MANIPULATION = "rapid_price_manipulation"
    CONTRACT_SECURITY_RISK = "contract_security_risk"
    HISTORICAL_PATTERN_MATCH = "historical_pattern_match"
    SENTIMENT_CONTRADICTION = "sentiment_contradiction"
    TECHNICAL_RED_FLAGS = "technical_red_flags"


@dataclass
class FailureScenario:
    """Individual failure scenario analysis"""
    pattern: FailurePattern
    probability: float  # 0.0 to 1.0
    impact_severity: float  # 0.0 to 1.0 (potential loss)
    risk_score: float  # Combined probability * severity
    evidence: List[str]
    mitigation_possible: bool
    confidence: float


@dataclass
class PreMortemAnalysis:
    """Complete pre-mortem analysis result"""
    trade_id: str
    token_address: str
    analyzed_at: datetime
    
    # Analysis results
    failure_scenarios: List[FailureScenario]
    overall_failure_probability: float
    max_potential_loss: float
    confidence_score: float
    
    # Decision
    veto_recommended: bool
    veto_reasons: List[VetoReason]
    risk_mitigation_suggestions: List[str]
    
    # Analysis metadata
    analysis_duration_ms: int
    patterns_analyzed: int
    historical_matches: int


class DevilsAdvocateSynapse:
    """The Colony's Devil's Advocate - Pre-mortem Analysis Engine"""
    
    def __init__(self):
        self.logger = setup_logger("DevilsAdvocateSynapse")
        
        # WCCA (Worst Case Constraint Analysis) Configuration
        self.acceptable_rel_threshold = 0.1  # Acceptable Risk-Adjusted Expected Loss (e.g., 0.1 SOL)
        
        # Failure pattern database
        self.failure_patterns: Dict[FailurePattern, Dict[str, Any]] = {}
        self.historical_failures: List[Dict[str, Any]] = []
        self.shadow_memory: Dict[str, Any] = {}  # shadow_memory.json data
        
        # Analysis configuration
        self.veto_threshold = 0.75  # 75% failure probability triggers veto
        self.max_analysis_time_ms = 500  # Maximum 500ms for analysis
        self.min_liquidity_threshold = 100.0  # SOL
        self.whale_activity_threshold = 0.7
        
        # Performance tracking
        self.total_analyses = 0
        self.vetoes_issued = 0
        self.trades_saved = 0  # Successful vetoes that prevented losses
        
        # Initialize failure patterns
        self._initialize_failure_patterns()
    
    async def initialize(self, shadow_memory_path: str = "shadow_memory.json"):
        """Initialize the Devil's Advocate Synapse"""
        self.logger.info("🕵️ Initializing Devil's Advocate Synapse...")
        
        # Load shadow memory of known failures
        await self._load_shadow_memory(shadow_memory_path)
        
        # Start background pattern learning
        asyncio.create_task(self._pattern_learning_loop())
        
        self.logger.info("✅ Devil's Advocate Synapse active - Pre-mortem analysis armed")
    
    async def conduct_pre_mortem_analysis(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        WCCA Survival Filter - Risk-Adjusted Expected Loss Analysis
        
        Implements non-negotiable veto system using R-EL calculation:
        R_EL = P(Loss) * |Position_Size|
        
        Returns:
            {"veto": True/False, "reason": str (if veto)}
        """
        start_time = datetime.now()
        
        try:
            # Extract trade information
            token_address = trade_params.get('token_address', '')
            position_size_sol = abs(float(trade_params.get('amount', 0.0)))
            
            self.logger.debug(f"🔍 WCCA analyzing R-EL for {token_address[:8]} | Position: {position_size_sol:.4f} SOL")
            
            # Calculate Risk-Adjusted Expected Loss for catastrophic failure patterns
            catastrophic_patterns = [FailurePattern.RUG_PULL, FailurePattern.HONEYPOT]
            max_rel = 0.0
            worst_pattern = None
            
            for pattern in catastrophic_patterns:
                # Get failure probability from specialized modules
                if pattern == FailurePattern.RUG_PULL:
                    failure_probability = await self._get_rug_pull_probability(trade_params)
                elif pattern == FailurePattern.HONEYPOT:
                    failure_probability = await self._get_honeypot_probability(trade_params)
                else:
                    failure_probability = 0.0
                
                # Calculate R-EL: P(Loss) * |Position_Size|
                rel = failure_probability * position_size_sol
                
                if rel > max_rel:
                    max_rel = rel
                    worst_pattern = pattern
                
                self.logger.debug(f"📊 {pattern.value} R-EL: {rel:.4f} SOL (P={failure_probability:.3f})")
            
            # VETO DECISION: If any R-EL exceeds threshold
            if max_rel > self.acceptable_rel_threshold:
                veto_reason = f"{worst_pattern.value.upper()} R-EL of {max_rel:.4f} SOL exceeds threshold of {self.acceptable_rel_threshold} SOL"
                
                self.logger.warning(f"🚫 WCCA VETO | {token_address[:8]} | {veto_reason}")
                
                return {
                    "veto": True,
                    "reason": veto_reason,
                    "rel_calculated": max_rel,
                    "threshold": self.acceptable_rel_threshold,
                    "worst_pattern": worst_pattern.value
                }
            
            # CLEAR DECISION: All R-EL within acceptable limits
            analysis_duration = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(f"✅ WCCA CLEAR | {token_address[:8]} | Max R-EL: {max_rel:.4f} SOL | Duration: {analysis_duration:.1f}ms")
            
            return {
                "veto": False,
                "max_rel": max_rel,
                "analysis_duration_ms": analysis_duration
            }
            
        except Exception as e:
            self.logger.error(f"❌ WCCA analysis error: {e}")
            # FAIL-SAFE: Veto on analysis failure
            return {
                "veto": True,
                "reason": f"WCCA analysis system failure: {str(e)}"
                         }
    
    async def _get_rug_pull_probability(self, trade_params: Dict[str, Any]) -> float:
        """
        Calculate rug pull probability from specialized rug detection modules
        
        Args:
            trade_params: Trade parameters containing token info
            
        Returns:
            float: Probability of rug pull (0.0 to 1.0)
        """
        try:
            token_address = trade_params.get('token_address', '')
            
            # Initialize probability factors
            probability_factors = []
            
            # Check if token is in known rug database
            if token_address in self.shadow_memory.get('known_rugs', []):
                probability_factors.append(0.95)  # 95% if known rug
                
            # Token age factor
            token_age_hours = trade_params.get('token_age_hours', 24)
            if token_age_hours < 1:
                probability_factors.append(0.8)   # 80% for <1 hour old
            elif token_age_hours < 6:
                probability_factors.append(0.6)   # 60% for <6 hours old
            elif token_age_hours < 24:
                probability_factors.append(0.4)   # 40% for <24 hours old
            else:
                probability_factors.append(0.1)   # 10% baseline for older tokens
                
            # Liquidity concentration factor
            liquidity_concentration = trade_params.get('liquidity_concentration', 0.5)
            if liquidity_concentration > 0.8:
                probability_factors.append(0.7)   # 70% for high concentration
            elif liquidity_concentration > 0.6:
                probability_factors.append(0.5)   # 50% for medium concentration
                
            # Dev holdings factor
            dev_holdings_percent = trade_params.get('dev_holdings_percent', 0.0)
            if dev_holdings_percent > 50:
                probability_factors.append(0.8)   # 80% if dev holds >50%
            elif dev_holdings_percent > 20:
                probability_factors.append(0.6)   # 60% if dev holds >20%
                
            # Enhanced rug detector score (if available)
            rug_detector_score = trade_params.get('rug_detector_score', None)
            if rug_detector_score is not None:
                probability_factors.append(float(rug_detector_score))
            
            # Calculate final probability using geometric mean to avoid over-amplification
            if probability_factors:
                # Use max of individual factors rather than multiplication to avoid near-zero results
                final_probability = max(probability_factors)
            else:
                final_probability = 0.2  # Default 20% baseline risk
                
            return min(0.99, final_probability)  # Cap at 99%
            
        except Exception as e:
            self.logger.error(f"Error calculating rug pull probability: {e}")
            return 0.5  # Default to moderate risk on error
    
    async def _get_honeypot_probability(self, trade_params: Dict[str, Any]) -> float:
        """
        Calculate honeypot probability
        
        Args:
            trade_params: Trade parameters containing token info
            
        Returns:
            float: Probability of honeypot (0.0 to 1.0)
        """
        try:
            token_address = trade_params.get('token_address', '')
            
            # Initialize probability factors
            probability_factors = []
            
            # Check contract verification status
            is_verified = trade_params.get('contract_verified', True)
            if not is_verified:
                probability_factors.append(0.7)  # 70% if unverified
                
            # Check for suspicious contract patterns
            has_transfer_restrictions = trade_params.get('has_transfer_restrictions', False)
            if has_transfer_restrictions:
                probability_factors.append(0.8)  # 80% if transfer restrictions
                
            # Check sell/buy ratio anomalies
            sell_buy_ratio = trade_params.get('sell_buy_ratio', 1.0)
            if sell_buy_ratio < 0.1:  # Very few sells compared to buys
                probability_factors.append(0.9)  # 90% honeypot probability
            elif sell_buy_ratio < 0.3:
                probability_factors.append(0.6)  # 60% honeypot probability
                
            # Check for blacklist functionality
            has_blacklist = trade_params.get('has_blacklist_function', False)
            if has_blacklist:
                probability_factors.append(0.7)  # 70% if blacklist function exists
                
            # Calculate final probability
            if probability_factors:
                final_probability = max(probability_factors)
            else:
                final_probability = 0.1  # Default 10% baseline risk
                
            return min(0.99, final_probability)  # Cap at 99%
            
        except Exception as e:
            self.logger.error(f"Error calculating honeypot probability: {e}")
            return 0.3  # Default to moderate risk on error
    
    async def _analyze_failure_scenarios(self, trade_params: Dict[str, Any]) -> List[FailureScenario]:
        """Analyze all possible failure scenarios for the trade"""
        scenarios = []
        
        # Analyze each failure pattern
        for pattern, pattern_config in self.failure_patterns.items():
            scenario = await self._analyze_single_pattern(pattern, trade_params, pattern_config)
            if scenario.probability > 0.1:  # Only include scenarios with >10% probability
                scenarios.append(scenario)
        
        # Sort by risk score (highest first)
        scenarios.sort(key=lambda x: x.risk_score, reverse=True)
        
        return scenarios
    
    async def _analyze_single_pattern(self, pattern: FailurePattern, trade_params: Dict[str, Any], pattern_config: Dict[str, Any]) -> FailureScenario:
        """Analyze a single failure pattern"""
        try:
            token_address = trade_params.get('token_address', '')
            amount_sol = trade_params.get('amount', 0.0)
            
            # Initialize scenario
            scenario = FailureScenario(
                pattern=pattern,
                probability=0.0,
                impact_severity=0.0,
                risk_score=0.0,
                evidence=[],
                mitigation_possible=True,
                confidence=0.5
            )
            
            # Pattern-specific analysis
            if pattern == FailurePattern.RUG_PULL:
                scenario = await self._analyze_rug_pull_risk(trade_params, scenario)
            elif pattern == FailurePattern.SANDWICH_ATTACK:
                scenario = await self._analyze_sandwich_attack_risk(trade_params, scenario)
            elif pattern == FailurePattern.LIQUIDITY_DRAIN:
                scenario = await self._analyze_liquidity_drain_risk(trade_params, scenario)
            elif pattern == FailurePattern.WHALE_DUMP:
                scenario = await self._analyze_whale_dump_risk(trade_params, scenario)
            elif pattern == FailurePattern.MEV_FRONT_RUN:
                scenario = await self._analyze_mev_front_run_risk(trade_params, scenario)
            elif pattern == FailurePattern.HONEYPOT:
                scenario = await self._analyze_honeypot_risk(trade_params, scenario)
            # Add more pattern-specific analyses as needed
            
            # Calculate final risk score
            scenario.risk_score = scenario.probability * scenario.impact_severity
            
            return scenario
            
        except Exception as e:
            self.logger.error(f"Error analyzing pattern {pattern.value}: {e}")
            return FailureScenario(
                pattern=pattern,
                probability=0.5,  # Default to moderate risk if analysis fails
                impact_severity=0.8,
                risk_score=0.4,
                evidence=[f"Analysis error: {str(e)}"],
                mitigation_possible=False,
                confidence=0.0
            )
    
    async def _analyze_rug_pull_risk(self, trade_params: Dict[str, Any], scenario: FailureScenario) -> FailureScenario:
        """Analyze rug pull risk factors"""
        token_address = trade_params.get('token_address', '')
        evidence = []
        probability_factors = []
        
        # Check shadow memory for known rug patterns
        if token_address in self.shadow_memory.get('known_rugs', []):
            evidence.append("Token found in known rug database")
            probability_factors.append(0.9)
        
        # Check token age (newer tokens = higher rug risk)
        token_age_hours = trade_params.get('token_age_hours', 24)
        if token_age_hours < 1:
            evidence.append(f"Extremely new token ({token_age_hours:.1f} hours old)")
            probability_factors.append(0.8)
        elif token_age_hours < 6:
            evidence.append(f"Very new token ({token_age_hours:.1f} hours old)")
            probability_factors.append(0.6)
        elif token_age_hours < 24:
            evidence.append(f"New token ({token_age_hours:.1f} hours old)")
            probability_factors.append(0.4)
        
        # Check liquidity concentration
        liquidity_concentration = trade_params.get('liquidity_concentration', 0.5)
        if liquidity_concentration > 0.8:
            evidence.append(f"High liquidity concentration ({liquidity_concentration:.1%})")
            probability_factors.append(0.7)
        
        # Check dev wallet holdings
        dev_holdings_percent = trade_params.get('dev_holdings_percent', 0.0)
        if dev_holdings_percent > 50:
            evidence.append(f"Dev holds {dev_holdings_percent:.1f}% of supply")
            probability_factors.append(0.8)
        elif dev_holdings_percent > 20:
            evidence.append(f"Dev holds {dev_holdings_percent:.1f}% of supply")
            probability_factors.append(0.5)
        
        # Calculate probability
        if probability_factors:
            scenario.probability = min(1.0, max(probability_factors) + 0.1 * (len(probability_factors) - 1))
        else:
            scenario.probability = 0.1  # Base rug risk for all tokens
        
        scenario.impact_severity = 0.95  # Rug pulls typically result in 95%+ loss
        scenario.evidence = evidence
        scenario.confidence = 0.8 if evidence else 0.3
        scenario.mitigation_possible = False  # Rug pulls are hard to mitigate
        
        return scenario
    
    async def _analyze_sandwich_attack_risk(self, trade_params: Dict[str, Any], scenario: FailureScenario) -> FailureScenario:
        """Analyze sandwich attack risk factors"""
        evidence = []
        probability_factors = []
        
        # Check trade size (larger trades = higher sandwich risk)
        amount_sol = trade_params.get('amount', 0.0)
        if amount_sol > 50:
            evidence.append(f"Large trade size ({amount_sol:.1f} SOL)")
            probability_factors.append(0.8)
        elif amount_sol > 20:
            evidence.append(f"Medium trade size ({amount_sol:.1f} SOL)")
            probability_factors.append(0.5)
        elif amount_sol > 5:
            evidence.append(f"Notable trade size ({amount_sol:.1f} SOL)")
            probability_factors.append(0.3)
        
        # Check slippage tolerance
        max_slippage = trade_params.get('max_slippage', 2.0)
        if max_slippage > 5.0:
            evidence.append(f"High slippage tolerance ({max_slippage:.1f}%)")
            probability_factors.append(0.6)
        
        # Check current network congestion
        network_congestion = trade_params.get('network_congestion', 0.5)
        if network_congestion > 0.7:
            evidence.append(f"High network congestion ({network_congestion:.1%})")
            probability_factors.append(0.7)
        
        # Calculate probability
        scenario.probability = max(probability_factors) if probability_factors else 0.2
        scenario.impact_severity = min(0.15, max_slippage * 0.02)  # Sandwich attacks typically cause 5-15% loss
        scenario.evidence = evidence
        scenario.confidence = 0.7
        scenario.mitigation_possible = True  # Can be mitigated with better slippage/timing
        
        return scenario
    
    async def _analyze_liquidity_drain_risk(self, trade_params: Dict[str, Any], scenario: FailureScenario) -> FailureScenario:
        """Analyze liquidity drain risk factors"""
        evidence = []
        
        # Check total liquidity
        total_liquidity_sol = trade_params.get('total_liquidity_sol', 0.0)
        trade_amount = trade_params.get('amount', 0.0)
        
        if total_liquidity_sol < self.min_liquidity_threshold:
            evidence.append(f"Low total liquidity ({total_liquidity_sol:.1f} SOL)")
            scenario.probability = 0.8
        elif trade_amount > total_liquidity_sol * 0.1:
            evidence.append(f"Trade size is {(trade_amount/total_liquidity_sol):.1%} of total liquidity")
            scenario.probability = 0.6
        else:
            scenario.probability = 0.1
        
        scenario.impact_severity = min(0.5, trade_amount / max(total_liquidity_sol, 1) * 2)
        scenario.evidence = evidence
        scenario.confidence = 0.8
        scenario.mitigation_possible = True
        
        return scenario
    
    async def _analyze_whale_dump_risk(self, trade_params: Dict[str, Any], scenario: FailureScenario) -> FailureScenario:
        """Analyze whale dump risk factors"""
        evidence = []
        
        # Check for recent whale activity
        whale_activity_score = trade_params.get('whale_activity_score', 0.0)
        if whale_activity_score > self.whale_activity_threshold:
            evidence.append(f"High whale activity detected ({whale_activity_score:.1%})")
            scenario.probability = 0.7
        else:
            scenario.probability = 0.2
        
        scenario.impact_severity = 0.6  # Whale dumps can cause 30-60% price impact
        scenario.evidence = evidence
        scenario.confidence = 0.6
        scenario.mitigation_possible = True
        
        return scenario
    
    async def _analyze_mev_front_run_risk(self, trade_params: Dict[str, Any], scenario: FailureScenario) -> FailureScenario:
        """Analyze MEV front-running risk factors"""
        evidence = []
        
        # Check gas price and timing
        gas_price = trade_params.get('gas_price_gwei', 0.0)
        if gas_price < 50:  # Low gas price makes front-running easier
            evidence.append(f"Low gas price ({gas_price:.0f} gwei)")
            scenario.probability = 0.5
        else:
            scenario.probability = 0.2
        
        scenario.impact_severity = 0.1  # MEV typically causes 5-10% loss
        scenario.evidence = evidence
        scenario.confidence = 0.5
        scenario.mitigation_possible = True
        
        return scenario
    
    async def _analyze_honeypot_risk(self, trade_params: Dict[str, Any], scenario: FailureScenario) -> FailureScenario:
        """Analyze honeypot contract risk factors"""
        evidence = []
        
        # Check contract verification and age
        contract_verified = trade_params.get('contract_verified', False)
        if not contract_verified:
            evidence.append("Contract not verified")
            scenario.probability = 0.6
        else:
            scenario.probability = 0.1
        
        scenario.impact_severity = 0.9  # Honeypots typically result in 90%+ loss
        scenario.evidence = evidence
        scenario.confidence = 0.7
        scenario.mitigation_possible = False
        
        return scenario
    
    def _calculate_overall_failure_probability(self, scenarios: List[FailureScenario]) -> float:
        """Calculate overall failure probability from all scenarios"""
        if not scenarios:
            return 0.0
        
        # Use the maximum risk score as the overall probability
        # This represents the most likely failure scenario
        max_risk = max(scenario.risk_score for scenario in scenarios)
        
        # Apply confidence weighting
        weighted_risks = [scenario.risk_score * scenario.confidence for scenario in scenarios]
        avg_weighted_risk = sum(weighted_risks) / len(weighted_risks) if weighted_risks else 0.0
        
        # Combine max risk and average weighted risk
        return min(1.0, max_risk * 0.7 + avg_weighted_risk * 0.3)
    
    def _calculate_max_potential_loss(self, scenarios: List[FailureScenario], trade_amount: float) -> float:
        """Calculate maximum potential loss from failure scenarios"""
        if not scenarios:
            return 0.0
        
        max_impact = max(scenario.impact_severity for scenario in scenarios)
        return trade_amount * max_impact
    
    def _calculate_analysis_confidence(self, scenarios: List[FailureScenario]) -> float:
        """Calculate confidence in the analysis"""
        if not scenarios:
            return 0.0
        
        avg_confidence = sum(scenario.confidence for scenario in scenarios) / len(scenarios)
        return avg_confidence
    
    def _determine_veto_recommendation(self, scenarios: List[FailureScenario], overall_probability: float) -> Tuple[bool, List[VetoReason]]:
        """Determine if trade should be vetoed and why"""
        veto_reasons = []
        
        # Check overall failure probability
        if overall_probability >= self.veto_threshold:
            veto_reasons.append(VetoReason.HIGH_FAILURE_PROBABILITY)
        
        # Check specific high-impact scenarios
        for scenario in scenarios:
            if scenario.risk_score > 0.8:
                if scenario.pattern == FailurePattern.RUG_PULL:
                    veto_reasons.append(VetoReason.CONTRACT_SECURITY_RISK)
                elif scenario.pattern == FailurePattern.LIQUIDITY_DRAIN:
                    veto_reasons.append(VetoReason.INSUFFICIENT_LIQUIDITY)
                elif scenario.pattern == FailurePattern.WHALE_DUMP:
                    veto_reasons.append(VetoReason.SUSPICIOUS_WHALE_ACTIVITY)
        
        # Check for pattern matches in shadow memory
        if any(scenario.evidence and "known rug database" in str(scenario.evidence) for scenario in scenarios):
            veto_reasons.append(VetoReason.HISTORICAL_PATTERN_MATCH)
        
        return len(veto_reasons) > 0, veto_reasons
    
    def _generate_mitigation_suggestions(self, scenarios: List[FailureScenario]) -> List[str]:
        """Generate risk mitigation suggestions"""
        suggestions = []
        
        for scenario in scenarios:
            if scenario.mitigation_possible and scenario.risk_score > 0.3:
                if scenario.pattern == FailurePattern.SANDWICH_ATTACK:
                    suggestions.append("Reduce trade size or increase gas price to avoid sandwich attacks")
                elif scenario.pattern == FailurePattern.LIQUIDITY_DRAIN:
                    suggestions.append("Wait for higher liquidity or reduce position size")
                elif scenario.pattern == FailurePattern.MEV_FRONT_RUN:
                    suggestions.append("Use higher gas price or delay execution")
                elif scenario.pattern == FailurePattern.WHALE_DUMP:
                    suggestions.append("Monitor whale wallets and wait for activity to subside")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _count_historical_matches(self, token_address: str) -> int:
        """Count historical pattern matches for this token"""
        matches = 0
        for failure in self.historical_failures:
            if failure.get('token_address') == token_address:
                matches += 1
        return matches
    
    def _initialize_failure_patterns(self):
        """Initialize failure pattern configurations"""
        self.failure_patterns = {
            FailurePattern.RUG_PULL: {
                'weight': 0.9,
                'impact': 0.95,
                'detection_confidence': 0.8
            },
            FailurePattern.SANDWICH_ATTACK: {
                'weight': 0.6,
                'impact': 0.15,
                'detection_confidence': 0.7
            },
            FailurePattern.LIQUIDITY_DRAIN: {
                'weight': 0.7,
                'impact': 0.5,
                'detection_confidence': 0.8
            },
            FailurePattern.WHALE_DUMP: {
                'weight': 0.5,
                'impact': 0.6,
                'detection_confidence': 0.6
            },
            FailurePattern.MEV_FRONT_RUN: {
                'weight': 0.4,
                'impact': 0.1,
                'detection_confidence': 0.5
            },
            FailurePattern.HONEYPOT: {
                'weight': 0.8,
                'impact': 0.9,
                'detection_confidence': 0.7
            }
        }
    
    async def _load_shadow_memory(self, file_path: str):
        """Load shadow memory of known failures"""
        try:
            with open(file_path, 'r') as f:
                self.shadow_memory = json.load(f)
            self.logger.info(f"📚 Loaded shadow memory: {len(self.shadow_memory.get('known_rugs', []))} known failures")
        except FileNotFoundError:
            self.logger.warning(f"Shadow memory file not found: {file_path}")
            self.shadow_memory = {'known_rugs': [], 'failed_patterns': []}
        except Exception as e:
            self.logger.error(f"Error loading shadow memory: {e}")
            self.shadow_memory = {'known_rugs': [], 'failed_patterns': []}
    
    async def _pattern_learning_loop(self):
        """Continuously learn from new failure patterns"""
        while True:
            try:
                # Update failure patterns based on recent observations
                await self._update_failure_patterns()
                await asyncio.sleep(3600)  # Update every hour
            except Exception as e:
                self.logger.error(f"Error in pattern learning: {e}")
                await asyncio.sleep(3600)
    
    async def _update_failure_patterns(self):
        """Update failure pattern configurations based on learning"""
        # This method would implement machine learning to update pattern weights
        # based on observed outcomes. For now, it's a placeholder.
        pass
    
    def get_synapse_status(self) -> Dict[str, Any]:
        """Get comprehensive Devil's Advocate Synapse status"""
        return {
            'total_analyses': self.total_analyses,
            'vetoes_issued': self.vetoes_issued,
            'veto_rate': self.vetoes_issued / max(self.total_analyses, 1),
            'trades_saved': self.trades_saved,
            'patterns_loaded': len(self.failure_patterns),
            'shadow_memory_entries': len(self.shadow_memory.get('known_rugs', [])),
            'veto_threshold': self.veto_threshold,
            'avg_analysis_time_target_ms': self.max_analysis_time_ms,
            'protection_active': True
        } 