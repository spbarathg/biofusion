"""
DEVIL'S ADVOCATE SYNAPSE - ADVANCED WCCA RISK ASSESSMENT ENGINE
==============================================================

The Devil's Advocate Synapse conducts mathematical Worst Case Contingency Analysis (WCCA)
using integrated intelligence from all specialized analysis modules to calculate
Risk-Adjusted Expected Loss (R-EL) and issue vetoes when thresholds are exceeded.

This system embodies the Colony's second-order thinking capability, asking:
"This trade failed catastrophically. What was the most probable cause?"

WCCA Mathematical Framework:
- R-EL = P(Loss) × |Position_Size| × Impact_Severity
- Integration with EnhancedRugDetector for rug analysis
- Integration with SentimentFirstAI for sentiment-based risk
- Integration with TechnicalAnalyzer for technical risk assessment
- Bayesian probability fusion for multi-source risk assessment
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math

from worker_ant_v1.utils.logger import get_logger
from worker_ant_v1.intelligence.enhanced_rug_detector import get_rug_detector
from worker_ant_v1.intelligence.sentiment_first_ai import get_sentiment_first_ai
from worker_ant_v1.intelligence.technical_analyzer import get_technical_analyzer
from worker_ant_v1.utils.market_data_fetcher import get_market_data_fetcher

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
    """The Colony's Devil's Advocate - Advanced WCCA Risk Assessment Engine"""
    
    def __init__(self):
        self.logger = get_logger("DevilsAdvocateSynapse")
        
        # WCCA (Worst Case Contingency Analysis) Configuration
        self.acceptable_rel_threshold = 0.1  # Acceptable Risk-Adjusted Expected Loss (e.g., 0.1 SOL)
        
        # Integrated intelligence modules
        self.rug_detector = None
        self.sentiment_ai = None
        self.technical_analyzer = None
        self.market_data_fetcher = None
        
        # Failure pattern database
        self.failure_patterns: Dict[FailurePattern, Dict[str, Any]] = {}
        self.historical_failures: List[Dict[str, Any]] = []
        self.shadow_memory: Dict[str, Any] = {}  # shadow_memory.json data
        
        # Analysis configuration
        self.veto_threshold = 0.75  # 75% failure probability triggers veto
        self.max_analysis_time_ms = 500  # Maximum 500ms for analysis
        self.min_liquidity_threshold = 100.0  # SOL
        self.whale_activity_threshold = 0.7
        
        # Bayesian probability fusion weights
        self.risk_source_weights = {
            'rug_detector': 0.4,     # Highest weight for rug detection
            'sentiment': 0.25,       # Sentiment analysis weight
            'technical': 0.2,        # Technical analysis weight
            'liquidity': 0.1,        # Liquidity analysis weight
            'pattern_match': 0.05    # Historical pattern matching
        }
        
        # Performance tracking
        self.total_analyses = 0
        self.vetoes_issued = 0
        self.trades_saved = 0  # Successful vetoes that prevented losses
        self.average_analysis_time_ms = 0.0
        
        # Initialize failure patterns
        self._initialize_failure_patterns()
    
    async def initialize(self, shadow_memory_path: str = "shadow_memory.json") -> bool:
        """Initialize the Devil's Advocate Synapse with integrated intelligence modules"""
        try:
            self.logger.info("🕵️ Initializing Devil's Advocate Synapse...")
            
            # Initialize integrated intelligence modules
            self.logger.info("🔗 Initializing integrated intelligence modules...")
            
            self.rug_detector = await get_rug_detector()
            self.sentiment_ai = await get_sentiment_first_ai()
            self.technical_analyzer = await get_technical_analyzer()
            self.market_data_fetcher = await get_market_data_fetcher()
            
            self.logger.info("✅ All intelligence modules initialized successfully")
            
            # Load shadow memory of known failures
            await self._load_shadow_memory(shadow_memory_path)
            
            # Test integrated analysis
            await self._test_integrated_analysis()
            
            # Start background pattern learning
            asyncio.create_task(self._pattern_learning_loop())
            
            self.logger.info("✅ Devil's Advocate Synapse active - Advanced WCCA analysis armed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Devil's Advocate Synapse: {e}")
            return False
    
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
            
            # Calculate Risk-Adjusted Expected Loss for ALL catastrophic failure patterns
            # Include all patterns that could result in >80% loss
            catastrophic_patterns = [
                FailurePattern.RUG_PULL,
                FailurePattern.HONEYPOT,
                FailurePattern.SLOW_RUG,
                FailurePattern.LIQUIDITY_DRAIN,
                FailurePattern.CONTRACT_EXPLOIT,
                FailurePattern.WHALE_DUMP
            ]
            
            max_rel = 0.0
            worst_pattern = None
            rel_breakdown = {}
            
            for pattern in catastrophic_patterns:
                # Get failure probability from specialized analysis
                if pattern == FailurePattern.RUG_PULL:
                    failure_probability = await self._get_rug_pull_probability(trade_params)
                elif pattern == FailurePattern.HONEYPOT:
                    failure_probability = await self._get_honeypot_probability(trade_params)
                elif pattern == FailurePattern.SLOW_RUG:
                    failure_probability = await self._get_slow_rug_probability(trade_params)
                elif pattern == FailurePattern.LIQUIDITY_DRAIN:
                    failure_probability = await self._get_liquidity_drain_probability(trade_params)
                elif pattern == FailurePattern.CONTRACT_EXPLOIT:
                    failure_probability = await self._get_contract_exploit_probability(trade_params)
                elif pattern == FailurePattern.WHALE_DUMP:
                    failure_probability = await self._get_whale_dump_probability(trade_params)
                else:
                    failure_probability = 0.0
                
                # Calculate R-EL: P(Loss) * |Position_Size| * Impact_Severity
                # Apply impact multiplier for different patterns
                impact_multiplier = self._get_impact_multiplier(pattern)
                rel = failure_probability * position_size_sol * impact_multiplier
                rel_breakdown[pattern.value] = rel
                
                if rel > max_rel:
                    max_rel = rel
                    worst_pattern = pattern
                
                self.logger.debug(f"📊 {pattern.value} R-EL: {rel:.4f} SOL (P={failure_probability:.3f}, Impact={impact_multiplier:.1f})")
            
            # VETO DECISION: If any R-EL exceeds threshold
            if max_rel > self.acceptable_rel_threshold:
                veto_reason = f"{worst_pattern.value.upper()} R-EL of {max_rel:.4f} SOL exceeds threshold of {self.acceptable_rel_threshold} SOL"
                
                self.logger.warning(f"🚫 WCCA VETO | {token_address[:8]} | {veto_reason}")
                
                return {
                    "veto": True,
                    "reason": veto_reason,
                    "rel_calculated": max_rel,
                    "threshold": self.acceptable_rel_threshold,
                    "worst_pattern": worst_pattern.value,
                    "rel_breakdown": rel_breakdown,
                    "patterns_analyzed": len(catastrophic_patterns)
                }
            
            # CLEAR DECISION: All R-EL within acceptable limits
            analysis_duration = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(f"✅ WCCA CLEAR | {token_address[:8]} | Max R-EL: {max_rel:.4f} SOL | "
                           f"Patterns: {len(catastrophic_patterns)} | Duration: {analysis_duration:.1f}ms")
            
            return {
                "veto": False,
                "max_rel": max_rel,
                "rel_breakdown": rel_breakdown,
                "patterns_analyzed": len(catastrophic_patterns),
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
        Calculate rug pull probability using integrated EnhancedRugDetector
        
        Args:
            trade_params: Trade parameters containing token info
            
        Returns:
            float: Probability of rug pull (0.0 to 1.0)
        """
        try:
            if not self.rug_detector:
                self.logger.warning("Rug detector not initialized, using fallback analysis")
                return await self._fallback_rug_analysis(trade_params)
            
            token_address = trade_params.get('token_address', '')
            token_symbol = trade_params.get('token_symbol', '')
            token_name = trade_params.get('token_name', '')
            
            # Get comprehensive rug analysis from EnhancedRugDetector
            rug_analysis = await self.rug_detector.analyze_token(token_address, token_name, token_symbol)
            
            # Convert detection level to probability
            detection_level = rug_analysis.detection_level
            base_probability = rug_analysis.overall_risk
            
            # Apply confidence weighting
            confidence_factor = rug_analysis.confidence_score
            
            # Specific rug type probabilities (use the highest)
            rug_probabilities = [
                rug_analysis.honeypot_probability,
                rug_analysis.slow_rug_probability,
                rug_analysis.flash_rug_probability,
                rug_analysis.dev_exit_probability
            ]
            
            max_rug_probability = max(rug_probabilities) if rug_probabilities else base_probability
            
            # Weighted combination of base risk and specific rug probabilities
            final_probability = (base_probability * 0.6 + max_rug_probability * 0.4) * confidence_factor
            
            self.logger.debug(f"🔍 Rug analysis for {token_address[:8]}: "
                            f"Base={base_probability:.3f}, Max={max_rug_probability:.3f}, "
                            f"Final={final_probability:.3f}, Confidence={confidence_factor:.3f}")
            
            return min(0.99, final_probability)  # Cap at 99%
            
        except Exception as e:
            self.logger.error(f"Error in integrated rug pull analysis: {e}")
            return await self._fallback_rug_analysis(trade_params)
    
    async def _get_honeypot_probability(self, trade_params: Dict[str, Any]) -> float:
        """
        Calculate honeypot probability using integrated EnhancedRugDetector
        
        Args:
            trade_params: Trade parameters containing token info
            
        Returns:
            float: Probability of honeypot (0.0 to 1.0)
        """
        try:
            if not self.rug_detector:
                self.logger.warning("Rug detector not initialized, using fallback honeypot analysis")
                return await self._fallback_honeypot_analysis(trade_params)
            
            token_address = trade_params.get('token_address', '')
            token_symbol = trade_params.get('token_symbol', '')
            token_name = trade_params.get('token_name', '')
            
            # Get comprehensive rug analysis from EnhancedRugDetector
            rug_analysis = await self.rug_detector.analyze_token(token_address, token_name, token_symbol)
            
            # Primary honeypot probability from detector
            honeypot_probability = rug_analysis.honeypot_probability
            
            # Apply confidence weighting
            confidence_factor = rug_analysis.confidence_score
            
            # Cross-check with sentiment analysis for market manipulation signals
            if self.sentiment_ai:
                sentiment_result = await self.sentiment_ai.analyze_token_sentiment(token_address, token_symbol)
                
                # High positive sentiment with honeypot indicators might indicate manipulation
                if sentiment_result.overall_sentiment_score > 0.7 and honeypot_probability > 0.5:
                    manipulation_factor = 1.2  # Increase probability by 20%
                elif sentiment_result.overall_sentiment_score < 0.3 and honeypot_probability > 0.3:
                    manipulation_factor = 1.1  # Slight increase
                else:
                    manipulation_factor = 1.0
                
                honeypot_probability *= manipulation_factor
                
                self.logger.debug(f"🍯 Honeypot + Sentiment for {token_address[:8]}: "
                                f"Base={rug_analysis.honeypot_probability:.3f}, "
                                f"Sentiment={sentiment_result.overall_sentiment_score:.3f}, "
                                f"Manipulation={manipulation_factor:.3f}")
            
            # Apply confidence factor and cap result
            final_probability = honeypot_probability * confidence_factor
            
            self.logger.debug(f"🍯 Honeypot analysis for {token_address[:8]}: "
                            f"Final={final_probability:.3f}, Confidence={confidence_factor:.3f}")
            
            return min(0.99, final_probability)  # Cap at 99%
            
        except Exception as e:
            self.logger.error(f"Error in integrated honeypot analysis: {e}")
            return await self._fallback_honeypot_analysis(trade_params)
    
    async def _get_slow_rug_probability(self, trade_params: Dict[str, Any]) -> float:
        """
        Calculate slow rug pull probability using integrated EnhancedRugDetector and SentimentFirstAI
        
        Args:
            trade_params: Trade parameters containing token info
            
        Returns:
            float: Probability of slow rug pull (0.0 to 1.0)
        """
        try:
            if not self.rug_detector:
                self.logger.warning("Rug detector not initialized, using fallback slow rug analysis")
                return await self._fallback_slow_rug_analysis(trade_params)
            
            token_address = trade_params.get('token_address', '')
            token_symbol = trade_params.get('token_symbol', '')
            token_name = trade_params.get('token_name', '')
            
            # Get comprehensive rug analysis from EnhancedRugDetector
            rug_analysis = await self.rug_detector.analyze_token(token_address, token_name, token_symbol)
            
            # Primary slow rug probability from detector
            slow_rug_probability = rug_analysis.slow_rug_probability
            confidence_factor = rug_analysis.confidence_score
            
            # Cross-reference with sentiment analysis for deteriorating sentiment patterns
            sentiment_modifier = 1.0
            if self.sentiment_ai:
                sentiment_result = await self.sentiment_ai.analyze_token_sentiment(token_address, token_symbol)
                
                # Declining sentiment momentum indicates potential slow rug
                sentiment_momentum = sentiment_result.sentiment_momentum
                community_health = sentiment_result.community_health
                
                # If sentiment is declining while price might be stable (slow rug pattern)
                if sentiment_momentum < -0.3:  # Strong negative momentum
                    sentiment_modifier = 1.3  # Increase probability by 30%
                elif sentiment_momentum < -0.1:  # Mild negative momentum
                    sentiment_modifier = 1.1  # Increase probability by 10%
                
                # Community health deterioration is a strong slow rug signal
                if community_health < 0.3:
                    sentiment_modifier *= 1.2  # Additional 20% increase
                
                self.logger.debug(f"🐌 Slow Rug + Sentiment for {token_address[:8]}: "
                                f"Base={slow_rug_probability:.3f}, "
                                f"Momentum={sentiment_momentum:.3f}, "
                                f"Health={community_health:.3f}, "
                                f"Modifier={sentiment_modifier:.3f}")
            
            # Apply technical analysis for trend deterioration
            technical_modifier = 1.0
            if self.technical_analyzer:
                tech_analysis = await self.technical_analyzer.analyze_token(token_address, token_symbol)
                
                # Bearish divergence or declining volume can indicate slow rug
                if tech_analysis.overall_trend_direction.value == "BEARISH":
                    technical_modifier = 1.2  # 20% increase
                elif tech_analysis.overall_trend_direction.value == "SIDEWAYS":
                    # Sideways with declining volume might indicate slow exit
                    if tech_analysis.overall_score < 0.4:
                        technical_modifier = 1.1  # 10% increase
                
                self.logger.debug(f"🐌 Slow Rug + Technical for {token_address[:8]}: "
                                f"Trend={tech_analysis.overall_trend_direction.value}, "
                                f"Score={tech_analysis.overall_score:.3f}, "
                                f"Modifier={technical_modifier:.3f}")
            
            # Combine all factors
            final_probability = slow_rug_probability * confidence_factor * sentiment_modifier * technical_modifier
            
            self.logger.debug(f"🐌 Slow rug analysis for {token_address[:8]}: "
                            f"Final={final_probability:.3f}, Base={slow_rug_probability:.3f}, "
                            f"Confidence={confidence_factor:.3f}")
            
            return min(0.99, final_probability)  # Cap at 99%
            
        except Exception as e:
            self.logger.error(f"Error in integrated slow rug analysis: {e}")
            return await self._fallback_slow_rug_analysis(trade_params)
    
    async def _get_liquidity_drain_probability(self, trade_params: Dict[str, Any]) -> float:
        """
        Calculate liquidity drain probability using integrated technical analysis and market data
        
        Args:
            trade_params: Trade parameters containing token info
            
        Returns:
            float: Probability of liquidity drain (0.0 to 1.0)
        """
        try:
            token_address = trade_params.get('token_address', '')
            token_symbol = trade_params.get('token_symbol', '')
            
            # Base probability from contract analysis
            base_probability = 0.1  # Default baseline
            
            # Use EnhancedRugDetector for liquidity health analysis
            if self.rug_detector:
                rug_analysis = await self.rug_detector.analyze_token(token_address, '', token_symbol)
                # Use the liquidity-related risk factors from the detector
                base_probability = max(base_probability, rug_analysis.overall_risk * 0.6)  # 60% weight
            
            # Technical analysis for volume and liquidity trends
            technical_modifier = 1.0
            volume_decline_factor = 1.0
            
            if self.technical_analyzer:
                tech_analysis = await self.technical_analyzer.analyze_token(token_address, token_symbol)
                
                # Volume indicators are critical for liquidity drain detection
                volume_indicators = [ind for ind in tech_analysis.indicators if 'volume' in ind.name.lower()]
                
                if volume_indicators:
                    avg_volume_score = sum(ind.value for ind in volume_indicators) / len(volume_indicators)
                    
                    # Declining volume indicates potential liquidity issues
                    if avg_volume_score < 0.3:  # Very low volume
                        volume_decline_factor = 1.5  # 50% increase
                    elif avg_volume_score < 0.5:  # Low volume
                        volume_decline_factor = 1.2  # 20% increase
                
                # Check for volatility spikes that might indicate liquidity problems
                volatility = tech_analysis.volatility_score
                if volatility > 0.8:  # High volatility
                    technical_modifier = 1.3  # Often indicates thin liquidity
                elif volatility > 0.6:  # Moderate volatility
                    technical_modifier = 1.1
                
                self.logger.debug(f"💧 Liquidity + Technical for {token_address[:8]}: "
                                f"Volume={avg_volume_score:.3f}, "
                                f"Volatility={volatility:.3f}, "
                                f"Modifiers=Volume:{volume_decline_factor:.3f}, Tech:{technical_modifier:.3f}")
            
            # Market data analysis for liquidity concentration
            market_data_modifier = 1.0
            if self.market_data_fetcher:
                # Analyze DEX liquidity distribution and concentration
                # This would ideally fetch real liquidity data
                # For now, we simulate based on typical patterns
                
                # High-risk scenarios:
                # 1. Single large LP provider (concentration risk)
                # 2. Recent large LP withdrawals
                # 3. Extremely low total liquidity relative to market cap
                
                # Simulate liquidity concentration analysis
                total_liquidity = trade_params.get('total_liquidity_sol', 0.0)
                market_cap_sol = trade_params.get('market_cap_sol', 0.0)
                
                if market_cap_sol > 0 and total_liquidity > 0:
                    liquidity_ratio = total_liquidity / market_cap_sol
                    
                    if liquidity_ratio < 0.01:  # Less than 1% liquidity
                        market_data_modifier = 2.0  # Double the risk
                    elif liquidity_ratio < 0.05:  # Less than 5% liquidity
                        market_data_modifier = 1.5  # 50% increase
                    elif liquidity_ratio < 0.1:  # Less than 10% liquidity
                        market_data_modifier = 1.2  # 20% increase
                
                self.logger.debug(f"💧 Liquidity + Market Data for {token_address[:8]}: "
                                f"Ratio={liquidity_ratio:.4f}, "
                                f"Modifier={market_data_modifier:.3f}")
            
            # Combine all factors
            final_probability = base_probability * technical_modifier * volume_decline_factor * market_data_modifier
            
            self.logger.debug(f"💧 Liquidity drain analysis for {token_address[:8]}: "
                            f"Final={final_probability:.3f}, Base={base_probability:.3f}")
            
            return min(0.99, final_probability)  # Cap at 99%
            
        except Exception as e:
            self.logger.error(f"Error in integrated liquidity drain analysis: {e}")
            return await self._fallback_liquidity_analysis(trade_params)
    
    async def _get_contract_exploit_probability(self, trade_params: Dict[str, Any]) -> float:
        """
        Calculate contract exploit probability using integrated EnhancedRugDetector
        
        Args:
            trade_params: Trade parameters containing token info
            
        Returns:
            float: Probability of contract exploit (0.0 to 1.0)
        """
        try:
            if not self.rug_detector:
                self.logger.warning("Rug detector not initialized, using fallback contract analysis")
                return await self._fallback_contract_analysis(trade_params)
            
            token_address = trade_params.get('token_address', '')
            token_symbol = trade_params.get('token_symbol', '')
            token_name = trade_params.get('token_name', '')
            
            # Get comprehensive rug analysis from EnhancedRugDetector
            rug_analysis = await self.rug_detector.analyze_token(token_address, token_name, token_symbol)
            
            # Extract contract-specific risk factors
            contract_risk_factors = []
            
            # Use the detector's risk factors for contract analysis
            for risk_factor in rug_analysis.risk_factors:
                if any(keyword in risk_factor.description.lower() for keyword in 
                      ['contract', 'code', 'verify', 'proxy', 'ownership', 'mint', 'burn', 'pause']):
                    contract_risk_factors.append(risk_factor)
            
            # Calculate base probability from contract-specific risks
            if contract_risk_factors:
                # Weight high-severity contract risks more heavily
                weighted_scores = []
                for risk in contract_risk_factors:
                    weight = 1.0
                    if risk.severity.value == "HIGH":
                        weight = 2.0
                    elif risk.severity.value == "CRITICAL":
                        weight = 3.0
                    elif risk.severity.value == "LOW":
                        weight = 0.5
                    
                    weighted_scores.append(risk.score * weight)
                
                base_probability = min(0.9, sum(weighted_scores) / len(weighted_scores))
            else:
                # Use overall risk as fallback
                base_probability = rug_analysis.overall_risk * 0.7  # 70% weight for contract focus
            
            # Apply confidence factor
            confidence_factor = rug_analysis.confidence_score
            
            # Cross-check with sentiment for potential social engineering attacks
            sentiment_modifier = 1.0
            if self.sentiment_ai:
                sentiment_result = await self.sentiment_ai.analyze_token_sentiment(token_address, token_symbol)
                
                # Extremely positive sentiment around a risky contract might indicate social engineering
                if sentiment_result.overall_sentiment_score > 0.8 and base_probability > 0.5:
                    sentiment_modifier = 1.2  # 20% increase - potential hype-driven exploit
                elif sentiment_result.overall_sentiment_score < 0.2:
                    sentiment_modifier = 1.1  # 10% increase - negative sentiment might indicate known issues
                
                self.logger.debug(f"🔓 Contract + Sentiment for {token_address[:8]}: "
                                f"Base={base_probability:.3f}, "
                                f"Sentiment={sentiment_result.overall_sentiment_score:.3f}, "
                                f"Modifier={sentiment_modifier:.3f}")
            
            # Final calculation
            final_probability = base_probability * confidence_factor * sentiment_modifier
            
            self.logger.debug(f"🔓 Contract exploit analysis for {token_address[:8]}: "
                            f"Final={final_probability:.3f}, Base={base_probability:.3f}, "
                            f"RiskFactors={len(contract_risk_factors)}, "
                            f"Confidence={confidence_factor:.3f}")
            
            return min(0.99, final_probability)  # Cap at 99%
            
        except Exception as e:
            self.logger.error(f"Error in integrated contract exploit analysis: {e}")
            return await self._fallback_contract_analysis(trade_params)
    
    async def _get_whale_dump_probability(self, trade_params: Dict[str, Any]) -> float:
        """
        Calculate whale dump probability using integrated sentiment and technical analysis
        
        Args:
            trade_params: Trade parameters containing token info
            
        Returns:
            float: Probability of whale dump (0.0 to 1.0)
        """
        try:
            token_address = trade_params.get('token_address', '')
            token_symbol = trade_params.get('token_symbol', '')
            
            # Base probability from whale concentration metrics
            base_probability = 0.1  # Default baseline
            
            # Analyze whale concentration using available data
            top_10_holder_percent = trade_params.get('top_10_holder_percent', 0.0)
            if top_10_holder_percent > 70:  # Top 10 hold >70%
                base_probability = 0.7
            elif top_10_holder_percent > 50:  # Top 10 hold >50%
                base_probability = 0.5
            elif top_10_holder_percent > 30:  # Top 10 hold >30%
                base_probability = 0.3
            
            # Sentiment analysis for whale dump patterns
            sentiment_modifier = 1.0
            if self.sentiment_ai:
                sentiment_result = await self.sentiment_ai.analyze_token_sentiment(token_address, token_symbol)
                
                # High sentiment with concentrated holdings indicates potential dump setup
                sentiment_score = sentiment_result.overall_sentiment_score
                viral_potential = sentiment_result.viral_potential
                
                # Classic pump-and-dump pattern: high sentiment + high concentration
                if sentiment_score > 0.8 and base_probability > 0.5:
                    sentiment_modifier = 1.4  # 40% increase - classic dump setup
                elif sentiment_score > 0.7 and viral_potential > 0.7:
                    sentiment_modifier = 1.2  # 20% increase - potential viral dump
                elif sentiment_score < 0.3:  # Already negative sentiment
                    sentiment_modifier = 0.8  # 20% decrease - might already be dumping
                
                self.logger.debug(f"🐋 Whale + Sentiment for {token_address[:8]}: "
                                f"Base={base_probability:.3f}, "
                                f"Sentiment={sentiment_score:.3f}, "
                                f"Viral={viral_potential:.3f}, "
                                f"Modifier={sentiment_modifier:.3f}")
            
            # Technical analysis for whale movement patterns
            technical_modifier = 1.0
            volume_spike_factor = 1.0
            
            if self.technical_analyzer:
                tech_analysis = await self.technical_analyzer.analyze_token(token_address, token_symbol)
                
                # Large volume spikes often precede whale dumps
                volume_indicators = [ind for ind in tech_analysis.indicators if 'volume' in ind.name.lower()]
                if volume_indicators:
                    avg_volume = sum(ind.value for ind in volume_indicators) / len(volume_indicators)
                    
                    # Very high volume with price topping might indicate distribution
                    if avg_volume > 0.8:
                        volume_spike_factor = 1.3  # 30% increase
                    elif avg_volume > 0.6:
                        volume_spike_factor = 1.1  # 10% increase
                
                # RSI and momentum divergence patterns
                momentum_indicators = [ind for ind in tech_analysis.indicators if 
                                     any(name in ind.name.lower() for name in ['rsi', 'momentum', 'macd'])]
                
                if momentum_indicators:
                    avg_momentum = sum(ind.value for ind in momentum_indicators) / len(momentum_indicators)
                    
                    # Overbought conditions with high whale concentration
                    if avg_momentum > 0.8 and base_probability > 0.4:
                        technical_modifier = 1.25  # 25% increase - overbought + concentration
                    elif avg_momentum < 0.2:  # Already oversold
                        technical_modifier = 0.9  # 10% decrease - might already be dumped
                
                # High volatility can indicate whale activity
                if tech_analysis.volatility_score > 0.7:
                    technical_modifier *= 1.1  # Additional 10% for high volatility
                
                self.logger.debug(f"🐋 Whale + Technical for {token_address[:8]}: "
                                f"Volume={avg_volume:.3f}, "
                                f"Volatility={tech_analysis.volatility_score:.3f}, "
                                f"Modifiers=Volume:{volume_spike_factor:.3f}, Tech:{technical_modifier:.3f}")
            
            # Market timing analysis
            timing_modifier = 1.0
            if self.market_data_fetcher:
                # Analyze market conditions for optimal whale dump timing
                # Whales often dump during:
                # 1. High retail FOMO (high sentiment + high volume)
                # 2. Market rally peaks
                # 3. When retail is most active
                
                # Simulate market timing analysis
                current_hour = datetime.now().hour
                
                # Peak trading hours when retail is most active
                if 14 <= current_hour <= 16:  # 2-4 PM UTC (US market hours)
                    timing_modifier = 1.15  # 15% increase during peak retail hours
                elif 9 <= current_hour <= 11:  # 9-11 AM UTC
                    timing_modifier = 1.1   # 10% increase during morning hours
                
                self.logger.debug(f"🐋 Whale + Timing for {token_address[:8]}: "
                                f"Hour={current_hour}, Modifier={timing_modifier:.3f}")
            
            # Combine all factors
            final_probability = base_probability * sentiment_modifier * technical_modifier * volume_spike_factor * timing_modifier
            
            self.logger.debug(f"🐋 Whale dump analysis for {token_address[:8]}: "
                            f"Final={final_probability:.3f}, Base={base_probability:.3f}, "
                            f"Concentration={top_10_holder_percent:.1f}%")
            
            return min(0.99, final_probability)  # Cap at 99%
            
        except Exception as e:
            self.logger.error(f"Error in integrated whale dump analysis: {e}")
            return await self._fallback_whale_analysis(trade_params)
    
    def _get_impact_multiplier(self, pattern: FailurePattern) -> float:
        """Get impact severity multiplier for different failure patterns"""
        impact_multipliers = {
            FailurePattern.RUG_PULL: 0.95,          # 95% loss expected
            FailurePattern.HONEYPOT: 0.90,          # 90% loss expected
            FailurePattern.SLOW_RUG: 0.80,          # 80% loss expected
            FailurePattern.CONTRACT_EXPLOIT: 0.85,   # 85% loss expected
            FailurePattern.LIQUIDITY_DRAIN: 0.70,   # 70% loss expected
            FailurePattern.WHALE_DUMP: 0.60,        # 60% loss expected
            FailurePattern.SANDWICH_ATTACK: 0.15,   # 15% loss expected
            FailurePattern.MEV_FRONT_RUN: 0.10,     # 10% loss expected
            FailurePattern.PUMP_AND_DUMP: 0.50,     # 50% loss expected
            FailurePattern.FAKE_VOLUME: 0.30,       # 30% loss expected
        }
        
        return impact_multipliers.get(pattern, 0.5)  # Default 50% impact
    
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
    
    # Fallback analysis methods for when intelligence modules are not available
    
    async def _fallback_rug_analysis(self, trade_params: Dict[str, Any]) -> float:
        """Fallback rug pull analysis using basic heuristics"""
        try:
            probability_factors = []
            
            # Basic contract verification check
            is_verified = trade_params.get('contract_verified', True)
            if not is_verified:
                probability_factors.append(0.6)
            
            # Token age check
            token_age_hours = trade_params.get('token_age_hours', 24)
            if token_age_hours < 1:  # Very new token
                probability_factors.append(0.8)
            elif token_age_hours < 6:
                probability_factors.append(0.5)
            
            # Liquidity check
            total_liquidity = trade_params.get('total_liquidity_sol', 0.0)
            market_cap_sol = trade_params.get('market_cap_sol', 1.0)
            liquidity_ratio = total_liquidity / market_cap_sol if market_cap_sol > 0 else 0
            
            if liquidity_ratio < 0.01:
                probability_factors.append(0.9)
            elif liquidity_ratio < 0.05:
                probability_factors.append(0.6)
            
            return max(probability_factors) if probability_factors else 0.2
            
        except Exception as e:
            self.logger.error(f"Error in fallback rug analysis: {e}")
            return 0.3
    
    async def _fallback_honeypot_analysis(self, trade_params: Dict[str, Any]) -> float:
        """Fallback honeypot analysis using basic heuristics"""
        try:
            probability_factors = []
            
            # Check for transfer restrictions
            has_transfer_restrictions = trade_params.get('has_transfer_restrictions', False)
            if has_transfer_restrictions:
                probability_factors.append(0.8)
            
            # Check sell/buy ratio
            sell_buy_ratio = trade_params.get('sell_buy_ratio', 1.0)
            if sell_buy_ratio < 0.1:
                probability_factors.append(0.9)
            elif sell_buy_ratio < 0.3:
                probability_factors.append(0.6)
            
            # Check for blacklist function
            has_blacklist = trade_params.get('has_blacklist_function', False)
            if has_blacklist:
                probability_factors.append(0.7)
            
            return max(probability_factors) if probability_factors else 0.1
            
        except Exception as e:
            self.logger.error(f"Error in fallback honeypot analysis: {e}")
            return 0.3
    
    async def _fallback_slow_rug_analysis(self, trade_params: Dict[str, Any]) -> float:
        """Fallback slow rug analysis using basic heuristics"""
        try:
            probability_factors = []
            
            # Liquidity trend
            liquidity_trend_7d = trade_params.get('liquidity_trend_7d', 0.0)
            if liquidity_trend_7d < -0.2:
                probability_factors.append(0.7)
            elif liquidity_trend_7d < -0.1:
                probability_factors.append(0.4)
            
            # Sell pressure
            sell_buy_ratio = trade_params.get('sell_buy_ratio', 1.0)
            if sell_buy_ratio > 3.0:
                probability_factors.append(0.6)
            elif sell_buy_ratio > 2.0:
                probability_factors.append(0.4)
            
            # Dev activity
            dev_activity_suspicious = trade_params.get('dev_activity_suspicious', False)
            if dev_activity_suspicious:
                probability_factors.append(0.8)
            
            return max(probability_factors) if probability_factors else 0.1
            
        except Exception as e:
            self.logger.error(f"Error in fallback slow rug analysis: {e}")
            return 0.2
    
    async def _fallback_liquidity_analysis(self, trade_params: Dict[str, Any]) -> float:
        """Fallback liquidity analysis using basic heuristics"""
        try:
            probability_factors = []
            
            # Liquidity to market cap ratio
            total_liquidity = trade_params.get('total_liquidity_sol', 0.0)
            market_cap_sol = trade_params.get('market_cap_sol', 1.0)
            liquidity_ratio = total_liquidity / market_cap_sol if market_cap_sol > 0 else 0
            
            if liquidity_ratio < 0.01:
                probability_factors.append(0.9)
            elif liquidity_ratio < 0.05:
                probability_factors.append(0.6)
            elif liquidity_ratio < 0.1:
                probability_factors.append(0.3)
            
            # LP concentration
            liquidity_concentration = trade_params.get('liquidity_concentration', 0.5)
            if liquidity_concentration > 0.9:
                probability_factors.append(0.8)
            elif liquidity_concentration > 0.7:
                probability_factors.append(0.5)
            
            # Recent LP removals
            large_lp_removals = trade_params.get('large_lp_removals_24h', 0)
            if large_lp_removals > 2:
                probability_factors.append(0.7)
            elif large_lp_removals > 0:
                probability_factors.append(0.4)
            
            return max(probability_factors) if probability_factors else 0.1
            
        except Exception as e:
            self.logger.error(f"Error in fallback liquidity analysis: {e}")
            return 0.2
    
    async def _fallback_contract_analysis(self, trade_params: Dict[str, Any]) -> float:
        """Fallback contract analysis using basic heuristics"""
        try:
            probability_factors = []
            
            # Contract verification
            contract_verified = trade_params.get('contract_verified', True)
            if not contract_verified:
                probability_factors.append(0.6)
            
            # Known vulnerabilities
            has_vulnerabilities = trade_params.get('has_known_vulnerabilities', False)
            if has_vulnerabilities:
                probability_factors.append(0.9)
            
            # Contract age vs complexity
            token_age_hours = trade_params.get('token_age_hours', 24)
            contract_complexity = trade_params.get('contract_complexity_score', 0.5)
            
            if token_age_hours < 6 and contract_complexity > 0.8:
                probability_factors.append(0.7)
            elif token_age_hours < 24 and contract_complexity > 0.9:
                probability_factors.append(0.5)
            
            # Proxy without transparency
            is_proxy = trade_params.get('is_proxy', False)
            is_transparent = trade_params.get('is_transparent', True)
            if is_proxy and not is_transparent:
                probability_factors.append(0.8)
            
            return max(probability_factors) if probability_factors else 0.1
            
        except Exception as e:
            self.logger.error(f"Error in fallback contract analysis: {e}")
            return 0.2
    
    async def _fallback_whale_analysis(self, trade_params: Dict[str, Any]) -> float:
        """Fallback whale analysis using basic heuristics"""
        try:
            probability_factors = []
            
            # Whale activity score
            whale_activity_score = trade_params.get('whale_activity_score', 0.0)
            if whale_activity_score > 0.8:
                probability_factors.append(0.7)
            elif whale_activity_score > 0.6:
                probability_factors.append(0.5)
            
            # Token concentration
            top_10_holder_percent = trade_params.get('top_10_holder_percent', 0.0)
            if top_10_holder_percent > 70:
                probability_factors.append(0.8)
            elif top_10_holder_percent > 50:
                probability_factors.append(0.6)
            
            # Large transactions
            large_transactions_24h = trade_params.get('large_transactions_24h', 0)
            if large_transactions_24h > 5:
                probability_factors.append(0.6)
            elif large_transactions_24h > 2:
                probability_factors.append(0.4)
            
            # Sentiment vs activity
            market_sentiment = trade_params.get('market_sentiment', 0.5)
            if market_sentiment > 0.8 and whale_activity_score > 0.6:
                probability_factors.append(0.7)
            
            return max(probability_factors) if probability_factors else 0.1
            
        except Exception as e:
            self.logger.error(f"Error in fallback whale analysis: {e}")
            return 0.2 